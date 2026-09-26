"""Two-stage DDI pipeline — Kaggle 2x Tesla T4 training run.

Launched detached (nohup) so the notebook kernel stays responsive:
    nohup python -u kaggle_train.py --split random > runs/random.log 2>&1 &
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import DDIDataset, collate_ddi_batch, split_dataset, smiles_to_graph
from src.export_onnx import emulate_int8_weights, export_model_to_onnx
from src.explain import attribution_drift_report, explain_ddi_interaction, match_cyp_substructures
from src.model import MolecularGNN_DDI
from src.personalization import PatientRiskAdjustment
from src.utils import calculate_metrics, get_device, set_seed

CLINICAL_PAIRS = [("Warfarin", "Aspirin"), ("Simvastatin", "Amiodarone")]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_loader(df, graph_cache, batch_size, shuffle, workers):
    ds = DDIDataset(df)
    ds.graph_cache = graph_cache          # shared via fork, no per-worker rebuild
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_ddi_batch,
                      num_workers=workers, persistent_workers=workers > 0, pin_memory=True)


def to_device(batch, device):
    def move(g):
        out = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in g.items()}
        return out
    return move(batch["d1"]), move(batch["d2"]), batch["labels"].to(device, non_blocking=True)


def run_epoch(model, loader, optimizer, criterion, device, train, max_steps=0):
    model.train(train)
    total, preds, labels = 0.0, [], []
    n = 0
    for batch in loader:
        d1, d2, y = to_device(batch, device)
        with torch.set_grad_enabled(train):
            out = model(d1, d2)
            loss = criterion(out["logits"], y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
        total += loss.item() * y.size(0)
        preds.extend(out["prob"].detach().float().cpu().tolist())
        labels.extend(y.detach().cpu().tolist())
        n += y.size(0)
        if max_steps and n >= max_steps * y.size(0):
            break
    metrics = calculate_metrics(labels, preds)
    metrics["loss"] = total / max(n, 1)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tdc_drugbank_ddi.csv")
    ap.add_argument("--split", default="random", choices=["random", "scaffold", "cold_start"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=0, help="steps per epoch cap (smoke runs)")
    ap.add_argument("--drift_samples", type=int, default=200)
    ap.add_argument("--out", default="runs/random")
    ap.add_argument("--seed", type=int, default=42,
                    help="training seed (hashed splits are seed-independent)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    set_seed(a.seed)
    device = get_device()
    log(f"device={device} gpus={torch.cuda.device_count()} torch={torch.__version__}")
    if device.type == "cuda":
        for i in range(torch.cuda.device_count()):
            log(f"  cuda:{i} {torch.cuda.get_device_name(i)}")

    df = pd.read_csv(a.data)
    log(f"dataset {a.data}: {df.shape} labels={df.interaction.value_counts().to_dict()}")
    train_df, val_df, test_df = split_dataset(df, a.split, require_holdout=True)
    log(f"split[{a.split}] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    t0 = time.time()
    graph_cache = {}
    for s in pd.unique(pd.concat([df.drug1_smiles, df.drug2_smiles])):
        try:
            graph_cache[s] = smiles_to_graph(s)
        except ValueError:
            pass  # unparsable rows are dropped later (DDIDataset/split), never faked
    log(f"graph cache: {len(graph_cache)} unique drugs in {time.time()-t0:.1f}s")

    train_loader = make_loader(train_df, graph_cache, a.batch_size, True, a.workers)
    val_loader = make_loader(val_df, graph_cache, a.batch_size, False, a.workers)
    test_loader = make_loader(test_df, graph_cache, a.batch_size, False, a.workers)

    model = MolecularGNN_DDI(in_atom_features=24, hidden_dim=a.hidden_dim, num_gnn_layers=a.layers)
    # One GPU per run (two concurrent runs via CUDA_VISIBLE_DEVICES=0/1) instead
    # of nn.DataParallel: scatter() splits edge_index [2, E] along dim 0 and
    # breaks graph batches. DDP is the path if one run must span both GPUs.
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"params={n_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    criterion = nn.BCEWithLogitsLoss()

    best_auroc, best_path, bad = -1.0, os.path.join(a.out, "best_model.pt"), 0
    history = []
    for epoch in range(1, a.epochs + 1):
        t = time.time()
        tr = run_epoch(model, train_loader, optimizer, criterion, device, True, a.max_steps)
        va = run_epoch(model, val_loader, None, criterion, device, False, a.max_steps)
        scheduler.step(va["auroc"])
        log(f"epoch {epoch:02d} train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
            f"val_auroc={va['auroc']:.4f} val_auprc={va['auprc']:.4f} val_f1={va['f1']:.4f} "
            f"({time.time()-t:.0f}s, lr={optimizer.param_groups[0]['lr']:.2e})")
        history.append({"epoch": epoch, "train": tr, "val": va, "seconds": round(time.time() - t, 1)})
        core = model
        torch.save(core.state_dict(), os.path.join(a.out, "latest.pt"))
        if va["auroc"] > best_auroc:
            best_auroc, bad = va["auroc"], 0
            torch.save(core.state_dict(), best_path)
            log(f"  ↳ new best val AUROC {best_auroc:.4f} -> {best_path}")
        else:
            bad += 1
            if bad >= a.patience:
                log(f"early stop at epoch {epoch} (no val improvement for {a.patience})")
                break
        json.dump(history, open(os.path.join(a.out, "history.json"), "w"), indent=2)

    core = model
    core.load_state_dict(torch.load(best_path, map_location=device))
    core.eval()
    test = run_epoch(model, test_loader, None, criterion, device, False, a.max_steps)
    log(f"TEST[{a.split}] auroc={test['auroc']:.4f} auprc={test['auprc']:.4f} acc={test['accuracy']:.4f} f1={test['f1']:.4f}")

    # --- Stage 1 export: FP32 always, INT8 attempted + verified or reported ---
    onnx_path = os.path.join(a.out, "molecular_gnn_ddi.onnx")
    onnx_status = export_model_to_onnx(best_path, onnx_path, hidden_dim=a.hidden_dim, num_gnn_layers=a.layers,
                                       quantize_int8=True, int8_output_path=os.path.join(a.out, "molecular_gnn_ddi_int8.onnx"))

    # --- Attribution drift FP32 vs INT8 (weight-only emulation of the exported scheme) ---
    drift, clinical, n = None, {}, 0
    int8_model = emulate_int8_weights(copy.deepcopy(core))
    pairs = []
    for row in test_df.head(a.drift_samples).itertuples():
        b = collate_ddi_batch([{"d1_name": row.drug1_name, "d2_name": row.drug2_name,
                                "d1_graph": graph_cache[row.drug1_smiles],
                                "d2_graph": graph_cache[row.drug2_smiles],
                                "label": torch.tensor(float(row.interaction))}])
        d1 = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b["d1"].items()}
        d2 = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b["d2"].items()}
        a1 = explain_ddi_interaction(core, d1, d2, top_k=5)
        a2 = explain_ddi_interaction(int8_model, d1, d2, top_k=5)
        pairs.append((a1["drug1_atom_importance"], a2["drug1_atom_importance"]))
        pairs.append((a1["drug2_atom_importance"], a2["drug2_atom_importance"]))
        n += 1
    drift = attribution_drift_report(pairs, k=5)
    log(f"attribution drift n={drift['n_pairs']} mean_rho={drift['mean_spearman_rho']:.4f} "
        f"min_rho={drift['min_spearman_rho']:.4f} mean_J5={drift['mean_top_k_jaccard']:.4f}")

    # TDC rows carry DrugBank accessions, not drug names: resolve the clinical
    # panel by canonical SMILES (a name lookup silently resolves to nothing).
    from rdkit import Chem

    def _canon(smi):
        if not smi:
            return None
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol) if mol is not None else None

    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "drugs_database.json")
    db_smiles = json.load(open(db_path)) if os.path.exists(db_path) else {}
    canon_to_smiles = {}
    for smi in graph_cache:
        c = _canon(smi)
        if c is not None:
            canon_to_smiles.setdefault(c, smi)
    adj = PatientRiskAdjustment()
    for n1, n2 in CLINICAL_PAIRS:
        s1 = canon_to_smiles.get(_canon(db_smiles.get(n1)))
        s2 = canon_to_smiles.get(_canon(db_smiles.get(n2)))
        if s1 is None or s2 is None:
            log(f"clinical pair {n1}+{n2} absent from dataset (no SMILES match)")
            continue
        b = collate_ddi_batch([{"d1_name": n1, "d2_name": n2, "d1_graph": graph_cache[s1],
                                "d2_graph": graph_cache[s2], "label": torch.tensor(0.0)}])
        d1 = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b["d1"].items()}
        d2 = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b["d2"].items()}
        with torch.no_grad():
            p = float(core(d1, d2)["prob"].item())
        t2 = adj.adjust(p, {"tier": 2, "age": 68, "is_pregnant": False})
        t3 = adj.adjust(p, {"tier": 3, "age": 68, "egfr": 35.0, "cyp_genotype": "CYP2C9 *2/*3"})
        clinical[f"{n1}+{n2}"] = {"molecular_probability": round(p, 4),
                                  "tier2_risk": round(t2["risk_score"], 4),
                                  "tier3_risk": round(t3["risk_score"], 4),
                                  "cyp_motifs_drug1": sorted(match_cyp_substructures(s1)["cyp_matches"]),
                                  "cyp_motifs_drug2": sorted(match_cyp_substructures(s2)["cyp_matches"])}
    log(f"clinical panel: {json.dumps(clinical)}")

    summary = {
        "split": a.split, "epochs_run": len(history), "best_val_auroc": best_auroc, "test": test,
        "params": n_params, "gpus": torch.cuda.device_count(), "batch_size": a.batch_size,
        "hidden_dim": a.hidden_dim, "lr": a.lr, "train_rows": len(train_df), "val_rows": len(val_df),
        "test_rows": len(test_df), "split_stats": split_dataset.stats, "onnx_export_ok": onnx_status["fp32"],
        "onnx_fp32_ok": onnx_status["fp32"], "onnx_int8_ok": onnx_status["int8"],
        "drift_samples": n, "attribution_drift": drift,
        "clinical_panel": clinical, "history": history,
        "onnx": {f: round(os.path.getsize(os.path.join(a.out, f)) / 1024, 2)
                 for f in os.listdir(a.out) if f.endswith(".onnx")},
        "artifacts": sorted(os.listdir(a.out)),
    }
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    log(f"summary written to {a.out}/summary.json")
    log("DONE")


if __name__ == "__main__":
    main()
