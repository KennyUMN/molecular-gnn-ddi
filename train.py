import argparse
import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.utils import set_seed, get_device, calculate_metrics
from src.dataset import DDIDataset, collate_ddi_batch, split_dataset, ecfp4_matrix
from src.model import MolecularGNN_DDI
from src.export_onnx import export_model_to_onnx

def run_classical_baseline(train_df, val_df, test_df):
    """Non-DL control: ECFP4 (Morgan 1024-bit) + XGBoost."""
    import xgboost as xgb

    x_train, y_train = ecfp4_matrix(train_df)
    x_val, y_val = ecfp4_matrix(val_df)
    x_test, y_test = ecfp4_matrix(test_df)
    if len(x_train) == 0 or len(np.unique(y_train)) < 2:
        print("⚠️  ECFP4+XGBoost baseline skipped (degenerate training split)")
        return None

    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.8, eval_metric="logloss",
        # ponytail: n_jobs=1 — XGBoost hangs (OpenMP deadlock) with n_jobs>1 in
        # this macOS venv. Raise only if the fit time matters on the full dataset.
        n_jobs=1,
    )
    clf.fit(x_train, y_train)
    metrics = calculate_metrics(y_test, clf.predict_proba(x_test)[:, 1]) if len(x_test) else {}
    print("\n" + "-" * 75)
    print("🧪 NON-DL BASELINE (ECFP4 Morgan 1024-bit + XGBoost):")
    for k in ("auroc", "auprc", "accuracy", "f1"):
        if k in metrics:
            print(f"   • {k.upper():<9}  {metrics[k]:.4f}")
    print("-" * 75)
    return metrics

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in dataloader:
        d1 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["d1"].items()}
        d2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["d2"].items()}
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        out = model(d1, d2)
        loss = criterion(out["logits"], labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_preds.extend(out["prob"].detach().cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    metrics = calculate_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(dataloader.dataset)
    return metrics

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            d1 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["d1"].items()}
            d2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["d2"].items()}
            labels = batch["labels"].to(device)

            out = model(d1, d2)
            loss = criterion(out["logits"], labels)

            total_loss += loss.item() * labels.size(0)
            all_preds.extend(out["prob"].cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    metrics = calculate_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(dataloader.dataset)
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train Molecular GNN for DDI Prediction")
    parser.add_argument("--data_path", type=str, default="data/sample_ddi.csv")
    parser.add_argument("--split_type", type=str, default="random", choices=["random", "scaffold", "cold_start"], help="Splitting regime")
    parser.add_argument("--baseline", action="store_true", help="Also train the ECFP4 + XGBoost non-DL baseline")
    parser.add_argument("--quantize_int8", action="store_true", help="Export the INT8 ONNX binary alongside FP32")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="models")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"🔬 PharmaGNN Training initialized on: {device} | Split Regime: {args.split_type.upper()}")
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.data_path):
        from data.download_dataset import generate_sample_dataset
        generate_sample_dataset()

    df = pd.read_csv(args.data_path)
    train_df, val_df, test_df = split_dataset(df, split_type=args.split_type, seed=args.seed, require_holdout=True)
    print(f"📊 Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_loader = DataLoader(DDIDataset(train_df), batch_size=args.batch_size, shuffle=True, collate_fn=collate_ddi_batch)
    val_loader = DataLoader(DDIDataset(val_df), batch_size=args.batch_size, shuffle=False, collate_fn=collate_ddi_batch)
    test_loader = DataLoader(DDIDataset(test_df), batch_size=args.batch_size, shuffle=False, collate_fn=collate_ddi_batch)

    model = MolecularGNN_DDI(in_atom_features=24, hidden_dim=args.hidden_dim, num_gnn_layers=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auroc = 0.0
    best_model_path = os.path.join(args.output_dir, "best_model.pt")

    print("\n" + "="*75)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val AUROC':<10} | {'Val AUPRC':<10} | {'Val F1'}")
    print("="*75)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(f"{epoch:<8} | {train_metrics['loss']:<12.4f} | {val_metrics['loss']:<10.4f} | {val_metrics['auroc']:<10.4f} | {val_metrics['auprc']:<10.4f} | {val_metrics['f1']:<8.4f}")

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            torch.save(model.state_dict(), best_model_path)

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    test_metrics = evaluate(model, test_loader, criterion, device)

    print("\n" + "="*75)
    print(f"📊 FINAL BENCHMARK TEST RESULTS ({args.split_type.upper()} SPLIT):")
    print(f"   • AUROC:       {test_metrics['auroc']:.4f}")
    print(f"   • AUPRC:       {test_metrics['auprc']:.4f}")
    print(f"   • Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"   • F1-Score:    {test_metrics['f1']:.4f}")
    print("="*75)

    onnx_path = os.path.join(args.output_dir, "molecular_gnn_ddi.onnx")
    onnx_status = export_model_to_onnx(best_model_path, onnx_path, hidden_dim=args.hidden_dim, quantize_int8=args.quantize_int8)
    print(f"ONNX export: fp32={'ok' if onnx_status['fp32'] else 'FAILED'} "
          f"int8={onnx_status['int8']}")

    if args.baseline:
        run_classical_baseline(train_df, val_df, test_df)

if __name__ == "__main__":
    main()
