import hashlib
import json
import os
import re
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# Common elements in FDA approved drug molecules
ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Na', 'K', 'Unknown']
HYBRIDIZATION_LIST = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'Unknown']

def one_hot_encoding(value, choices):
    encoding = [0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1
    else:
        encoding[-1] = 1
    return encoding

def smiles_to_graph_rdkit(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    atom_features = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        hyb = str(atom.GetHybridization())
        deg = atom.GetTotalDegree()
        charge = atom.GetFormalCharge()
        aromatic = 1 if atom.GetIsAromatic() else 0
        h_count = atom.GetTotalNumHs()

        feat = (
            one_hot_encoding(sym, ATOM_LIST) +
            one_hot_encoding(hyb, HYBRIDIZATION_LIST) +
            [deg / 6.0, charge, aromatic, h_count / 4.0]
        )
        atom_features.append(feat)

    x = torch.tensor(atom_features, dtype=torch.float)

    # Connectivity (edge features are not used by GATv2Layer, so not computed)
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Undirected graph: add both directions
        edge_indices.extend([[i, j], [j, i]])

    if len(edge_indices) == 0:
        # Isolated single atom molecule
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    return {
        "x": x,
        "edge_index": edge_index,
        "num_nodes": x.size(0)
    }

def canonical_smiles(smiles):
    """Canonical SMILES — the identity key for matching molecules across
    datasets that key rows by name or by DrugBank accession. Stereo-insensitive
    on purpose: DrugBank/TDC entries carry explicit [H]/stereo forms while
    curated files are flat, and identity here means 'same drug entry', not 'same
    stereoisomer'. Memoized: split over 380k rows sees ~1.7k unique SMILES."""
    cached = _canon_memo.get(smiles, _MISSING)
    if cached is not _MISSING:
        return cached
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles or "")
    out = Chem.MolToSmiles(mol, isomericSmiles=False) if mol is not None else None
    _canon_memo[smiles] = out
    return out


_canon_memo = {}
_MISSING = object()


def smiles_to_graph(smiles):
    """RDKit SMILES -> graph. Raises on missing RDKit or unparsable input.

    No silent fallback parser: a fabricated chain graph looks like data and
    poisons training, and an environment without RDKit must never 'succeed'
    on fake molecules (audit finding: silent failure)."""
    from rdkit import Chem  # ImportError propagates: RDKit is a hard requirement
    if Chem.MolFromSmiles(smiles or "") is None:
        raise ValueError(f"unparsable SMILES: {smiles!r}")
    return smiles_to_graph_rdkit(smiles)


# ---------------------------------------------------------------------------
# Frozen hash-based splits (deterministic across machines and PYTHONHASHSEED)
#   random      : transductive 80:10:10, hashed per drug pair
#   scaffold    : 80:10:10 over Bemis-Murcko ring systems (scaffold-disjoint)
#   cold_start  : 80:10:10 over drugs, zero drug overlap between splits
# ---------------------------------------------------------------------------
HOLDOUT_PAIRS_RAW = [("Warfarin", "Aspirin"), ("Simvastatin", "Amiodarone")]
SPLIT_REGIMES = ("random", "scaffold", "cold_start")


def _hashed_bucket(payload, buckets=10):
    """md5 -> bucket. Stable across processes (unlike built-in hash())."""
    digest = hashlib.md5(str(payload).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


def _entity_bucket_table(keys, val_buckets=(8,), test_buckets=(9,)):
    table = {}
    for key in keys:
        b = _hashed_bucket(key)
        table[key] = "test" if b in test_buckets else ("val" if b in val_buckets else "train")
    return table


def normalize_drug_pair(name_a, name_b):
    return tuple(sorted((str(name_a).strip().lower(), str(name_b).strip().lower())))


# Canonical (sorted, lowercased) so membership tests match normalize_drug_pair.
HOLDOUT_PAIRS = {normalize_drug_pair(a, b) for a, b in HOLDOUT_PAIRS_RAW}


def _holdout_by_smiles():
    """Canonical-SMILES identity of the hold-out pairs, resolved through the
    curated data/drugs_database.json (the file that defines these names).

    Name matching alone is not enough: TDC rows key drugs by DrugBank
    accession ('DB00682'), so name-keyed hold-outs silently miss and the
    "held-out" clinical pairs leak into training (audit finding #1)."""
    global _HOUT_SMILES
    if _HOUT_SMILES is None:
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "..", "data", "drugs_database.json")) as f:
            db = json.load(f)
        _HOUT_SMILES = {}
        for a, b in HOLDOUT_PAIRS_RAW:
            for name in (a, b):
                if name not in db:
                    raise KeyError(f"hold-out drug {name!r} missing from data/drugs_database.json")
            _HOUT_SMILES[frozenset((canonical_smiles(db[a]), canonical_smiles(db[b])))] = (a, b)
    return _HOUT_SMILES


_HOUT_SMILES = None


def _holdout_by_accession():
    """Pair identity by DrugBank accession, resolved from the dataset's own
    drug map (data/drugs_tdc.json labels 'Name (DBxxxxx)').

    This is the EXACT identity for TDC rows (keyed by accession) and survives
    SMILES notation drift (explicit [H]/stereo in TDC vs flat curated forms).
    Accessions are never hardcoded — they are looked up from the map at runtime."""
    global _HOUT_ACC
    if _HOUT_ACC is None:
        _HOUT_ACC = {}
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "data", "drugs_tdc.json")
        if os.path.exists(path):
            with open(path) as f:
                labels = json.load(f)
            by_name = {}
            for label in labels:
                m = re.match(r"^(.*)\s+\((DB\d+)\)$", label)
                if m:
                    by_name[m.group(1).strip().lower()] = m.group(2)
            for a, b in HOLDOUT_PAIRS_RAW:
                if a.lower() in by_name and b.lower() in by_name:
                    _HOUT_ACC[normalize_drug_pair(by_name[a.lower()], by_name[b.lower()])] = (a, b)
    return _HOUT_ACC


_HOUT_ACC = None


_HOUT_BY_NAME = {normalize_drug_pair(a, b): (a, b) for a, b in HOLDOUT_PAIRS_RAW}


def holdout_label(name1, name2, smiles1, smiles2):
    """(a, b) raw hold-out pair label if this row IS a hold-out pair (matched by
    name, DrugBank accession, or canonical SMILES), else None. Single identity
    choke point."""
    key = normalize_drug_pair(name1, name2)
    lbl = _HOUT_BY_NAME.get(key) or _holdout_by_accession().get(key)
    if lbl:
        return lbl
    return _holdout_by_smiles().get(frozenset((canonical_smiles(smiles1), canonical_smiles(smiles2))))


def murcko_scaffold(smiles):
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(smiles or "")
        if mol is None:
            return "unparsable"
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "acyclic"
    except ImportError:
        return "no_rdkit"


def split_dataset(df, split_type="random", seed=42, require_holdout=False):
    """Three-way hashed split.

    Returns (train_df, val_df, test_df). For `random`, the clinical hold-out
    pairs (HOLDOUT_PAIRS) are forced into test. For the `scaffold`/`cold_start`
    entity splits they are excluded from all three splits instead: their drugs
    also appear in other pairs, so forcing them into test would share drugs
    with train and void the zero-overlap guarantee. They remain available as a
    clinical case-study set (see kaggle_train.py's clinical panel).
    `seed` is accepted for API compatibility but hashed splits are
    seed-independent by design.

    Hold-out rows are identified by name OR canonical SMILES (see
    holdout_label). With require_holdout=True a hold-out pair matching zero
    rows raises instead of warning: a silent miss leaks the clinical panel
    into training.

    Per-call stats are recorded on split_dataset.stats, including
    rows_dropped_straddlers (entity-split rows dropped because the pair
    straddles the train/eval boundary).
    """
    if split_type not in SPLIT_REGIMES:
        raise ValueError(f"unknown split_type={split_type!r}, expected one of {SPLIT_REGIMES}")

    df = df.reset_index(drop=True)
    records = df.to_dict("records")
    hout_label_of = [holdout_label(r["drug1_name"], r["drug2_name"],
                                  r["drug1_smiles"], r["drug2_smiles"]) for r in records]
    pair_buckets = [normalize_drug_pair(r["drug1_name"], r["drug2_name"]) for r in records]

    if split_type == "random":
        bucket_of = {p: _hashed_bucket("|".join(p)) for p in set(pair_buckets)}
    else:
        # Per-drug entity split: a pair is only train if BOTH drugs are train drugs.
        keys, key_of_row = set(), []
        if split_type == "scaffold":
            scaffold_by_name = {}
            for _, r in df.iterrows():
                for nm, smi in ((r["drug1_name"], r["drug1_smiles"]), (r["drug2_name"], r["drug2_smiles"])):
                    key = str(nm).strip().lower()
                    if key not in scaffold_by_name:      # setdefault would recompute per row
                        scaffold_by_name[key] = murcko_scaffold(smi)
            for p in pair_buckets:
                keys.add(scaffold_by_name.get(p[0], p[0]))
                keys.add(scaffold_by_name.get(p[1], p[1]))
            for p in pair_buckets:
                key_of_row.append(tuple(sorted((scaffold_by_name.get(p[0], p[0]),
                                                scaffold_by_name.get(p[1], p[1])))))
        else:  # cold_start — split over drug identities
            for p in pair_buckets:
                keys.update(p)
            key_of_row = list(pair_buckets)

        table = _entity_bucket_table(keys)
        order = {"test": 2, "val": 1, "train": 0}
        bucket_of, dropped = {}, 0
        for pair, row_keys in zip(pair_buckets, key_of_row):
            ranks = {order[table[key]] for key in row_keys}
            if len(ranks) > 1:
                # Straddling pair (one drug train-side, the other eval-side):
                # dropped so no scaffold/drug is shared across splits.
                bucket_of[pair] = None
                dropped += 1
                continue
            bucket_of[pair] = {0: 0, 1: 8, 2: 9}[ranks.pop()]  # hash scale used by `random`
        print(f"   ↳ {dropped} straddling pairs dropped (strict entity-disjoint split)")

    rows = {"train": [], "val": [], "test": []}
    held_out_seen, dropped_rows, holdout_excluded, dropped_unparsable = set(), 0, 0, 0
    force_holdout = split_type == "random"
    for idx, (row, pair, hlabel) in enumerate(zip(records, pair_buckets, hout_label_of)):
        if canonical_smiles(row["drug1_smiles"]) is None or canonical_smiles(row["drug2_smiles"]) is None:
            dropped_unparsable += 1  # counted drop at the split choke point
            continue
        if hlabel is not None:
            if force_holdout:
                rows["test"].append(idx)
                held_out_seen.add(hlabel)
            else:
                # Entity split: keep the clinical hold-out pairs out of
                # train/val/test entirely — assigning them to test would leak
                # their drugs/scaffolds across the split boundary.
                holdout_excluded += 1
            continue
        b = bucket_of[pair]
        if b is None:
            dropped_rows += 1
            continue
        rows["test" if b == 9 else ("val" if b == 8 else "train")].append(idx)

    train_df = df.iloc[rows["train"]].reset_index(drop=True)
    val_df = df.iloc[rows["val"]].reset_index(drop=True)
    test_df = df.iloc[rows["test"]].reset_index(drop=True)

    split_dataset.stats = {
        "split_type": split_type,
        "rows_dropped_straddlers": dropped_rows,
        "rows_excluded_holdout_pairs": holdout_excluded,
        "rows_dropped_unparsable": dropped_unparsable,
    }

    if force_holdout:
        missing = {lbl for lbl in HOLDOUT_PAIRS_RAW} - held_out_seen
        if missing:
            msg = (f"clinical hold-out pairs matched zero rows: {sorted(missing)} — "
                   f"hold-out rows would leak into training (check name/SMILES identity)")
            if require_holdout:
                raise ValueError(msg)
            print(f"⚠️  {msg}")
        print(f"📐 Split regime [{split_type}] -> Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}"
              f" | held-out clinical pairs in test: {len(held_out_seen)}")
    else:
        if require_holdout and holdout_excluded == 0:
            raise ValueError("clinical hold-out pairs matched zero rows in an entity split — "
                             "hold-out rows would leak into training (check name/SMILES identity)")
        print(f"📐 Split regime [{split_type}] -> Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}"
              f" | dropped: {dropped_rows} straddlers, {holdout_excluded} clinical hold-out rows")
    return train_df, val_df, test_df


def ecfp4_matrix(df, n_bits=1024, radius=2):
    """ECFP4 (Morgan) fingerprints for both drugs, concatenated -> [N, 2 * n_bits].

    Rows with unparsable SMILES are dropped (counted), not zero-fingerprinted:
    a silent zero vector looks like a real molecule to the baseline."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    def fp(smiles):
        mol = Chem.MolFromSmiles(smiles or "")
        if mol is None:
            return None
        bits = gen.GetFingerprint(mol).ToBitString()
        return (np.frombuffer(bits.encode("ascii"), dtype=np.uint8) - ord("0")).astype(np.float32)

    rows, ys = [], []
    for _, r in df.iterrows():
        f1, f2 = fp(r["drug1_smiles"]), fp(r["drug2_smiles"])
        if f1 is None or f2 is None:
            continue
        rows.append(np.concatenate([f1, f2]))
        ys.append(r["interaction"])
    dropped = len(df) - len(rows)
    if dropped:
        print(f"   ↳ {dropped} rows dropped (unparsable SMILES)")
    X = np.stack(rows) if rows else np.zeros((0, 2 * n_bits), dtype=np.float32)
    y = np.array(ys, dtype=np.float32) if ys else np.zeros(0, dtype=np.float32)
    return X, y

class DDIDataset(Dataset):
    def __init__(self, df, cache_graphs=True):
        self.df = df.reset_index(drop=True)
        self.cache_graphs = cache_graphs
        self.graph_cache = {}
        # Drop rows with unparsable SMILES up front (counted, never faked).
        ok = []
        for i, row in self.df.iterrows():
            try:
                self.get_graph(row["drug1_smiles"])
                self.get_graph(row["drug2_smiles"])
            except ValueError:
                continue
            ok.append(i)
        self.dropped_unparsable = len(self.df) - len(ok)
        if self.dropped_unparsable:
            self.df = self.df.iloc[ok].reset_index(drop=True)
            print(f"   ↳ {self.dropped_unparsable} rows dropped (unparsable SMILES)")

    def __len__(self):
        return len(self.df)

    def get_graph(self, smiles):
        if self.cache_graphs:
            if smiles not in self.graph_cache:
                self.graph_cache[smiles] = smiles_to_graph(smiles)
            return self.graph_cache[smiles]
        return smiles_to_graph(smiles)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        g1 = self.get_graph(row['drug1_smiles'])
        g2 = self.get_graph(row['drug2_smiles'])
        label = torch.tensor(row['interaction'], dtype=torch.float)

        return {
            "d1_name": row['drug1_name'],
            "d2_name": row['drug2_name'],
            "d1_graph": g1,
            "d2_graph": g2,
            "label": label
        }

def collate_ddi_batch(batch):
    # Batch multiple PyG molecular graphs together
    def batch_graphs(graph_list):
        batch_x = []
        batch_edge_index = []
        batch_mapping = []
        node_offset = 0

        for b_idx, g in enumerate(graph_list):
            num_nodes = g["num_nodes"]
            batch_x.append(g["x"])
            
            if g["edge_index"].size(1) > 0:
                shifted_edge_index = g["edge_index"] + node_offset
                batch_edge_index.append(shifted_edge_index)
            
            batch_mapping.append(torch.full((num_nodes,), b_idx, dtype=torch.long))
            node_offset += num_nodes

        return {
            "x": torch.cat(batch_x, dim=0),
            "edge_index": torch.cat(batch_edge_index, dim=1) if batch_edge_index else torch.empty((2, 0), dtype=torch.long),
            "batch": torch.cat(batch_mapping, dim=0),
            "num_graphs": len(graph_list)
        }

    d1_graphs = batch_graphs([b["d1_graph"] for b in batch])
    d2_graphs = batch_graphs([b["d2_graph"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])

    return {
        "d1": d1_graphs,
        "d2": d2_graphs,
        "labels": labels,
        "d1_names": [b["d1_name"] for b in batch],
        "d2_names": [b["d2_name"] for b in batch]
    }
