import unittest
import os
import torch
import numpy as np
import pandas as pd
from rdkit import Chem

from src.dataset import (
    smiles_to_graph,
    split_dataset,
    DDIDataset,
    collate_ddi_batch,
    ecfp4_matrix,
    HOLDOUT_PAIRS_RAW
)
from src.model import MolecularGNN_DDI
from src.explain import (
    CYP450_SMARTS,
    match_cyp_substructures,
    spearman_rank_correlation,
    top_k_jaccard_overlap,
    attribution_drift_report
)
from src.personalization import (
    PatientRiskAdjustment,
    is_poor_metabolizer,
    parse_cyp_genotype
)

class TestDatasetAndSplits(unittest.TestCase):
    def setUp(self):
        self.csv_path = "data/sample_ddi.csv"
        self.assertTrue(os.path.exists(self.csv_path), "sample_ddi.csv must exist")
        self.df = pd.read_csv(self.csv_path)

    def test_smiles_validity(self):
        """All SMILES in the dataset must be chemically valid in RDKit."""
        for _, row in self.df.iterrows():
            m1 = Chem.MolFromSmiles(row["drug1_smiles"])
            m2 = Chem.MolFromSmiles(row["drug2_smiles"])
            self.assertIsNotNone(m1, f"Invalid SMILES for {row['drug1_name']}")
            self.assertIsNotNone(m2, f"Invalid SMILES for {row['drug2_name']}")

    def test_feature_dimensions(self):
        """Node features must be 24-dim (bond features unused by GATv2Layer)."""
        g = smiles_to_graph(self.df.iloc[0]["drug1_smiles"])
        self.assertEqual(g["x"].shape[1], 24)
        self.assertEqual(g["edge_index"].shape[0], 2)
        self.assertNotIn("edge_attr", g)

    def test_three_way_splits_leakage_free(self):
        """random: holdout pairs forced into test. Entity splits: excluded entirely."""
        for regime in ("random", "scaffold", "cold_start"):
            train_df, val_df, test_df = split_dataset(self.df, split_type=regime)
            train_pairs = {tuple(sorted((r["drug1_name"].lower(), r["drug2_name"].lower()))) for _, r in train_df.iterrows()}
            test_pairs = {tuple(sorted((r["drug1_name"].lower(), r["drug2_name"].lower()))) for _, r in test_df.iterrows()}

            for h1, h2 in HOLDOUT_PAIRS_RAW:
                p = tuple(sorted((h1.lower(), h2.lower())))
                self.assertNotIn(p, train_pairs,
                                f"Leakage detected: {h1}+{h2} in train set of {regime}")
                if regime == "random":
                    self.assertIn(p, test_pairs,
                                  f"Holdout pair {h1}+{h2} missing from test set of {regime}")
                else:
                    self.assertNotIn(p, test_pairs,
                                     f"Holdout pair {h1}+{h2} must be excluded from entity split {regime}")

    def test_ecfp4_matrix(self):
        """ECFP4 fingerprint generator must produce [N, 2048] binary vector."""
        X, y = ecfp4_matrix(self.df.iloc[:5], n_bits=1024)
        self.assertEqual(X.shape, (5, 2048))
        self.assertEqual(len(y), 5)


class TestModelSymmetry(unittest.TestCase):
    def setUp(self):
        self.model = MolecularGNN_DDI(in_atom_features=24, hidden_dim=64, num_substructures=4)
        self.model.eval()
        df = pd.read_csv("data/sample_ddi.csv")
        ds = DDIDataset(df.iloc[:2])
        self.batch = collate_ddi_batch([ds[0], ds[1]])

    def test_commutativity(self):
        """In eval mode, f(A, B) must exactly match f(B, A)."""
        with torch.no_grad():
            out_ab = self.model(self.batch["d1"], self.batch["d2"])
            out_ba = self.model(self.batch["d2"], self.batch["d1"])

        diff_logits = (out_ab["logits"] - out_ba["logits"]).abs().max().item()
        diff_probs = (out_ab["prob"] - out_ba["prob"]).abs().max().item()
        self.assertLess(diff_logits, 1e-5, f"Logits asymmetry: {diff_logits}")
        self.assertLess(diff_probs, 1e-5, f"Probability asymmetry: {diff_probs}")


class TestModelLearning(unittest.TestCase):
    def test_overfit_small_batch(self):
        """Model must be able to overfit a small 4-pair batch to prove gradient flow and learning capacity."""
        torch.manual_seed(42)
        model = MolecularGNN_DDI(in_atom_features=24, hidden_dim=64, num_substructures=4)
        model.train()
        df = pd.read_csv("data/sample_ddi.csv")
        ds = DDIDataset(df.iloc[:4])
        batch = collate_ddi_batch([ds[i] for i in range(4)])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()

        initial_loss = None
        for step in range(30):
            optimizer.zero_grad()
            out = model(batch["d1"], batch["d2"])
            loss = criterion(out["logits"], batch["labels"])
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        self.assertLess(final_loss, initial_loss * 0.35,
                        f"Model failed to overfit small batch: init={initial_loss:.4f}, final={final_loss:.4f}")


class TestExplainabilityAndAttributionDrift(unittest.TestCase):
    def test_cyp_substructures(self):
        """Aspirin contains CYP2C9 benzoic acid/carboxylate; Warfarin contains coumarin core."""
        aspirin_smi = "CC(=O)OC1=CC=CC=C1C(=O)O"
        warfarin_smi = "CC(=O)CC(C1=CC=CC=C1)C2=C(O)C3=CC=CC=C3OC2=O"

        asp_matches = match_cyp_substructures(aspirin_smi)
        war_matches = match_cyp_substructures(warfarin_smi)

        self.assertIn("CYP2C9_carboxylate", asp_matches["cyp_matches"])
        self.assertIn("CYP2C9_coumarin", war_matches["cyp_matches"])

    def test_spearman_and_jaccard(self):
        """Attribution drift metrics must compute rank correlation correctly."""
        v1 = np.array([0.1, 0.4, 0.9, -0.2, 0.05])
        v2 = np.array([0.12, 0.38, 0.85, -0.18, 0.04]) # high rank agreement
        rho = spearman_rank_correlation(v1, v2)
        jacc = top_k_jaccard_overlap(v1, v2, k=2)

        self.assertGreater(rho, 0.95)
        self.assertEqual(jacc, 1.0)

    def test_attribution_drift_report(self):
        pairs = [(np.array([1, 2, 3]), np.array([1.1, 1.9, 3.2]))]
        report = attribution_drift_report(pairs, k=2)
        self.assertTrue(report["passes_rho_0.85"])


class TestPersonalizationLayer(unittest.TestCase):
    def setUp(self):
        self.adj = PatientRiskAdjustment()

    def test_tier1_graceful_degradation(self):
        """Tier 1 zero-knowledge profile must leave molecular score untouched."""
        res_none = self.adj.adjust(0.65, None)
        self.assertEqual(res_none["tier_used"], 1)
        self.assertAlmostEqual(res_none["risk_score"], 0.65, places=5)

        res_empty = self.adj.adjust(0.65, {})
        self.assertEqual(res_empty["tier_used"], 1)
        self.assertAlmostEqual(res_empty["risk_score"], 0.65, places=5)

    def test_tier2_demographic_penalty(self):
        """Age >= 65 must add demographic penalty."""
        res = self.adj.adjust(0.5, {"age": 70, "tier": 2})
        self.assertEqual(res["tier_used"], 2)
        self.assertGreater(res["risk_score"], 0.5)
        self.assertIn("age_penalty", res["detail"])

    def test_tier3_renal_and_cyp(self):
        """eGFR < 90 and CPIC poor metabolizer must elevate risk score."""
        self.assertTrue(is_poor_metabolizer("CYP2C9 *2/*3"))
        self.assertFalse(is_poor_metabolizer("CYP2C9 *1/*1"))

        res = self.adj.adjust(0.5, {"tier": 3, "age": 70, "egfr": 45, "cyp_genotype": "CYP2C9 *2/*3"})
        self.assertEqual(res["tier_used"], 3)
        self.assertGreater(res["risk_score"], 0.60)
        self.assertGreater(res["delta_pgx"], 0.0)
        self.assertIn("cyp_genotype", res["detail"])

if __name__ == "__main__":
    unittest.main()
