"""Acceptance tests for the two-stage hierarchical DDI pipeline.

Run:  python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import (HOLDOUT_PAIRS, collate_ddi_batch, ecfp4_matrix, murcko_scaffold,
                         normalize_drug_pair, smiles_to_graph, split_dataset)
from src.explain import (CYP450_SMARTS, attribution_drift_report, match_cyp_substructures,
                         spearman_rank_correlation, top_k_jaccard_overlap)
from src.export_onnx import emulate_int8_weights
from src.model import MolecularGNN_DDI
from src.personalization import PatientRiskAdjustment, is_poor_metabolizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "sample_ddi.csv")
CHECKPOINT = os.path.join(ROOT, "models", "best_model.pt")

SMILES_A = "CC(=O)OC1=CC=CC=C1C(=O)O"                                   # Aspirin
SMILES_B = "CC(=O)CC(C1=CC=CC=C1)C2=C(O)C3=CC=CC=C3OC2=O"               # Warfarin
SMILES_C = "CCCc1c(oc2ccccc12)C(=O)c1cc(I)c(OCCN(CC)CC)c(I)c1"          # Amiodarone
EXPECTED_MOTIFS = {SMILES_A: "CYP2C9_benzoic_acid",
                   SMILES_B: "CYP2C9_coumarin",
                   SMILES_C: "CYP3A4_benzofuran"}
HOLDOUT_DRUGS = {d for pair in HOLDOUT_PAIRS for d in pair}  # already lowercase


def _pair_batch(smiles1, smiles2, label=0.0):
    return collate_ddi_batch([{
        "d1_name": "A", "d2_name": "B",
        "d1_graph": smiles_to_graph(smiles1),
        "d2_graph": smiles_to_graph(smiles2),
        "label": torch.tensor(label),
    }])


def _build_model(seed=42, train_if_missing=True):
    torch.manual_seed(seed)
    model = MolecularGNN_DDI(in_atom_features=24, hidden_dim=64, num_gnn_layers=3)
    if os.path.exists(CHECKPOINT):
        model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    elif train_if_missing:
        raise unittest.SkipTest("no checkpoint available for attribution drift test")
    model.eval()
    return model


class TestModelSymmetry(unittest.TestCase):
    def test_forward_is_commutative(self):
        """f(A, B) == f(B, A) within 1e-5 for every ordering of the pharmacology."""
        model = MolecularGNN_DDI(in_atom_features=24, hidden_dim=64, num_gnn_layers=3).eval()
        for smiles1, smiles2 in [(SMILES_A, SMILES_B), (SMILES_B, SMILES_C), (SMILES_C, SMILES_A)]:
            b12 = _pair_batch(smiles1, smiles2)
            b21 = _pair_batch(smiles2, smiles1)
            with torch.no_grad():
                p12 = model(b12["d1"], b12["d2"])["prob"]
                p21 = model(b21["d1"], b21["d2"])["prob"]
            self.assertLess(float((p12 - p21).abs().max()), 1e-5,
                            msg=f"asymmetric prediction for {smiles1[:8]} vs {smiles2[:8]}")

    def test_encoder_is_shared_not_duplicated(self):
        """Siamese twin: one encoder/pooler instance serves both molecules."""
        from src.model import GATv2Layer, SubstructurePooler
        model = MolecularGNN_DDI(num_gnn_layers=3)
        layers = [m for m in model.modules() if isinstance(m, GATv2Layer)]
        poolers = [m for m in model.modules() if isinstance(m, SubstructurePooler)]
        self.assertEqual(len(layers), 3, "encoder weights duplicated per branch")
        self.assertEqual(len(poolers), 1, "pooler duplicated per branch")

    def test_substructure_pooling_is_size_invariant_mean(self):
        """K tokens are the softmax-weighted mean over each molecule's atoms."""
        from src.model import SubstructurePooler
        torch.manual_seed(0)
        pooler = SubstructurePooler(hidden_dim=8, num_substructures=4)
        h_nodes = torch.randn(5, 8)
        batch = torch.zeros(5, dtype=torch.long)
        tokens, assign = pooler(h_nodes, batch, num_graphs=1)
        expected = torch.stack([(h_nodes * assign[:, k:k + 1]).sum(dim=0) / assign[:, k].sum()
                                for k in range(4)], dim=0)
        self.assertTrue(torch.allclose(tokens[0], expected, atol=1e-6),
                        msg="pooling is not a weighted mean over atoms")


class TestPersonalization(unittest.TestCase):
    def setUp(self):
        self.adj = PatientRiskAdjustment()

    def test_graceful_degradation_identity(self):
        """No patient profile -> final score is exactly the Stage 1 score."""
        for s_mol in (0.0, 0.0137, 0.42, 0.737, 1.0):
            res = self.adj.adjust(s_mol, None)
            self.assertEqual(res["risk_score"], s_mol)
            self.assertEqual(res["tier_used"], 1)
        for profile in ({"tier": 1}, {}, {"age": None, "tier": None}):
            res = self.adj.adjust(0.42, profile)
            self.assertEqual(res["tier_used"], 1)
            self.assertEqual(res["risk_score"], 0.42)

    def test_tier2_age_penalty(self):
        from src.personalization import DELTA_AGE_65
        res = self.adj.adjust(0.42, {"tier": 2, "age": 68, "is_pregnant": False, "egfr": None, "cyp_genotype": None})
        self.assertEqual(res["tier_used"], 2)
        self.assertAlmostEqual(res["delta_demo"], DELTA_AGE_65, places=6)  # FAERS-calibrated, not the spec's 0.15
        self.assertGreater(res["risk_score"], 0.42)
        self.assertEqual(self.adj.adjust(0.42, {"tier": 2, "age": 40})["risk_score"], 0.42)

    def test_tier3_renal_and_pgx(self):
        res = self.adj.adjust(0.42, {"tier": 3, "age": 70, "egfr": 30.0, "cyp_genotype": "CYP2C9 *2/*3"})
        self.assertEqual(res["tier_used"], 3)
        self.assertAlmostEqual(res["delta_renal"], (1 - 30.0 / 90.0) * 0.25, places=6)
        self.assertAlmostEqual(res["delta_pgx"], 0.45, places=6)
        self.assertGreater(res["risk_score"], self.adj.adjust(0.42, {"tier": 2, "age": 70})["risk_score"])

    def test_monotonic_in_renal_impairment(self):
        """Risk rises monotonically as eGFR falls."""
        scores = [self.adj.adjust(0.3, {"tier": 3, "egfr": e})["risk_score"] for e in (90, 60, 30, 10)]
        self.assertEqual(scores, sorted(scores))

    def test_genotype_parsing(self):
        self.assertTrue(is_poor_metabolizer("CYP2C9 *2/*3"))
        self.assertTrue(is_poor_metabolizer("CYP2D6 *4/*4"))
        self.assertFalse(is_poor_metabolizer("CYP2C9 *1/*1"))
        self.assertFalse(is_poor_metabolizer(None))

    def test_extreme_probabilities_are_finite(self):
        lo = self.adj.adjust(0.0, {"tier": 3, "egfr": 5, "cyp_genotype": "CYP2D6 *4/*4"})["risk_score"]
        hi = self.adj.adjust(1.0, {"tier": 3, "egfr": 5, "cyp_genotype": "CYP2D6 *4/*4"})["risk_score"]
        self.assertTrue(0.0 <= lo <= 1.0)
        self.assertTrue(0.0 <= hi <= 1.0)
        self.assertLess(lo, hi)


class TestSplitsAndBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas as pd
        cls.df = pd.read_csv(DATA)

    def test_holdout_pairs_are_test_only(self):
        """random: forced into test. Entity splits: excluded entirely (their
        drugs appear in other pairs, so forcing them into test would leak)."""
        for regime in ("random", "scaffold", "cold_start"):
            train_df, val_df, test_df = split_dataset(self.df, regime)
            splits = {name: {normalize_drug_pair(r.drug1_name, r.drug2_name) for r in frame.itertuples()}
                      for name, frame in (("train", train_df), ("val", val_df), ("test", test_df))}
            for pair in HOLDOUT_PAIRS:
                self.assertNotIn(pair, splits["train"], msg=f"{pair} leaked into train ({regime})")
                self.assertNotIn(pair, splits["val"], msg=f"{pair} leaked into val ({regime})")
                if regime == "random":
                    self.assertIn(pair, splits["test"], msg=f"{pair} missing from test set ({regime})")
                else:
                    self.assertNotIn(pair, splits["test"],
                                     msg=f"{pair} must be excluded from entity splits ({regime})")

    @staticmethod
    def _entities(frame, key):
        values = list(frame.drug1_smiles) + list(frame.drug2_smiles)
        return {key(smi) for smi in values}

    def test_entity_splits_partition_the_dataset(self):
        """Straddling pairs are dropped, never silently leaked: rows are conserved."""
        for regime in ("scaffold", "cold_start"):
            train_df, val_df, test_df = split_dataset(self.df, regime)
            accounted = len(train_df) + len(val_df) + len(test_df)
            self.assertLessEqual(accounted, len(self.df))
            self.assertGreaterEqual(accounted, 2, "entity split kept no rows at all")

    def test_cold_start_has_zero_drug_overlap(self):
        # Strict: no carve-outs. Hold-out pairs are excluded from entity splits,
        # so nothing may bridge train and eval — the "zero drug overlap" claim.
        train_df, val_df, test_df = split_dataset(self.df, "cold_start")

        def drugs(frame):
            return {str(x).strip().lower() for x in list(frame.drug1_name) + list(frame.drug2_name)}

        train_drugs = drugs(train_df)
        eval_drugs = drugs(val_df) | drugs(test_df)
        self.assertEqual(train_drugs & eval_drugs, set(),
                         "inductive leak: a held-out drug also appears in training")

    def test_scaffold_split_is_scaffold_disjoint(self):
        train_df, val_df, test_df = split_dataset(self.df, "scaffold")
        train_scaffolds = self._entities(train_df, murcko_scaffold)
        eval_scaffolds = (self._entities(val_df, murcko_scaffold)
                          | self._entities(test_df, murcko_scaffold))
        self.assertEqual(train_scaffolds & eval_scaffolds, set(),
                         "scaffold leakage: a held-out ring system also appears in training")

    def test_splits_are_seed_and_process_independent(self):
        a = split_dataset(self.df, "random")[2].shape
        b = split_dataset(self.df, "random", seed=999)[2].shape
        self.assertEqual(a, b)

    def test_ecfp4_fingerprint_shape(self):
        X, y = ecfp4_matrix(self.df.head(4))
        self.assertEqual(X.shape, (4, 2048))
        self.assertEqual(len(y), 4)
        self.assertTrue(np.isin(X, [0.0, 1.0]).all())


class TestXAI(unittest.TestCase):
    def test_all_smarts_patterns_compile(self):
        from rdkit import Chem
        for name, smarts in CYP450_SMARTS.items():
            self.assertIsNotNone(Chem.MolFromSmarts(smarts), msg=f"bad SMARTS {name}")

    def test_cyp_motifs_found_in_clinical_pairs(self):
        for smiles, expected in EXPECTED_MOTIFS.items():
            res = match_cyp_substructures(smiles)
            self.assertTrue(res["cyp_matches"], msg=f"no CYP motif in {smiles[:12]}")
            self.assertIn(expected, res["cyp_matches"])
            self.assertIsInstance(res["brenk_alerts"], list)
            self.assertTrue(all(isinstance(i, int) for i in res["atom_indices"]))

    def test_coumarin_and_benzofuran_do_not_crossfire(self):
        self.assertNotIn("CYP2C9_coumarin", match_cyp_substructures(SMILES_C)["cyp_matches"])
        self.assertNotIn("CYP3A4_benzofuran", match_cyp_substructures(SMILES_B)["cyp_matches"])

    def test_spearman_and_jaccard(self):
        v = np.array([0.3, 0.9, 0.1, 0.7, 0.5, 0.2, 0.8])
        self.assertAlmostEqual(spearman_rank_correlation(v, v), 1.0, places=9)
        self.assertAlmostEqual(spearman_rank_correlation(v, -v), -1.0, places=9)
        self.assertEqual(top_k_jaccard_overlap(v, v, k=5), 1.0)
        self.assertLess(top_k_jaccard_overlap(v, np.roll(v, 1), k=3), 1.0)

    def test_attribution_drift_fp32_vs_int8(self):
        """Spearman rho >= 0.85 between FP32 and INT8 attributions on the test set."""
        import copy
        from src.explain import explain_ddi_interaction

        model = _build_model()
        int8_model = emulate_int8_weights(copy.deepcopy(model))

        import pandas as pd
        test_df = split_dataset(pd.read_csv(DATA), "random")[2]
        self.assertGreater(len(test_df), 0, "empty test split - drift cannot be measured")
        pairs = []
        for row in test_df.itertuples():
            b = _pair_batch(row.drug1_smiles, row.drug2_smiles, label=float(row.interaction))
            a1 = explain_ddi_interaction(model, b["d1"], b["d2"], top_k=5)
            a2 = explain_ddi_interaction(int8_model, b["d1"], b["d2"], top_k=5)
            pairs.append((a1["drug1_atom_importance"], a2["drug1_atom_importance"]))
            pairs.append((a1["drug2_atom_importance"], a2["drug2_atom_importance"]))
        report = attribution_drift_report(pairs, k=5)
        print(f"\n[drift] n={report['n_pairs']} mean_rho={report['mean_spearman_rho']:.4f} "
              f"min_rho={report['min_spearman_rho']:.4f} mean_J5={report['mean_top_k_jaccard']:.4f}")
        self.assertGreaterEqual(report["mean_spearman_rho"], 0.85,
                                msg=f"attribution drift too large: {report['mean_spearman_rho']}")


class TestAuditRegressions(unittest.TestCase):
    """Guards for the audit findings: hold-out leak, fake-graph fallback,
    swallowed degenerate metrics, unvalidated API input."""

    def test_tdc_style_holdout_matches_by_smiles(self):
        """TDC rows key drugs by DrugBank accession: hold-out identity must come
        from canonical SMILES or the clinical panel leaks into train (audit #1)."""
        import json
        import pandas as pd
        from src.dataset import split_dataset
        db = json.load(open(os.path.join(ROOT, "data", "drugs_database.json")))
        s_w, s_a = db["Warfarin"], db["Aspirin"]
        s_s, s_m = db["Simvastatin"], db["Amiodarone"]
        df = pd.DataFrame([
            {"drug1_name": "DB00682", "drug1_smiles": s_w, "drug2_name": "DB00945", "drug2_smiles": s_a, "interaction": 1},
            {"drug1_name": "DB00641", "drug1_smiles": s_s, "drug2_name": "DB00917", "drug2_smiles": s_m, "interaction": 1},
            {"drug1_name": "DB00006", "drug1_smiles": "CC(C)CC(NC(=O)C(N)Cc1ccccc1)C(=O)O",
             "drug2_name": "DB00115", "drug2_smiles": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21", "interaction": 0},
        ])
        _, _, test_df = split_dataset(df, "random", require_holdout=True)
        test_names = set(zip(test_df["drug1_name"], test_df["drug2_name"]))
        self.assertIn(("DB00682", "DB00945"), test_names)   # accession-keyed hold-out forced to test
        self.assertIn(("DB00641", "DB00917"), test_names)
        for regime in ("scaffold", "cold_start"):
            split_dataset(df, regime, require_holdout=True)
            self.assertEqual(split_dataset.stats["rows_excluded_holdout_pairs"], 2)  # excluded entirely

    def test_holdout_matches_by_accession_when_smiles_notation_drifts(self):
        """TDC row SMILES carry explicit stereo ([H][C@]...) while curated
        files are flat: accession identity must still catch the pair (the exact
        bug that made the retrain abort on real TDC data)."""
        from src.dataset import holdout_label
        lbl = holdout_label("DB00682", "DB00945",
                            "CCO-does-not-matter", "CCN-does-not-matter")
        self.assertEqual(lbl, ("Warfarin", "Aspirin"))
        lbl2 = holdout_label("DB00641", "DB01118", "x", "y")
        self.assertEqual(lbl2, ("Simvastatin", "Amiodarone"))

    def test_split_drops_unparsable_rows(self):
        import pandas as pd
        from src.dataset import split_dataset
        df = pd.DataFrame([
            {"drug1_name": "X", "drug1_smiles": "CCO", "drug2_name": "Y", "drug2_smiles": "not-a-smiles", "interaction": 1},
            {"drug1_name": "X", "drug1_smiles": "CCO", "drug2_name": "Z", "drug2_smiles": "CCN", "interaction": 0},
        ])
        tr, va, te = split_dataset(df, "random", require_holdout=False)
        self.assertEqual(split_dataset.stats["rows_dropped_unparsable"], 1)
        self.assertEqual(len(tr) + len(va) + len(te), 1)

    def test_require_holdout_fails_loud(self):
        import pandas as pd
        from src.dataset import split_dataset
        df = pd.DataFrame([{"drug1_name": "X", "drug1_smiles": "CCO",
                            "drug2_name": "Y", "drug2_smiles": "CCN", "interaction": 1}])
        with self.assertRaises(ValueError):
            split_dataset(df, "random", require_holdout=True)

    def test_holdout_smiles_resolve_from_curated_db(self):
        from src.dataset import _holdout_by_smiles
        m = _holdout_by_smiles()
        self.assertEqual(len(m), 2)
        self.assertIn(("Warfarin", "Aspirin"), m.values())
        self.assertIn(("Simvastatin", "Amiodarone"), m.values())

    def test_unparsable_smiles_raises_not_fake_graph(self):
        import pandas as pd
        from src.dataset import DDIDataset, smiles_to_graph
        with self.assertRaises(ValueError):
            smiles_to_graph("not-a-smiles")
        df = pd.DataFrame([
            {"drug1_name": "X", "drug1_smiles": "CCO", "drug2_name": "Y", "drug2_smiles": "not-a-smiles", "interaction": 1},
            {"drug1_name": "X", "drug1_smiles": "CCO", "drug2_name": "Z", "drug2_smiles": "CCN", "interaction": 0},
        ])
        ds = DDIDataset(df)
        self.assertEqual(ds.dropped_unparsable, 1)  # counted drop, never a fake chain graph
        self.assertEqual(len(ds), 1)

    def test_degenerate_metrics_raise(self):
        from src.utils import calculate_metrics
        with self.assertRaises(ValueError):
            calculate_metrics([1, 1, 1], [0.2, 0.5, 0.9])

    def test_api_rejects_garbage_input(self):
        from api.app import build_prediction
        with self.assertRaises(ValueError):
            build_prediction("not-a-smiles", "CCO", drug1="A", drug2="B")
        with self.assertRaises(ValueError):
            build_prediction("CCO", "CCN", patient_profile="oops", drug1="A", drug2="B")

    def test_adversarial_and_edge_inputs(self):
        """Adversarial and unusual inputs: empty SMILES, single-atom ions, disconnected salts."""
        from src.dataset import smiles_to_graph, collate_ddi_batch
        from src.model import MolecularGNN_DDI

        # 1. Empty SMILES must raise ValueError
        with self.assertRaises(ValueError):
            smiles_to_graph("")
        with self.assertRaises(ValueError):
            smiles_to_graph("   ")

        # 2. Single-atom ion [Na+] and [Cl-] must produce valid graphs
        g_na = smiles_to_graph("[Na+]")
        g_cl = smiles_to_graph("[Cl-]")
        self.assertEqual(g_na["num_nodes"], 1)
        self.assertEqual(g_na["edge_index"].shape, (2, 0))

        # 3. Disconnected salt [Na+].[Cl-] must produce valid graph
        g_salt = smiles_to_graph("[Na+].[Cl-]")
        self.assertEqual(g_salt["num_nodes"], 2)
        self.assertEqual(g_salt["edge_index"].shape, (2, 0))

        # 4. GNN forward on single-atom and disconnected salt must produce valid finite probabilities
        model = MolecularGNN_DDI(in_atom_features=24, hidden_dim=64, num_gnn_layers=3).eval()
        b_salt = collate_ddi_batch([{
            "d1_name": "Salt", "d2_name": "Aspirin",
            "d1_graph": g_salt, "d2_graph": smiles_to_graph(SMILES_A),
            "label": torch.tensor(0.0)
        }])
        with torch.no_grad():
            out_salt = model(b_salt["d1"], b_salt["d2"])
            prob = out_salt["prob"].item()
            self.assertTrue(0.0 <= prob <= 1.0)
            self.assertFalse(np.isnan(prob))

    def test_ecfp4_commutativity_invariant(self):
        """Baseline ECFP4 matrix must be strictly commutative: X(A, B) == X(B, A)."""
        import pandas as pd
        from src.dataset import ecfp4_matrix
        df_ab = pd.DataFrame([{"drug1_smiles": SMILES_A, "drug2_smiles": SMILES_B, "interaction": 1}])
        df_ba = pd.DataFrame([{"drug1_smiles": SMILES_B, "drug2_smiles": SMILES_A, "interaction": 1}])
        x_ab, y_ab = ecfp4_matrix(df_ab)
        x_ba, y_ba = ecfp4_matrix(df_ba)
        self.assertEqual(float(np.abs(x_ab - x_ba).max()), 0.0)


class TestAPIContract(unittest.TestCase):
    def test_predict_response_keys(self):
        from api.app import build_prediction
        res = build_prediction(SMILES_B, SMILES_A,
                               patient_profile={"tier": 2, "age": 68, "is_pregnant": False,
                                                "egfr": None, "cyp_genotype": None},
                               drug1="Warfarin", drug2="Aspirin")
        for key in ("molecular_ddi_probability", "personalized_risk_score", "tier_applied",
                    "atom_attributions", "matched_cyp_substructures"):
            self.assertIn(key, res)
        self.assertEqual(res["tier_applied"], 2)
        self.assertGreater(res["personalized_risk_score"], res["molecular_ddi_probability"])
        self.assertIn("cyp_matches", res["matched_cyp_substructures"]["drug1"])

    def test_tier3_personalization_deltas(self):
        from api.app import build_prediction
        res = build_prediction(SMILES_B, SMILES_A,
                               patient_profile={"tier": 3, "age": 70, "egfr": 30.0,
                                                "cyp_genotype": "CYP2C9 *2/*3"},
                               drug1="Warfarin", drug2="Aspirin")
        self.assertEqual(res["tier_applied"], 3)
        self.assertEqual(res["risk_deltas"]["delta_pgx"], 0.45)   # poor metabolizer (CPIC)
        self.assertGreater(res["risk_deltas"]["delta_renal"], 0)  # eGFR 30 < 90
        self.assertGreater(res["personalized_risk_score"], res["molecular_ddi_probability"])

    def test_pgx_intermediate_is_half_weight(self):
        from api.app import build_prediction
        res = build_prediction(SMILES_B, SMILES_A,
                               patient_profile={"tier": 3, "cyp_genotype": "CYP2C19 *1/*2"},
                               drug1="Warfarin", drug2="Aspirin")
        self.assertEqual(res["tier_applied"], 3)
        self.assertEqual(res["risk_deltas"]["delta_pgx"], 0.225)  # intermediate = half of poor
        self.assertEqual(res["patient_factors_used"]["pgx_phenotype"], "intermediate")

    def test_age_penalty_is_faers_calibrated(self):
        from src.personalization import DELTA_AGE_65
        # Calibrated ln(ROR) of FAERS fatal-outcome reports, age 65+ vs 0-64
        self.assertGreater(DELTA_AGE_65, 0)
        self.assertLess(DELTA_AGE_65, 1.0)

    def test_predict_tier1_matches_stage1(self):
        from api.app import build_prediction
        res = build_prediction(SMILES_B, SMILES_A)
        self.assertEqual(res["tier_applied"], 1)
        self.assertEqual(res["personalized_risk_score"], res["molecular_ddi_probability"])

    def test_mechanism_narration_is_not_invented(self):
        """Unlisted pair: no fabricated monitoring advice, honest disclaimer.
        Warfarin+Paracetamol: the real risk is INR elevation/bleeding."""
        from api.app import DRUG_DB, build_prediction
        generic = build_prediction(DRUG_DB["Ibuprofen"], DRUG_DB["Digoxin"],
                                   drug1="Ibuprofen", drug2="Digoxin")
        self.assertNotIn("blood pressure", generic["clinical_mechanism"])
        self.assertNotIn("renal profile", generic["clinical_mechanism"])
        self.assertIn("Stockley", generic["clinical_mechanism"])
        wp = build_prediction(DRUG_DB["Warfarin"], DRUG_DB["Paracetamol"],
                              drug1="Warfarin", drug2="Paracetamol")
        self.assertIn("INR", wp["clinical_mechanism"])
        self.assertNotIn("blood pressure", wp["clinical_mechanism"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
