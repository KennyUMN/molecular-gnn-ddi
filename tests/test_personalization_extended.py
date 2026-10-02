import unittest
from src.personalization import PatientRiskAdjustment, classify_genotype, is_poor_metabolizer


class TestExtendedPersonalization(unittest.TestCase):
    def setUp(self):
        self.adj = PatientRiskAdjustment()

    def test_slco1b1_transporter_genotype(self):
        """SLCO1B1 *5/*5 maps to poor function and adds PGx penalty for statins."""
        gene, fns, bucket = classify_genotype("SLCO1B1 *5/*5")
        self.assertEqual(gene, "SLCO1B1")
        self.assertEqual(bucket, "poor")
        self.assertTrue(is_poor_metabolizer("SLCO1B1 *5/*5"))
        
        res = self.adj.adjust(0.42, {"tier": 3, "cyp_genotype": "SLCO1B1 *5/*5"})
        self.assertEqual(res["tier_used"], 3)
        self.assertAlmostEqual(res["delta_pgx"], 0.45, places=6)
        self.assertEqual(res["detail"]["pgx_phenotype"], "poor")

    def test_hla_b1502_asian_severe_cutaneous_alert(self):
        """HLA-B*15:02 carrier maps to high risk bucket."""
        gene, fns, bucket = classify_genotype("HLA-B *15:02")
        self.assertEqual(gene, "HLA-B")
        self.assertEqual(bucket, "poor")
        
        res = self.adj.adjust(0.35, {"tier": 3, "genotype": "HLA-B *15:02"})
        self.assertEqual(res["tier_used"], 3)
        self.assertGreater(res["risk_score"], 0.35)
        self.assertAlmostEqual(res["delta_pgx"], 0.45, places=6)

    def test_crediblemeds_cardiac_qtc_risk(self):
        """Baseline QTc >= 470 ms triggers severe prolongation penalty."""
        res_severe = self.adj.adjust(0.40, {"tier": 3, "baseline_qtc": 485.0})
        self.assertEqual(res_severe["tier_used"], 3)
        self.assertAlmostEqual(res_severe["delta_cardiac"], 0.35, places=6)
        self.assertEqual(res_severe["detail"]["cardiac_qtc_risk"], "severe_prolongation")
        self.assertGreater(res_severe["risk_score"], 0.40)

        # Borderline prolongation (450 - 469 ms)
        res_borderline = self.adj.adjust(0.40, {"tier": 3, "baseline_qtc": 458.0})
        self.assertAlmostEqual(res_borderline["delta_cardiac"], 0.20, places=6)
        self.assertLess(res_borderline["risk_score"], res_severe["risk_score"])

        # Normal QTc (<450 ms) -> no penalty
        res_normal = self.adj.adjust(0.40, {"tier": 3, "baseline_qtc": 415.0})
        self.assertEqual(res_normal["delta_cardiac"], 0.0)
        self.assertEqual(res_normal["risk_score"], 0.40)

    def test_herbal_and_jamu_interactions(self):
        """Co-consumed herbal supplements (Kunyit, Sambiloto, Grapefruit) trigger HDI penalties."""
        # Kunyit (Curcumin) CYP3A4/P-gp penalty
        res_kunyit = self.adj.adjust(0.40, {"tier": 3, "jamu": "Kunyit"})
        self.assertEqual(res_kunyit["tier_used"], 3)
        self.assertAlmostEqual(res_kunyit["delta_herbal"], 0.15, places=6)
        self.assertIn("Kunyit", res_kunyit["detail"]["herbal_supplements"])

        # Grapefruit irreversible CYP3A4 suicide inhibition
        res_grapefruit = self.adj.adjust(0.40, {"tier": 3, "herbal_supplements": ["Grapefruit"]})
        self.assertAlmostEqual(res_grapefruit["delta_herbal"], 0.30, places=6)

        # Multiple herbs are additive and capped at 0.40
        res_multi = self.adj.adjust(0.40, {"tier": 3, "supplements": "Kunyit, Sambiloto, Grapefruit"})
        self.assertAlmostEqual(res_multi["delta_herbal"], 0.40, places=6)

    def test_full_tier3_stacking_with_cardiac_and_herbal(self):
        """All tiers stack together in logit space cleanly."""
        profile = {
            "tier": 3,
            "age": 72,
            "egfr": 35.0,
            "cyp_genotype": "CYP2C9 *2/*3",
            "baseline_qtc": 475.0,
            "jamu": "Kunyit"
        }
        res = self.adj.adjust(0.30, profile)
        self.assertEqual(res["tier_used"], 3)
        self.assertGreater(res["delta_demo"], 0)
        self.assertGreater(res["delta_renal"], 0)
        self.assertGreater(res["delta_pgx"], 0)
        self.assertGreater(res["delta_cardiac"], 0)
        self.assertGreater(res["delta_herbal"], 0)
        self.assertGreater(res["risk_score"], 0.65)

    def test_otc_safe_pairs_false_positive_elimination(self):
        """Paracetamol + Ibuprofen, Paracetamol + Aspirin, and Simvastatin + Metformin
        must be classified as Safe despite in-silico 2D reactive alerts."""
        from api.app import build_prediction, DRUG_DB

        # 1. Paracetamol + Ibuprofen
        res_pi = build_prediction(
            DRUG_DB["Paracetamol"], DRUG_DB["Ibuprofen"],
            drug1="Paracetamol", drug2="Ibuprofen"
        )
        self.assertEqual(res_pi["severity"], "Safe / Synergistic")
        self.assertEqual(res_pi["risk_level"], "Safe")
        self.assertLessEqual(res_pi["molecular_ddi_probability"], 0.10)
        self.assertTrue(res_pi["clinical_curation_applied"])
        self.assertGreater(res_pi["raw_in_silico_gnn_prob"], 0.50)  # proved raw GNN had false positive, but was corrected!

        # 2. Paracetamol + Aspirin
        res_pa = build_prediction(
            DRUG_DB["Paracetamol"], DRUG_DB["Aspirin"],
            drug1="Paracetamol", drug2="Aspirin"
        )
        self.assertEqual(res_pa["severity"], "Low / Safe")
        self.assertEqual(res_pa["risk_level"], "Safe")
        self.assertLessEqual(res_pa["molecular_ddi_probability"], 0.15)
        self.assertTrue(res_pa["clinical_curation_applied"])

        # 3. Simvastatin + Metformin
        res_sm = build_prediction(
            DRUG_DB["Simvastatin"], DRUG_DB["Metformin"],
            drug1="Simvastatin", drug2="Metformin"
        )
        self.assertEqual(res_sm["severity"], "Safe")
        self.assertEqual(res_sm["risk_level"], "Safe")
        self.assertLessEqual(res_sm["molecular_ddi_probability"], 0.10)
        self.assertTrue(res_sm["clinical_curation_applied"])


if __name__ == "__main__":
    unittest.main()
