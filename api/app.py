import os
import sys
import json
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import smiles_to_graph, collate_ddi_batch
from src.explain import draw_highlighted_molecule, explain_ddi_interaction, match_cyp_substructures
from src.personalization import PatientRiskAdjustment

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DRUGS_FILE = os.path.join(DATA_DIR, "drugs_database.json")
SAMPLE_DDI_FILE = os.path.join(DATA_DIR, "sample_ddi.csv")
# Prefer the Kaggle-trained checkpoint (runs_kaggle/random). NOTE: models/best_model.pt
# is STALE (older run, Sep 16) — it is kept only as a last-resort fallback; the served
# weights are runs_kaggle/random/best_model.pt whenever present.
_CKPT_CANDIDATES = [os.environ.get("DDI_CHECKPOINT"),
                    os.path.join(PROJECT_ROOT, "runs_kaggle", "random", "best_model.pt"),
                    os.path.join(PROJECT_ROOT, "models", "best_model.pt")]
MODEL_CHECKPOINT = next((p for p in _CKPT_CANDIDATES if p and os.path.exists(p)), _CKPT_CANDIDATES[-1])

RISK_ADJUSTMENT = PatientRiskAdjustment()

# Load drug database
if os.path.exists(DRUGS_FILE):
    with open(DRUGS_FILE, "r") as f:
        DRUG_DB = json.load(f)
else:
    DRUG_DB = {
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Warfarin": "CC(=O)CC(C1=CC=CC=C1)C2=C(O)C3=CC=CC=C3OC2=O",
        "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    }

# Known verified clinical interactions (Stage 1 fallback + clinical narration)
KNOWN_RULES = {
    ("Warfarin", "Aspirin"): {"prob": 0.94, "severity": "Contraindicated", "desc": "Severe gastrointestinal and systemic bleeding risk via synergistic platelet inhibition and INR elevation.", "atoms_d1": [4, 7, 8], "atoms_d2": [2, 3, 8]},
    ("Warfarin", "Ibuprofen"): {"prob": 0.91, "severity": "Major", "desc": "Significant gastrointestinal bleeding and ulceration via non-selective COX-1 inhibition.", "atoms_d1": [4, 5, 8], "atoms_d2": [8, 9, 10]},
    ("Warfarin", "Paracetamol"): {"prob": 0.55, "severity": "Moderate", "desc": "Enhanced anticoagulant effect: regular paracetamol use (especially 1-2 weeks or longer) can raise INR and increase bleeding risk, dose and duration dependent. Monitor INR and watch for signs of bleeding.", "atoms_d1": [4, 5, 8], "atoms_d2": [1, 2]},
    ("Simvastatin", "Amiodarone"): {"prob": 0.96, "severity": "Major", "desc": "Rhabdomyolysis and acute renal failure triggered by CYP3A4 inhibition.", "atoms_d1": [12, 14], "atoms_d2": [8, 9]},
    ("Sildenafil", "Nitroglycerin"): {"prob": 0.99, "severity": "Fatal Contraindicated", "desc": "Severe, refractory hypotension and cardiogenic shock via synergistic cGMP vasodilation.", "atoms_d1": [11, 14, 15], "atoms_d2": [1, 3, 5]},
    ("Simvastatin", "Fluconazole"): {"prob": 0.93, "severity": "Major", "desc": "Marked elevation of systemic statin exposure leading to myopathy via CYP3A4 inhibition.", "atoms_d1": [10, 11], "atoms_d2": [2, 4]},
    ("Methotrexate", "Aspirin"): {"prob": 0.88, "severity": "Major", "desc": "Reduced methotrexate renal clearance causing fatal bone marrow suppression.", "atoms_d1": [15, 18], "atoms_d2": [2, 3, 8]},
    ("Clopidogrel", "Omeprazole"): {"prob": 0.82, "severity": "Moderate", "desc": "Attenuated antiplatelet activation of clopidogrel via competitive CYP2C19 inhibition.", "atoms_d1": [6, 7], "atoms_d2": [4, 5, 6]},
    ("Digoxin", "Amiodarone"): {"prob": 0.89, "severity": "Major", "desc": "Digitalis toxicity and lethal AV block via P-glycoprotein efflux inhibition.", "atoms_d1": [12, 16], "atoms_d2": [8, 9]},
    ("Paracetamol", "Aspirin"): {"prob": 0.12, "severity": "Low / Safe", "desc": "No significant adverse pharmacokinetic interference at therapeutic doses.", "atoms_d1": [1, 2], "atoms_d2": [1, 2]},
    ("Simvastatin", "Metformin"): {"prob": 0.08, "severity": "Safe", "desc": "Standard guideline-recommended dual therapy for dyslipidemia and diabetes.", "atoms_d1": [2], "atoms_d2": [1]},
    ("Paracetamol", "Ibuprofen"): {"prob": 0.08, "severity": "Safe / Synergistic", "desc": "Standard over-the-counter synergistic dual analgesic therapy. Minimal pharmacokinetic competition at standard therapeutic doses (separate glucuronidation/sulfation and CYP2C9 elimination pathways).", "atoms_d1": [3, 7], "atoms_d2": [10, 12]},
    ("Amoxicillin", "Clavulanate"): {"prob": 0.05, "severity": "Safe / Synergistic", "desc": "Standard beta-lactamase inhibitor combination therapy (co-amoxiclav) for bacterial infections.", "atoms_d1": [2], "atoms_d2": [1]},
    ("Omeprazole", "Paracetamol"): {"prob": 0.06, "severity": "Safe", "desc": "Co-administration has no clinically significant pharmacokinetic interaction.", "atoms_d1": [4], "atoms_d2": [1]},
    ("Atorvastatin", "Aspirin"): {"prob": 0.08, "severity": "Safe", "desc": "Standard guideline-recommended dual therapy for secondary cardiovascular prevention.", "atoms_d1": [5], "atoms_d2": [2]},
    ("Metformin", "Aspirin"): {"prob": 0.07, "severity": "Safe", "desc": "Established co-administration in patients with type 2 diabetes and cardiovascular disease.", "atoms_d1": [1], "atoms_d2": [2]}
}

_STAGE1_CACHE = {}


def find_clinical_rule(drug1, drug2, smiles1=None, smiles2=None):
    """Resolve pair against KNOWN_RULES using name, case-insensitivity, or canonical SMILES."""
    if (drug1, drug2) in KNOWN_RULES:
        return KNOWN_RULES[(drug1, drug2)]
    if (drug2, drug1) in KNOWN_RULES:
        return KNOWN_RULES[(drug2, drug1)]

    n1, n2 = str(drug1 or "").strip().lower(), str(drug2 or "").strip().lower()
    for (r1, r2), rule in KNOWN_RULES.items():
        if (r1.lower(), r2.lower()) in ((n1, n2), (n2, n1)):
            return rule

    if smiles1 and smiles2:
        try:
            from src.dataset import canonical_smiles
            c1, c2 = canonical_smiles(smiles1), canonical_smiles(smiles2)
            if c1 and c2:
                for (r1, r2), rule in KNOWN_RULES.items():
                    s1 = DRUG_DB.get(r1)
                    s2 = DRUG_DB.get(r2)
                    if s1 and s2:
                        cr1, cr2 = canonical_smiles(s1), canonical_smiles(s2)
                        if (cr1, cr2) in ((c1, c2), (c2, c1)):
                            return rule
        except Exception:
            pass
    return None


def load_stage1_model():
    """Lazily load the Stage 1 GNN. Returns None when torch/checkpoint absent."""
    if "model" in _STAGE1_CACHE:
        return _STAGE1_CACHE["model"]
    model = None
    try:
        if os.path.exists(MODEL_CHECKPOINT):
            import torch
            from src.export_onnx import load_trained_model
            model = load_trained_model(MODEL_CHECKPOINT)
            model.eval()
            print(f"🧠 Stage 1 GNN loaded from {MODEL_CHECKPOINT}")
        else:
            print("⚠️  No Stage 1 checkpoint found - using curated clinical fallback scores")
    except Exception as exc:  # torch missing, state_dict mismatch, ...
        print(f"⚠️  Stage 1 GNN unavailable ({exc}) - using curated clinical fallback scores")
        model = None
    _STAGE1_CACHE["model"] = model
    return model


def stage1_predict(drug1, drug2, smiles1, smiles2, top_k=5):
    """Stage 1: molecular DDI probability + atom attributions."""
    model = load_stage1_model()
    if model is not None:
        try:
            import torch
            batch = collate_ddi_batch([{
                "d1_name": drug1, "d2_name": drug2,
                "d1_graph": smiles_to_graph(smiles1),
                "d2_graph": smiles_to_graph(smiles2),
                "label": torch.tensor(0.0),
            }])
            attribution = explain_ddi_interaction(model, batch["d1"], batch["d2"], top_k=top_k)
            return attribution["prediction_prob"], attribution, "gnn_fp32"
        except Exception as exc:
            print(f"⚠️  Stage 1 GNN inference failed ({exc}) - falling back")

    rule = find_clinical_rule(drug1, drug2, smiles1, smiles2)
    if rule is None:
        return None, None, "unavailable"
    prob = rule["prob"]
    attributions = {
        "prediction_prob": prob,
        "risk_level": "High" if prob > 0.7 else ("Medium" if prob > 0.4 else "Safe"),
        "drug1_atom_importance": None,
        "drug2_atom_importance": None,
        "drug1_highlight_atoms": rule["atoms_d1"],
        "drug2_highlight_atoms": rule["atoms_d2"],
        "backend": "curated_fallback",
    }
    return prob, attributions, "curated_fallback"


def build_prediction(smiles1, smiles2, patient_profile=None, drug1=None, drug2=None):
    started = time.perf_counter()
    drug1 = drug1 or smiles1[:12]
    drug2 = drug2 or smiles2[:12]
    # Trust boundary: reject garbage input loudly instead of serving a
    # prediction over a fabricated molecule graph.
    if patient_profile is not None and not isinstance(patient_profile, dict):
        raise ValueError("patient_profile must be an object with patient fields")
    from rdkit import Chem
    for smi in (smiles1, smiles2):
        if Chem.MolFromSmiles(str(smi) or "") is None:
            raise ValueError(f"unparsable SMILES: {smi!r}")

    raw_gnn_prob, attribution, backend = stage1_predict(drug1, drug2, smiles1, smiles2)
    rule = find_clinical_rule(drug1, drug2, smiles1, smiles2)

    if raw_gnn_prob is None and rule is None:
        return None

    cyp1 = match_cyp_substructures(smiles1)
    cyp2 = match_cyp_substructures(smiles2)

    # Clinical Ground-Truth Harmonization:
    # When a pair has established clinical consensus (e.g. verified safe OTC combination
    # or confirmed black-box alert), the clinical ground truth calibrates in-silico 2D
    # structural alert false positives/negatives while preserving atom gradient attributions.
    if rule is not None:
        molecular_prob = rule["prob"]
        severity, desc = rule["severity"], rule["desc"]
        clinical_curated = True
    else:
        molecular_prob = raw_gnn_prob
        clinical_curated = False
        motifs = sorted(set(cyp1["cyp_matches"]) | set(cyp2["cyp_matches"]))
        if molecular_prob > 0.7:
            severity, band = "Major Warning", "high"
        elif molecular_prob > 0.4:
            severity, band = "Moderate", "moderate"
        else:
            severity, band = "Low / Safe", "low"
        hint = (f" Detected motifs: {', '.join(motifs)}." if motifs
                else " No known CYP450 motif detected in either molecule.")
        desc = (f"Model-predicted interaction probability is {band} ({molecular_prob:.0%}), "
                f"with no curated clinical record for this pair.{hint} "
                f"The model cannot name a mechanism: verify against a real reference "
                f"(Stockley's, DrugBank, FDA label) before any therapeutic decision.")

    stage2 = RISK_ADJUSTMENT.adjust(molecular_prob, patient_profile)

    if stage2["risk_score"] > 0.7:
        risk_level = "High"
    elif stage2["risk_score"] > 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Safe"

    return {
        # --- Stage 1: molecular screening -------------------------------------
        "molecular_ddi_probability": round(molecular_prob, 4),
        "raw_in_silico_gnn_prob": round(raw_gnn_prob, 4) if raw_gnn_prob is not None else None,
        "clinical_curation_applied": clinical_curated,
        "interaction_probability": round(molecular_prob, 4),  # legacy client key
        "risk_level": risk_level,
        "severity": severity,
        "clinical_mechanism": desc,
        # --- Stage 2: patient-centric personalization -------------------------
        "personalized_risk_score": round(stage2["risk_score"], 4),
        "tier_applied": stage2["tier_used"],
        "risk_deltas": {
            "delta_demo": round(stage2["delta_demo"], 4),
            "delta_renal": round(stage2["delta_renal"], 4),
            "delta_pgx": round(stage2["delta_pgx"], 4),
            "delta_cardiac": round(stage2.get("delta_cardiac", 0.0), 4),
            "delta_herbal": round(stage2.get("delta_herbal", 0.0), 4),
        },
        "patient_factors_used": stage2["detail"],
        # --- Explainability ---------------------------------------------------
        "atom_attributions": {
            "drug1": attribution["drug1_atom_importance"],
            "drug2": attribution["drug2_atom_importance"],
            "drug1_top_atoms": attribution["drug1_highlight_atoms"],
            "drug2_top_atoms": attribution["drug2_highlight_atoms"],
            "method": "vanilla_gradient_saliency" if backend.startswith("gnn") else "curated_ground_truth",
        },
        "matched_cyp_substructures": {
            "drug1": cyp1,
            "drug2": cyp2,
        },
        # --- Metadata ---------------------------------------------------------
        "xai_toxic_substructure": {  # legacy client key
            "drug1_atom_indices": attribution["drug1_highlight_atoms"],
            "drug2_atom_indices": attribution["drug2_highlight_atoms"],
            "chemical_moiety_identified": (
                "CYP450 inhibitor motifs: " + ", ".join(sorted(set(cyp1["cyp_matches"]) | set(cyp2["cyp_matches"])))
                if (cyp1["cyp_matches"] or cyp2["cyp_matches"]) else "no verified toxicophore motif detected"),
        },
        "drug1": drug1,
        "drug2": drug2,
        "drug1_smiles": smiles1,
        "drug2_smiles": smiles2,
        "stage1_backend": backend,
        "runtime_engine": "PyTorch GNN (Stage 1) + Tiered Personalization (Stage 2)",
        "inference_latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


class DDIRequestHandler(SimpleHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/drugs":
            drug_list = [{"name": k, "smiles": v} for k, v in DRUG_DB.items()]
            tdc_file = os.path.join(DATA_DIR, "drugs_tdc.json")
            if os.path.exists(tdc_file):
                with open(tdc_file) as f:
                    drug_list.extend({"name": k, "smiles": v} for k, v in json.load(f).items())
            self._send_json({"count": len(drug_list), "drugs": drug_list})
            return

        elif parsed.path in ("/api/predict", "/predict"):
            qs = urllib.parse.parse_qs(parsed.query)
            d1 = qs.get("drug1", [""])[0].strip()
            d2 = qs.get("drug2", [""])[0].strip()
            if not d1 or not d2:
                self._send_json({"error": "Both drug1 and drug2 must be specified."}, status=400)
                return
            smiles1 = DRUG_DB.get(d1, qs.get("drug1_smiles", [None])[0])
            smiles2 = DRUG_DB.get(d2, qs.get("drug2_smiles", [None])[0])
            unknown = [name for name, smi in ((d1, smiles1), (d2, smiles2)) if smi is None]
            if unknown:
                self._send_json({"error": f"Unknown drug name(s): {', '.join(unknown)}. "
                                          "Pass drug1_smiles/drug2_smiles for drugs outside the database."},
                                status=400)
                return
            try:
                result = build_prediction(smiles1, smiles2, patient_profile=None, drug1=d1, drug2=d2)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=422)
                return
            if result is None:
                self._send_json({"error": "No prediction available: GNN unavailable and no curated rule for this pair."},
                                status=503)
                return
            self._send_json(result)
            return

        elif parsed.path == "/api/render":
            qs = urllib.parse.parse_qs(parsed.query)
            smiles = qs.get("smiles", [""])[0].strip()
            try:
                atoms = [int(x) for x in qs.get("atoms", [""])[0].split(",") if x.strip()]
            except ValueError:
                atoms = []
            if not smiles:
                self._send_json({"error": "smiles required"}, status=400)
                return
            import tempfile
            fd, tmp = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            try:
                if not draw_highlighted_molecule(smiles, atoms, tmp):
                    self._send_json({"error": "could not parse/render SMILES"}, status=422)
                    return
                with open(tmp, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            finally:
                os.remove(tmp)
            return

        # Serve frontend or static files
        if parsed.path in ["/", "/index.html", "/mobile"]:
            client_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock_mobile_client.html")
            if os.path.exists(client_path):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(client_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        if parsed.path in ["/slides", "/presentation", "/presentation_slides.html"]:
            slides_path = os.path.join(PROJECT_ROOT, "presentation_slides.html")
            if os.path.exists(slides_path):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(slides_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        if parsed.path in ["/pdf", "/slides.pdf"]:
            pdf_path = os.path.join(PROJECT_ROOT, "PharmaGNN_Progress_Presentation_Slides.pdf")
            if os.path.exists(pdf_path):
                self.send_response(200)
                self.send_header("Content-Type", "application/pdf")
                self.end_headers()
                with open(pdf_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        if parsed.path.startswith("/paper/"):
            img_name = os.path.basename(parsed.path)
            if img_name in ("empirical_training_curves.png", "split_generalization_comparison.png", "scaffold_leakage_diagram.png"):
                img_path = os.path.join(PROJECT_ROOT, "paper", img_name)
                if os.path.exists(img_path):
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.end_headers()
                    with open(img_path, "rb") as f:
                        self.wfile.write(f.read())
                    return

        # Nothing else is routable: never fall through to
        # SimpleHTTPRequestHandler (it would serve the whole repo over the
        # network: source, checkpoints, 42 MB data files).
        self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path not in ("/predict", "/api/predict"):
            self._send_json({"error": f"Unknown endpoint {parsed.path}"}, status=404)
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": f"Invalid JSON payload: {exc}"}, status=400)
            return

        smiles1 = str(payload.get("drug1_smiles") or "").strip()
        smiles2 = str(payload.get("drug2_smiles") or "").strip()
        if not smiles1 or not smiles2:
            self._send_json({"error": "drug1_smiles and drug2_smiles are required."}, status=400)
            return

        try:
            result = build_prediction(
                smiles1, smiles2,
                patient_profile=payload.get("patient_profile"),
                drug1=payload.get("drug1") or payload.get("drug1_name"),
                drug2=payload.get("drug2") or payload.get("drug2_name"),
            )
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=422)
            return
        except Exception as exc:  # never drop the connection without an error body
            self._send_json({"error": f"internal error: {exc}"}, status=500)
            return
        if result is None:
            self._send_json({"error": "No prediction available: GNN unavailable and no curated rule for this pair."},
                            status=503)
            return
        self._send_json(result)

def run_server(port=8080):
    server = HTTPServer(("0.0.0.0", port), DDIRequestHandler)
    print(f"🚀 Molecular GNN DDI Mobile API Server running at http://localhost:{port}")
    print(f"📱 Open browser to test the IF570 Mobile App Simulator!")
    server.serve_forever()

if __name__ == "__main__":
    run_server()
