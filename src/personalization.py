"""Stage 2 — Patient-Centric Personalization Layer.

Three-tier graceful degradation: a missing patient profile never blocks the
pipeline, it just falls back to the pure molecular (Stage 1) score.

    Risk_final = sigmoid(logit(S_mol) + delta_demo + delta_renal + delta_pgx)

Deltas are population-level add-ons expressed in logit space. The pgx allele
table and the age delta are calibrated from real sources by
data/calibrate_deltas.py (ClinPGx/CPIC allele functions; FAERS fatal-outcome
reporting odds) into data/delta_calibration.json. Pregnancy and hepatic
penalties have no clean public field and keep the spec's placeholder knobs
(MASTER_PROJECT_SPECIFICATION.md 3.3): re-fit when MIMIC-IV/JADER is in.
"""
import json
import math
import os

# --- calibration artifact (produced by data/calibrate_deltas.py) --------------
try:
    with open(os.path.join(os.path.dirname(__file__), "..", "data",
                           "delta_calibration.json")) as _f:
        _CAL = json.load(_f)
except (OSError, ValueError):
    _CAL = {}  # graceful: fall back to the spec knobs and the legacy allele set

# {gene: {"*2": "decreased", ...}} from ClinPGx haplotype functionTerm (CPIC)
ALLELE_FUNCTION = _CAL.get("pgx_allele_function", {})
# ln of the FAERS fatal-outcome ROR (age >= 65 vs 0-64, pooled known pairs,
# n ~ 29k reports with age); floor 0: no risk credit from age. The spec's 0.15
# applies only when the calibration file is absent.
DELTA_AGE_65 = max(0.0, float(
    _CAL.get("faers_age", {}).get("fatal_outcome", {}).get("ln_ror", 0.15)))

# Legacy fallback used only when the calibration file is absent.
POOR_METABOLIZER_ALLELES = {
    "CYP2C9": {"*2", "*3", "*5", "*6", "*11", "*13"},
    "CYP2C19": {"*2", "*3", "*4", "*5", "*6", "*7", "*8"},
    "CYP2D6": {"*3", "*4", "*5", "*6", "*7", "*8", "*11", "*12"},
    "CYP3A4": {"*20", "*22"},
    "SLCO1B1": {"*5", "*15", "*17", "*37"},
    "HLA-B": {"*15:02", "*57:01", "*58:01"},
}


def _norm_allele(tok):
    tok = tok.strip().upper().replace(" ", "")
    if not tok:
        return None
    return tok if tok.startswith("*") else "*" + tok


def _allele_list(genotype):
    """Like parse_cyp_genotype but keeps duplicates: 'CYP2C9 *2/*2' -> (gene, [*2, *2])."""
    if not genotype:
        return None, []
    if isinstance(genotype, dict):
        gene = None
        out = []
        for g, alleles in genotype.items():
            gene = gene or str(g).upper()
            toks = alleles if isinstance(alleles, (list, tuple, set)) else str(alleles).split("/")
            out += [_norm_allele(str(a)) for a in toks]
        return gene, [a for a in out if a]
    text = str(genotype).upper().replace(" ", "")
    gene = None
    for g in POOR_METABOLIZER_ALLELES:
        if g in text:
            gene = g
            text = text.replace(g, "")
            break
    out = [_norm_allele(a) for a in text.split("/")]
    return gene, [a for a in out if a]


def parse_cyp_genotype(genotype):
    """'CYP2C9 *2/*3' -> ('CYP2C9', {'*2', '*3'}). Tolerant of dict/list input."""
    gene, alleles = _allele_list(genotype)
    return gene, set(alleles)


# Allele-function pair -> metabolizer bucket. ponytail: function-pair rules,
# not per-gene CPIC activity scores (e.g. CYP2D6 *10/*10 collapses to 'poor'
# where clinical tables call it 'intermediate'). Upgrade path: per-gene
# activity-score tables. Bucket weights are risk knobs (poor = the spec's 0.45,
# intermediate = half), not clinical diplotype calls.
PGX_DELTA = {
    "poor": 0.45,           # spec 3.3: poor metabolizer -> +0.45
    "intermediate": 0.225,  # interpolation: half of the poor-metabolizer weight
    "normal": 0.0,
    "rapid": 0.0,
    "ultrarapid": 0.0,
    "unknown": 0.0,
}

# Canonical (alphabetically sorted) function pairs.
_POOR_PAIRS = {("decreased", "decreased"), ("decreased", "no_function"),
               ("no_function", "no_function")}
_INTERMEDIATE_PAIRS = {("decreased", "increased"), ("decreased", "normal"),
                       ("increased", "no_function"), ("no_function", "normal")}


def classify_genotype(genotype):
    """-> (gene, [function, ...], bucket). A single allele is read as homozygous."""
    gene, alleles = _allele_list(genotype)
    if not alleles:
        return None, [], "unknown"
    if len(alleles) == 1:
        alleles = alleles * 2
    table = ALLELE_FUNCTION.get(gene, {})
    fns = [table.get(a) for a in alleles]
    if any(f is None for f in fns):
        return gene, [f for f in fns if f], "unknown"
    pair = tuple(sorted(fns))
    if pair in _POOR_PAIRS:
        bucket = "poor"
    elif pair in _INTERMEDIATE_PAIRS:
        bucket = "intermediate"
    elif pair == ("normal", "normal"):
        bucket = "normal"
    elif pair == ("increased", "normal"):
        bucket = "rapid"
    elif pair == ("increased", "increased"):
        bucket = "ultrarapid"
    else:
        bucket = "unknown"
    return gene, fns, bucket


def is_poor_metabolizer(genotype):
    """True if the diplotype maps to the poor-metabolizer bucket."""
    if not ALLELE_FUNCTION:  # legacy fallback when calibration is absent
        gene, alleles = parse_cyp_genotype(genotype)
        if not alleles:
            return False
        pool = POOR_METABOLIZER_ALLELES.get(gene, set().union(*POOR_METABOLIZER_ALLELES.values()))
        return bool(alleles & pool)
    return classify_genotype(genotype)[2] == "poor"


class PatientRiskAdjustment:
    # Tier 2: demographic penalties (logit space). Age is FAERS-calibrated
    # (DELTA_AGE_65); pregnancy and hepatic stay spec placeholders (ponytail:
    # no clean public field; refit from MIMIC-IV/JADER when available).
    DELTA_PREGNANT = 0.10
    DELTA_HEPATIC = 0.12
    # Tier 3: Biomarkers, Pharmacogenomics, Organ clearance & Herbals
    DELTA_RENAL_MAX = 0.25       # spec: (1 - eGFR/90) * 0.25, capped at 0.25
    DELTA_EGFR_REFERENCE = 90.0
    DELTA_CARDIAC_QTC = 0.35     # CredibleMeds Known Risk / baseline QTc >= 470 ms
    DELTA_CARDIAC_POSSIBLE = 0.20 # CredibleMeds Possible Risk / baseline QTc 450-469 ms
    DELTA_HERBAL_DEFAULT = 0.15  # KNApSAcK Jamu / Supp.AI default HDI penalty
    EPS = 1e-6

    def __init__(self, age_threshold=65, **overrides):
        self.age_threshold = age_threshold
        self.DELTA_AGE_65 = DELTA_AGE_65
        for k, v in overrides.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def adjust(self, s_mol, patient_profile=None):
        """s_mol: Stage 1 molecular probability. Returns dict with risk score,
        tier actually applied, and the individual delta components."""
        s_mol = float(s_mol)
        tier = self._requested_tier(patient_profile)
        delta_demo = delta_renal = delta_pgx = delta_cardiac = delta_herbal = 0.0
        detail = {}

        if tier >= 2:
            p = patient_profile or {}
            if float(p.get("age") or 0) >= self.age_threshold:
                delta_demo += self.DELTA_AGE_65
                detail["age_penalty"] = round(self.DELTA_AGE_65, 4)
            if p.get("is_pregnant"):
                delta_demo += self.DELTA_PREGNANT
                detail["pregnancy_penalty"] = self.DELTA_PREGNANT
            if p.get("hepatic_impairment") or p.get("has_ulcer_history") or p.get("comorbidity"):
                delta_demo += self.DELTA_HEPATIC
                detail["hepatic_penalty"] = self.DELTA_HEPATIC

        if tier >= 3:
            p = patient_profile or {}
            egfr = p.get("egfr")
            if egfr is not None:
                egfr = max(0.0, float(egfr))
                delta_renal = max(0.0, (1.0 - egfr / self.DELTA_EGFR_REFERENCE) * self.DELTA_RENAL_MAX)
                detail["egfr"] = egfr
            genotype = p.get("cyp_genotype") or p.get("genotype")
            if genotype:
                _, fns, bucket = classify_genotype(genotype)
                delta_pgx = PGX_DELTA.get(bucket, 0.0)
                if bucket != "unknown":
                    detail["pgx_phenotype"] = bucket
                if delta_pgx:
                    detail["cyp_genotype"] = str(genotype)
                    detail["genotype"] = str(genotype)
                    detail["pgx_functions"] = fns

            # CredibleMeds Cardiac QTc Risk (ECG baseline >= 470 ms or known risk flag)
            baseline_qtc = p.get("baseline_qtc")
            if baseline_qtc is not None:
                baseline_qtc = float(baseline_qtc)
                detail["baseline_qtc"] = baseline_qtc
                if baseline_qtc >= 470.0:
                    delta_cardiac = self.DELTA_CARDIAC_QTC
                    detail["cardiac_qtc_risk"] = "severe_prolongation"
                elif baseline_qtc >= 450.0:
                    delta_cardiac = self.DELTA_CARDIAC_POSSIBLE
                    detail["cardiac_qtc_risk"] = "borderline_prolongation"
            elif p.get("has_qtc_risk") or p.get("cardiac_risk"):
                delta_cardiac = self.DELTA_CARDIAC_QTC
                detail["cardiac_qtc_risk"] = "clinical_high_risk"

            # Herbal-Drug & Indonesian Jamu Interactions (KNApSAcK / Supp.AI)
            herbals = p.get("herbal_supplements") or p.get("jamu") or p.get("supplements")
            if herbals:
                if isinstance(herbals, str):
                    herbals = [h.strip() for h in herbals.split(",") if h.strip()]
                herbal_db = _CAL.get("herbal_interactions", {}).get("herbals", {})
                h_penalties = []
                for h in herbals:
                    h_clean = h.lower().strip().replace(" ", "_").replace("'", "")
                    matched_info = herbal_db.get(h_clean)
                    if not matched_info:
                        for k, v in herbal_db.items():
                            if k in h_clean or h_clean in k or h_clean in v.get("english_name", "").lower():
                                matched_info = v
                                break
                    weight = matched_info.get("penalty_weight", self.DELTA_HERBAL_DEFAULT) if matched_info else self.DELTA_HERBAL_DEFAULT
                    h_penalties.append((h, weight))
                if h_penalties:
                    delta_herbal = min(0.40, sum(w for _, w in h_penalties))
                    detail["herbal_supplements"] = [h for h, _ in h_penalties]
                    detail["herbal_penalty"] = round(delta_herbal, 4)

        total_delta = delta_demo + delta_renal + delta_pgx + delta_cardiac + delta_herbal

        if tier == 1 or total_delta == 0.0:
            # Graceful degradation / neutral profile: identity with the Stage 1
            # output, exactly (no logit->sigmoid round-trip error).
            return {
                "risk_score": s_mol,
                "molecular_probability": s_mol,
                "tier_used": 1 if tier == 1 else tier,
                "delta_demo": 0.0,
                "delta_renal": 0.0,
                "delta_pgx": 0.0,
                "delta_cardiac": 0.0,
                "delta_herbal": 0.0,
                "detail": detail,
            }

        logit = math.log(max(s_mol, self.EPS) / max(1.0 - s_mol, self.EPS))
        risk = 1.0 / (1.0 + math.exp(-(logit + total_delta)))
        return {
            "risk_score": float(risk),
            "molecular_probability": s_mol,
            "tier_used": tier,
            "delta_demo": float(delta_demo),
            "delta_renal": float(delta_renal),
            "delta_pgx": float(delta_pgx),
            "delta_cardiac": float(delta_cardiac),
            "delta_herbal": float(delta_herbal),
            "detail": detail,
        }

    def _requested_tier(self, patient_profile):
        if not patient_profile:
            return 1
        try:
            requested = int(patient_profile.get("tier") or 0)
        except (TypeError, ValueError):
            requested = 0
        # A tier is only usable if its inputs are actually present; the reported
        # tier is the highest tier with data, so an empty profile degrades to 1.
        has_biomarkers = (patient_profile.get("egfr") is not None or
                          patient_profile.get("cyp_genotype") or
                          patient_profile.get("genotype") or
                          patient_profile.get("baseline_qtc") is not None or
                          patient_profile.get("has_qtc_risk") or
                          patient_profile.get("herbal_supplements") or
                          patient_profile.get("jamu") or
                          patient_profile.get("supplements"))
        has_demographics = any(patient_profile.get(k) is not None for k in
                               ("age", "is_pregnant", "hepatic_impairment",
                                "has_ulcer_history", "comorbidity"))
        available = 3 if has_biomarkers else (2 if has_demographics else 1)
        return max(1, min(requested or available, available))

if __name__ == "__main__":
    # Data guard: the calibration table must cover the spec 3.3 PM examples.
    for g, a in (("CYP2C9", "*2"), ("CYP2C9", "*3"), ("CYP2C19", "*2"),
                 ("CYP2D6", "*4"), ("CYP3A4", "*22")):
        assert ALLELE_FUNCTION.get(g, {}).get(a), f"allele function missing for {g} {a}"
    # Function-pair buckets; the spec's poor-metabolizer examples must land 'poor'.
    assert classify_genotype("CYP2C9 *2/*3")[2] == "poor"
    assert classify_genotype("CYP2C19 *2/*2")[2] == "poor"
    assert classify_genotype("CYP2D6 *4/*4")[2] == "poor"
    assert classify_genotype("CYP3A4 *22/*22")[2] == "poor"
    assert classify_genotype("CYP2C19 *2/*3")[2] == "poor"
    assert classify_genotype("CYP2C19 *1/*2")[2] == "intermediate"
    assert classify_genotype("CYP2C19 *1/*1")[2] == "normal"
    assert classify_genotype("CYP2C19 *1/*17")[2] == "rapid"
    assert PGX_DELTA["intermediate"] == PGX_DELTA["poor"] / 2

    adj = PatientRiskAdjustment()
    # Tier 1 fallback is an exact passthrough
    assert adj.adjust(0.42, None)["risk_score"] == 0.42
    assert adj.adjust(0.42, {"tier": 1})["tier_used"] == 1
    # Tier 2: age penalty raises risk, monotonic
    t2 = adj.adjust(0.42, {"tier": 2, "age": 68, "is_pregnant": False})
    assert t2["tier_used"] == 2 and t2["risk_score"] > 0.42
    assert t2["delta_demo"] == DELTA_AGE_65
    assert adj.adjust(0.42, {"tier": 2, "age": 40})["risk_score"] == 0.42
    # Tier 3: renal + PGx stacking; *2/*3 = poor = full weight
    t3 = adj.adjust(0.42, {"tier": 3, "age": 70, "egfr": 30.0, "cyp_genotype": "CYP2C9 *2/*3"})
    assert t3["tier_used"] == 3 and t3["delta_renal"] > 0 and t3["delta_pgx"] == 0.45
    assert t3["risk_score"] > t2["risk_score"]
    # Intermediate metabolizer gets half the pgx weight
    t_im = adj.adjust(0.42, {"tier": 3, "cyp_genotype": "CYP2C19 *1/*2"})
    assert t_im["delta_pgx"] == 0.225 and t_im["detail"]["pgx_phenotype"] == "intermediate"
    # Unknown alleles degrade to zero pgx delta, never to a wrong penalty
    assert adj.adjust(0.42, {"tier": 3, "cyp_genotype": "CYP2C9 *99/*99"})["delta_pgx"] == 0.0
    # Degradation: asks for tier 3 but gives no biomarkers -> falls back to the
    # highest tier with actual data; a totally empty profile degrades to 1.
    assert adj.adjust(0.42, {"tier": 3, "age": 70})["tier_used"] == 2
    assert adj.adjust(0.42, {"tier": 3, "egfr": None, "cyp_genotype": None})["tier_used"] == 1
    # Extreme probabilities must not blow up logit
    assert 0.0 <= adj.adjust(1.0, {"tier": 3, "egfr": 10, "cyp_genotype": "CYP2D6 *4/*4"})["risk_score"] <= 1.0
    assert adj.adjust(0.0, {"tier": 3, "egfr": 10})["risk_score"] >= 0.0
    print("personalization self-check OK; delta_age =", DELTA_AGE_65)
