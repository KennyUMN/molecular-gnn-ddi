"""Calibrate Stage 2 personalization deltas from real sources. Writes data/delta_calibration.json.

Two sources:
  1. PGx allele functions: ClinPGx (ex-PharmGKB) REST API, haplotype records per
     gene carry `functionTerm` (Normal/Decreased/No/Increased function, CPIC
     sourced). api.pharmgkb.org is being retired 2026-07-20, api.clinpgx.org is
     the live hostname.
  2. Demographic delta: openFDA FAERS. For reports co-mentioning known
     interacting pairs, the reporting odds ratio of a serious outcome
     (hospitalisation/death) in age >= 65 vs age 0-64 gives delta_age = ln(ROR).

The phenotype bucket from an allele-function pair is NOT taken from a database
(CPIC diplotype tables are per-guideline PDFs): it is the transparent rule set in
src/personalization.py, asserted against the poor-metabolizer examples in
MASTER_PROJECT_SPECIFICATION.md 3.3.

Output: data/delta_calibration.json, consumed by src/personalization.py.
"""
import json
import math
import os
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "molecular-gnn-ddi-build/1.0 (dataset build; contact: local)"}
CLINPGX = "https://api.clinpgx.org/v1/data"
OPENFDA = "https://api.fda.gov/drug/event.json"
GENES = ["CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]
# Known interacting pairs used to pool the FAERS age stratum (generic_name form;
# note US naming: paracetamol == acetaminophen).
PAIRS = [
    ("warfarin", "aspirin"),
    ("warfarin", "acetaminophen"),
    ("warfarin", "ibuprofen"),
    ("simvastatin", "amiodarone"),
    ("clopidogrel", "omeprazole"),
    ("digoxin", "amiodarone"),
]

_FUNC_MAP = {
    "Normal function": "normal",
    "Decreased function": "decreased",
    "No function": "no_function",
    "Increased function": "increased",
}


def _get_json(url, params=None, headers=UA, timeout=60, attempts=3):
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(3 * (i + 1))


def pull_allele_functions():
    """{gene: {"*2": "decreased", ...}} from ClinPGx haplotypes (functionTerm)."""
    out = {}
    for gene in GENES:
        data = _get_json(f"{CLINPGX}/haplotype", {"gene.symbol": gene})["data"]
        table = {}
        for h in data:
            term = (h.get("functionTerm") or {}).get("term")
            fn = _FUNC_MAP.get(term, "unknown")
            if fn == "unknown":
                continue  # None / Uncertain function: not usable as a risk input
            name = (h.get("name") or h.get("symbol") or "").strip()
            if name.startswith(gene):
                name = name[len(gene):]
            if name:
                table[name] = fn
        out[gene] = dict(sorted(table.items()))
        print(f"  {gene}: {len(table)} function-mapped alleles")
        time.sleep(1)
    return out


def _openfda_total(search):
    d = _get_json(OPENFDA, {"search": search, "limit": 1})
    return d["meta"]["results"]["total"]


def pull_faers_age_ror():
    """2x2 (outcome x age stratum) pooled over known pairs -> ln(ROR) per endpoint.

    Two endpoints are measured: any serious outcome (serious:1) and fatal
    outcome (seriousnessdeath:1). openFDA has no negation search, so the
    no-event cell is stratum total minus the event count. Crude reporting odds,
    both endpoints reported unfiltered; the caller decides which matches the
    delta's semantics.
    """
    pair_expr = " OR ".join(
        f'(patient.drug.openfda.generic_name:"{a}" AND patient.drug.openfda.generic_name:"{b}")'
        for a, b in PAIRS)
    age_exprs = {"age_65_plus": "patient.patientonsetage:[65 TO 200]",
                 "age_0_64": "patient.patientonsetage:[0 TO 64]"}
    totals = {}
    for age_lbl, age_expr in age_exprs.items():
        totals[age_lbl] = _openfda_total(f"({pair_expr}) AND {age_expr}")
        print(f"  {age_lbl:10s} total           : {totals[age_lbl]}")
        time.sleep(1.2)  # openFDA unkeyed limit is 40 req/min
    out = {}
    for ep_lbl, event in (("serious_outcome", "serious:1"),
                          ("fatal_outcome", "seriousnessdeath:1")):
        ev = {}
        for age_lbl, age_expr in age_exprs.items():
            ev[age_lbl] = _openfda_total(f"({pair_expr}) AND {age_expr} AND {event}")
            print(f"  {ep_lbl:15s} {age_lbl:10s} event  : {ev[age_lbl]}")
            time.sleep(1.2)
        cells = {
            "age_65_plus__event": ev["age_65_plus"],
            "age_65_plus__no_event": totals["age_65_plus"] - ev["age_65_plus"],
            "age_0_64__event": ev["age_0_64"],
            "age_0_64__no_event": totals["age_0_64"] - ev["age_0_64"],
        }
        a, b = cells["age_65_plus__event"], cells["age_65_plus__no_event"]
        c, d = cells["age_0_64__event"], cells["age_0_64__no_event"]
        # Haldane-Anscombe correction keeps the ROR defined with any zero cell.
        ror = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        out[ep_lbl] = {
            "cells": cells,
            "ror_age65_vs_0_64": round(ror, 4),
            "ln_ror": round(math.log(ror), 4),
            "n_reports_with_age": a + b + c + d,
        }
    return out


def main():
    out = {
        "retrieved": time.strftime("%Y-%m-%d"),
        "sources": {
            "pgx_allele_function": "https://api.clinpgx.org/v1/data/haplotype?gene.symbol=GENE (functionTerm, CPIC sourced)",
            "faers_age": "https://api.fda.gov/drug/event.json counts (pairs: %s)" % ", ".join("+".join(p) for p in PAIRS),
        },
    }
    print("PGx allele functions (ClinPGx):")
    out["pgx_allele_function"] = pull_allele_functions()
    print("FAERS age stratum (openFDA, pooled known pairs):")
    out["faers_age"] = pull_faers_age_ror()

    path = os.path.join(os.path.dirname(__file__), "delta_calibration.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"Wrote {path}")
    faers = out["faers_age"]
    for ep in faers:
        print(f"  {ep}: ROR = {faers[ep]['ror_age65_vs_0_64']} (ln = {faers[ep]['ln_ror']}), n = {faers[ep]['n_reports_with_age']}")


if __name__ == "__main__":
    main()
