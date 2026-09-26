"""Build data/tdc_drugbank_ddi.csv from the TDC DrugBank DDI dataset.

Downloads all known DrugBank drug-drug interaction pairs via PyTDC
(~191,808 pairs), binarizes the interaction-type label (any known type -> 1),
and adds a 1:1 set of sampled negatives: random drug pairs from the same drug
universe that are not known positives. Pairs are canonical (sorted DrugBank
IDs), so orientation plays no role. Output columns match DDIDataset/split_dataset:
drug1_name, drug1_smiles, drug2_name, drug2_smiles, interaction, severity, description.

Also writes data/drugs_tdc.json ("Name (DBxxxxx)" -> SMILES) for the website
drug picker; names are resolved from Wikidata (P715) unless --no-names.
"""
import argparse
import json
import os
import random
import re

import pandas as pd

_STEREO_PREFIX = re.compile(r"^\((?:[+\-]|[0-9]+[RS EZ0-9,]+)\)-|^rac-")


def _tidy(name):
    """Strip stereo/racemic prefixes and trademark marks:
    '(+)-pseudoephedrine' -> 'pseudoephedrine', 'Femara®' -> 'Femara'."""
    name = (name or "").replace("®", "").replace("™", "").replace("℠", "").strip()
    while True:
        stripped = _STEREO_PREFIX.sub("", name)
        if stripped == name:
            return name
        name = stripped


def _display(name):
    return name[0].upper() + name[1:]


def pick_name(label, aliases):
    """Clean common name for display.

    The Wikidata label is usually the generic name ('letrozole'); aliases carry
    brands ('Femara') and synonyms. So: use the tidied label whenever it is
    clean, and only score aliases when the label is a bracketed systematic
    name. ponytail: heuristic scoring (systematic tokens and all-caps
    abbreviations penalised). Returns None when nothing clean exists so the
    caller falls back to the bare accession. Upgrade path: curated overrides.
    """
    clean = _tidy(label)
    if clean and not any(ch in clean for ch in "[]()"):
        return _display(clean)
    scored = []
    for rank, cand in enumerate(aliases):
        c = _tidy(cand)
        if not c or any(ch in c for ch in "[]()"):
            continue
        systematic = len(re.findall(r"-yl\b|-ene\b|-oic\b|\d", c))
        shouty = 30 if c.isupper() else 0
        scored.append(((systematic * 100 + shouty + len(c), rank), c))
    if not scored:
        return None
    return _display(min(scored)[1])


def pubchem_names_for(db_ids):
    """Per-ID name fallback via PubChem PUG-REST, for the few IDs Wikidata misses."""
    import time
    import urllib.request

    out = {}
    for dbid in db_ids:
        time.sleep(0.5)  # unkeyed PUG-REST limit is 5 req/s
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{dbid}/property/Title/JSON"
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                props = json.load(r)["PropertyTable"]["Properties"]
            if props:
                name = pick_name(props[0].get("Title") or "", [])
                if name:
                    out[dbid] = name
        except Exception:
            continue
    return out


def wikidata_names(db_ids):
    """DrugBank accession -> clean common name from Wikidata (P715), one bulk query.

    ponytail: best-effort enrichment, IDs missing from Wikidata or without any
    clean name fall back to the bare accession label. Upgrade path: per-ID
    PubChem PUG-REST fallback.
    """
    import urllib.parse
    import urllib.request

    values = " ".join(f'"{i}"' for i in db_ids)
    query = (
        "SELECT ?db ?label ?alt WHERE { "
        f"VALUES ?db {{ {values} }} "
        "?item wdt:P715 ?db . ?item rdfs:label ?label . FILTER(LANG(?label)='en') "
        "OPTIONAL { ?item skos:altLabel ?alt . FILTER(LANG(?alt)='en') } }"
    )
    req = urllib.request.Request(
        "https://query.wikidata.org/sparql", data=query.encode("utf-8"), headers={
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/sparql-query",
            "User-Agent": "molecular-gnn-ddi-build/1.0 (dataset build; contact: local)",
        })
    with urllib.request.urlopen(req, timeout=120) as r:
        bindings = json.load(r)["results"]["bindings"]

    table = {}
    for b in bindings:
        e = table.setdefault(b["db"]["value"], {"label": b["label"]["value"], "aliases": []})
        alt = b.get("alt", {}).get("value")
        if alt and alt not in e["aliases"]:
            e["aliases"].append(alt)
    return {dbid: name for dbid, e in table.items()
            if (name := pick_name(e["label"], e["aliases"]))}



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tdc_drugbank_ddi.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-names", action="store_true",
                    help="skip Wikidata name lookup (labels stay bare DrugBank IDs)")
    a = ap.parse_args()

    from tdc.multi_pred import DDI
    data = DDI(name="DrugBank")
    df = data.get_data()  # columns: Drug1_ID, Drug1 (SMILES), Drug2_ID, Drug2 (SMILES), Y

    positives = {}
    smiles_by_id = {}
    for r in df.itertuples():
        pair = tuple(sorted((str(r.Drug1_ID), str(r.Drug2_ID))))
        positives.setdefault(pair, int(r.Y))
        smiles_by_id[pair[0]] = r.Drug1 if pair[0] == str(r.Drug1_ID) else r.Drug2
        smiles_by_id[pair[1]] = r.Drug2 if pair[1] == str(r.Drug2_ID) else r.Drug1

    drugs = sorted(smiles_by_id)
    rng = random.Random(a.seed)
    negatives = set()
    while len(negatives) < len(positives):
        d1, d2 = rng.sample(drugs, 2)
        pair = (d1, d2) if d1 < d2 else (d2, d1)
        if pair in positives or pair in negatives:
            continue
        negatives.add(pair)

    rows = []
    for (d1, d2), y in positives.items():
        rows.append({"drug1_name": d1, "drug1_smiles": smiles_by_id[d1],
                     "drug2_name": d2, "drug2_smiles": smiles_by_id[d2],
                     "interaction": 1, "severity": "Known DDI",
                     "description": f"DrugBank DDI (TDC type {y})"})
    for d1, d2 in negatives:
        rows.append({"drug1_name": d1, "drug1_smiles": smiles_by_id[d1],
                     "drug2_name": d2, "drug2_smiles": smiles_by_id[d2],
                     "interaction": 0, "severity": "None",
                     "description": "No known interaction (sampled negative)"})

    out = pd.DataFrame(rows).sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False)

    # Full drug universe (label -> SMILES) for the API/website drug picker.
    drugs_out = os.path.join(os.path.dirname(a.out) or ".", "drugs_tdc.json")
    names = {} if a.no_names else wikidata_names(list(smiles_by_id))
    if names:
        missing = sorted(set(smiles_by_id) - set(names))
        if missing:
            print(f"  resolving {len(missing)} IDs missing from Wikidata via PubChem...")
            names.update(pubchem_names_for(missing))
    labeled = {f"{names[i]} ({i})" if i in names else i: s for i, s in smiles_by_id.items()}
    with open(drugs_out, "w") as f:
        json.dump(dict(sorted(labeled.items())), f)

    print(f"Wrote {len(out)} rows -> {a.out}")
    print(f"  positives: {len(positives)} | negatives: {len(negatives)} | unique drugs: {len(drugs)}")
    print(f"  drug picker: {len(labeled)} entries ({len(names)} named via Wikidata) -> {drugs_out}")
    if names:
        # Verify before shipping: no systematic/bracketed names may leak into the UI.
        bad = [n for n in names.values() if any(ch in n for ch in "[]()")]
        assert not bad, f"unverified labels leaked: {bad[:5]}"
        print("  name check (random 15):")
        for i in sorted(random.sample(sorted(names), 15)):
            print(f"    {i} -> {names[i]}")
        longest = sorted(names, key=lambda i: -len(names[i]))[:8]
        print("  longest labels:")
        for i in longest:
            print(f"    {i} -> {names[i]}")
        leftover = sorted(set(smiles_by_id) - set(names))
        print(f"  unnamed (bare ID kept): {len(leftover)}: {leftover[:10]}")


if __name__ == "__main__":
    main()
