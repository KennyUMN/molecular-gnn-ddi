import torch
import numpy as np

# ---------------------------------------------------------------------------
# CYP450 / metabolic ground truth
# Ashby alerts cover DNA-reactive genotoxicants, not enzyme-mediated DDI, so
# the ground truth for the metabolic axis is a SMARTS panel of CYP inhibitor
# motifs plus RDKit's Brenk unwanted-substructure catalog.
# ---------------------------------------------------------------------------
CYP450_SMARTS = {
    # CYP3A4: azole N-heterocycles, benzofuran-type aromatics, tertiary amines
    "CYP3A4_imidazole": "c1cncn1",
    "CYP3A4_triazole": "n1cncn1",
    "CYP3A4_benzofuran": "o1ccc2ccccc12",
    "CYP3A4_tertiary_amine": "[NX3;H0;!$(N=*);!$(N-C=O);!$(N#*)]([#6])[#6]",
    # CYP2C9: carboxylate / aryl carboxylic acid and the coumarin core
    "CYP2C9_benzoic_acid": "c1ccccc1C(=O)O",
    "CYP2C9_carboxylate": "[CX3](=[OX1])[OX2H1]",
    "CYP2C9_coumarin": "O=c1ccc2ccccc2o1",
    # CYP1A2: flat poly-aromatics and flavone cores
    "CYP1A2_polyaromatic": "c1ccc2ccccc2c1",
    "CYP1A2_flavone": "O=c1cc(-c2ccccc2)oc2ccccc12",
    # CYP2D6: protonatable basic amine (distance rule applied on top, see below)
    "CYP2D6_basic_amine": "[NX3;H2,H1,H0;!$(NC=O);!$(N=*);+0]",
}

_COMPILED_SMARTS = {}
_BRENK_CATALOG = None


def _pattern(smarts):
    from rdkit import Chem
    if smarts not in _COMPILED_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            raise ValueError(f"invalid SMARTS pattern: {smarts}")
        _COMPILED_SMARTS[smarts] = patt
    return _COMPILED_SMARTS[smarts]


def get_brenk_catalog():
    from rdkit.Chem import FilterCatalog
    global _BRENK_CATALOG
    if _BRENK_CATALOG is None:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
        _BRENK_CATALOG = FilterCatalog.FilterCatalog(params)
    return _BRENK_CATALOG


def cyp2d6_basic_amine_hits(mol, min_bonds=3, max_bonds=5):
    """Basic amine held at a pharmacophore distance from an aromatic ring.

    ponytail: topological bond count proxies the 5-7 Angstrom criterion
    (3-5 bonds ~= 4.5-7.5 A on sp2/sp3 frameworks). Upgrade path: embed 3D
    conformers (Chem.EmbedMolecule) and measure the real N-to-ring-centroid
    distance when 3D data is available. The bond window is the calibration knob.
    """
    from rdkit import Chem
    basic_n = _pattern(CYP450_SMARTS["CYP2D6_basic_amine"])
    amines = sorted({a for match in mol.GetSubstructMatches(basic_n) for a in match})
    aromatics = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    if not amines or not aromatics:
        return []
    dist = Chem.GetDistanceMatrix(mol)
    hits = set()
    for n_idx in amines:
        for ar_idx in aromatics:
            if min_bonds <= dist[n_idx][ar_idx] <= max_bonds:
                hits.update((n_idx, ar_idx))
    return sorted(hits)


def match_cyp_substructures(smiles):
    """SMARTS-level CYP450 inhibitor / toxicophore ground truth for one molecule.

    Returns {'cyp_matches': {name: [atom_idx, ...]}, 'atom_indices': [...],
             'brenk_alerts': [alert_name, ...]}.
    """
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"cyp_matches": {}, "atom_indices": [], "brenk_alerts": []}

    cyp_matches = {}
    for name, smarts in CYP450_SMARTS.items():
        patt = _pattern(smarts)
        hits = sorted({a for match in mol.GetSubstructMatches(patt) for a in match})
        if name == "CYP2D6_basic_amine":
            hits = cyp2d6_basic_amine_hits(mol)
            if hits:
                name = "CYP2D6_basic_amine_5_7A"
        if hits:
            cyp_matches[name] = hits

    brenk_alerts = [entry.GetDescription() for entry in get_brenk_catalog().GetMatches(mol)]
    atom_indices = sorted({a for hits in cyp_matches.values() for a in hits})
    return {"cyp_matches": cyp_matches, "atom_indices": atom_indices, "brenk_alerts": brenk_alerts}


# ---------------------------------------------------------------------------
# Attribution drift metrics (FP32 vs INT8)
# KL-divergence is invalid here: gradient-saliency attributions are L2 norms
# (non-negative, not a probability distribution), so a probability divergence
# on them is undefined. Rank agreement (Spearman rho) and top-k set overlap
# (Jaccard) are the valid pair.
# ---------------------------------------------------------------------------
def _rankdata(values):
    """Average-rank transform (numpy-only, ties handled)."""
    values = np.asarray(values, dtype=np.float64)
    order = values.argsort(kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    # average tied ranks
    sorted_vals = values[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def spearman_rank_correlation(attr_fp32, attr_int8):
    """Rank correlation between FP32 and INT8 atom importance vectors."""
    a = np.asarray(attr_fp32, dtype=np.float64).ravel()
    b = np.asarray(attr_int8, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.size < 2:
        return 1.0
    ra, rb = _rankdata(a), _rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        # Constant attribution vector: ranks carry no information.
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def top_k_jaccard_overlap(attr_fp32, attr_int8, k=5):
    """Jaccard overlap of the k highest-attribution atoms of FP32 vs INT8."""
    a = np.asarray(attr_fp32, dtype=np.float64).ravel()
    b = np.asarray(attr_int8, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    k = min(k, a.size)
    if k == 0:
        return 1.0
    top_a = set(np.argsort(-np.abs(a), kind="mergesort")[:k].tolist())
    top_b = set(np.argsort(-np.abs(b), kind="mergesort")[:k].tolist())
    return len(top_a & top_b) / len(top_a | top_b)


def attribution_drift(attr_fp32, attr_int8, k=5):
    return {
        "spearman_rho": spearman_rank_correlation(attr_fp32, attr_int8),
        "top_k_jaccard": top_k_jaccard_overlap(attr_fp32, attr_int8, k=k),
        "k": k,
    }


def attribution_drift_report(pairs, k=5):
    """pairs: iterable of (fp32_vector, int8_vector). Aggregates over a test set.

    Percentiles and the >= k+1 atom subset are reported alongside the mean: a
    molecule with 2 atoms produces rho = -1 by construction (two ranks, swapped)
    and would otherwise dominate the minimum.
    """
    reports, sizes = [], []
    for a, b in pairs:
        reports.append(attribution_drift(a, b, k=k))
        sizes.append(len(np.asarray(a).ravel()))
    if not reports:
        return {"n_pairs": 0, "mean_spearman_rho": None, "min_spearman_rho": None,
                "mean_top_k_jaccard": None, "passes_rho_0.85": False}
    rhos = np.array([r["spearman_rho"] for r in reports])
    jac = np.array([r["top_k_jaccard"] for r in reports])
    sizes = np.array(sizes)
    big = sizes >= k + 1
    return {
        "n_pairs": len(reports),
        "mean_spearman_rho": float(rhos.mean()),
        "min_spearman_rho": float(rhos.min()),
        "p05_spearman_rho": float(np.percentile(rhos, 5)),
        "p50_spearman_rho": float(np.percentile(rhos, 50)),
        "mean_top_k_jaccard": float(jac.mean()),
        "n_pairs_ge_k_plus_1_atoms": int(big.sum()),
        "mean_spearman_rho_ge_k+1_atoms": float(rhos[big].mean()) if big.any() else None,
        "min_spearman_rho_ge_k+1_atoms": float(rhos[big].min()) if big.any() else None,
        "passes_rho_0.85": bool(rhos.mean() >= 0.85),
        "per_pair": reports,
    }


def explain_ddi_interaction(model, g1, g2, top_k=3):
    """
    Computes vanilla gradient saliency (a single backward pass on the input
    node features - no baseline, no integration steps) to determine which
    atoms in Drug A and Drug B are most responsible for the predicted interaction.
    """
    model.eval()
    
    # Enable gradient tracking on node features
    x1 = g1['x'].clone().detach().requires_grad_(True)
    x2 = g2['x'].clone().detach().requires_grad_(True)

    g1_clone = {k: v for k, v in g1.items()}
    g2_clone = {k: v for k, v in g2.items()}
    g1_clone['x'] = x1
    g2_clone['x'] = x2

    model.zero_grad()
    out = model(g1_clone, g2_clone)
    prob = out['prob']
    # Backprop the pre-sigmoid logit: sigmoid saturation kills the gradient
    # exactly when the model is confident (audit finding).
    out['logits'].backward()

    # Atom-level saliency: L2 norm of feature gradients
    saliency_1 = x1.grad.norm(dim=-1).detach().cpu().numpy()
    saliency_2 = x2.grad.norm(dim=-1).detach().cpu().numpy()

    # Normalize to [0, 1]
    if saliency_1.max() > 0:
        saliency_1 = saliency_1 / saliency_1.max()
    if saliency_2.max() > 0:
        saliency_2 = saliency_2 / saliency_2.max()

    top_atoms_1 = np.argsort(saliency_1)[::-1][:top_k].tolist()
    top_atoms_2 = np.argsort(saliency_2)[::-1][:top_k].tolist()

    return {
        "prediction_prob": float(prob.item()),
        "risk_level": "High" if prob.item() > 0.7 else ("Medium" if prob.item() > 0.4 else "Low"),
        "drug1_atom_importance": saliency_1.tolist(),
        "drug2_atom_importance": saliency_2.tolist(),
        "drug1_highlight_atoms": top_atoms_1,
        "drug2_highlight_atoms": top_atoms_2
    }

def draw_highlighted_molecule(smiles, highlight_atom_indices, output_filepath):
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        # Highlight top atoms in soft red/coral
        highlight_colors = {idx: (0.9, 0.2, 0.2) for idx in highlight_atom_indices}
        
        drawer = Draw.rdMolDraw2D.MolDraw2DCairo(400, 300)
        Draw.rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, mol, highlightAtoms=highlight_atom_indices,
            highlightAtomColors=highlight_colors
        )
        drawer.FinishDrawing()
        with open(output_filepath, "wb") as f:
            f.write(drawer.GetDrawingText())
        return True
    except Exception as e:
        print(f"RDKit visualization note: {e}")
        return False


if __name__ == "__main__":
    # Ground truth panel: motifs that a DDI XAI explanation must recover.
    panel = {
        "warfarin": "CC(=O)CC(C1=CC=CC=C1)C2=C(O)C3=CC=CC=C3OC2=O",        # coumarin -> CYP2C9
        "amiodarone": "CCCc1c(oc2ccccc12)C(=O)c1cc(I)c(OCCN(CC)CC)c(I)c1",  # benzofuran + amine -> CYP3A4/2D6
        "fluconazole": "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F",                # triazole -> CYP3A4
        "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",                              # benzoic acid -> CYP2C9
    }
    expected = {
        "warfarin": "CYP2C9_coumarin",
        "amiodarone": "CYP3A4_benzofuran",
        "fluconazole": "CYP3A4_triazole",
        "aspirin": "CYP2C9_benzoic_acid",
    }
    for name, smiles in panel.items():
        res = match_cyp_substructures(smiles)
        assert res["cyp_matches"], f"no CYP motif found for {name}"
        assert expected[name] in res["cyp_matches"], f"{name}: expected {expected[name]}, got {sorted(res['cyp_matches'])}"
    # coumarin pattern must not fire on a benzofuran, and vice versa
    assert "CYP2C9_coumarin" not in match_cyp_substructures(panel["amiodarone"])["cyp_matches"]
    assert "CYP3A4_benzofuran" not in match_cyp_substructures(panel["warfarin"])["cyp_matches"]
    # 2D6 needs the amine and an aromatic ring within the bond window
    assert "CYP2D6_basic_amine_5_7A" in match_cyp_substructures(panel["amiodarone"])["cyp_matches"]

    # Drift metrics behave: identity -> rho 1 / jaccard 1, reversed -> rho -1
    v = np.array([0.3, 0.9, 0.1, 0.7, 0.5, 0.2])
    assert abs(spearman_rank_correlation(v, v) - 1.0) < 1e-9
    assert abs(spearman_rank_correlation(v, -v) + 1.0) < 1e-9
    assert top_k_jaccard_overlap(v, v, k=3) == 1.0
    assert top_k_jaccard_overlap(v, np.zeros_like(v), k=3) < 1.0
    assert attribution_drift_report([(v, v)])["passes_rho_0.85"]
    print("explain self-check OK")
