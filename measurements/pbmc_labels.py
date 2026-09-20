#!/usr/bin/env python3
"""Marker-derived PBMC cell types for the macaque and rat corpora, which ship no author annotation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "model"), str(REPO / "eval"), str(REPO / "target" / "py")]

from scrm_eval import metrics, registry  # noqa: E402

P = registry.PROJ

PANEL_UPPER = {
    "T cell":     ["CD3D", "CD3E", "CD3G", "CD2", "TRAC", "IL7R", "LCK"],
    "NK cell":    ["NKG7", "GNLY", "KLRD1", "PRF1", "NCR1", "KLRB1"],
    "B cell":     ["MS4A1", "CD79A", "CD79B", "BANK1", "CD19", "IGHM"],
    "monocyte":   ["LYZ", "CD14", "S100A8", "S100A9", "FCN1", "VCAN", "CSF1R"],
    "dendritic cell": ["FCER1A", "CST3", "IRF8", "FLT3", "CLEC9A"],
    "platelet":   ["PPBP", "PF4", "ITGA2B", "GP9"],
}
PANEL_TITLE = {
    "T cell":     ["Cd3d", "Cd3e", "Cd3g", "Cd2", "Trac", "Il7r", "Lck"],
    "NK cell":    ["Nkg7", "Gnly", "Klrd1", "Prf1", "Ncr1", "Klrb1"],
    "B cell":     ["Ms4a1", "Cd79a", "Cd79b", "Bank1", "Cd19", "Ighm"],
    "monocyte":   ["Lyz2", "Cd14", "S100a8", "S100a9", "Fcn1", "Vcan", "Csf1r"],
    "dendritic cell": ["Fcer1a", "Cst3", "Irf8", "Flt3", "Clec9a"],
    "platelet":   ["Ppbp", "Pf4", "Itga2b", "Gp9"],
}
PANELS = {"macaque": PANEL_UPPER, "rat": PANEL_TITLE, "human": PANEL_UPPER, "mouse": PANEL_TITLE}


def main(argv) -> int:
    import scanpy as sc
    from scrm_eval import targets as T

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.05,
                    help="a cluster's best score must beat the runner-up by this, else `unassigned`")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--restrict-to", default=None,
                    help="comma-separated lineages to keep, for a task MATCHED across species. "
                         "Human PBMC supports 6 types over the 100-cell floor while macaque and rat "
                         "support 4, and protocol B accuracy depends on the number of classes -- so "
                         "comparing 6-way human against 4-way macaque would confound species with "
                         "task difficulty. Restricting all three to the shared four makes the "
                         "comparison about the species, which is the only reason to make it.")
    ap.add_argument("--suffix", default="", help="written as marker_labels{suffix}.json")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv[1:])

    species = T.SPECIES[a.dataset]
    panel = PANELS.get(species)
    if panel is None:
        print(f"no marker panel for species {species!r}", file=sys.stderr)
        return 2

    ad = T.load_counts(a.dataset)
    ad.var_names_make_unique()
    n0 = ad.n_obs
    sc.pp.filter_cells(ad, min_genes=200)
    sc.pp.filter_genes(ad, min_cells=3)
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    present = {k: [g for g in v if g in set(ad.var_names)] for k, v in panel.items()}
    print(f"{a.dataset}  species={species}  {n0} -> {ad.n_obs} cells after QC, {ad.n_vars} genes")
    for k, v in present.items():
        print(f"    {k:16} {len(v)}/{len(panel[k])} markers: {v}")
    if any(len(v) < 2 for v in present.values()):
        print("REFUSING: a lineage has fewer than 2 markers; the score would be a single gene",
              file=sys.stderr)
        return 3

    ad.layers["lognorm"] = ad.X.copy()
    sc.pp.highly_variable_genes(ad, n_top_genes=2000)
    ad = ad[:, ad.var.highly_variable].copy()
    sc.pp.scale(ad, max_value=10)
    sc.tl.pca(ad, n_comps=50, svd_solver="arpack", random_state=a.seed)
    sc.pp.neighbors(ad, n_neighbors=15, random_state=a.seed)
    # metrics.graph_cluster_labels, not sc.tl.leiden: every scanpy Leiden flavour needs
    # leidenalg or python-igraph, both GPL, and this project is MIT. Same graph, same seed.
    ad.obs["leiden"] = pd.Categorical(
        [str(v) for v in metrics.graph_cluster_labels(
            ad.obsp["connectivities"], resolution=a.resolution, seed=a.seed)])

    full = T.load_counts(a.dataset)
    full.var_names_make_unique()
    full = full[ad.obs_names].copy()
    sc.pp.normalize_total(full, target_sum=1e4)
    sc.pp.log1p(full)
    for k, v in present.items():
        sc.tl.score_genes(full, v, score_name=f"s::{k}", random_state=a.seed)
    full.obs["leiden"] = ad.obs["leiden"].values

    types = list(present)
    per_cluster, mapping = {}, {}
    for cl in sorted(full.obs["leiden"].unique(), key=lambda x: int(x)):
        m = full.obs["leiden"] == cl
        sco = {t: float(full.obs.loc[m, f"s::{t}"].mean()) for t in types}
        order = sorted(sco, key=lambda t: -sco[t])
        best, second = order[0], order[1]
        gap = sco[best] - sco[second]
        lab = best if gap >= a.margin else "unassigned"
        mapping[cl] = lab
        per_cluster[cl] = {"n_cells": int(m.sum()), "scores": sco,
                           "best": best, "runner_up": second, "margin": gap, "assigned": lab}
        print(f"  cluster {cl:>3} n={int(m.sum()):5}  {lab:16} "
              f"(best {best} {sco[best]:+.3f}, next {second} {sco[second]:+.3f}, gap {gap:+.3f})")

    labels = {bc.split("-")[0]: mapping[cl] for bc, cl in zip(full.obs_names, full.obs["leiden"])}
    labels = {k: v for k, v in labels.items() if v != "unassigned"}
    if a.restrict_to:
        keep_t = {x.strip() for x in a.restrict_to.split(",")}
        unknown = keep_t - set(types)
        if unknown:
            print(f"REFUSING: --restrict-to names lineages not in the panel: {sorted(unknown)}",
                  file=sys.stderr)
            return 5
        before = len(labels)
        labels = {k: v for k, v in labels.items() if v in keep_t}
        print(f"  --restrict-to {sorted(keep_t)}: {before} -> {len(labels)} cells")
    from collections import Counter
    counts = Counter(labels.values())
    print(f"\n  {len(labels)} labelled cells, {len(counts)} types")
    for t, n in counts.most_common():
        print(f"    {t:16} {n:5}  {'OK for protocol B' if n >= 100 else 'below the 100-cell floor'}")
    keep = [t for t, n in counts.items() if n >= 100]
    print(f"  types clearing protocol B's floor: {len(keep)} -> {sorted(keep)}")
    if len(keep) < 2:
        print("REFUSING: fewer than 2 types clear the floor; a 1-class probe is not a probe",
              file=sys.stderr)
        return 4

    out = P / "raw" / a.dataset / f"marker_labels{a.suffix}.json"
    prov = {
        "_what": "MARKER-DERIVED cell types, not an author annotation. See measurements/pbmc_labels.py.",
        "_circularity": ("Labels come from graph clustering on these counts. count_scanpy is PCA on the same "
                         "matrix, so its accuracy here is partly circular and is an upper bound. "
                         "scrm reads bytes and the count FMs use their own vocabularies, so they "
                         "are not circular on these labels."),
        "dataset": a.dataset, "species": species, "resolution": a.resolution,
        "margin": a.margin, "seed": a.seed,
        "n_cells_matrix": n0, "n_cells_after_qc": int(ad.n_obs), "n_labelled": len(labels),
        "markers_used": present, "markers_declared": panel,
        "clusters": per_cluster, "types_over_100": sorted(keep),
    }
    if a.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(labels, indent=0))
        tmp.replace(out)
        pv = out.with_name(f"marker_labels{a.suffix}.provenance.json")
        ptmp = pv.with_suffix(".json.tmp")
        ptmp.write_text(json.dumps(prov, indent=2))
        ptmp.replace(pv)
        print(f"\n  wrote {out}\n  wrote {pv}")
    else:
        print("\n  (dry run; pass --write to save)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
