"""E6/E9 target construction: count-matrix-derived prediction targets, aligned to corpus cells."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics, registry

CACHE_DIR = Path(os.environ.get(
    "SCRM_TARGET_CACHE", "") or registry.PROJ / "eval_artifacts" / "target_cache")
_PANEL_KEY_FIELDS = ("panel", "digest", "top_k", "resolution", "seed", "alpha",
                     "test", "correction", "tie_correct", "min_cells", "graph")

#: The graph the marker panels are clustered on. `algo` is part of this dict on purpose: `graph`
#: is one of `_PANEL_KEY_FIELDS`, so changing the algorithm changes every panel key and no cached
#: panel built by the previous algorithm can be served for the new one.
CLUSTER_GRAPH = {"target_sum": 1e4, "n_comps_max": 50, "n_neighbors": 15,
                 "algo": metrics.CLUSTER_ALGO, "directed": False}


def _counts_digest(adata) -> str:
    """Identify the matrix itself, so a cache entry cannot outlive the data it was built from."""
    h = hashlib.sha256()
    h.update(f"{adata.n_obs}x{adata.n_vars}".encode())
    h.update("\x00".join(map(str, adata.var_names)).encode())
    h.update("\x00".join(map(str, adata.obs_names)).encode())
    x = adata.X
    nnz = int(x.nnz) if hasattr(x, "nnz") else int(np.count_nonzero(x))
    h.update(f"{float(x.sum()):.6f}:{nnz}".encode())
    return h.hexdigest()[:20]


def _panel_key(**parts) -> str:
    unknown = set(parts) - set(_PANEL_KEY_FIELDS)
    if unknown:
        raise TypeError(
            f"_panel_key got fields not in the cache key: {sorted(unknown)}. Add them to "
            f"_PANEL_KEY_FIELDS -- a parameter outside the key means a stale panel is served silently."
        )
    body = json.dumps({k: parts[k] for k in sorted(parts)}, sort_keys=True, default=str)
    return hashlib.sha256(body.encode()).hexdigest()[:24]


def _cached_panel(key: str, label: str, compute):
    """Return `(indices, labels, prov)` from disk, or compute it and store it atomically."""
    f = CACHE_DIR / f"{key}.npz"
    if f.exists():
        try:
            z = np.load(f, allow_pickle=False)
            prov = json.loads(str(z["prov_json"]))
            prov["from_cache"] = True
            print(f"[targets] {label}: reusing cached panel {f.name}", flush=True)
            return z["indices"], z["labels"], prov
        except Exception as e:
            print(f"[targets] {label}: cache {f.name} unreadable ({e}); recomputing", flush=True)
    t0 = time.monotonic()
    print(f"[targets] {label}: computing (no cache entry) ...", flush=True)
    indices, labels, prov = compute()
    print(f"[targets] {label}: done in {time.monotonic() - t0:.0f}s "
          f"-> {len(indices)} genes", flush=True)
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f.name + f".tmp{os.getpid()}.npz")
        np.savez(tmp, indices=np.asarray(indices), labels=np.asarray(labels),
                 prov_json=np.array(json.dumps(prov)))
        os.replace(tmp, f)
    except OSError as e:
        print(f"[targets] {label}: could not write cache ({e}); continuing", flush=True)
    return indices, labels, prov


class TargetError(Exception):
    """A target set that cannot be built correctly."""


MIN_CELLS = 20


def load_counts(dataset_or_dir):
    """A 10x matrix as AnnData (cells x genes), v2 (`genes.tsv`) or v3 (`features.tsv.gz`) layout."""
    import scanpy as sc

    path = registry.count_matrices().get(dataset_or_dir, Path(dataset_or_dir))
    if str(path).endswith((".h5ad", ".h5")):
        if not Path(path).is_file():
            raise TargetError(f"no count matrix file at {path}")
        adata = (sc.read_10x_h5(path) if str(path).endswith(".h5")
                 else sc.read_h5ad(path))
        adata.var_names_make_unique()
    else:
        if not Path(path).is_dir():
            raise TargetError(f"no count matrix directory at {path}")
        adata = sc.read_10x_mtx(path, gex_only=True)
    adata.var_names_make_unique()

    if isinstance(dataset_or_dir, str) and dataset_or_dir in registry.COUNT_MATRICES:
        from . import controls

        _, _, cells = controls.corpus_facts(dataset_or_dir)
        keep = set(cells)
        mask = [b.split("-")[0] in keep for b in adata.obs_names]
        n = int(sum(mask))
        if n == 0:
            raise TargetError(f"{dataset_or_dir}: no matrix barcode matches a corpus cell")
        if n < len(mask):
            print(f"[targets] {dataset_or_dir}: {n:,} of {len(mask):,} matrix barcodes are corpus "
                  f"cells; targets are built on those", flush=True)
            adata = adata[mask].copy()

    n_before = adata.n_vars
    import scanpy as sc  # noqa: F811  (local, mirrors the import above)

    sc.pp.filter_genes(adata, min_cells=MIN_CELLS)
    if adata.n_vars < n_before:
        print(f"[targets] genes {n_before:,} -> {adata.n_vars:,} at min_cells={MIN_CELLS}", flush=True)
    return adata


def _bare(barcodes) -> list[str]:
    """10x barcodes carry a '-1' suffix; corpus cell names are the bare 16 bp."""
    return [b.split("-")[0] for b in barcodes]


@dataclass
class TargetSet:
    """Targets aligned to a caller-supplied cell order (the artifact's), with provenance."""

    matrix: np.ndarray
    gene_names: list[str]
    has_counts: np.ndarray
    panel: str
    provenance: dict = field(default_factory=dict)


def align_to_cells(adata, cells: list[str]):
    """Row indices of `cells` in `adata`, and the has_counts mask for cells it lacks."""
    by_bare = {b: i for i, b in enumerate(_bare(adata.obs_names))}
    rows = np.array([by_bare.get(c.split("-")[0], -1) for c in cells], dtype=np.int64)
    return rows, rows >= 0


def hvg_panel(adata, n_top: int = 2000) -> np.ndarray:
    """E6 panel (a): seurat_v3 highly-variable genes on the raw counts, and nothing but the HVGs."""
    import scanpy as sc

    a = adata.copy()
    sc.pp.highly_variable_genes(a, n_top_genes=min(n_top, a.n_vars), flavor="seurat_v3")
    return np.flatnonzero(a.var["highly_variable"].to_numpy())


def qc_mask(adata, n_mads: float = 5.0):
    """Ch. 2's MAD outlier rule: which cells a standard QC pass would KEEP, and why."""
    import numpy as np
    import scanpy as sc

    a = adata.copy()
    mito = [g for g in a.var_names if str(g).lower().startswith(("mt-", "mt."))]
    a.var["mt"] = a.var_names.isin(mito)
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], inplace=True, percent_top=None, log1p=True)

    def outlier(col, k=n_mads):
        x = np.asarray(a.obs[col], dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) or 1e-12
        return np.abs(x - med) > k * 1.4826 * mad

    bad = outlier("log1p_total_counts") | outlier("log1p_n_genes_by_counts")
    if "pct_counts_mt" in a.obs:
        bad |= outlier("pct_counts_mt", 3.0)
    keep = ~bad
    return keep, {"rule": f"MAD>{n_mads} on log1p counts/genes, MAD>3 on pct_mt",
                  "n_mito_genes": len(mito), "n_kept": int(keep.sum()), "n_cells": int(len(keep))}


def hvg_panel_pearson(adata, n_top: int = 2000) -> np.ndarray:
    """E6 panel (b): HVGs by ANALYTIC PEARSON RESIDUALS (Ch. 6's recommended selection)."""
    import scanpy as sc

    a = adata.copy()
    sc.experimental.pp.highly_variable_genes(
        a, flavor="pearson_residuals", n_top_genes=min(n_top, a.n_vars))
    return np.flatnonzero(a.var["highly_variable"].to_numpy())


def _cluster_labels_and_dense(adata, resolution: float, seed: int):
    """`(cluster_labels, dense log-normalised matrix)`, computed once per (adata, resolution, seed)."""
    import scanpy as sc

    g = CLUSTER_GRAPH
    key = ("_cluster_dense", float(resolution), int(seed), tuple(sorted(g.items())))
    cached = adata.uns.get(key) if hasattr(adata, "uns") else None
    if cached is not None:
        return cached

    a = adata.copy()
    sc.pp.normalize_total(a, target_sum=g["target_sum"])
    sc.pp.log1p(a)
    sc.pp.pca(a, n_comps=min(g["n_comps_max"], a.n_obs - 1, a.n_vars - 1), random_state=seed)
    sc.pp.neighbors(a, random_state=seed, n_neighbors=g["n_neighbors"])
    labels = metrics.graph_cluster_labels(a.obsp["connectivities"], resolution=resolution, seed=seed)
    x = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    if hasattr(adata, "uns"):
        adata.uns[key] = (labels, x)
    return labels, x


def leiden_marker_panel_wilcoxon(adata, *, top_k: int = 50, resolution: float = 1.0,
                                 seed: int = 0, alpha: float = 0.05):
    """Per-cluster markers by WILCOXON RANK-SUM with BH correction — Ch. 11's recommended test."""
    def compute():
        import scanpy as sc

        labels, _ = _cluster_labels_and_dense(adata, resolution, seed)
        a = adata.copy()
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        a.obs["leiden"] = pd.Categorical([str(v) for v in labels])
        n_clusters = len(set(labels.tolist()))
        print(f"[targets] wilcoxon DE: {a.n_vars} genes x {n_clusters} clusters, "
              f"{a.n_obs} cells (single-threaded, no progress from scanpy)", flush=True)
        sc.tl.rank_genes_groups(a, "leiden", method="wilcoxon", tie_correct=True, pts=False)
        r = a.uns["rank_genes_groups"]
        chosen: set[int] = set()
        n_sig = {}
        pos = {g: i for i, g in enumerate(adata.var_names)}
        for grp in r["names"].dtype.names:
            names = r["names"][grp]
            padj = r["pvals_adj"][grp]
            keep = [nm for nm, q in zip(names, padj) if q is not None and q == q and q < alpha]
            n_sig[grp] = len(keep)
            for nm in keep[:top_k]:
                if nm in pos:
                    chosen.add(pos[nm])
        prov = {"top_k": top_k, "resolution": resolution, "seed": seed,
                "n_clusters": len(r["names"].dtype.names), "test": "wilcoxon",
                "correction": "benjamini-hochberg", "alpha": alpha,
                "tie_correct": True, "n_significant_per_cluster": n_sig}
        return np.array(sorted(chosen), dtype=np.int64), labels, prov

    key = _panel_key(panel="cluster_markers_wilcoxon", digest=_counts_digest(adata),
                     top_k=top_k, resolution=resolution, seed=seed, alpha=alpha,
                     test="wilcoxon", correction="benjamini-hochberg", tie_correct=True,
                     graph=CLUSTER_GRAPH)
    return _cached_panel(key, "leiden_markers (wilcoxon+BH)", compute)


def leiden_marker_panel(adata, *, top_k: int = 50, resolution: float = 1.0, seed: int = 0):
    """E6 panel (b), the legacy definition: per-Leiden-cluster top-K one-vs-rest F-statistic
    genes (union over clusters), computed on log1p-normalised counts. Returns (gene_indices,
    leiden_labels, provenance).
    """
    from sklearn.feature_selection import f_classif

    labels, x = _cluster_labels_and_dense(adata, resolution, seed)
    chosen: set[int] = set()
    for cluster in np.unique(labels):
        f, _ = f_classif(x, (labels == cluster).astype(int))
        f = np.nan_to_num(f, nan=0.0)
        chosen.update(np.argsort(f)[::-1][:top_k].tolist())
    prov = {"top_k": top_k, "resolution": resolution, "seed": seed, "n_clusters": int(labels.max()) + 1}
    return np.array(sorted(chosen), dtype=np.int64), labels, prov


def binary_expression_targets(
    adata, cells: list[str], gene_idx: np.ndarray, *, panel: str, threshold: int = 0
) -> TargetSet:
    """E6: detected-or-not (count > `threshold`) for the panel genes, in the caller's cell order."""
    rows, has = align_to_cells(adata, cells)
    x = adata.X[:, gene_idx]
    x = (x > threshold).toarray() if hasattr(x, "toarray") else np.asarray(x) > threshold
    out = np.zeros((len(cells), len(gene_idx)), dtype=bool)
    out[has] = x[rows[has]]
    return TargetSet(
        matrix=out,
        gene_names=[adata.var_names[i] for i in gene_idx],
        has_counts=has,
        panel=panel,
        provenance={"detection_threshold": threshold},
    )


def raw_count_targets(adata, cells: list[str], gene_idx: np.ndarray, *, panel: str) -> TargetSet:
    """RAW integer counts for the panel genes — the target `count_regression_probe` needs."""
    rows, has = align_to_cells(adata, cells)
    x = adata.X[:, gene_idx]
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    out = np.zeros((len(cells), len(gene_idx)), dtype=np.float32)
    out[has] = x[rows[has]]
    return TargetSet(matrix=out, gene_names=[adata.var_names[i] for i in gene_idx],
                     has_counts=has, panel=panel, provenance={"transform": "none (raw counts)"})


def expression_targets(adata, cells: list[str], gene_idx: np.ndarray, *, panel: str) -> TargetSet:
    """E6b: GRADED expression (log1p of CP10K-normalised counts) for the panel genes."""
    import scanpy as sc

    a = adata.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    rows, has = align_to_cells(a, cells)
    x = a.X[:, gene_idx]
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    out = np.zeros((len(cells), len(gene_idx)), dtype=np.float32)
    out[has] = x[rows[has]]
    return TargetSet(
        matrix=out,
        gene_names=[adata.var_names[i] for i in gene_idx],
        has_counts=has,
        panel=panel,
        provenance={"transform": "normalize_total(1e4) -> log1p"},
    )


def module_scores(adata, gene_sets: dict[str, list[str]], cells: list[str], *, tmin: int = 15) -> TargetSet:
    """E9: per-cell module activity via AUCell (decoupler), in the caller's cell order."""
    import decoupler as dc
    import pandas as pd
    import scanpy as sc

    a = adata.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    net = pd.DataFrame([(src, g) for src, genes in gene_sets.items() for g in genes],
                       columns=["source", "target"])
    dc.mt.aucell(a, net, tmin=tmin, verbose=False)
    sdf = a.obsm["score_aucell"]
    names = sorted(map(str, sdf.columns))
    if not names:
        raise TargetError(f"no module had >= {tmin} genes present in the matrix")
    scores = sdf[names].to_numpy(dtype=np.float32)
    rows, has = align_to_cells(a, cells)
    out = np.zeros((len(cells), len(names)), dtype=np.float32)
    out[has] = scores[rows[has]]
    return TargetSet(matrix=out, gene_names=names, has_counts=has, panel="modules",
                     provenance={"scorer": "decoupler.mt.aucell", "tmin": tmin,
                                 "n_up": "decoupler default (top 5% of features)",
                                 "n_modules": len(names)})


GENE_SETS = {
    "mouse": str(registry.PROJ / "gene_sets" / "mh.all.v2024.1.Mm.symbols.gmt"),
    "human": str(registry.PROJ / "gene_sets" / "h.all.v2024.1.Hs.symbols.gmt"),
}
SPECIES = {
    "crispr_a549_5k": "human",
    "nuclei_2k": "mouse", "tm_10X_P7_2": "mouse", "heart_10k_v3": "mouse", "neuron_10k_v3": "mouse",
    "mouse_brain_5k": "mouse", "neuron_1k_v3": "mouse", "neurons_5k": "mouse",
    "sod1_spinal_cord": "mouse",
    "sc_mixology_5cl": "human", "pbmc_1k_v2": "human", "pbmc_10k_v3": "human",
    "got_ET01": "human",
    "2500_rhesus_monkey_pbmcs_singleplex_3p_gem_x": "macaque",
    "2500_wistar_rat_pbmcs_singleplex_3p_gem_x": "rat",
    "fly_antenna": "fly",
}


def gene_set_path(dataset: str) -> str | None:
    """The hallmark GMT for a dataset's species. Unknown dataset is an error, not a guess."""
    if dataset not in SPECIES:
        raise TargetError(f"unknown species for {dataset!r}; add it to targets.SPECIES")
    return GENE_SETS.get(SPECIES[dataset])


def read_gmt(path) -> dict[str, list[str]]:
    """MSigDB .gmt: one module per line, `name <tab> url <tab> gene...`."""
    sets: dict[str, list[str]] = {}
    for line in Path(path).read_text().splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) > 2:
            sets[parts[0]] = parts[2:]
    return sets
