"""The metrics, and nothing else."""

from __future__ import annotations

import numpy as np

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

DIM_EXPECTED = 384


def _as_matrix(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D (cells, dim) array, got shape {a.shape}")
    if a.shape[0] < 2:
        raise ValueError(f"need at least 2 cells, got {a.shape[0]}")
    if not np.isfinite(a).all():
        raise ValueError("embedding contains NaN or inf")
    return a


def centred(x):
    """`x - x.mean(axis=0)`: the ONE place an angular readout removes the shared mean vector."""
    a = _as_matrix(x)
    return a - a.mean(axis=0, keepdims=True)


def effective_rank(x) -> float:
    """`exp(H(p))` where `p` normalises the singular values of the **centred** matrix."""
    a = _as_matrix(x)
    a = a - a.mean(axis=0, keepdims=True)
    s = np.linalg.svd(a, compute_uv=False)
    total = s.sum()
    if total <= 0:
        return 1.0
    p = s / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def silhouette(x, labels, *, metric: str = "cosine") -> float:
    """Mean silhouette of a reference clustering in the embedding, cosine by default."""
    a = _as_matrix(x)
    y = np.asarray(labels)
    if y.shape[0] != a.shape[0]:
        raise ValueError(f"{a.shape[0]} cells but {y.shape[0]} labels")
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(silhouette_score(a, y, metric=metric))


def knn_label_purity(x, labels, *, k: int = 15, chance_correct: bool = True) -> dict[str, float]:
    """Fraction of each cell's `k` nearest neighbours sharing its reference label."""
    from sklearn.neighbors import NearestNeighbors

    a = centred(x)
    y = np.asarray(labels)
    if y.shape[0] != a.shape[0]:
        raise ValueError(f"{a.shape[0]} cells but {y.shape[0]} labels")
    k = min(k, a.shape[0] - 1)
    if k < 1:
        raise ValueError("need at least 2 cells to have a neighbour")

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(a)
    idx = nn.kneighbors(a, return_distance=False)[:, 1:]
    purity = float((y[idx] == y[:, None]).mean())

    _, counts = np.unique(y, return_counts=True)
    chance = float(((counts / counts.sum()) ** 2).sum())
    adj = (purity - chance) / (1.0 - chance) if chance < 1.0 else float("nan")
    return {"purity": purity, "chance": chance, "purity_adj": float(adj), "k": float(k)}


def cluster_agreement(found, reference) -> dict[str, float]:
    """ARI and NMI between a clustering found in the embedding and the reference clustering."""
    f, r = np.asarray(found), np.asarray(reference)
    if f.shape != r.shape:
        raise ValueError(f"{f.shape} found labels vs {r.shape} reference labels")
    return {
        "ari": float(adjusted_rand_score(r, f)),
        "nmi": float(normalized_mutual_info_score(r, f)),
    }


#: The community-detection algorithm, and the reason it is this one. Louvain via networkx
#: (BSD-3-Clause). NOT `leidenalg` (GPL-3.0-or-later) and NOT `python-igraph` (GPL), which
#: `scanpy.tl.leiden` requires for every one of its flavours: this project ships under MIT, so
#: neither may appear in `requirements.txt` nor be imported here. The protocol around the
#: clustering is unchanged -- the same scanpy kNN connectivity graph, the same resolution, the
#: same seed -- so only the community-detection step differs, and every panel derived from it
#: records this constant in its provenance.
CLUSTER_ALGO = "louvain_networkx"


def cluster_provenance(resolution: float = 1.0, n_neighbors: int = 15, seed: int = 0) -> dict:
    """Everything about the clustering protocol that a reported ARI depends on."""
    from importlib.metadata import version

    return {
        "algo": CLUSTER_ALGO,
        "resolution": resolution,
        "n_neighbors": n_neighbors,
        "seed": seed,
        "directed": False,
        "scanpy": version("scanpy"),
        "networkx": version("networkx"),
    }


def graph_cluster_labels(connectivities, *, resolution: float, seed: int) -> np.ndarray:
    """THE clustering implementation: Louvain communities on a kNN connectivity graph.

    Both callers -- the ARI path in this module and the marker-panel path in `targets` -- come
    here, so there is one algorithm and one place it can change. Communities are numbered by
    (size descending, lowest member) so the labelling is deterministic across runs; ARI does not
    depend on the numbering but a cached panel keyed on labels would.
    """
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    graph = nx.from_scipy_sparse_array(connectivities)
    communities = louvain_communities(graph, resolution=resolution, seed=seed, weight="weight")
    labels = np.empty(connectivities.shape[0], dtype=np.int64)
    for i, community in enumerate(sorted(communities, key=lambda c: (-len(c), min(c)))):
        for node in community:
            labels[node] = i
    return labels


def cluster_labels(x, *, resolution: float = 1.0, n_neighbors: int = 15, seed: int = 0) -> np.ndarray:
    """Cluster the embedding on its own kNN graph, deterministically."""
    import anndata
    import scanpy as sc

    a = _as_matrix(x)
    ad = anndata.AnnData(a.astype(np.float32))
    sc.pp.neighbors(ad, n_neighbors=n_neighbors, use_rep="X", random_state=seed)
    return graph_cluster_labels(ad.obsp["connectivities"], resolution=resolution, seed=seed)


def linear_cka(a, b) -> float:
    """Centred Kernel Alignment between two representations of the same cells. No fitting, no labels."""
    x, y = _as_matrix(a), _as_matrix(b)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{x.shape[0]} rows vs {y.shape[0]}: CKA compares the same cells")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    xy = np.linalg.norm(x.T @ y, ord="fro") ** 2
    xx = np.linalg.norm(x.T @ x, ord="fro")
    yy = np.linalg.norm(y.T @ y, ord="fro")
    if xx == 0 or yy == 0:
        return 0.0
    return float(xy / (xx * yy))


def pc_regression(embedding, targets, *, folds: int = 5, seed: int = 0) -> dict:
    """Cross-validated R² for predicting each count-space PC from the embedding."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold

    x = _as_matrix(embedding)
    t = _as_matrix(targets)
    if x.shape[0] != t.shape[0]:
        raise ValueError(f"{x.shape[0]} cells in the embedding vs {t.shape[0]} in the targets")

    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    pred = np.empty_like(t)
    for tr, te in kf.split(x):
        model = RidgeCV(alphas=np.logspace(-2, 4, 13))
        model.fit(x[tr], t[tr])
        pred[te] = model.predict(x[te])

    resid = ((t - pred) ** 2).sum(axis=0)
    total = ((t - t.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    r2 = np.where(total > 0, 1.0 - resid / total, 0.0)
    weights = total / total.sum() if total.sum() > 0 else np.full(t.shape[1], 1 / t.shape[1])
    return {
        "r2_per_pc": [float(v) for v in r2],
        "r2_mean": float(r2.mean()),
        "r2_variance_weighted": float((r2 * weights).sum()),
    }


def residualise(targets, nuisance):
    """Remove everything linearly predictable from `nuisance` out of `targets`."""
    t = _as_matrix(targets)
    n = _as_matrix(nuisance)
    if t.shape[0] != n.shape[0]:
        raise ValueError(f"{t.shape[0]} target rows vs {n.shape[0]} nuisance rows")
    design = np.column_stack([np.ones(n.shape[0]), n - n.mean(axis=0, keepdims=True)])
    coef, *_ = np.linalg.lstsq(design, t, rcond=None)
    return t - design @ coef


def neighbor_overlap(a, b, *, k: int = 15) -> float:
    """Fraction of each cell's `k` nearest neighbours shared between two representations."""
    from sklearn.neighbors import NearestNeighbors

    x, y = centred(a), centred(b)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{x.shape[0]} rows vs {y.shape[0]}")
    k = min(k, x.shape[0] - 1)
    ix = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(x).kneighbors(x, return_distance=False)[:, 1:]
    iy = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(y).kneighbors(y, return_distance=False)[:, 1:]
    shared = [len(set(r1) & set(r2)) for r1, r2 in zip(ix, iy)]
    return float(np.mean(shared) / k)


def sensitive(a, b, *, max_cosine: float = 0.999) -> tuple[bool, float | None]:
    """The ablation gate: do two embeddings actually differ?"""
    x, y = _as_matrix(a), _as_matrix(b)
    if x.shape != y.shape:
        return True, None
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    nx = np.linalg.norm(x, axis=1)
    ny = np.linalg.norm(y, axis=1)
    ok = (nx > 0) & (ny > 0)
    if not ok.any():
        return False, None
    cos = (x[ok] * y[ok]).sum(axis=1) / (nx[ok] * ny[ok])
    mean_cos = float(np.clip(cos, -1.0, 1.0).mean())
    return mean_cos < max_cosine, mean_cos
