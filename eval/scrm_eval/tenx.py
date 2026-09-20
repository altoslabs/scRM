"""10x CellRanger's own analysis bundle: the independent label source and the `counts_10x_pca10` control."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import registry

DEFAULT_ROOT = registry.PROJ / "raw" / "analysis"

_GRAPHCLUST = "analysis/clustering/*graphclust/clusters.csv"
_KMEANS = "analysis/clustering/*kmeans_{k}_clusters/clusters.csv"
_PCA = "analysis/pca/*components/projection.csv"


def _one(root: Path, pattern: str, what: str) -> Path:
    hits = sorted(root.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"no {what} under {root} (looked for {pattern})")
    if len(hits) > 1:
        raise ValueError(f"{len(hits)} candidates for {what} under {root}: {[h.name for h in hits]}")
    return hits[0]


def _read_two_column(path: Path) -> dict[str, str]:
    """`Barcode,Value` with the GEM-well suffix stripped, as the corpus stores cell names."""
    out: dict[str, str] = {}
    with path.open() as fh:
        r = csv.reader(fh)
        header = next(r)
        if len(header) != 2:
            raise ValueError(f"{path} has {len(header)} columns, expected Barcode,Value")
        for bc, value in r:
            bare = bc.split("-")[0]
            if bare in out and out[bare] != value:
                raise ValueError(
                    f"{path} maps barcode {bare} twice with different values ({out[bare]!r} then "
                    f"{value!r}): this is an aggregated multi-well bundle, and the corpus's bare "
                    f"16 bp cell names cannot address its cells. Join on the full barcode instead."
                )
            out[bare] = value
    return out


def clusters(dataset: str, *, root: Path = DEFAULT_ROOT, k: int | None = None) -> dict[str, int]:
    """10x's clustering for `dataset`: graph-based by default, k-means when `k` is given."""
    d = root / dataset
    pattern = _GRAPHCLUST if k is None else _KMEANS.format(k=k)
    what = "graph clustering" if k is None else f"k-means at k={k}"
    return {bc: int(v) for bc, v in _read_two_column(_one(d, pattern, what)).items()}


PCA_COMPONENTS = 10


def pca(dataset: str, *, root: Path = DEFAULT_ROOT) -> tuple[list[str], np.ndarray]:
    """10x's PCA projection: `(barcodes, matrix)`, the `counts_10x_pca10` control."""
    path = _one(root / dataset, _PCA, "PCA projection")
    barcodes: list[str] = []
    rowlist: list[list[float]] = []
    with path.open() as fh:
        r = csv.reader(fh)
        header = next(r)
        if header[0] != "Barcode":
            raise ValueError(f"{path} does not start with a Barcode column: {header[:3]}")
        for row in r:
            barcodes.append(row[0].split("-")[0])
            rowlist.append([float(x) for x in row[1:]])
    mat = np.asarray(rowlist, dtype=np.float32)
    if mat.ndim != 2 or mat.shape[1] != PCA_COMPONENTS:
        raise ValueError(
            f"{path} holds {mat.shape[1] if mat.ndim == 2 else '?'} components, but this is the "
            f"`counts_10x_pca10` control and its name states {PCA_COMPONENTS}. A control whose "
            f"width does not match its label invites the cross-table comparison this module exists "
            f"to prevent."
        )
    return barcodes, mat


def align(cell_names, mapping: dict[str, int]):
    """Inner join of corpus cells against a 10x label map, in corpus order."""
    names = [str(n) for n in cell_names]
    keep = [i for i, n in enumerate(names) if n in mapping]
    labels = np.asarray([mapping[names[i]] for i in keep], dtype=np.int32)
    return np.asarray(keep, dtype=np.int64), labels
