"""The control representations, written as stamped artifacts through the same path as the model."""

from __future__ import annotations

import hashlib
import gzip
import json
import subprocess
from pathlib import Path

import numpy as np

from . import artifact, registry, targets, tenx
from scrm_eval import panel as _panel


def __getattr__(name: str):
    if name == "CORPUS_ROOT":
        return registry.corpus_root()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
PRODUCERS = ("count_scanpy", "counts_10x_pca10", "depth2", "kmer_bag")
BUILT_PRODUCERS = tuple(p for p in PRODUCERS if p != "counts_10x_pca10")


def code_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parents[2],
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parents[2],
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def corpus_facts(dataset: str, corpus=None, root: Path | None = None) -> tuple[dict, dict, list[str]]:
    """`(compat_key, {dataset: stream_digest}, cell_names)` for one dataset, from the corpus itself."""
    import sys

    _tp = str(Path(__file__).resolve().parents[2] / "target" / "py")
    if _tp not in sys.path:
        sys.path.insert(0, _tp)
    import scrm_py

    path = (root or registry.corpus_root()) / dataset
    c = corpus if corpus is not None else scrm_py.Corpus(
        path=str(path), seq_len=16384, max_umi_len=1600, min_v=1
    )
    compat = c.compat_key()
    entry = _pinned().pinned_entry(path, dataset)
    declared = _pinned().declared(path)
    if dict(declared) != dict(compat):
        raise ValueError(
            f"{path}: manifest declares {declared} but the loader reports {compat} -- "
            "two readers of one molecule definition disagree"
        )
    return compat, {dataset: entry["stream_sha256"]}, list(c.cell_names())


def _pinned():
    """`pinned_digests` as a module, loaded by path and cached."""
    global _PINNED_MOD
    if _PINNED_MOD is None:
        import importlib.util

        f = Path(__file__).resolve().parents[2] / "crates/scrm-corpus/tests/pinned_digests.py"
        spec = importlib.util.spec_from_file_location("scrm_pinned_digests", f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _PINNED_MOD = mod
    return _PINNED_MOD


_PINNED_MOD = None


PERCOLATION_DEGREE = 1.0


def corpus_caveats(dataset: str) -> dict:
    """Risk facts a corpus measured about itself that nothing downstream was reading."""
    man = registry.corpus_root() / dataset / "manifest.json"
    if not man.exists():
        return {}
    d = json.loads(man.read_text())
    umi = d.get("measured", {}).get("umi_stats", {})
    counts = d.get("measured", {}).get("counts", {})
    deg = umi.get("mean_hamming1_degree")
    called, shallow = counts.get("cells_called"), counts.get("dropped_cells_shallow")
    out = {
        "mean_hamming1_degree": deg,
        "umi_graph_percolates": None if deg is None else bool(deg >= PERCOLATION_DEGREE),
        "percolation_threshold": PERCOLATION_DEGREE,
        "cells_called": called,
        "cells_dropped_shallow": shallow,
        "trim_config_stamped": bool(
            set((d.get("declared", {}).get("config", {}).get("trim") or {}).keys()) - {"decode_backend"}
        ),
    }
    if called and shallow is not None:
        out["cells_dropped_shallow_fraction"] = shallow / called
    return out


def _read_counts(dataset: str, adata=None):
    """The filtered count matrix, restricted to the corpus's cells and in corpus order."""
    if adata is not None:
        return adata.copy(), list(adata.obs_names)
    import scanpy as sc

    path = registry.count_matrix(dataset)
    if str(path).endswith((".h5ad", ".h5")):
        ad = sc.read_10x_h5(path) if str(path).endswith(".h5") else sc.read_h5ad(path)
        ad.var_names_make_unique()
    else:
        ad = sc.read_10x_mtx(path, var_names="gene_symbols", cache=False, gex_only=True)
    ad.obs_names = [b.split("-")[0] for b in ad.obs_names]
    _, _, cells = corpus_facts(dataset)
    have = set(ad.obs_names)
    keep = [n for n in cells if n in have]
    return ad[keep].copy(), keep


def filtered_matrix(dataset: str, out_dir: Path, adata=None) -> Path:
    """Write a 10x-format matrix holding EXACTLY the corpus's cells, and return its directory."""
    import scipy.io
    import scipy.sparse as sp

    ad, cells = _read_counts(dataset, adata)
    out_dir.mkdir(parents=True, exist_ok=True)
    x = ad.X.tocsr() if sp.issparse(ad.X) else sp.csr_matrix(ad.X)
    with gzip.open(out_dir / "matrix.mtx.gz", "wb") as fh:
        scipy.io.mmwrite(fh, x.T.astype(np.int32), field="integer")
    with gzip.open(out_dir / "barcodes.tsv.gz", "wt") as fh:
        fh.writelines(f"{b}-1\n" for b in cells)
    raw = ad.var["gene_ids"] if "gene_ids" in ad.var.columns else ad.var_names
    ids, patched = [], 0
    for gid, sym in zip(raw, ad.var_names):
        text = "" if gid is None else str(gid).strip()
        if text in ("", "nan", "NaN", "None", "na", "NA"):
            ids.append(str(sym))
            patched += 1
        else:
            ids.append(text)
    if patched:
        print(f"[filtered] {dataset}: {patched:,} of {len(ids):,} genes had no usable gene id; "
              f"used the gene symbol instead", flush=True)
    with gzip.open(out_dir / "features.tsv.gz", "wt") as fh:
        fh.writelines(f"{i}\t{sym}\tGene Expression\n" for i, sym in zip(ids, ad.var_names))
    print(f"[filtered] {dataset}: {len(cells):,} corpus cells x {ad.shape[1]:,} genes -> {out_dir}",
          flush=True)
    return out_dir


def count_scanpy(dataset: str, adata=None) -> tuple[list[str], np.ndarray]:
    """`normalize_total(1e4) → log1p → HVG-2000 → StandardScaler → PCA-50`, verbatim."""
    import scanpy as sc
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    ad, cells = _read_counts(dataset, adata)
    sc.pp.filter_genes(ad, min_cells=targets.MIN_CELLS)
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=2000)
    ad = ad[:, ad.var.highly_variable].copy()
    x = StandardScaler().fit_transform(np.asarray(ad.X.todense()))
    return cells, PCA(n_components=50, random_state=0).fit_transform(x).astype(np.float32)


KMER_K = 6


def _kmer_frequencies(dataset: str, k: int, root: Path | None = None) -> tuple[list[str], np.ndarray]:
    """Per-cell k-mer COMPOSITION over the corpus's own tokens. No model, no counts, no reference."""
    import pyarrow as pa
    from pyarrow import ipc

    root = (root or registry.corpus_root()) / dataset
    manifest = json.loads((root / "manifest.json").read_text())
    order = corpus_facts(dataset, root=root.parent)[2]
    index = {name: i for i, name in enumerate(order)}
    n_feat = 4 ** k
    acc = np.zeros(len(order) * n_feat, dtype=np.int64)
    pw = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)[::-1].copy()
    ones = np.ones(k, dtype=np.int64)
    for entry in manifest["shards"]:
        with pa.memory_map(str(root / entry["file"]), "rb") as src:
            reader = ipc.open_file(src)
            for b in range(reader.num_record_batches):
                batch = reader.get_batch(b)
                tok = batch.column("tokens")
                offs = np.asarray(tok.offsets, dtype=np.int64)
                vals = np.asarray(tok.values, dtype=np.uint8)
                if len(vals) < k:
                    continue
                cid = batch.column("cell_id")
                names = cid.dictionary.to_pylist()
                to_global = np.array([index[n] for n in names], dtype=np.int64)
                cell_of_row = to_global[np.asarray(cid.indices, dtype=np.int64)]

                code = np.convolve(vals.astype(np.int64), pw, mode="valid")
                is_base = (vals < 4).astype(np.int64)
                whole = np.convolve(is_base, ones, mode="valid") == k
                start = np.arange(code.size, dtype=np.int64)
                row_of_pos = np.repeat(np.arange(batch.num_rows, dtype=np.int64), np.diff(offs))
                row = row_of_pos[: code.size]
                inside = (start + k) <= offs[row + 1]
                keep = whole & inside
                if not keep.any():
                    continue
                flat = cell_of_row[row[keep]] * n_feat + code[keep]
                acc += np.bincount(flat, minlength=acc.size)
    counts = acc.reshape(len(order), n_feat).astype(np.float64)
    total = counts.sum(axis=1, keepdims=True)
    freq = np.divide(counts, np.where(total > 0, total, 1.0))
    return list(order), np.log1p(freq * 1e4).astype(np.float32)


def kmer_bag(dataset: str, adata=None) -> tuple[list[str], np.ndarray]:
    """`_kmer_frequencies` reduced to 50 components, the width every other control reports."""
    from sklearn.decomposition import PCA

    cells, x = _kmer_frequencies(dataset, KMER_K)
    n = min(50, min(x.shape) - 1) if min(x.shape) > 1 else 1
    return cells, PCA(n_components=n, random_state=0).fit_transform(x).astype(np.float32)


def depth2(dataset: str, adata=None) -> tuple[list[str], np.ndarray]:
    """`[log1p(total counts), log1p(genes detected)]`, raw."""
    ad, cells = _read_counts(dataset, adata)
    x = ad.X
    total = np.asarray(x.sum(axis=1)).ravel()
    genes = np.asarray((x > 0).sum(axis=1)).ravel()
    return cells, np.column_stack([np.log1p(total), np.log1p(genes)]).astype(np.float32)


def counts_10x_pca10(dataset: str, adata=None) -> tuple[list[str], np.ndarray]:
    """10x's own 10-component PCA, joined to the corpus in corpus order."""
    if adata is not None:
        raise ValueError("counts_10x_pca10 is 10x's published PCA; it cannot be recomputed on a "
                         "downsampled matrix, so it has no place in a budget sweep")
    _, _, cells = corpus_facts(dataset)
    bcs, mat = tenx.pca(dataset)
    by = dict(zip(bcs, mat))
    keep = [n for n in cells if n in by]
    return keep, np.stack([by[n] for n in keep]).astype(np.float32)


BUILDERS = {
    "count_scanpy": count_scanpy,
    "counts_10x_pca10": counts_10x_pca10,
    "depth2": depth2,
    "kmer_bag": kmer_bag,
}


def build(dataset: str, producer: str, out_dir: Path, *, adata=None, notes=None) -> Path:
    """Compute one control and write it as a stamped artifact."""
    if producer not in BUILDERS:
        raise ValueError(f"unknown control {producer!r}; known: {sorted(BUILDERS)}")
    cells, mat = BUILDERS[producer](dataset, adata)
    compat, digests, _ = corpus_facts(dataset)
    stamp = artifact.Stamp(
        producer=producer,
        dim=mat.shape[1],
        n_cells=len(cells),
        n_windows=mat.shape[0],
        cell_order_sha256=artifact.cell_order_digest(cells),
        corpus_compat_key=dict(compat) if isinstance(compat, dict) else {"compat_key": list(compat)},
        corpus_digests=digests,
        code_rev=code_rev(),
        w_eval=1,
        eval_seed=0,
        seq_len=16384,
        notes={"dataset": dataset, "rows": "one per cell; controls have no windows", **(notes or {})},
    )
    out = Path(out_dir) / f"{dataset}.{producer}.arrow"
    artifact.write(out, cells, np.zeros(len(cells), dtype=np.uint16), mat, stamp)
    return out


#: Declared in `panel.py`, which has no imports, because `eval/countfm/embed_count_fm.py`
#: runs under the foundation models' interpreter and cannot import this module.
PANEL = _panel.PANEL


def from_npz(dataset: str, producer: str, npz_path, out_dir: Path, *, notes=None) -> Path:
    """Wrap a count-FM embedding (`{emb, barcode}`) as a stamped artifact."""
    raw = np.load(npz_path, allow_pickle=True)
    for key in ("emb", "barcode"):
        if key not in raw:
            raise ValueError(f"{npz_path} has no {key!r}; keys are {sorted(raw.files)}")
    emb = np.asarray(raw["emb"], dtype=np.float32)
    bar = [str(b).split("-")[0] for b in raw["barcode"]]
    if emb.shape[0] != len(bar):
        raise ValueError(f"{npz_path}: {emb.shape[0]} embeddings but {len(bar)} barcodes")

    compat, digests, cells = corpus_facts(dataset)
    by = dict(zip(bar, emb))
    keep = [c for c in cells if c in by]
    if not keep:
        raise ValueError(f"{npz_path} shares no cells with {dataset}")
    mat = np.stack([by[c] for c in keep])

    stamp = artifact.Stamp(
        producer=producer,
        dim=mat.shape[1],
        n_cells=len(keep),
        n_windows=mat.shape[0],
        cell_order_sha256=artifact.cell_order_digest(keep),
        corpus_compat_key=dict(compat) if isinstance(compat, dict) else {"compat_key": list(compat)},
        corpus_digests=digests,
        code_rev=code_rev(),
        w_eval=1,
        eval_seed=0,
        seq_len=16384,
        notes={
            "dataset": dataset,
            "source_npz": str(npz_path),
            "cells_in_npz": len(bar),
            "cells_kept": len(keep),
            "rows": "one per cell; count-FM baselines have no windows",
            **(notes or {}),
        },
    )
    out = Path(out_dir) / f"{dataset}.{producer}.arrow"
    artifact.write(out, keep, np.zeros(len(keep), dtype=np.uint16), mat, stamp)
    return out


def digest_file(path: Path) -> str:
    """sha256 of a written artifact, so a cached control can be proved unchanged rather than trusted."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
