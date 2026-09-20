#!/usr/bin/env python3
"""Turn the NeurIPS 2021 BMMC multiome h5ad into what the corpus builder and E3 need."""

import gzip
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.io
import scipy.sparse as sp

SAMPLES = ("s1d1", "s1d3")


def die(msg: str) -> None:
    raise SystemExit(f"bmmc_prep: {msg}")


def pick(cols, *candidates):
    for c in candidates:
        if c in cols:
            return c
    die(f"none of {candidates} in {list(cols)}")


def main() -> None:
    h5, out_root = Path(sys.argv[1]), Path(sys.argv[2])
    a = ad.read_h5ad(h5)
    print(f"[h5ad] {a.shape[0]:,} cells x {a.shape[1]:,} features", flush=True)
    print(f"[obs] columns: {list(a.obs.columns)}", flush=True)
    print(f"[var] columns: {list(a.var.columns)}", flush=True)
    print(f"[obs_names] first three: {list(a.obs_names[:3])}", flush=True)

    batch_col = pick(a.obs.columns, "batch", "Samplename", "sample")
    label_col = pick(a.obs.columns, "cell_type", "cell_type_l1", "celltype")
    print(f"[obs] batch={batch_col!r} label={label_col!r}", flush=True)
    print(f"[obs] batches present: {sorted(set(a.obs[batch_col].astype(str)))}", flush=True)
    print(f"[obs] {a.obs[label_col].nunique()} distinct curated labels", flush=True)

    if "feature_types" in a.var.columns:
        kinds = sorted(set(a.var["feature_types"].astype(str)))
        print(f"[var] feature_types: {kinds}", flush=True)
        gex = a.var["feature_types"].astype(str) == "GEX"
        if not gex.any():
            die(f"no GEX features; feature_types were {kinds}")
        a = a[:, gex.values]
        print(f"[var] restricted to GEX: {a.shape[1]:,} genes", flush=True)

    for s in SAMPLES:
        keep = (a.obs[batch_col].astype(str) == s).values
        if not keep.any():
            die(f"sample {s!r} absent; batches were {sorted(set(a.obs[batch_col].astype(str)))}")
        sub = a[keep]
        bcs = [n.split("-")[0] for n in sub.obs_names]
        bad = [b for b in bcs[:200] if len(b) != 16 or set(b) - set("ACGTN")]
        if bad:
            die(f"{s}: barcodes are not bare 16 bp ACGT, e.g. {bad[:3]} from {list(sub.obs_names[:2])}")
        if len(set(bcs)) != len(bcs):
            die(f"{s}: {len(bcs) - len(set(bcs))} duplicate barcodes within the sample")

        d = out_root / f"bmmc_{s}" / "tenx_matrix"
        d.mkdir(parents=True, exist_ok=True)
        x = sub.X
        x = sp.csr_matrix(x) if not sp.issparse(x) else x.tocsr()
        with gzip.open(d / "matrix.mtx.gz", "wb") as fh:
            scipy.io.mmwrite(fh, x.T.astype(np.int32), field="integer")
        with gzip.open(d / "barcodes.tsv.gz", "wt") as fh:
            fh.writelines(f"{b}-1\n" for b in bcs)
        ids = sub.var["gene_id"] if "gene_id" in sub.var.columns else sub.var_names
        with gzip.open(d / "features.tsv.gz", "wt") as fh:
            fh.writelines(f"{i}\t{sym}\tGene Expression\n" for i, sym in zip(ids, sub.var_names))
        lab = out_root / f"bmmc_{s}" / "cell_type.tsv"
        with open(lab, "w") as fh:
            fh.writelines(f"{b}\t{c}\n" for b, c in zip(bcs, sub.obs[label_col].astype(str)))
        print(f"[{s}] {sub.shape[0]:,} cells x {sub.shape[1]:,} genes -> {d} (+ {lab.name})", flush=True)


if __name__ == "__main__":
    main()
