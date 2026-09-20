"""Certify OUR UCE invocation against THEIR published embeddings, on THEIR data.

Owner, 2026-08-26, wanting UCE in our benchmark and scRM compared on their datasets. The
certification question this answers is narrower and more important than either: when we run UCE
ourselves, do we get what they got? Their published per-species h5ads
(`fm/uce/their_data/*.h5ad`, fetched individually -- the 146 GB export_data monolith is
quota-throttled and unnecessary) each carry raw counts AND their own `X_uce`, so embedding the same
cells with our invocation gives two vectors per cell: theirs and ours.

If per-cell cosine is ~1, every UCE number reported anywhere in this project rests on a verified
invocation. If it is not, the difference is a bug in our setup and must be found before any UCE row
is published -- which is exactly the check that cannot be done on our own datasets, because there is
no reference embedding for them.

Reported per dataset: per-cell cosine (mean, sd, min), the same for a shuffled pairing as the null,
and -- because two runs of a stochastic model need not be bit-identical -- the agreement of the two
embeddings on the downstream task: a label probe on theirs vs on ours, same folds.

    python measurements/uce_certify_invocation.py --dataset their_mouse_kidney_norn \
        --their /projects/.../their_data/mouse_kidney_norn.h5ad --labels cell_type
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))

from scrm_eval import registry  # noqa: E402

P = registry.PROJ
OUT = P / "eval_reports" / "uce"


def cos(A, B):
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    return np.einsum("ij,ij->i", A, B) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, help="the name used for our npz under fm/uce/npz")
    ap.add_argument("--their", required=True, type=Path, help="their h5ad, carrying X_uce")
    ap.add_argument("--labels", default=None, help="obs column to probe, if any")
    a = ap.parse_args()

    import scanpy as sc

    ours_npz = P / "fm" / "uce" / "npz" / f"{a.dataset}.uce.npz"
    z = np.load(ours_npz, allow_pickle=True)
    ours, bars = z["emb"], [str(b) for b in z["barcode"]]
    ad = sc.read(a.their)
    theirs_all = np.asarray(ad.obsm["X_uce"], dtype=np.float32)

    # Align by barcode. UCE filters cells (min counts / genes), so ours is a SUBSET in general; the
    # join must be explicit rather than positional, or a filtered cell silently shifts every row.
    idx = {str(b): i for i, b in enumerate(ad.obs_names)}
    keep = [(i, idx[b]) for i, b in enumerate(bars) if b in idx]
    if not keep:
        raise SystemExit(f"no barcode overlap: ours {bars[:3]} vs theirs {list(ad.obs_names[:3])}")
    oi = [i for i, _ in keep]
    ti = [j for _, j in keep]
    A, B = ours[oi], theirs_all[ti]

    same = cos(A, B)
    rng = np.random.default_rng(0)
    null = cos(A, B[rng.permutation(len(B))])
    rep = {
        "dataset": a.dataset,
        "their_file": a.their.name,
        "cells_ours": int(len(bars)),
        "cells_theirs": int(ad.n_obs),
        "cells_matched": int(len(keep)),
        "dim": int(A.shape[1]),
        "per_cell_cosine_ours_vs_theirs": {"mean": float(same.mean()), "sd": float(same.std()),
                                           "min": float(same.min()), "p01": float(np.quantile(same, .01))},
        "shuffled_pairing_null": {"mean": float(null.mean()), "sd": float(null.std())},
    }
    if a.labels and a.labels in ad.obs.columns:
        from scrm_eval import probes

        y = np.asarray(ad.obs[a.labels].values[ti], dtype=object).astype(str)
        for name, X in (("theirs", B), ("ours", A)):
            r = probes.annotation_probe(X, y, readers=("knn_cosine",), n_jobs=4)["knn_cosine"]
            rep.setdefault("label_probe_knn_macro_f1", {})[name] = {"macro": r.macro, "seed_sd": r.seed_sd}
        rep["label_probe_column"] = a.labels
        rep["label_probe_classes"] = int(len(set(y)))
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"certify_invocation.{a.dataset}.json"
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rep, indent=2))
    tmp.replace(dest)
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
