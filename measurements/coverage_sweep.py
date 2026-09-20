"""Does using MORE READS per cell make the embedding better? The published tables cannot say."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "eval"), str(REPO / "target" / "py"), str(REPO / "model")]
from scrm_eval import artifact, reduce as _reduce, registry, targets  # noqa: E402
from scrm_eval.probes import binary_expression_probe  # noqa: E402
from scrm_eval.reduce import reduce_windows  # noqa: E402
from scrm_eval.run_probes import DETECTION_THRESHOLD  # noqa: E402

P = registry.PROJ


def group_size(stamp: artifact.Stamp) -> int:
    """How many WINDOWS one artifact row holds, READ FROM THE STAMP and never assumed."""
    rows = (stamp.notes or {}).get("rows")
    if rows is None:
        raise SystemExit(
            "the artifact's stamp records no notes['rows'], so it cannot say whether a row is one "
            "window or a group of them. Re-embed with a writer that stamps it; do not guess."
        )
    if rows == "windows":
        return 1
    if rows != "pairs":
        raise SystemExit(f"unknown notes['rows'] unit {rows!r}: this sweep knows 'windows' and 'pairs'")
    ro = stamp.readout or {}
    if "views" in ro:
        v = int(ro["views"])
        if v < 1:
            raise SystemExit(f"readout declares views={v!r}, which cannot be a group size")
        return v
    levels = [str(x) for x in ro.get("levels", ())]
    if ro.get("pairing") == "adjacent_v1" or any(x.endswith("->pair") for x in levels):
        return 2
    raise SystemExit(
        f"stamp says rows are 'pairs' but the readout spec {ro!r} declares no group size, so the "
        "windows-per-row factor is unknown. Refusing to guess it (design 12.3)."
    )


def prefix_rows(emb: artifact.Embeddings, k: int, group: int) -> np.ndarray:
    """Mask keeping each cell's first k WINDOWS. k <= 0 keeps everything."""
    if k <= 0:
        return np.ones(len(emb.cell_ids), dtype=bool)
    if k % group:
        raise SystemExit(
            f"k={k} windows is not a multiple of this artifact's group size {group}: a group is the "
            "smallest prefix that exists, because pool_pairs drops a partial trailing group."
        )
    return np.asarray(emb.windows) < k // group


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", required=True, help="arm, e.g. armA")
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128, 256, 0],
                    help="windows per cell to keep; 0 means all. Multiples of the group size.")
    ap.add_argument("--panel", default="hvg2000")
    ap.add_argument("-o", "--out", type=Path, default=None)
    a = ap.parse_args()

    path = P / "eval_runs" / a.run / f"{a.dataset}.{a.run}.allreads.arrow"
    emb = artifact.read(path)
    adata = targets.load_counts(a.dataset)

    n_top = int(a.panel.removeprefix("hvg"))
    idx = targets.hvg_panel(adata, n_top=n_top)

    rule = _reduce.rule_for(emb.stamp)
    group = group_size(emb.stamp)
    print(f"  rows={emb.stamp.notes.get('rows')!r}  windows per row={group}  "
          f"({emb.stamp.n_windows:,} rows = {emb.stamp.n_windows * group:,} windows)", flush=True)
    out = []
    for k in a.ks:
        keep = prefix_rows(emb, k, group)
        cells, x = reduce_windows(emb.cell_ids[keep], emb.matrix[keep], rule)
        cells = [str(c) for c in cells]
        t = targets.binary_expression_targets(adata, cells, idx, panel=a.panel,
                                              threshold=DETECTION_THRESHOLD)
        r = binary_expression_probe(x, t.matrix, t.has_counts)
        med_rows = int(np.median(np.bincount(np.unique(emb.cell_ids[keep], return_inverse=True)[1])))
        med = med_rows * group
        out.append({"k": k, "n_cells": len(cells), "median_windows_per_cell": med,
                    "median_rows_per_cell": med_rows, "windows_per_row": group,
                    "macro": r.macro, "seed_sd": r.seed_sd, "n_scored": r.n_scored})
        print(f"  k={k or 'all':>4}  cells={len(cells):5d}  median_windows={med:4d}  "
              f"AUROC={r.macro:.4f} +/- {r.seed_sd:.4f}  ({r.n_scored} genes)", flush=True)

    res = {"dataset": a.dataset, "run": a.run, "panel": a.panel,
           "unit": "windows", "windows_per_row": group,
           "rows_unit": emb.stamp.notes.get("rows"),
           "detection_threshold": DETECTION_THRESHOLD, "points": out}
    if a.out:
        a.out.write_text(json.dumps(res, indent=2))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
