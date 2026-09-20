"""Real-data positive control for EVERY probe family, run before any number is published."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "eval"), str(REPO / "target" / "py")]

from scrm_eval import artifact as A, registry, targets  # noqa: E402
from scrm_eval.probes import (binary_expression_probe, count_regression_probe,  # noqa: E402
                               expression_regression_probe, module_activity_probe)
from scrm_eval.reduce import reduce_windows  # noqa: E402
from scrm_eval import run_probes  # noqa: E402

ART = registry.PROJ / "eval_artifacts"


def _artifacts(dataset: str) -> tuple[Path, Path]:
    """The signal artifact and the depth floor this control contrasts, for `dataset`."""
    return (ART / f"{dataset}.count_scanpy.arrow", ART / f"{dataset}.depth2.arrow")


def check_dataset(dataset: str, gmt=None, include_count_likelihood: bool = False) -> bool:
    """Score a PCA of the counts and the depth floor on every probe family, and print both."""
    signal, floor = (A.read(p) for p in _artifacts(dataset))
    ad = targets.load_counts(dataset)
    gmt = gmt or targets.gene_set_path(dataset)
    if gmt is None:
        print(f"[probe_sanity] {dataset}: E9 skipped -- MSigDB publishes no hallmark collection for "
              f"{targets.SPECIES[dataset]}, so this dataset has no module target set.", flush=True)

    rows = {}
    for name, emb in (("count_scanpy", signal), ("depth_floor", floor)):
        cells, x = reduce_windows(emb.cell_ids, emb.matrix, "mean")
        cells = [str(c) for c in cells]
        depth = run_probes.depth_nuisance(ad, cells)

        idx = targets.hvg_panel(ad, n_top=200)
        det = targets.binary_expression_targets(
            ad, cells, idx, panel="hvg200", threshold=run_probes.DETECTION_THRESHOLD)
        graded = targets.expression_targets(ad, cells, idx, panel="graded")

        rows[name] = {
            "E6 detection AUROC": binary_expression_probe(x, det.matrix, det.has_counts).macro,
            "E6 graded r": expression_regression_probe(x, graded.matrix, graded.has_counts).macro,
            "E6 graded r (depth-res)": expression_regression_probe(
                x, graded.matrix, graded.has_counts, depth=depth).macro,
        }
        if gmt is not None:
            mods = targets.module_scores(ad, targets.read_gmt(gmt), cells)
            rows[name]["E9 pathway r"] = module_activity_probe(
                x, mods.matrix, mods.has_counts).macro
            rows[name]["E9 pathway r (depth-res)"] = module_activity_probe(
                x, mods.matrix, mods.has_counts, depth=depth).macro
        if include_count_likelihood:
            raw = targets.raw_count_targets(ad, cells, idx[:30], panel="raw30")
            rows[name]["E6c count GLM"] = count_regression_probe(
                x, raw.matrix, raw.has_counts, depth, family="nb").macro

    print(f"\n{dataset}: does a PCA of the counts beat depth-only on count-derived targets?\n")
    print(f"{'probe':22s} {'count_scanpy':>13s} {'depth floor':>12s}   verdict")
    ok = True
    for probe in rows["count_scanpy"]:
        s, f = rows["count_scanpy"][probe], rows["depth_floor"][probe]
        passed = np.isfinite(s) and np.isfinite(f) and s > f and s > 0.0
        ok &= bool(passed)
        print(f"{probe:22s} {s:>13.3f} {f:>12.3f}   {'PASS' if passed else 'FAIL'}")
    print()
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", help="one dataset; omit with --check for every evaluation dataset")
    ap.add_argument("--check", action="store_true",
                    help="every evaluation dataset whose two control artifacts are on disk")
    ap.add_argument("--gmt", default=None, help="default: the dataset's species hallmark sets")
    ap.add_argument("--include-count-likelihood", action="store_true",
                    help="also check the count-GLM probe, which currently FAILS this control")
    args = ap.parse_args()
    if bool(args.dataset) == bool(args.check):
        ap.error("pass --dataset <name> or --check, not both and not neither")

    if args.dataset:
        return 0 if check_dataset(args.dataset, args.gmt, args.include_count_likelihood) else 1

    ok, ran = True, 0
    for ds in registry.datasets("eval"):
        missing = [p.name for p in _artifacts(ds) if not p.exists()]
        if missing:
            print(f"[probe_sanity] {ds}: SKIPPED, no {', '.join(missing)} on disk", flush=True)
            continue
        ran += 1
        ok &= check_dataset(ds, args.gmt, args.include_count_likelihood)
    if not ran:
        print(f"[probe_sanity] no evaluation dataset has both control artifacts under {ART}")
        return 1
    print(f"[probe_sanity] {ran} dataset(s) checked: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
