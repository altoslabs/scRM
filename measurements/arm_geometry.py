"""Effective rank per arm, per dataset — the collapse measure, as a stamped report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "eval"), str(REPO / "target" / "py")]
from scrm_eval import artifact, metrics, registry  # noqa: E402
from scrm_eval import reduce as _reduce  # noqa: E402
from scrm_eval.reduce import reduce_windows  # noqa: E402

P = registry.PROJ
REPORTS = P / "eval_reports"


def measure(path: Path, override: str | None = None) -> dict:
    """Geometry of one artifact, reduced by the rule its OWN stamp implies."""
    e = artifact.read(path)
    rule = _reduce.rule_for(e.stamp, override=override)
    cid, mat = reduce_windows(np.asarray(e.cell_ids), np.asarray(e.matrix), rule=rule)
    x = np.asarray(mat, dtype=np.float64)
    er = metrics.effective_rank(x)
    n = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(0)
    sub = n[rng.choice(len(n), size=min(1500, len(n)), replace=False)]
    cos = float((sub @ sub.T)[np.triu_indices(len(sub), 1)].mean())
    nrm = np.linalg.norm(x, axis=1)
    return {"reduction": rule, "n_cells": int(x.shape[0]), "dim": int(x.shape[1]),
            "effective_rank": float(er), "rank_fraction": float(er / x.shape[1]),
            "mean_cosine": cos, "norm_cv": float(nrm.std() / max(nrm.mean(), 1e-12)),
            "w_eval": int(e.stamp.w_eval or 0)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", default=list(registry.datasets("eval")))
    ap.add_argument("--reduction", default=None,
                    help="force one rule on EVERY artifact, arms and baselines alike; "
                         "omit so each is read by reduce.rule_for(stamp)")
    a = ap.parse_args(argv)
    out: dict = {"reduction": a.reduction or "per-stamp via reduce.rule_for",
                 "scrm_rule": _reduce.SCRM_RULE, "baseline_rule": "mean",
                 "arms": {}, "baselines": {}}
    for ds in a.datasets:
        for run in registry.runs():
            f = P / "eval_runs" / run / f"{ds}.{run}.allreads.arrow"
            if f.exists():
                out["arms"].setdefault(run, {})[ds] = measure(f, a.reduction)
        for p in sorted((P / "eval_artifacts").glob(f"{ds}.*.arrow")):
            prod = p.name.split(".")[1]
            if prod in ("allmodels",):
                continue
            out["baselines"].setdefault(prod, {})[ds] = measure(p, a.reduction)
        print(f"[geom] {ds}: {sum(ds in v for v in out['arms'].values())} arms, "
              f"{sum(ds in v for v in out['baselines'].values())} baselines", flush=True)

    for group in ("arms", "baselines"):
        for name, per in out[group].items():
            v = [d["effective_rank"] for d in per.values()]
            fr = [d["rank_fraction"] for d in per.values()]
            out[group][name]["_mean"] = {"effective_rank": float(np.mean(v)),
                                        "rank_fraction": float(np.mean(fr)),
                                        "n_datasets": len(v)}
    REPORTS.mkdir(parents=True, exist_ok=True)
    dest = REPORTS / "arm_geometry.json"
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2, sort_keys=True))
    tmp.replace(dest)
    print(f"\n{'name':30s} {'eff.rank':>9s} {'rank%':>7s} {'n_ds':>5s}")
    rows = [(n, d["_mean"]) for g in ("arms", "baselines") for n, d in out[g].items()]
    for n, m in sorted(rows, key=lambda r: -r[1]["effective_rank"]):
        print(f"{n:30s} {m['effective_rank']:9.2f} {100*m['rank_fraction']:6.2f}% "
              f"{m['n_datasets']:5d}")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
