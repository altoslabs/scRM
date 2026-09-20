#!/usr/bin/env python3
"""Derive the seed floor from the reseed sets on disk, instead of trusting a constant."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "model"), str(REPO / "eval"), str(REPO / "target" / "py")]

from scrm_eval import registry as R  # noqa: E402

P = R.PROJ
PANEL, METRIC = R.REPORT_PANEL, "binary_expression"
sys.path.insert(0, str(REPO / "measurements"))
from ablation import DS_EVAL  # noqa: E402

SEED_SUFFIX = re.compile(r"^(?P<base>.+)-s(?P<seed>\d{8})$")


def scores() -> dict[str, dict[str, float]]:
    """`arm -> {dataset: E6 AUROC}` from the probes reports. Transcribes nothing."""
    out: dict[str, dict[str, float]] = {}
    for ds in DS_EVAL:
        f = P / "eval_artifacts" / f"{ds}.allmodels.probes.json"
        if not f.is_file():
            continue
        for key, e in json.loads(f.read_text())["artifacts"].items():
            v = ((e.get(METRIC) or {}).get(PANEL) or {}).get("macro")
            if v is None:
                continue
            name = key.removeprefix(f"{ds}.").removesuffix(".allreads").removesuffix(".full")
            out.setdefault(name, {})[ds] = v
    return out


def reseed_sets() -> dict[str, list[str]]:
    """`base arm -> [base, base-sNNNNNNNN, ...]`, for every base with at least one reseed."""
    known = {r for r, _, _ in R.REQUIRED_RUNS} | {r for r, _, _ in R.OPTIONAL_RUNS}
    groups: dict[str, set[str]] = {}
    for arm in known:
        m = SEED_SUFFIX.match(arm)
        if not m:
            continue
        base = m.group("base")
        if base in known:
            groups.setdefault(base, {base}).add(arm)
    return {b: sorted(v) for b, v in groups.items()}


def measure(sc: dict, members: list[str]) -> dict | None:
    """Mean range and worst per-dataset range, or None if the set is not fully scored."""
    full = [m for m in members if len(set(DS_EVAL) & set(sc.get(m, {}))) == len(DS_EVAL)]
    if len(full) < 2:
        return {"members": members, "scored": [len(sc.get(m, {})) for m in members],
                "usable": full, "incomplete": True}
    means = {m: st.fmean(sc[m][d] for d in DS_EVAL) for m in full}
    per_ds = {d: max(sc[m][d] for m in full) - min(sc[m][d] for m in full) for d in DS_EVAL}
    worst_ds = max(per_ds, key=per_ds.get)
    return {"members": full, "n": len(full), "incomplete": False,
            "mean_range": max(means.values()) - min(means.values()),
            "means": means, "worst_dataset": worst_ds, "worst_range": per_ds[worst_ds],
            "per_dataset": per_ds}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                   help="exit non-zero if registry.SEED_FLOOR is below the widest measured set")
    ap.add_argument("-o", "--out", type=Path, default=P / "eval_reports" / "seed_floor.json")
    a = ap.parse_args(argv)

    sc = scores()
    sets = reseed_sets()
    if not sets:
        print("no reseed sets found in the registry")
        return 1
    fbase = R.seed_base(R.FLAGSHIP)
    print(f"declared registry.SEED_FLOOR = {R.SEED_FLOOR}  "
          f"(SEED_FLOOR_SET = {R.SEED_FLOOR_SET})")
    print(f"flagship = {R.FLAGSHIP}  (reseed base {fbase})\n")
    report = {"declared_seed_floor": R.SEED_FLOOR, "flagship": R.FLAGSHIP,
              "flagship_base": fbase, "sets": {}}
    usable = {}
    for base, members in sorted(sets.items()):
        r = measure(sc, members)
        report["sets"][base] = r
        if r["incomplete"]:
            print(f"  {base:44s} INCOMPLETE — scored {r['scored']} of {len(DS_EVAL)} per member")
            continue
        usable[base] = r
        flag = "  <== FLAGSHIP's own architecture" if base == fbase else ""
        print(f"  {base:44s} n={r['n']}  mean range {r['mean_range']:.4f}  "
              f"worst dataset {r['worst_range']:.4f} ({r['worst_dataset']}){flag}")
    if not usable:
        print(f"\nno reseed set is fully scored on all {len(DS_EVAL)} datasets; "
              f"the floor cannot be derived yet")
        report["widest"] = None
    else:
        widest = max(usable, key=lambda b: usable[b]["mean_range"])
        w = usable[widest]["mean_range"]
        report["widest"] = {"set": widest, "mean_range": w}
        print(f"\nWIDEST fully-scored set: {widest} at {w:.4f} on the mean")
        if fbase in usable:
            f = usable[fbase]
            report["flagship_floor"] = {
                "set": fbase, "n": f["n"], "mean_range": f["mean_range"],
                "worst_dataset": f["worst_dataset"], "worst_range": f["worst_range"],
                "prescribed": math.ceil(f["mean_range"] * 10_000) / 10_000}
            print(f"the flagship's OWN floor (THE RULE): {f['mean_range']:.4f}  "
                  f"worst dataset {f['worst_range']:.4f} ({f['worst_dataset']})")
            print(f"  the rule therefore prescribes SEED_FLOOR = "
                  f"{report['flagship_floor']['prescribed']}")
            print(f"  for reference, the worst set anywhere is {widest} at {w:.4f}; declared "
                  f"SEED_FLOOR = {R.SEED_FLOOR}")
        else:
            print(f"the flagship {R.FLAGSHIP} has no fully-scored reseed set yet, so no floor can be "
                  "derived under the current rule")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    tmp.replace(a.out)
    print(f"\nwrote {a.out}")
    own = (usable.get(fbase) or {}).get("mean_range")
    if a.check:
        if own is None:
            print(f"FAILED: the flagship {R.FLAGSHIP} has no fully-scored reseed set, so "
                  "SEED_FLOOR cannot be derived under the current rule.")
            return 1
        want = math.ceil(own * 10_000) / 10_000
        if R.SEED_FLOOR < own or abs(R.SEED_FLOOR - want) > 1e-9:
            print(f"FAILED: SEED_FLOOR {R.SEED_FLOOR} does not match the flagship's own measured "
                  f"spread {own:.4f}. Re-ratify the constant, or say which set it comes from.")
            return 1
        msg = f"OK: SEED_FLOOR {R.SEED_FLOOR} matches the flagship's own spread {own:.4f}."
        wide = report["widest"]
        if wide["mean_range"] > R.SEED_FLOOR:
            msg += (f" NOTE: {wide['set']} spreads {wide['mean_range']:.4f}, wider than this floor, "
                    "so a delta measured against an arm of THAT set is not bounded by it.")
        print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
