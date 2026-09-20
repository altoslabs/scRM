#!/usr/bin/env python3
"""Is a published gap noise? Paired significance tests between representations."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "model"), str(REPO / "eval"), str(REPO / "target" / "py")]

from scrm_eval import registry as R, reports  # noqa: E402

P = R.PROJ


def load(metric: str, panel: str) -> dict[str, dict[str, float]]:
    """`model -> {dataset: macro}` from the scored reports on disk. Transcribes nothing."""
    out: dict[str, dict[str, float]] = {}
    for f in glob.glob(str(P / "eval_artifacts" / "*.allmodels.probes.json")):
        for key, e in json.loads(Path(f).read_text())["artifacts"].items():
            blk = (e.get(metric) or {}).get(panel) or {}
            if blk.get("macro") is None:
                continue
            ds = key.split(".")[0]
            name = key.removeprefix(f"{ds}.").removesuffix(".allreads").removesuffix(".full")
            out.setdefault(name, {})[ds] = blk["macro"]
    return out


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values, monotone in p."""
    m = len(p)
    order = np.argsort(p)
    q = np.empty_like(p)
    running = 1.0
    for rank, i in enumerate(reversed(order)):
        running = min(running, p[i] * m / (m - rank))
        q[i] = running
    return q


def compare(subject: str, others: list[str], per: dict, datasets: tuple[str, ...]) -> list[dict]:
    rows = []
    for b in others:
        common = [d for d in datasets if d in per.get(subject, {}) and d in per.get(b, {})]
        if len(common) < 6:
            continue
        a = np.array([per[subject][d] for d in common])
        c = np.array([per[b][d] for d in common])
        diff = a - c
        ahead = int((diff > 0).sum())
        rows.append({
            "baseline": b, "n": len(common), "mean_delta": float(diff.mean()),
            "sd_delta": float(diff.std(ddof=1)), "ahead": ahead,
            "sign_p": float(stats.binomtest(ahead, len(common), 0.5).pvalue),
            "t_p": float(stats.ttest_rel(a, c).pvalue),
            "wilcoxon_p": float(stats.wilcoxon(diff).pvalue),
        })
    if rows:
        for r, q in zip(rows, bh(np.array([r["t_p"] for r in rows]))):
            r["t_q"] = float(q)
    return rows


def verdict(r: dict, floor: float) -> str:
    """The word this file is willing to put on a gap."""
    if r["t_q"] >= 0.05:
        return "indistinguishable"
    if abs(r["mean_delta"]) <= floor:
        return "significant but inside the seed floor"
    return "better" if r["mean_delta"] > 0 else "WORSE"


def readouts() -> list[tuple[str, str]]:
    """Every `(metric, panel)` the scored reports actually carry, DERIVED rather than listed."""
    found: set[tuple[str, str]] = set()
    for f in glob.glob(str(P / "eval_artifacts" / "*.allmodels.probes.json")):
        for e in json.loads(Path(f).read_text())["artifacts"].values():
            for metric, blk in e.items():
                if not isinstance(blk, dict):
                    continue
                for panel, v in blk.items():
                    if isinstance(v, dict) and "macro" in v:
                        found.add((metric, panel))
    return sorted(found)


def per_target(report: Path, subject: str, comparators: list[str]) -> dict:
    """Targets on which `subject` leads EVERY comparator, paired over the screen's targets."""
    d = json.loads(Path(report).read_text())
    res, targets = d["results"], d["targets"]
    def _v(model, target):
        return reports.per_target(d, res[model], target)

    subj = np.asarray([_v(subject, t) for t in targets], dtype=float)
    have = [b for b in comparators if b in res]
    led = []
    for i, t in enumerate(targets):
        others = [_v(b, t) for b in have]
        others = [o for o in others if o is not None]
        if others and all(np.isfinite(o) and subj[i] > o for o in others):
            led.append({"target": t, "subject": float(subj[i]), "best_other": float(max(others)),
                        "margin": float(subj[i] - max(others))})
    paired = []
    for b in have:
        o = np.asarray([_v(b, t) for t in targets], dtype=float)
        m = np.isfinite(subj) & np.isfinite(o)
        diff = subj[m] - o[m]
        paired.append({"baseline": b, "macro": res[b]["macro"], "mean_delta": float(diff.mean()),
                       "ahead": int((diff > 0).sum()), "n": int(m.sum()),
                       "sign_p": float(stats.binomtest(int((diff > 0).sum()), int(m.sum()), 0.5).pvalue),
                       "t_p": float(stats.ttest_rel(subj[m], o[m]).pvalue)})
    if paired:
        for r, q in zip(paired, bh(np.array([r["t_p"] for r in paired]))):
            r["t_q"] = float(q)
    return {"subject_macro": res[subject]["macro"], "n_targets": len(targets),
            "comparators": have, "led": led, "paired": paired,
            "expected_by_chance": len(targets) * 0.5 ** len(have)}


def count_representations() -> list[str]:
    """The representations whose INPUT is the count matrix, which is what a count comparison means."""
    from scrm_eval import controls

    return ["count_scanpy", *controls.PANEL]



def behind_best_count_fm(subject: str) -> dict:
    """The sentence in sec:identity, DERIVED: is `subject` behind the best count FM on every unit?

    Four decisions, each matching what the rest of the chain already does, so this cannot drift
    away from the tables the sentence cites:

    * SCOPE is `registry.datasets("eval")`. The per-baseline `readouts` below take every dataset
      scored on disk, which includes the E7 screen dataset and therefore counts ten where every
      published table counts nine -- `results_page.DS` is role-filtered and this must match it.
    * READOUTS are `registry.REPORT_PANELS`, the panels the paper publishes, not every
      `(metric, panel)` pair `readouts()` happens to discover.
    * "BEST count foundation model" is the per-unit MAX over `controls.PANEL`, the same five
      models E8's columns range over -- not one named model, and not `count_representations()`,
      which adds the `count_scanpy` PCA control and answers a different question.
    * UNITS differ per family on purpose: a dataset for gene detection, a dataset x panel for
      pathway (hence "measurements"), a target for the screen. The reported bound is the MAX of
      the three two-sided exact binomial p-values.
    """
    from scrm_eval import controls

    scope = set(R.datasets("eval"))
    fms = list(controls.PANEL)
    out: dict = {
        "scope": sorted(scope), "count_fms": fms,
        "definition": "behind = subject below the per-unit max over count_fms",
        "families": {},
    }
    for metric, panels in R.REPORT_PANELS.items():
        units, behind = [], 0
        for panel in panels:
            per = load(metric, panel)
            if subject not in per:
                continue
            for ds in sorted(scope & set(per[subject])):
                theirs = [per[m][ds] for m in fms if ds in per.get(m, {})]
                if not theirs:
                    continue
                units.append(f"{ds}/{panel}")
                behind += int(per[subject][ds] < max(theirs))
        if units:
            out["families"][metric] = {
                "n": len(units), "behind": behind, "units": units,
                "sign_p": float(stats.binomtest(behind, len(units), 0.5).pvalue),
            }
    f = P / "eval_reports" / f"e7_vs_control.{R.E7_DS}.json" if hasattr(R, "E7_DS") else None
    if f is None or not f.is_file():
        f = P / "eval_reports" / "e7_vs_control.crispr_a549_5k.json"
    if f.is_file():
        d = json.loads(f.read_text())
        res, targets = d["results"], d["targets"]
        have = [m for m in fms if m in res]
        behind, n = 0, 0
        for t in targets:
            mine = reports.per_target(d, res[subject], t)
            theirs = [reports.per_target(d, res[m], t) for m in have]
            theirs = [v for v in theirs if v is not None]
            if mine is None or not theirs:
                continue
            n += 1
            behind += int(mine < max(theirs))
        if n:
            out["families"]["screen"] = {
                "n": n, "behind": behind, "count_fms_present": have,
                "sign_p": float(stats.binomtest(behind, n, 0.5).pvalue),
            }
    if out["families"]:
        out["bound_p"] = max(v["sign_p"] for v in out["families"].values())
    return out

def in_domain_report(subject: str, reference: str) -> dict:
    """A model scored on ONE dataset, compared to `reference` across that dataset's panels."""
    seeds = [r for r in R.SEED_FLOOR_SET if r != reference]
    out: dict = {
        "subject": subject, "reference": reference,
        "test": "per-panel paired difference on one dataset, against the reference's reseed spread",
        "why_not_the_dataset_paired_test": (
            "the subject is IN-DOMAIN on its only scored dataset (registry.IN_DOMAIN), so the "
            "dataset-paired test has n = 1 and is undefined rather than merely weak"
        ),
        "panels_are_not_replicates": (
            "the panels of one dataset are views of one measurement, so they are listed and not "
            "aggregated into a p-value"
        ),
        "reseed_set": list(seeds), "panels": {},
    }
    for metric, panel in readouts():
        per = load(metric, panel)
        if subject not in per or reference not in per:
            continue
        shared = set(per[subject]) & set(per[reference])
        if len(shared) != 1:
            continue
        ds = next(iter(shared))
        vals = [per[r][ds] for r in seeds if r in per and ds in per[r]]
        spread = (max(vals) - min(vals)) if len(vals) > 1 else None
        delta = per[subject][ds] - per[reference][ds]
        out["panels"][f"{metric}/{panel}"] = {
            "dataset": ds, "subject": per[subject][ds], "reference": per[reference][ds],
            "delta": delta, "reseed_spread": spread,
            "delta_over_spread": (abs(delta) / spread) if spread else None,
            "clears_reseed_spread": (abs(delta) > spread) if spread else None,
        }
    out["n_panels"] = len(out["panels"])
    return out


def build_report(subject: str) -> dict:
    """Every comparison of `subject` against every count-side representation, on every readout."""
    from scrm_eval import controls, scoring

    out: dict = {
        "subject": subject,
        "seed_floor": R.SEED_FLOOR,
        "seed_floor_is_one_sided": True,
        "seed_floor_note": (
            "registry.SEED_FLOOR is the SUBJECT's own reseed spread. A frozen published checkpoint "
            "has no reseed set, so nothing here measures how much a baseline would move if it were "
            "retrained: the floor is a lower bound on a subject-versus-baseline gap's noise, never "
            "an upper one."
        ),
        "unit_of_analysis": "dataset for the panel readouts; target for the CRISPR screen",
        "tests": ["sign (exact binomial)", "paired t", "Wilcoxon signed-rank"],
        "multiplicity": "Benjamini-Hochberg over every baseline compared, per readout",
        "scoring_version": scoring.scoring_version(),
        "code_rev": controls.code_rev(),
        "readouts": {},
    }
    for metric, panel in readouts():
        per = load(metric, panel)
        if subject not in per:
            continue
        datasets = tuple(sorted(per[subject]))
        family = [m for m in R.baselines() if m in per]
        rows = compare(subject, family, per, datasets)
        if not rows:
            continue
        for r in rows:
            r["verdict"] = verdict(r, R.SEED_FLOOR)
        out["readouts"][f"{metric}/{panel}"] = {
            "metric": metric, "panel": panel, "datasets": list(datasets),
            "bh_family": [r["baseline"] for r in rows], "comparisons": rows,
        }
    others = [r for r in R.runs(required_only=True) if r != subject]
    if others:
        out["other_maintained_models"] = {o: in_domain_report(o, subject) for o in others}
    f = P / "eval_reports" / "e7_vs_control.crispr_a549_5k.json"
    if f.is_file():
        counts = count_representations()
        others = [b for b in R.baselines() if b not in counts]
        full = per_target(f, subject, counts + others)
        full["led_vs_count_representations"] = per_target(f, subject, counts)["led"]
        full["count_representations"] = counts
        out["crispr_screen"] = full
    out["behind_best_count_fm"] = behind_best_count_fm(subject)
    return out



def signif_macros(rep: dict) -> str:
    """The sentence's numbers as macros. The prose carried them as literals, and one was wrong.

    The bound is emitted as an EXACT dyadic fraction where it is one. The prose said
    `p <= 0.0039` while the value is 0.00390625, so the published inequality was strictly false
    by rounding; `2/512` states it exactly and cannot be rounded the wrong way by a later editor.
    """
    import math

    b = rep.get("behind_best_count_fm") or {}
    fams = b.get("families") or {}
    ns = R.TEX_NS

    def mac(name, val):
        return "\\newcommand{\\" + ns + name + "}{" + val + "}"

    def exact(p_):
        """`2/512` when the p-value is 2*2^-n, else a decimal rounded UP so `<=` stays true."""
        if p_ and p_ > 0:
            n = math.log2(2.0 / p_)
            if abs(n - round(n)) < 1e-9:
                return f"2/{2 ** int(round(n))}"
        return f"{math.ceil(p_ * 1e4) / 1e4:.4f}"

    out = ["% GENERATED by measurements/paired_significance.py --write-tex. Do not edit by hand.",
           "% behind = the subject below the per-unit max over controls.PANEL, over "
           "registry.datasets('eval')."]
    gd, pw, sc = fams.get("binary_expression"), fams.get("module_activity"), fams.get("screen")
    if gd:
        out += [mac("SignDetectionBehind", str(gd["behind"])), mac("SignDetectionN", str(gd["n"]))]
    if pw:
        out += [mac("SignPathwayBehind", str(pw["behind"])), mac("SignPathwayN", str(pw["n"]))]
    if sc:
        out += [mac("SignScreenBehind", str(sc["behind"])), mac("SignScreenN", str(sc["n"]))]
    if b.get("bound_p") is not None:
        out.append(mac("SignTestBound", exact(b["bound_p"])))
    return "\n".join(out)

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="binary_expression")
    ap.add_argument("--panel", default="hvg2000")
    ap.add_argument("--model", default=None,
                    help=f"the subject of the comparison (default: registry.FLAGSHIP, {R.FLAGSHIP})")
    ap.add_argument("--e7", action="store_true",
                    help="the CRISPR screen instead: paired over targets, plus the per-target "
                         "selection the main text makes a claim about")
    ap.add_argument("--write", action="store_true",
                    help="write every readout's comparisons to the report the page reads")
    ap.add_argument("-o", "--out", type=Path,
                    default=P / "eval_reports" / "paired_significance.json")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write-tex", type=Path, metavar="DIR",
                    help="also write signif_n.tex, the macros sec:identity states in prose")
    a = ap.parse_args(argv)

    # Each flag names one output and guards only that output, so either on its own does the work.
    # A producer that exits 0 having written no file is indistinguishable from success to a caller
    # that checks only the exit code.
    if a.write or a.write_tex:
        rep = build_report(a.model or R.FLAGSHIP)
        if a.write:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            tmp = a.out.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(rep, indent=2, sort_keys=True, default=str))
            tmp.replace(a.out)
            n = len(rep["readouts"])
            worse = sum(1 for b in rep["readouts"].values()
                        for c in b["comparisons"] if c["verdict"] == "WORSE")
            print(f"wrote {a.out}: {n} readouts, "
                  f"{sum(len(b['comparisons']) for b in rep['readouts'].values())} comparisons, "
                  f"{worse} of them significantly against the subject")
        if a.write_tex:
            a.write_tex.mkdir(parents=True, exist_ok=True)
            mf = a.write_tex / "signif_n.tex"
            tmp2 = mf.with_suffix(".tex.tmp")
            tmp2.write_text(signif_macros(rep) + "\n")
            tmp2.replace(mf)
            print(f"wrote {mf}")
        return 0
    if a.e7:
        f = P / "eval_reports" / "e7_vs_control.crispr_a549_5k.json"
        if not f.is_file():
            print(f"FAILED: {f} is not on disk")
            return 1
        counts = count_representations()
        r = per_target(f, a.model or R.FLAGSHIP, counts + [b for b in R.baselines() if b not in counts])
        if a.json:
            print(json.dumps(r, indent=2))
            return 0
        print(f"subject: {a.model or R.FLAGSHIP}   macro over {r['n_targets']} targets = "
              f"{r['subject_macro']:.4f}")
        print(f"\n{'baseline':18s} {'macro':>8} {'mean D':>9} {'ahead':>8} {'sign p':>8} "
              f"{'t p':>8} {'t q':>8}")
        print("-" * 72)
        for row in sorted(r["paired"], key=lambda x: x["mean_delta"]):
            print(f"{row['baseline']:18s} {row['macro']:8.4f} {row['mean_delta']:+9.4f} "
                  f"{row['ahead']:3d}/{row['n']:<4d} {row['sign_p']:8.4f} {row['t_p']:8.4f} "
                  f"{row['t_q']:8.4f}")
        led = per_target(f, a.model or R.FLAGSHIP, counts)["led"]
        print(f"\nagainst the {len(counts)} count representations, targets led: {len(led)} of "
              f"{r['n_targets']}")
        for x in sorted(led, key=lambda x: -x["margin"]):
            print(f"   {x['target']:10s} {x['subject']:.4f} vs best count {x['best_other']:.4f}"
                  f"   margin {x['margin']:+.4f}")
        print(f"against all {len(r['comparators'])} comparators, targets led: {len(r['led'])}")
        print("\nSelecting the targets that came out ahead, after seeing all of them, is a "
              "selection and not a test.")
        return 0

    subject = a.model or R.FLAGSHIP
    per = load(a.metric, a.panel)
    if subject not in per:
        print(f"FAILED: {subject!r} has no scored rows for {a.metric}/{a.panel}")
        return 1
    datasets = tuple(sorted(per[subject]))
    others = [m for m in R.baselines() if m in per]
    rows = compare(subject, others, per, datasets)
    if not rows:
        print("FAILED: no baseline shares at least 6 datasets with the subject")
        return 1

    if a.json:
        print(json.dumps({"subject": subject, "metric": a.metric, "panel": a.panel,
                          "datasets": list(datasets), "seed_floor": R.SEED_FLOOR,
                          "comparisons": rows}, indent=2))
        return 0

    print(f"subject: {subject}")
    print(f"metric:  {a.metric} / {a.panel}, macro over the panel, paired over {len(datasets)} datasets")
    print(f"floor:   registry.SEED_FLOOR = {R.SEED_FLOOR} (the subject's own reseed spread)")
    print()
    print(f"{'baseline':18s} {'mean D':>9} {'sd(D)':>8} {'ahead':>7} {'sign p':>8} {'t p':>8} "
          f"{'t q (BH)':>9}  verdict")
    print("-" * 104)
    for r in sorted(rows, key=lambda r: r["mean_delta"]):
        print(f"{r['baseline']:18s} {r['mean_delta']:+9.4f} {r['sd_delta']:8.4f} "
              f"{r['ahead']:3d}/{r['n']:<3d} {r['sign_p']:8.4f} {r['t_p']:8.4f} {r['t_q']:9.4f}  "
              f"{verdict(r, R.SEED_FLOOR)}")
    print()
    print("q is Benjamini-Hochberg over every baseline compared, not over the ones that passed.")
    print("A verdict of 'indistinguishable' means this evaluation cannot separate the two "
          "representations; it is a finding, not a missing measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
