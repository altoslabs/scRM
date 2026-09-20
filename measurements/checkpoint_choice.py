#!/usr/bin/env python3
"""Which checkpoint an arm is REPORTED at, measured rather than assumed."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "model"), str(REPO / "eval"), str(REPO / "target" / "py")]

from scrm_eval import artifact, controls, registry as R, run_probes  # noqa: E402

P = R.PROJ
CKPTSEL = P / "eval_ckptsel"
METRIC = "binary_expression"


def _published(run: str, ds: str) -> Path:
    """The publishable artifact path. One spelling, matching `eval_matrix.sbatch`'s `$art`."""
    return P / "eval_runs" / run / f"{ds}.{run}.allreads.arrow"


def candidates() -> list[tuple[str, str, str, Path]]:
    """`(run, ckpt_stem, dataset, path)` for every candidate embed on disk."""
    out = []
    if not CKPTSEL.is_dir():
        return out
    known = sorted(R.runs(), key=len, reverse=True)
    for d in sorted(CKPTSEL.iterdir()):
        if not d.is_dir():
            continue
        run = next((r for r in known if d.name.startswith(r + ".")), None)
        if run is None:
            print(f"  {d.name}: no declared run is a prefix of this directory — SKIPPED", file=sys.stderr)
            continue
        stem = d.name[len(run) + 1:]
        for f in sorted(d.glob(f"*.{run}.allreads.arrow")):
            out.append((run, stem, f.name.split(f".{run}.")[0], f))
    return out


def compare(run: str, stem: str, ds: str, cand: Path, out_dir: Path) -> dict:
    """Score the candidate, the reported checkpoint and the reseed set in ONE report."""
    rep = _published(run, ds)
    if not rep.is_file():
        return {"run": run, "candidate": stem, "dataset": ds, "verdict": "NO REPORTED ARTIFACT",
                "why": f"{rep.relative_to(P)} does not exist, so there is nothing to beat"}
    arts = {f"{run}.{stem}": artifact.read(cand), run: artifact.read(rep)}
    seeds = {}
    for s in R.SEED_FLOOR_SET:
        f = _published(s, ds)
        if f.is_file() and s not in arts:
            arts[s] = artifact.read(f)
            seeds[s] = None
    report = run_probes.probe_report(arts, ds, None, None)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"{ds}.{run}.{stem}.probes.json.tmp"
    tmp.write_text(json.dumps(report, indent=2))
    scored = out_dir / f"{ds}.{run}.{stem}.probes.json"
    tmp.replace(scored)

    def macro(key: str):
        e = (report["artifacts"].get(key) or {}).get(METRIC) or {}
        return (e.get(R.SPEC_PANEL) or {}).get("macro")

    got = {k: macro(k) for k in arts}
    bar_vals = [got[s] for s in R.SEED_FLOOR_SET if got.get(s) is not None]
    row = {"run": run, "candidate": stem, "dataset": ds, "panel": R.SPEC_PANEL, "metric": METRIC,
           "scored_report": str(scored.relative_to(P)), "auroc": got,
           "reseed_members_scored": len(bar_vals), "reseed_members_declared": len(R.SEED_FLOOR_SET),
           "registry_seed_floor": R.SEED_FLOOR, "reported_ckpt": R.reported_ckpt(run)}
    c, r = got.get(f"{run}.{stem}"), got.get(run)
    if c is None or r is None:
        row["verdict"] = "NOT SCORED"
        row["why"] = f"{R.SPEC_PANEL} is absent for the candidate or the reported checkpoint"
        return row
    row["delta"] = c - r
    if len(bar_vals) < 2:
        row["verdict"] = "UNDECIDABLE"
        row["why"] = (f"the noise bar needs at least two of {R.SEED_FLOOR_SET} scored on {ds}; "
                      f"{len(bar_vals)} are. registry.SEED_FLOOR is a nine-dataset mean on another "
                      "panel and is NOT substituted for it.")
        return row
    row["noise"] = max(bar_vals) - min(bar_vals)
    row["bar_source"] = (f"reseeds of {R.FLAGSHIP} on {ds}/{R.SPEC_PANEL}, not of {run} itself; "
                         f"{run} has no reseed set")
    row["verdict"] = "ADOPT" if row["delta"] > row["noise"] else "KEEP"
    row["why"] = (f"candidate {c:.4f} vs reported {r:.4f} = {row['delta']:+.4f} against a "
                  f"{row['noise']:.4f} spread over {len(bar_vals)} reseeds on {ds}/{R.SPEC_PANEL}")
    return row


def stamp_audit() -> list[dict]:
    """Every publishable artifact, and whether its stamp is the checkpoint the registry reports."""
    rows = []
    for run in R.runs():
        d = P / "eval_runs" / run
        if not d.is_dir():
            continue
        want = R.reported_ckpt(run)
        f = P / "runs" / run / want
        digest = controls.digest_file(f) if f.is_file() else None
        others = {c.name: controls.digest_file(c)
                  for c in sorted((P / "runs" / run).glob("*.pt")) if c.name != want}
        for art in sorted(d.glob(f"*.{run}.allreads.arrow")):
            got = artifact.read_stamp(art).checkpoint_sha256
            ok = digest is not None and got == digest
            rows.append({"run": run, "artifact": art.name, "reported_ckpt": want, "ok": ok,
                         "stamp_checkpoint_sha256": got,
                         "matches_instead": next((n for n, h in others.items() if h == got), None),
                         "reported_ckpt_present": digest is not None})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero when a verdict disagrees with registry.REPORTED_CKPT, or "
                         "when a publishable artifact was embedded from another checkpoint")
    ap.add_argument("--no-score", action="store_true",
                    help="run the stamp audit only; scores nothing and needs no count matrix")
    ap.add_argument("-o", "--out", type=Path, default=P / "eval_reports" / "checkpoint_choice.json")
    a = ap.parse_args(argv)

    print(f"reported checkpoint per arm (registry.REPORTED_CKPT, {len(R.REPORTED_CKPT)} exceptions):")
    for run in R.runs():
        mark = "  <== MEASURED EXCEPTION" if run in R.REPORTED_CKPT else ""
        print(f"  {run:44s} {R.reported_ckpt(run)}{mark}")

    print("\nstamp audit — every publishable artifact against the checkpoint the registry reports")
    audit = stamp_audit()
    bad = [r for r in audit if not r["ok"]]
    for r in bad:
        extra = f" (it is {r['matches_instead']})" if r["matches_instead"] else ""
        missing = "" if r["reported_ckpt_present"] else f" — {r['reported_ckpt']} is not on disk"
        print(f"  WRONG CHECKPOINT: {r['artifact']} is not {r['run']}/{r['reported_ckpt']}{extra}{missing}")
    print(f"  {len(audit) - len(bad)} of {len(audit)} publishable artifacts match")

    rows = []
    cands = []
    if not a.no_score:
        cands = candidates()
        print(f"\n{len(cands)} candidate embed(s) under {CKPTSEL.relative_to(P)}")
        for run, stem, ds, f in cands:
            rows.append(compare(run, stem, ds, f, P / "eval_reports" / "ckptsel"))
            r = rows[-1]
            print(f"  {run}.{stem} on {ds}: {r['verdict']} — {r.get('why', '')}")
    ds_owed = R.spec_dataset()
    owed = [(r, c, ds_owed) for r in R.runs() for c in R.ckpt_candidates(r)]
    have = {(r, f"{stem}.pt", ds) for r, stem, ds, _ in cands}
    unmet = [t for t in owed if t not in have]
    if a.no_score:
        print(f"\nthe registry owes {len(owed)} candidate comparison(s); NOT CHECKED under "
              "--no-score, which scores nothing. Run without it for the full gate.")
        unmet = []
    elif owed:
        print(f"\nthe registry owes {len(owed)} candidate comparison(s); "
              f"{len(owed) - len(unmet)} present")
        for r, c, ds in unmet:
            src = P / "runs" / r / c
            print(f"  MISSING: {r} vs {c} on {ds} — "
                  + (f"RUN={r} CKPT={c} ONE_DS={ds} sbatch --array=0 model/jobs/eval_matrix.sbatch"
                     if src.is_file() else f"{src.relative_to(P)} does not exist"))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"reported_ckpt": {r: R.reported_ckpt(r) for r in R.runs()},
                               "exceptions": dict(R.REPORTED_CKPT), "comparisons": rows,
                               "stamp_audit": audit}, indent=2, sort_keys=True, default=str))
    tmp.replace(a.out)
    print(f"\nwrote {a.out}")

    if not a.check:
        return 0
    fail = []
    for r, c, ds in unmet:
        fail.append(f"{r}: the registry declares {c} a candidate that must be measured against "
                    f"{R.reported_ckpt(r)} on {ds}, and no such comparison exists")
    if bad:
        fail.append(f"{len(bad)} publishable artifact(s) were embedded from a checkpoint the registry "
                    "does not report — re-embed them, or correct REPORTED_CKPT")
    for r in rows:
        named = R.reported_ckpt(r["run"]) == f"{r['candidate']}.pt"
        if r["verdict"] == "ADOPT" and not named:
            fail.append(f"{r['run']}: {r['candidate']} beats {r['reported_ckpt']} by "
                        f"{r['delta']:+.4f} over a {r['noise']:.4f} bar and REPORTED_CKPT does not "
                        "name it")
        if named and r["verdict"] != "ADOPT":
            fail.append(f"{r['run']}: REPORTED_CKPT names {r['candidate']} but the measurement says "
                        f"{r['verdict']} ({r.get('why', '')})")
    if fail:
        print("\nFAILED:")
        for m in fail:
            print(f"  {m}")
        return 1
    if a.no_score:
        print(f"\nOK (stamp half only): all {len(audit)} publishable artifacts carry the digest of "
              f"the checkpoint the registry reports. The {len(owed)} candidate comparison(s) were "
              "NOT measured; run without --no-score for that half.")
    else:
        print("\nOK: every arm is reported at the checkpoint the measurement supports, and every "
              "publishable artifact carries that checkpoint's digest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
