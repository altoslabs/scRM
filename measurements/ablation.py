#!/usr/bin/env python3
"""Ablation study for scrm: seven conditions, each removing one alternative explanation."""

from __future__ import annotations

import argparse
import collections
import json
import statistics as stats
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "model"), str(REPO / "eval"), str(REPO / "target" / "py")]

from scrm_eval import registry as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
from results_page import TEX_NS  # noqa: E402  the paper's macro namespace, declared once

P = R.PROJ

PANEL, METRIC = R.REPORT_PANEL, "binary_expression"

from results_page import TEX_MISSING  # noqa: E402

DS_EVAL = tuple(R.datasets("eval"))

FLAGSHIP = R.FLAGSHIP


class Condition:
    """One row, or one block of member rows, of the study."""

    def __init__(self, cid: str, label: str, members: tuple[str, ...], kills: str | None):
        self.cid, self.label, self.members, self.kills = cid, label, members, kills

    @property
    def status(self) -> str:
        """"ready", or what is missing. Covers TRAINED and EMBEDDED, which are disk facts."""
        if not self.members:
            return "no members declared yet"
        arms, gaps = set(R.runs()), []
        for m in self.members:
            if m not in arms:
                continue
            if not (P / "runs" / m / R.reported_ckpt(m)).is_file():
                gaps.append(f"{m} not trained")
                continue
            n = sum((P / "eval_runs" / m / f"{d}.{m}.allreads.arrow").is_file() for d in DS_EVAL)
            if n < len(DS_EVAL):
                gaps.append(f"{m} embedded on {n}/{len(DS_EVAL)}")
        return "ready" if not gaps else "; ".join(gaps)


CONDITIONS = (
    Condition("C1", "flagship", (FLAGSHIP,), None),
    Condition("C2", "untrained backbone, 3 random inits",
              ("armS96-untrained-i1", "armS96-untrained-i2", "armS96-untrained-i3"),
              "the architecture and the pooling do it, not the pretraining"),
    Condition("C3", "permuted molecules, depth held exactly",
              ("armS96-c3k-sigreg-lam002-permuted",),
              "it is not cell-specific sequence at all"),
    Condition("C4", "depth only", ("depth2",), "it is library size"),
    Condition("C5", "k-mer composition, no model (k=6)", ("kmer_bag",), "no learning is needed"),
    Condition("C7", "mean pooling instead of attention",
              ("armS96ML-c3k-sigreg-lam002", "armS96MA-c3k-sigreg-lam002"),
              "the learned pooling is not load-bearing"),
)

GRID_2x2: dict[tuple[str, str], str] = {}


def scores() -> dict[str, dict[str, float]]:
    """`member -> {dataset: E6 AUROC}`, read from the scored reports on disk."""
    out: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for ds in DS_EVAL:
        f = P / "eval_artifacts" / f"{ds}.allmodels.probes.json"
        if not f.is_file():
            continue
        for key, e in json.loads(f.read_text())["artifacts"].items():
            v = ((e.get(METRIC) or {}).get(PANEL) or {}).get("macro")
            if v is None:
                continue
            name = key.removeprefix(f"{ds}.").removesuffix(".allreads").removesuffix(".full")
            out[name][ds] = v
    return dict(out)


def _geometry_report() -> dict[str, dict]:
    """`member -> its `_mean` block` of `eval_reports/arm_geometry.json`, or `{}` when absent."""
    f = P / "eval_reports" / "arm_geometry.json"
    if not f.is_file():
        return {}
    d = json.loads(f.read_text())
    out = {}
    for block in ("arms", "baselines"):
        for name, per in (d.get(block) or {}).items():
            if per.get("_mean"):
                out[name] = per["_mean"]
    return out


def geometry() -> dict[str, float]:
    """`member -> mean effective rank`, or `{}` when the geometry report is absent."""
    return {n: m["effective_rank"] for n, m in _geometry_report().items()
            if m.get("effective_rank") is not None}


def _dims() -> dict[str, int]:
    """`member -> the width its effective rank is out of`, from the same report."""
    out = {}
    for n, m in _geometry_report().items():
        r, frac = m.get("effective_rank"), m.get("rank_fraction")
        if r and frac:
            out[n] = round(r / frac)
    return out


def _spread(values: list[float]) -> float:
    """P90 - P10, the unreliability statistic, with interpolated percentiles."""
    if len(values) < 2:
        return 0.0
    s = sorted(values)

    def q(p: float) -> float:
        i = p * (len(s) - 1)
        lo = int(i)
        return s[lo] if lo == len(s) - 1 else s[lo] + (i - lo) * (s[lo + 1] - s[lo])

    return q(0.9) - q(0.1)


def row(member: str, sc: dict, base: dict | None, geo: dict) -> dict:
    """One member's level, delta against `base`, sign count, dataset spread, rank and verdict."""
    per = sc.get(member, {})
    common = [d for d in DS_EVAL if d in per]
    r = {"member": member, "n_datasets": len(common),
         "level": stats.fmean(per[d] for d in common) if common else None,
         "dataset_spread": _spread([per[d] for d in common]),
         "effective_rank": geo.get(member),
         "delta": None, "better": None, "verdict": "INCOMPLETE"}
    if base and len(common) == len(DS_EVAL) and all(d in base for d in DS_EVAL):
        r["delta"] = stats.fmean(per[d] - base[d] for d in DS_EVAL)
        r["better"] = sum(1 for d in DS_EVAL if per[d] > base[d])
        if abs(r["delta"]) <= R.SEED_FLOOR:
            r["verdict"] = "null"
        elif r["better"] in (0, len(DS_EVAL)):
            r["verdict"] = "ESTABLISHED"
        else:
            r["verdict"] = "clears floor, mixed"
    elif len(common) == len(DS_EVAL):
        r["verdict"] = "reference"
    return r


def _measured_spread() -> float | None:
    """The flagship's own reseed spread, read from `seed_floor.py`'s artifact. None if unmeasured."""
    v = _flagship_floor().get("mean_range")
    return float(v) if v is not None else None


def _flagship_floor() -> dict:
    """`seed_floor.py`'s `flagship_floor` block, or `{}` when it has not been written yet."""
    f = R.PROJ / "eval_reports" / "seed_floor.json"
    if not f.is_file():
        return {}
    v = json.loads(f.read_text()).get("flagship_floor")
    return v if isinstance(v, dict) else {}


def _worst_dataset_range() -> float | None:
    """The widest SINGLE-DATASET reseed range in the same set, or None when unmeasured."""
    v = _flagship_floor().get("worst_range")
    return None if v is None else float(v)


MEMBER_LABEL = {
    FLAGSHIP: "reported model",
    "armS96-untrained-i1": "untrained, init 1",
    "armS96-untrained-i2": "untrained, init 2",
    "armS96-untrained-i3": "untrained, init 3",
    f"{R.seed_base(FLAGSHIP)}-permuted": "molecules permuted across cells, depth held",
    "depth2": "depth only",
    "kmer_bag": "hexamer frequencies, nothing trained",
    "armS96ML-c3k-sigreg-lam002": "window and view pools as means",
    "armS96MA-c3k-sigreg-lam002": "all four pools as means",
}


def _tex_label(member: str) -> str:
    """The paper's prose name for `member`, or fail."""
    if member not in MEMBER_LABEL:
        raise KeyError(
            f"{member} is a member of a declared condition but has no entry in MEMBER_LABEL, so the "
            f"generated table would print its arm name. Add the prose the appendix should show."
        )
    return _tex_escape(MEMBER_LABEL[member])


def _tex_cells(r: dict) -> tuple[str, str, str]:
    """`(level, delta, sign)` as LaTeX math, from the SAME dict the terminal row reads."""
    lvl = f"${r['level']:.4f}$" if r["level"] is not None else TEX_MISSING + "{level}"
    if r["delta"] is not None:
        dlt = f"${r['delta']:+.4f}$"
    elif r["member"] == FLAGSHIP:
        dlt = "---"
    else:
        dlt = TEX_MISSING + "{delta}"
    n = (f"${r['better']}/{len(DS_EVAL)}$" if r["better"] is not None
         else ("---" if r["member"] == FLAGSHIP else TEX_MISSING + "{n}"))
    return lvl, dlt, n


def _tex_escape(t: str) -> str:
    """LaTeX-safe prose for the label and `kills` columns, which are written as Python strings."""
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}")):
        t = t.replace(a, b)
    return t


def _series(items: list[str]) -> str:
    """`a`, `a and b`, `a, b and c` -- one condition's members as the paper punctuates a list."""
    if len(items) < 3:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _thousands(n: int) -> str:
    """`1{,}182{,}720`: LaTeX's thousands separator, which `tools/verify_generated.py` undoes."""
    return f"{n:,}".replace(",", "{,}")


_CID_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def _cid_stem(cid: str) -> str:
    """The macro-name stem for a condition id: `C7 -> Cseven`."""
    return "C" + _CID_WORDS[int(cid[1:])]


_GEOMETRY_QUOTED = (("Flag", FLAGSHIP), ("Count", "count_scanpy"),
                    ("Hex", "kmer_bag"), ("Geneformer", "geneformer"))


def _macros(collect, sc: dict, base: dict) -> str:
    """Every figure App. `sec:ablations` quotes, as `\newcommand` lines."""
    out: list[str] = []

    def cmd(name: str, value: str) -> None:
        out.append("\\newcommand{\\" + TEX_NS + name + "}{" + value + "}")

    cmd("AblationN", str(len(DS_EVAL)))
    cmd("SeedFloor", f"{R.SEED_FLOOR:.4f}")
    cmd("FlagSeed", str(R.recipe(R.FLAGSHIP)["seed"]))
    _ms = _measured_spread()
    cmd("SeedSpreadMeasured",
        f"{_ms:.5f}" if _ms is not None else TEX_MISSING + "{seed_spread}")

    _wr = _worst_dataset_range()
    cmd("SeedWorstRange", f"${_wr:.4f}$" if _wr is not None else TEX_MISSING + "{seed_worst}")
    if _wr is not None:
        cmd("SeedWorstVsFloor", "wider than" if _wr > R.SEED_FLOOR else "inside")
    cmd("SignTestP", f"1/{2 ** (len(DS_EVAL) - 1)}")
    cmd("SeedN", str(len(R.SEED_FLOOR_SET)))
    cmd("ReportPanel", _tex_escape(PANEL))

    flag = collect(FLAGSHIP)
    cmd("FlagLevel", f"${flag['level']:.4f}$" if flag["level"] is not None
        else TEX_MISSING + "{flag_level}")
    cmd("FlagSpread", f"${flag['dataset_spread']:.4f}$")

    deltas: list[float] = []
    for c in CONDITIONS:
        rows = [collect(m) for m in c.members]
        stem, one = _cid_stem(c.cid), len(c.members) == 1
        lvl = [f"${r['level']:.4f}$" if r["level"] is not None else TEX_MISSING + "{level}"
               for r in rows]
        cmd(stem + ("Level" if one else "Levels"), _series(lvl))
        if FLAGSHIP not in c.members:
            dlt = [f"${r['delta']:+.4f}$" if r["delta"] is not None else TEX_MISSING + "{delta}"
                   for r in rows]
            cmd(stem + ("Delta" if one else "Deltas"), _series(dlt))
            ahead = [f"${r['better']}$" if r["better"] is not None else TEX_MISSING + "{n}"
                     for r in rows]
            cmd(stem + "Ahead", _series(ahead))
            cmd(stem + ("Verdict" if one else "Verdicts"),
                _series([r["verdict"].lower() for r in rows]))
            deltas += [r["delta"] for r in rows if r["delta"] is not None]
            for field, fmt in (("better", lambda v: f"${v}$"), ("verdict", lambda v: v.lower())):
                vals = {r[field] for r in rows}
                if not one and len(vals) == 1 and None not in vals:
                    cmd(stem + ("Ahead" if field == "better" else "Verdict") + "All",
                        fmt(rows[0][field]))
        if not one:
            for r, suf in zip(rows, "ABCDEFG"):
                cmd(f"{stem}Level{suf}", f"${r['level']:.4f}$" if r["level"] is not None
                    else TEX_MISSING + "{level}")
                if r["delta"] is not None:
                    cmd(f"{stem}Delta{suf}", f"${r['delta']:+.4f}$")
                    cmd(f"{stem}Ahead{suf}", f"${r['better']}$")
                    cmd(f"{stem}Verdict{suf}", r["verdict"].lower())
            levels = [r["level"] for r in rows if r["level"] is not None]
            if len(levels) > 1:
                spread = max(levels) - min(levels)
                cmd(stem + "Spread", f"${spread:.4f}$")
                cmd(stem + "SpreadVsFloor", "inside" if spread <= R.SEED_FLOOR else "outside")
        ref = R.TRAINING[FLAGSHIP][0]
        off = [R.TRAINING[m][0] for m in c.members
               if m in R.TRAINING and R.TRAINING[m][0] != ref]
        if off:
            cmd(stem + "ParamCuts",
                _series([f"${_thousands(ref - n)}$ (${(ref - n) / ref:.1%}$)".replace("%", r"\%")
                         for n in off]))
    cmd("FlagParams", f"${_thousands(R.TRAINING[FLAGSHIP][0])}$")

    if deltas:
        widest = max(abs(d) for d in deltas)
        cmd("MaxAbsDelta", f"${widest:.4f}$")
        cmd("FlagSpreadVsDelta",
            "larger than" if flag["dataset_spread"] > widest else "smaller than")

    hex_per = sc.get("kmer_bag", {})
    wins = sorted((base[d] - hex_per[d] for d in DS_EVAL if d in hex_per and d in base
                   and base[d] > hex_per[d]), reverse=True)
    cmd("CfiveBehind", f"${len(wins)}$")
    cmd("CfiveMargins", _series([f"${w:.4f}$" for w in wins]) if wins
        else TEX_MISSING + "{c5_margins}")
    from scrm_eval.controls import KMER_K
    cmd("HexK", f"${KMER_K}$")
    cmd("HexFeatures", f"$4^{KMER_K} = {_thousands(4 ** KMER_K)}$")

    ranks, dims = geometry(), _dims()
    for name, member in _GEOMETRY_QUOTED:
        rk, dm = ranks.get(member), dims.get(member)
        cmd(name + "Rank", f"${rk:.1f}$" if rk is not None else TEX_MISSING + "{rank}")
        cmd(name + "Dim", f"${dm}$" if dm is not None else TEX_MISSING + "{dim}")
    return "\n".join(out)


def tex_tables(collect, sc: dict, base: dict) -> dict[str, str]:
    """The bodies of `tab:ablation` and `tab:lambda2x2`, derived from the same `collect` the
    terminal render uses.
    """
    body = []
    for c in CONDITIONS:
        if not c.members:
            gap = f"{TEX_MISSING}{{{c.cid}}}"
            body.append(f"{c.cid} & {_tex_escape(c.label)} & {_tex_escape(c.kills or '')} & "
                        f"{gap} & {gap} & {gap} & not built \\\\")
            continue
        rows = [collect(m) for m in c.members]
        for i, r in enumerate(rows):
            lvl, dlt, n = _tex_cells(r)
            cid = c.cid if i == 0 else ""
            kills = _tex_escape(c.kills or "")
            if i == 0 and len(rows) > 1:
                kills = (f"\\multirow{{{len(rows)}}}{{0.24\\textwidth}}{{{kills}}}")
            elif i:
                kills = ""
            body.append(f"{cid} & {_tex_label(r['member'])} & {kills} & {lvl} & {dlt} & {n} & "
                        f"{r['verdict'].lower()} \\\\")
    grid = []
    for (reg, lam), member in GRID_2x2.items():
        r = collect(member)
        lvl, dlt, n = _tex_cells(r)
        grid.append(f"{_tex_escape(reg)} & {_tex_escape(str(lam))} & {_tex_label(member)} & "
                    f"{lvl} & {dlt} & {n} & {r['verdict'].lower()} \\\\")
    # lambda2x2_body is not emitted: the paper references it nowhere, and this release supports
    # only what the paper shows.
    return {"ablation_body": "\n".join(body),
            "ablation_n": _macros(collect, sc, base)}


def _cells(r: dict) -> tuple[str, str, str, str]:
    """The four formatted numbers of a member row: level, delta, sign count, effective rank."""
    lvl = f"{r['level']:.4f}" if r["level"] is not None else "—"
    if r["delta"] is not None:
        dlt = f"{r['delta']:+.4f}"
    else:
        dlt = "  (base)" if r["member"] == FLAGSHIP else "—"
    bt = f"{r['better']}/{len(DS_EVAL)}" if r["better"] is not None else "—"
    rk = f"{r['effective_rank']:.1f}" if r["effective_rank"] is not None else "n/a"
    return lvl, dlt, bt, rk


def render(check_only: bool, tex_dir: Path | None = None, tex_check: bool = False) -> int:
    """Print the study, or with `check_only` just its completeness verdict. Returns an exit code."""
    sc, geo = scores(), geometry()
    base = sc.get(FLAGSHIP, {})
    if len(base) != len(DS_EVAL):
        print(f"FAILED: the flagship {FLAGSHIP} is scored on {len(base)} of {len(DS_EVAL)} datasets; "
              "every delta in the study is measured against it")
        return 1

    missing: list[str] = []
    incomplete: list[str] = []

    def collect(member: str) -> dict:
        r = row(member, sc, None if member == FLAGSHIP else base, geo)
        if r["n_datasets"] < len(DS_EVAL):
            incomplete.append(f"{member}: {r['n_datasets']}/{len(DS_EVAL)} datasets")
        return r

    show = not check_only
    if show:
        print(f"metric: E6 detection AUROC ({PANEL}), mean over {len(DS_EVAL)} held-out datasets")
        print(f"floor:  {R.SEED_FLOOR} on the mean — the flagship's own reseed spread over "
              f"{len(R.SEED_FLOOR_SET)} seeds and all {len(DS_EVAL)} datasets.")
        ref = R.TRAINING[R.FLAGSHIP][0]
        off = [(m, R.TRAINING[m][0]) for c in CONDITIONS for m in c.members
               if m in R.TRAINING and R.TRAINING[m][0] != ref]
        if off:
            print("        NOT parameter-matched to the flagship's "
                  f"{ref:,}, so the floor bounds one side only: "
                  + ", ".join(f"{m} at {n:,} ({(n - ref) / ref:+.1%})" for m, n in off))
        else:
            print(f"        every member holds the flagship's {ref:,} parameters, so the floor "
                  "bounds both sides of every delta.")
        print(f"{'':4} {'condition / member':46} {'level':>7} {'delta':>8} "
              f"{'n/' + str(len(DS_EVAL)):>5} {'ds spread':>9} {'rank':>6}  verdict")
        print("-" * 118)

    for c in CONDITIONS:
        if show:
            head = f"{c.cid}  {c.label}"
            print(f"{head:51}" + (f"   [{c.status}]" if c.status != "ready" else ""))
            if c.kills:
                print(f"     kills: {c.kills}")
        if not c.members:
            missing.append(f"{c.cid} {c.label}: no members declared yet ({c.status})")
            continue
        rows = [collect(m) for m in c.members]
        if not show:
            continue
        for r in rows:
            lvl, dlt, bt, rk = _cells(r)
            print(f"     {r['member']:46} {lvl:>7} {dlt:>8} {bt:>5} "
                  f"{r['dataset_spread']:>9.4f} {rk:>6}  {r['verdict']}")
        levels = [r["level"] for r in rows if r["level"] is not None]
        if len(levels) > 1:
            print(f"     {'-> seed/init spread across members':46} "
                  f"{max(levels) - min(levels):>7.4f}   (level range; NOT the dataset spread)")

    if show:
        print()
        print("the 2x2 (regulariser x lambda) -- the interaction is the result")
    for (reg, lam), member in GRID_2x2.items():
        r = collect(member)
        if show:
            lvl, dlt, _, _ = _cells(r)
            print(f"     {reg:7} lambda={lam:14} {member:34} {lvl:>7} {dlt:>8}  {r['verdict']}")

    if show:
        print()
    if missing or incomplete:
        for m in missing:
            print(f"  NOT BUILT   {m}")
        for m in sorted(set(incomplete)):
            print(f"  INCOMPLETE  {m}")
        print(f"FAILED: {len(missing)} condition(s) not built, "
              f"{len(set(incomplete))} member(s) short of {len(DS_EVAL)} datasets. "
              "A table with a blank cell is not a result.")
        if tex_dir:
            _tex_dispose(tex_dir, check=tex_check, collect=collect, sc=sc, base=base)
        return 1
    print(f"OK: every declared member is scored on all {len(DS_EVAL)} datasets")
    if tex_dir:
        return _tex_dispose(tex_dir, check=tex_check, collect=collect, sc=sc, base=base)
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Ablation study for scrm: seven conditions, each removing one alternative "
                    "explanation. See the module docstring for the design and the floor rule.")
    ap.add_argument("--check", action="store_true",
                    help="print only the completeness verdict, and exit non-zero if any declared "
                         "member is missing or scored on fewer than all held-out datasets")
    ap.add_argument("--write-tex", type=Path, metavar="DIR",
                    help="write generated/{ablation_body,ablation_n}.tex into DIR")
    ap.add_argument("--check-tex", type=Path, metavar="DIR",
                    help="compare DIR's fragments against a fresh derivation; non-zero on drift")
    a = ap.parse_args(argv[1:])
    if a.write_tex and a.check_tex:
        ap.error("--write-tex and --check-tex are one derivation with two dispositions; pass one")
    return render(a.check, tex_dir=a.write_tex or a.check_tex,
                  tex_check=bool(a.check_tex))


def _tex_dispose(out_dir: Path, *, check: bool, collect, sc: dict, base: dict) -> int:
    """Write or verify this module's fragments. Returns an exit code, so it can gate `main`."""
    frags = tex_tables(collect, sc, base)
    problems = 0
    for name, body in sorted(frags.items()):
        f = Path(out_dir) / f"{name}.tex"
        body = body.rstrip("\n") + "\n"
        n_missing = body.count(TEX_MISSING.lstrip(chr(92)))
        if n_missing:
            print(f"  [tex] {f.name}: {n_missing} cell(s) no report can produce")
            problems += n_missing
        import results_page as _rp
        for b in _rp.validate_tex(name, body):
            print(f"  [tex] MALFORMED {b}")
            problems += 1
        if check:
            have = f.read_text() if f.is_file() else None
            if have != body:
                print(f"  [tex] STALE {f}: "
                      f"{'absent' if have is None else 'differs from a fresh derivation'}")
                problems += 1
            continue
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        f.write_text(body)
        print(f"  [tex] wrote {f}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
