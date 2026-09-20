"""Derive the paper's generated table fragments from the stamped reports on disk.

Nothing is transcribed: every value is read programmatically from a stamped report. This
module emits the tex fragments the paper \\input{}s and nothing else — there is no page.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from scrm_eval import reports as _reports
from scrm_eval import artifact as _artifact, controls, reduce as _reduce, registry as _registry  # noqa: E402

P = _registry.PROJ

#: OUR producer key, DERIVED from the declaration in `artifact.MODEL_PRODUCERS` rather than
#: repeated as a literal. Every stamp comparison and every report lookup below goes through this
#: name. The single-member assertion holds the fragment code to ONE "ours" column per table.
_OURS = sorted(_artifact.MODEL_PRODUCERS)
assert len(_OURS) == 1, (
    f"results_page reports ONE 'ours' column per fragment; artifact.MODEL_PRODUCERS "
    f"declares {_OURS}. Decide what each table shows per producer before adding a second."
)
OURS = _OURS[0]

#: How the model is NAMED where a human reads it. OURS is a PRODUCER KEY -- the string an
#: artifact stamps itself with -- which is right for looking a row up and wrong for typesetting.
OURS_LABEL = "scRM"


def scrm_rule_applied(rep: dict) -> bool:
    """Did this report's scrm rows come from a STANDARDISED embedding?"""
    applied = rep.get("reduction_applied")
    if not isinstance(applied, dict):
        return False
    return applied.get(OURS) == _reduce.SCRM_RULE


full_coverage = _registry.full_coverage
TRAIN_ONLY = tuple(d for d, _, role in _registry.DATASETS if role == "train")
_DS_PREFERRED = ("sc_mixology_5cl", "fly_antenna", "nuclei_2k", "pbmc_10k_v3", "got_ET01",
                 "tm_10X_P7_2", "heart_10k_v3", "neuron_10k_v3", "pbmc_1k_v2")

E2_DS = "sc_mixology_5cl"
E3_DS = "tm_10X_P7_2"
E7_DS = "crispr_a549_5k"
for _t, _d in (("E2", E2_DS), ("E3", E3_DS)):
    assert _d in _registry.datasets("eval"), f"{_t}'s dataset {_d} is not a registered eval dataset"
    assert _registry.label_file(_d), f"{_t}'s dataset {_d} has no independent label file"
_DS_ALL = _registry.datasets("eval")
DS = tuple(d for d in _DS_PREFERRED if d in _DS_ALL) + tuple(
    d for d in _DS_ALL if d not in _DS_PREFERRED)
assert not set(DS) & set(TRAIN_ONLY), "an evaluation table may not contain a training dataset"
FAMILY_SUFFIXES = ("-1pass",)


def model_family(run: str) -> str:
    """Strip TRAINING LENGTH from a run name; everything else is the model's identity."""
    for suf in FAMILY_SUFFIXES:
        if run.endswith(suf):
            return run[: -len(suf)]
    return run


def _e6_per_dataset() -> dict[str, dict[str, float]]:
    """run -> {dataset: E6 hvg2000}. Kept PER DATASET, not pre-averaged, because the selection has to
    compare members on the datasets they SHARE.
    """
    import collections
    score: dict[str, dict[str, float]] = collections.defaultdict(dict)
    refused: list[str] = []
    ungated: list[str] = []
    for ds in DS:
        f = P / f"eval_artifacts/{ds}.allmodels.probes.json"
        if not f.exists():
            continue
        try:
            need = full_coverage(ds)
        except (KeyError, FileNotFoundError, _registry.StaleRequirement):
            ungated.append(ds)
            continue
        for k, e in (json.load(open(f)).get("artifacts") or {}).items():
            if e.get("producer") not in _artifact.MODEL_PRODUCERS:
                continue
            if (e.get("w_eval") or 0) < need:
                refused.append(f"{ds}/{k} (w_eval={e.get('w_eval')})")
                continue
            run = k.removeprefix(f"{ds}.").removesuffix(".allreads").removesuffix(".full")
            v = ((e.get("binary_expression") or {}).get("hvg2000") or {}).get("macro")
            if v is not None:
                score[run][ds] = v
    if refused:
        print(f"  [selection] REFUSED {len(refused)} artifact(s) below full coverage: "
              + ", ".join(refused[:6]) + (" ..." if len(refused) > 6 else ""))
    if ungated:
        print(f"  [selection] SKIPPED {len(ungated)} dataset(s) with no pinned all-reads "
              f"requirement, so selection cannot be gated on them: {', '.join(ungated)}")
    return dict(score)


def best_per_family(runs) -> dict[str, str]:
    """family -> the run with the highest mean E6 hvg2000, from the reports on disk."""
    import collections
    per = _E6
    fam: dict[str, list[str]] = collections.defaultdict(list)
    for r in runs:
        fam[model_family(r)].append(r)
    out = {}
    for family, members in fam.items():
        rated = [r for r in members if per.get(r)]
        if not rated:
            out[family] = sorted(members)[0]
            continue
        common = set.intersection(*(set(per[r]) for r in rated))
        if common:
            out[family] = max(rated, key=lambda r: sum(per[r][d] for d in common) / len(common))
        else:
            out[family] = max(rated, key=lambda r: len(per[r]))
    return out

_ARM_LABELS = _registry.labels()
_ARM_ORDER = tuple(_registry.runs())
ALL_ARMS = tuple((r, _ARM_LABELS[r]) for r in _ARM_ORDER)
_E6 = _e6_per_dataset()
_SELECTED = best_per_family(_ARM_ORDER)
ARMS = tuple((r, _ARM_LABELS[r]) for r in _ARM_ORDER if _SELECTED.get(model_family(r)) == r)
_SCORED = {r for r, v in _E6.items() if v}
NOT_SELECTED = {r: (_SELECTED[model_family(r)],
                    "scored lower" if r in _SCORED else "not scored yet")
                for r in _ARM_ORDER if _SELECTED.get(model_family(r)) not in (r, None)}


_ARM_OF = {label: key for key, label in ALL_ARMS}

DROPPED_PRODUCERS = {"counts_10x_pca10"}
NICE = {
    "depth2": "depth-only floor", "genejepa": "GeneJepa", "scvi_pretrained": "scVI",
    "geneformer": "Geneformer", "scgpt": "scGPT", "uce": "UCE (multi-species)",
    "count_scanpy": "counts, scanpy PCA-50 (reference)",
    "counts_10x_pca10": "counts, 10x PCA-10 (legacy)",
    "kmer_bag": "k-mer composition, no model (k=6)",
}

IN_DOMAIN = dict(_registry.IN_DOMAIN)


def in_domain(label: str, dataset: str) -> bool:
    """Is this arm's score on this dataset an in-domain score?"""
    return _ARM_OF.get(label) in IN_DOMAIN.get(dataset, ())


def collect():
    fam: dict[str, dict[str, dict[str, float]]] = {}
    CI: dict = {}

    def put(metric, model, ds, panel, v, ci=None, sd=None):
        col = f"{ds}|{panel}"
        fam.setdefault(metric, {}).setdefault(model, {})[col] = v
        if ci or sd is not None:
            CI[(metric, model, col)] = (ci or (), sd)

    def harvest(entry, model, ds):
        for panel, r in entry.get("binary_expression", {}).items():
            put("detection AUROC", model, ds, panel, r["macro"], r.get("ci95"), r.get("seed_sd"))
        for variant, r in entry.get("count_likelihood", {}).items():
            put("count deviance explained", model, ds, variant, r["macro"], r.get("ci95"), r.get("seed_sd"))
        for variant, r in entry.get("graded_expression", {}).items():
            put("graded expression r", model, ds, variant, r["macro"], r.get("ci95"), r.get("seed_sd"))
        for variant, r in entry.get("module_activity", {}).items():
            put("pathway activity r", model, ds, variant, r["macro"], r.get("ci95"), r.get("seed_sd"))

    label_of = dict(ARMS)
    skipped: list[str] = []
    for ds in DS:
        combined = P / f"eval_artifacts/{ds}.allmodels.probes.json"
        if combined.exists():
            for key, e in json.load(open(combined))["artifacts"].items():
                if e["producer"] not in KNOWN_PRODUCERS:
                    raise SystemExit(
                        f"{combined.name}: {key} names producer {e['producer']!r}, which is neither "
                        f"a model producer nor a baseline. Known: {sorted(KNOWN_PRODUCERS)}. A "
                        f"producer this file does not recognise would be treated as a baseline and "
                        f"so would bypass the coverage refusal below. If this report predates the "
                        f"rename, re-score it; do not add the old name here."
                    )
                if e["producer"] in _artifact.MODEL_PRODUCERS and e.get("w_eval", 0) < full_coverage(ds):
                    skipped.append(f"{ds}/{key} (w_eval={e.get('w_eval')})")
                    continue
                if e["producer"] in _artifact.MODEL_PRODUCERS:
                    run = key.removeprefix(f"{ds}.")
                    for suffix in (".allreads", ".full"):
                        run = run.removesuffix(suffix)
                    if run in NOT_SELECTED:
                        continue
                    if run not in label_of:
                        raise SystemExit(
                            f"{ds}/{key}: run {run!r} is not a known arm. Refusing to render a table "
                            f"that silently omits a trained model (rule 2). Known: {sorted(label_of)}"
                        )
                    model = label_of.get(run, run)
                else:
                    if e["producer"] in DROPPED_PRODUCERS:
                        continue
                    model = NICE.get(e["producer"], e["producer"])
                harvest(e, model, ds)
            continue

    e2: dict[str, dict] = {}

    ann = P / f"eval_artifacts/{E2_DS}.allmodels.annotation.json"
    _ann_rep = _annotation_report(ann)
    if _ann_rep is not None:
        rep = _ann_rep
        for key, e in (rep.get("artifacts") or {}).items():
            run = key.removeprefix(f"{E2_DS}.")
            for suffix in (".allreads", ".full"):
                run = run.removesuffix(suffix)
            if run not in dict(ARMS):
                nice = NICE.get(e.get("producer"))
                if not nice or e.get("producer") in DROPPED_PRODUCERS:
                    continue
                def _norm(x):
                    return x.replace(" (reference)", "").strip()
                for dup in [k for k in e2 if k != nice and _norm(k) == _norm(nice)]:
                    del e2[dup]
                e2[nice] = {m: [e[m]["macro"], e[m].get("baseline", 0.0), e[m]["n_cells"]]
                            for m in ("linear", "knn_cosine", "mlp") if m in e}
                continue
            if e.get("w_eval", 0) < full_coverage(E2_DS):
                skipped.append(f"{E2_DS}/{key} E2 (w_eval={e.get('w_eval')})")
                continue
            label = dict(ARMS)[run]
            if in_domain(label, E2_DS):
                label += " (IN-DOMAIN: trained on this dataset)"
            e2[label] = {m: [e[m]["macro"], e[m].get("baseline", 0.0), e[m]["n_cells"]]
                         for m in ("linear", "knn_cosine", "mlp") if m in e}

    mlm = {}
    for run, label in ARMS:
        lg = P / f"runs/{run}/log.jsonl"
        if lg.exists():
            lines = lg.read_text().splitlines()
            if lines:
                last = json.loads(lines[-1])
                mlm[label] = (last.get("loss_mlm_bits"), last.get("step"), last.get("eff_rank_cell"))
    if skipped:
        print(f"REFUSED {len(skipped)} non-full-coverage rows: {skipped[:6]}{' …' if len(skipped) > 6 else ''}")
    return fam, e2, mlm, CI


_E8_PAPER_ROWS = (
    ("E2", "cell-line identity", "read genotypes", "macro-F1"),
    ("E7", "perturbation response", "CRISPR guides", "macro AUROC"),
    ("E6", "gene detection", "the count matrix", "AUROC"),
    ("E3", "cell-type annotation", "count-matrix clusters", "macro-F1"),
    ("E9", "pathway activity", "the count matrix", "Pearson $r$"),
)


def contrast_table(P, fam, e2, e3):
    """E8: for each task, the worst and best count model against the reported scrm arm.

    Returns the paper's table body. The comparison is the deliverable; there is no page.
    """
    COUNTS = {NICE[k] for k in controls.PANEL if k in NICE}

    def best(d, dataset=None):
        """(worst count, best count, best held-out scrm) as (value, label) triples for one mapping."""
        cnt = [(v, k) for k, v in d.items() if k in COUNTS and v is not None]
        ours = [(v, k) for k, v in d.items()
                if v is not None and _ARM_OF.get(k) == _registry.FLAGSHIP]
        if len(cnt) < 2:
            return (None, None, (max(ours) if ours else None), len(cnt))
        return (min(cnt), max(cnt), (max(ours) if ours else None), len(cnt))

    rows = []
    for tid, task, ds, src, d in (
            ("E2", "cell-line identity", E2_DS, "demuxlet genotypes (variants in reads)",
             {k: v["linear"][0] for k, v in e2.items()}),
            ("E3", "cell-type annotation", E3_DS, "expert cell-ontology curation",
             e3 or {})):
        w, c, o, ncnt = best(d, ds)
        if c and o:
            rows.append((tid, task, src, "independent of counts", w, c, o, ncnt))
        elif o and ncnt < 2:
            print(f"  [{tid}] REFUSED from the E8 table: {ncnt} count foundation model(s) have a "
                  f"value, and 'worst FM | best FM' is not a range over one")
    for tid, task, metric, family in (("E6", "gene detection", "detection AUROC",
                                       "binary_expression"),
                                      ("E9", "pathway activity", "pathway activity r",
                                       "module_activity")):
        want = _registry.REPORT_PANELS[family]
        d = {m: {c: v for c, v in cols.items() if c.split("|", 1)[1] in want}
             for m, cols in (fam.get(metric) or {}).items()}
        d = {m: cols for m, cols in d.items() if cols}
        all_cols = set()
        for cols in d.values():
            all_cols |= {c for c, v in cols.items() if v is not None}
        flat, short = {}, []
        for k, cols in d.items():
            vals = [v for c, v in cols.items()
                    if v is not None and not in_domain(k, c.split("|")[0])]
            if len(vals) < len(all_cols):
                short.append((k, len(vals)))
                continue
            flat[k] = sum(vals) / len(vals)
        if short:
            print(f"  [{tid}] {len(short)} model(s) below full coverage "
                  f"({len(all_cols)} columns), excluded from the contrast:")
            for k, n in sorted(short, key=lambda x: -x[1]):
                print(f"        {n:>3}/{len(all_cols)}  {k}")
        w, c, o, ncnt = best(flat)
        if c and o:
            rows.append((tid, task, "the count matrix itself", "count-derived", w, c, o, ncnt))
        else:
            print(f"  [{tid}] REFUSED from the E8 table: no count/scrm pair holds all "
                  f"{len(all_cols)} columns")
    f7 = P / "eval_reports" / f"e7_vs_control.{E7_DS}.json"
    if f7.exists():
        R7 = json.loads(f7.read_text())["results"]
        d7 = {NICE.get(k, dict(ALL_ARMS).get(k, k)): v["macro"] for k, v in R7.items()}
        w, c, o, ncnt = best(d7, E7_DS)
        if c and o:
            rows.append(("E7", "perturbation response", "CRISPR guide capture (a separate library)",
                         "independent of counts", w, c, o, ncnt))
    if not rows:
        print("  [E8] no task has both a count and a held-out scrm row, so every cell is marked "
              "missing rather than dotted")
    else:
        print("  [E8] the paper's Table 1: worst count FM & scrm (reported) & best count FM")
        for tid, task, _s, _r, (wv, wk), (cv, ck), (ov, ok), ncnt in rows:
            print(f"        {task:22s} ${wv:.3f}$ & ${ov:.3f}$ & ${cv:.3f}$   "
                  f"(worst={wk}, best={ck}, ours={ok}, over {ncnt} count FMs)")
    got = {tid: r for r in rows for tid in (r[0],)}
    body = []
    for tid, task, src, metric in _E8_PAPER_ROWS:
        r = got.get(tid)
        if r is None:
            cells = f"{TEX_MISSING}{{{tid}}} & {TEX_MISSING}{{{tid}}} & {TEX_MISSING}{{{tid}}}"
        else:
            (wv, _wk), (cv, _ck), (ov, _ok) = r[4], r[5], r[6]
            cells = f"${wv:.3f}$ & ${ov:.3f}$ & ${cv:.3f}$"
        body.append(f"{task:21s}& {src:22s}& {metric:13s}& {cells} \\\\")
    #: The E2 gap the main text quotes, produced by the table's own cells.
    _e2 = got.get("E2")
    _gap = (_e2[5][0] - _e2[6][0]) if _e2 else None
    macros = ("\\newcommand{\\" + TEX_NS + "EeightEtwoGap}{"
              + (f"${_gap:.3f}$" if _gap is not None else TEX_MISSING + "{E2}") + "}")
    return "\n".join(body), macros


_TEX: dict[str, str] = {}

#: The paper's LaTeX macro namespace, from the single declaration in `registry`. Every fragment
#: this file writes binds `\\<TEX_NS>...`, and the paper resolves the same prefix.
TEX_NS = _registry.TEX_NS
TEX_MISSING = "\\" + TEX_NS + "Missing"


TEX_COLUMNS = {"e8_body": 6, "ablation_body": 7,
               "dataurls_body": 4, "c5_body": 5, "specialize_body": 10, "e7_body": 9,
               "e7full_body": 9, "e7spec_body": 8, "e7rank_body": 4}


def validate_tex(name: str, body: str) -> list[str]:
    """Structural problems in one fragment: column count, brace balance. Empty means well-formed."""
    bad = []
    want = TEX_COLUMNS.get(name)
    for i, line in enumerate(body.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("%") or line.startswith("\\newcommand"):
            continue
        mc = "multicolumn{" in line
        if mc:
            span = line.split("multicolumn{", 1)[1].split("}", 1)[0]
            if want is not None and span.isdigit() and int(span) > want:
                bad.append(f"{name}:{i} multicolumn spans {span} of {want} columns")
            if not (span.isdigit() and int(span) == 1):
                want = None
        if want is not None:
            n = len([c for j, c in enumerate(line) if c == "&" and (j == 0 or line[j - 1] != "\\")])
            if n != want - 1:
                bad.append(f"{name}:{i} has {n + 1} columns, the table declares {want}")
        if line.count("{") != line.count("}"):
            bad.append(f"{name}:{i} unbalanced braces ({line.count('{')} open, "
                       f"{line.count('}')} close)")
        if not line.endswith(chr(92) * 2):
            bad.append(f"{name}:{i} does not end a LaTeX row")
    return bad


def emit_tex(name: str, body: str) -> None:
    """Record one generated fragment. Overwriting is a defect, not an update."""
    if name in _TEX:
        raise KeyError(f"{name} emitted twice; two producers claim one fragment of the paper")
    _TEX[name] = body.rstrip("\n") + "\n"


def write_tex(out_dir: Path, *, check: bool = False) -> int:
    """Write every fragment under `out_dir`, or with `check` compare and report drift."""
    out_dir = Path(out_dir)
    problems = 0
    if not check:
        out_dir.mkdir(parents=True, exist_ok=True)
    for name, body in sorted(_TEX.items()):
        f = out_dir / f"{name}.tex"
        n_missing = body.count(TEX_MISSING)
        if n_missing:
            problems += n_missing
            # "no report can produce it" and "nobody named a data root" are two diagnoses.
            # Name which one applies.
            why = (f"NO DATA ROOT, so no report was even opened -- {_registry.ROOT_HINT}"
                   if _registry.root_unset() else "no report can produce yet")
            print(f"  [tex] {f.name}: {n_missing} cell(s) {why}", flush=True)
        for b in validate_tex(name, body):
            print(f"  [tex] MALFORMED {b}", flush=True)
            problems += 1
        if check:
            have = f.read_text() if f.is_file() else None
            if have != body:
                problems += 1
                why = "absent" if have is None else "differs from a fresh derivation"
                print(f"  [tex] STALE {f}: {why}", flush=True)
            continue
        f.write_text(body)
        print(f"  [tex] wrote {f} ({len(body.splitlines())} lines)", flush=True)
    if not _TEX:
        print("  [tex] NOTHING EMITTED -- no producer wrote a fragment, which is a defect, not "
              "an empty paper"
              + (f". {_registry.ROOT_HINT}" if _registry.root_unset() else ""), flush=True)
        problems += 1
    return problems


_C5_BLOCKS = (
    ("cell-line identity", "annotation", E2_DS, None),
    ("cell-type annotation, mouse", "annotation", E3_DS, None),
    ("cell-type annotation, fly", "annotation", "fly_antenna", "broad"),
    ("gene detection", "probes", "binary_expression", None),
    ("pathway activity", "probes", "module_activity", None),
    ("perturbation response, the CRISPR screen", "e7", None, None),
)
_C5_READERS = (("linear", "linear"), ("knn_cosine", "$k$NN"), ("mlp", "MLP"))


def _annotation_report(path):
    """One annotation report, refusing a producer this file does not recognise. `None` if absent.

    `collect` has refused an unknown producer in a PROBES report since the rename, and says in
    its own message why: an unrecognised producer is treated as a baseline and so bypasses the
    coverage refusal. The annotation readers had no such gate, and their failure is quieter.
    `_c5_annotation` keys its output off the artifact KEY rather than the stamped producer, so a
    pre-rename entry parses to a name no panel contains, is dropped, and "the best count FM"
    becomes a maximum over a subset -- a number that is wrong without being missing. Same
    refusal, same reason, applied wherever an annotation report is opened.
    """
    if not path.is_file():
        return None
    try:
        rep = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for key, e in (rep.get("artifacts") or {}).items():
        prod = e.get("producer")
        if prod not in KNOWN_PRODUCERS:
            raise SystemExit(
                f"{path.name}: {key} names producer {prod!r}, which is neither a model producer "
                f"nor a baseline. Known: {sorted(KNOWN_PRODUCERS)}. An annotation entry this file "
                f"does not recognise is DROPPED rather than refused, so it would silently shrink "
                f"the panel that 'the best count FM' maximises over. If this report predates the "
                f"rename, re-score it; do not add the old name here."
            )
    return rep


def _c5_annotation(P, ds, grain):
    """`{producer: {reader: macro}}` from one annotation report, or None if it has not landed."""
    name = f"{ds}.allmodels.annotation" + (f"_{grain}" if grain else "") + ".json"
    rep = _annotation_report(P / "eval_artifacts" / name)
    if rep is None:
        return None
    out = {}
    for key, e in (rep.get("artifacts") or {}).items():
        prod = key[len(ds) + 1:].removesuffix(".allreads")
        out[prod] = {r: (e.get(r) or {}).get("macro") for r, _ in _C5_READERS}
    return out


def _c5_probes(P, family):
    """`{producer: value}` for one probes family on `registry.REPORT_PANEL`, meaned over datasets."""
    want = _registry.REPORT_PANELS.get(family)
    if want is None:
        raise KeyError(
            f"{family!r} has no entry in registry.REPORT_PANELS, so there is no declared answer to "
            f"'which keys does this row report'. Declare it; do not average what happens to exist."
        )
    per, n_ds, n_cols = {}, 0, 0
    for f in sorted(P.glob("eval_artifacts/*.allmodels.probes.json")):
        try:
            rep = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        ds = rep.get("dataset")
        arts = rep.get("artifacts") or {}
        got = False
        for key, e in arts.items():
            prod = key[len(ds) + 1:].removesuffix(".allreads")
            panels = (e.get(family) or {})
            vals = [float(panels[k]["macro"]) for k in want
                    if isinstance(panels.get(k), dict) and panels[k].get("macro") is not None]
            if not vals:
                continue
            per.setdefault(prod, []).append(sum(vals) / len(vals))
            n_cols = max(n_cols, len(vals))
            got = True
        n_ds += 1 if got else 0
    return {k: sum(v) / len(v) for k, v in per.items()}, n_ds, n_cols


def c5_tex(P) -> str:
    """`tab:c5`'s body: ours, the hexamer table, the best count FM, per block."""
    flag = _registry.FLAGSHIP
    panel = [p for p in controls.PANEL]
    rows = []

    def cells(vals: dict, label_suffix=""):
        ours = vals.get(flag)
        bag = vals.get("kmer_bag")
        fms = [(vals.get(p), p) for p in panel if vals.get(p) is not None]
        best = max(fms)[0] if fms else None
        out = []
        for v in (ours, bag, best):
            out.append(f"${v:.4f}$" if v is not None else TEX_MISSING + "{c5}")
        return out

    for label, source, key, grain in _C5_BLOCKS:
        if source == "annotation":
            got = _c5_annotation(P, key, grain)
            for i, (reader, rname) in enumerate(_C5_READERS):
                head = label if i == 0 else ""
                if got is None:
                    c = [TEX_MISSING + "{c5}"] * 3
                else:
                    c = cells({p: (v or {}).get(reader) for p, v in got.items()})
                rows.append(f"{head} & {rname} & " + " & ".join(c) + r" \\")
        elif source == "probes":
            vals, n_ds, n_cols = _c5_probes(P, key)
            metric = "AUROC" if key == "binary_expression" else "Pearson $r$"
            head = f"{label}, {n_ds} datasets" if n_ds else label
            rows.append(f"{head} & {metric}, ${n_cols * n_ds}$ columns & "
                        + " & ".join(cells(vals)) + r" \\")
        else:
            f7 = P / "eval_reports" / f"e7_vs_control.{E7_DS}.json"
            if not f7.is_file():
                rows.append(f"{label} & macro AUROC & "
                            + " & ".join([TEX_MISSING + "{E7}"] * 3) + r" \\")
                continue
            r7 = json.loads(f7.read_text())["results"]
            vals = {k: v["macro"] for k, v in r7.items()}
            _r7full = json.loads(f7.read_text())
            _tg = _r7full["targets"]
            trio_keys = [flag, "kmer_bag", max(((vals.get(q), q) for q in panel
                                                if vals.get(q) is not None), default=(None, None))[1]]
            wins = {k: sum(1 for t in _tg
                           if k in r7 and all(_reports.per_target(_r7full, r7[k], t)
                                              >= _reports.per_target(_r7full, r7[o], t)
                                              for o in trio_keys if o in r7))
                    for k in trio_keys if k}
            trio = []
            for k in trio_keys:
                if k is None or k not in vals:
                    trio.append(TEX_MISSING + "{E7}")
                else:
                    trio.append(f"${vals[k]:.4f}$ (${wins[k]}$)")
            rows.append(f"{label} & macro AUROC & " + " & ".join(trio) + r" \\")
    return "\n".join(rows)


#: Every producer a report may name. A report is plain JSON with no stamp, so unlike an artifact it
#: cannot fail to load; an unknown producer name is refused here instead.
KNOWN_PRODUCERS = (
    frozenset(_artifact.MODEL_PRODUCERS) | frozenset(controls.BUILDERS) | frozenset(controls.PANEL)
)


_SPEC_SHORT = {"geneformer": "Genef.", "scgpt": "scGPT", "count_scanpy": r"\texttt{scanpy}",
               "uce": "UCE", "scvi_pretrained": "scVI", "genejepa": "GeneJEPA",
               "depth2": "floor"}


_E7_STORIES = (
    ("we are ahead where the perturbation changes transcript structure",
     (("MORF4L1", "exon exclusion"), ("SRPK1", "exon inclusion"))),
    ("every model is ahead where it changes the expression profile",
     (("ELOF1", "transcription rate"), ("CTNNB1", "transcription activation"))),
    ("no model detects a change in translation",
     (("PAIP2", "translation rate"), ("EIF4EBP1", "translation initiation"))),
)
#: tab:e7's leading rows. The chance level the NTC A/B splits measure is reported in the appendix
#: through \scrmEsevenNTC rather than as a table row.
_E7_OVERALL = (
    ("overall", "macro", True),
    ("control specificity", "macro_control_accuracy", True),
)


def _tex_rows(lines: list[str]) -> str:
    r"""Join generated rows, folding a lone `\midrule` onto the row that follows it."""
    out, pending = [], ""
    for ln in lines:
        if ln.strip() in (r"\midrule", r"\toprule", r"\bottomrule"):
            pending += ln.strip() + " "
            continue
        out.append(pending + ln)
        pending = ""
    assert not pending, (
        f"a horizontal rule with no row after it: {pending.strip()!r} ends the fragment. "
        f"The emitter appended a rule before a group that turned out empty. lines={lines!r}"
    )
    return "\n".join(out)


def _e7_hex_qualitative(rep: dict) -> str:
    """The main text's hexamer clause -- three facts, no numbers, and it FAILS if one stops holding."""
    res = rep.get("results") or {}
    bag = res.get("kmer_bag")
    counts = [k for k in (("count_scanpy",) + tuple(controls.PANEL)) if k in res]
    genes = [g for g, _f in _E7_STORIES[0][1]]
    if not bag or not counts or len(genes) != 2:
        return TEX_MISSING + "{hex}"
    margin = {}
    for t in rep.get("targets") or []:
        mine = _e7_target(rep, bag, t, "auroc")
        theirs = [v for v in (_e7_target(rep, res[k], t, "auroc") for k in counts) if v is not None]
        if mine is not None and theirs:
            margin[t] = mine - max(theirs)
    if not margin or not all(g in margin for g in genes):
        return TEX_MISSING + "{hex}"
    behind = sum(1 for v in margin.values() if v < 0)
    if behind * 2 <= len(margin):
        return TEX_MISSING + "{hex}"
    if not all(margin[g] > 0 for g in genes):
        return TEX_MISSING + "{hex}"
    top = max(margin, key=lambda t: margin[t])
    clause = ("A bag of hexamer frequencies taken from the same reads, with nothing trained, is "
              "behind the count-based models on most of the panel, yet it is ahead of every one of "
              "them on both of these targets")
    if top == genes[0]:
        clause += f" -- on \\emph{{{genes[0]}}} by more than on any other perturbation in the screen"
    return clause


def _e7_seed_spread(rep: dict) -> str:
    """The reported model's seed spread: the macro's range, then story 1's targets."""
    res = rep.get("results") or {}
    base = _registry.seed_base(_registry.FLAGSHIP)
    seeds = sorted(k for k in res if _registry.seed_base(k) == base)
    if len(seeds) < 2:
        return TEX_MISSING + "{seeds}"
    macros = [res[k]["macro"] for k in seeds if res[k].get("macro") is not None]
    if len(macros) < 2:
        # Two seed ROWS is not two seed MACROS; a spread over one value is not a spread.
        return TEX_MISSING + "{seeds}"
    out = [f"${len(seeds)}$ seeds of the reported model spread the macro by "
           f"${max(macros) - min(macros):.3f}$"]
    for gene, _function in _E7_STORIES[0][1]:
        vals = [v for v in (_e7_target(rep, res[k], gene, "auroc") for k in seeds) if v is not None]
        if len(vals) > 1:
            out.append(f"\\emph{{{gene}}} by ${max(vals) - min(vals):.3f}$")
    return ", ".join(out[:-1]) + " and " + out[-1] if len(out) > 1 else out[0]


def _e7_hex_story_one(rep: dict) -> str:
    """The hexamer table against the reported model on story 1's two targets, as a clause."""
    res = rep.get("results") or {}
    ours, bag = res.get(_registry.FLAGSHIP), res.get("kmer_bag")
    genes = [g for g, _f in _E7_STORIES[0][1]]
    if not ours or not bag:
        return TEX_MISSING + "{hex}"
    parts = []
    for g in genes:
        a, b = _e7_target(rep, bag, g, "auroc"), _e7_target(rep, ours, g, "auroc")
        if a is None or b is None:
            return TEX_MISSING + "{hex}"
        parts.append(f"${a:.3f}$ on \\emph{{{g}}} against ${b:.3f}$")
    return ", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0]


def _e7_report(P) -> dict:
    """The one perturbation report every E7 fragment reads. Named once, like `E7_DS`."""
    f = P / "eval_reports" / f"e7_vs_control.{E7_DS}.json"
    return json.loads(f.read_text()) if f.is_file() else {}


def _e7_columns() -> list[tuple[str, str]]:
    """`tab:e7`'s model columns: the reported arm, the count pipeline, then `controls.PANEL`."""
    cols = [(_registry.FLAGSHIP, OURS_LABEL), ("count_scanpy", _SPEC_SHORT["count_scanpy"])]
    return cols + [(k, _SPEC_SHORT[k]) for k in controls.PANEL]


def _e7_row(vals: list[float | None], *, bold: bool, digits: int = 3) -> str:
    """One row's model cells: the leading score bold, an unproducible cell marked, never blank."""
    top = max((v for v in vals if v is not None), default=None)
    out = []
    for v in vals:
        if v is None:
            out.append(TEX_MISSING + "{e7}")
        elif bold and top is not None and v == top:
            out.append(f"$\\mathbf{{{v:.{digits}f}}}$")
        else:
            out.append(f"${v:.{digits}f}$")
    return " & ".join(out)


def e7_tex(P) -> tuple[str, str]:
    """`tab:e7` -- the perturbation table -- header row, three panel rows and the three stories."""
    rep = _e7_report(P)
    res = rep.get("results") or {}
    cols = _e7_columns()
    dash = r"\multicolumn{1}{c}{---}"
    lines = [" & ".join(["target", "gene function"] + [lab for _k, lab in cols]) + r" \\",
             r"\midrule"]
    for label, field, bold in _E7_OVERALL:
        vals = [(res.get(k) or {}).get(field) for k, _lab in cols]
        lines.append(f"{label} & {dash} & " + _e7_row(vals, bold=bold) + r" \\")
    for _story, targets in _E7_STORIES:
        lines.append(r"\midrule")
        for gene, function in targets:
            vals = [_e7_target(rep, res.get(k) or {}, gene, "auroc") for k, _lab in cols]
            lines.append(f"\\emph{{{gene}}} & {function} & " + _e7_row(vals, bold=True) + r" \\")
    macros = "\n".join([
        "\\newcommand{\\" + TEX_NS + "EsevenTargets}{" + str(len(rep.get("targets") or []) or "??") + "}",
        "\\newcommand{\\" + TEX_NS + "EsevenNTC}{" + f"{rep.get('n_ntc', 0):,}".replace(",", "{,}") + "}",
        "\\newcommand{\\" + TEX_NS + "EsevenLeads}{" + _e7_leads(rep) + "}",
        "\\newcommand{\\" + TEX_NS + "HexStoryOne}{" + _e7_hex_story_one(rep) + "}",
        "\\newcommand{\\" + TEX_NS + "EsevenSeedSpread}{" + _e7_seed_spread(rep) + "}",
        "\\newcommand{\\" + TEX_NS + "HexQualitative}{" + _e7_hex_qualitative(rep) + "}",
        "\\newcommand{\\" + TEX_NS + "EsevenHeldControls}{" + str(
            next((int(v["fp_control_cells_tested"]) for v in (res or {}).values()
                  if v.get("fp_control_cells_tested")), "??")) + "}",
    ])
    return _tex_rows(lines), macros


def _e7_target(rep: dict, entry: dict, gene: str, field: str):
    """One (model, target) number, from the by-name dict this contrast computes."""
    by = entry.get("per_target_by_name")
    if isinstance(by, dict):
        rec = by.get(gene) or {}
        if field.startswith("fp:"):
            return (rec.get("false_positive") or {}).get(field[3:])
        return rec.get(field)
    if field != "auroc":
        return None
    tg, pt = rep.get("targets") or [], entry.get("per_target") or []
    return pt[tg.index(gene)] if gene in tg and len(pt) == len(tg) else None


def _e7_bold_target(gene: str) -> str:
    """The target's name, bolded iff `tab:e7` shows it -- which both appendix captions promise."""
    shown = {g for _s, ts in _E7_STORIES for g, _f in ts}
    body = f"\\emph{{{gene}}}"
    return f"\\textbf{{{body}}}" if gene in shown else body


def e7full_tex(P) -> str:
    """`tab:e7full` -- every model at every target, no maximum taken anywhere."""
    rep = _e7_report(P)
    res, cols = rep.get("results") or {}, _e7_columns()
    lines = [" & ".join(["target", "$n_+$"] + [lab for _k, lab in cols]) + r" \\", r"\midrule"]
    for gene in sorted(rep.get("targets") or []):
        npos = next((v for v in (_e7_target(rep, res.get(k) or {}, gene, "n_pos")
                                 for k, _l in cols) if v is not None), None)
        vals = [_e7_target(rep, res.get(k) or {}, gene, "auroc") for k, _lab in cols]
        n = TEX_MISSING + "{e7}" if npos is None else f"${int(npos)}$"
        lines.append(f"{_e7_bold_target(gene)} & {n} & " + _e7_row(vals, bold=True) + r" \\")
    lines.append(r"\midrule")
    vals = [(res.get(k) or {}).get("macro") for k, _lab in cols]
    lines.append(r"macro & \multicolumn{1}{c}{---} & " + _e7_row(vals, bold=True) + r" \\")
    return _tex_rows(lines)


def _e7_leads(rep: dict) -> str:
    """The sentence naming every target where the reported model beats EVERY count model."""
    res = rep.get("results") or {}
    ours = res.get(_registry.FLAGSHIP) or {}
    counts = [k for k in (("count_scanpy",) + tuple(controls.PANEL)) if k in res]
    if not ours or not counts:
        return TEX_MISSING + "{e7leads}"
    won = []
    for gene in rep.get("targets") or []:
        mine = _e7_target(rep, ours, gene, "auroc")
        theirs = [_e7_target(rep, res[k], gene, "auroc") for k in counts]
        theirs = [v for v in theirs if v is not None]
        if mine is None or not theirs:
            continue
        if mine > max(theirs):
            won.append((mine - max(theirs), gene))
    if not won:
        return "no target"
    won.sort(reverse=True)
    n = len(rep.get("targets") or [])
    items = [f"\\emph{{{g}}} by ${d:.3f}$" for d, g in won]
    body = ", ".join(items[:-1]) + (" and " + items[-1] if len(items) > 1 else items[-1])
    return f"{len(won)} of the ${n}$ targets, " + body


def specialize_tex(P) -> tuple[str, str]:
    """`tab:specialize`'s two rows and its caption's macros."""
    ds = _registry.spec_dataset()
    panel = _registry.SPEC_PANEL
    f = P / "eval_artifacts" / f"{ds}.allmodels.probes.json"
    rep = json.loads(f.read_text()) if f.is_file() else {"artifacts": {}}
    vals, genes = {}, None
    for key, e in (rep.get("artifacts") or {}).items():
        prod = key[len(ds) + 1:].removesuffix(".allreads")
        cell = ((e.get("binary_expression") or {}).get(panel) or {})
        if cell.get("macro") is not None:
            vals[prod] = float(cell["macro"])
            genes = genes or cell.get("n_scored")
    flag = _registry.FLAGSHIP
    # THE ARM THE CAPTION DESCRIBES, and the deltas it actually carries.
    #
    # The arm comes from the DECLARATION, never from `max` over the scores. The in-domain arm
    # differs from the flagship in its seed as well as its corpus, so every delta beyond the
    # corpus is published in `\scrmSpecDelta`.
    arms = _registry.IN_DOMAIN[ds]
    if len(arms) != 1:
        raise KeyError(
            f"the specialisation table describes ONE arm; IN_DOMAIN[{ds!r}] declares {list(arms)}. "
            f"Name which one the caption is about before registering a second."
        )
    spec = arms[0]
    _delta = dict(_registry.ABLATION_DELTAS.get(spec) or {})
    _extra = sorted(set(_delta) - {"corpora"})
    if "corpora" not in _delta:
        # An empty fragment would render as an empty table in the paper, so this refuses out loud.
        raise SystemExit(
            f"[specialize] REFUSED: {spec} is declared in-domain on {ds}, but its ABLATION_DELTAS "
            f"entry {sorted(_delta)} does not change the corpus, so this table cannot be captioned "
            f"as a corpus effect at all. Fix the declaration or retire the table."
        )
    short = dict(_SPEC_SHORT)
    short[flag] = OURS_LABEL
    short[spec] = rf"{OURS_LABEL}$^\dagger$"
    have = [(v, k) for k, v in vals.items() if k in short]
    have.sort(reverse=True)
    if not have:
        body = (" & " + " & ".join([TEX_MISSING + "{spec}"] * len(short)) + r" \\" + "\n"
                + r"\midrule AUROC & "
                + " & ".join([TEX_MISSING + "{spec}"] * len(short)) + r" \\")
    else:
        top = have[0][0]
        head = " & " + " & ".join(short[k] for _v, k in have) + r" \\"
        cells = " & ".join((f"$\\mathbf{{{v:.3f}}}$" if v == top else f"${v:.3f}$")
                           for v, _k in have)
        body = head + "\n" + r"\midrule AUROC & " + cells + r" \\"
    rec = _registry.recipe(spec)
    flagrec = _registry.recipe(flag)
    _changes = ", ".join(f"{k} ({flagrec.get(k)} $\\to$ {rec.get(k)})" for k in _extra)
    spec_delta = f"the corpus AND {_changes}" if _extra else "the corpus and nothing else"
    print(f"  [specialize] {spec} differs from {flag} in: {spec_delta}. The caption must say so; "
          f"it is published as \\{TEX_NS}SpecDelta.", flush=True)
    # The panel SELECTS `n_genes` and SCORES `n_scored` of them; the caption states both, so both
    # are derived.
    selected = ((rep.get("panels") or {}).get(panel) or {}).get("n_genes")
    #: The genes the panel selects but cannot score: detected in every cell of a fold or in none,
    #: which leaves AUROC undefined. Derived as the difference of the two counts it sits between.
    undefined = (selected - genes) if (selected is not None and genes is not None) else None
    #: The two AUROCs the main text quotes side by side, and the gain between them.
    _spec_v, _flag_v = vals.get(spec), vals.get(flag)
    _gain = (_spec_v - _flag_v) if (_spec_v is not None and _flag_v is not None) else None
    #: The reseed span ON THIS DATASET AND PANEL, which is what the main text compares the gain
    #: against. registry.SEED_FLOOR is a span over datasets and is NOT this number.
    _seeds = [vals[r] for r in _registry.SEED_FLOOR_SET if r in vals]
    _span = (max(_seeds) - min(_seeds)) if len(_seeds) > 1 else None
    macros = "\n".join([
        "\\newcommand{\\" + TEX_NS + "SpecGenes}{" + (str(genes) if genes else "??") + "}",
        "\\newcommand{\\" + TEX_NS + "SpecPanelGenes}{" + (str(selected) if selected else "??") + "}",
        "\\newcommand{\\" + TEX_NS + "SpecGenesUndefined}{"
        + (str(undefined) if undefined is not None else "??") + "}",
        "\\newcommand{\\" + TEX_NS + "SpecCells}{" + f"{rec['train_cells']:,}".replace(",", "{,}") + "}",
        "\\newcommand{\\" + TEX_NS + "FlagCells}{" + f"{flagrec['train_cells']:,}".replace(",", "{,}") + "}",
        "\\newcommand{\\" + TEX_NS + "SpecAUROC}{"
        + (f"${_spec_v:.3f}$" if _spec_v is not None else TEX_MISSING + "{spec}") + "}",
        "\\newcommand{\\" + TEX_NS + "SpecFlagAUROC}{"
        + (f"${_flag_v:.3f}$" if _flag_v is not None else TEX_MISSING + "{spec}") + "}",
        "\\newcommand{\\" + TEX_NS + "SpecGain}{"
        + (f"${_gain:.3f}$" if _gain is not None else TEX_MISSING + "{spec}") + "}",
        "\\newcommand{\\" + TEX_NS + "SpecSeedSpan}{"
        + (f"${_span:.3f}$" if _span is not None else TEX_MISSING + "{spec}") + "}",
        "\\newcommand{\\" + TEX_NS + "SpecSeedN}{" + str(len(_seeds)) + "}",
        #: How many times the reseed span the in-domain gain is.
        "\\newcommand{\\" + TEX_NS + "SpecGainVsSeed}{"
        + (f"${_gain / _span:.1f}$" if (_gain is not None and _span) else TEX_MISSING + "{spec}")
        + "}",
    ])
    return body, macros



def corpora_tex() -> str:
    """app:corpora's mouse-vs-other claim: how much easier the mouse datasets are.

    The appendix states two deltas on the reported detection panel -- how much higher the
    depth-only floor and the count pipeline score on the mouse datasets than on the rest. Both
    were literals, and both had drifted (0.024 -> 0.041 and 0.017 -> 0.012). The split is by
    `registry.species()`, so adding a dataset moves the claim with the data rather than leaving
    a sentence behind.
    """
    panel = _registry.REPORT_PANEL
    sp = _registry.species()
    mouse = [d for d in _registry.datasets("eval") if sp.get(d) == "mouse"]
    other = [d for d in _registry.datasets("eval") if sp.get(d) != "mouse"]

    def mean_on(model, dss):
        vs = []
        for ds in dss:
            f = P / f"eval_artifacts/{ds}.allmodels.probes.json"
            if not f.is_file():
                continue
            for _k, e in (json.load(f.open()).get("artifacts") or {}).items():
                if e.get("producer") == model:
                    v = ((e.get("binary_expression") or {}).get(panel) or {}).get("macro")
                    if v is not None:
                        vs.append(v)
        return (sum(vs) / len(vs)) if vs else None

    out = ["% GENERATED by measurements/results_page.py --write-tex. Do not edit by hand.",
           f"% Panel {panel}. Mouse: {', '.join(mouse)}. Other: {', '.join(other)}."]
    for model, tag in (("depth2", "MouseDepthDelta"), ("count_scanpy", "MouseCountDelta")):
        a, b = mean_on(model, mouse), mean_on(model, other)
        val = f"${a - b:.3f}$" if (a is not None and b is not None) else TEX_MISSING + "{corpora}"
        out.append("\\newcommand{\\" + TEX_NS + tag + "}{" + val + "}")
    out.append("\\newcommand{\\" + TEX_NS + "MouseN}{" + str(len(mouse)) + "}")
    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-tex", type=Path, metavar="DIR",
                    help="write the paper's generated table fragments into DIR "
                         "(e.g. $SCRM_REPO-paper/generated)")
    ap.add_argument("--check-tex", type=Path, metavar="DIR",
                    help="compare DIR's fragments against a fresh derivation; non-zero on drift")
    args = ap.parse_args()
    if args.write_tex and args.check_tex:
        ap.error("--write-tex and --check-tex are the same derivation with two dispositions; "
                 "pass one")
    if not (args.write_tex or args.check_tex):
        ap.error("pass --write-tex DIR or --check-tex DIR; this generator emits the paper's "
                 "fragments and nothing else")
    if _registry.root_unset():
        print(f"[tex] NO DATA ROOT. {_registry.ROOT_HINT}", flush=True)
        print("[tex] every fragment would be derived from nothing READ, not from nothing "
              "measured, so none of them is publishable.", flush=True)
    fam, e2, _mlm, _CI = collect()

    # E3's linear macro-F1 per model, read from the combined annotation report. The coverage gate
    # applies here as everywhere: an arm below its dataset's own requirement is refused rather than
    # averaged in.
    e3_linear = {}
    # The refusal is PRINTED, like the three other coverage sites (see `refused`/`skipped` above).
    _e3_refused: list[str] = []
    _e3f = P / f"eval_artifacts/{E3_DS}.allmodels.annotation.json"
    if _e3f.exists():
        _lab = dict(ARMS)
        for _k, _e in (json.loads(_e3f.read_text()).get("artifacts") or {}).items():
            _run = _k.removeprefix(f"{E3_DS}.")
            for _s in (".allreads", ".full"):
                _run = _run.removesuffix(_s)
            if _e.get("producer") in _artifact.MODEL_PRODUCERS:
                if _e.get("w_eval", 0) < full_coverage(E3_DS):
                    _e3_refused.append(f"{E3_DS}/{_k} E3 (w_eval={_e.get('w_eval')})")
                    continue
                _name = _lab.get(_run, _run)
            else:
                if _e.get("producer") in DROPPED_PRODUCERS:
                    continue
                _name = NICE.get(_e.get("producer"), _e.get("producer"))
            _m = (_e.get("linear") or {}).get("macro")
            if _m is not None:
                e3_linear[_name] = _m

    if _e3_refused:
        print(f"  [E3] {len(_e3_refused)} model(s) below full coverage ({full_coverage(E3_DS)} "
              f"windows), excluded from the annotation column:", flush=True)
        for _r in sorted(_e3_refused):
            print(f"        {_r}", flush=True)

    _e8_body, _e8_macros = contrast_table(P, fam, e2, e3_linear)
    emit_tex("e8_body", _e8_body)
    emit_tex("e8_n", _e8_macros)
    emit_tex("corpora_n", corpora_tex())
    emit_tex("c5_body", c5_tex(P))
    _spec_body, _spec_macros = specialize_tex(P)
    emit_tex("specialize_body", _spec_body)
    emit_tex("specialize_n", _spec_macros)
    _e7_body, _e7_macros = e7_tex(P)
    emit_tex("e7_body", _e7_body)
    emit_tex("e7_n", _e7_macros)
    emit_tex("e7full_body", e7full_tex(P))

    tex_dir = args.write_tex or args.check_tex
    problems = write_tex(tex_dir, check=bool(args.check_tex))
    if problems:
        verb = "differ from a fresh derivation or cannot be produced" if args.check_tex else \
               "were written with cells no report can produce"
        print(f"[tex] {problems} problem(s): fragments {verb}. The paper's numbers are only as "
              f"current as these files, so this is a failure and not a warning.")
        raise SystemExit(1)
    print(f"[tex] {len(_TEX)} fragment(s) "
          f"{'verified against' if args.check_tex else 'written to'} {tex_dir}")


if __name__ == "__main__":
    main()
