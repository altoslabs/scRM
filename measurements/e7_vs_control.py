"""E7: is a cell perturbed? Each CRISPR target against the screen's NON-TARGETING controls."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO / "target" / "py"))

from scrm_eval import controls, artifact, reduce as _reduce, registry  # noqa: E402

P = registry.PROJ
DS = "crispr_a549_5k"

BASELINES = tuple(controls.BUILT_PRODUCERS) + tuple(controls.PANEL)

PRESUMED_ESSENTIAL = ("PPIB", "ELOF1", "RAB1A", "SET", "PAIP2", "EIF4EBP1",
                      "SNRNP70", "SNRNP40", "TPR", "RBBP4")
NEGATIVE_CONTROL = "Non-Targeting"


def discovered_arms(min_w: int | None = None) -> list[str]:
    """Every scrm arm with a full-coverage artifact for this dataset."""
    from scrm_eval import registry as R

    return list(R.arms_with_full_coverage(DS))


def labels() -> tuple[dict[str, set[str]], list[str]]:
    """cell -> the targets its guides hit (all guides), and the 26 effective targets to score."""
    raw = P / "raw" / DS
    eff = set(json.loads((raw / "effective_targets.json").read_text()))
    cpp = json.loads((raw / "crispr_analysis" / "cells_per_protospacer.json").read_text())
    fr = [ln.split(",") for ln in (raw / "crispr_analysis" / "feature_reference.csv").read_text().splitlines()]
    gi, ti = fr[0].index("id"), fr[0].index("target_gene_name")
    g2t = {r[gi]: r[ti] for r in fr[1:] if len(r) > max(gi, ti)}
    cell2t: dict[str, set[str]] = {}
    for guide, cs in cpp.items():
        for c in cs:
            cell2t.setdefault(str(c).split("-")[0], set()).add(g2t.get(guide, guide))
    return cell2t, sorted(eff)


def matrix(name: str):
    """The reduced cell matrix for one representation, by name."""
    if "." in name:
        run, stem = name.split(".", 1)
        p = P / "eval_ckptsel" / f"{run}.{stem}" / f"{DS}.{run}.allreads.arrow"
        if not p.exists():
            return None
        e = artifact.read(p)
        cells, x = _reduce.reduce_windows(e.cell_ids, e.matrix, _reduce.rule_for(e.stamp))
        return [str(c).split("-")[0] for c in cells], x
    p = P / "eval_runs" / name / f"{DS}.{name}.allreads.arrow"
    if not p.exists():
        p = P / "eval_artifacts" / f"{DS}.{name}.arrow"
    if not p.exists():
        return None
    e = artifact.read(p)
    cells, x = _reduce.reduce_windows(e.cell_ids, e.matrix, _reduce.rule_for(e.stamp))
    return [str(c).split("-")[0] for c in cells], x


READERS = ("ridge", "mlp", "mlp_ovr")

READER = "ridge"


def _oof_scores(x, y, seeds: int = 3, folds: int = 5, reader: str = "ridge"):
    """Out-of-fold scores for a binary label, averaged over `seeds` fold partitions."""
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from scrm_eval.probes import _annotation_reader, _folds, _ridge

    per_seed = []
    for seed in range(seeds):
        pred = np.full(len(y), np.nan)
        for tr, te in _folds(len(y), folds, seed):
            if y[tr].sum() == 0 or y[tr].sum() == len(tr):
                continue
            if reader == "mlp":
                m = _annotation_reader("mlp", x.shape[1], knn_k=1)
                m.fit(x[tr], y[tr].astype(int))
                pred[te] = m.predict_proba(x[te])[:, 1]
            else:
                m = make_pipeline(StandardScaler(), _ridge())
                m.fit(x[tr], y[tr].astype(np.float64))
                pred[te] = m.predict(x[te])
        per_seed.append(pred)
    return per_seed


def _mean_key(ok, key):
    vals = [v["false_positive"][key] for v in ok
            if isinstance(v.get("false_positive"), dict) and v["false_positive"].get(key) is not None]
    return float(sum(vals) / len(vals)) if vals else None


def _mean_fpr(ok):
    vals = [v["false_positive"]["fpr"] for v in ok
            if isinstance(v.get("false_positive"), dict)]
    return float(sum(vals) / len(vals)) if vals else None


def _shared_false_positives(ok):
    """How concentrated the false positives are across targets."""
    import collections

    hits = collections.Counter()
    tested = 0
    for v in ok:
        fp = v.get("false_positive")
        if not isinstance(fp, dict):
            continue
        tested = max(tested, fp.get("n_control_tested") or 0)
        hits.update(fp.get("flagged_cells") or [])
    if not hits:
        return {"fp_max_targets_per_cell": None, "fp_cells_flagged": 0,
                "fp_control_cells_tested": tested}
    counts = sorted(hits.values(), reverse=True)
    n_t = sum(1 for v in ok if isinstance(v.get("false_positive"), dict))
    ref_max = None
    if n_t and tested:
        import numpy as np
        pbar = sum(v["false_positive"]["fpr"] for v in ok
                   if isinstance(v.get("false_positive"), dict)) / n_t
        rng = np.random.default_rng(0)
        sims = [int(rng.binomial(n_t, pbar, size=tested).max()) for _ in range(200)]
        ref_max = float(np.mean(sims))
    return {"fp_max_targets_per_cell": counts[0],
            "fp_cells_flagged": len(hits),
            "fp_control_cells_tested": tested,
            "fp_n_targets": n_t,
            "fp_binomial_expected_max": ref_max,
            "fp_excess_over_binomial": (counts[0] - ref_max) if ref_max is not None else None,
            "fp_mean_targets_per_flagged_cell": float(sum(counts) / len(counts))}


def false_positive_rate(x, pos, ntc_idx, rng, tpr_at: float = 0.5):
    """How often does a target's detector cry wolf on cells that are PROVABLY negative?"""
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from scrm_eval.probes import _ridge

    perm = rng.permutation(ntc_idx)
    half = len(perm) // 2
    train_neg, test_neg = perm[:half], perm[half:]
    if len(test_neg) < 10 or len(pos) < 10:
        return None
    oof = _oof_scores(x[np.concatenate([pos, train_neg])],
                      np.concatenate([np.ones(len(pos), bool), np.zeros(len(train_neg), bool)]),
                      reader=READER)
    pos_scores = np.nanmean([p[: len(pos)] for p in oof], axis=0)
    pos_scores = pos_scores[~np.isnan(pos_scores)]
    if len(pos_scores) < 5:
        return None
    m = make_pipeline(StandardScaler(), _ridge())
    tr = np.concatenate([pos, train_neg])
    m.fit(x[tr], np.concatenate([np.ones(len(pos)), np.zeros(len(train_neg))]))
    held_neg = m.predict(x[test_neg])
    k = len(pos)
    pool = np.concatenate([pos_scores, held_neg])
    if len(pool) <= k:
        return None
    thr = float(np.sort(pool)[-k])
    tpr = float((pos_scores >= thr).mean())
    fpr = float((held_neg >= thr).mean())
    held = held_neg
    return {
            "control_accuracy": 1.0 - fpr,
            "fpr": fpr,
            "tpr": tpr,
            "threshold_rule": "top-k, k = number of true positives (prevalence-matched)",
            "n_positives": int(len(pos_scores)),
            "n_control_tested": int(len(test_neg)),
            "n_flagged": int((held_neg >= thr).sum()),
            "flagged_cells": [int(i) for i in np.asarray(test_neg)[held > thr]]}


def _fpr_one_thread(x, pos, ntc_idx, target: str, seed: int):
    """`false_positive_rate` under ONE BLAS thread and its OWN random stream."""
    import numpy as np
    from threadpoolctl import threadpool_limits

    r = np.random.default_rng(seed + 1 + int.from_bytes(target.encode()[:6], "little") % 10_000_000)
    with threadpool_limits(1):
        return false_positive_rate(x, pos, ntc_idx, r)


def target_vs_ntc(x, cell2t, cells, targets, ntc_cells, null_reps: int, seed: int = 0):
    """Per target: AUROC **and AUPRC** against the UNPERTURBED cells, plus a measured null at the
    same n and the same class balance.
    """
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    idx = {c: i for i, c in enumerate(cells)}
    ntc_idx = np.array([idx[c] for c in ntc_cells if c in idx])
    out = {}

    def _both(a_idx, b_idx):
        """(AUROC, AUPRC) averaged over the seed partitions, from ONE set of scores."""
        sel = np.concatenate([a_idx, b_idx])
        y = np.zeros(len(sel), dtype=bool)
        y[: len(a_idx)] = True
        au, ap = [], []
        for pred in _oof_scores(x[sel], y, reader=READER):
            ok = ~np.isnan(pred)
            if ok.sum() < 4 or len(set(y[ok])) < 2:
                continue
            au.append(roc_auc_score(y[ok], pred[ok]))
            ap.append(average_precision_score(y[ok], pred[ok]))
        return (float(np.mean(au)) if au else float("nan"),
                float(np.mean(ap)) if ap else float("nan"))

    def _one(t):
        pos = np.array([idx[c] for c in cells if t in cell2t.get(c, ())])
        n_t = len(pos)
        if n_t < 10 or len(ntc_idx) - n_t < 10:
            return t, {"n_pos": int(n_t), "skipped": "too few cells for a matched null"}
        from threadpoolctl import threadpool_limits
        with threadpool_limits(1):
            r = np.random.default_rng(seed + int.from_bytes(t.encode()[:6], "little") % 10_000_000)
            tau, tap = [], []
            a, b = _both(pos, ntc_idx)
            tau.append(a); tap.append(b)
            tau = np.asarray([v for v in tau if v == v]); tap = np.asarray([v for v in tap if v == v])
            auroc = float(tau.mean()) if len(tau) else float("nan")
            auprc = float(tap.mean()) if len(tap) else float("nan")
            base = n_t / (n_t + len(ntc_idx))
            nau, nap = [], []
            for _ in range(null_reps):
                perm = r.permutation(ntc_idx)
                a, b = _both(perm[:n_t], perm[n_t:])
                nau.append(a); nap.append(b)
            pau, pap = [], []
            for _ in range(null_reps):
                mix = np.concatenate([pos, r.permutation(ntc_idx)[: len(ntc_idx) - n_t]])
                sh = r.permutation(mix)
                a, b = _both(sh[:n_t], sh[n_t:])
                pau.append(a); pap.append(b)
            pau = np.asarray([v for v in pau if v == v]); pap = np.asarray([v for v in pap if v == v])
        nau = np.asarray([v for v in nau if v == v]); nap = np.asarray([v for v in nap if v == v])

        def _z(val, arr):
            if val != val or len(arr) < 2:
                return None
            sd = float(arr.std(ddof=1))
            return (val - float(arr.mean())) / sd if sd > 0 else None

        return t, {
            "n_pos": int(n_t), "n_neg": int(len(ntc_idx)), "base_rate": base,
            "auroc": auroc, "auprc": auprc,
            "n_scored_cells": int(n_t + len(ntc_idx)),
            "auroc_null_mean": float(nau.mean()) if len(nau) else None,
            "auroc_null_sd": float(nau.std(ddof=1)) if len(nau) > 1 else None,
            "auprc_null_mean": float(nap.mean()) if len(nap) else None,
            "auprc_null_sd": float(nap.std(ddof=1)) if len(nap) > 1 else None,
            "z_auroc": _z(auroc, nau), "z_auprc": _z(auprc, nap),
            "perm_auroc_mean": float(pau.mean()) if len(pau) else None,
            "perm_auroc_sd": float(pau.std(ddof=1)) if len(pau) > 1 else None,
            "perm_auprc_mean": float(pap.mean()) if len(pap) else None,
            "z_auroc_perm": _z(auroc, pau), "z_auprc_perm": _z(auprc, pap),
            "composition_artefact": (float(pau.mean()) - float(nau.mean()))
                                    if len(pau) and len(nau) else None,
            "null_reps": int(min(len(nau), len(nap))),
            "false_positive": _fpr_one_thread(x, pos, ntc_idx, t, seed),
        }

    import os

    from joblib import Parallel, delayed

    n_jobs = min(len(targets), len(os.sched_getaffinity(0)))
    out = dict(Parallel(n_jobs=n_jobs)(delayed(_one)(t) for t in targets))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reader", default="ridge", choices=READERS,
                    help="the estimator fitted inside each fold. `ridge` is the published reader; "
                         "`mlp` answers whether the guide signal is linearly separable at all")
    ap.add_argument("--models", nargs="+", default=None,
                    help="default: the seven baselines plus EVERY full-coverage scrm arm on disk")
    ap.add_argument("--null-reps", type=int, default=20,
                    help="A/B splits of the NTC cells used to measure chance at each target's n")
    ap.add_argument("--verify", action="store_true",
                    help="recompute and diff against the report on disk instead of writing it")
    ap.add_argument("-o", "--out", type=Path,
                    default=None)
    a = ap.parse_args()
    global READER
    READER = a.reader
    if a.out is None:
        a.out = P / "eval_reports" / f"e7_vs_control.{DS}.json"

    cell2t, targets = labels()
    models = a.models or (list(BASELINES) + discovered_arms())
    print(f"{len(models)} representations: {sum(1 for m in models if m.startswith('arm'))} scrm arms "
          f"+ {sum(1 for m in models if not m.startswith('arm'))} baselines\n", flush=True)
    rep = {"targets": targets, "results": {}, "negative_class": "ntc"}
    if True:
        import json as _json
        _t = _json.loads((P / "raw" / DS / "labels_target.json").read_text())
        ntc = [k.split("-")[0] for k, v in _t.items() if v == "Non-Targeting"]
        rep["n_ntc"] = len(ntc)
        rep["null_reps"] = a.null_reps
        print(f"negative class = {len(ntc)} Non-Targeting cells; "
              f"chance measured from {a.null_reps} A/B splits per target\n", flush=True)
    for m in models:
        got = matrix(m)
        if got is None:
            print(f"{m:26s} no artifact, skipped", flush=True)
            continue
        cells, x = got
        if True:
            res = target_vs_ntc(np.asarray(x, dtype=np.float32), cell2t, cells, targets, ntc,
                                a.null_reps)
            ok = [v for v in res.values() if "auroc" in v]
            def _mean(k):
                vals = [v[k] for v in ok if v.get(k) is not None and v[k] == v[k]]
                return float(np.mean(vals)) if vals else None
            zr = [v["z_auroc"] for v in ok if v.get("z_auroc") is not None]
            zp = [v["z_auprc"] for v in ok if v.get("z_auprc") is not None]
            _per = {t: res.get(t, {}).get("auroc") for t in targets}
            _ceil = [v for t, v in _per.items() if t in PRESUMED_ESSENTIAL and v is not None]
            _restp = [v for t, v in _per.items() if t not in PRESUMED_ESSENTIAL and v is not None]
            rep["results"][m] = {
                "per_target_by_name": res,
                "per_target": [res.get(t, {}).get("auroc") for t in targets],
                "macro": _mean("auroc"),
                "ceiling_essential": (sum(_ceil) / len(_ceil)) if _ceil else None,
                "ceiling_targets": [t for t in PRESUMED_ESSENTIAL if _per.get(t) is not None],
                "rest_of_panel": (sum(_restp) / len(_restp)) if _restp else None,
                "n_scored": len(ok),
                "macro_auroc": _mean("auroc"), "macro_auroc_null": _mean("auroc_null_mean"),
                "macro_auprc": _mean("auprc"), "macro_auprc_null": _mean("auprc_null_mean"),
                "mean_base_rate": _mean("base_rate"),
                "auroc_above_2sd": sum(1 for z in zr if z >= 2.0),
                "auprc_above_2sd": sum(1 for z in zp if z >= 2.0),
                "macro_perm_floor": _mean("perm_auroc_mean"),
                "composition_artefact": _mean("composition_artefact"),
                "auroc_above_2sd_perm": sum(
                    1 for v in ok if (v.get("z_auroc_perm") or -9) >= 2.0),
                "macro_fpr": _mean_fpr(ok),
                "macro_control_accuracy": (1.0 - _mean_fpr(ok)) if _mean_fpr(ok) is not None else None,
                "macro_tpr": _mean_key(ok, "tpr"),
                **_shared_false_positives(ok)}
            e = rep["results"][m]
            def _f(v):
                return "n/a" if v is None else f"{v:.4f}"
            print(f"{m:26s} AUROC {_f(e['macro_auroc'])}  floor(NTC) {_f(e['macro_auroc_null'])}  "
                  f"floor(perm) {_f(e['macro_perm_floor'])}  "
                  f"artefact {_f(e['composition_artefact'])}  "
                  f"detectable {e['auroc_above_2sd']}/{e['n_scored']} vs NTC, "
                  f"{e['auroc_above_2sd_perm']}/{e['n_scored']} vs perm  "
                  f"AUPRC {_f(e['macro_auprc'])}  "
                  f"perturbed-cell accuracy {_f(e.get('macro_tpr'))}  "
                  f"control accuracy {_f(e.get('macro_control_accuracy'))} "
                  f"(FPR {_f(e.get('macro_fpr'))}) "
                  f"(worst control cell: {e.get('fp_max_targets_per_cell')} targets vs "
                  f"{_f(e.get('fp_binomial_expected_max'))} expected by chance)",
                  flush=True)
            continue
        raise SystemExit(f"{m}: target_vs_ntc should have handled every model; "
                        f"this line is unreachable by construction")

    if a.verify:
        old = json.loads(a.out.read_text())
        assert old["targets"] == rep["targets"], "target list differs"
        worst = 0.0
        for m, v in rep["results"].items():
            if m not in old["results"]:
                print(f"  {m}: not in the published report")
                continue
            def _macro(rec):
                return rec.get("macro", rec.get("macro_auroc"))

            d = abs(_macro(v) - _macro(old["results"][m]))
            dp = max(abs(p - q) for p, q in zip(v["per_target"], old["results"][m]["per_target"]))
            worst = max(worst, d, dp)
            print(f"  {m:26s} d(macro)={d:.2e}  max d(per-target)={dp:.2e}")
        print(f"\nworst absolute difference: {worst:.3e} (tolerance 1e-4, BLAS summation order)")
        return 0 if worst < 1e-4 else 1

    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.name + ".tmp")
    tmp.write_text(json.dumps(rep, indent=2))
    tmp.replace(a.out)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
