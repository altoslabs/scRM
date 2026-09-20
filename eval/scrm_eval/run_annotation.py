"""E2 runner: score stamped embedding artifacts against external cell labels (macro-F1)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from . import artifact, scoring
from .probes import annotation_probe
from .reduce import reduce_windows


def _write(out: Path, report: dict) -> None:
    """tmp + os.replace, the same discipline run_probes uses."""
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2))
    os.replace(tmp, out)


def _probe_entry(x, y) -> dict:
    """One artifact's readers, as plain JSON-able data. Module level so loky can pickle it."""
    return {name: dataclasses.asdict(res) | {"per_target": [round(float(v), 4) for v in res.per_target]}
            for name, res in annotation_probe(x, y, n_jobs=1).items()}


def _tagged(tag: str, x, y) -> tuple[str, dict]:
    """`_probe_entry` with its artifact's tag carried through. Module level so loky can pickle it."""
    return tag, _probe_entry(x, y)


def _reduce_rule(emb, reduction):
    """`reduction` if the caller forced one, else the rule this artifact's stamp implies."""
    from .reduce import rule_for

    return rule_for(emb.stamp, override=reduction)


def _reuse_key(emb: artifact.Embeddings, reduction: str | None, labels: dict[str, str]) -> str:
    """This artifact's row identity: the SAME contract run_probes uses. See `scoring.py`."""
    return scoring.reuse_key(
        emb, reduction_rule=_reduce_rule(emb, reduction),
        context={"labels": scoring.digest(sorted(labels.items()))})


def annotation_report(arts: dict[str, artifact.Embeddings], labels: dict[str, str], reduction: str,
                      out: Path | None = None) -> dict:
    """Score every artifact in `arts` against `labels`, writing `out` as each one finishes."""
    first = next(iter(arts.values()))
    for other in arts.values():
        artifact.same_population(first.stamp, other.stamp)
    scoring.refuse_mixed_precision(arts)
    applied = {p_: _reduce_rule(e, reduction)
               for p_, e in sorted({e.stamp.producer: e for e in arts.values()}.items())}
    report: dict = {"reduction": reduction, "reduction_applied": applied,
                    "n_labels": len(labels), "artifacts": {}}
    done: dict = {}
    if out is not None and out.exists():
        try:
            done = json.loads(out.read_text()).get("artifacts") or {}
        except (OSError, json.JSONDecodeError) as e:
            print(f"  [resume] {out.name} unreadable ({e}); starting clean", flush=True)
    t0 = time.monotonic()
    todo = []
    for tag, emb in arts.items():
        keep = done.get(tag)
        if keep and keep.get("_reuse_key") == _reuse_key(emb, reduction, labels):
            report["artifacts"][tag] = keep
            print(f"  {tag} reused from disk "
                  f"(linear={round((keep.get('linear') or {}).get('macro', float('nan')), 4)})",
                  flush=True)
            continue
        todo.append((tag, emb))
    if report["artifacts"] and out is not None:
        _write(out, report)

    n_jobs = int(os.environ.get("SCRM_ANNOT_JOBS") or os.environ.get("SLURM_CPUS_PER_TASK") or 1)
    n_jobs = max(1, min(len(todo), n_jobs))
    todo.sort(key=lambda te: -int(te[1].matrix.shape[0]) * int(te[1].matrix.shape[1]))
    by_tag = dict(todo)
    print(f"  {len(todo)} artifacts to score, {n_jobs} at a time", flush=True)

    def _record(tag: str, entry: dict) -> None:
        emb = by_tag[tag]
        report["artifacts"][tag] = {
            "producer": emb.stamp.producer,
            "w_eval": emb.stamp.w_eval,
            "forward_precision": emb.stamp.forward_precision or "fp32+eager",
            "_reuse_key": _reuse_key(emb, reduction, labels),
            **entry,
        }
        lin = (entry.get("linear") or {}).get("macro")
        print(f"  [{len(report['artifacts'])}/{len(arts)}] {tag} "
              f"linear={lin if lin is None else round(lin, 4)} "
              f"({time.monotonic() - t0:.0f}s elapsed)", flush=True)
        if out is not None:
            report["artifacts"] = {t: report["artifacts"][t] for t in arts if t in report["artifacts"]}
            _write(out, report)

    def _jobs():
        """Reduce lazily, one artifact per dispatch."""
        for tag, emb in todo:
            cells, x = reduce_windows(emb.cell_ids, emb.matrix, _reduce_rule(emb, reduction))
            y = np.asarray([labels.get(str(c).split("-")[0], "") for c in cells])
            yield joblib.delayed(_tagged)(tag, x, y)

    with threadpool_limits(limits=1):
        if n_jobs > 1:
            with joblib.parallel_backend("loky", n_jobs=n_jobs, inner_max_num_threads=1):
                for tag, entry in joblib.Parallel(return_as="generator_unordered")(_jobs()):
                    _record(tag, entry)
        else:
            for func, a, kw in _jobs():
                _record(*func(*a, **kw))
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", nargs="+", required=True, type=Path)
    p.add_argument("--labels", required=True, type=Path, help="JSON: bare barcode -> class label")
    p.add_argument("--reduction", default=None,
                   help="force one rule on every artifact; omit so each is read by "
                        "reduce.rule_for(stamp)")
    p.add_argument("-o", "--out", type=Path, required=True)
    args = p.parse_args(argv)

    labels = {k.split("-")[0]: v for k, v in json.loads(args.labels.read_text()).items()}
    arts = {path.name.removesuffix(".arrow"): artifact.read(path) for path in args.artifacts}
    report = annotation_report(arts, labels, args.reduction, out=args.out)
    report["labels_file"] = str(args.labels)
    _write(args.out, report)
    print(f"annotation report -> {args.out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
