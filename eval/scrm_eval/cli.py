"""scrm-eval: score stamped embedding artifacts into one report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import artifact, controls, metrics, reduce, tenx


REF_K = 10


def protocol_labels(mat: np.ndarray, *, k: int = REF_K, seed: int = 0) -> np.ndarray:
    """The one clustering protocol, applied to any representation."""
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(mat)


def evaluate(emb, dataset: str, *, rule: str, reference=None, depth=None, independent_labels=None) -> dict:
    """Every metric for one artifact under one window→cell rule."""
    cells, mat = reduce.reduce_windows(emb.cell_ids, emb.matrix, rule)
    row = {c: i for i, c in enumerate(cells)}
    out: dict = {
        "rule": rule,
        "n_cells": int(mat.shape[0]),
        "dim": int(mat.shape[1]),
        "effective_rank": metrics.effective_rank(mat),
    }

    if reference is not None:
        r_cells, r_mat = reference
        shared = [c for c in r_cells if c in row]
        if len(shared) > REF_K:
            ours = mat[[row[c] for c in shared]]
            theirs = r_mat[[i for i, c in enumerate(r_cells) if c in row]]

            ari = metrics.cluster_agreement(protocol_labels(ours), protocol_labels(theirs))
            out.update({f"ari_symmetric_{m}": v for m, v in ari.items()})

            out["cka_vs_count_scanpy"] = metrics.linear_cka(ours, theirs)
            out["neighbor_overlap_vs_count_scanpy"] = metrics.neighbor_overlap(ours, theirs)
            reg = metrics.pc_regression(ours, theirs)
            out["pc_r2_variance_weighted"] = reg["r2_variance_weighted"]
            out["pc_r2_per_pc"] = reg["r2_per_pc"]
            out["reference_dim"] = int(theirs.shape[1])

            if depth is not None:
                d_cells, d_mat = depth
                d_row = {c: i for i, c in enumerate(d_cells)}
                if all(c in d_row for c in shared):
                    nuis = d_mat[[d_row[c] for c in shared]]
                    out["pc_r2_depth_residualised"] = metrics.pc_regression(
                        ours, metrics.residualise(theirs, nuis)
                    )["r2_variance_weighted"]

    if independent_labels:
        keep, y = tenx.align(cells, independent_labels)
        if len(keep) > 2 and len(set(y.tolist())) > 1:
            sub = mat[keep]
            out["labelled_cells"] = int(len(keep))
            out["reference_clusters"] = int(len(set(y.tolist())))
            out["silhouette_within_dataset_only"] = metrics.silhouette(sub, y)
            out.update({f"purity_{k}": v for k, v in metrics.knn_label_purity(sub, y).items()})
    return out


def check_dataset(dataset: str, loaded) -> None:
    """`--dataset` must name a corpus the artifacts were actually computed on."""
    for e in loaded:
        covers = e.stamp.corpus_digests or {}
        if dataset not in covers:
            raise artifact.StampError(
                f"--dataset {dataset!r} names no corpus this artifact was computed on: its stamp "
                f"covers {sorted(covers)}. The 10x reference labels and every purity/silhouette "
                f"number would come from a different dataset than the embeddings."
            )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="scrm-eval", description=__doc__)
    ap.add_argument("--artifacts", nargs="*", default=[], help="stamped .arrow embeddings to evaluate")
    ap.add_argument("--dataset", required=True, help="which corpus dataset the artifacts cover")
    ap.add_argument("--rules", nargs="*", default=None,
                    help=f"window->cell rules to compare, e.g. --rules mean mean_standardised; "
                         f"omit so each artifact is read by reduce.rule_for(stamp). known {reduce.RULES}")
    ap.add_argument("--build-controls", action="store_true", help="compute and write the control artifacts")
    ap.add_argument("--out-dir", default="eval/artifacts", help="where controls are written")
    ap.add_argument("-o", "--out", default=None, help="report JSON path")
    args = ap.parse_args(argv)

    paths = [Path(p) for p in args.artifacts]
    if args.build_controls:
        out_dir = Path(args.out_dir)
        for producer in controls.BUILT_PRODUCERS:
            p = controls.build(args.dataset, producer, out_dir)
            print(f"  wrote {p.name}  sha256 {controls.digest_file(p)[:16]}…")
            paths.append(p)
    if not paths:
        ap.error("nothing to evaluate: pass --artifacts and/or --build-controls")

    loaded = [artifact.read(p) for p in paths]
    for other in loaded[1:]:
        artifact.same_population(loaded[0].stamp, other.stamp)
    check_dataset(args.dataset, loaded)

    reference = depth = None
    by_producer = {e.stamp.producer: e for e in loaded}
    if "count_scanpy" in by_producer:
        e = by_producer["count_scanpy"]
        reference = reduce.reduce_windows(e.cell_ids, e.matrix, reduce.rule_for(e.stamp))
    if "depth2" in by_producer:
        e = by_producer["depth2"]
        depth = reduce.reduce_windows(e.cell_ids, e.matrix, reduce.rule_for(e.stamp))
    try:
        independent = tenx.clusters(args.dataset)
    except FileNotFoundError:
        independent = None
        print(f"[eval] {args.dataset}: no CellRanger analysis bundle, so the independent-label "
              "metrics are SKIPPED for every model on this dataset", flush=True)

    report = {
        "dataset": args.dataset,
        "reference": "count_scanpy" if reference is not None else None,
        "reference_protocol": {"clustering": "kmeans", "k": REF_K, "seed": 0, "applied_to": "both sides"},
        "independent_labels": "10x graph clustering" if independent is not None else None,
        "depth_nuisance": "depth2" if depth is not None else None,
        "results": [],
        "sensitivity": [],
    }
    for e in loaded:
        for rule in (args.rules or [None]):
            row = evaluate(
                    e, args.dataset, rule=reduce.rule_for(e.stamp, override=rule),
                    reference=reference, depth=depth,
                    independent_labels=independent,
                )
            row["producer"] = e.stamp.producer
            row["stamp"] = {
                "code_rev": e.stamp.code_rev,
                "w_eval": e.stamp.w_eval,
                "eval_seed": e.stamp.eval_seed,
                "readout": e.stamp.readout,
                "checkpoint_sha256": e.stamp.checkpoint_sha256,
                "corpus_digests": e.stamp.corpus_digests,
            }
            report["results"].append(row)

    for i, a in enumerate(loaded):
        for b in loaded[i + 1 :]:
            ca, ma = reduce.reduce_windows(a.cell_ids, a.matrix, reduce.rule_for(a.stamp))
            cb, mb = reduce.reduce_windows(b.cell_ids, b.matrix, reduce.rule_for(b.stamp))
            ok, cos = metrics.sensitive(ma, mb)
            row = {"a": a.stamp.producer, "b": b.stamp.producer, "differ": bool(ok), "mean_cosine": cos}
            if cos is None:
                row["reason"] = (
                    f"different dimensionality ({ma.shape[1]} vs {mb.shape[1]})"
                    if ma.shape != mb.shape else
                    "no cell has a non-zero centred norm: both embeddings are constant across "
                    "cells, so no per-cell cosine exists"
                )
            report["sensitivity"].append(row)

    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"\nwrote {args.out}")
    _print_table(report)
    return 0


def _print_table(report: dict) -> None:
    """The console summary of a report that has already been written."""
    print(f"\n=== {report['dataset']} ===")
    head = (
        f"{'producer':18} {'rule':7} {'dim':>4} {'eff.rank':>9} {'silh*':>7} {'pur_adj':>8} "
        f"{'ARIsym':>7} {'CKA':>6} {'nbr':>6} {'PC R2':>7} {'PC R2-d':>8}"
    )
    print(head)
    for r in report["results"]:
        print(
            f"{r['producer']:18} {r['rule']:7} {r['dim']:>4} {r['effective_rank']:>9.2f} "
            f"{r.get('silhouette_within_dataset_only', float('nan')):>7.3f} "
            f"{r.get('purity_purity_adj', float('nan')):>8.3f} "
            f"{r.get('ari_symmetric_ari', float('nan')):>7.3f} "
            f"{r.get('cka_vs_count_scanpy', float('nan')):>6.3f} "
            f"{r.get('neighbor_overlap_vs_count_scanpy', float('nan')):>6.3f} "
            f"{r.get('pc_r2_variance_weighted', float('nan')):>7.3f} "
            f"{r.get('pc_r2_depth_residualised', float('nan')):>8.3f}"
        )
    print("\n* silhouette compares methods WITHIN this dataset only -- never across datasets.")
    print("  ARIsym: one k-means protocol at k=10 applied to BOTH the reference and the embedding.")
    print("  count_scanpy's own row is the CEILING, not a bar -- it is the reference scoring itself.")
    print("  Leiden-vs-graphclust ARI is not reported.")
    if report["sensitivity"]:
        print("\nsensitivity gate -- a 'no difference' claim needs these to say differ=True:")
        for s in report["sensitivity"]:
            cos = s["mean_cosine"]
            shown = f"mean_cosine={cos:.4f}" if cos is not None else s.get("reason", "no cosine")
            print(f"  {s['a']:18} vs {s['b']:18} differ={str(s['differ']):5} {shown}")


if __name__ == "__main__":
    raise SystemExit(main())
