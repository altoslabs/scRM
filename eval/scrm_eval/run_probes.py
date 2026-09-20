"""E6/E9 runner: score stamped embedding artifacts against count-matrix targets, for one dataset."""

from __future__ import annotations

import argparse
import dataclasses
import os
import json
from pathlib import Path

import numpy as np
import joblib
from threadpoolctl import threadpool_limits

from . import metrics, artifact, controls, reduce, scoring, targets
from .probes import binary_expression_probe, expression_regression_probe, module_activity_probe
from .reduce import reduce_windows


DETECTION_THRESHOLD = 0


MIN_COMMON_FRACTION = 0.95
MIN_SALVAGED_FRACTION = 0.85


def common_population(arts: dict[str, artifact.Embeddings]) -> tuple[list[str], dict[str, int]]:
    """The cells every artifact covers, in the first artifact's order, plus who lost what."""
    orders = {tag: list(dict.fromkeys(e.cell_ids.tolist())) for tag, e in arts.items()}
    common = set.intersection(*(set(o) for o in orders.values()))
    first_tag = next(iter(arts))
    cells = [str(c) for c in orders[first_tag] if c in common]
    if not cells:
        raise artifact.StampError("the artifacts share no cells at all")
    largest = max(len(o) for o in orders.values())
    sizes = {t: len(o) for t, o in orders.items()}
    salvaged = [t for t, e in arts.items() if (e.stamp.notes or {}).get("salvaged")]
    short = [t for t, o in orders.items() if len(o) < largest]
    unexplained = [t for t in short if t not in salvaged]
    # A salvage stamp explains the cells ITS OWN artifact never reached, so the lenient floor wants a
    # short artifact and every short artifact stamped. Equal-length artifacts hold no shortfall for a
    # stamp to explain, whatever they are stamped.
    explained = bool(short) and not unexplained
    floor = MIN_SALVAGED_FRACTION if explained else MIN_COMMON_FRACTION
    if len(cells) < floor * largest:
        why = (f"unexplained shortfall in {unexplained}, so these are not the same population"
               if unexplained else
               "even the salvaged-artifact floor is not met, so these are not the same population"
               if explained else
               "no artifact is short, so no salvage stamp explains anything here: the artifacts hold "
               "different cells and are not the same population")
        raise artifact.StampError(
            f"only {len(cells)} cells are common to all artifacts but the largest has {largest} "
            f"({len(cells) / largest:.1%} < {floor:.0%}); sizes={sizes}. Refusing to score -- {why}."
        )
    # No salvage shortfall can explain two FULL-LENGTH artifacts disagreeing, so they answer to the
    # strict floor even when the lenient one governs the whole intersection.
    firm = [set(orders[t]) for t in orders if t not in short]
    if explained and len(firm) > 1:
        n_firm = len(set.intersection(*firm))
        if n_firm < MIN_COMMON_FRACTION * largest:
            raise artifact.StampError(
                f"the full-length artifacts share only {n_firm} cells of {largest} "
                f"({n_firm / largest:.1%} < {MIN_COMMON_FRACTION:.0%}); sizes={sizes}. Refusing to "
                f"score -- a salvage stamp explains only the cells its own artifact never reached, "
                f"so these are not the same population."
            )
    dropped = {tag: len(o) - len(cells) for tag, o in orders.items()}
    return cells, dropped


def depth_nuisance(adata, cells: list[str]) -> np.ndarray:
    """The depth nuisance every depth-residualised probe regresses out, on the targets' scale.

    Two columns per cell, `[log1p(total counts), log1p(genes detected)]`, matching
    `controls.depth2`. A cell absent from the matrix keeps a zero row.
    """
    rows, has = targets.align_to_cells(adata, cells)
    depth = np.zeros((len(cells), 2))
    sub = adata.X[rows[has]]
    depth[has, 0] = np.log1p(np.asarray(sub.sum(axis=1)).ravel())
    depth[has, 1] = np.log1p(np.asarray((sub > 0).sum(axis=1)).ravel())
    return depth


def _reuse_key(emb: artifact.Embeddings, reduction: str | None, population: str,
               counts: str, gene_sets: str | None) -> str:
    """This artifact's row identity, in a run over `population` cells of the `counts` matrix."""
    return scoring.reuse_key(emb, reduction_rule=_reduce_rule(emb, reduction),
                             context={"population": population, "counts": counts,
                                      "gene_sets": gene_sets})


def _reduce_rule(emb, reduction):
    """`reduction` if the caller forced one, else the rule this artifact's stamp implies."""
    from .reduce import rule_for

    return rule_for(emb.stamp, override=reduction)


def probe_report(arts: dict[str, artifact.Embeddings], dataset: str, gmt: Path | None, reduction: str,
                 reuse_from: Path | None = None) -> dict:
    """Build this dataset's targets once, then score every artifact against them."""
    first = next(iter(arts.values()))
    for other in arts.values():
        artifact.same_corpus(first.stamp, other.stamp)
    scoring.refuse_mixed_precision(arts)

    cells, dropped_cells = common_population(arts)
    population = artifact.cell_order_digest(cells)
    not_scored: dict[str, str] = {}
    adata = targets.load_counts(dataset)
    counts_id = targets._counts_digest(adata)
    rows, has = targets.align_to_cells(adata, cells)
    depth = depth_nuisance(adata, cells)

    panels = {}
    for n_top in (200, 500, 2000):
        idx = targets.hvg_panel(adata, n_top=n_top)
        panels[f"hvg{n_top}"] = targets.binary_expression_targets(
            adata, cells, idx, panel=f"hvg{n_top}", threshold=DETECTION_THRESHOLD)
        panels[f"hvg{n_top}"].provenance = {"n_top": n_top, "detection_threshold": DETECTION_THRESHOLD}
    try:
        pidx = targets.hvg_panel_pearson(adata, n_top=2000)
        panels["pearson2000"] = targets.binary_expression_targets(
            adata, cells, pidx, panel="pearson2000", threshold=DETECTION_THRESHOLD)
        panels["pearson2000"].provenance = {"n_top": 2000, "flavor": "pearson_residuals",
                                            "detection_threshold": DETECTION_THRESHOLD}
        overlap = len(set(pidx.tolist()) & set(targets.hvg_panel(adata, n_top=2000).tolist()))
        panels["pearson2000"].provenance["overlap_with_seurat_v3"] = overlap
    except Exception as exc:
        print(f"[probes] pearson-residual panel unavailable: {exc}", flush=True)
        not_scored["pearson2000"] = f"panel unavailable: {type(exc).__name__}: {exc}"

    for n_top in (500, 1000, 2000):
        idx = targets.hvg_panel(adata, n_top=n_top)
        key = f"legacy_hvg{n_top}"
        panels[key] = targets.binary_expression_targets(adata, cells, idx, panel=key, threshold=1)
    wil_idx, _, wil_prov = targets.leiden_marker_panel_wilcoxon(adata, top_k=100)
    panels["markers_wilcoxon_gt0"] = targets.binary_expression_targets(
        adata, cells, wil_idx, panel="markers_wilcoxon_gt0", threshold=DETECTION_THRESHOLD)
    panels["markers_wilcoxon_gt0"].provenance = {**wil_prov,
                                                 "detection_threshold": DETECTION_THRESHOLD,
                                                 "circular": True, "algo": metrics.CLUSTER_ALGO,
                                                 "why": "markers derived from graph clustering on the same counts"}
    panels["markers_wilcoxon_gt1"] = targets.binary_expression_targets(
        adata, cells, wil_idx, panel="markers_wilcoxon_gt1", threshold=1)
    panels["markers_wilcoxon_gt1"].provenance = {**wil_prov, "detection_threshold": 1,
                                                 "circular": True, "algo": metrics.CLUSTER_ALGO,
                                                 "why": "markers derived from graph clustering on the same counts"}
    legacy_marker_idx, _, legacy_prov = targets.leiden_marker_panel(adata, top_k=100)
    panels["markers_fstat_gt0"] = targets.binary_expression_targets(
        adata, cells, legacy_marker_idx, panel="markers_fstat_gt0",
        threshold=DETECTION_THRESHOLD)
    panels["markers_fstat_gt0"].provenance = {**legacy_prov, "test": "f_classif",
                                              "detection_threshold": DETECTION_THRESHOLD,
                                              "circular": True, "algo": metrics.CLUSTER_ALGO,
                                              "why": "markers derived from graph clustering on the same counts"}
    panels["legacy_markers100"] = targets.binary_expression_targets(
        adata, cells, legacy_marker_idx, panel="legacy_markers100", threshold=1)
    panels["legacy_markers100"].provenance = {**legacy_prov, "detection_threshold": 1,
                                              "circular": True, "algo": metrics.CLUSTER_ALGO,
                                              "why": "markers derived from graph clustering on the same counts"}

    qc_keep, qc_prov = targets.qc_mask(adata)
    qc_row = np.zeros(len(cells), dtype=bool)
    qc_row[has] = np.asarray(qc_keep)[rows[has]]

    hvg_idx = targets.hvg_panel(adata)
    marker_idx, _, marker_prov = targets.leiden_marker_panel_wilcoxon(adata)
    markers = targets.binary_expression_targets(
        adata, cells, marker_idx, panel="leiden_markers", threshold=DETECTION_THRESHOLD)
    markers.provenance = {**marker_prov, "detection_threshold": DETECTION_THRESHOLD,
                          "circular": True, "algo": metrics.CLUSTER_ALGO,
                          "why": "markers derived from graph clustering on the same counts"}
    graded = targets.expression_targets(adata, cells, hvg_idx, panel="hvg_graded")
    modules = None
    if gmt is None:
        known = dataset in targets.SPECIES
        found = targets.gene_set_path(dataset) if known else None
        gmt = Path(found) if found else None
        if gmt is None:
            not_scored["module_activity"] = (
                f"E9 skipped: MSigDB publishes no hallmark collection for "
                f"{targets.SPECIES[dataset]}, so this dataset has no module target set. Scoring it "
                f"against a borrowed human GMT through an ortholog bridge would invent the exact "
                f"gene-identity dependency this dataset exists to avoid."
                if known else
                f"E9 skipped: {dataset!r} is not in targets.SPECIES, so no hallmark collection "
                f"could be resolved for it and none was passed with --gmt."
            )
    gene_sets_id = None
    if gmt is not None:
        _sets = targets.read_gmt(gmt)
        modules = targets.module_scores(adata, _sets, cells)
        gene_sets_id = scoring.digest(
            [(k, sorted(v)) for k, v in sorted(_sets.items())])
        modules.provenance = {"gmt": str(gmt), "gene_sets_sha256": gene_sets_id}

    report: dict = {
        "dataset": dataset,
        "reduction": reduction,
        "reduction_applied": {p: reduce.rule_for(e.stamp, override=reduction)
                              for p, e in sorted({e.stamp.producer: e for e in arts.values()}.items())},
        "corpus_caveats": controls.corpus_caveats(dataset),
        "population": {
            "n_cells_scored": len(cells),
            "cells_dropped_to_intersect": dropped_cells,
            "cell_order_sha256": artifact.cell_order_digest(cells),
        },
        "panels": {
            t.panel: {"n_genes": len(t.gene_names), "n_cells_with_counts": int(t.has_counts.sum()),
                      "provenance": t.provenance}
            for t in (list(panels.values()) + [markers, graded] + ([modules] if modules is not None else []))
        },
        "qc": {"mad_outliers": qc_prov, "n_cells_qc_pass": int(qc_row.sum()),
               "n_cells_scored": len(cells)},
        "not_scored": not_scored,
        "artifacts": {},
    }

    def _one(tag, emb):
        own, x = reduce_windows(emb.cell_ids, emb.matrix, _reduce_rule(emb, reduction))
        pos = {str(c): i for i, c in enumerate(own)}
        try:
            take = np.fromiter((pos[c] for c in cells), dtype=np.int64, count=len(cells))
        except KeyError as e:
            raise artifact.StampError(f"{tag} lacks cell {e.args[0]} that the intersection contains")
        x = x[take]
        entry = {
            "producer": emb.stamp.producer,
            "w_eval": emb.stamp.w_eval,
            "forward_precision": emb.stamp.forward_precision or "fp32+eager",
            "binary_expression": {
                t.panel: dataclasses.asdict(binary_expression_probe(x, t.matrix, t.has_counts)) | {"per_target": None}
                for t in (list(panels.values()) + [markers])
            },
            "graded_expression": {
                variant: dataclasses.asdict(
                    expression_regression_probe(x, graded.matrix, graded.has_counts, depth=d)
                ) | {"per_target": None}
                for variant, d in (("raw", None), ("depth_residualised", depth))
            },
        }
        if modules is not None:
            entry["module_activity"] = {
                variant: dataclasses.asdict(
                    module_activity_probe(x, modules.matrix, modules.has_counts, depth=d)
                ) | {"per_target": None}
                for variant, d in (("raw", None), ("depth_residualised", depth))
            }
        entry["_reuse_key"] = _reuse_key(emb, reduction, population, counts_id,
                                         gene_sets_id)
        return tag, entry

    prior: dict[str, dict] = {}
    if reuse_from is not None and reuse_from.exists():
        try:
            old = json.loads(reuse_from.read_text())
            for tag, entry in (old.get("artifacts") or {}).items():
                if tag in arts and entry.get("_reuse_key") == _reuse_key(
                        arts[tag], reduction, population, counts_id, gene_sets_id):
                    prior[tag] = entry
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[probes] prior report unusable ({exc}); computing everything", flush=True)
            prior = {}
    items = [(t, e) for t, e in arts.items() if t not in prior]
    if prior:
        print(f"[probes] {dataset}: carrying forward {len(prior)} unchanged artifact(s), "
              f"computing {len(items)}", flush=True)
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SCRM_PROBE_JOBS") or 1)
    n_jobs = max(1, min(len(items), n_jobs))
    with threadpool_limits(limits=1):
        if n_jobs > 1:
            with joblib.parallel_backend("loky", n_jobs=n_jobs, inner_max_num_threads=1):
                scored = joblib.Parallel()(joblib.delayed(_one)(t, e) for t, e in items)
        else:
            scored = [_one(t, e) for t, e in items]
    for tag, entry in prior.items():
        report["artifacts"][tag] = entry
    for tag, entry in scored:
        report["artifacts"][tag] = entry
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", nargs="+", required=True, type=Path)
    p.add_argument("--dataset", required=True)
    p.add_argument("--gmt", type=Path, default=None, help="MSigDB .gmt for E9; omitted = E9 skipped")
    p.add_argument("--reduction", default=None,
                   help="force one rule on every artifact; omit so each is read by "
                        "reduce.rule_for(stamp)")
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--no-reuse", action="store_true",
                   help="recompute every artifact even if an unchanged row exists in --out")
    args = p.parse_args(argv)

    arts = {path.name.removesuffix(".arrow"): artifact.read(path) for path in args.artifacts}
    report = probe_report(arts, args.dataset, args.gmt, args.reduction,
                          reuse_from=None if args.no_reuse else args.out)
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2))
    os.replace(tmp, args.out)
    print(f"probe report -> {args.out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
