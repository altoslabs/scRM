"""Is "more passes is worse" overfitting, or representation collapse? Measured per token-pass."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "model"), str(REPO / "eval"), str(REPO / "target" / "py")]
from threadpoolctl import threadpool_limits  # noqa: E402

from scrm_eval import controls, metrics, registry, targets  # noqa: E402
from scrm_eval.probes import binary_expression_probe  # noqa: E402
from scrm_eval.run_probes import DETECTION_THRESHOLD  # noqa: E402

P = registry.PROJ


def _runs() -> list[tuple[str, str, str]]:
    """(run, arm, regulariser) for the objective pair, DERIVED from the registry."""
    reg = registry.FLAGSHIP_RECIPE["regularizer"]
    out = [(registry.FLAGSHIP, registry.FLAGSHIP_RECIPE["arm"], reg)]
    for run, delta in registry.ABLATION_DELTAS.items():
        if set(delta) == {"regularizer"}:
            out.append((run, registry.recipe(run)["arm"], delta["regularizer"]))
    return out


SEED = 0


def snapshots(run: str) -> list[tuple[str, Path]]:
    d = P / "runs" / run
    out = [(f"pass{i}", d / f"pass{i}.pt") for i in (1, 2, 3)]
    out = [(n, p) for n, p in out if p.exists()]
    if (d / "latest.pt").exists():
        out.append(("latest", d / "latest.pt"))
    return out


def cell_vectors(ck, corpus, picked, names, w_cap: int, eval_seed: int) -> tuple[np.ndarray, list[str]]:
    import torch
    from scrm_eval import embed as ev

    vecs, got = [], []
    for ci in picked:
        b, _real, n_real = ev.cell_windows(corpus, int(ci), w_cap, eval_seed=eval_seed)
        if not n_real:
            continue
        tok = np.asarray(b.tokens)[:n_real]
        acc, n_win = None, 0
        for lo in range(0, n_real, 128):
            hi = min(lo + 128, n_real)
            with torch.no_grad():
                v = ck.model.embed_windows(tok[lo:hi], np.asarray(b.valid)[lo:hi],
                                           np.asarray(b.contig_segment_id)[lo:hi],
                                           np.asarray(b.umi_segment_id)[lo:hi],
                                           np.asarray(b.patch_start_ids)[lo:hi])
            v = np.asarray(v, dtype=np.float64)
            acc = v.sum(axis=0) if acc is None else acc + v.sum(axis=0)
            n_win += v.shape[0]
        vecs.append(acc / max(n_win, 1))
        got.append(names[ci])
    return np.stack(vecs).astype(np.float32), got


def geometry(x: np.ndarray) -> dict:
    er = metrics.effective_rank(x)
    n = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(0)
    sub = n[rng.choice(len(n), size=min(1200, len(n)), replace=False)]
    cos = float((sub @ sub.T)[np.triu_indices(len(sub), 1)].mean())
    return {"dim": int(x.shape[1]), "effective_rank": float(er),
            "rank_fraction": float(er / x.shape[1]), "mean_cosine": cos}


def main(argv=None) -> int:
    import scrm_py
    from scrm_model.evalio import model_factory
    from scrm_model.run_eval import eval_ready
    from scrm_eval import embed as ev

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default=registry.datasets("eval")[0])
    ap.add_argument("--cells", type=int, default=600)
    ap.add_argument("--w-cap", type=int, default=128)
    ap.add_argument("--eval-seed", type=int, default=20260818)
    a = ap.parse_args(argv)

    corpus = scrm_py.Corpus(path=str(registry.corpus_root() / a.dataset), seq_len=16384,
                             max_umi_len=1600, min_v=1)
    names = list(corpus.cell_names())
    rng = np.random.default_rng(SEED)
    picked = sorted(rng.choice(len(names), size=min(a.cells, len(names)), replace=False).tolist())

    adata = targets.load_counts(a.dataset)
    lab = metrics.cluster_labels(controls.count_scanpy(a.dataset)[1], seed=SEED)
    by_lab = dict(zip(controls._read_counts(a.dataset)[1], lab))

    out = {"dataset": a.dataset, "cells_requested": a.cells, "w_cap": a.w_cap,
           "eval_seed": a.eval_seed, "runs": {}}
    for run, arm, reg in _runs():
        snaps = snapshots(run)
        if not snaps:
            print(f"[collapse] {run}: no snapshots", flush=True)
            continue
        print(f"\n[collapse] {run} ({reg})", flush=True)
        for label, path in snaps:
            ck = ev.Checkpoint.load(eval_ready(path, arm, P / "eval_runs"),
                                    model_factory=model_factory)
            x, cells = cell_vectors(ck, corpus, picked, names, a.w_cap, a.eval_seed)
            row = {"regularizer": reg, "n_cells": len(cells), **geometry(x)}
            keep = [c for c in cells if c in by_lab]
            idx = [i for i, c in enumerate(cells) if c in by_lab]
            if len(keep) > 50:
                y = np.array([by_lab[c] for c in keep])
                with threadpool_limits(limits=1):
                    row["knn_purity"] = float(metrics.knn_label_purity(x[idx], y, k=15)["purity_adj"])
            ts = targets.binary_expression_targets(
                adata, cells, targets.hvg_panel(adata, n_top=200), panel="hvg200",
                threshold=DETECTION_THRESHOLD)
            with threadpool_limits(limits=1):
                r = binary_expression_probe(x, ts.matrix, ts.has_counts, folds=5, seeds=3)
            row["auroc_hvg200"] = float(r.macro)
            out["runs"].setdefault(run, {})[label] = row
            print(f"  {label:8s} rank {row['effective_rank']:7.2f} / {row['dim']:3d} "
                  f"({100*row['rank_fraction']:5.2f}%)  cos {row['mean_cosine']:.4f}  "
                  f"auroc200 {row['auroc_hvg200']:.4f}  knn {row.get('knn_purity', float('nan')):.4f}",
                  flush=True)
            dest = P / "eval_reports" / f"collapse_trajectory.{a.dataset}.json"
            tmp = dest.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(out, indent=2, sort_keys=True))
            tmp.replace(dest)
    print(f"\n[collapse] wrote {P / 'eval_reports' / f'collapse_trajectory.{a.dataset}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
