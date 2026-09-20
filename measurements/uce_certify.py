"""Certify our transcription of UCE's scIB protocol against THEIR data and THEIR embeddings."""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export-data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    import os

    import scanpy as sc
    from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation

    hits = sorted(a.export_data.rglob("new_tabula_scib.h5ad")) or sorted(a.export_data.rglob("*scib*.h5ad"))
    if not hits:
        raise SystemExit(f"no scib h5ad under {a.export_data}; found: "
                         f"{[p.name for p in a.export_data.rglob('*.h5ad')][:20]}")
    ad = sc.read(hits[0])
    print(f"[certify] {hits[0].name}: {ad.n_obs} cells, obsm keys {sorted(ad.obsm)}", flush=True)

    want = ["X_uce", "X_geneformer", "X_scgpt", "X_tgpt", "X_scvi", "X_scarches"]
    keys = [k for k in want if k in ad.obsm]
    label, batch = "cell_ontology_class", "sample_id"
    for col in (label, batch):
        if col not in ad.obs.columns:
            raise SystemExit(f"their h5ad has no obs column {col!r}; has {list(ad.obs.columns)[:20]}")

    bm = Benchmarker(
        ad, batch_key=batch, label_key=label, embedding_obsm_keys=keys,
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(pcr_comparison=False),
        n_jobs=min(64, len(os.sched_getaffinity(0))),
    )
    bm.prepare()
    bm.benchmark()
    try:
        df = bm.get_results(min_max_scale=False)
    except KeyError:
        df = bm._results.transpose()
        df = df.loc[df.index != "Metric Type"]
    rep = {
        "source_h5ad": str(hits[0]),
        "n_cells": int(ad.n_obs),
        "n_types": int(ad.obs[label].nunique()),
        "n_batches": int(ad.obs[batch].nunique()),
        "embeddings": keys,
        "protocol": "scib.ipynb benchmark(do_batchcons=True), BioConservation defaults, "
                    "BatchCorrection(pcr_comparison=False), min_max_scale=False",
        "deviations": [
            "installed scib-metrics renamed silhouette_batch -> bras",
            "scib-metrics default CPU neighbours instead of their faiss GPU brute-force computer",
        ],
        "results": json.loads(df.to_json(orient="index")),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rep, indent=2))
    tmp.replace(a.out)
    print(df.to_string(), flush=True)
    print(f"[certify] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
