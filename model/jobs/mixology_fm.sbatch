#!/bin/bash
#SBATCH --job-name=scrm-mixfm
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --gres=disk:tmp:20G
#SBATCH --requeue

if [ -z "${SLURM_JOB_ID:-}" ]; then >&2 echo "Run with: sbatch $0"; exit 1; fi
set -uo pipefail
PY=${SCRM_PY_SCFM:-python3}
PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
W=$PROJ/countfm/scfm_weights
MODELS=(geneformer scgpt scvi_pretrained genejepa)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
NPZ=$PROJ/fm/sc_mixology_5cl.$M.npz
# Reuse an npz only if it LOADS and carries the record `controls.from_npz` reads.
# FORCE=1 re-embeds regardless.
if [ -s "$NPZ" ] && [ "${FORCE:-0}" != "1" ]; then
    if "${SCRM_PY:-python3}" - "$NPZ" <<'PYEOF'
import json
import sys

import numpy as np

with np.load(sys.argv[1], allow_pickle=True) as z:
    for key in ("emb", "barcode", "provenance"):
        if key not in z:
            raise SystemExit(f"no {key!r}; keys are {sorted(z.files)}")
    rows = np.asarray(z["emb"]).shape[0]
    if rows != len(z["barcode"]):
        raise SystemExit(f"{rows} embeddings but {len(z['barcode'])} barcodes")
    json.loads(str(z["provenance"]))
PYEOF
    then
        echo "have $NPZ"; exit 0
    fi
    echo "re-embedding $NPZ: on disk but this tree cannot read it or it carries no provenance"
fi
extra=()
case "$M" in scvi_pretrained) extra+=(--scvi-model-dir "$W/scvi_ts_lung") ;; esac
srun "$PY" "$REPO/eval/countfm/embed_count_fm.py" \
    --model "$M" \
    --matrix $PROJ/raw/sc_mixology_meta/sc_mixology_5cl_matrix \
    --out-npz "$NPZ" \
    --weights-root "$W" \
    --device cuda --batch-size 32 "${extra[@]}"
