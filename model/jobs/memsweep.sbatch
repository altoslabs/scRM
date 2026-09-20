#!/bin/bash
#SBATCH --job-name=scrm-memsweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:10G

if [ -z "${SLURM_JOB_ID:-}" ]; then
    >&2 echo "Run with: sbatch $0"
    exit 1
fi
set -euo pipefail
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
CORPORA_ROOT=${SCRM_CORPUS_ROOT:-${CORPUS_ROOT:-${CORPORA_ROOT:-$PROJ/corpus}}}
export CORPORA_ROOT SCRM_CORPUS_ROOT="$CORPORA_ROOT"
echo "corpus root: $CORPORA_ROOT"
PY=${SCRM_PY:-python3}
export PYTHONPATH=$REPO/model:$REPO/target/py:$REPO/eval

srun $PY -m scrm_model.bench_memory --corpus "$CORPORA_ROOT"/*
