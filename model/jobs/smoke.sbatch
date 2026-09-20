#!/bin/bash
#SBATCH --job-name=scrm-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
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

srun $PY -m scrm_model.bench_loader --corpus \
    $CORPORA_ROOT/mouse_brain_5k \
    $CORPORA_ROOT/neuron_1k_v3 \
    $CORPORA_ROOT/neurons_5k \
    $CORPORA_ROOT/sod1_spinal_cord

srun $PY -m scrm_model.train \
    --corpus $CORPORA_ROOT/mouse_brain_5k \
                 $CORPORA_ROOT/neuron_1k_v3 \
                 $CORPORA_ROOT/neurons_5k \
                 $CORPORA_ROOT/sod1_spinal_cord \
    --steps 60 --ckpt-every 30 \
    --out $PROJ/runs/smoke_flagship
