#!/bin/bash
#SBATCH --job-name=scrm-collapse
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:10G
set -uo pipefail
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
PY=${SCRM_PY:-python3}
export PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py
export LD_LIBRARY_PATH=$(dirname "$(dirname "${SCRM_PY:-python3}")")/lib
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
CORPORA_ROOT=${SCRM_CORPUS_ROOT:-${CORPUS_ROOT:-${CORPORA_ROOT:-$PROJ/corpus}}}
export CORPORA_ROOT SCRM_CORPUS_ROOT="$CORPORA_ROOT"
DS=${DS:-$($PY -m scrm_eval.registry --datasets | head -1)}
echo "corpus root: $CORPORA_ROOT"
echo "dataset: $DS"
[ -f "$CORPORA_ROOT/$DS/manifest.json" ] || {
    echo "no corpus for $DS under $CORPORA_ROOT -- set SCRM_CORPUS_ROOT to the tree that holds it" >&2
    exit 1
}
srun $PY $REPO/measurements/collapse_trajectory.py --dataset "$DS" \
    --cells "${1:-600}" --w-cap "${2:-128}"
