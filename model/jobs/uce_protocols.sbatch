#!/bin/bash
#SBATCH --job-name=scrm-uce-protocols
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --requeue
set -uo pipefail
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}; PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
cd "$REPO"
export PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py:$PROJ/envs/uce-pylib
export JAX_PLATFORMS=cpu
if [ -n "${DATASET:-}" ]; then
    SCOPE=(--dataset "$DATASET")
else
    SCOPE=(--all)
fi
echo "uce_protocols: scope=${SCOPE[*]} protocols=${PROTOCOLS:-b} CENTRE=${CENTRE:-0}"
srun "${SCRM_PY:-python3}" measurements/uce_protocols.py \
     "${SCOPE[@]}" --protocols "${PROTOCOLS:-b}"
