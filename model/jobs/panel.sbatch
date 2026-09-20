#!/bin/bash
#SBATCH --job-name=scrm-panel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:10G

if [ -z "${SLURM_JOB_ID:-}" ]; then
    >&2 echo "Run with: sbatch $0"
    exit 1
fi
set -euo pipefail
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
PY=${SCRM_PY:-python3}
export PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py

PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
for ARM in A C; do
    srun $PY -m scrm_model.run_eval \
        --checkpoint $PROJ/runs/arm$ARM/latest.pt \
        --arm $ARM --tag arm$ARM \
        --out-dir $PROJ/eval_runs/arm$ARM
done
