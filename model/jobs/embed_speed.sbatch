#!/bin/bash
#SBATCH --job-name=scrm-embed-speed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:10G
set -euo pipefail
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
export PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py
export LD_LIBRARY_PATH=$(dirname "$(dirname "${SCRM_PY:-python3}")")/lib
srun "${SCRM_PY:-python3}" $SCRM_REPO/measurements/embed_speed.py
