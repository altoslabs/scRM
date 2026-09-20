#!/bin/bash
#SBATCH --job-name=scrm-arm-geometry
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --requeue
set -uo pipefail
cd "${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
export PYTHONPATH=model:eval:target/py
PY=${SCRM_PY:-python3}
srun $PY measurements/arm_geometry.py; rc_arm=$?
echo "arm_geometry rc=$rc_arm"
[ $rc_arm -eq 0 ]
