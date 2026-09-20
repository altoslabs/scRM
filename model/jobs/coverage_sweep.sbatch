#!/bin/bash
#SBATCH --job-name=scrm-sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:10G
#SBATCH --array=0-3
set -euo pipefail
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
PY=${SCRM_PY:-python3}
export LD_LIBRARY_PATH=$(dirname "$(dirname "${SCRM_PY:-python3}")")/lib
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
OUT=$PROJ/eval_artifacts/sweeps
mkdir -p "$OUT"
FLAGSHIP=armS96-c3k-sigreg-lam002
CASES=("tm_10X_P7_2 $FLAGSHIP" "nuclei_2k $FLAGSHIP")
set -- ${CASES[$SLURM_ARRAY_TASK_ID]}
DS=$1; RUN=$2
srun $PY $REPO/measurements/coverage_sweep.py --dataset "$DS" --run "$RUN" \
    --ks 2 4 8 16 32 64 128 256 0 --panel hvg2000 \
    -o "$OUT/$DS.$RUN.hvg2000.json"
