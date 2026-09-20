#!/bin/bash
#SBATCH --job-name=scrm-uce-theireval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:10G
set -uo pipefail
WHICH=${1:-b}
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
PY=$PROJ/envs/uce/bin/python
[ -x "$PY" ] || PY=${SCRM_PY:-python3}
export PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py:$REPO/measurements
export LD_LIBRARY_PATH=$(dirname "$(dirname "${SCRM_PY:-python3}")")/lib
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32

if [ "$WHICH" = "a" ] || [ "$WHICH" = "all" ]; then
    export PYTHONPATH=$PYTHONPATH:$PROJ/envs/scib-lib
fi

EMB=${2:-}; DRAW=${3:-}
ARG=(--protocol "$WHICH"); [ "$WHICH" = "all" ] && ARG=(--all)
SUFFIX="$WHICH"
if [ -n "$EMB" ]; then ARG+=(--embedding "$EMB"); SUFFIX="${SUFFIX}_${EMB}"; fi
if [ -n "$DRAW" ]; then ARG+=(--resample "$DRAW"); SUFFIX="${SUFFIX}_d${DRAW}"; fi
echo "protocol=$WHICH embedding=${EMB:-all} draw=${DRAW:-all} -> their_eval_${SUFFIX}.json"
srun $PY $REPO/measurements/uce_their_eval.py "${ARG[@]}" \
    -o "$PROJ/eval_reports/uce/their_eval_${SUFFIX}.json"
