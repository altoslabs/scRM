#!/bin/bash
#SBATCH --job-name=scrm-annot1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:10G
set -uo pipefail
DS=${1:?usage: sbatch annotate_one.sbatch <dataset>}
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}; PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
PY=${SCRM_PY:-python3}
export PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py
export LD_LIBRARY_PATH=$(dirname "$(dirname "${SCRM_PY:-python3}")")/lib
CORPORA_ROOT=${SCRM_CORPUS_ROOT:-${CORPUS_ROOT:-${CORPORA_ROOT:-$PROJ/corpus}}}
export CORPORA_ROOT SCRM_CORPUS_ROOT="$CORPORA_ROOT"
echo "corpus root: $CORPORA_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mapfile -t REQUIRED < <($PY -m scrm_eval.registry --required)
mapfile -t OPTIONAL < <($PY -m scrm_eval.registry --optional)
RUNS=("${REQUIRED[@]}" "${OPTIONAL[@]}")
LABELS=$($PY -m scrm_eval.registry --label-file "$DS" 2>/dev/null || true)
[ -n "$LABELS" ] && [ -s "$LABELS" ] || { echo "$DS: no independent label file, nothing to annotate"; exit 0; }
echo "$DS: labels $LABELS"

ARTS=()
for run in "${RUNS[@]}"; do
    a=$PROJ/eval_runs/$run/$DS.$run.allreads.arrow
    [ -s "$a" ] && ARTS+=("$a")
done
for b in $PROJ/eval_artifacts/$DS.*.arrow; do [ -s "$b" ] && ARTS+=("$b"); done
echo "$(date +%H:%M:%S) $DS: annotating ${#ARTS[@]} artifacts"

AOUT=$PROJ/eval_artifacts/$DS.allmodels.annotation.json
srun --exclusive -n1 $PY -m scrm_eval.run_annotation \
    --artifacts "${ARTS[@]}" --labels "$LABELS" -o "$AOUT" \
    || { echo "$(date +%H:%M:%S) $DS: ANNOTATION FAILED"; exit 1; }
echo "$(date +%H:%M:%S) $DS: DONE"
