#!/bin/bash
#SBATCH --job-name=scrm-score1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:10G
set -uo pipefail
DS=${1:?usage: sbatch score_one.sbatch <dataset>}
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}; PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
export SCRM_REPO="$REPO"
PY=${SCRM_PY:-python3}
export PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py
export LD_LIBRARY_PATH=$(dirname "$(dirname "${SCRM_PY:-python3}")")/lib
CORPORA_ROOT=${SCRM_CORPUS_ROOT:-${CORPUS_ROOT:-${CORPORA_ROOT:-$PROJ/corpus}}}
export CORPORA_ROOT SCRM_CORPUS_ROOT="$CORPORA_ROOT"
echo "corpus root: $CORPORA_ROOT"
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
mapfile -t REQUIRED < <($PY -m scrm_eval.registry --required)
mapfile -t OPTIONAL < <($PY -m scrm_eval.registry --optional)
OUT=$PROJ/eval_artifacts/$DS.allmodels.probes.json
DEADLINE=$(( $(date +%s) + 7*3600 ))

while [ $(date +%s) -lt $DEADLINE ]; do
    if [ -s "$OUT" ] && [ "${FORCE:-0}" != "1" ]; then echo "$DS already scored"; exit 0; fi

    $PY - "$DS" <<'MERGEEOF' || true
import glob, sys
from pathlib import Path
import os
sys.path[:0] = [os.environ["SCRM_REPO"] + p for p in ("/model", "/eval", "/target/py")]
from scrm_eval import embed as ev
from scrm_eval import registry as _R
ds = sys.argv[1]
proj = _R.PROJ
for run in _R.runs():
    out = proj / "eval_runs" / run / f"{ds}.{run}.allreads.arrow"
    if out.exists():
        continue
    for nsh in (2, 3, 4, 6):
        shards = sorted(glob.glob(str(proj / "eval_runs" / run / f"{ds}.{run}.shard*-of-{nsh}.arrow")))
        if len(shards) == nsh:
            try:
                ev.merge_shards(shards, out, dataset=ds)
            except Exception as exc:
                print(f"  [merge] {run}: {exc}", flush=True)
            break
MERGEEOF

    ARTS=(); missing=0
    for run in "${REQUIRED[@]}"; do
        a=$PROJ/eval_runs/$run/$DS.$run.allreads.arrow
        if [ -s "$a" ]; then ARTS+=("$a"); else missing=$((missing+1)); fi
    done
    for run in "${OPTIONAL[@]}"; do
        a=$PROJ/eval_runs/$run/$DS.$run.allreads.arrow
        [ -s "$a" ] && ARTS+=("$a")
    done
    if [ $missing -gt 0 ]; then
        echo "$(date +%H:%M:%S) $DS: waiting on $missing/${#REQUIRED[@]} model artifacts"
        sleep 180; continue
    fi
    for b in $PROJ/eval_artifacts/$DS.*.arrow; do [ -s "$b" ] && ARTS+=("$b"); done
    echo "$(date +%H:%M:%S) $DS: scoring ${#ARTS[@]} artifacts -> $OUT"
    $PY -m scrm_eval.run_probes --artifacts "${ARTS[@]}" --dataset "$DS" -o "$OUT" \
        || { echo "$(date +%H:%M:%S) $DS: FAILED"; exit 1; }

    GRAINS=()
    case "$DS" in
        sc_mixology_5cl)
            GRAINS+=("$PROJ/raw/sc_mixology_meta/sc_mixology_5cl_labels_sng.json|$PROJ/eval_artifacts/$DS.allmodels.annotation.json");;
        tm_10X_P7_2)
            GRAINS+=("$PROJ/raw/tabula_muris/10X_P7_2_labels.json|$PROJ/eval_artifacts/$DS.allmodels.annotation.json");;
        fly_antenna)
            # broad only: results_page._C5_BLOCKS takes "broad". The fine labels stay on disk
            # for measurements/uce_protocols.py.
            GRAINS+=("$PROJ/raw/fly_antenna_meta/fly_antenna_labels_broad.json|$PROJ/eval_artifacts/$DS.allmodels.annotation_broad.json");;
    esac
    for pair in ${GRAINS+"${GRAINS[@]}"}; do
        LABELS=${pair%%|*}; AOUT=${pair##*|}
        [ -s "$LABELS" ] || { echo "$(date +%H:%M:%S) $DS: labels missing at $LABELS, skipped"; continue; }
        echo "$(date +%H:%M:%S) $DS: annotation -> $AOUT"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    $PY -m scrm_eval.run_annotation --artifacts "${ARTS[@]}" --labels "$LABELS" -o "$AOUT" \
            || { echo "$(date +%H:%M:%S) $DS: ANNOTATION FAILED"; exit 1; }
    done
    echo "$(date +%H:%M:%S) $DS: DONE"; exit 0
done
echo "$DS: deadline reached without 9/9"
