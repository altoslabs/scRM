#!/bin/bash
#SBATCH --job-name=scrm-pin-digests
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:10G
set -euo pipefail
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
export PYTHONPATH=$REPO/target/py
export LD_LIBRARY_PATH=$(dirname "$(dirname "${SCRM_PY:-python3}")")/lib
PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
CORPORA_ROOT=${SCRM_CORPUS_ROOT:-${CORPUS_ROOT:-${CORPORA_ROOT:-$PROJ/corpus}}}
export CORPORA_ROOT SCRM_CORPUS_ROOT="$CORPORA_ROOT"

MODE=${MODE:-}
echo "pinned_digests ${MODE:-verify} on $CORPORA_ROOT"
srun "${SCRM_PY:-python3}" \
    $REPO/crates/scrm-corpus/tests/pinned_digests.py \
    "$CORPORA_ROOT" ${MODE:+"$MODE"}
echo "digests ${MODE:-verify} complete on $CORPORA_ROOT"
