#!/bin/bash
#SBATCH --job-name=scrm-bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:10G
set -uo pipefail
ARM=${1:-}
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
export SCRM_REPO="$REPO"
PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
PY=${SCRM_PY:-python3}
export PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py
export LD_LIBRARY_PATH=$(dirname "$(dirname "${SCRM_PY:-python3}")")/lib

mapfile -t CORPORA < <($PY - <<'PYEOF'
import json, sys
import os
REPO = os.environ["SCRM_REPO"]
sys.path[:0] = [REPO + p for p in ("/eval",)]
from scrm_eval.registry import corpus_root
sets=json.load(open(f"{REPO}/model/jobs/training_sets.json"))["sets"]
root = corpus_root()
for c in sets["mouse56"]["corpora"]:
    print(root / c)
PYEOF
)
echo "[bench] arm=${ARM:-<flagship, from config.FLAGSHIP_RECIPE>} over ${#CORPORA[@]} corpora"
BATCHES=${2:-128 256 512 1024}
srun $PY -m scrm_model.bench_speed ${ARM:+--arm "$ARM"} --corpus "${CORPORA[@]}" --batch-cells $BATCHES
