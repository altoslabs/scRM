#!/bin/bash
#SBATCH --job-name=scrm-heldout-base
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:20G
set -uo pipefail
MODELS=(controls geneformer scgpt scvi_pretrained genejepa controls geneformer scgpt scvi_pretrained genejepa)
DSETS=(tm_10X_P7_2 tm_10X_P7_2 tm_10X_P7_2 tm_10X_P7_2 tm_10X_P7_2)
i=$SLURM_ARRAY_TASK_ID
M=${MODELS[$i]}; DS=${DSETS[$i]}
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}; PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
export SCRM_REPO="$REPO"
W=$PROJ/countfm/scfm_weights
export PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py
export LD_LIBRARY_PATH=$(dirname "$(dirname "${SCRM_PY:-python3}")")/lib
cd "$REPO"

if [ "$M" = "controls" ]; then
    srun "${SCRM_PY:-python3}" - "$DS" <<'PYEOF'
import sys
import os
sys.path[:0] = [os.environ["SCRM_REPO"] + p for p in ("/eval", "/target/py")]
from scrm_eval import registry
from pathlib import Path
from scrm_eval import controls
ds = sys.argv[1]
for producer in controls.BUILT_PRODUCERS:
    print(controls.build(ds, producer, registry.PROJ / "eval_artifacts"), flush=True)
PYEOF
    exit $?
fi

NPZ=$PROJ/fm/$DS.$M.npz
if [ ! -s "$NPZ" ]; then
    extra=(); case "$M" in scvi_pretrained) extra+=(--scvi-model-dir "$W/scvi_ts_lung") ;; esac
    srun "${SCRM_PY_SCFM:-python3}" \
        $REPO/eval/countfm/embed_count_fm.py \
        --model "$M" --matrix "$("${SCRM_PY:-python3}" -c "
import sys; sys.path[:0]=['$REPO/eval']
from scrm_eval import registry; print(registry.count_matrix('$DS'))")" \
        --out-npz "$NPZ" --weights-root "$W" \
        --ortholog-table "$W/ortholog_mouse_human.tsv" \
        --device cuda --batch-size 32 "${extra[@]}" || exit 1
fi
"${SCRM_PY:-python3}" - "$DS" "$M" "$NPZ" <<'PYEOF'
import sys
import os
sys.path[:0] = [os.environ["SCRM_REPO"] + p for p in ("/eval", "/target/py")]
from scrm_eval import registry
from pathlib import Path
from scrm_eval import controls
ds, producer, npz = sys.argv[1], sys.argv[2], sys.argv[3]
print(controls.from_npz(ds, producer, npz, registry.PROJ / "eval_artifacts"), flush=True)
PYEOF
