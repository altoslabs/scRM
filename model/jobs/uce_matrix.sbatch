#!/bin/bash
#SBATCH --job-name=scrm-uce
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:10G
set -euo pipefail
PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
export SCRM_REPO="$REPO"
UPY=${SCRM_PY:-python3}
SPY=${SCRM_PY_SCFM:-python3}
export LD_LIBRARY_PATH=$(dirname "$(dirname "${SCRM_PY:-python3}")")/lib

mapfile -t DSETS < <(PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py $UPY -m scrm_eval.registry --datasets)
DS=${DSETS[$SLURM_ARRAY_TASK_ID]}
if [ -n "${ONE_DS:-}" ]; then
    PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py $UPY -m scrm_eval.registry --all-datasets 2>/dev/null \
        | grep -qx "$ONE_DS" || { echo "ONE_DS=$ONE_DS is not a registered dataset; refusing to guess"; exit 1; }
    DS=$ONE_DS
fi
SP=$(PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py $UPY -c "
import sys; sys.path[:0]=['$REPO/eval']
from scrm_eval.targets import SPECIES; print(SPECIES['$DS'])")
case "$SP" in
    macaque) echo "[uce] $DS: species=macaque -> embedding as human (matrix is human-annotated, ENSG ids)"
             SP=human;;
esac
MAT=$(PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py $UPY -c "
import sys; sys.path[:0]=['$REPO/eval','$REPO/target/py','$REPO/measurements']
from pathlib import Path
from scrm_eval import controls
import baseline_population as bp
d = Path('$PROJ/fm/cells_$DS')
print(controls.filtered_matrix('$DS', d) if bp.matrix_dir('$DS')[0] != 'OK' else d)" | tail -1)
[ -n "$MAT" ] && [ -d "$MAT" ] || { echo "could not resolve a filtered matrix for $DS"; exit 1; }
NPZ=$PROJ/fm/uce/npz/$DS.uce.npz
ART=$PROJ/eval_artifacts/$DS.uce.arrow
# Reuse an artifact only if it LOADS; FORCE=1 rebuilds regardless.
if [ -s "$ART" ] && [ "${FORCE:-0}" != "1" ]; then
    if $PY -c "
import sys
sys.path[:0] = ['$SCRM_REPO/eval', '$SCRM_REPO/target/py']
from scrm_eval import artifact
artifact.read('$ART')" 2>/dev/null; then
        echo "$ART exists and loads; skipping"; exit 0
    fi
    echo "rebuilding $ART: on disk but this tree cannot read its stamp"
fi

echo "[uce] $DS species=$SP (native, no bridge)"
PYTHONPATH=$PROJ/envs/uce-scfm-extra srun $SPY \
    $REPO/measurements/embed_uce.py --dataset "$DS" --matrix-dir "$MAT" --species "$SP" --out-npz "$NPZ"

$UPY - "$DS" uce "$NPZ" <<'PYEOF'
import sys
import os
sys.path[:0] = [os.environ["SCRM_REPO"] + p for p in ("/eval", "/target/py")]
from scrm_eval import registry
from pathlib import Path
from scrm_eval import controls
print(controls.from_npz(sys.argv[1], sys.argv[2], sys.argv[3], registry.PROJ / "eval_artifacts"), flush=True)
PYEOF
echo "done: $DS.uce"
