#!/bin/bash
#SBATCH --job-name=scrm-mixbase
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:10G

if [ -z "${SLURM_JOB_ID:-}" ]; then >&2 echo "Run with: sbatch $0"; exit 1; fi
set -euo pipefail
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
PY=${SCRM_PY:-python3}
export PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py
export LD_LIBRARY_PATH=$(dirname "$(dirname "${SCRM_PY:-python3}")")/lib
cd "$REPO"

srun $PY - <<'PYEOF'
from pathlib import Path
from scrm_eval import controls
from scrm_eval import registry

ART = registry.PROJ / "eval_artifacts"
FM = registry.PROJ / "fm"
ds = "sc_mixology_5cl"
for producer in controls.BUILT_PRODUCERS:
    p = ART / f"{ds}.{producer}.arrow"
    print(p if p.exists() else controls.build(ds, producer, ART), flush=True)
for npz in sorted(FM.glob(f"{ds}.*.npz")):
    producer = npz.name[len(ds) + 1 : -4]
    p = ART / f"{ds}.{producer}.arrow"
    print(p if p.exists() else controls.from_npz(ds, producer, npz, ART), flush=True)
PYEOF
$PY -m scrm_eval.run_annotation \
    --artifacts $PROJ/eval_artifacts/sc_mixology_5cl.*.arrow \
    --labels $PROJ/raw/sc_mixology_meta/sc_mixology_5cl_labels_sng.json \
    -o $PROJ/eval_artifacts/sc_mixology_5cl.baselines.annotation.json
echo mixology baselines complete
