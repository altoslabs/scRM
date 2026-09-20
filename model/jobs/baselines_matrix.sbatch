#!/bin/bash
#SBATCH --job-name=scrm-basematrix
#SBATCH --array=0-79%10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1,disk:tmp:20G
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --requeue
set -uo pipefail
# The controls are ours; the count FMs are their authors'. MODELS=controls builds the
# controls; MODELS="geneformer scgpt ..." embeds those, which needs eval/countfm/fetch.py
# to have pulled the publishers' code and weights first.
read -r -a MODELS <<< "${MODELS:-controls}"
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}; PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
export SCRM_REPO="$REPO"
CORPORA_ROOT=${SCRM_CORPUS_ROOT:-${CORPUS_ROOT:-${CORPORA_ROOT:-$PROJ/corpus}}}
export CORPORA_ROOT SCRM_CORPUS_ROOT="$CORPORA_ROOT"
echo "corpus root: $CORPORA_ROOT"
W=$PROJ/countfm/scfm_weights
PY=${SCRM_PY:-python3}
export PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py
export LD_LIBRARY_PATH=$(dirname "$(dirname "${SCRM_PY:-python3}")")/lib

if [ -n "${ONE_DS:-}" ]; then
    $PY -m scrm_eval.registry --all-datasets 2>/dev/null | grep -qx "$ONE_DS" \
        || { echo "ONE_DS=$ONE_DS is not a registered dataset; refusing to guess"; exit 1; }
    DSETS=("$ONE_DS")
else
    mapfile -t DSETS < <($PY -m scrm_eval.registry --datasets)
fi
mapfile -t -O "${#MODELS[@]}" MODELS < <($PY -m scrm_eval.registry --panel)
ND=${#DSETS[@]}
i=$SLURM_ARRAY_TASK_ID
NM=${#MODELS[@]}
if [ "$i" -ge $(( NM * ND )) ]; then
    echo "task $i is outside $NM models x $ND datasets -- nothing to do"; exit 0
fi
M=${MODELS[$((i / ND))]}; DS=${DSETS[$((i % ND))]}
cd "$REPO"
[ -f "$CORPORA_ROOT/$DS/manifest.json" ] || { echo "corpus $DS not built under $CORPORA_ROOT yet"; exit 0; }

if [ "$M" = "controls" ]; then
    srun "${SCRM_PY:-python3}" - "$DS" <<'PYEOF'
import sys
import os
sys.path[:0] = [os.environ["SCRM_REPO"] + p for p in ("/eval", "/target/py")]
from scrm_eval import registry
from pathlib import Path
from scrm_eval import controls
ds = sys.argv[1]
art = registry.PROJ / "eval_artifacts"
# Reuse a control only if it LOADS; otherwise rebuild. FORCE=1 rebuilds regardless.
from scrm_eval import artifact
force = os.environ.get("FORCE") == "1"
for producer in controls.BUILT_PRODUCERS:
    p = art / f"{ds}.{producer}.arrow"
    reusable = False
    if p.exists() and not force:
        try:
            artifact.read(p)
            reusable = True
        except Exception as exc:
            print(f"rebuilding {p.name}: {type(exc).__name__}: {str(exc)[:80]}", flush=True)
    print(p if reusable else controls.build(ds, producer, art), flush=True)
PYEOF
    exit $?
fi

if [ -s "$PROJ/eval_artifacts/$DS.$M.arrow" ]; then
    if $PY measurements/baseline_population.py --dataset "$DS" --producer "$M" --check -q; then
        echo "have $DS.$M"; exit 0
    fi
    echo "REFUSING TO SKIP: $DS.$M is on disk but does not cover the corpus cells the count matrix"
    echo "has. Move the cached inputs aside as printed above, then resubmit this task."
    exit 1
fi
if [ "$M" = uce ]; then
    echo "[baseline] $M is embedded by model/jobs/uce_matrix.sbatch (per-species ESM2 protein"
    echo "           embeddings, not a shared human vocabulary), not by embed_count_fm.py."
    echo "           Run:  ONE_DS=$DS sbatch --array=0 model/jobs/uce_matrix.sbatch"
    exit 0
fi
NPZ=$PROJ/fm/$DS.$M.npz
if [ ! -s "$NPZ" ]; then
    extra=(); case "$M" in scvi_pretrained) extra+=(--scvi-model-dir "$W/scvi_ts_lung") ;; esac
    read -r MAT SP < <("${SCRM_PY:-python3}" -c "
import sys; sys.path[:0]=['$REPO/eval', '$REPO/measurements']
from pathlib import Path
from scrm_eval import controls
from scrm_eval.targets import SPECIES
d = Path('$PROJ/fm/cells_$DS')
import baseline_population as bp
print(controls.filtered_matrix('$DS', d) if bp.matrix_dir('$DS')[0] != 'OK' else d, SPECIES['$DS'])" | tail -1)
    [ -n "$MAT" ] && [ -n "$SP" ] || { echo "could not resolve matrix/species for $DS"; exit 1; }
    BRIDGE=none
    case "$SP" in
        mouse)   BRIDGE=$W/ortholog_mouse_human.tsv;;
        fly)     BRIDGE=$W/ortholog_fly_human.tsv;;
        rat)     BRIDGE=$W/ortholog_rat_human.tsv;;
        macaque) ;;   # already human-annotated: no bridge
        human)   ;;
        *)       echo "unknown species '$SP' for $DS -- refusing to guess a bridge"; exit 1;;
    esac
    [ "$BRIDGE" = none ] || extra+=(--ortholog-table "$BRIDGE")
    echo "[baseline] $M / $DS : species=$SP ortholog=$(basename "$BRIDGE")"
    srun "${SCRM_PY_SCFM:-python3}" \
        $REPO/eval/countfm/embed_count_fm.py \
        --model "$M" --matrix "$MAT" --out-npz "$NPZ" --weights-root "$W" \
        --device "${DEVICE:-cuda}" --batch-size 32 "${extra[@]}" || exit 1
fi
"${SCRM_PY:-python3}" - "$DS" "$M" "$NPZ" <<'PYEOF'
import sys
import os
sys.path[:0] = [os.environ["SCRM_REPO"] + p for p in ("/eval", "/target/py")]
from scrm_eval import registry
from pathlib import Path
from scrm_eval import controls
print(controls.from_npz(sys.argv[1], sys.argv[2], sys.argv[3], registry.PROJ / "eval_artifacts"), flush=True)
PYEOF
