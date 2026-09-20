#!/bin/bash
#SBATCH --job-name=scrm-evalmatrix
#SBATCH --array=0-1499
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1,disk:tmp:10G
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --requeue
set -euo pipefail
PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
PY=${SCRM_PY:-python3}
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
export SCRM_REPO="$REPO"
export PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py
export LD_LIBRARY_PATH=$(dirname "$(dirname "${SCRM_PY:-python3}")")/lib
mapfile -t RUNS < <($PY -m scrm_eval.registry --runs)
mapfile -t ARMS < <($PY -m scrm_eval.registry --arms)
mapfile -t DSETS < <($PY -m scrm_eval.registry --datasets)
if [ -n "${ONE_DS:-}" ]; then
    $PY -m scrm_eval.registry --all-datasets 2>/dev/null | grep -qx "$ONE_DS" \
        || { echo "ONE_DS=$ONE_DS is not a registered dataset; refusing to guess"; exit 1; }
    DSETS=("$ONE_DS")
    echo "[eval_matrix] ONE_DS=$ONE_DS (index selects the run only)"
fi
NR=${#RUNS[@]}
ND=${#DSETS[@]}
if [ "$NR" -eq 0 ] || [ ${#DSETS[@]} -eq 0 ] || [ "$NR" -ne ${#ARMS[@]} ]; then
    echo "registry read failed: RUNS=$NR ARMS=${#ARMS[@]} DSETS=${#DSETS[@]}" >&2; exit 1
fi

if [ -z "${RUN:-}" ] && [ -z "${ONE_DS:-}" ] && [ -n "${SLURM_ARRAY_TASK_MAX:-}" ]; then
    NEED=$(( NR * ND ))
    if [ "$(( SLURM_ARRAY_TASK_MAX + 1 ))" -lt "$NEED" ]; then
        echo "REFUSING: --array=0-$SLURM_ARRAY_TASK_MAX covers $(( SLURM_ARRAY_TASK_MAX + 1 )) tasks," >&2
        echo "  but the registry holds $NR runs x $ND datasets = $NEED. The runs past the end would" >&2
        echo "  be SILENTLY never embedded. Resubmit with --array=0-$(( NEED - 1 )), and bump the" >&2
        echo "  #SBATCH --array line in this file so the default submission is correct too." >&2
        exit 1
    fi
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge $(( NR * ${#DSETS[@]} )) ]; then
    echo "task $SLURM_ARRAY_TASK_ID is outside $NR runs x ${#DSETS[@]} datasets" >&2; exit 1
fi
i=$SLURM_ARRAY_TASK_ID
if [ -n "${RUN:-}" ]; then
    ARM=""
    for _j in "${!RUNS[@]}"; do
        [ "${RUNS[$_j]}" = "$RUN" ] && ARM="${ARMS[$_j]}"
    done
    [ -n "$ARM" ] || { echo "RUN=$RUN is not registered; refusing to guess its arm"; exit 1; }
    DS=${DSETS[$((i % ND))]}
    echo "[eval_matrix] pinned RUN=$RUN ARM=$ARM DS=$DS (index selects the dataset only)"
else
    RUN=${RUNS[$((i % NR))]}; ARM=${ARMS[$((i % NR))]}; DS=${DSETS[$((i / NR))]}
fi
REPORTED_CKPT=$($PY -m scrm_eval.registry --ckpt "$RUN") || {
    echo "[eval_matrix] registry does not declare a reported checkpoint for $RUN" >&2; exit 1; }
CKPT=${CKPT:-$REPORTED_CKPT}
if [ "$CKPT" = "$REPORTED_CKPT" ]; then
    OUT=$PROJ/eval_runs/$RUN
else
    OUT=$PROJ/eval_ckptsel/$RUN.${CKPT%.pt}
    echo "[eval_matrix] CANDIDATE checkpoint $CKPT (reported: $REPORTED_CKPT) -> $OUT" \
         "(not a publishable artifact tree)"
fi
mkdir -p "$OUT"
if [ ! -s "$PROJ/runs/$RUN/$CKPT" ]; then
    echo "$PROJ/runs/$RUN/$CKPT does not exist." >&2
    $PY -m scrm_eval.registry --recipe "$RUN" 2>/dev/null | grep -q '^UNTRAINED=' \
        && echo "  $RUN is declared UNTRAINED: create it with measurements/untrained_control.py --write" >&2 \
        || echo "  train it with: sbatch model/jobs/arm_ablation.sbatch $RUN" >&2
    exit 1
fi
art=$OUT/$DS.$RUN.allreads.arrow

CORPORA_ROOT=${SCRM_CORPUS_ROOT:-${CORPUS_ROOT:-${CORPORA_ROOT:-$PROJ/corpus}}}
export CORPORA_ROOT SCRM_CORPUS_ROOT="$CORPORA_ROOT"
echo "embedding from corpus root: $CORPORA_ROOT"
[ -f "$CORPORA_ROOT/$DS/manifest.json" ] || { echo "corpus $DS not built under $CORPORA_ROOT yet"; exit 0; }
# Reuse an artifact only if it LOADS. FORCE=1 re-embeds regardless.
if [ -s "$art" ] && [ "${FORCE:-0}" != "1" ]; then
    if $PY -c "
import sys
sys.path[:0] = ['$SCRM_REPO/eval', '$SCRM_REPO/target/py']
from scrm_eval import artifact
artifact.read('$art')" 2>/dev/null; then
        echo "have $art"; exit 0
    fi
    echo "re-embedding $art: it is on disk but this tree cannot read its stamp"
fi
W=$($PY "${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"/measurements/windows_required.py --dataset "$DS")
echo "$RUN / $DS : embedding every read (w_eval=$W)"

srun $PY - "$ARM" "$RUN" "$DS" "$art" "$W" "$CKPT" <<'PYEOF'
import sys
import os
sys.path[:0] = [os.environ["SCRM_REPO"] + p for p in ("/model", "/eval", "/target/py")]
from scrm_eval import registry
from pathlib import Path
import scrm_py
from scrm_eval import embed as ev
from scrm_model.evalio import model_factory
from scrm_model.run_eval import eval_ready
arm, run, ds, art, w, ckpt_name = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
proj = registry.PROJ
ckpt = ev.Checkpoint.load(eval_ready(proj / "runs" / run / ckpt_name, arm, art.parent), model_factory=model_factory)
import os
window_batch = int(os.environ.get("WINDOW_BATCH", "512"))
corpora_root = Path(os.environ["CORPORA_ROOT"])
ev.embed_to_file(ckpt, scrm_py.Corpus([str(corpora_root / ds)]), art,
                 dataset=ds, w_eval=w, eval_seed=20260818, window_batch=window_batch)
print(f"embedded ALL reads (w_eval={w}, window_batch={window_batch}) -> {art}", flush=True)
PYEOF
echo "done: $RUN / $DS"
