#!/bin/bash
#SBATCH --job-name=scrm-arm-ablation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=h100:1
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:10G
set -euo pipefail

RUN=${1:?usage: sbatch $0 <run>   (a run named in registry.ABLATION_DELTAS)}

PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
CORPORA_ROOT=${SCRM_CORPUS_ROOT:-${CORPUS_ROOT:-${CORPORA_ROOT:-$PROJ/corpus}}}
export CORPORA_ROOT SCRM_CORPUS_ROOT="$CORPORA_ROOT"
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
PY=${SCRM_PY:-python3}
export PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py
export LD_LIBRARY_PATH=$(dirname "$(dirname "${SCRM_PY:-python3}")")/lib

eval "$($PY -m scrm_eval.registry --recipe "$RUN")"
: "${ARM:?}" "${REGULARIZER:?}" "${LAMB:?}" "${SEED:?}" "${STEPS:?}" "${TRAIN_CELLS:?}" "${CORPORA:?}"
if [ -n "${UNTRAINED:-}" ]; then
    echo "$RUN is declared UNTRAINED (init $UNTRAINED): produce it with measurements/untrained_control.py"
    exit 1
fi
echo "run:         $RUN"
echo "corpus root: $CORPORA_ROOT"
if [ -n "${STEPS_OVERRIDE:-}" ]; then
    echo "[arm_ablation] STEPS OVERRIDE: recipe says $STEPS, training to $STEPS_OVERRIDE" \
         "(checkpoint-selection run; registry.recipe is unchanged)"
    STEPS=$STEPS_OVERRIDE
fi
echo "recipe:      arm=$ARM reg=$REGULARIZER lamb=$LAMB seed=$SEED steps=$STEPS cells=$TRAIN_CELLS"
echo "corpora:     $CORPORA"

read -r -a DS <<< "$CORPORA"
for d in "${DS[@]}"; do
    grep -q "\"$d\"" "$REPO/crates/scrm-corpus/tests/expected_stream_digests.json" \
        || { echo "$d is not pinned; run pinned_digests.py --report / --write and commit first"; exit 1; }
    [ -e "$CORPORA_ROOT/$d/manifest.json" ] \
        || { echo "$CORPORA_ROOT/$d has no manifest -- a directory of shards is not yet a corpus"; exit 1; }
done

if [ -n "${RESUME_LONGER:-}" ] && [ -e "$PROJ/runs/${RUN}/latest.pt" ]; then
    $PY - "$PROJ/runs/${RUN}/latest.pt" "$CORPORA_ROOT" "${DS[@]}" <<'PYCHK' || exit 1
import json, sys
from pathlib import Path
import torch
ckpt, root, *dss = sys.argv[1:]
c = torch.load(ckpt, map_location="cpu", weights_only=False)
have = c.get("corpus_compat_key") or {}
if not have:
    print("REFUSING RESUME: the checkpoint records no corpus_compat_key, so nothing can be checked. "
          "A guard that cannot verify is not a guard that passed.")
    raise SystemExit(1)
bad = {}
for d in dss:
    cfg = json.loads((Path(root) / d / "manifest.json").read_text())["declared"]["config"]
    for k, v in have.items():
        if k in cfg and cfg[k] != v:
            bad[f"{d}.{k}"] = (v, cfg[k])
if bad:
    print(f"REFUSING RESUME: the checkpoint's molecule definition differs from this corpus: {bad}")
    print("  That is exactly the corpus mixing the already-trained guard exists to prevent.")
    raise SystemExit(1)
print(f"[arm_ablation] resume OK: step {c.get('step')}, molecule definition matches "
      f"({len(dss)} corpus/corpora checked; corpus_digests is not recorded in the checkpoint, so "
      f"WHICH dataset comes from registry.recipe)")
PYCHK
elif [ -e "$PROJ/runs/${RUN}/latest.pt" ] && [ "${SLURM_RESTART_COUNT:-0}" = 0 ]; then
    echo "$RUN already trained; delete it, pass RESUME_LONGER=1 to train further, or pick another name"
    exit 1
fi

CORPUS_ARGS=()
for d in "${DS[@]}"; do CORPUS_ARGS+=("$CORPORA_ROOT/$d"); done

srun --ntasks=1 $PY -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=1 \
    -m scrm_model.train \
    --corpus "${CORPUS_ARGS[@]}" \
    --arm "$ARM" --steps "$STEPS" --seed "$SEED" \
    --lamb "$LAMB" \
    --train-frac "$TRAIN_FRAC" --train-cells "$TRAIN_CELLS" \
    --out "$PROJ/runs/${RUN}"
echo "done: $RUN"
