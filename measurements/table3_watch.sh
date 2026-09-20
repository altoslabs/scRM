#!/usr/bin/env bash
set -uo pipefail
REPO=$SCRM_REPO; PROJ=$SCRM_PROJ
export SCRM_CORPUS_ROOT=${SCRM_CORPUS_ROOT:-$PROJ/corpus_v3}
PY=${SCRM_PY:-python3}
export LD_LIBRARY_PATH=$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py
DS=$($PY -c "import sys; sys.path.insert(0,'$REPO/eval'); from scrm_eval import registry as R; print(R.spec_dataset())")
cd "$REPO"
echo "watching for pass checkpoints; Table 3 dataset = $DS"
while :; do
    for RUN in armS96-c3k-sigreg-lam002 armS96-sc_mixology_5cl; do
        for CK in $(ls $PROJ/runs/$RUN/pass*.pt 2>/dev/null | xargs -r -n1 basename); do
            ART=$PROJ/eval_ckptsel/$RUN.${CK%.pt}/$DS.$RUN.allreads.arrow
            [ -s "$ART" ] && continue
            OWN=$PROJ/eval_ckptsel/$RUN.${CK%.pt}/$DS.$RUN.allreads.arrow.resume/owner
            if [ -e "$OWN" ] && [ $(( $(date +%s) - $(stat -c %Y "$OWN") )) -lt 900 ]; then
                continue
            fi
            J=$(CKPT=$CK RUN=$RUN ONE_DS=$DS sbatch --parsable --array=0 \
                model/jobs/eval_matrix.sbatch 2>&1)
            echo "$(date -u +%H:%M) submitted $RUN $CK -> $J"
        done
    done
    sleep 600
done
