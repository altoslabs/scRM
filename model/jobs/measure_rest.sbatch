#!/bin/bash
#SBATCH --job-name=scrm-measure-rest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:10G
set -uo pipefail
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
PY=${SCRM_PY:-python3}
export PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py
export LD_LIBRARY_PATH=$(dirname "$(dirname "${SCRM_PY:-python3}")")/lib
CORPORA_ROOT=${SCRM_CORPUS_ROOT:-${CORPUS_ROOT:-${CORPORA_ROOT:-$PROJ/corpus}}}
export CORPORA_ROOT SCRM_CORPUS_ROOT="$CORPORA_ROOT"
cd "$REPO"
echo "corpus root: $CORPORA_ROOT"

FAILED=()
run() {
    local name=$1; shift
    echo; echo "===== $name"
    if "$@"; then echo "----- $name OK"; else echo "----- $name FAILED (rc $?)"; FAILED+=("$name"); fi
}

R=$PROJ/eval_reports

if $PY measurements/seed_floor.py --check; then
    echo "----- seed_floor OK"; FLOOR_OK=1
else
    echo "----- seed_floor FAILED (rc $?)"; FAILED+=(seed_floor); FLOOR_OK=0
fi
run ckpt_choice       $PY measurements/checkpoint_choice.py --check
run e4_vocab_reach    $PY measurements/e4_vocab_reach.py
run e7_vs_control     $PY measurements/e7_vs_control.py -o "$R/e7_vs_control.crispr_a549_5k.json"
run e7_isoform        $PY measurements/e7_isoform_structure.py
run uce_certify       $PY measurements/uce_certify_invocation.py --dataset their_mouse_kidney_norn --their "$PROJ/fm/uce/their_data/mouse_kidney_norn.h5ad" --labels cell_type
if [ "$FLOOR_OK" = 1 ]; then
    run paired_signif     $PY measurements/paired_significance.py --write
    run paired_signif_e7  $PY measurements/paired_significance.py --e7 --write
else
    echo; echo "===== paired_significance NOT RUN"
    echo "registry.SEED_FLOOR is not ratified against the reseed sets on disk. Re-ratify the"
    echo "constant from $R/seed_floor.json, then run the two --write calls by hand."
    FAILED+=(paired_signif_blocked_on_seed_floor paired_signif_e7_blocked_on_seed_floor)
fi

echo
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED (${#FAILED[@]}): ${FAILED[*]}"
    echo "Of these, results_page.py reads only e7_vs_control; the rest feed the appendix helper."
    exit 1
fi
echo "every measurement in this job wrote its report"
