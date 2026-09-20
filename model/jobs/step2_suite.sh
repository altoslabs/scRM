#!/bin/bash
set -uo pipefail
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
PY=${SCRM_PY:-python3}
export PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py
export LD_LIBRARY_PATH=$(dirname "$(dirname "${SCRM_PY:-python3}")")/lib
export SCRM_REPO=$REPO
export SBATCH_ACCOUNT=${SCRM_SLURM_ACCOUNT:?set SCRM_SLURM_ACCOUNT to your slurm account}
export SBATCH_PARTITION=${SCRM_SLURM_PARTITION:?set SCRM_SLURM_PARTITION to your partition}
[ -n "${SCRM_LOG_DIR:-}" ] && export SBATCH_OUTPUT="$SCRM_LOG_DIR/%x-%j.out"
cd "$REPO"

: "${SCRM_CORPUS_ROOT:?set it to the corpus tree, e.g. $PROJ/corpus_v3 -- the default $PROJ/corpus is empty and every stage would skip}"
export CORPORA_ROOT="$SCRM_CORPUS_ROOT"
DRY=${DRY:-}
_DRYC=$(mktemp); echo 0 > "$_DRYC"; trap 'rm -f "$_DRYC"' EXIT
sub() {
    if [ -n "$DRY" ]; then
        local n=$(( $(cat "$_DRYC") + 1 )); echo "$n" > "$_DRYC"
        echo "    would: sbatch $*" >&2; echo "DRY$n"
    else
        sbatch "$@" | awk '{print $NF}'
    fi
}

mapfile -t RUNS  < <($PY -m scrm_eval.registry --runs)
mapfile -t DSETS < <($PY -m scrm_eval.registry --datasets)
mapfile -t LDS   < <($PY -m scrm_eval.registry --labelled-datasets)
[ ${#RUNS[@]} -gt 0 ] && [ ${#DSETS[@]} -gt 0 ] || { echo "registry read failed" >&2; exit 1; }
echo "corpus root: $SCRM_CORPUS_ROOT"
echo "scope:       ${#RUNS[@]} arms x ${#DSETS[@]} eval datasets"

fail=0
for d in "${DSETS[@]}"; do
    [ -e "$SCRM_CORPUS_ROOT/$d/manifest.json" ] || { echo "  MISSING corpus: $d"; fail=1; }
    grep -q "\"$d\"" "$REPO/crates/scrm-corpus/tests/expected_stream_digests.json" \
        || { echo "  UNPINNED corpus: $d -- embed() cannot stamp it"; fail=1; }
done
$PY - <<'PYEOF' || fail=1
import sys
from scrm_eval import registry
bad = {}
for d in registry.datasets("eval") + registry.datasets("eval-pending"):
    try:
        registry.full_coverage(d)
    except Exception as exc:
        bad[d] = f"{type(exc).__name__}: {exc}"
for d, m in sorted(bad.items()):
    print(f"  NO USABLE w_eval: {d}: {m}")
sys.exit(1 if bad else 0)
PYEOF
$PY - <<'PYEOF' || fail=1
import sys
import torch
from scrm_eval import registry
bad = []
for run in registry.runs():
    rec = registry.recipe(run)
    ck = registry.reported_ckpt(run)
    f = registry.PROJ / "runs" / run / ck
    if not f.is_file():
        bad.append(f"{run}: no {ck}")
        continue
    if rec.get("untrained"):
        continue
    step = torch.load(f, map_location="cpu", weights_only=False).get("step")
    if step != rec["steps"]:
        bad.append(f"{run}: checkpoint is at step {step} of a declared {rec['steps']} -- STILL TRAINING")
for b in bad:
    print(f"  {b}")
sys.exit(1 if bad else 0)
PYEOF
echo "--- baseline populations (measurements/baseline_population.py) ---"
if [ -n "${SKIP_BASELINE_AUDIT:-}" ]; then
    echo "SKIPPED by SKIP_BASELINE_AUDIT. Whether the count-side baselines describe this corpus" >&2
    echo "        is then unchecked. A stale baseline is scored as though current, and the" >&2
    echo "        population loss reaches the report looking like a property of the data." >&2
elif $PY measurements/baseline_population.py --check -q; then
    echo "baseline populations: OK"
elif [ -n "${STRICT_BASELINES:-}" ]; then
    fail=1
else
    echo "WARNING: stale count-side baselines above. They will be scored as though current, and the" >&2
    echo "         population loss will appear in the report as if it were a property of the data." >&2
    echo "         Rebuild them as printed, or re-run with STRICT_BASELINES=1 to make this fatal." >&2
fi

[ "$fail" = 0 ] || { echo "preconditions failed; nothing submitted" >&2; exit 1; }
echo "preconditions: OK"

A=()
for r in "${RUNS[@]}"; do
    want=0
    for d in "${DSETS[@]}"; do [ -s "$PROJ/eval_runs/$r/$d.$r.allreads.arrow" ] || want=1; done
    [ "$want" = 0 ] && { echo "  [A] $r: all ${#DSETS[@]} datasets embedded"; continue; }
    id=$(RUN=$r sub --array=0-$(( ${#DSETS[@]} - 1 )) model/jobs/eval_matrix.sbatch)
    echo "  [A] $r -> $id"
    A+=("$id")
done
FLAG=$($PY -m scrm_eval.registry --flagship)
for d in $($PY -m scrm_eval.registry --all-datasets | grep -vxF -f <(printf '%s\n' "${DSETS[@]}") | grep -v permuted); do
    case "$d" in
        2500_*|crispr_a549_5k) ;;
        *) continue ;;   # a training corpus: an evaluation table may not contain one
    esac
    [ -s "$PROJ/eval_runs/$FLAG/$d.$FLAG.allreads.arrow" ] && { echo "  [A] $FLAG/$d: embedded"; continue; }
    id=$(RUN=$FLAG ONE_DS=$d sub --array=0 model/jobs/eval_matrix.sbatch)
    echo "  [A] $FLAG / $d -> $id"
    A+=("$id")
done
dep() { local ids="$1"; [ -z "$ids" ] && echo "" || echo "--dependency=afterany:${ids}"; }
ADEP=$(IFS=:; echo "${A[*]:-}")

B=()
for d in "${DSETS[@]}"; do
    [ -s "$PROJ/eval_artifacts/$d.allmodels.probes.json" ] && { echo "  [B] $d: probes scored"; continue; }
    id=$(sub $(dep "$ADEP") model/jobs/score_one.sbatch "$d")
    echo "  [B] score $d -> $id"; B+=("$id")
done
for d in "${LDS[@]}"; do
    [ -e "$SCRM_CORPUS_ROOT/$d/manifest.json" ] || continue
    ls "$PROJ/eval_artifacts/$d".allmodels.annotation*.json >/dev/null 2>&1 \
        && { echo "  [B] $d: annotated"; continue; }
    id=$(sub $(dep "$ADEP") model/jobs/annotate_one.sbatch "$d")
    echo "  [B] annotate $d -> $id"; B+=("$id")
done
BDEP=$(IFS=:; echo "${B[*]:-}")
[ -z "$BDEP" ] && BDEP="$ADEP"

for j in uce_protocols uce_their_eval arm_geometry collapse_trajectory; do
    id=$(sub $(dep "$BDEP") "model/jobs/$j.sbatch")
    echo "  [C] $j -> $id"
done
id=$(sub $(dep "$BDEP") model/jobs/measure_rest.sbatch)
echo "  [C] measure_rest (seed_floor, ckpt_choice, e4_vocab_reach, e7_vs_control, e7_isoform, paired_significance) -> $id"

cat <<'TXT'

  [D] gate and [E] render are not submitted by this script. Run them by hand and READ them:

        PYTHONPATH=model:eval:target/py python measurements/ablation.py --check     # must exit 0
        PYTHONPATH=model:eval:target/py python measurements/seed_floor.py --check   # must exit 0
        PYTHONPATH=model:eval:target/py python measurements/checkpoint_choice.py --check  # must exit 0
        PYTHONPATH=model:eval:target/py python measurements/paired_significance.py --write
        PYTHONPATH=model:eval:target/py python measurements/figure_species.py -o <paper>/figures/species_data.tex
        PYTHONPATH=model:eval:target/py python measurements/results_page.py --write-tex $SCRM_PROJ/generated_oss

  `ablation.py --check` exits non-zero until every member of the study exists, and
  `seed_floor.py --check` fails whenever SEED_FLOOR sits below the widest fully-scored reseed
  spread.
TXT
