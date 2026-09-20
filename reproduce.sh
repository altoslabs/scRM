#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
REPO=$PWD
PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
PY=${SCRM_PY:?set SCRM_PY to the interpreter of the environment envs/scrm.yml pins (absolute path)}
export SCRM_PY="$PY"
# corpus_rel is the corpus root this code builds.
export SCRM_CORPUS_ROOT=${SCRM_CORPUS_ROOT:-$PROJ/corpus_rel}
export CORPORA_ROOT=$SCRM_CORPUS_ROOT
if [ -z "${LD_LIBRARY_PATH:-}" ]; then
    PYBIN=$(command -v "$PY" 2>/dev/null)
    [ -n "$PYBIN" ] || { printf 'refusing: SCRM_PY=%s names no executable on PATH\n' "$PY" >&2; exit 1; }
    PYLIB=$(dirname "$(dirname "$PYBIN")")/lib
    case "$PYLIB" in
        /*) ;;
        *)  printf "refusing: the interpreter's lib/ resolved to the RELATIVE path '%s' (from\n" "$PYLIB" >&2
            printf "  SCRM_PY='%s' -> '%s'). Set SCRM_PY to an absolute interpreter path.\n" "$PY" "$PYBIN" >&2
            exit 1 ;;
    esac
    [ -d "$PYLIB" ] || {
        printf "refusing: '%s' does not exist, so LD_LIBRARY_PATH would name nothing (from\n" "$PYLIB" >&2
        printf "  SCRM_PY='%s' -> '%s'). Measured: the model suite then fails two test_ddp_parity\n" "$PY" "$PYBIN" >&2
        printf "  tests with scipy's \"CXXABI_1.3.15 not found\", which is the environment, not the code.\n" >&2
        exit 1
    }
    export LD_LIBRARY_PATH=$PYLIB
fi
export PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py
export SCRM_REPO=$REPO
PAPER=${SCRM_PAPER:-../scrm-paper}

export SBATCH_ACCOUNT=${SCRM_SLURM_ACCOUNT:?set SCRM_SLURM_ACCOUNT to your slurm account}
export SBATCH_PARTITION=${SCRM_SLURM_PARTITION:?set SCRM_SLURM_PARTITION to your partition}
[ -n "${SCRM_LOG_DIR:-}" ] && export SBATCH_OUTPUT="$SCRM_LOG_DIR/%x-%j.out"

DRY=0; STAGES=()
for a in "$@"; do
    case "$a" in
        --dry-run|-n) DRY=1 ;;
        all) STAGES=(0 1 2 3 4 5 6) ;;
        [0-6]) STAGES+=("$a") ;;
        *) echo "usage: $0 [--dry-run] {0..6|all}" >&2; exit 2 ;;
    esac
done
[ ${#STAGES[@]} -gt 0 ] || { echo "usage: $0 [--dry-run] {0..6|all}" >&2; exit 2; }

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
run()  { printf '   $ %s\n' "$*"; [ "$DRY" = 1 ] && return 0; "$@"; }
fail() { printf '\033[1;31m   FAILED: %s\033[0m\n' "$*" >&2; exit 1; }
note() { printf '   %s\n' "$*"; }
# Read-only. Runs under --dry-run; announces itself on stderr, leaving stdout to the caller.
query() { printf '   ? %s\n' "$*" >&2; "$@"; }
# Printed, never run.
manual() { printf '   \033[1mMANUAL\033[0m (this script does NOT run it): %s\n' "$*"; }

banner() {
    cat <<'TXT'

reproduce.sh is a PER-STAGE SUBMITTER AND PLAN PRINTER. It does NOT sequence the stages.
  * No stage passes --dependency to the next, so a stage sbatches its jobs and RETURNS while they
    are still queued. Run ONE stage, wait for its jobs to land (squeue --me), then run the next.
  * Re-run a stage until it reports nothing left to submit: each stage skips what is already on
    disk, so convergence is by repetition, not by waiting.
  * The pin and the training are MANUAL. Stage 2 submits the corpus builds but only PRINTS the two
    pin_corpus.sbatch commands; stage 3 submits nothing at all. The MANUAL lines are the commands
    you run yourself.
  * Stage 5 is the one real DAG: model/jobs/step2_suite.sh chains embed -> score -> measure with
    --dependency inside itself, and checks its preconditions before submitting anything.
  * Stage 6 needs the private paper tree and refuses without it.
TXT
}

stage0() {
    say "0. CODE GATE -- rustfmt, clippy, cargo test, and all four pytest suites"
    echo "   check.sh is the single entrypoint. It needs no corpus, no GPU and no compiled extension."
    run ./check.sh || fail "stage 0: the code does not pass its own gate"
}

stage1() {
    say "1. ACQUIRE -- verify every one of the paper's 20 corpora can be acquired"
    echo "   model/jobs/corpora.tsv is the declaration; verify_acquisition.py reads it and resolves"
    echo "   each row's whitelist SCHEME. There is no separate fetch stage: the one executor in"
    echo "   stage 2 fetches its own reads, so acquisition is verified here and performed there."
    run $PY measurements/verify_acquisition.py || fail "stage 1: acquisition cannot be verified"
}

stage2() {
    say "2. CORPUS -- build all 20, then PIN, in that order"
    echo "   Pin THEN embed: controls.corpus_facts reads the stream digest out of the fixture, so"
    echo "   an UNPINNED corpus cannot be stamped at all."
    query $PY measurements/verify_acquisition.py --plan \
        | grep -E '^CORPUS' | while IFS='|' read -r _ launcher key; do
        echo "   # $key"
        run sbatch model/jobs/$launcher "$key"
    done
    note "The builds above do not block the pin, and the pin is NOT submitted by this script."
    note "Re-run this stage until it submits nothing, then, with every build landed, run:"
    manual "sbatch model/jobs/pin_corpus.sbatch $SCRM_CORPUS_ROOT --report   # read-only, asks what WOULD change"
    manual "sbatch model/jobs/pin_corpus.sbatch $SCRM_CORPUS_ROOT --write    # commit must say which entries moved and why"
    note "A pinned-digest MISMATCH is not a reason to run --write: it means the pipeline changed"
    note "the data or the corpus is corrupt, and both need diagnosing."
    note "What is pinned in this tree TODAY is the corpora key of"
    note "crates/scrm-corpus/tests/expected_stream_digests.json -- all twenty, under one bounds key."
}

stage3() {
    say "3. TRAIN -- the registered runs (THIS STAGE SUBMITS NOTHING)"
    note "Every arm is one sbatch you run yourself. It is printed rather than submitted so that you"
    note "see arm_ablation.sbatch refuse an unpinned corpus here, not discover it six hours in."
    note "The arm list is registry.runs():"
    query $PY -m scrm_eval.registry --runs || fail "stage 3: cannot read the arm list"
    note ""
    note "arm_ablation.sbatch takes a run NAME. It resolves the recipe with"
    note "  \$PY -m scrm_eval.registry --recipe <run>   (FLAGSHIP_RECIPE + that run's declared delta)"
    note "and REFUSES: a corpus absent from expected_stream_digests.json, a shard directory with no"
    note "manifest.json, and a run whose \$SCRM_PROJ/runs/<run>/latest.pt already exists (pass"
    note "RESUME_LONGER=1 to train one further). It needs SCRM_CORPUS_ROOT exported -- its own"
    note "default is \$SCRM_PROJ/corpus, which is not the release's root."
    note ""
    note "The three runs score_one.sbatch BLOCKS on (registry.REQUIRED_RUNS):"
    for r in $(query $PY -m scrm_eval.registry --required); do
        manual "sbatch model/jobs/arm_ablation.sbatch $r"
    done
    note "The three UNTRAINED controls are not trained at all -- they are random init. Run this"
    note "AFTER the flagship: it copies the architecture out of the flagship's own checkpoint and"
    note "exits 'does not exist' without it. Drop --write to check and write nothing."
    manual "$PY measurements/untrained_control.py --write"
    note "Every other OPTIONAL run is the same one-name sbatch:"
    for r in $(query $PY -m scrm_eval.registry --optional); do
        case "$r" in
            *untrained*) continue ;;
            *) manual "sbatch model/jobs/arm_ablation.sbatch $r" ;;
        esac
    done
    note ""
    note "Nothing waits for these. Stage 5 CHECKS them: step2_suite.sh refuses to submit until every"
    note "run in registry.runs() has its reported checkpoint at its declared step count."
}

stage4() {
    say "4. BASELINES -- count controls, the 4 count FMs, and UCE"
    echo "   embedder_equiv FIRST: it measures that the count-FM embedder and the weights are"
    echo "   the ones that produced the panel."
    run sbatch model/jobs/embedder_equiv.sbatch
    run sbatch model/jobs/baselines_matrix.sbatch
    run sbatch model/jobs/uce_matrix.sbatch
    echo "   Then, before any scoring reads them:"
    run $PY measurements/baseline_population.py --check \
        || echo "   (advisory: stale baselines listed above -- see the rebuild commands it printed)"
}

stage5() {
    say "5. EVAL -- embed, score, measure, as one dependency graph"
    echo "   step2_suite.sh checks its preconditions BEFORE submitting anything."
    run ./model/jobs/step2_suite.sh || fail "stage 5: the eval DAG refused to submit"
}

paper_required() {
    [ "$DRY" = 1 ] && return 0
    [ -d "$PAPER" ] && return 0
    printf '\033[1;31m   NOT PUBLICLY REPRODUCIBLE: stage 6 needs the paper repository\033[0m\n' >&2
    cat >&2 <<MSG
   Looked for: $PAPER   (set SCRM_PAPER, or default ../scrm-paper)

   That is a separate, PRIVATE repository: main.tex, generated/, tools/derive_appendix_numbers.py,
   tools/verify_generated.py. This repository cannot supply it, so stage 6 cannot run here.

   Stages 0-5 need nothing private and produce every report stage 6 typesets. The numbers
   themselves are in the stamped reports on disk and on the page:

       $PY measurements/results_page.py --write-tex <dir>  # every number the paper shows

   Use --dry-run to see what stage 6 would submit.
MSG
    exit 1
}

stage6() {
    say "6. PAPER -- write every generated table, then verify none has drifted"
    paper_required
    echo "   The paper's tables are \\input{generated/<name>} fragments; the producers write them;"
    echo "   --check-tex fails on drift."
    run $PY measurements/results_page.py --write-tex $PAPER/generated \
        || fail "stage 6: a generated table has cells no report can produce (see [tex] lines above)"
    run $PY measurements/ablation.py --write-tex $PAPER/generated \
        || fail "stage 6: the ablation study has a cell no report can produce"
    run $PY measurements/figure_species.py -o $PAPER/figures/species_data.tex \
        || fail "stage 6: Figure 2's body could not be derived (a plotted model has no row)"
    run $PY measurements/verify_acquisition.py --write-tex $PAPER/generated \
        || fail "stage 6: tab:dataurls has a cell no manifest can produce"
    run $PY measurements/paired_significance.py --write --write-tex $PAPER/generated \
        || fail "stage 6: the sign test behind sec:identity could not be derived"
    run $PY measurements/e7_isoform_structure.py --write-tex $PAPER/generated \
        || fail "stage 6: app:perturbation's isoform macros could not be derived"
    run $PY measurements/umi_compression.py -o $PAPER/generated \
        || fail "stage 6: app:pipeline_detail's compression macros could not be derived"
    echo "   Appendix numbers:"
    run $PY $PAPER/tools/derive_appendix_numbers.py
    say "6b. VERIFY -- the paper's committed fragments ARE these numbers"
    run $PY measurements/results_page.py --check-tex $PAPER/generated \
        || fail "stage 6b: the paper carries a number this workflow does not produce"
    run $PY measurements/ablation.py --check-tex $PAPER/generated \
        || fail "stage 6b: the appendix carries an ablation number this workflow does not produce"
    run $PY measurements/verify_acquisition.py --check-tex $PAPER/generated \
        || fail "stage 6b: tab:dataurls has drifted from the manifests"
    note "signif_n, e7iso_n and compression_n are macro fragments: they are WRITTEN above"
    note "but have no --check-tex, so a stale one cannot be caught here. Rewriting them"
    note "every run is what keeps them current; do not hand-edit them in the paper."
    echo "   Both checks also validate STRUCTURE -- column count, brace balance, row terminator."
}

stage6c() {
    say "6c. BUILD THE PAPER and verify every generated number is ON THE PAGE"
    paper_required
    echo "   A fragment can be well-formed and not typeset, so the check reads the numbers out of"
    echo "   the built PDF."
    export PATH=${SCRM_TEX_BIN:-}:$PATH
    run tectonic $PAPER/main.tex --outdir $PAPER/build --keep-logs \
        || fail "stage 6c: the paper does not build"
    run $PY $PAPER/tools/verify_generated.py $PAPER/build/main.pdf \
        || fail "stage 6c: a generated number is not on the page, or a cell renders as ??"
}

banner
for s in "${STAGES[@]}"; do "stage$s"; [ "$s" = 6 ] && stage6c; done
say "done: stages ${STAGES[*]}${DRY:+ (dry run)}"
note "Stages were run in order but NOT chained: each returned as soon as it had submitted. Check"
note "squeue --me, then re-run the stages above until each submits nothing."
