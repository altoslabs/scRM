#!/bin/bash

set -uo pipefail
cd "$(dirname "$0")"

PY=${SCRM_PY:-python3}
# Resolve the interpreter before deriving anything from its path.
PYBIN=$(command -v "$PY" 2>/dev/null)
if [ -z "$PYBIN" ]; then
    >&2 echo "refusing: SCRM_PY=${SCRM_PY:-<unset, defaulted to python3>} names no executable"
    >&2 echo "  on PATH, so nothing below could run. Point SCRM_PY at the interpreter of the"
    >&2 echo "  environment envs/scrm.yml pins, e.g."
    >&2 echo "    SCRM_PY=<micromamba envs dir>/scrm/bin/python ./check.sh"
    exit 1
fi
PYLIB=$(dirname "$(dirname "$PYBIN")")/lib
case "$PYLIB" in
    /*) ;;
    *)  >&2 echo "refusing: the interpreter's lib/ resolved to the RELATIVE path '$PYLIB'"
        >&2 echo "  (from SCRM_PY='$PY' -> '$PYBIN'). LD_LIBRARY_PATH would then name nothing and"
        >&2 echo "  the model suite fails on scipy's CXXABI_1.3.15. Set SCRM_PY to an absolute path."
        exit 1 ;;
esac
if [ ! -d "$PYLIB" ]; then
    >&2 echo "refusing: '$PYLIB' does not exist, so LD_LIBRARY_PATH would name nothing"
    >&2 echo "  (from SCRM_PY='$PY' -> '$PYBIN'). The model suite then fails two test_ddp_parity"
    >&2 echo "  tests with scipy's 'CXXABI_1.3.15 not found', which is the environment, not the code."
    >&2 echo "  SCRM_PY must be the interpreter of the environment whose lib/ holds its libstdc++."
    exit 1
fi
export LD_LIBRARY_PATH=$PYLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/scrm-numba-cache-$(id -u)"
mkdir -p "$NUMBA_CACHE_DIR" || { echo "cannot create NUMBA_CACHE_DIR=$NUMBA_CACHE_DIR" >&2; exit 1; }
CORPUS_ROOT_RESOLVED=$("$PYBIN" -c 'import sys; sys.path[:0] = ["eval", "model"]
from scrm_eval import registry
print(registry.corpus_root())')
if [ -z "$CORPUS_ROOT_RESOLVED" ]; then
    >&2 echo "refusing: registry.corpus_root() printed nothing -- the import failed, and its error is"
    >&2 echo "  on stderr just above. A checker that cannot read the corpus root cannot report on one."
    exit 1
fi
# Did the root resolve to a real tree holding a corpus?
CORPUS_ROOT_OK=0
case "$CORPUS_ROOT_RESOLVED" in
    *'$'* | [!/]*)
        # The root came back unexpanded, or is not an absolute path.
        echo "corpus root UNRESOLVED: $CORPUS_ROOT_RESOLVED"
        echo "  That is not a path -- SCRM_PROJ is unset, so registry.corpus_root() returned its own"
        echo "  default with the variable name still in it. Nothing on disk is named that, so every"
        echo "  corpus-dependent test below SKIPS and this run checks CODE ONLY."
        echo "  To check a corpus: SCRM_CORPUS_ROOT=<corpus tree> ./check.sh <corpus dir> ..."
        ;;
    *)
        # Test the root for an actual corpus, not for its parent directory.
        if ! compgen -G "$CORPUS_ROOT_RESOLVED/*/manifest.json" >/dev/null 2>&1; then
            >&2 echo "corpus root holds no corpus: $CORPUS_ROOT_RESOLVED"
            >&2 echo "  cargo test --workspace WILL skip six corpus tests across scrm-pack and scrm-correct."
            >&2 echo "  Set one of SCRM_CORPUS_ROOT, CORPORA_ROOT, CORPUS_ROOT to the tree that holds the"
            >&2 echo "  corpora, e.g.  SCRM_CORPUS_ROOT=<corpus tree> ./check.sh"
            exit 1
        fi
# Name the tree under test: the commit, and whether the working tree is dirty.
SCRM_HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo "not-a-git-tree")
SCRM_DIRT=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
echo "tree: $SCRM_HEAD  uncommitted paths: $SCRM_DIRT"
if [ "$SCRM_DIRT" != "0" ]; then
    echo "  WARNING: the working tree is DIRTY. Whatever this run prints describes the files on"
    echo "  disk, not $SCRM_HEAD. Do not cite it for a commit; commit first, or run a clean checkout."
fi

        CORPUS_ROOT_OK=1
        echo "corpus root: $CORPUS_ROOT_RESOLVED"
        ;;
esac

fail=0
roots=""

step() {
    echo
    echo "=== $1 ==="
    shift
    if "$@"; then
        echo "  PASS"
    else
        echo "  FAIL: $*"
        fail=1
    fi
}

step "rustfmt (style_edition 2024, max_width 120)" cargo fmt --check
step "clippy, all targets" cargo clippy --workspace --all-targets -- -D warnings
# Six corpus tests print "SKIP:" and return instead of failing when their data is absent, so the
# harness runs with `-- --nocapture` and the SKIP lines are counted.
CARGO_SKIPS=0
cargo_test_workspace() {
    local log status
    log=$(mktemp "${TMPDIR:-/tmp}/scrm-cargo-test.XXXXXX") || { >&2 echo "cannot create a temp file"; return 1; }
    cargo test --workspace -- --nocapture 2>&1 | tee "$log"
    status=${PIPESTATUS[0]}
    CARGO_SKIPS=$(grep -c "SKIP:" "$log")
    if [ "$CARGO_SKIPS" -gt 0 ]; then
        echo
        echo "  $CARGO_SKIPS test(s) printed SKIP: instead of failing. Each returned early because its data"
        echo "  was absent, so it verified NOTHING and libtest still counted it as a pass:"
        grep "SKIP:" "$log" | sed "s/^/    /"
    fi
    rm -f "$log"
    [ "$status" -eq 0 ] || return "$status"
    if [ "$CARGO_SKIPS" -gt 0 ] && [ "$CORPUS_ROOT_OK" -eq 1 ]; then
        echo "  The corpus root RESOLVED to $CORPUS_ROOT_RESOLVED and holds a corpus, so nothing above"
        echo "  had a reason to skip. Counting $CARGO_SKIPS silent skip(s) as a FAILURE. Check that SCRM_PROJ"
        echo "  is set (real_barcode_list reads it, not the corpus root) and that the Python and Rust"
        echo "  roots agree: both read SCRM_CORPUS_ROOT, CORPORA_ROOT, CORPUS_ROOT, then SCRM_PROJ/corpus."
        return 1
    fi
    if [ "$CARGO_SKIPS" -gt 0 ]; then
        echo "  WARNING: this is the CODE-ONLY run announced above (the corpus root did not resolve), so"
        echo "  those $CARGO_SKIPS skip(s) are expected -- but they are NOT coverage. Nothing in this run"
        echo "  read a corpus. Re-run with SCRM_CORPUS_ROOT=<corpus tree> to actually check them."
    fi
    return 0
}

step "cargo test, whole workspace" cargo_test_workspace

step "pytest, the eval suite" \
    "$PY" -m pytest eval/tests -q -rs -p no:cacheprovider

step "pytest, the model suite" \
    "$PY" -m pytest model/tests -q -rs -p no:cacheprovider

step "pytest, the corpus checkers" \
    "$PY" -m pytest crates/scrm-corpus/tests -q -rs -p no:cacheprovider

if [ ! -f target/py/scrm_py.so ]; then
    echo
    echo "=== pytest, the Python packer API ==="
    echo "  FAIL: no target/py/scrm_py.so, so the only independent reimplementation of packing"
    echo "        would not run at all. Build it with:"
    echo "        cargo build --release -p scrm-py && mkdir -p target/py &&"
    echo "        cp target/release/libscrm_py.so target/py/scrm_py.so"
    fail=1
# scrm-py depends on scrm-core, scrm-pack and scrm-asm; scrm-pack depends on scrm-corpus.
elif [ -n "$(find crates/scrm-py/src crates/scrm-pack/src crates/scrm-core/src \
              crates/scrm-asm/src crates/scrm-corpus/src Cargo.lock \
              -type f -newer target/py/scrm_py.so -print -quit 2>/dev/null)" ]; then
    echo
    echo "=== pytest, the Python packer API ==="
    echo "  FAIL: target/py/scrm_py.so is OLDER than the packer sources it is built from, so this"
    echo "        suite would report on code that is not in the tree. Rebuild it with:"
    echo "        cargo build --release -p scrm-py && cp target/release/libscrm_py.so target/py/scrm_py.so"
    echo "  newer than the extension:"
    find crates/scrm-py/src crates/scrm-pack/src crates/scrm-core/src \
         -type f -newer target/py/scrm_py.so -printf "        %TY-%Tm-%Td %TH:%TM  %p\n" 2>/dev/null | sort | head -10
    fail=1
else
    step "pytest, the Python packer API" \
        env PYTHONPATH=target/py "$PY" -m pytest crates/scrm-py/tests -q -rs -p no:cacheprovider
fi

for corpus in "$@"; do
    name=$(basename "$corpus")
    if [ ! -f "$corpus/manifest.json" ]; then
        echo
        echo "=== $name ==="
        echo "  FAIL: $corpus has no manifest.json -- incomplete (still being written?), not a corpus"
        fail=1
        continue
    fi
    step "pyarrow_compat: $name" \
        "$PY" crates/scrm-corpus/tests/pyarrow_compat.py "$corpus"
    step "cell accounting: $name" \
        "$PY" crates/scrm-corpus/tests/cell_accounting.py "$corpus"
    roots="$roots$(dirname "$corpus")\n"
done

for root in $(printf '%b' "$roots" | sort -u); do
    step "pinned digests: $root" \
        "$PY" crates/scrm-corpus/tests/pinned_digests.py "$root"
done

if [ -n "${SCRM_PROJ:-}" ]; then
    step "seed floor vs registry.SEED_FLOOR" \
        env PYTHONPATH=model:eval:target/py "$PY" measurements/seed_floor.py --check
    step "probe sanity: a PCA of the counts beats the depth floor on every probe family" \
        env PYTHONPATH=model:eval:target/py "$PY" measurements/probe_sanity.py --check
else
    echo
    echo "=== seed floor vs registry.SEED_FLOOR ==="
    echo "  SKIPPED: SCRM_PROJ is not set, so there are no eval_reports to derive the floor from."
    echo "           SCRM_PROJ=<project root> ./check.sh  runs it."
    echo
    echo "=== probe sanity: a PCA of the counts beats the depth floor on every probe family ==="
    echo "  SKIPPED: SCRM_PROJ is not set, so there are no control artifacts to score."
    echo "           SCRM_PROJ=<project root> ./check.sh  runs it."
fi

echo
if [ "$#" -eq 0 ]; then
    echo "=== the corpus checkers ==="
    echo "  NOT RUN: pyarrow_compat, cell_accounting and pinned_digests run once per corpus named on"
    echo "           the command line, and none was."
    if [ "$CORPUS_ROOT_OK" -eq 1 ]; then
        echo "           The root resolved and holds a corpus, so any SKIP: the cargo step printed was"
        echo "           counted there as a FAILURE, never as coverage."
        echo "           The skip lines the pytest suites printed state their own reasons (a GPU, an"
        echo "           unset SCRM_PROJ) and are unrelated to these three checkers."
    else
        echo "           The corpus root did not resolve, so the $CARGO_SKIPS cargo SKIP: line(s) above and"
        echo "           the pytest \"no corpus root\" skips are the ones that wanted a corpus."
    fi
    echo "           ./check.sh <corpus dir> ...  runs them."
fi

echo
if [ "$fail" -eq 0 ]; then
    if [ "$#" -eq 0 ]; then
        echo "CODE CHECKS PASSED -- the corpus half did not run (see above)"
    else
        echo "ALL CHECKS PASSED"
    fi
else
    echo "SOMETHING FAILED -- see the FAIL lines above"
fi
exit "$fail"
