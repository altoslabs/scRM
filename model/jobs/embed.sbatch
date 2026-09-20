#!/bin/bash
# Embed one arm over one dataset's corpus.
#
#   sbatch --partition=... --account=... model/jobs/embed.sbatch <dataset> [run]
#   SHARD=<i> NSHARD=<n> sbatch ... model/jobs/embed.sbatch <dataset> [run]
#
# run defaults to registry.FLAGSHIP. w_eval comes from registry.full_coverage(dataset) and the
# checkpoint from registry.reported_ckpt(run). One GPU per job. The corpus must be pinned:
# controls.corpus_facts reads its stream digest from the pinned fixture.
#
# SHARD/NSHARD embed one CONTIGUOUS cell range of the dataset, writing
# `<ds>.<run>.shard<i>-of-<n>.arrow` instead of the whole artifact. NSHARD must be one of
# 2, 3, 4 or 6, the counts score_one.sbatch merges. `embed.merge_shards` refuses a gap, an
# overlap or a partial cover.
#SBATCH --job-name=scrm-embed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:10G
set -euo pipefail

DS=${1:?usage: sbatch $0 <dataset> [run]}
NSHARD=${NSHARD:-}
SHARD=${SHARD:-}
if [ -n "$NSHARD" ] || [ -n "$SHARD" ]; then
    [ -n "$NSHARD" ] && [ -n "$SHARD" ] \
        || { >&2 echo "REFUSING: SHARD and NSHARD are one request; set both or neither"; exit 1; }
    case "$NSHARD" in
        2|3|4|6) ;;
        *) >&2 echo "REFUSING: NSHARD=$NSHARD; score_one.sbatch merges only 2, 3, 4 or 6 shards," \
                    "so any other count writes a set nothing will ever merge"; exit 1 ;;
    esac
    [ "$SHARD" -ge 0 ] && [ "$SHARD" -lt "$NSHARD" ] \
        || { >&2 echo "REFUSING: SHARD=$SHARD is out of range for NSHARD=$NSHARD"; exit 1; }
fi
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root}
PY=${SCRM_PY:-$(command -v python3 || true)}
[ -x "$PY" ] || { >&2 echo "REFUSING: no interpreter; set SCRM_PY to an absolute path"; exit 1; }
export LD_LIBRARY_PATH=$(cd "$(dirname "$(dirname "$PY")")/lib" && pwd)
export PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py
export SCRM_PROJ="$PROJ" SCRM_REPO="$REPO"
CORPORA_ROOT=${SCRM_CORPUS_ROOT:-${CORPUS_ROOT:-${CORPORA_ROOT:-$PROJ/corpus}}}
export CORPORA_ROOT SCRM_CORPUS_ROOT="$CORPORA_ROOT"
cd "$REPO"

srun "$PY" - "$DS" "${2:-}" "${SHARD:-}" "${NSHARD:-}" <<'PYEOF'
import os
import sys
from pathlib import Path

import scrm_py
from scrm_eval import artifact, embed as ev, registry
from scrm_model.evalio import model_factory
from scrm_model.run_eval import eval_ready

ds = sys.argv[1]
run = sys.argv[2] or registry.FLAGSHIP
proj = registry.PROJ

w = registry.full_coverage(ds)
print(f"{ds}: w_eval = full_coverage = {w}", flush=True)

shard = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
nshard = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None

out = proj / "eval_runs" / run
out.mkdir(parents=True, exist_ok=True)
if nshard is None:
    art = out / f"{ds}.{run}.allreads.arrow"
    cell_range = None
else:
    art = out / f"{ds}.{run}.shard{shard}-of-{nshard}.arrow"
    # A merge consumes its shards, so skip on the merged artifact rather than the shard path:
    # once `<ds>.<run>.allreads.arrow` exists and READS, this shard is already folded into it
    # and the shards themselves may be gone.
    merged = out / f"{ds}.{run}.allreads.arrow"
    if merged.is_file() and os.environ.get("FORCE", "0") != "1":
        try:
            artifact.read(merged)
        except Exception as exc:
            print(f"{merged} is on disk but unreadable ({type(exc).__name__}); embedding the shard",
                  flush=True)
        else:
            print(f"already merged -> {merged}; shard {shard}/{nshard} is not needed")
            raise SystemExit(0)
# Reuse an artifact only if it LOADS. FORCE=1 re-embeds regardless.
if art.is_file() and os.environ.get("FORCE", "0") != "1":
    try:
        artifact.read(art)
    except Exception as exc:
        print(f"re-embedding {art}: on disk but this tree cannot read it "
              f"({type(exc).__name__})", flush=True)
    else:
        print(f"already embedded -> {art}")
        raise SystemExit(0)

ckpt_src = proj / "runs" / run / registry.reported_ckpt(run)
if not ckpt_src.is_file():
    raise SystemExit(f"{ckpt_src} is absent; registry.reported_ckpt({run!r}) names it")

corpus_dir = registry.corpus_root() / ds
if not (corpus_dir / "manifest.json").is_file():
    raise SystemExit(f"{corpus_dir} has no manifest")

ckpt = ev.Checkpoint.load(eval_ready(ckpt_src, registry.recipe(run)["arm"], out),
                          model_factory=model_factory)
corpus = scrm_py.Corpus([str(corpus_dir)])
if nshard is not None:
    # Contiguous, exhaustive, ceil-divided: shard n-1 takes the remainder, so the ranges cover
    # every cell exactly once. merge_shards asserts that independently.
    n_cells = corpus.n_cells()
    per = -(-n_cells // nshard)
    lo, hi = shard * per, min((shard + 1) * per, n_cells)
    if lo >= hi:
        raise SystemExit(f"shard {shard} of {nshard} is empty for {n_cells} cells; use a smaller NSHARD")
    cell_range = (lo, hi)
    print(f"{ds}: shard {shard}/{nshard} = cells {lo}:{hi} of {n_cells}", flush=True)
ev.embed_to_file(ckpt, corpus, art, dataset=ds, w_eval=w, eval_seed=20260818,
                 cell_range=cell_range)
print(f"embedded -> {art}", flush=True)
PYEOF
echo "== embed complete: $DS"
