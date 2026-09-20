#!/bin/bash
#SBATCH --job-name=scrm-restamp
#SBATCH --array=0-31
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:10G

if [ -z "${SLURM_JOB_ID:-}" ]; then >&2 echo "Run with: sbatch $0"; exit 1; fi
set -euo pipefail
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
CORPORA_ROOT=${SCRM_CORPUS_ROOT:-${CORPUS_ROOT:-${CORPORA_ROOT:-$PROJ/corpus}}}
export CORPORA_ROOT SCRM_CORPUS_ROOT="$CORPORA_ROOT"
export PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py
export LD_LIBRARY_PATH=$(dirname "$(dirname "${SCRM_PY:-python3}")")/lib
cd "$REPO"
echo "corpus root: $CORPORA_ROOT"

srun "${SCRM_PY:-python3}" - <<'PYEOF'
import os
import traceback
from scrm_eval import artifact, controls, registry

ART = registry.PROJ / "eval_artifacts"
FM = registry.PROJ / "fm"
ART.mkdir(parents=True, exist_ok=True)

one = os.environ.get("DS") or None
todo = sorted(d for d in registry.count_matrices()
              if (registry.corpus_root() / d / "manifest.json").exists())
if one:
    if one not in todo:
        raise SystemExit(f"DS={one} has no count matrix or no built corpus under {registry.corpus_root()}")
    todo = [one]
else:
    hi = os.environ.get("SLURM_ARRAY_TASK_MAX")
    if hi is not None and int(hi) + 1 < len(todo):
        raise SystemExit(f"REFUSING: --array=0-{hi} covers {int(hi) + 1} tasks but {len(todo)} "
                         f"datasets have a count matrix and a corpus. Resubmit with "
                         f"--array=0-{len(todo) - 1} and bump the #SBATCH --array line.")
    i = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if i >= len(todo):
        print(f"task {i} is outside {len(todo)} datasets -- nothing to do")
        raise SystemExit(0)
    todo = [todo[i]]
print(f"dataset: {todo[0]}\n", flush=True)

failed: list[str] = []
for ds in sorted(todo):
    for producer in controls.BUILT_PRODUCERS:
        try:
            print(controls.build(ds, producer, ART), flush=True)
        except Exception as exc:
            failed.append(f"{ds}.{producer}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    npzs = [(p.name[len(ds) + 1:-4], p) for p in sorted(FM.glob(f"{ds}.*.npz"))]
    uce = FM / "uce" / "npz" / f"{ds}.uce.npz"
    if uce.exists():
        npzs.append(("uce", uce))
    for producer, path in npzs:
        try:
            print(controls.from_npz(ds, producer, path, ART), flush=True)
        except Exception as exc:
            failed.append(f"{ds}.{producer}: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    stamps = [(p.name, artifact.read(p).stamp) for p in sorted(ART.glob(f"{ds}.*.arrow"))]
    for _, s in stamps[1:]:
        artifact.same_corpus(stamps[0][1], s)
    counts = sorted({s.n_cells for _, s in stamps})
    print(f"{ds}: {len(stamps)} artifacts on one corpus content, cell counts {counts}\n", flush=True)

if failed:
    print("FAILED:")
    for line in failed:
        print(f"  {line}")
    raise SystemExit(1)
print("every baseline artifact re-derived against this corpus tree")
PYEOF
echo restamp complete
