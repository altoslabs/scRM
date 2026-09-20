#!/bin/bash
#SBATCH --job-name=scrm-merge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:10G
set -uo pipefail
DS=${1:?usage: sbatch merge_shards.sbatch <dataset> <n_shards>}
NSH=${2:?n_shards}
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
export SCRM_REPO="$REPO"
export PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py
export LD_LIBRARY_PATH=$(dirname "$(dirname "${SCRM_PY:-python3}")")/lib
srun "${SCRM_PY:-python3}" - "$DS" "$NSH" <<'PYEOF'
import sys, glob
from pathlib import Path
import os
sys.path[:0] = [os.environ["SCRM_REPO"] + p for p in ("/model", "/eval", "/target/py")]
from scrm_eval import embed as ev

ds, nsh = sys.argv[1], int(sys.argv[2])
from scrm_eval import registry as _R
runs = list(_R.runs())
proj = _R.PROJ
ok = missing = 0
for run in runs:
    out = proj / "eval_runs" / run / f"{ds}.{run}.allreads.arrow"
    if out.exists():
        print(f"  {run}: whole artifact already present", flush=True); ok += 1; continue
    shards = sorted(glob.glob(str(proj / "eval_runs" / run / f"{ds}.{run}.shard*-of-{nsh}.arrow")))
    if len(shards) != nsh:
        print(f"  {run}: {len(shards)}/{nsh} shards -- skipping, merge_shards refuses a partial cover",
              flush=True)
        missing += 1
        continue
    ev.merge_shards(shards, out, dataset=ds)
    ok += 1
print(f"merged {ok}/{len(runs)} models for {ds}; {missing} still incomplete", flush=True)
sys.exit(1 if missing else 0)
PYEOF
