#!/bin/bash
#SBATCH --job-name=scrm-pin-corpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --requeue
set -uo pipefail
ROOT=${1:?usage: sbatch $0 <corpus root> [--write|--report]}
MODE=${2:-}
PY=${SCRM_PY:-python3}
cd "${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
echo "pin_corpus: root=$ROOT mode=${MODE:-verify} cpus=${SLURM_CPUS_PER_TASK:-?}"
srun $PY crates/scrm-corpus/tests/pinned_digests.py "$ROOT" $MODE
