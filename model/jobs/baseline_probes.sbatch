#!/bin/bash
#SBATCH --job-name=scrm-baseprobes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:10G

if [ -z "${SLURM_JOB_ID:-}" ]; then >&2 echo "Run with: sbatch $0"; exit 1; fi
set -euo pipefail
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
export PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py
export LD_LIBRARY_PATH=$(dirname "$(dirname "${SCRM_PY:-python3}")")/lib

for ds in mouse_brain_5k neuron_1k_v3 neurons_5k nuclei_2k sod1_spinal_cord; do
    out=$PROJ/eval_artifacts/$ds.baselines.probes.json
    if [ -s "$out" ]; then echo "SKIP $ds"; continue; fi
    srun "${SCRM_PY:-python3}" -m scrm_eval.run_probes \
        --artifacts $PROJ/eval_artifacts/$ds.*.arrow \
        --dataset "$ds" \
        --gmt $PROJ/gene_sets/mh.all.v2024.1.Mm.symbols.gmt \
        -o "$out"
done
echo baseline probes complete
