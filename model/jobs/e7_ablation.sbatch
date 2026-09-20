#!/bin/bash
#SBATCH --job-name=scrm-e7-ablation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:10G
set -uo pipefail
cd "${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
export PYTHONPATH=model:eval:target/py
PY=${SCRM_PY:-python3}
PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
R=$PROJ/eval_reports
srun $PY measurements/e7_vs_control.py -o $R/e7_vs_control.crispr_a549_5k.json; rc1=$?
echo "e7_vs_control rc=$rc1"
[ $rc1 -eq 0 ]
