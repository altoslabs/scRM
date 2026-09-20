#!/bin/bash
#SBATCH --job-name=scrm-corpus-v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:50G

if [ -z "${SLURM_JOB_ID:-}" ]; then
    >&2 echo "Run with: sbatch $0"
    exit 1
fi
set -euo pipefail

REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
BIN=$REPO/target/release/scrm-corpus
PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
SHEET=$PROJ/samplesheet.csv
OUT=$PROJ/corpus_v2

mkdir -p "$OUT"
for exp in mouse_brain_5k neurons_5k sod1_spinal_cord neuron_1k_v3 nuclei_2k; do
    if [ -f "$OUT/$exp/manifest.json" ]; then
        echo "skip $exp: manifest exists"
        continue
    fi
    if [ -d "$OUT/$exp" ]; then
        echo "clearing partial $exp"
        rm -rf "${OUT:?}/${exp:?}"
    fi
    sub=$OUT/.sheet-$exp.csv
    head -1 "$SHEET" > "$sub"
    awk -F, -v e="$exp" 'NR > 1 && $1 == e' "$SHEET" >> "$sub"
    srun $BIN --samplesheet "$sub" --out "$OUT" --threads 32

done
echo "corpus_v2 complete"
