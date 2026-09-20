#!/bin/bash
#SBATCH --job-name=scrm-mouse-index
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:50G
set -euo pipefail
PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
D=$PROJ/quant_mouse
mkdir -p "$D"
FA=$D/Mus_musculus.GRCm39.cdna.all.fa.gz
[ -s "$FA" ] || curl -sS --max-time 3600 -o "$FA" \
  "https://ftp.ensembl.org/pub/release-112/fasta/mus_musculus/cdna/Mus_musculus.GRCm39.cdna.all.fa.gz"
ls -la "$FA"
export PATH=${SCRM_SIMPLEAF_BIN:-simpleaf}:${SCRM_AF_BIN:-alevin-fry}:$PATH
[ -d "$D/idx/index" ] || srun ${SCRM_SIMPLEAF_BIN:-simpleaf}/piscem build \
    -s "$FA" -k 31 -m 19 -t 32 -o "$D/idx/index"
ls -la "$D/idx/"
echo "mouse index complete"
