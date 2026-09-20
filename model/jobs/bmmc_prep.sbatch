#!/bin/bash
#SBATCH --job-name=scrm-bmmc-prep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:20G
set -euo pipefail
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
RAW=$PROJ/raw
H5=$RAW/bmmc_neurips/multiome_BMMC_processed.h5ad.gz
URL=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz

mkdir -p "$(dirname "$H5")"
if [ ! -s "$H5" ]; then
    echo "downloading $URL"
    curl -sSL --fail --retry 5 --retry-delay 10 --max-time 10800 -o "$H5.part" "$URL"
    mv "$H5.part" "$H5"
fi
ls -la "$H5"

PLAIN=${H5%.gz}
if [ ! -s "$PLAIN" ]; then
    echo "decompressing to $PLAIN"
    gunzip -c "$H5" > "$PLAIN.part"
    mv "$PLAIN.part" "$PLAIN"
fi
ls -la "$PLAIN"

export LD_LIBRARY_PATH=$(dirname "$(dirname "${SCRM_PY:-python3}")")/lib
srun "${SCRM_PY:-python3}" \
    $REPO/measurements/bmmc_prep.py "$PLAIN" "$RAW"
echo "bmmc prep done"
