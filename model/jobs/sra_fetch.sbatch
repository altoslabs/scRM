#!/bin/bash
#SBATCH --job-name=scrm-sra-fetch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:200G
set -euo pipefail
SRR=${1:?usage: sbatch $0 <SRR accession> <dest dir under raw/>}
DEST=${2:?dest dir name under $SCRM_PROJ/raw}
BIN=$(dirname "${SCRM_PY:-python3}")
PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
D=$PROJ/raw/$DEST
T=${TMPDIR:-/tmp}
mkdir -p "$D"

mkdir -p "$D"
n=0
for f in $(find "$D" -name "${SRR}_*.fastq.gz" 2>/dev/null); do
    if pigz -t "$f" 2>/dev/null; then
        n=$((n+1))
    else
        echo "CORRUPT $f -- deleting so this run refetches it"
        rm -f "$f"
    fi
done
if [ "$n" -ge 2 ]; then echo "already have $n intact ${SRR} fastq.gz under $D"; exit 0; fi

mkdir -p "$T/fqtmp"
cd "$T"
SRADIR=$PROJ/work/sra/$SRR
mkdir -p "$SRADIR"
srun $BIN/prefetch --max-size 200G -O "$SRADIR" "$SRR"
SRAFILE=$(find "$SRADIR" -name "$SRR.sra" | head -1)
[ -s "$SRAFILE" ] || { echo "prefetch left no $SRR.sra under $SRADIR"; exit 1; }
echo "sra: $SRAFILE ($(du -h "$SRAFILE" | cut -f1)); job scratch now holds only the FASTQs"
srun $BIN/fasterq-dump --split-files --include-technical -e 8 \
     --disk-limit-tmp 180G --disk-limit 180G \
     -t "$T/fqtmp" -O "$T" "$SRAFILE"
ls -la "$T"/${SRR}*.fastq

got=$(ls -1 "$T"/${SRR}_*.fastq 2>/dev/null | wc -l)
[ "$got" -ge 2 ] || { echo "fasterq-dump produced $got read file(s); expected >= 2 for 10x paired data"; exit 1; }

for f in "$T"/${SRR}_*.fastq; do
    b=$(basename "$f")
    srun pigz -p "${SLURM_CPUS_PER_TASK:-8}" -c "$f" > "$D/${b}.gz.part"
    pigz -t "$D/${b}.gz.part" || { echo "the gz just written does not decompress: $b"; rm -f "$D/${b}.gz.part"; exit 1; }
    mv "$D/${b}.gz.part" "$D/${b}.gz"
    rm -f "$f"
done
rm -rf "$SRADIR"
echo "wrote: $(find "$D" -name "${SRR}_*.fastq.gz" -printf '%f %s\n')"
