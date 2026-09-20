#!/bin/bash
#SBATCH --job-name=scrm-uce-rat-species
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:40G
set -uo pipefail
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
PY=${SCRM_PY:-python3}
PE_DIR=$PROJ/fm/uce/model_files/protein_embeddings
OUT=$PROJ/fm/uce/species_build
PE=$PE_DIR/Rattus_norvegicus.gene_symbol_to_embedding_ESM2.pt
FASTA=$OUT/Rattus_norvegicus.mRatBN7.2.pep.all.fa   # fly keeps its FASTA here too
mkdir -p "$OUT" "$PE_DIR"
cd "$OUT"

cat > resolve_drive.py <<'PYEOF'
import re, sys
from urllib.parse import urlencode
page = sys.stdin.read()
m = re.search(r'action="([^"]+)"', page)
if not m:
    raise SystemExit("no confirm form in the Drive page (quota, or the sharing changed)")
fields = dict(re.findall(r'name="([^"]+)" value="([^"]*)"', page))
print(m.group(1) + "?" + urlencode(fields))
PYEOF

if [ ! -s "$PE" ]; then
    FID=1mY4tgVxZqZkS7nzj7zsbp_WMoSzWZoVP
    echo "[rat] resolving Drive confirm form for the ESM2 embeddings"
    curl -sL -c ck.txt -b ck.txt --max-time 120 \
         "https://drive.google.com/uc?id=$FID&export=download" -o page.html || exit 1
    URL=$($PY resolve_drive.py < page.html) || exit 1
    case "$URL" in http*) ;; *) echo "[rat] could not resolve a URL: $URL"; exit 1;; esac
    echo "[rat] downloading (resumable)"
    curl -L -c ck.txt -b ck.txt --max-time 3600 -C - "$URL" -o "$PE.part" || exit 1
    SZ=$(stat -c%s "$PE.part")
    [ "$SZ" -gt 50000000 ] || { echo "[rat] downloaded only $SZ bytes -- Drive served an error page"; exit 1; }
    $PY -c "
import torch,sys
d=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
print(f'[rat] embeddings load OK: {len(d):,} gene symbols')" "$PE.part" || exit 1
    mv "$PE.part" "$PE"
fi
echo "[rat] embeddings: $PE ($(stat -c%s "$PE") bytes)"

if [ ! -s "$FASTA" ]; then
    echo "[rat] fetching the mRatBN7.2 proteome (Ensembl 110)"
    curl -sL --max-time 1800 -o "$FASTA.gz" \
        "https://ftp.ensembl.org/pub/release-110/fasta/rattus_norvegicus/pep/Rattus_norvegicus.mRatBN7.2.pep.all.fa.gz" \
        || { echo "[rat] proteome download failed"; exit 1; }
    gunzip -f "$FASTA.gz" || exit 1
fi
echo "[rat] proteome: $(grep -c '^>' "$FASTA") protein records"

$PY $REPO/measurements/uce_add_species.py --species rat --taxonomy-id 10116 \
    --fasta "$FASTA" --embeddings "$PE" --assembly mRatBN7.2 --out-dir "$OUT" || exit 1

CSV=$PROJ/fm/uce/model_files/new_species_protein_embeddings.csv
grep -q "^rat," "$CSV" || echo "rat,$PE" >> "$CSV"
echo "[rat] registered in $CSV:"; grep "^rat," "$CSV"
echo "[rat] DONE. Next: ONE_DS=2500_wistar_rat_pbmcs_singleplex_3p_gem_x sbatch --array=0 model/jobs/uce_matrix.sbatch"
