#!/bin/bash
#SBATCH --job-name=scrm-uce-theirdata
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:10G
set -uo pipefail
PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
PY=${SCRM_PY:-python3}
STAGE=$PROJ/work/uce_export
OUT=$PROJ/fm/uce/export_data
mkdir -p "$STAGE" "$OUT"
cd "$STAGE"

WANT=(
    'export_data/new_tabula_scib.h5ad'
    'export_data/brain_atlas_uce_scgpt_geneformer.h5ad'
    'export_data/fly_cell_atlas_uce_adata_33layer.h5ad'
    'export_data/fly_cell_atlas_uce_adata_33layer_umap.h5ad'
    'export_data/tabula_sapiens_subset_shuffled_uce_adata.h5ad'
    'export_data/tabula_sapiens_subset_unshuffled_uce_adata.h5ad'
)
have=1
for w in "${WANT[@]}"; do [ -s "$OUT/$(basename "$w")" ] || have=0; done
if [ "$have" = 1 ]; then echo "[their] all members already on Lustre; nothing to fetch"; exit 0; fi

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

FID=1O9a0UkqPHT_LOXxf695c1J-SyHL47eGI
resolve() {
    curl -sL -c ck.txt -b ck.txt --max-time 120 \
         "https://drive.google.com/uc?id=$FID&export=download" -o page.html || return 1
    $PY resolve_drive.py < page.html
}

echo "[their] resolving the Drive confirm form"
URL=$(resolve) || { echo "[their] could not fetch the confirm page"; exit 1; }
case "$URL" in http*) ;; *) echo "[their] could not resolve a download URL: $URL"; exit 1;; esac

probe() {
    local u; u=$(resolve) || return 1
    curl -sL -b ck.txt -c ck.txt -D hdr.txt --max-time 180 -r 0-1023 "$u" -o /dev/null || return 1
    tr -d '\r' < hdr.txt | grep -i '^content-range:' | sed -E 's#.*/([0-9]+)$#\1#' | head -1
}
EXPECT=$(probe || echo 0)
EXPECT=${EXPECT:-0}
echo "[their] server reports total $EXPECT bytes"
if [ "$EXPECT" -lt 100000000000 ]; then
    echo "[their] REFUSING: $EXPECT is not the export_data bundle -- Drive served an error or quota"
    echo "[their] page, or the ranged probe was itself refused. A per-file quota clears on its own."
    exit 1
fi

CHUNK=${CHUNK:-2000000000}
stall=0
while :; do
    have=$(stat -c %s export_data.tar.gz 2>/dev/null || echo 0)
    [ "$have" -ge "$EXPECT" ] && { echo "[their] have all $have bytes"; break; }
    end=$(( have + CHUNK - 1 )); [ "$end" -ge "$EXPECT" ] && end=$(( EXPECT - 1 ))
    URL=$(resolve) || { echo "[their] resolve failed; waiting"; sleep 120; stall=$((stall+1)); }
    if [ -n "${URL:-}" ]; then
        rm -f chunk.bin
        curl -sL -b ck.txt -c ck.txt --max-time 1800 -r "$have-$end" "$URL" -o chunk.bin || true
        got=$(stat -c %s chunk.bin 2>/dev/null || echo 0)
        want=$(( end - have + 1 ))
        if [ "$got" -eq "$want" ]; then
            cat chunk.bin >> export_data.tar.gz
            rm -f chunk.bin
            stall=0
            pct=$(( (have + got) * 100 / EXPECT ))
            echo "[their] $(( (have + got) / 1000000000 )) / $(( EXPECT / 1000000000 )) GB (${pct}%)"
        else
            echo "[their] chunk at $have: got $got of $want -- refused; retrying"
            [ "$got" -gt 0 ] && [ "$got" -lt 4096 ] && head -c 120 chunk.bin | tr -d '\n' && echo
            rm -f chunk.bin
            stall=$((stall+1))
            sleep $(( stall < 10 ? 60 : 300 ))
        fi
    fi
    if [ "$stall" -ge 40 ]; then
        echo "[their] 40 consecutive refusals -- Drive's per-file quota is closed to us right now."
        echo "[their] $(stat -c %s export_data.tar.gz 2>/dev/null || echo 0) bytes are on Lustre at"
        echo "[their] $STAGE and resubmitting this job resumes from there. The quota clears on its own."
        exit 1
    fi
done

have=$(stat -c %s export_data.tar.gz 2>/dev/null || echo 0)
if [ "$have" -lt "$EXPECT" ]; then
    echo "[their] INCOMPLETE: $have of $EXPECT bytes after 24 attempts. The partial file is on"
    echo "[their] node-local NVMe and dies with this job, so resubmitting restarts -- if this keeps"
    echo "[their] happening, stage to Lustre instead and trade IO for resumability."
    exit 1
fi
magic=$(head -c 2 export_data.tar.gz | od -An -tx1 | tr -d ' ')
[ "$magic" = "1f8b" ] || { echo "[their] not gzip (magic $magic) -- refusing to extract"; exit 1; }

echo "[their] extracting the six members the eval notebooks load"
tar xzf export_data.tar.gz -C "$OUT" --strip-components=1 --wildcards --no-anchored \
    'new_tabula_scib.h5ad' 'brain_atlas_uce_scgpt_geneformer.h5ad' \
    'fly_cell_atlas_uce_adata_33layer.h5ad' 'fly_cell_atlas_uce_adata_33layer_umap.h5ad' \
    'tabula_sapiens_subset_shuffled_uce_adata.h5ad' \
    'tabula_sapiens_subset_unshuffled_uce_adata.h5ad' \
  || echo "[their] tar reported a problem; listing what did land"
ls -la "$OUT"

missing=0
for w in "${WANT[@]}"; do
    b=$(basename "$w")
    if [ -s "$OUT/$b" ]; then echo "  OK      $b  $(du -h "$OUT/$b" | cut -f1)"
    else echo "  MISSING $b"; missing=$((missing+1)); fi
done
echo "[their] $missing of ${#WANT[@]} members missing"
echo "[their] NEXT: measurements/uce_their_eval.py runs their protocols on these and compares"
echo "[their]       each number against the value stored in their own notebook outputs."
