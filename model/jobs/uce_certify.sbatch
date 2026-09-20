#!/bin/bash
#SBATCH --job-name=scrm-uce-certify
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:120G
set -euo pipefail
PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
PY=${SCRM_PY:-python3}
export PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py:$PROJ/envs/uce-pylib
export LD_LIBRARY_PATH=$(dirname "$(dirname "${SCRM_PY:-python3}")")/lib
STAGE=${TMPDIR:-/tmp}/uce_export
mkdir -p "$STAGE"
cd "$STAGE"

FID=1O9a0UkqPHT_LOXxf695c1J-SyHL47eGI
echo "[certify] resolving the Drive confirm form"
PAGE=$(curl -sL --max-time 60 "https://drive.google.com/uc?id=$FID&export=download")
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
URL=$(printf '%s' "$PAGE" | $PY resolve_drive.py)
case "$URL" in http*) ;; *) echo "[certify] could not resolve a download URL: $URL"; exit 1;; esac
echo "[certify] downloading export_data to $STAGE (node-local)"
curl -sL --max-time 7200 -o export_data.tar.gz "$URL"
ls -la export_data.tar.gz
tar xzf export_data.tar.gz
find . -name "*scib*" -o -name "*tabula*" | head -20

$PY $REPO/measurements/uce_certify.py --export-data "$STAGE" \
    --out "$PROJ/eval_reports/uce/certify_tabula.json"
echo "done: certification report written"
