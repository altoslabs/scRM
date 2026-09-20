#!/bin/bash
#SBATCH --job-name=scrm-embedder-equiv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1,disk:tmp:20G
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --requeue
set -uo pipefail
PROJ=$SCRM_PROJ
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
DS=pbmc_1k_v2          # 996 cells, human -> no ortholog bridge
M=geneformer
OUT=$PROJ/fm/.equiv-check/$DS.$M.npz
mkdir -p "$(dirname "$OUT")"
# The embedder under test is the one in the release tree.
REF=$PROJ/fm/$DS.$M.npz
echo "count-FM embedder: $REPO/eval/countfm/embed_count_fm.py"
md5sum "$REPO/eval/countfm/embed_count_fm.py"
# Checked before the GPU run: the comparison is against the published panel npz, which is data
# and is not in the release.
if [ ! -s "$REF" ]; then
    echo "no published embedding to compare against at $REF."
    echo "This job measures that the embedder + the fetched weights REPRODUCE our panel; the"
    echo "panel npz is an output, so a fresh clone has none. Verify the weights themselves with:"
    echo "  \$SCRM_PY_SCFM eval/countfm/fetch.py --weights-root \$SCRM_PROJ/countfm/scfm_weights --check"
    exit 1
fi
srun "${SCRM_PY_SCFM:-python3}" \
    $REPO/eval/countfm/embed_count_fm.py \
    --model "$M" --matrix "$PROJ/fm/cells_$DS" --out-npz "$OUT" \
    --weights-root "$PROJ/countfm/scfm_weights" --device cuda --batch-size 32 || exit 1

"${SCRM_PY:-python3}" - "$REF" "$OUT" <<'PYEOF'
import sys
import numpy as np
old, new = (np.load(p, allow_pickle=True) for p in sys.argv[1:3])
bo = [str(b) for b in old["barcode"]]; bn = [str(b) for b in new["barcode"]]
print(f"barcodes: old {len(bo)}  new {len(bn)}  identical order: {bo == bn}")
if bo != bn:
    print(f"  set equal: {set(bo) == set(bn)}")
eo, en = np.asarray(old["emb"]), np.asarray(new["emb"])
print(f"emb: old {eo.shape} new {en.shape}")
if eo.shape != en.shape:
    print("SHAPE MISMATCH -- the rescued code is NOT the code that produced the panel"); raise SystemExit(1)
pos = {b: i for i, b in enumerate(bn)}
common = [b for b in bo if b in pos]
a = eo[[bo.index(b) for b in common]]; b_ = en[[pos[b] for b in common]]
d = np.abs(a - b_)
print(f"compared {len(common)} cells x {eo.shape[1]} dims")
print(f"  bitwise identical: {np.array_equal(a, b_)}")
print(f"  max |diff| = {d.max():.3e}   mean |diff| = {d.mean():.3e}")
print(f"  max |old|  = {np.abs(a).max():.3e}")
print("VERDICT:", "SAME CODE+WEIGHTS" if d.max() < 1e-4 else "DIFFERENT -- do not mix columns")
PYEOF
