#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:10G

if [ -z "${SLURM_JOB_ID:-}" ]; then >&2 echo "Run with: sbatch $0 {arm} {run} {ckpt}"; exit 1; fi
set -euo pipefail
ARM=${1:?arm}; RUN=${2:?run}; CKPT=${3:-latest}
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
export SCRM_REPO="$REPO"
PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
PY=${SCRM_PY:-python3}
export PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py
export LD_LIBRARY_PATH=$(dirname "$(dirname "${SCRM_PY:-python3}")")/lib
OUT=$PROJ/eval_runs/$RUN
mkdir -p "$OUT"

srun $PY - "$ARM" "$RUN" "$CKPT" <<'PYEOF'
import json
import sys
import os
sys.path[:0] = [os.environ["SCRM_REPO"] + p for p in ("/model", "/eval", "/target/py")]
from scrm_eval.registry import corpus_root
from scrm_eval import registry
from pathlib import Path
import numpy as np
import scrm_py
from scrm_eval import embed as ev, targets
from scrm_eval.metrics import residualise
from scrm_eval.probes import _folds
from scrm_eval.reduce import reduce_windows
from scrm_model.evalio import model_factory
from scrm_model.run_eval import eval_ready
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

arm, run, ckpt_name = sys.argv[1:4]
proj = registry.PROJ
out = proj / "eval_runs" / run
art = out / f"sc_mixology_5cl.{run}.{ckpt_name}.wfull.arrow"
if not art.exists():
    ckpt = ev.Checkpoint.load(eval_ready(proj / "runs" / run / f"{ckpt_name}.pt", arm, out), model_factory=model_factory)
    corpus = scrm_py.Corpus([str(corpus_root() / "sc_mixology_5cl")])
    ev.embed_to_file(ckpt, corpus, art, dataset="sc_mixology_5cl", w_eval=256, eval_seed=20260818)
    print(f"embedded w_eval=256 -> {art}", flush=True)

from scrm_eval import artifact as arti
emb = arti.read(art)
cells, x = reduce_windows(emb.cell_ids, emb.matrix, "mean")
cells = [str(c) for c in cells]
labels = {k.split("-")[0]: v for k, v in json.loads(
    (proj / "raw/sc_mixology_meta/sc_mixology_5cl_labels_sng.json").read_text()).items()}
y = np.asarray([labels.get(c.split("-")[0], "") for c in cells])

adata = targets.load_counts("sc_mixology_5cl")
rows, has_counts = targets.align_to_cells(adata, cells)
size = np.zeros((len(cells), 2))
size[has_counts, 0] = np.log1p(np.asarray(adata.X[rows[has_counts]].sum(axis=1)).ravel())
size[has_counts, 1] = np.log1p(np.asarray((adata.X[rows[has_counts]] > 0).sum(axis=1)).ravel())

keep = (y != "") & has_counts
x, y, size = x[keep], y[keep], size[keep]
x_sr = np.asarray(residualise(x.astype(np.float64), size)).reshape(len(x), -1)

report = {}
for variant, xv in (("raw", x), ("size_removed", x_sr)):
    for name, make in (("linear", lambda: LogisticRegression(max_iter=2000)),
                       ("knn_cosine", lambda: KNeighborsClassifier(n_neighbors=10, metric="cosine"))):
        accs, f1s = [], []
        for s in range(3):
            preds = np.empty(len(xv), dtype=object)
            for tr, te in _folds(len(xv), 5, seed=s):
                preds[te] = make().fit(xv[tr], y[tr]).predict(xv[te])
            accs.append(accuracy_score(y, preds.astype(str)))
            f1s.append(f1_score(y, preds.astype(str), average="macro"))
        report[f"{variant}.{name}"] = {"accuracy": float(np.mean(accs)), "macro_f1": float(np.mean(f1s))}
        print(f"{variant:13s} {name:11s} acc {np.mean(accs):.4f}  f1 {np.mean(f1s):.4f}", flush=True)
report["n_cells"] = int(keep.sum())
report["w_eval"] = 256
(out / f"sc_mixology_5cl.{run}.{ckpt_name}.wfull.protocol_match.json").write_text(json.dumps(report, indent=2))
PYEOF
echo done
