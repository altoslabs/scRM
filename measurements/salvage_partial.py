"""Assemble a real artifact out of resume chunks, for work that is computed but not finished."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "eval"), str(REPO / "target" / "py"), str(REPO / "model")]
from scrm_eval import artifact, controls  # noqa: E402


def assemble(resume_dir: Path):
    """(cells, windows, matrix, key) from every contiguous chunk, in order."""
    chunks = sorted(resume_dir.glob("chunk-*.npz"))
    if not chunks:
        raise SystemExit(f"{resume_dir}: no chunks")
    key = json.loads((resume_dir / "key.json").read_text())
    cells: list[str] = []
    windows: list[int] = []
    mats: list[np.ndarray] = []
    nxt = 0
    for c in chunks:
        with np.load(c, allow_pickle=False) as z:
            start, next_start = int(z["start"]), int(z["next_start"])
            if start != nxt:
                print(f"  gap at {c.name}: expected start {nxt}, got {start}; stopping", flush=True)
                break
            blob = str(z["cells"])
            these = blob.split("\n") if blob else []
            w = [int(x) for x in z["windows"]]
            m = np.asarray(z["matrix"], dtype=np.float32)
            if not (len(these) == len(w) == m.shape[0]):
                print(f"  {c.name}: {len(these)} cells vs {len(w)} ordinals vs {m.shape[0]} rows; stopping")
                break
            cells += these
            windows += w
            mats.append(m)
            nxt = next_start
    if not mats:
        raise SystemExit(f"{resume_dir}: nothing contiguous to assemble")
    return cells, np.asarray(windows, dtype=np.uint16), np.vstack(mats), key, nxt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("resume_dirs", nargs="+", type=Path)
    ap.add_argument("--out-suffix", default="partial")
    a = ap.parse_args()
    for rd in a.resume_dirs:
        cells, windows, matrix, key, reached = assemble(rd)
        ds = key["dataset"]
        if "corpus_def" not in key:
            raise SystemExit(
                f"{rd}: the resume key has no `corpus_def`, so which molecule definition produced "
                f"these chunks cannot be established. Stamping it from the dataset name would "
                f"assert a corpus identity the vectors may not have -- the v3 quarantine. Re-embed "
                f"instead; the chunks are not salvageable."
            )
        compat, digests, _ = controls.corpus_facts(ds)
        if key["corpus_def"] is not None and dict(compat) != dict(key["corpus_def"]):
            raise SystemExit(
                f"{rd}: the chunks were computed under molecule definition {key['corpus_def']} and "
                f"the corpus at {controls.CORPUS_ROOT}/{ds} now declares {dict(compat)}. Salvaging "
                f"would stamp one corpus's identity onto another's vectors."
            )
        stamp = artifact.Stamp(
            producer="scrm",
            dim=int(matrix.shape[1]),
            n_cells=len(dict.fromkeys(cells)),
            n_windows=int(matrix.shape[0]),
            cell_order_sha256=artifact.cell_order_digest(dict.fromkeys(cells)),
            corpus_compat_key=dict(compat) if isinstance(compat, dict) else {"compat_key": list(compat)},
            corpus_digests=digests,
            code_rev=controls.code_rev(),
            w_eval=int(key["w_eval"]),
            eval_seed=int(key["eval_seed"]),
            seq_len=int(key["seq_len"]),
            checkpoint_sha256=key.get("checkpoint_sha256"),
            arch_sha=key.get("arch_sha"),
            readout={"salvaged_from": "resume chunks"},
            forward_precision=key.get("forward_precision"),
            notes={"dataset": ds, "rows": "pairs" if key.get("pair_mode") else "windows",
                   "pair_v": key.get("pair_v"),
                   "salvaged": True, "cells_reached": reached,
                   "source": str(rd.name)},
        )
        out = rd.parent / rd.name.replace(".arrow.resume", f".{a.out_suffix}.arrow")
        artifact.write(out, np.asarray(cells), windows, matrix, stamp)
        print(f"  {out.name}: {stamp.n_cells:,} cells, {stamp.n_windows:,} rows, "
              f"w_eval={stamp.w_eval}, {stamp.forward_precision}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
