"""How many windows a corpus's DEEPEST cell needs — i.e. what `w_eval` must be to read all reads."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "target" / "py"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import scrm_py  # noqa: E402
from scrm_eval import registry  # noqa: E402

CORPUS = registry.corpus_root()
OUT = registry.PROJ / "eval_artifacts" / "windows_required.json"
SEQ_LEN = 16384
PACKING_MARGIN = 1.05


def _exact_tokens_per_cell(dataset: str) -> np.ndarray:
    """Total tokens per cell, read from the shards. One separator per molecule, as the tiler emits."""
    import pyarrow as pa
    from pyarrow import ipc

    totals: dict[str, int] = {}
    for shard in sorted((CORPUS / dataset).glob("*.arrow")):
        with pa.memory_map(str(shard), "rb") as src:
            reader = ipc.open_file(src)
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                cells = batch.column("cell_id").dictionary_decode().to_pylist()
                lengths = np.diff(np.asarray(batch.column("tokens").offsets))
                for cell, length in zip(cells, lengths):
                    totals[cell] = totals.get(cell, 0) + int(length) + 1
    return np.asarray(list(totals.values()), dtype=np.int64)


def _corpus_identity(dataset: str) -> dict:
    """What corpus this requirement was measured on: its declared molecule definition and digest."""
    man = json.loads((CORPUS / dataset / "manifest.json").read_text())
    return {
        "corpus_molecule_rule": man["declared"]["config"]["bubble_resolution"],
        "corpus_tokens_sha256_root": man.get("tokens_sha256_root") or None,
        "corpus_n_rows": man.get("n_rows"),
        "corpus_n_tokens": man.get("n_tokens"),
    }


def required(dataset: str) -> dict:
    mpc = np.asarray(scrm_py.Corpus([str(CORPUS / dataset)]).molecules_per_cell())
    tokens = _exact_tokens_per_cell(dataset)
    per_cell = np.ceil(tokens / SEQ_LEN)
    need = int(math.ceil(per_cell.max() * PACKING_MARGIN))
    need += need % 2
    return {
        **_corpus_identity(dataset),
        "w_eval": need,
        "cells": int(len(mpc)),
        "max_molecules": int(mpc.max()),
        "max_tokens_in_a_cell": int(tokens.max()),
        "exact_window_bound": int(per_cell.max()),
        "total_windows_estimate": int(per_cell.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--repin", action="store_true",
                    help="allow --write to CHANGE a w_eval that embeddings are already using")
    ap.add_argument("--dataset", help="print just this one (for a job script)")
    args = ap.parse_args()

    if args.dataset:
        try:
            print(registry.full_coverage(args.dataset))
            return 0
        except (KeyError, FileNotFoundError):
            pass
        print(required(args.dataset)["w_eval"])
        return 0

    out = {}
    for d in sorted(p.name for p in CORPUS.iterdir() if (p / "manifest.json").exists()):
        out[d] = required(d)
        r = out[d]
        print(f"{d:18s} w_eval={r['w_eval']:>5d}  cells={r['cells']:>7,}  "
              f"deepest={r['max_molecules']:>8,} mol  total≈{r['total_windows_estimate']:>9,} windows")
    if args.write:
        prev = json.loads(OUT.read_text()) if OUT.exists() else {}
        moved = {d: (prev[d]["w_eval"], out[d]["w_eval"]) for d in out
                 if d in prev and prev[d]["w_eval"] != out[d]["w_eval"]}
        if moved and not args.repin:
            for d, (was, now) in sorted(moved.items()):
                print(f"  REFUSING: {d} w_eval {was} -> {now}")
            print("\nThese datasets have embeddings keyed to the old w_eval; rewriting the file makes\n"
                  "every running task discard its checkpoints and restart from cell 0. Pass --repin\n"
                  "ONLY with the fleet drained, and expect those datasets to recompute in full.")
            return 1
        if moved:
            for d, (was, now) in sorted(moved.items()):
                print(f"  REPINNED {d}: w_eval {was} -> {now} (checkpoints for it are now void)")
        tmp = OUT.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out, indent=2, sort_keys=True))
        tmp.replace(OUT)
        print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
