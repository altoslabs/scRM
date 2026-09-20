"""Provenance for the count-FM embedding npz files.

`embed_count_fm.py` imports `stamp` from here to build the record, and `savez_stamped` to write
it beside the arrays.

An npz without a stamp is an unattributable embedding, which is what
`eval/scrm_eval/artifact.py` refuses on the Arrow side. The same rule applies here, one layer
earlier: the npz records the model, the matrix it read, the code that produced it and the
arguments it was given, and `controls.from_npz` carries that into the Arrow stamp.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=Path(__file__).resolve().parent, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    """Digest a file, or the sorted digests of a directory's files, so a model directory is
    identified as a whole rather than by whichever file happens to be first."""
    if path.is_file():
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    if path.is_dir():
        h = hashlib.sha256()
        for f in sorted(p for p in path.rglob("*") if p.is_file()):
            h.update(str(f.relative_to(path)).encode())
            h.update(_sha256(f).encode())
        return h.hexdigest()
    return ""


def stamp(args: Any, *, corpus: Path | str | None = None, extra: dict | None = None) -> dict:
    """The record: what ran, on what, with which arguments."""
    argv = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()} if args else {}
    rec = {
        "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "producer": "scrm/eval/countfm/embed_count_fm.py",
        "scrm_sha": _git("rev-parse", "HEAD"),
        "scrm_dirty": _git("status", "--porcelain") != "",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "args": argv,
    }
    if corpus is not None:
        p = Path(corpus)
        rec["source"] = str(p)
        rec["source_sha256"] = _sha256(p)
    if extra:
        rec.update(extra)
    return rec


def savez_stamped(out: Path | str, arrays: dict[str, np.ndarray], record: dict) -> Path:
    """Write the arrays with the record alongside, tmp-then-rename."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    # WRITTEN THROUGH A FILE HANDLE, because `np.savez` APPENDS `.npz` to a path that does not
    # end in it: given `x.npz.tmp` it writes `x.npz.tmp.npz`, and the rename then fails on a
    # file that was never created. A handle is taken as given.
    with tmp.open("wb") as fh:
        np.savez(fh, **arrays, provenance=np.array(json.dumps(record, sort_keys=True)))
    tmp.replace(out)
    return out
