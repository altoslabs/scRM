"""The count-FM npz writer: it must land at the name it was given, with its record."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "countfm"))
from provenance import savez_stamped, stamp  # noqa: E402

from scrm_eval import panel  # noqa: E402


def _a_model() -> str:
    """Some model the embedder handles, asked of the declaration rather than typed. The suite
    refuses a hardcoded name, and rightly: a test that names one keeps a second copy of the
    panel, which is what `panel.py` exists to prevent."""
    return panel.EMBEDDABLE[0]


def test_the_file_lands_at_the_name_it_was_given(tmp_path):
    """`np.savez` APPENDS `.npz` to a path that does not end in it, so writing the temporary
    file by path produced `x.npz.tmp.npz` and the rename failed on a file that never existed.
    Every count-FM embedding died at the final write, after the model had run.
    """
    out = tmp_path / f"ds.{_a_model()}.npz"
    savez_stamped(out, {"emb": np.zeros((3, 4), dtype=np.float32)}, {"model": _a_model()})
    assert out.is_file(), "the npz must be at the requested path"
    assert not list(tmp_path.glob("*.tmp*")), f"no temporary left behind: {list(tmp_path.iterdir())}"


def test_the_record_travels_with_the_arrays(tmp_path):
    out = tmp_path / f"ds.{_a_model()}.npz"
    savez_stamped(out, {"emb": np.ones((2, 2), dtype=np.float32)}, {"model": _a_model(), "n_cells": 2})
    with np.load(out, allow_pickle=False) as z:
        assert z["emb"].shape == (2, 2)
        rec = json.loads(str(z["provenance"]))
    assert rec["model"] == _a_model() and rec["n_cells"] == 2


def test_a_stamp_names_what_ran_and_on_what(tmp_path):
    src = tmp_path / "matrix.h5ad"
    src.write_bytes(b"not really a matrix")
    # argparse.Namespace is what the embedder passes, and `stamp` reads it with `vars()`.
    rec = stamp(argparse.Namespace(model=_a_model(), matrix=src), corpus=src, extra={"n_cells": 7})
    assert rec["producer"].endswith("embed_count_fm.py")
    assert rec["source"] == str(src) and len(rec["source_sha256"]) == 64
    assert rec["n_cells"] == 7 and rec["args"]["model"] == _a_model()


def test_a_directory_source_is_digested_as_a_whole(tmp_path):
    """A model directory is identified by all its files, not whichever comes first."""
    d = tmp_path / "model"
    (d / "sub").mkdir(parents=True)
    (d / "a.bin").write_bytes(b"a")
    (d / "sub" / "b.bin").write_bytes(b"b")
    first = stamp(None, corpus=d)["source_sha256"]
    (d / "sub" / "b.bin").write_bytes(b"changed")
    assert stamp(None, corpus=d)["source_sha256"] != first
    assert len(first) == 64


@pytest.mark.parametrize("missing", ["nope.npz", "sub/dir/deep.npz"])
def test_the_parent_directory_is_created(tmp_path, missing):
    out = tmp_path / missing
    savez_stamped(out, {"emb": np.zeros((1, 1), dtype=np.float32)}, {})
    assert out.is_file()
