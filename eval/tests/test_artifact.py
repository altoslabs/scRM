"""The artifact and the reductions, against synthetic per-window arrays."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scrm_eval import artifact, controls, reduce, registry  # noqa: E402
from scrm_eval.artifact import Stamp, StampError  # noqa: E402

DIM = 8
COMPAT = {"alphabet_size": 7, "k": 19, "dedup_rule": "1MM_CR_flow"}

A_DATASET = registry.datasets("train")[0]
A_COUNT_PRODUCER = controls.PRODUCERS[0]
A_MODEL_PRODUCER = sorted(artifact.MODEL_PRODUCERS)[0]

DIGESTS = {A_DATASET: "a" * 64}


def rows(n_cells: int, w: int, dim: int = DIM, *, seed: int = 0):
    """`n_cells` cells with `w` windows each, contiguous, as the packer writes them."""
    rng = np.random.default_rng(seed)
    ids = [f"CELL{i:04d}" for i in range(n_cells) for _ in range(w)]
    win = np.array([j for _ in range(n_cells) for j in range(w)], dtype=np.uint16)
    mat = rng.normal(size=(n_cells * w, dim)).astype(np.float32)
    return ids, win, mat


def stamp_for(ids, mat, *, producer=A_COUNT_PRODUCER, **over) -> Stamp:
    base = dict(
        producer=producer,
        dim=mat.shape[1],
        n_cells=len(dict.fromkeys(ids)),
        n_windows=mat.shape[0],
        cell_order_sha256=artifact.cell_order_digest(dict.fromkeys(ids)),
        corpus_compat_key=COMPAT,
        corpus_digests=DIGESTS,
        code_rev="deadbeef",
        w_eval=4,
        eval_seed=0,
        seq_len=16384,
    )
    if producer in artifact.MODEL_PRODUCERS:
        base.update(checkpoint_sha256="c" * 64, arch_sha="d" * 64, readout={"rule": "mean"})
    base.update(over)
    return Stamp(**base)


def test_a_written_artifact_reads_back_identical(tmp_path):
    ids, win, mat = rows(5, 4)
    s = stamp_for(ids, mat)
    p = tmp_path / "emb.arrow"
    artifact.write(p, ids, win, mat, s)
    got = artifact.read(p)
    assert got.cell_ids.tolist() == ids
    assert got.windows.tolist() == win.tolist()
    assert np.array_equal(got.matrix, mat)
    assert got.stamp == s


def test_the_stamp_survives_json_round_trip():
    ids, _, mat = rows(3, 2)
    s = stamp_for(ids, mat)
    assert Stamp.from_json(s.to_json()) == s


def test_reading_an_unstamped_arrow_file_is_refused(tmp_path):
    """An Arrow file carrying correct data and no stamp must be refused."""
    import pyarrow as pa
    from pyarrow import ipc

    t = pa.table({"cell_id": pa.array(["A", "B"]), "window": pa.array([0, 1], type=pa.uint16())})
    p = tmp_path / "bare.arrow"
    with ipc.new_file(pa.OSFile(str(p), "wb"), t.schema) as w:
        w.write_table(t)
    with pytest.raises(StampError, match="carries no scrm_eval_stamp"):
        artifact.read(p)


def test_a_model_embedding_without_a_readout_spec_is_refused(tmp_path):
    """A `producer == "scrm"` stamp with no readout spec must be refused."""
    ids, win, mat = rows(4, 2)
    for missing in ("readout", "arch_sha", "checkpoint_sha256"):
        s = stamp_for(ids, mat, producer=A_MODEL_PRODUCER, **{missing: None})
        with pytest.raises(StampError, match=missing):
            artifact.write(tmp_path / "x.arrow", ids, win, mat, s)


def test_a_baseline_may_omit_the_model_only_fields(tmp_path):
    """A baseline stamp must read through the one reader without the model-only fields."""
    ids, win, mat = rows(4, 2)
    s = stamp_for(ids, mat, producer=A_COUNT_PRODUCER)
    assert s.readout is None
    artifact.write(tmp_path / "b.arrow", ids, win, mat, s)
    assert artifact.read(tmp_path / "b.arrow").stamp.producer == A_COUNT_PRODUCER


@pytest.mark.parametrize("field", ["producer", "dim", "corpus_digests", "code_rev", "w_eval"])
def test_a_stamp_missing_a_required_field_is_refused(field):
    ids, _, mat = rows(3, 2)
    raw = __import__("json").loads(stamp_for(ids, mat).to_json())
    del raw[field]
    with pytest.raises(StampError, match="missing required fields"):
        Stamp.from_json(__import__("json").dumps(raw))


def test_a_stamp_from_a_newer_writer_is_refused_not_silently_truncated():
    """A stamp carrying an unknown field must be refused, not truncated."""
    ids, _, mat = rows(3, 2)
    raw = __import__("json").loads(stamp_for(ids, mat).to_json())
    raw["temperature_schedule"] = "cosine"
    with pytest.raises(StampError, match="does not know"):
        Stamp.from_json(__import__("json").dumps(raw))


@pytest.mark.parametrize(
    "over,match",
    [
        ({"producer": ""}, "no producer"),
        ({"dim": 0}, "must be positive"),
        ({"w_eval": -1}, "must be positive"),
        ({"cell_order_sha256": "short"}, "not a sha256"),
        ({"corpus_digests": {}}, "no corpus digests"),
        ({"corpus_compat_key": {}}, "no corpus compat_key"),
    ],
)
def test_a_meaningless_stamp_is_refused(over, match):
    ids, _, mat = rows(3, 2)
    with pytest.raises(StampError, match=match):
        stamp_for(ids, mat, **over).validate()


@pytest.mark.parametrize("value", ["bf16+eager", "bf16+compiled", "fp32+eager", None])
def test_a_well_formed_forward_precision_is_accepted(value):
    ids, _, mat = rows(4, 2)
    stamp_for(ids, mat, producer=A_MODEL_PRODUCER, forward_precision=value).validate()


@pytest.mark.parametrize("value", ["bf16-compiled", "bf16", "bfloat16+eager", "bf16+compile", ""])
def test_a_forward_precision_outside_the_writers_grammar_is_refused(value):
    """`forward_precision` must be refused unless it matches the writer's grammar exactly."""
    ids, _, mat = rows(4, 2)
    with pytest.raises(StampError, match="forward_precision"):
        stamp_for(ids, mat, producer=A_MODEL_PRODUCER, forward_precision=value).validate()


def test_the_writers_own_grammar_is_the_one_validated():
    """The regex must accept exactly what `embed.Checkpoint.forward_precision` composes."""
    for dtype in ("bf16", "fp32"):
        for compiled in (True, False):
            got = f"{dtype}+{'compiled' if compiled else 'eager'}"
            assert artifact._PRECISION.fullmatch(got), got


@pytest.mark.parametrize(
    "over,match",
    [
        ({"dim": DIM + 1}, "dim"),
        ({"n_windows": 999}, "n_windows"),
        ({"n_cells": 999}, "n_cells"),
        ({"cell_order_sha256": "f" * 64}, "cell_order_sha256"),
    ],
)
def test_writing_a_stamp_that_disagrees_with_the_data_is_refused(tmp_path, over, match):
    """`write()` must recompute the stamp from the arrays and refuse one that disagrees."""
    ids, win, mat = rows(5, 3)
    with pytest.raises(StampError, match=match):
        artifact.write(tmp_path / "x.arrow", ids, win, mat, stamp_for(ids, mat, **over))


def test_ragged_inputs_are_refused(tmp_path):
    ids, win, mat = rows(4, 2)
    with pytest.raises(ValueError, match="ragged"):
        artifact.write(tmp_path / "x.arrow", ids[:-1], win, mat, stamp_for(ids, mat))


def test_non_finite_embeddings_are_refused(tmp_path):
    ids, win, mat = rows(4, 2)
    mat = mat.copy()
    mat[3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        artifact.write(tmp_path / "x.arrow", ids, win, mat, stamp_for(ids, mat))


def test_a_second_writer_to_one_destination_cannot_publish_an_unreadable_artifact(tmp_path, monkeypatch):
    """Two writers in flight to one artifact path must publish only a readable artifact."""
    dest = tmp_path / "x.arrow"
    a_ids, a_win, a_mat = rows(6, 2, seed=1)
    b_ids, b_win, b_mat = rows(6, 2, seed=2)

    real_new_file = artifact.ipc.new_file
    fired = []

    def interleaved(sink, schema, *a, **k):
        if not fired:
            fired.append(True)
            artifact.write(dest, b_ids, b_win, b_mat, stamp_for(b_ids, b_mat))
        return real_new_file(sink, schema, *a, **k)

    monkeypatch.setattr(artifact.ipc, "new_file", interleaved)
    artifact.write(dest, a_ids, a_win, a_mat, stamp_for(a_ids, a_mat))
    monkeypatch.undo()

    assert fired, "the interleave never happened, so this fixture asserts nothing"
    got = artifact.read(dest)
    assert np.allclose(got.matrix, a_mat)


def test_a_write_that_fails_publishes_nothing_and_leaves_nothing_behind(tmp_path, monkeypatch):
    """A failed write must publish nothing and leave no staging file behind."""
    dest = tmp_path / "x.arrow"
    ids, win, mat = rows(4, 2)

    def boom(*a, **k):
        raise ValueError("disk on fire")

    monkeypatch.setattr(artifact.ipc, "new_file", boom)
    with pytest.raises(ValueError, match="disk on fire"):
        artifact.write(dest, ids, win, mat, stamp_for(ids, mat))
    monkeypatch.undo()

    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_two_artifacts_over_the_same_cells_are_comparable():
    ids, _, mat = rows(6, 2)
    artifact.same_population(stamp_for(ids, mat, producer=A_MODEL_PRODUCER), stamp_for(ids, mat))


def test_different_cell_sets_are_refused():
    a_ids, _, a_mat = rows(6, 2)
    b_ids, _, b_mat = rows(5, 2)
    with pytest.raises(StampError, match="different cells"):
        artifact.same_population(stamp_for(a_ids, a_mat), stamp_for(b_ids, b_mat))


def test_a_different_cell_ORDER_is_refused():
    """`cell_order_sha256` must differ for the same cells in a different order, so row i of one
    artifact is never compared against row i of another.
    """
    ids, _, mat = rows(6, 2)
    flipped = [c for i in range(5, -1, -1) for c in [f"CELL{i:04d}"] * 2]
    with pytest.raises(StampError, match="different cells or a different order"):
        artifact.same_population(stamp_for(ids, mat), stamp_for(flipped, mat))


def test_a_corpus_content_difference_is_refused_and_named():
    """A corpus content difference must be refused and must name the dataset that differs."""
    ids, _, mat = rows(4, 2)
    other = {A_DATASET: "b" * 64}
    with pytest.raises(StampError, match=re.escape(f"differing=['{A_DATASET}']")):
        artifact.same_population(stamp_for(ids, mat), stamp_for(ids, mat, corpus_digests=other))


def test_every_rule_returns_one_row_per_cell_in_artifact_order():
    ids, _, mat = rows(7, 3)
    want = [f"CELL{i:04d}" for i in range(7)]
    for rule in reduce.RULES:
        cells, out = reduce.reduce_windows(ids, mat, rule)
        assert cells.tolist() == want, rule
        assert out.shape == (7, DIM), rule


def test_the_reductions_compute_what_they_claim():
    """Each reduction rule must compute the statistic it names, by value and not only by shape."""
    ids = ["A", "A", "A", "B"]
    mat = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 11.0], [5.0, 5.0]])
    exp = {
        "mean": [2.0, 5.0],
        "median": [2.0, 3.0],
        "max": [4.0, 11.0],
        "first": [0.0, 1.0],
    }
    for rule, want in exp.items():
        _, out = reduce.reduce_windows(ids, mat, rule)
        assert out[0].tolist() == pytest.approx(want), rule
    _, out = reduce.reduce_windows(ids, mat, "l2_mean")
    unit = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 11.0]])
    unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    assert out[0].tolist() == pytest.approx(unit.mean(axis=0).tolist())


def test_a_single_window_cell_is_unchanged_by_every_rule():
    """Every reduction rule must be the identity on a one-window cell."""
    ids = ["A", "B", "C"]
    mat = np.arange(9, dtype=float).reshape(3, 3)
    for rule in reduce.RULES:
        _, out = reduce.reduce_windows(ids, mat, rule)
        if rule == "l2_mean":
            continue
        if rule == "mean_standardised":
            assert np.allclose(out.mean(axis=0), 0.0), "must leave a zero column mean"
            assert np.allclose(out.std(axis=0), 1.0, atol=1e-6), "and a unit column sd"
            continue
        if rule == "mean_centred":
            assert np.allclose(out.mean(axis=0), 0.0), "mean_centred must leave a zero column mean"
            assert np.allclose(out, mat - mat.mean(axis=0)), "and differ from `mean` by exactly that"
            continue
        assert np.array_equal(out, mat), rule


def test_non_contiguous_windows_of_one_cell_are_refused():
    """Windows of one cell that are not contiguous must be refused rather than averaged."""
    ids = ["A", "B", "A"]
    mat = np.ones((3, 4))
    with pytest.raises(ValueError, match="not contiguous"):
        reduce.reduce_windows(ids, mat)


def test_an_unknown_rule_is_refused():
    with pytest.raises(ValueError, match="unknown reduction"):
        reduce.reduce_windows(["A"], np.ones((1, 2)), "softmax_pool")


def test_all_rules_produces_every_arm_from_one_pass():
    """`all_rules` must produce every window-to-cell arm from a single embed pass."""
    ids, _, mat = rows(5, 4)
    got = reduce.all_rules(ids, mat)
    assert set(got) == set(reduce.RULES)
    assert all(v[1].shape == (5, DIM) for v in got.values())
    means = got["mean"][1]
    assert not np.allclose(means, got["max"][1])
    assert not np.allclose(means, got["first"][1])
