"""The block masker's obligations."""

from __future__ import annotations

import numpy as np

from conftest import NEWLINE, TAB, make_batch

from scrm_model.masking import LENGTH_BINS, MaskOut, block_mask

V = 2


def batch_with_cells(specs, seq_len, *, seed=0):
    b = make_batch(specs, seq_len, seed=seed)
    out = {k: v.numpy() for k, v in b.items()}
    out["cell_index"] = (np.arange(len(specs)) // V).astype(np.uint32)
    return out


SPECS = [
    [[60, 40], [95], [19, 30]],
    [[210]],
    [[19], [19], [64], [30, 22]],
    [[130], [88]],
]


def run(batch, *, epoch_seed=11, lo=0.2, hi=0.4) -> MaskOut:
    return block_mask(batch, epoch_seed=epoch_seed, v=V, min_ratio=lo, max_ratio=hi)


def test_no_masked_position_is_a_separator_or_padding():
    batch = batch_with_cells(SPECS, 256)
    out = run(batch)
    assert not (out.mask & (batch["tokens"] == TAB)).any()
    assert not (out.mask & (batch["tokens"] == NEWLINE)).any()
    assert not (out.mask & ~batch["valid"]).any()
    assert out.mask.any()


def test_every_contig_and_every_umi_keeps_a_visible_base():
    batch = batch_with_cells(SPECS, 256)
    out = run(batch, lo=0.4, hi=0.6)
    tokens, valid, cseg, useg = (batch[k] for k in ("tokens", "valid", "contig_segment_id", "umi_segment_id"))
    is_base = valid & (tokens != TAB) & (tokens != NEWLINE)
    for w in range(len(SPECS)):
        for seg_arr, what in ((cseg, "contig"), (useg, "umi")):
            for s in np.unique(seg_arr[w][valid[w]]):
                at = (seg_arr[w] == s) & is_base[w]
                visible = at & ~out.mask[w]
                assert visible.any(), f"window {w} {what} {s} lost every visible base"


def test_every_masked_run_is_at_least_the_block_floor():
    batch = batch_with_cells(SPECS, 256)
    out = run(batch)
    from scrm_model.masking import _BLOCK_FLOOR

    for w in range(len(SPECS)):
        m = out.mask[w].astype(np.int8)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], m, [0]])))
        runs = edges[1::2] - edges[0::2]
        assert (runs >= _BLOCK_FLOOR).all(), (
            f"window {w} has a masked run of {runs.min()} < {_BLOCK_FLOOR}")


def test_pure_function_of_seed_view_and_epoch():
    batch = batch_with_cells(SPECS, 256)
    a, b = run(batch, epoch_seed=11), run(batch, epoch_seed=11)
    assert np.array_equal(a.mask, b.mask), "same seed must reproduce the mask bitwise"
    assert np.array_equal(a.tokens_student, b.tokens_student)
    assert not np.array_equal(a.mask[0], a.mask[1])
    c = run(batch, epoch_seed=12)
    assert not np.array_equal(a.mask, c.mask), "consecutive epochs must differ"


def test_the_teacher_input_is_untouched():
    batch = batch_with_cells(SPECS, 256)
    before = batch["tokens"].copy()
    out = run(batch)
    assert np.array_equal(batch["tokens"], before), "the masker mutated its input"
    assert (out.tokens_student[out.mask] == 7).all()
    assert np.array_equal(out.tokens_student[~out.mask], batch["tokens"][~out.mask])


def test_n_is_masked_in_the_input_but_never_a_loss_target():
    batch = batch_with_cells(SPECS, 256, seed=4)
    tokens = batch["tokens"]
    is_base = batch["valid"] & (tokens != TAB) & (tokens != NEWLINE)
    tokens[is_base & (np.random.default_rng(0).random(tokens.shape) < 0.3)] = 4
    out = run(batch)
    assert (out.mask & (tokens == 4)).any(), "the fixture must actually mask an N"
    assert not (out.mlm_keep & (tokens == 4)).any(), "an N reached the loss"
    assert not (out.mlm_keep & ~out.mask).any()


def test_realised_ratio_is_measured_and_plausible():
    batch = batch_with_cells([[[200], [180]], [[240]], [[150], [190]], [[220]]], 512)
    out = run(batch, lo=0.2, hi=0.4)
    assert (out.realised_ratio > 0.10).all() and (out.realised_ratio < 0.45).all(), (
        f"realised {out.realised_ratio} outside the sampled band with overlap slack"
    )


def test_the_short_contig_hole_is_visible_in_the_binned_log():
    from scrm_model.masking import _BLOCK_FLOOR

    r = 0.2
    unmaskable = max(1, int(np.ceil((_BLOCK_FLOOR - 0.5) / r)) - 1)
    assert round(r * unmaskable) < _BLOCK_FLOOR
    maskable = int(np.ceil(_BLOCK_FLOOR / r)) + 1
    assert round(r * maskable) >= _BLOCK_FLOOR
    batch = batch_with_cells(
        [[[unmaskable], [unmaskable], [200]], [[unmaskable], [240]]] * 2, 512)
    out = run(batch, lo=r, hi=r)
    short_bin = np.digitize([unmaskable], LENGTH_BINS)[0] - 1
    long_bin = np.digitize([200], LENGTH_BINS)[0] - 1
    assert out.realised_by_length[short_bin, 0] > 0, "fixture bins are wrong"
    assert out.realised_by_length[short_bin, 1] == 0.0, (
        f"{unmaskable}-byte contigs cannot be masked at r={r} with floor {_BLOCK_FLOOR}")
    assert out.realised_by_length[long_bin, 1] > 0.1


def test_seed_derivation_matches_the_packer_bit_for_bit():
    from scrm_model.masking import _derive_seed

    def scalar_splitmix64(state):
        m = (1 << 64) - 1
        state = (state + 0x9E3779B97F4A7C15) & m
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & m
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & m
        return state, z ^ (z >> 31)

    def scalar_derive(epoch_seed, a, b):
        m = (1 << 64) - 1
        s = epoch_seed ^ 0xA0761D6478BD642F
        s = (s + (a * 0x9E3779B97F4A7C15)) & m
        s, _ = scalar_splitmix64(s)
        s = (s + (b * 0xC2B2AE3D27D4EB4F)) & m
        _, out = scalar_splitmix64(s)
        return out

    rng = np.random.default_rng(0)
    e = rng.integers(0, 1 << 63, size=16, dtype=np.uint64)
    a = rng.integers(0, 1 << 63, size=16, dtype=np.uint64)
    b = rng.integers(0, 1 << 63, size=16, dtype=np.uint64)
    got = _derive_seed(e, a, b)
    want = [scalar_derive(int(x), int(y), int(z)) for x, y, z in zip(e, a, b)]
    assert got.tolist() == want


def test_the_module_docstring_states_the_geometry_the_code_HAS():
    """`masking`'s module docstring must state the live region cap and the live mask-ratio schedule."""
    from scrm_model import masking, optim

    doc = masking.__doc__
    assert "min(_MAX_REGIONS, n_mask // _BLOCK_FLOOR)" in doc
    assert f"min({masking._MAX_REGIONS}, n_mask // {masking._BLOCK_FLOOR})" in doc
    lo0, hi0 = optim.mask_ratio_at(0, 100)
    lo1, hi1 = optim.mask_ratio_at(100, 100)
    assert f"U(.{int(lo0 * 100)},.{int(hi0 * 100)}) -> U(.{int(lo1 * 100)},.{int(hi1 * 100)})" in doc
    for stale in ("min(8, n_mask", "U(.2,.4) -> U(.4,.6)\n"):
        assert stale not in doc, f"the docstring still states {stale!r} as the mechanism"
