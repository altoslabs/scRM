"""The block masker: per window, per view, **per contig** — a pure function of
`(epoch_seed, cell, view, contig)`, never carried state.

r ~ U(min_ratio, max_ratio), scheduled U(.15,.25) -> U(.20,.30), and the masked bases are split
into n_region = min(_MAX_REGIONS, n_mask // _BLOCK_FLOOR) = min(3, n_mask // 3) contiguous blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MASK_DOMAIN = np.uint64(0x6D61_736B)
_TAB, _NEWLINE, _N = 5, 6, 4
_MASK_ID = 7
_BLOCK_FLOOR = 3
_MAX_REGIONS = 3

_U64 = np.uint64


def _splitmix64(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One splitmix64 step, vectorised; returns (new_state, output). Mirrors the Rust bit for bit."""
    state = state + _U64(0x9E37_79B9_7F4A_7C15)
    z = state.copy()
    z = (z ^ (z >> _U64(30))) * _U64(0xBF58_476D_1CE4_E5B9)
    z = (z ^ (z >> _U64(27))) * _U64(0x94D0_49BB_1331_11EB)
    return state, z ^ (z >> _U64(31))


def _derive_seed(epoch_seed: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`scrm_pack::derive_seed`, vectorised, same constants."""
    s = epoch_seed ^ _U64(0xA076_1D64_78BD_642F)
    s = s + a * _U64(0x9E37_79B9_7F4A_7C15)
    s, _ = _splitmix64(s)
    s = s + b * _U64(0xC2B2_AE3D_27D4_EB4F)
    _, out = _splitmix64(s)
    return out


def _uniforms(seed: np.ndarray, k: int) -> np.ndarray:
    """(len(seed), k) doubles in [0, 1): k fixed splitmix64 iterates per seed — a pure function."""
    out = np.empty((seed.shape[0], k), dtype=np.float64)
    state = seed.copy()
    for i in range(k):
        state, z = _splitmix64(state)
        out[:, i] = (z >> _U64(11)).astype(np.float64) * (2.0**-53)
    return out


@dataclass
class MaskOut:
    tokens_student: np.ndarray
    mask: np.ndarray
    mlm_keep: np.ndarray
    realised_ratio: np.ndarray
    realised_by_length: np.ndarray
    length_bins: np.ndarray


LENGTH_BINS = np.array([0, 24, 32, 48, 64, 96, 128, 192, 1 << 30])


def block_mask(
    batch: dict[str, np.ndarray],
    *,
    epoch_seed: int,
    v: int,
    min_ratio: float,
    max_ratio: float,
    compute_bins: bool = True,
) -> MaskOut:
    tokens = np.asarray(batch["tokens"])
    valid = np.asarray(batch["valid"]).astype(bool, copy=False)
    cseg = np.asarray(batch["contig_segment_id"])
    cell_index = np.asarray(batch["cell_index"]).astype(np.uint64)
    w_n, s_len = tokens.shape
    view = (np.arange(w_n, dtype=np.uint64)) % _U64(v)

    prev_same = np.zeros_like(valid)
    prev_same[:, 1:] = valid[:, 1:] & valid[:, :-1] & (cseg[:, 1:] == cseg[:, :-1])
    is_start = valid & ~prev_same
    c_window, c_first = np.nonzero(is_start)
    used = valid.sum(axis=1)
    c_end = np.empty_like(c_first)
    same_w = c_window[1:] == c_window[:-1]
    c_end[:-1] = np.where(same_w, c_first[1:], used[c_window[:-1]])
    if len(c_end):
        c_end[-1] = used[c_window[-1]]
    c_span = c_end - c_first
    c_id = cseg[c_window, c_first].astype(np.uint64)
    last_tok = tokens[c_window, c_end - 1]
    c_bases = c_span - ((last_tok == _TAB) | (last_tok == _NEWLINE)).astype(np.int64)

    win_seed = _derive_seed(
        np.full(w_n, epoch_seed, dtype=np.uint64) ^ MASK_DOMAIN, cell_index, view
    )
    c_seed = _derive_seed(win_seed[c_window], c_id, np.zeros_like(c_id))
    K = _MAX_REGIONS
    u = _uniforms(c_seed, 2 * K)

    r = min_ratio + u[:, 0] * (max_ratio - min_ratio)
    n_mask = np.rint(r * c_bases).astype(np.int64)
    n_region = np.minimum(K, n_mask // _BLOCK_FLOOR)

    surplus = np.where(n_region > 0, n_mask - _BLOCK_FLOOR * n_region, 0).astype(np.float64)
    cuts = np.sort(u[:, 1:K], axis=1) * (surplus[:, None] + 1.0)
    cuts = np.floor(cuts)
    used = np.arange(K - 1)[None, :] < np.maximum(n_region[:, None] - 1, 0)
    cuts = np.where(used, cuts, surplus[:, None])
    bounds = np.concatenate(
        [np.zeros((len(c_first), 1)), cuts, surplus[:, None]], axis=1)
    bounds = np.maximum.accumulate(bounds, axis=1)
    part = np.diff(bounds, axis=1).astype(np.int64)
    sizes = np.where(np.arange(K) < n_region[:, None], _BLOCK_FLOOR + part, 0)

    starts = np.floor(u[:, K:2 * K] * np.maximum(c_bases[:, None] - sizes + 1, 1)).astype(np.int64)

    live = sizes > 0
    b_window = np.broadcast_to(c_window[:, None], sizes.shape)[live]
    b_start = (c_first[:, None] + starts)[live]
    b_end = b_start + sizes[live]
    row = w_n * (s_len + 1)
    up = np.bincount(b_window * (s_len + 1) + b_start, minlength=row)
    down = np.bincount(b_window * (s_len + 1) + b_end, minlength=row)
    diff = (up - down).astype(np.int16).reshape(w_n, s_len + 1)
    mask = np.cumsum(diff[:, :-1], axis=1) > 0

    tokens_student = tokens.copy()
    tokens_student[mask] = _MASK_ID
    is_base = valid & (tokens < _TAB)
    mlm_keep = mask & (tokens < _N)

    realised_ratio = mask.sum(1) / np.maximum(is_base.sum(1), 1)

    by_len = np.zeros((len(LENGTH_BINS) - 1, 2), dtype=np.float64)
    if compute_bins and len(c_first):
        mask_cum = np.cumsum(mask, axis=1)
        m_end = mask_cum[c_window, c_end - 1]
        m_before = np.where(c_first > 0, mask_cum[c_window, np.maximum(c_first - 1, 0)], 0)
        masked_per_contig = m_end - m_before
        bin_of = np.digitize(c_bases, LENGTH_BINS) - 1
        np.add.at(by_len[:, 0], bin_of, 1)
        sums = np.zeros((len(LENGTH_BINS) - 1, 2), dtype=np.float64)
        np.add.at(sums[:, 0], bin_of, masked_per_contig)
        np.add.at(sums[:, 1], bin_of, c_bases)
        with np.errstate(invalid="ignore", divide="ignore"):
            by_len[:, 1] = np.where(sums[:, 1] > 0, sums[:, 0] / sums[:, 1], 0.0)

    return MaskOut(
        tokens_student=tokens_student,
        mask=mask,
        mlm_keep=mlm_keep,
        realised_ratio=realised_ratio,
        realised_by_length=by_len,
        length_bins=LENGTH_BINS,
    )
