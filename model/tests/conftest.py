"""Shared fixtures: paths, a tiny ArchConfig, and a synthetic-batch builder."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "model"))
sys.path.insert(0, str(REPO / "eval"))

TAB, NEWLINE, N, MASK = 5, 6, 4, 7


def make_batch(
    specs: list[list[list[int]]],
    seq_len: int,
    *,
    seed: int = 0,
    junk_padding: bool = False,
    n_share: float = 0.0,
) -> dict[str, torch.Tensor]:
    """One window per spec; a spec is a list of molecules; a molecule is a list of contig lengths."""
    rng = np.random.default_rng(seed)
    junk_rng = np.random.default_rng(seed + 10_000)
    w_n = len(specs)
    tokens = np.full((w_n, seq_len), NEWLINE, dtype=np.int64)
    valid = np.zeros((w_n, seq_len), dtype=bool)
    cseg = np.zeros((w_n, seq_len), dtype=np.int64)
    useg = np.zeros((w_n, seq_len), dtype=np.int64)
    n_patches = np.zeros(w_n, dtype=np.int64)

    for w, mols in enumerate(specs):
        cursor, n_contigs = 0, 0
        for mol_i, contigs in enumerate(mols):
            for k, ln in enumerate(contigs):
                bases = rng.integers(0, 4, size=ln)
                if n_share:
                    bases[rng.random(ln) < n_share] = N
                sep = TAB if k < len(contigs) - 1 else (NEWLINE if mol_i < len(mols) - 1 else None)
                span = ln + (1 if sep is not None else 0)
                assert cursor + span <= seq_len, "fixture spec overflows the window"
                tokens[w, cursor : cursor + ln] = bases
                if sep is not None:
                    tokens[w, cursor + ln] = sep
                cseg[w, cursor : cursor + span] = n_contigs
                useg[w, cursor : cursor + span] = mol_i
                cursor += span
                n_contigs += 1
        valid[w, :cursor] = True
        n_patches[w] = len(mols)
        if junk_padding:
            tokens[w, cursor:] = junk_rng.integers(0, 7, size=seq_len - cursor)
            cseg[w, cursor:] = junk_rng.integers(0, 50, size=seq_len - cursor)
            useg[w, cursor:] = junk_rng.integers(0, 50, size=seq_len - cursor)

    return {
        "tokens": torch.from_numpy(tokens),
        "valid": torch.from_numpy(valid),
        "contig_segment_id": torch.from_numpy(cseg),
        "umi_segment_id": torch.from_numpy(useg),
        "n_patches": torch.from_numpy(n_patches),
    }


def mask_some(batch: dict, *, seed: int = 1, ratio: float = 0.3) -> dict:
    """A trivial stand-in for the masker: MASK random valid non-separator positions."""
    g = torch.Generator().manual_seed(seed)
    tokens = batch["tokens"]
    maskable = batch["valid"] & (tokens != TAB) & (tokens != NEWLINE)
    coins = torch.rand(tokens.shape, generator=g) < ratio
    masked = maskable & coins
    tokens_student = tokens.clone()
    tokens_student[masked] = MASK
    out = dict(batch)
    out["tokens_student"] = tokens_student
    out["mlm_keep"] = masked & (tokens != N)
    return out


@pytest.fixture
def tiny_cfg():
    from scrm_model.config import ArchConfig

    return ArchConfig(
        n_layers_local=2,
        dim_local=8,
        n_heads_local=2,
        n_layers_global=2,
        dim_global=16,
        n_heads_global=2,
        multiple_of_local=8,
        multiple_of_global=8,
        proj_hidden=32,
        proj_dim=16,
        attn_impl="sdpa",
        rope_max_pos=64,
    )


@pytest.fixture
def pair_specs():
    """4 windows = 2 cells x 2 views, ragged: single-molecule windows, multi-contig molecules, and
    one window of many short molecules.
    """
    return [
        [[12, 7], [9], [4, 4, 4]],
        [[30]],
        [[6], [5], [7], [4], [8]],
        [[10, 10], [11]],
    ]
