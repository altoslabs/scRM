"""Obligations of the uniform-random molecule dropout (LeVJEPA arXiv 2608.27395 §4.1, adapted)."""
from __future__ import annotations

import torch

from scrm_model.model import drop_molecules


def _slots(n_real: list[int], p_n: int) -> torch.Tensor:
    m = torch.zeros(len(n_real), p_n, dtype=torch.bool)
    for i, n in enumerate(n_real):
        m[i, :n] = True
    return m


def test_rho_zero_returns_the_input_untouched():
    """`rho=0.0` must be the identity, not a draw that happens to keep everything."""
    slots = _slots([7, 3, 12], 16)
    assert drop_molecules(slots, 0.0) is slots


def test_keeps_exactly_ceil_of_the_real_count_per_row():
    torch.manual_seed(0)
    n_real = [100, 40, 7, 1]
    slots = _slots(n_real, 128)
    for rho in (0.5, 0.8, 0.95, 0.99):
        keep = drop_molecules(slots, rho)
        for i, n in enumerate(n_real):
            want = max(1, -(-int(n * (1.0 - rho)) // 1))
            want = max(1, int(torch.ceil(torch.tensor(n * (1.0 - rho))).item()))
            assert int(keep[i].sum()) == want, f"rho={rho} row={i} n_real={n}"


def test_never_keeps_a_padding_slot():
    """A padding slot must never be kept, at any rho."""
    torch.manual_seed(1)
    slots = _slots([5, 2, 1], 64)
    for rho in (0.1, 0.5, 0.9, 0.999):
        keep = drop_molecules(slots, rho)
        assert not (keep & ~slots).any(), f"padding kept at rho={rho}"


def test_at_least_one_slot_survives_even_at_extreme_rho():
    """At least one slot must survive at any rho, so no row pools over an all-False mask."""
    slots = _slots([1, 2, 3], 8)
    keep = drop_molecules(slots, 0.999999)
    assert (keep.sum(1) >= 1).all()


def test_rows_are_drawn_independently():
    """Rows must be drawn independently; a row is a (cell, view) pair."""
    torch.manual_seed(2)
    slots = _slots([50] * 2, 64)
    keep = drop_molecules(slots, 0.5)
    assert not torch.equal(keep[0], keep[1]), "two rows drew the identical subset"


def test_selection_is_uniform_over_real_slots():
    """Selection must be uniform over the real slots, with no position bias."""
    torch.manual_seed(3)
    p_n, n_real, trials = 32, 20, 4000
    slots = _slots([n_real], p_n)
    counts = torch.zeros(p_n)
    for _ in range(trials):
        counts += drop_molecules(slots, 0.5).squeeze(0).float()
    real = counts[:n_real]
    expected = trials * 10 / n_real
    assert (real - expected).abs().max() < 0.12 * expected, real
    assert counts[n_real:].sum() == 0


def test_eval_mode_never_drops_and_train_mode_does(tiny_cfg):
    """The forward's gate must never drop in eval mode and must drop in train mode."""
    from conftest import make_batch
    from scrm_model.model import ScrmModel

    specs = [[[4]] * 8, [[4]] * 8]
    batch = make_batch(specs, seq_len=64, seed=5)
    model = ScrmModel(tiny_cfg)

    model.eval()
    with torch.no_grad():
        model.mol_drop = 0.0
        base = model(batch).window
        model.mol_drop = 0.95
        dropped_in_eval = model(batch).window
    assert torch.equal(base, dropped_in_eval), "mol_drop changed the window in EVAL mode"

    model.train()
    torch.manual_seed(0)
    with torch.no_grad():
        moved = model(batch).window
    assert not torch.allclose(moved, base), "mol_drop did nothing in TRAIN mode"
