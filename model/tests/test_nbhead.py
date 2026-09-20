"""The NB head over a learned molecule codebook: counts, likelihood, and the gated default."""

from __future__ import annotations

import torch

from scrm_model.nbhead import MoleculeCodebook, NBReconstruction, code_counts


def test_code_counts_matches_a_naive_count():
    """scatter_add over a linearised (group, code) index must equal the obvious double loop."""
    torch.manual_seed(0)
    n, codes, groups = 500, 7, 4
    idx = torch.randint(0, codes, (n,))
    group = torch.randint(0, groups, (n,))
    got = code_counts(idx, group, groups, codes)
    want = torch.zeros(groups, codes)
    for i in range(n):
        want[group[i], idx[i]] += 1
    assert torch.equal(got, want)
    assert got.sum().item() == n


def test_nb_loss_is_lower_when_the_prediction_matches():
    """A cell whose predicted composition matches its counts must score better than a mismatch."""
    torch.manual_seed(0)
    codes, dim = 6, 8
    head = NBReconstruction(dim, codes)
    counts = torch.tensor([[10.0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 10.0]])
    cell = torch.zeros(2, dim)
    with torch.no_grad():
        head.rate.weight.zero_()
        head.rate.bias.copy_(torch.tensor([5.0, 0, 0, 0, 0, -5.0]))
    aligned = head(cell, counts[:1])
    misaligned = head(cell, counts[1:])
    assert aligned.item() < misaligned.item()


def test_codebook_is_deterministic_in_eval_and_assigns_every_molecule():
    """In eval the codebook must not move, and every molecule must get exactly one code."""
    torch.manual_seed(0)
    cb = MoleculeCodebook(dim=8, codes=16)
    cb.eval()
    x = torch.randn(64, 8)
    before = cb.embed.clone()
    q1, i1, c1 = cb(x)
    q2, i2, c2 = cb(x)
    assert torch.equal(i1, i2), "assignment must be deterministic"
    assert torch.equal(cb.embed, before), "eval must not update the codebook"
    assert i1.shape == (64,) and int(i1.min()) >= 0 and int(i1.max()) < 16
    assert q1.shape == x.shape


def test_straight_through_passes_gradient_to_the_encoder():
    """The quantiser must be transparent to gradient, or the trunk never learns from this term."""
    cb = MoleculeCodebook(dim=8, codes=16)
    cb.eval()
    x = torch.randn(32, 8, requires_grad=True)
    q, _, _ = cb(x)
    q.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
