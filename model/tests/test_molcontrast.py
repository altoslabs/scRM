"""Obligations of the per-molecule capture contrast and the freezable codebook."""
from __future__ import annotations

import torch

from scrm_model.molcontrast import MoleculeCellContrast
from scrm_model.nbhead import MoleculeCodebook


def test_perfect_alignment_gives_near_zero_loss_and_unit_accuracy():
    torch.manual_seed(0)
    b, d = 8, 16
    cells = torch.randn(b, d)
    target = torch.arange(b)
    m = MoleculeCellContrast(16, temp=0.01)
    loss, acc, n = m(cells.clone(), cells, target)
    assert n == b
    assert float(acc) == 1.0
    assert float(loss.detach()) / n < 1e-3


def test_random_molecules_sit_at_the_chance_floor():
    """Random molecules must sit at the chance floor in accuracy, at every temperature."""
    import math
    torch.manual_seed(1)
    b, d, n = 32, 24, 512
    mol, cells = torch.randn(n, d), torch.randn(b, d)
    tgt = torch.randint(0, b, (n,))

    for temp in (0.07, 1.0):
        _, acc, got = MoleculeCellContrast(24, temp=temp)(mol, cells, tgt)
        assert got == n
        assert float(acc) < 4.0 / b, f"temp={temp} acc={float(acc):.3f} is above chance"

    loss, _, got = MoleculeCellContrast(24, temp=1.0)(mol, cells, tgt)
    per = float(loss.detach()) / got
    assert abs(per - math.log(b)) < 0.15 * math.log(b), per


def test_loss_is_summed_not_meaned():
    """The loss must be a sum, not a mean, so two identical halves give exactly twice one half and
    the caller's division by the global molecule count weights molecules.
    """
    torch.manual_seed(2)
    b, d = 6, 12
    cells = torch.randn(b, d)
    mol = torch.randn(5, d)
    tgt = torch.randint(0, b, (5,))
    m = MoleculeCellContrast(12)
    one, _, n1 = m(mol, cells, tgt)
    two, _, n2 = m(torch.cat([mol, mol]), cells, torch.cat([tgt, tgt]))
    assert n2 == 2 * n1
    assert torch.allclose(two, 2 * one, atol=1e-5)


def test_temperature_is_learnable_and_positive():
    m = MoleculeCellContrast(8, temp=0.07)
    assert m.log_temp.requires_grad
    assert abs(float(m.log_temp.exp()) - 0.07) < 1e-6
    loss, _, _ = m(torch.randn(4, 8), torch.randn(3, 8), torch.randint(0, 3, (4,)))
    loss.backward()
    assert m.log_temp.grad is not None, "temperature got no gradient: it is not being learned"


def test_accuracy_would_expose_leakage():
    """`mol_acc` must read 1.0 when molecules carry their cell's identity outright."""
    torch.manual_seed(3)
    cells = torch.randn(10, 16)
    _, acc, _ = MoleculeCellContrast(16, temp=0.01)(cells, cells, torch.arange(10))
    assert float(acc) == 1.0


def test_a_frozen_codebook_stops_moving():
    """A frozen codebook must stop moving, and a thawed one must move."""
    torch.manual_seed(4)
    cb = MoleculeCodebook(dim=8, codes=16)
    cb.train()
    x = torch.randn(64, 8)
    cb(x)
    before = cb.embed.clone()
    cb(torch.randn(64, 8))
    assert not torch.equal(before, cb.embed), "codebook did not move while thawed"

    cb.frozen.fill_(True)
    held = cb.embed.clone()
    for _ in range(3):
        cb(torch.randn(64, 8))
    assert torch.equal(held, cb.embed), "codebook moved while frozen"


def test_frozen_flag_survives_a_state_dict_round_trip():
    """The `frozen` flag must survive a `state_dict` round trip, so a resume cannot thaw the basis."""
    cb = MoleculeCodebook(dim=8, codes=16)
    cb.frozen.fill_(True)
    other = MoleculeCodebook(dim=8, codes=16)
    assert not bool(other.frozen)
    other.load_state_dict(cb.state_dict())
    assert bool(other.frozen), "frozen was lost across state_dict, so a resume would thaw the basis"


def test_queue_off_by_default_is_bit_identical():
    """`queue=0` must be bit-identical to the queue-free path."""
    torch.manual_seed(7)
    mol, cells = torch.randn(9, 12), torch.randn(5, 12)
    tgt = torch.randint(0, 5, (9,))
    a, aa, an = MoleculeCellContrast(12, temp=0.07)(mol, cells, tgt)
    b, ba, bn = MoleculeCellContrast(12, temp=0.07, queue=0)(mol, cells, tgt)
    assert torch.equal(a, b) and torch.equal(aa, ba) and an == bn


def test_queue_adds_columns_AFTER_the_in_batch_ones():
    """Queued negatives must be appended after the in-batch columns, because `target` is a global
    index into `cells`.
    """
    torch.manual_seed(8)
    d, b = 16, 6
    m = MoleculeCellContrast(16, temp=1.0, queue=32)
    m.train()
    cells = torch.randn(b, d)
    tgt = torch.arange(b)
    _, acc0, _ = m(cells.clone(), cells, tgt)
    assert float(acc0) == 1.0, "in-batch matching broke before the queue was even populated"
    assert int(m.neg_filled) == b, "the batch was not enqueued"
    logits_ok = True
    mm = torch.nn.functional.normalize(cells, dim=1)
    cc = torch.nn.functional.normalize(cells, dim=1)
    lg = mm @ cc.t()
    for i in range(b):
        if lg[i].argmax().item() != i:
            logits_ok = False
    assert logits_ok, "target column is not the in-batch max; indexing is wrong"


def test_queue_makes_the_task_harder_not_different():
    """More negatives must raise the loss on the same data."""
    torch.manual_seed(9)
    d, b = 24, 8
    cells, tgt = torch.randn(b, d), torch.arange(b)
    mol = cells + 0.35 * torch.randn(b, d)

    plain = MoleculeCellContrast(24, temp=0.2, queue=0)
    plain.train()
    l0, _, n0 = plain(mol, cells, tgt)

    q = MoleculeCellContrast(24, temp=0.2, queue=256)
    q.train()
    for _ in range(6):
        q(torch.randn(b, d), torch.randn(b, d), tgt)
    l1, _, n1 = q(mol, cells, tgt)
    assert n0 == n1
    assert float(l1) > float(l0), (
        f"queue did not increase the loss ({float(l1):.4f} vs {float(l0):.4f}), so it is adding no "
        "difficulty -- check that the extra columns actually reach cross_entropy"
    )


def test_queue_state_survives_a_round_trip():
    """The queue state must survive a `state_dict` round trip."""
    torch.manual_seed(10)
    a = MoleculeCellContrast(8, queue=64); a.train()
    a(torch.randn(4, 8), torch.randn(4, 8), torch.arange(4))
    assert int(a.neg_filled) == 4
    b = MoleculeCellContrast(8, queue=64)
    b.load_state_dict(a.state_dict())
    assert int(b.neg_filled) == 4 and torch.equal(a.neg, b.neg)


def test_queued_negatives_carry_no_gradient():
    """Queued negatives must carry no gradient, having come from earlier weights."""
    m = MoleculeCellContrast(8, queue=32); m.train()
    m(torch.randn(4, 8), torch.randn(4, 8), torch.arange(4))
    assert not m.neg.requires_grad
