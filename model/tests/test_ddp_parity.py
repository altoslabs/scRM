"""Multi-GPU arithmetic must equal single-GPU arithmetic."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import make_batch, mask_some  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
for extra in (REPO / "model", REPO / "target" / "py"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

SEQ = 64
SPECS = [
    [[12, 7], [9]],
    [[11, 8], [7]],
    [[30]],
    [[26]],
    [[6], [5], [7]],
    [[9], [8]],
    [[10, 10]],
    [[19]],
]


def _cfg():
    from scrm_model.config import ArchConfig

    return ArchConfig(
        n_layers_local=2, dim_local=8, n_heads_local=2,
        n_layers_global=2, dim_global=16, n_heads_global=2,
        multiple_of_local=8, multiple_of_global=8,
        proj_hidden=32, proj_dim=16, attn_impl="sdpa", rope_max_pos=64,
    )


CODES = 8


def _build(*, heads: bool):
    """Models seeded exactly as train.py does it: manual_seed before construction, every process."""
    from scrm_model.model import ScrmModel
    from scrm_model.molcontrast import MoleculeCellContrast
    from scrm_model.nbhead import MoleculeCodebook, NBReconstruction
    from scrm_model.projector import build_projector
    from scrm_model.losses import SIGReg

    torch.manual_seed(7)
    cfg = _cfg()
    student = ScrmModel(cfg)
    projector = build_projector(cfg)
    reg = SIGReg(num_slices=4 * cfg.proj_dim)
    nb = mol = None
    if heads:
        nb = (MoleculeCodebook(cfg.dim_global, codes=CODES),
              NBReconstruction(cfg.dim_global, CODES))
        mol = MoleculeCellContrast(cfg.dim_global, temp=0.07, queue=0)
    return student, projector, reg, nb, mol


def _step(batch, *, heads: bool = False):
    from scrm_model.losses import Weights, training_step
    from scrm_model.teacher import make_teacher

    student, projector, reg, nb, mol = _build(heads=heads)
    teacher = make_teacher(student, total_steps=10)
    teacher.update(student, step=0)
    weights = Weights(w_nb=1.0, w_mol=1.0, w_reg_cell=0.5, w_commit=0.25) if heads else Weights()
    out = training_step(student, teacher.model, projector, reg, batch, v=2,
                        weights=weights, nb=nb, mol=mol)
    out.loss.backward()
    trainable = list(student.parameters()) + list(projector.parameters())
    if heads:
        trainable += list(nb[1].parameters()) + list(mol.parameters())
    params = [p for p in trainable if p.grad is not None]
    return out, params


def _worker(rank: int, world: int, init_file: str, results, heads: bool = False):
    torch.set_num_threads(1)
    full = mask_some(make_batch(SPECS, SEQ, seed=3), seed=4, ratio=0.4)

    ref_out, ref_params = _step(full, heads=heads)
    ref_grads = [p.grad.clone() for p in ref_params]
    ref_reg = float(ref_out.terms["reg"].detach())
    ref_terms = {k: float(v.detach()) for k, v in ref_out.terms.items()}

    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world
    )
    from scrm_model.train import allreduce_grads

    half = {k: v[rank * 4 : (rank + 1) * 4] if v.shape[:1] == (8,) else v for k, v in full.items()}
    out, params = _step(half, heads=heads)
    allreduce_grads(params, world)

    checks = {
        "reg_matches": abs(float(out.terms["reg"].detach()) - ref_reg) < 1e-6,
        "guards_match": abs(out.logs["eff_rank_cell"] - ref_out.logs["eff_rank_cell"]) < 1e-4
        and abs(out.logs["eff_rank_window"] - ref_out.logs["eff_rank_window"]) < 1e-4,
        "grads_match": all(
            torch.allclose(g, r, atol=1e-6, rtol=1e-5) for g, r in zip((p.grad for p in params), ref_grads)
        ),
        "max_grad_diff": max(
            float((p.grad - r).abs().max()) for p, r in zip(params, ref_grads)
        ),
        "terms": sorted(out.terms),
        "term_vals": {k: float(v.detach()) for k, v in out.terms.items()},
        "ref_terms": ref_terms,
    }
    results[rank] = checks
    torch.distributed.destroy_process_group()


@pytest.mark.timeout(300)
def test_two_ranks_reproduce_the_single_process_loss_and_gradients(tmp_path):
    mp.set_start_method("spawn", force=True)
    with mp.Manager() as mgr:
        results = mgr.dict()
        mp.spawn(_worker, args=(2, str(tmp_path / "pg"), results), nprocs=2, join=True)
        for rank in (0, 1):
            r = results[rank]
            assert r["reg_matches"], f"rank {rank}: reg loss diverged from single-process"
            assert r["guards_match"], f"rank {rank}: gathered guard readings diverged"
            assert r["grads_match"], f"rank {rank}: gradients diverged, max |diff| {r['max_grad_diff']:.2e}"


@pytest.mark.timeout(300)
def test_the_nb_and_mol_terms_are_also_single_process_arithmetic(tmp_path):
    """The NB and molecule terms must also be single-process arithmetic under DDP."""
    mp.set_start_method("spawn", force=True)
    with mp.Manager() as mgr:
        results = mgr.dict()
        mp.spawn(_worker, args=(2, str(tmp_path / "pg-heads"), results, True), nprocs=2, join=True)
        for rank in (0, 1):
            r = results[rank]
            assert "nb" in r["terms"] and "mol" in r["terms"], \
                f"rank {rank}: the terms under test are absent: {r['terms']}"
            assert r["grads_match"], f"rank {rank}: gradients diverged, max |diff| {r['max_grad_diff']:.2e}"
        ref = results[0]["ref_terms"]
        for name in ref:
            got = sum(results[rank]["term_vals"][name] for rank in (0, 1)) / 2
            assert abs(got - ref[name]) < 1e-6, (
                f"term {name}: rank-mean {got:.6f} against single-process {ref[name]:.6f} "
                f"(diff {abs(got - ref[name]):.2e}) -- the DDP loss is not the single-GPU loss"
            )


def test_seq_len_ladder_holds_its_stated_invariants():
    """Every `seq_len` rung must satisfy the bounds the ladder states."""
    from scrm_model.config import MAX_UMI_LEN, SEQ_LEN, SEQ_LEN_OPTIONS
    from scrm_model.train import parse_args

    assert SEQ_LEN_OPTIONS[0] == SEQ_LEN, "the base rung is THE characterised default"
    for s in SEQ_LEN_OPTIONS:
        assert s % 16384 == 0, "rungs are whole multiples of the base window"
        assert s >= 10 * MAX_UMI_LEN, "the packer's geometry margin"
        assert s <= 49152, "the measured data ceiling: every cell must still fill v=2"
        assert 2 * s < 131070, "u16 segment ids must stay clear of their cap"
    with pytest.raises(SystemExit):
        parse_args(["--corpus", "x", "--out", "/tmp/x", "--seq-len", "20000"])
