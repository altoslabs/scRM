"""The four training guards."""

from __future__ import annotations

import copy

import pytest
import torch

from conftest import N, NEWLINE, TAB, make_batch, mask_some

from scrm_model import guards
from scrm_model.losses import Weights, training_step
from scrm_model.model import ScrmModel
from scrm_model.projector import build_projector
from scrm_model.losses import SIGReg

V = 2


def build(tiny_cfg):
    torch.manual_seed(0)
    student = ScrmModel(tiny_cfg)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    projector = build_projector(tiny_cfg)
    reg = SIGReg(num_slices=4 * tiny_cfg.proj_dim)
    return student, teacher, projector, reg


def step(student, teacher, projector, reg, batch, *, weights=Weights(), seed=7):
    torch.manual_seed(seed)
    return training_step(student, teacher, projector, reg, batch, v=V, weights=weights)


class TestGuard4PaddingBitwise:
    def test_backbone_outputs_are_bitwise_equal_and_finite_under_junk_padding(self, tiny_cfg, pair_specs):
        student, *_ = build(tiny_cfg)
        clean = make_batch(pair_specs, seq_len=64, seed=3)
        junk = make_batch(pair_specs, seq_len=64, seed=3, junk_padding=True)
        assert torch.equal(clean["tokens"][clean["valid"]], junk["tokens"][junk["valid"]])
        assert not torch.equal(clean["tokens"], junk["tokens"]), "the fixture must actually differ"

        with torch.no_grad():
            a = student(clean)
            b = student(junk)

        for name, t in [("h_bytes", a.h_bytes), ("umi", a.umi), ("window", a.window)]:
            assert torch.isfinite(t).all(), f"{name} is not finite"
        assert torch.equal(a.window, b.window), "padding contents leaked into the window embedding"
        assert torch.equal(a.umi, b.umi), "padding contents leaked into the UMI embeddings"
        assert torch.equal(a.h_bytes, b.h_bytes), "padding contents leaked into the byte states"

    def test_the_whole_loss_is_bitwise_equal_under_junk_padding(self, tiny_cfg, pair_specs):
        student, teacher, projector, reg = build(tiny_cfg)
        clean = mask_some(make_batch(pair_specs, seq_len=64, seed=3))
        junk = mask_some(make_batch(pair_specs, seq_len=64, seed=3, junk_padding=True))
        reg_state = {k: v.clone() for k, v in reg.state_dict().items()}
        out_a = step(student, teacher, projector, reg, clean)
        reg.load_state_dict(reg_state)
        out_b = step(student, teacher, projector, reg, junk)
        for k in out_a.terms:
            assert torch.equal(out_a.terms[k], out_b.terms[k]), f"loss term {k} read padding"

    def test_finite_at_the_padding_extremes(self, tiny_cfg):
        student, *_ = build(tiny_cfg)
        specs = [[[40]], [[5], [4], [6], [4], [7], [5]], [[30]], [[6, 6], [8]]]
        with torch.no_grad():
            out = student(make_batch(specs, seq_len=64, seed=5))
        assert torch.isfinite(out.window).all()
        assert torch.isfinite(out.umi).all()

    def test_padding_umi_slots_are_exactly_zero(self, tiny_cfg, pair_specs):
        student, *_ = build(tiny_cfg)
        batch = make_batch(pair_specs, seq_len=64, seed=3)
        with torch.no_grad():
            out = student(batch)
        for w, n in enumerate(batch["n_patches"].tolist()):
            tail = out.umi[w, n:]
            assert (tail == 0).all(), f"window {w}: padding UMI slots are not zeroed"


class TestGuard3GradientAudit:
    def test_every_student_and_projector_parameter_gets_gradient(self, tiny_cfg, pair_specs):
        student, teacher, projector, reg = build(tiny_cfg)
        batch = mask_some(make_batch(pair_specs, seq_len=64, seed=3, n_share=0.05))
        out = step(student, teacher, projector, reg, batch)
        out.loss.backward()
        dead = guards.dead_parameters(guards.gradient_audit(student, prefix="student.")) + guards.dead_parameters(
            guards.gradient_audit(projector, prefix="projector.")
        )
        assert dead == [], f"parameters with no gradient: {dead}"

    def test_the_audit_catches_an_untrained_view_pool(self, tiny_cfg, pair_specs):
        student, teacher, projector, reg = build(tiny_cfg)
        batch = mask_some(make_batch(pair_specs, seq_len=64, seed=3))
        out = step(student, teacher, projector, reg, batch, weights=Weights(w_ema_cell=0.0))
        out.loss.backward()
        dead = guards.dead_parameters(guards.gradient_audit(student))
        assert any("view_pool" in name for name in dead), f"audit missed the dead view pool: {dead}"

    def test_the_audit_catches_an_untrained_projector(self, tiny_cfg, pair_specs):
        student, teacher, projector, reg = build(tiny_cfg)
        batch = mask_some(make_batch(pair_specs, seq_len=64, seed=3))
        out = step(student, teacher, projector, reg, batch, weights=Weights(w_reg=0.0))
        out.loss.backward()
        dead = guards.dead_parameters(guards.gradient_audit(projector))
        assert len(dead) == len(list(projector.parameters())), "audit missed the dead projector"


class TestGuards12RankAndNormLogs:
    def test_every_step_logs_rank_and_norm_at_all_three_places(self, tiny_cfg, pair_specs):
        student, teacher, projector, reg = build(tiny_cfg)
        batch = mask_some(make_batch(pair_specs, seq_len=64, seed=3))
        out = step(student, teacher, projector, reg, batch)
        for key in ("eff_rank_window", "eff_rank_proj", "eff_rank_cell", "cell_norm"):
            assert key in out.logs, f"guard log {key} missing"
            assert out.logs[key] > 0
        assert 1.0 <= out.logs["eff_rank_window"] <= min(len(pair_specs), tiny_cfg.dim_global)
        assert 1.0 <= out.logs["eff_rank_cell"] <= min(len(pair_specs) // V, tiny_cfg.dim_global)

    def test_effective_rank_reports_collapse_as_one(self):
        collapsed = torch.ones(32, 16) * 3.7
        assert guards.effective_rank(collapsed) == pytest.approx(1.0)

    def test_effective_rank_matches_the_eval_suite_definition(self):
        from scrm_eval.metrics import effective_rank as eval_rank

        torch.manual_seed(0)
        x = torch.randn(48, 12)
        assert guards.effective_rank(x) == pytest.approx(eval_rank(x.numpy()), rel=1e-6)


class TestPerTermGradientNorms:
    def test_the_partition_table_holds_empirically(self, tiny_cfg, pair_specs):
        student, teacher, projector, reg = build(tiny_cfg)
        batch = mask_some(make_batch(pair_specs, seq_len=64, seed=3, n_share=0.05))
        out = step(student, teacher, projector, reg, batch)
        norms = guards.per_term_gradient_norms(out.terms, guards.parameter_groups(student, projector))

        must_be_zero = [
            ("mlm", "pools"), ("mlm", "global"), ("mlm", "window_pool"),
            ("mlm", "view_pool"), ("mlm", "projector"),
            ("ema_umi", "window_pool"), ("ema_umi", "view_pool"), ("ema_umi", "projector"),
            ("ema_cell", "projector"),
            ("reg", "view_pool"), ("inv", "view_pool"),
        ]
        must_reach = [
            ("mlm", "local"),
            ("ema_umi", "local"), ("ema_umi", "pools"), ("ema_umi", "global"),
            ("ema_cell", "view_pool"), ("ema_cell", "window_pool"), ("ema_cell", "local"),
            ("reg", "projector"), ("reg", "window_pool"), ("reg", "local"),
            ("inv", "projector"), ("inv", "global"),
        ]
        for term, group in must_be_zero:
            assert norms[f"gnorm/{term}/{group}"] == 0.0, f"{term} illegally reaches {group}"
        for term, group in must_reach:
            assert norms[f"gnorm/{term}/{group}"] > 0.0, f"{term} fails to reach {group}"


class TestLossAssembly:
    def test_the_four_terms_are_present_and_finite(self, tiny_cfg, pair_specs):
        student, teacher, projector, reg = build(tiny_cfg)
        batch = mask_some(make_batch(pair_specs, seq_len=64, seed=3, n_share=0.1))
        out = step(student, teacher, projector, reg, batch)
        for term in ("mlm", "ema_umi", "ema_cell", "reg", "inv"):
            assert term in out.terms
            assert torch.isfinite(out.terms[term]), f"{term} is not finite"

    def test_mlm_reads_only_kept_positions(self, tiny_cfg, pair_specs):
        student, teacher, projector, reg = build(tiny_cfg)
        batch = mask_some(make_batch(pair_specs, seq_len=64, seed=3))
        shifted = dict(batch)
        shifted["mlm_keep"] = torch.roll(batch["mlm_keep"], 1, dims=1) & batch["valid"] & (
            batch["tokens"] != TAB
        ) & (batch["tokens"] != NEWLINE) & (batch["tokens"] != N)
        a = step(student, teacher, projector, reg, batch)
        b = step(student, teacher, projector, reg, shifted)
        assert not torch.equal(a.terms["mlm"], b.terms["mlm"]), "shifting the mask did not move the loss"


class TestTheInstrumentCanSeeEveryWeightedTerM:
    """Every weighted loss term must have a row in the instrument's partition table."""

    def _heads(self, tiny_cfg):
        from scrm_model.molcontrast import MoleculeCellContrast
        from scrm_model.nbhead import MoleculeCodebook, NBReconstruction

        nb = (MoleculeCodebook(tiny_cfg.dim_global, codes=8),
              NBReconstruction(tiny_cfg.dim_global, 8))
        return nb, MoleculeCellContrast(tiny_cfg.dim_global, temp=0.07, queue=0)

    def test_the_nb_and_mol_heads_have_rows_and_their_terms_reach_them(self, tiny_cfg, pair_specs):
        student, teacher, projector, reg = build(tiny_cfg)
        nb, mol = self._heads(tiny_cfg)
        batch = mask_some(make_batch(pair_specs, seq_len=64, seed=3, n_share=0.05))
        torch.manual_seed(7)
        out = training_step(student, teacher, projector, reg, batch, v=V,
                            weights=Weights(w_nb=1.0, w_mol=1.0, w_commit=0.25), nb=nb, mol=mol)
        groups = guards.parameter_groups(student, projector, nb=nb, mol=mol)
        assert "nb_head" in groups and "mol_temp" in groups, groups.keys()
        norms = guards.per_term_gradient_norms(out.terms, groups)
        assert norms["gnorm/nb/nb_head"] > 0.0, "the NB term's own head must be measurable"
        assert norms["gnorm/mol/mol_temp"] > 0.0, "the learned temperature must be measurable"
        assert norms["gnorm/mlm/nb_head"] == 0.0
        assert norms["gnorm/mlm/mol_temp"] == 0.0
        assert norms["gnorm/nb/mol_temp"] == 0.0

    def test_the_rows_are_absent_exactly_when_the_heads_are(self, tiny_cfg):
        student, _, projector, _ = build(tiny_cfg)
        groups = guards.parameter_groups(student, projector)
        assert "nb_head" not in groups and "mol_temp" not in groups, \
            "a head that was never built has no parameters to measure"
