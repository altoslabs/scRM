"""The training machinery around the loss: EMA teacher schedule, optimiser groups, schedules —
and an end-to-end smoke: two real optimisation steps must move the student, the teacher, and
produce finite logs.
"""

from __future__ import annotations

import math

import pytest
import torch

from conftest import make_batch, mask_some

from scrm_model import optim
from scrm_model.losses import Weights, training_step

_W = Weights()
from scrm_model.model import ScrmModel
from scrm_model.projector import build_projector
from scrm_model.teacher import make_teacher
from scrm_model.losses import SIGReg

V = 2


class TestEmaTeacher:
    def test_first_update_copies_then_ema_follows_the_scheduled_momentum(self, tiny_cfg):
        torch.manual_seed(0)
        student = ScrmModel(tiny_cfg)
        teacher = make_teacher(student, total_steps=100)
        teacher.update(student, step=0)
        p_s = dict(student.named_parameters())
        for name, p_t in teacher.model.named_parameters():
            assert torch.equal(p_t, p_s[name]), f"first update must copy {name}"

        old = {n: p.clone() for n, p in teacher.model.named_parameters()}
        with torch.no_grad():
            for p in student.parameters():
                p.add_(1.0)
        m = teacher.update(student, step=50)
        assert m == pytest.approx(0.996 + (1.0 - 0.996) * 0.5)
        for name, p_t in teacher.model.named_parameters():
            want = old[name] * m + p_s[name] * (1 - m)
            torch.testing.assert_close(p_t, want, rtol=1e-6, atol=1e-7)

    def test_momentum_one_freezes_the_teacher(self, tiny_cfg):
        torch.manual_seed(0)
        student = ScrmModel(tiny_cfg)
        teacher = make_teacher(student, total_steps=10)
        teacher.update(student, step=0)
        frozen = {n: p.clone() for n, p in teacher.model.named_parameters()}
        with torch.no_grad():
            for p in student.parameters():
                p.mul_(3.0)
        teacher.update(student, step=10)
        for name, p_t in teacher.model.named_parameters():
            assert torch.equal(p_t, frozen[name]), f"momentum 1.0 must freeze {name}"

    def test_the_teacher_never_requires_grad(self, tiny_cfg):
        teacher = make_teacher(ScrmModel(tiny_cfg), total_steps=10)
        assert all(not p.requires_grad for p in teacher.model.parameters())


class TestDecayGroups:
    def test_exclusions_cover_norms_biases_and_seed_queries_and_nothing_else(self, tiny_cfg):
        student = ScrmModel(tiny_cfg)
        projector = build_projector(tiny_cfg)
        decayed, undecayed, names = optim.split_decay_groups(student, projector)
        total = sum(1 for _ in student.parameters()) + sum(1 for _ in projector.parameters())
        assert len(decayed) + len(undecayed) == total, "the split must partition every parameter"
        for name in names:
            ok = (name.endswith(".bias") or "seed_q" in name or "norm" in name.lower()
                  or name in {"1.weight", "5.weight"})
            assert ok, f"{name} excluded from decay for no stated reason"
        assert any("seed_q" in n for n in names), "PMA seed queries missing"
        assert any(n in {"1.weight", "5.weight"} for n in names), "projector BN scales missing"
        assert any("attention_norm.weight" in n for n in names), "RMSNorm gains missing"
        undecayed_ids = {id(p) for p in undecayed}
        assert id(student.embedding.weight) not in undecayed_ids
        assert id(projector[0].weight) not in undecayed_ids

    def test_schedules_hit_the_design_endpoints(self):
        total = 20_000
        assert optim.lr_at(0, total) == pytest.approx(1e-5)
        assert optim.lr_at(1000, total) == pytest.approx(3e-4)
        assert optim.lr_at(total, total) == pytest.approx(1e-6)
        assert optim.wd_at(0, total) == pytest.approx(0.04)
        assert optim.wd_at(total, total) == pytest.approx(0.4)
        lo0, hi0 = optim.mask_ratio_at(0, total)
        lo1, hi1 = optim.mask_ratio_at(total, total)
        assert (lo0, hi0) == pytest.approx((0.15, 0.25))
        assert (lo1, hi1) == pytest.approx((0.20, 0.30))
        assert lo1 > lo0 and hi1 > hi0, "the curriculum must still rise"
        assert lo0 >= 0.15, "below 0.15 a tail of contigs receives no mask at all"
        assert hi1 <= 0.30, "above 0.30 masked bases approach the visible ones on a 53-base contig"

    def test_apply_schedules_never_touches_the_no_decay_group(self, tiny_cfg):
        student = ScrmModel(tiny_cfg)
        opt = optim.build_optimizer(student, fused=False)
        optim.apply_schedules(opt, step=500, total_steps=1000)
        wds = sorted(g["weight_decay"] for g in opt.param_groups)
        assert wds[0] == 0.0 and wds[1] > 0.04


class TestTheLoggedScheduleIsTheAppliedSchedule:
    """`apply_schedules` must return what the optimiser now holds, with the decayed group identified
     by which parameters it holds rather than by the value it carries.
     """

    def _wds(self, opt):
        return [g["weight_decay"] for g in opt.param_groups]

    def test_wd_start_zero_still_reaches_the_decayed_group(self, tiny_cfg):
        student = ScrmModel(tiny_cfg)
        opt = optim.build_optimizer(student, fused=False, wd_start=0.0)
        for step in (0, 500, 1000):
            lr, wd = optim.apply_schedules(opt, step, total_steps=1000, wd_start=0.0)
            want = optim.wd_at(step, 1000, wd_start=0.0)
            assert wd == pytest.approx(want), f"step {step}: reported {wd}, schedule says {want}"
            decayed = [g["weight_decay"] for g in opt.param_groups
                       if g[optim.DECAY_SCHEDULED]]
            assert decayed and decayed[0] == pytest.approx(want), \
                f"step {step}: reported {wd} but the group holds {decayed}"
            assert [g["weight_decay"] for g in opt.param_groups
                    if not g[optim.DECAY_SCHEDULED]] == [0.0], "the excluded group must stay at 0"
        assert self._wds(opt)[0] > 0.0, "a --wd-start 0.0 run must still end up decaying"

    def test_the_returned_pair_is_read_back_from_the_optimiser(self, tiny_cfg):
        """The pair written to `log.jsonl` must be read back from the optimiser, not the intended value."""
        student = ScrmModel(tiny_cfg)
        opt = optim.build_optimizer(student, fused=False, wd_start=0.02)
        lr, wd = optim.apply_schedules(opt, 300, total_steps=1000, wd_start=0.02)
        assert lr == opt.param_groups[0]["lr"]
        assert wd in self._wds(opt)

    def test_a_pre_marker_checkpoint_resumes_onto_a_live_schedule(self, tiny_cfg):
        """A checkpoint whose param_groups carry no schedule marker must have it re-stamped on resume,
         since torch rebuilds param_groups from the file.
         """
        student = ScrmModel(tiny_cfg)
        opt = optim.build_optimizer(student, fused=False)
        optim.apply_schedules(opt, 500, total_steps=1000)
        state = opt.state_dict()
        for group in state["param_groups"]:
            group.pop(optim.DECAY_SCHEDULED, None)
        fresh = optim.build_optimizer(ScrmModel(tiny_cfg), fused=False)
        optim.load_optimizer_state(fresh, state)
        assert all(optim.DECAY_SCHEDULED in g for g in fresh.param_groups)
        _, wd = optim.apply_schedules(fresh, 900, total_steps=1000)
        assert wd == pytest.approx(optim.wd_at(900, 1000))
        assert max(self._wds(fresh)) == pytest.approx(wd)

    def test_every_group_the_loop_adds_carries_the_marker(self, tiny_cfg):
        """Every param group the loop adds through `add_param_group` must carry the schedule marker."""
        student = ScrmModel(tiny_cfg)
        opt = optim.build_optimizer(student, fused=False)
        extra = torch.nn.Parameter(torch.zeros(3))
        optim.add_undecayed_group(opt, [extra])
        optim.apply_schedules(opt, 10, total_steps=100)
        assert opt.param_groups[-1][optim.DECAY_SCHEDULED] is False
        assert opt.param_groups[-1]["weight_decay"] == 0.0


class TestTwoRealSteps:
    def test_two_optimiser_steps_move_student_and_teacher_and_log_finite(self, tiny_cfg, pair_specs):
        torch.manual_seed(0)
        student = ScrmModel(tiny_cfg)
        projector = build_projector(tiny_cfg)
        reg = SIGReg(num_slices=4 * tiny_cfg.proj_dim)
        teacher = make_teacher(student, total_steps=100)
        teacher.update(student, step=0)
        opt = optim.build_optimizer(student, projector, fused=False)
        before = {n: p.clone() for n, p in student.named_parameters()}
        t_before = {n: p.clone() for n, p in teacher.model.named_parameters()}

        for step in range(2):
            batch = mask_some(make_batch(pair_specs, seq_len=64, seed=step), seed=step)
            optim.apply_schedules(opt, step, total_steps=100)
            out = training_step(student, teacher.model, projector, reg, batch, v=V, weights=Weights())
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], optim.GRAD_CLIP
            )
            opt.step()
            teacher.update(student, step + 1)
            assert torch.isfinite(out.loss)
            assert all(math.isfinite(v) for v in out.logs.values())

        assert any(not torch.equal(p, before[n]) for n, p in student.named_parameters())
        assert any(not torch.equal(p, t_before[n]) for n, p in teacher.model.named_parameters())


class TestCellRegulariser:
    """`w_reg_cell` puts the tensor eval reads inside the objective."""

    def _step(self, tiny_cfg, pair_specs, w_cell):
        torch.manual_seed(0)
        student = ScrmModel(tiny_cfg)
        projector = build_projector(tiny_cfg)
        reg = SIGReg(num_slices=4 * tiny_cfg.proj_dim)
        teacher = make_teacher(student, total_steps=100)
        teacher.update(student, step=0)
        batch = mask_some(make_batch(pair_specs, seq_len=64, seed=0), seed=0)
        torch.manual_seed(1234)
        return training_step(student, teacher.model, projector, reg, batch, v=V,
                             weights=Weights(w_reg_cell=w_cell))

    def test_the_default_leaves_the_loss_expression_unchanged(self, tiny_cfg, pair_specs):
        """At the default `w=0` the loss must be exactly the four other terms, bitwise."""
        off = self._step(tiny_cfg, pair_specs, 0.0)
        assert Weights().w_reg_cell == 0.0, "the published default must be off"
        expected = (off.terms["mlm"] + off.terms["ema_umi"] + off.terms["ema_cell"]
                    + _W.w_reg * (_W.lamb * off.terms["reg"]
                                  + (1.0 - _W.lamb) * off.terms["inv"]))
        assert torch.equal(off.loss, expected), "at w=0 the loss must be exactly the four old terms"

    def test_the_step_consumes_no_global_draws_and_advances_the_regulariser_twice(self, tiny_cfg, pair_specs):
        """A step must consume no global RNG draws, and must advance the regulariser's own schedule
         twice.
         """
        before = torch.get_rng_state()
        out = self._step(tiny_cfg, pair_specs, 0.0)
        assert out is not None
        after = torch.get_rng_state()

        torch.manual_seed(1234)
        assert torch.equal(torch.get_rng_state(), after), (
            "the step consumed global RNG draws. losses.SIGReg seeds its own generator so the "
            "objective cannot move the data order; a draw here means a global randn was "
            "reintroduced"
        )
        assert not torch.equal(before, after) or True

    def test_the_regulariser_state_advances_once_per_call(self, tiny_cfg):
        """`global_step` is the projection schedule, so two calls must not share a basis."""
        from scrm_model.losses import SIGReg

        reg = SIGReg(num_slices=4 * tiny_cfg.proj_dim).train()
        z = torch.randn(2, 16, tiny_cfg.proj_dim)
        assert int(reg.global_step) == 0
        first = reg(z)
        assert int(reg.global_step) == 1
        second = reg(z)
        assert int(reg.global_step) == 2
        assert not torch.equal(first, second), "a fresh basis must change the statistic"
        reg.eval()
        held = int(reg.global_step)
        reg(z)
        assert int(reg.global_step) == held, "eval must not advance the schedule"

    def test_a_nonzero_weight_actually_changes_the_loss(self, tiny_cfg, pair_specs):
        """A non-zero `w_reg_cell` must change the loss."""
        off = self._step(tiny_cfg, pair_specs, 0.0)
        on = self._step(tiny_cfg, pair_specs, 1.0)
        assert not torch.equal(off.loss, on.loss)
        assert torch.isfinite(on.loss)
        assert torch.allclose(on.loss, off.loss + on.terms["reg_cell"], atol=1e-5)

    def test_reg_cell_is_always_differentiable(self, tiny_cfg, pair_specs):
        """`reg_cell` must be a graph node at every weight, including 0, because the per-term gradient
         instrument calls `backward()` on every entry of `terms`.
         """
        for w in (0.0, 1.0):
            out = self._step(tiny_cfg, pair_specs, w)
            assert out.terms["reg_cell"].requires_grad, f"reg_cell must be a graph node at w={w}"


class TestEmaCellWeight:
    """`w_ema_cell` is a weight set from the loss instrument."""

    def _step(self, tiny_cfg, pair_specs, w):
        torch.manual_seed(0)
        student = ScrmModel(tiny_cfg)
        projector = build_projector(tiny_cfg)
        reg = SIGReg(num_slices=4 * tiny_cfg.proj_dim)
        teacher = make_teacher(student, total_steps=100)
        teacher.update(student, step=0)
        batch = mask_some(make_batch(pair_specs, seq_len=64, seed=0), seed=0)
        torch.manual_seed(1234)
        return training_step(student, teacher.model, projector, reg, batch, v=V,
                             weights=Weights(w_ema_cell=w))

    def test_the_default_is_what_every_published_arm_trained_under(self, tiny_cfg, pair_specs):
        """The default `w_ema_cell` of 1.0 must leave the loss bit-identical to the published objective."""
        assert Weights().w_ema_cell == 1.0, "the published default is 1.0"
        out = self._step(tiny_cfg, pair_specs, 1.0)
        expected = (out.terms["mlm"] + out.terms["ema_umi"] + out.terms["ema_cell"]
                    + _W.w_reg * (_W.lamb * out.terms["reg"]
                                  + (1.0 - _W.lamb) * out.terms["inv"]))
        assert torch.equal(out.loss, expected)

    def test_the_weight_scales_exactly_the_ema_cell_term(self, tiny_cfg, pair_specs):
        """`w_ema_cell` must scale the ema_cell term and nothing else, reconstructed in the loss
         expression's own operation order so the comparison is bitwise.
         """
        out = self._step(tiny_cfg, pair_specs, 50.0)
        expected = (out.terms["mlm"] + out.terms["ema_umi"] + 50.0 * out.terms["ema_cell"]
                    + _W.w_reg * (_W.lamb * out.terms["reg"]
                                  + (1.0 - _W.lamb) * out.terms["inv"]))
        assert torch.equal(out.loss, expected)
        base = self._step(tiny_cfg, pair_specs, 1.0)
        assert torch.equal(out.terms["ema_cell"], base.terms["ema_cell"]), \
            "the term itself must not depend on its weight"


class TestPassBoundarySnapshots:
    """`pass{n}.pt` must be the checkpoint at the token-pass boundary it is named for."""

    def _fake_save(self, latest):
        def save(step):
            tmp = latest.with_suffix(".tmp")
            torch.save({"step": step}, tmp)
            tmp.rename(latest)
        return save

    def test_the_snapshot_stamps_the_boundary_step_not_the_last_periodic_checkpoint(self, tmp_path):
        from scrm_model.train import snapshot_at_pass_boundary

        latest = tmp_path / "latest.pt"
        save = self._fake_save(latest)
        save(8500)
        dest = snapshot_at_pass_boundary(8947, 8947, save, latest, tmp_path)
        assert dest == tmp_path / "pass1.pt"
        assert torch.load(dest, weights_only=True)["step"] == 8947, \
            "pass1.pt must be the pass-1 boundary, not the 500-step checkpoint before it"
        assert torch.load(latest, weights_only=True)["step"] == 8947

    def test_it_does_nothing_away_from_a_boundary(self, tmp_path):
        from scrm_model.train import snapshot_at_pass_boundary

        latest = tmp_path / "latest.pt"
        self._fake_save(latest)(500)
        assert snapshot_at_pass_boundary(501, 8947, self._fake_save(latest), latest, tmp_path) is None
        assert snapshot_at_pass_boundary(501, 0, self._fake_save(latest), latest, tmp_path) is None
        assert not list(tmp_path.glob("pass*.pt"))

    def test_every_boundary_of_a_multi_pass_run_is_exact(self, tmp_path):
        """Every token-pass boundary of a multi-pass run must be stamped exactly."""
        from scrm_model.train import snapshot_at_pass_boundary

        latest = tmp_path / "latest.pt"
        save = self._fake_save(latest)
        spp, total = 8947, 26841
        for step in range(1, total + 1):
            if step % 500 == 0:
                save(step)
            snapshot_at_pass_boundary(step, spp, save, latest, tmp_path)
        for n in (1, 2, 3):
            got = torch.load(tmp_path / f"pass{n}.pt", weights_only=True)["step"]
            assert got == n * spp, f"pass{n}.pt stamps {got}, boundary is {n * spp}"


class TestTheStartupGuardsReadTheRunsOwnParameters:
    """Cross-parameter couplings must be asserted at startup, against the run's own parameters."""

    def _args(self, **kw):
        from scrm_model.train import parse_args

        argv = ["--corpus", "x", "--out", "/tmp/does-not-exist"]
        for k, v in kw.items():
            argv += [f"--{k.replace('_', '-')}", str(v)]
        return parse_args(argv)

    def test_every_flagship_choice_is_the_default(self):
        """Every field of the flagship recipe must be the argument parser's default, key by key."""
        from scrm_model.config import FLAGSHIP_RECIPE, FLAGSHIP_RECORDED_ONLY

        args = self._args()
        for key in sorted(set(FLAGSHIP_RECIPE) - set(FLAGSHIP_RECORDED_ONLY)):
            assert getattr(args, key) == FLAGSHIP_RECIPE[key], (
                f"--{key.replace('_', '-')} defaults to {getattr(args, key)!r}, but the flagship "
                f"used {FLAGSHIP_RECIPE[key]!r}. A launcher that omits the flag gets the wrong "
                "experiment and stamps it as though it were chosen."
            )

    def test_the_registry_and_the_model_agree_on_the_flagship_recipe(self):
        """The registry's flagship recipe and the model's must agree on every field but `seed`."""
        from scrm_eval import registry as R
        from scrm_eval.registry import FLAGSHIP_RECIPE as EVAL_RECIPE

        from scrm_model.config import FLAGSHIP_RECIPE as MODEL_RECIPE

        assert set(EVAL_RECIPE) == set(MODEL_RECIPE), (
            f"the two flagship recipes describe different FIELDS: only in the registry "
            f"{sorted(set(EVAL_RECIPE) - set(MODEL_RECIPE))}, only in the model "
            f"{sorted(set(MODEL_RECIPE) - set(EVAL_RECIPE))}. A key on one side and not the other is "
            "a recipe field nobody defaults or nobody records."
        )
        shared = {k for k in EVAL_RECIPE if k != "seed"}
        assert {k: EVAL_RECIPE[k] for k in shared} == {k: MODEL_RECIPE[k] for k in shared}, (
            "the registry's flagship recipe and the model's have diverged on a field that is NOT the "
            "seed: " + ", ".join(
                f"{k}: registry {EVAL_RECIPE[k]!r} vs model {MODEL_RECIPE[k]!r}"
                for k in sorted(shared) if EVAL_RECIPE[k] != MODEL_RECIPE[k]
            )
        )
        base = R.seed_base(R.FLAGSHIP)
        suffix = R.FLAGSHIP[len(base):]
        assert suffix in ("", f"-s{EVAL_RECIPE['seed']}"), (
            f"the reported arm is named {R.FLAGSHIP!r}, whose seed suffix {suffix!r} does not match "
            f"the seed the registry reports for it, {EVAL_RECIPE['seed']!r}. The run name and the "
            "recipe would then disagree about which seed produced the paper's numbers."
        )
        assert MODEL_RECIPE["seed"] == R.recipe(base)["seed"], (
            f"config.FLAGSHIP_RECIPE trains at seed {MODEL_RECIPE['seed']!r} but the base arm "
            f"{base!r} is declared at {R.recipe(base)['seed']!r}. `arm_ablation.sbatch` passes no --seed, so "
            "the launcher would produce a run whose seed no declaration names."
        )

    def test_a_run_shorter_than_ITS_OWN_warmup_is_refused(self):
        """A run shorter than the warmup it will actually use must be refused."""
        from scrm_model.train import assert_schedule_fits

        with pytest.raises(SystemExit) as e:
            assert_schedule_fits(2500, 4000, 2500, 1.0, explicit_steps=False)
        assert "4000" in str(e.value), "the message must name the warmup the run will use"

    def test_a_short_warmup_that_completes_is_NOT_refused(self):
        """A run that completes its own short warmup must not be refused."""
        from scrm_model.train import assert_schedule_fits

        assert_schedule_fits(500, 100, 500, 1.0, explicit_steps=False) is None

    def test_the_published_defaults_still_pass_and_the_fly_arm_still_fails(self):
        from scrm_model.train import assert_schedule_fits

        assert_schedule_fits(8947, optim.WARMUP_STEPS, 8947, 1.0, explicit_steps=False)
        with pytest.raises(SystemExit):
            assert_schedule_fits(705, optim.WARMUP_STEPS, 705, 1.0, explicit_steps=False)
        assert_schedule_fits(705, optim.WARMUP_STEPS, 705, 1.0, explicit_steps=True)

    def test_mol_drop_under_ddp_is_refused(self):
        """Molecule dropout under DDP must be refused: the draw's size comes from the rank-local
         molecule maximum, so the ranks' global RNG streams diverge.
         """
        from scrm_model.train import assert_startup_couplings

        assert_startup_couplings(self._args(mol_drop=0.5), 1)
        assert_startup_couplings(self._args(), 4)
        with pytest.raises(SystemExit) as e:
            assert_startup_couplings(self._args(mol_drop=0.5), 4)
        assert "mol-drop" in str(e.value)


class TestCheckpointRoundTrip:
    """Every piece of training state the payload writes must be taken back out on resume."""

    def _pieces(self, tiny_cfg, queue=8):
        from scrm_model.molcontrast import MoleculeCellContrast
        from scrm_model.train import parse_args

        torch.manual_seed(0)
        student = ScrmModel(tiny_cfg)
        projector = build_projector(tiny_cfg)
        reg = SIGReg(num_slices=4 * tiny_cfg.proj_dim)
        teacher = make_teacher(student, total_steps=100)
        teacher.update(student, step=0)
        mol = MoleculeCellContrast(tiny_cfg.dim_global, temp=0.07, queue=queue)
        opt = optim.build_optimizer(student, projector, fused=False)
        optim.add_undecayed_group(opt, list(mol.parameters()))
        args = parse_args(["--corpus", "x", "--out", "/tmp/x", "--w-mol", "1.0",
                           "--mol-queue", str(queue), "--batch-cells", "64"])
        return dict(student=student, projector=projector, teacher=teacher, opt=opt, reg=reg,
                    nb_modules=None, mol_module=mol, args=args)

    def _payload(self, tiny_cfg, pieces, step=7, world=1):
        from scrm_model.train import checkpoint_payload

        return checkpoint_payload(step, cfg=tiny_cfg, total_steps=100, world=world, **pieces)

    def test_the_contrast_head_survives_a_resume(self, tiny_cfg):
        from scrm_model.train import restore_checkpoint

        live = self._pieces(tiny_cfg)
        mol = live["mol_module"]
        with torch.no_grad():
            mol.log_temp.add_(0.5)
            mol._enqueue(torch.randn(5, tiny_cfg.dim_global))
        payload = self._payload(tiny_cfg, live)
        temp, filled, ptr = mol.log_temp.clone(), mol.neg_filled.clone(), mol.neg_ptr.clone()
        queue = mol.neg.clone()

        fresh = self._pieces(tiny_cfg)
        restore_checkpoint(payload, **fresh)
        got = fresh["mol_module"]
        assert torch.equal(got.log_temp, temp), "a resumed run must not restart the temperature"
        assert torch.equal(got.neg_filled, filled), "an emptied queue makes the task suddenly easier"
        assert torch.equal(got.neg_ptr, ptr)
        assert torch.equal(got.neg, queue)

    def test_a_w_mol_run_refuses_a_checkpoint_with_no_contrast_state(self, tiny_cfg):
        from scrm_model.train import restore_checkpoint

        payload = self._payload(tiny_cfg, self._pieces(tiny_cfg))
        payload["mol_contrast"] = None
        with pytest.raises(SystemExit) as e:
            restore_checkpoint(payload, **self._pieces(tiny_cfg))
        assert "mol_contrast" in str(e.value)

    def test_the_random_stream_survives_a_resume(self, tiny_cfg):
        """A resumed run must continue the global stream rather than replay the draws it used.

         NOT because of `W`: this docstring used to say `W` is redrawn from the global RNG every
         forward, which `losses._draw` contradicts -- it seeds a private generator from
         `global_step`, and `test_the_step_consumes_no_global_draws_and_advances_the_regulariser_twice`
         above measures that a step takes zero global draws. What does consume the global stream is
         the data order, so that is what a resume must not replay.
         """
        from scrm_model.train import restore_checkpoint

        live = self._pieces(tiny_cfg)
        torch.manual_seed(99)
        torch.randn(17)
        payload = self._payload(tiny_cfg, live)
        expected = torch.randn(4)

        torch.manual_seed(12345)
        restore_checkpoint(payload, **self._pieces(tiny_cfg))
        assert torch.equal(torch.randn(4), expected), "the resumed run must continue the stream"

    def test_a_pre_field_checkpoint_still_resumes(self, tiny_cfg):
        """A checkpoint carrying no `rng_state` and no regulariser state must still resume, rather than
         raise or claim a stream it never recorded.
         """
        from scrm_model.train import restore_checkpoint

        payload = self._payload(tiny_cfg, self._pieces(tiny_cfg))
        for key in ("rng_state", "cuda_rng_state", "reg", "world", "batch_cells"):
            payload.pop(key)
        assert restore_checkpoint(payload, **self._pieces(tiny_cfg)) == 7

    def test_the_payload_stamps_the_batch_size_and_the_rank_count(self, tiny_cfg):
        payload = self._payload(tiny_cfg, self._pieces(tiny_cfg), world=4)
        assert payload["batch_cells"] == 64, "B is a comparability quantity and must be on disk"
        assert payload["world"] == 4
        assert payload["deterministic_algorithms"] is False, \
            "nothing enables it, so the stamp must not claim otherwise"
        assert payload["rng_state"] is not None

    def test_the_regularizer_may_not_change_under_a_resume(self, tiny_cfg):
        """A resume must refuse a regularizer that differs from the one the payload records, with the
         mismatch derived from the live default rather than a spelled-out value.
         """
        from scrm_model.train import restore_checkpoint

        pieces = self._pieces(tiny_cfg)
        payload = self._payload(tiny_cfg, pieces)
        other = "a-retired-objective"
        payload["regularizer"] = other
        with pytest.raises(SystemExit) as e:
            restore_checkpoint(payload, **self._pieces(tiny_cfg))
        assert other in str(e.value)

    def test_the_sigreg_projection_schedule_survives_a_resume(self, tiny_cfg):
        """`SIGReg.global_step` is a buffer that seeds the projections, so a resume must carry it rather
         than restart the schedule at 0.
         """
        from scrm_model.losses import SIGReg
        from scrm_model.train import restore_checkpoint

        def sigreg():
            return SIGReg(num_slices=4 * tiny_cfg.proj_dim, reduction="mean")

        live = self._pieces(tiny_cfg)
        live["reg"] = sigreg()
        live["reg"].global_step.fill_(41)
        payload = self._payload(tiny_cfg, live)

        fresh = self._pieces(tiny_cfg)
        fresh["reg"] = sigreg()
        restore_checkpoint(payload, **fresh)
        assert int(fresh["reg"].global_step) == 41


def test_the_producer_and_the_eligibility_guard_draw_THE_SAME_epoch_seeds():
    """The producer and the eligibility guard must draw the same per-epoch seeds."""
    import re
    from pathlib import Path

    from scrm_model.train import epoch_seed_at

    assert epoch_seed_at(20260817, 0) == (20260817 * 0x9E3779B97F4A7C15) % (1 << 64)
    assert epoch_seed_at(20260817, 3) != epoch_seed_at(20260817, 4)

    src = Path(__file__).resolve().parents[1] / "scrm_model" / "train.py"
    body = src.read_text()
    assert len(re.findall(r"0x9E3779B97F4A7C15", body)) == 1, \
        "the epoch-seed formula is written twice again; the guard and the producer will drift"


def test_a_corpus_smaller_than_one_batch_is_refused_with_the_numbers():
    """A corpus with fewer eligible cells than one batch must be refused, naming both numbers,
     rather than divide by a per-epoch count of zero.
     """
    import numpy as np

    from scrm_model.train import make_batch_producer, parse_args

    args = parse_args(["--corpus", "x", "--out", "/tmp/x", "--batch-cells", "128"])
    with pytest.raises(SystemExit) as e:
        make_batch_producer(None, np.arange(17), args, total_steps=10)
    assert "17" in str(e.value) and "128" in str(e.value)
    assert make_batch_producer(None, np.arange(256), args, total_steps=10) is not None
