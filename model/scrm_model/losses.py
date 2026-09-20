"""The assembled loss: four forwards, one teacher read at two levels, four terms.

FIRST-PARTY, and here is exactly what that rests on. `SIGReg` below takes from the published
description of the method (LeJEPA, Balestriero & LeCun, arXiv:2511.08544) and of the statistic it
evaluates (Epps & Pulley 1983, Biometrika 70(3)): the 17 integration points, the trapezoidal rule,
the `exp(-t^2/2)` weight and the `x N` scaling. What the papers do NOT specify, and what is
therefore ours, is the half-line symmetry fold with half-weight endpoints, the cos/sin real
decomposition in place of a complex characteristic function, and `T_MAX = 3` rather than the
paper's recommended domain. No third-party code is vendored anywhere in `model/scrm_model/`.

What this docstring does not assert is anything about process: what a reader can check is the
itemisation above, against the code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed
import torch.distributed.nn

from scrm_model import nbhead
import torch.nn.functional as F
from torch import nn

from . import guards
from .model import ScrmModel


#: `T_MAX` is the half-line bound; the integrand's symmetry doubles it, so the domain is [-3, 3].
#: The seed below is reduced `% (1 << 63)` to stay inside `manual_seed`'s range.
T_MAX = 3.0
N_POINTS = 17

_GOLDEN64 = 0x9E3779B97F4A7C15


class SIGReg(nn.Module):
    """The Epps-Pulley normality statistic on random 1-D slices, written first-party.

    The method is LeJEPA's (Balestriero & LeCun, arXiv:2511.08544) and the statistic is
    Epps & Pulley 1983, Biometrika 70(3). The module docstring itemises which parts come from
    those papers and which are this file's.
    """

    def __init__(self, num_slices: int, reduction: str = "mean"):
        super().__init__()
        if reduction != "mean":
            raise ValueError(f"reduction must be 'mean', got {reduction!r}")
        self.num_slices = num_slices
        self.reduction = reduction

        t = torch.linspace(0.0, T_MAX, N_POINTS, dtype=torch.float32)
        phi = torch.exp(-t**2 / 2)
        h = T_MAX / (N_POINTS - 1)
        trapezoid = torch.full((N_POINTS,), h, dtype=torch.float32)
        trapezoid[0] = h / 2
        trapezoid[-1] = h / 2

        self.register_buffer("t", t, persistent=False)
        self.register_buffer("phi", phi, persistent=False)
        self.register_buffer("weights", 2 * phi * trapezoid, persistent=False)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.int64), persistent=True)

    def _draw(self, z: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.global_step) * _GOLDEN64 % (1 << 63))
        W = torch.randn(z.shape[-1], self.num_slices, generator=generator, dtype=torch.float32)
        W = W.to(device=z.device, dtype=z.dtype)
        return W / W.norm(dim=0, keepdim=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        W = self._draw(z)
        if self.training:
            self.global_step += 1

        p = (z @ W).float()
        tp = self.t * p.unsqueeze(-1)
        err = (torch.cos(tp).mean(dim=1) - self.phi)**2 + torch.sin(tp).mean(dim=1)**2
        return (err @ self.weights * z.shape[1]).mean()


@dataclass(frozen=True)
class Weights:
    """The four loss-term weights, and `lamb`, which is not one of them: `lamb` belongs to
    the convex mix in `training_step` (see `config.FLAGSHIP_RECIPE`).
    """

    w_mlm: float = 1.0
    w_ema_umi: float = 1.0
    w_ema_cell: float = 1.0
    w_reg: float = 1.0
    lamb: float = 0.9
    w_reg_cell: float = 0.0
    w_nb: float = 0.0
    w_mol: float = 0.0
    smooth_l1_beta: float = 1.0
    w_commit: float = 1.0


@dataclass
class StepOut:
    loss: torch.Tensor
    terms: dict[str, torch.Tensor]
    logs: dict[str, float]


def _dist() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _gather_windows(win: torch.Tensor) -> torch.Tensor:
    """All-gather window rows (autograd-aware) BEFORE the projector under DDP."""
    if _dist():
        return torch.cat(torch.distributed.nn.all_gather(win.contiguous()), dim=0)
    return win


def _exact_mean(sum_loss: torch.Tensor, n_local: int) -> torch.Tensor:
    """A mean over a GLOBAL count that gradient-averaging cannot distort."""
    if _dist():
        n = torch.tensor([float(n_local)], device=sum_loss.device)
        torch.distributed.all_reduce(n)
        return sum_loss * (torch.distributed.get_world_size() / n.squeeze(0))
    return sum_loss / n_local


def _gather_rows(x: torch.Tensor) -> torch.Tensor:
    """Logging-only row concat across ranks (no grad), so the collapse guards read the GLOBAL
    batch's rank.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world = torch.distributed.get_world_size()
        out = [torch.empty_like(x) for _ in range(world)]
        torch.distributed.all_gather(out, x.contiguous())
        return torch.cat(out, dim=0)
    return x


def training_step(
    student: ScrmModel,
    teacher: ScrmModel,
    projector: nn.Module,
    reg: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    v: int = 2,
    weights: Weights = Weights(),
    nb: tuple[nn.Module, nn.Module] | None = None,
    mol: nn.Module | None = None,
) -> StepOut:
    w_n = batch["tokens"].shape[0]
    assert w_n % v == 0, "windows must be cell-major groups of v"
    b_n = w_n // v

    with torch.no_grad():
        pre = student.precompute(batch)
    out_s = student(batch, tokens=batch["tokens_student"], pre=pre)
    with torch.no_grad():
        out_t = teacher(batch, pre=pre)

    keep = batch["mlm_keep"]
    assert bool(keep.any()), "a step with nothing masked trains nothing and hides a masker bug"
    logits = student.mlm_logits(out_s.h_bytes)
    mlm = _exact_mean(
        F.cross_entropy(logits[keep], batch["tokens"][keep].long(), reduction="sum"), int(keep.sum())
    )

    p_n = out_s.umi.shape[1]
    real = torch.arange(p_n, device=out_s.umi.device).expand(w_n, p_n) < batch["n_patches"][:, None]
    ema_umi = _exact_mean(
        F.smooth_l1_loss(out_s.umi[real], out_t.umi[real], reduction="sum",
                         beta=weights.smooth_l1_beta), out_s.umi[real].numel()
    )

    cell_s = student.pool_views(out_s.window, v)
    with torch.no_grad():
        cell_t = teacher.pool_views(out_t.window, v)
    ema_cell = F.smooth_l1_loss(cell_s, cell_t, beta=weights.smooth_l1_beta)

    win_all = _gather_windows(out_s.window.view(b_n * v, -1))
    flat = projector(win_all)
    b_all = flat.shape[0] // v
    proj = flat.view(b_all, v, -1).transpose(0, 1).float()
    reg_loss = reg(proj)
    inv = (proj - proj.mean(dim=0, keepdim=True)).square().mean()
    reg_mix = weights.lamb * reg_loss + (1.0 - weights.lamb) * inv

    cell_g = _gather_windows(cell_s)
    reg_cell = reg(cell_g.float().unsqueeze(0))

    if nb is not None:
        codebook, nbrec = nb
        umi_real = out_s.umi[real]
        _, code_idx, commit_sum = codebook(umi_real)
        commit = _exact_mean(commit_sum, umi_real.numel())
        win_ids = torch.arange(w_n, device=out_s.umi.device).unsqueeze(1).expand(real.shape)
        win_of_mol = win_ids[real]
        counts = nbhead.code_counts(code_idx, win_of_mol // v, b_n, codebook.codes)
        nb_nll = nbrec(cell_s, counts) + weights.w_commit * commit
    else:
        nb_nll = None

    if mol is not None:
        umi_for_mol = out_s.umi[real]
        win_ids_m = torch.arange(w_n, device=out_s.umi.device).unsqueeze(1).expand(real.shape)
        local_cell = win_ids_m[real] // v
        rank = torch.distributed.get_rank() if _dist() else 0
        mol_sum, mol_acc, n_mol = mol(umi_for_mol, cell_g, local_cell + rank * b_n)
        mol_loss = _exact_mean(mol_sum, max(n_mol, 1))
    else:
        mol_loss, mol_acc = None, None

    loss = (
        weights.w_mlm * mlm
        + weights.w_ema_umi * ema_umi
        + weights.w_ema_cell * ema_cell
        + weights.w_reg * reg_mix
        + weights.w_reg_cell * reg_cell
        + (weights.w_nb * nb_nll if nb_nll is not None else 0.0)
        + (weights.w_mol * mol_loss if mol_loss is not None else 0.0)
    )

    with torch.no_grad():
        window_all = win_all.detach()
        flat_all = flat.detach()
        cell_all = _gather_rows(cell_s.detach())
    logs = {
        "eff_rank_window": guards.effective_rank(window_all),
        "eff_rank_proj": guards.effective_rank(flat_all),
        "eff_rank_cell": guards.effective_rank(cell_all),
        "cell_norm": float(cell_all.norm(dim=-1).mean()),
        "loss_mlm": float(mlm.detach()),
        "loss_mlm_bits": float(mlm.detach()) / 0.6931471805599453,
        "loss_ema_umi": float(ema_umi.detach()),
        "loss_ema_cell": float(ema_cell.detach()),
        "loss_nb": float(nb_nll.detach()) if nb_nll is not None else 0.0,
                "loss_commit": float(commit.detach()) if nb_nll is not None else 0.0,
                "loss_mol": float(mol_loss.detach()) if mol_loss is not None else 0.0,
                "mol_acc": float(mol_acc) if mol_acc is not None else 0.0,
        "loss_reg": float(reg_loss.detach()),
        "loss_inv": float(inv.detach()),
        "mlm_positions": int(keep.sum()),
    }
    terms = {"mlm": mlm, "ema_umi": ema_umi, "ema_cell": ema_cell, "reg": reg_loss, "inv": inv,
             "reg_cell": reg_cell}

    if nb_nll is not None:
        terms["nb"] = nb_nll
    if mol_loss is not None:
        terms["mol"] = mol_loss
    return StepOut(loss=loss, terms=terms, logs=logs)
