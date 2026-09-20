"""The optimiser: `AdamW(fused=True)`, betas (0.9, 0.95), with the schedules and the
weight-decay exclusions — BatchNorm params, RMSNorm gains, all biases, and the PMA seed queries.
"""

from __future__ import annotations

import math

import torch
from torch import nn

WARMUP_STEPS = 1000
LR_START, LR_MAX, LR_END = 1e-5, 3e-4, 1e-6
WD_START, WD_END = 0.04, 0.4
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0

_NORM_TYPES = (nn.RMSNorm, nn.BatchNorm1d, nn.LayerNorm)


def split_decay_groups(*modules: nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], list[str]]:
    """(decayed, undecayed, undecayed_names). Their union is every parameter."""
    no_decay_ids: set[int] = set()
    names: dict[int, str] = {}
    for m in modules:
        for mod in m.modules():
            if isinstance(mod, _NORM_TYPES):
                for p in mod.parameters(recurse=False):
                    no_decay_ids.add(id(p))
        for name, p in m.named_parameters():
            names[id(p)] = name
            if name.endswith(".bias") or "seed_q" in name:
                no_decay_ids.add(id(p))
    decayed, undecayed, undecayed_names = [], [], []
    seen: set[int] = set()
    for m in modules:
        for p in m.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if id(p) in no_decay_ids:
                undecayed.append(p)
                undecayed_names.append(names[id(p)])
            else:
                decayed.append(p)
    return decayed, undecayed, undecayed_names


DECAY_SCHEDULED = "scrm_decay_scheduled"


def build_optimizer(*modules: nn.Module, fused: bool | None = None,
                    wd_start: float = WD_START, lr_start: float = LR_START,
                    betas: tuple[float, float] = BETAS) -> torch.optim.AdamW:
    decayed, undecayed, _ = split_decay_groups(*modules)
    if fused is None:
        fused = torch.cuda.is_available()
    return torch.optim.AdamW(
        [
            {"params": decayed, "weight_decay": wd_start, DECAY_SCHEDULED: True},
            {"params": undecayed, "weight_decay": 0.0, DECAY_SCHEDULED: False},
        ],
        lr=lr_start,
        betas=betas,
        eps=1e-8,
        fused=fused,
    )


def add_undecayed_group(optimizer: torch.optim.Optimizer, params: list[torch.nn.Parameter]) -> None:
    """Add an extra no-decay group (the NB head, the contrast temperature) carrying the marker."""
    optimizer.add_param_group({"params": params, "weight_decay": 0.0, DECAY_SCHEDULED: False})


def load_optimizer_state(optimizer: torch.optim.Optimizer, state: dict) -> None:
    """`load_state_dict`, with the decay markers preserved."""
    marks = [g.get(DECAY_SCHEDULED, g["weight_decay"] != 0.0) for g in optimizer.param_groups]
    optimizer.load_state_dict(state)
    for group, mark in zip(optimizer.param_groups, marks):
        group[DECAY_SCHEDULED] = mark


def lr_at(step: int, total_steps: int, lr_max: float = LR_MAX, *,
          warmup: int = WARMUP_STEPS, lr_start: float = LR_START,
          lr_end: float = LR_END) -> float:
    """1e-5 → `lr_max` linear over the first 1000 steps, then cosine to 1e-6 (JEPA-DNA's shape)."""
    if step < warmup:
        return lr_start + (lr_max - lr_start) * step / max(warmup, 1)
    t = min((step - warmup) / max(total_steps - warmup, 1), 1.0)
    return lr_end + (lr_max - lr_end) * 0.5 * (1.0 + math.cos(math.pi * t))


def wd_at(step: int, total_steps: int, wd_end: float = WD_END, *,
          wd_start: float = WD_START) -> float:
    """0.04 → `wd_end`, cosine shape (i-JEPA's schedule). `wd_end = WD_START` holds it flat."""
    t = min(step / max(total_steps, 1), 1.0)
    return wd_end + (wd_start - wd_end) * 0.5 * (1.0 + math.cos(math.pi * t))


def apply_schedules(
    optimizer: torch.optim.Optimizer, step: int, total_steps: int, wd_end: float = WD_END,
    lr_max: float = LR_MAX, *, warmup: int = WARMUP_STEPS, lr_start: float = LR_START,
    lr_end: float = LR_END, wd_start: float = WD_START,
) -> tuple[float, float]:
    """Write the schedules into the optimiser and return WHAT WAS APPLIED, read back out of the
    param groups rather than what was intended.
    """
    lr = lr_at(step, total_steps, lr_max, warmup=warmup, lr_start=lr_start, lr_end=lr_end)
    wd = wd_at(step, total_steps, wd_end, wd_start=wd_start)
    for group in optimizer.param_groups:
        group["lr"] = lr
        if group[DECAY_SCHEDULED]:
            group["weight_decay"] = wd
    scheduled = [g["weight_decay"] for g in optimizer.param_groups if g[DECAY_SCHEDULED]]
    return optimizer.param_groups[0]["lr"], (scheduled[0] if scheduled else 0.0)


def mask_ratio_at(step: int, total_steps: int) -> tuple[float, float]:
    """U(.15, .25) early → U(.20, .30) late, linear in steps (token-pass units)."""
    t = min(step / max(total_steps, 1), 1.0)
    return 0.15 + 0.05 * t, 0.25 + 0.05 * t
