"""The single EMA teacher: `AveragedModel` with a scheduled momentum."""

from __future__ import annotations

import torch
from torch.optim.swa_utils import AveragedModel

from .model import ScrmModel


class EmaTeacher:
    def __init__(self, student: ScrmModel, *, total_steps: int, m_start: float = 0.996, m_end: float = 1.0) -> None:
        self.total_steps = total_steps
        self.m_start = m_start
        self.m_end = m_end
        self.momentum = m_start

        def ema_fn(avg_params: list[torch.Tensor], params: list[torch.Tensor], _num) -> None:
            torch._foreach_lerp_(avg_params, params, 1.0 - self.momentum)

        self.averaged = AveragedModel(student, multi_avg_fn=ema_fn, use_buffers=False)
        for p in self.averaged.parameters():
            p.requires_grad_(False)

    @property
    def model(self) -> ScrmModel:
        """The EMA copy, for forwards. A plain ScrmModel so `pool_views` etc. resolve."""
        return self.averaged.module

    def update(self, student: ScrmModel, step: int) -> float:
        """Advance the schedule (linear, per step — JEPA-DNA's) and fold in the student."""
        t = min(step / max(self.total_steps, 1), 1.0)
        self.momentum = self.m_start + (self.m_end - self.m_start) * t
        self.averaged.update_parameters(student)
        return self.momentum

    def state_dict(self) -> dict:
        return {"averaged": self.averaged.state_dict(), "momentum": self.momentum}

    def load_state_dict(self, state: dict) -> None:
        self.averaged.load_state_dict(state["averaged"])
        self.momentum = state["momentum"]


def make_teacher(student: ScrmModel, *, total_steps: int,
                 m_start: float = 0.996, m_end: float = 1.0) -> EmaTeacher:
    """Build the EMA teacher, forwarding the momentum schedule's endpoints."""
    return EmaTeacher(student, total_steps=total_steps, m_start=m_start, m_end=m_end)
