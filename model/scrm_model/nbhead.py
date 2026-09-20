"""A negative-binomial reconstruction head over a LEARNED molecule vocabulary."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MoleculeCodebook(nn.Module):
    """Vector-quantise molecule embeddings into K learned codes."""

    def __init__(self, dim: int, codes: int = 512, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.codes, self.decay, self.eps = codes, decay, eps
        self.register_buffer("frozen", torch.zeros((), dtype=torch.bool))
        self.register_buffer("embed", torch.randn(codes, dim))
        self.register_buffer("cluster_size", torch.zeros(codes))
        self.register_buffer("embed_avg", torch.randn(codes, dim).clone())

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, onehot: torch.Tensor) -> None:
        n = onehot.sum(0)
        dw = onehot.t() @ flat
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(n)
            torch.distributed.all_reduce(dw)
        self.cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(dw, alpha=1 - self.decay)
        total = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.eps) / (total + self.codes * self.eps) * total
        self.embed.copy_(self.embed_avg / smoothed.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`x` (n, dim) -> (quantised, code_index, commitment SUM)."""
        flat = x.detach().float()
        d = (flat.pow(2).sum(1, keepdim=True)
             - 2 * flat @ self.embed.t()
             + self.embed.pow(2).sum(1))
        idx = d.argmin(1)
        quant = F.embedding(idx, self.embed)
        if self.training and not bool(self.frozen):
            self._ema_update(flat, F.one_hot(idx, self.codes).to(flat.dtype))
        commit_sum = F.mse_loss(x.float(), quant.detach(), reduction="sum")
        return x + (quant - x).detach(), idx, commit_sum


class NBReconstruction(nn.Module):
    """Predict a cell's count vector over codes from its cell embedding, under NB (or ZINB)."""

    def __init__(self, dim: int, codes: int, zero_inflated: bool = False):
        super().__init__()
        self.codes, self.zero_inflated = codes, zero_inflated
        self.rate = nn.Linear(dim, codes)
        self.log_theta = nn.Parameter(torch.zeros(codes))
        self.gate = nn.Linear(dim, codes) if zero_inflated else None

    def forward(self, cell: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """`cell` (b, dim), `counts` (b, codes) integer -> mean NB negative log-likelihood."""
        expo = counts.sum(1, keepdim=True).clamp(min=1.0)
        mu = torch.softmax(self.rate(cell.float()), dim=1) * expo
        theta = self.log_theta.exp().clamp(min=1e-4, max=1e4)
        mu = mu.clamp(min=1e-8)
        ll = (torch.lgamma(counts + theta) - torch.lgamma(theta) - torch.lgamma(counts + 1.0)
              + theta * (theta.log() - (theta + mu).log())
              + counts * (mu.log() - (theta + mu).log()))
        if self.zero_inflated:
            logit = self.gate(cell.float())
            log_nb_zero = theta * (theta.log() - (theta + mu).log())
            zero = torch.logaddexp(
                F.logsigmoid(logit), F.logsigmoid(-logit) + log_nb_zero)
            ll = torch.where(counts < 0.5, zero, F.logsigmoid(-logit) + ll)
        return -ll.mean()


def code_counts(idx: torch.Tensor, group: torch.Tensor, n_groups: int, codes: int) -> torch.Tensor:
    """Molecule code indices -> per-group count vector, (n_groups, codes)."""
    flat = group.long() * codes + idx.long()
    out = torch.zeros(n_groups * codes, device=idx.device, dtype=torch.float32)
    out.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
    return out.view(n_groups, codes)
