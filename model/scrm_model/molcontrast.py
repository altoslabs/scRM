"""Per-molecule capture likelihood, without a molecule universe."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MoleculeCellContrast(nn.Module):
    """Score each molecule against every cell in the GLOBAL batch; its own cell is the target."""

    def __init__(self, dim: int, temp: float = 0.07, queue: int = 0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(temp)))))
        self.queue_size = queue
        if queue:
            self.register_buffer("neg", torch.zeros(queue, dim), persistent=True)
            self.register_buffer("neg_ptr", torch.zeros((), dtype=torch.long))
            self.register_buffer("neg_filled", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _enqueue(self, cells: torch.Tensor) -> None:
        """Ring-buffer the batch's cell vectors."""
        if not self.queue_size:
            return
        n = min(cells.shape[0], self.queue_size)
        idx = (self.neg_ptr + torch.arange(n, device=cells.device)) % self.queue_size
        self.neg[idx] = F.normalize(cells[:n].detach().float(), dim=1)
        self.neg_ptr.fill_(int((self.neg_ptr + n) % self.queue_size))
        self.neg_filled.fill_(int(min(int(self.neg_filled) + n, self.queue_size)))

    def forward(self, mol: torch.Tensor, cells: torch.Tensor, target: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """`mol` (n, d) local molecules, `cells` (B_global, d), `target` (n,) global cell index."""
        m = F.normalize(mol.float(), dim=1)
        c = F.normalize(cells.float(), dim=1)
        temp = self.log_temp.exp().clamp(min=1e-3)
        logits = (m @ c.t()) / temp
        filled = int(self.neg_filled) if self.queue_size else 0
        if filled:
            logits = torch.cat([logits, (m @ self.neg[:filled].t().to(m.dtype)) / temp], dim=1)
        loss = F.cross_entropy(logits, target, reduction="sum")
        with torch.no_grad():
            acc = (logits.argmax(1) == target).float().mean()
        if self.training and self.queue_size:
            self._enqueue(cells)
        return loss, acc, mol.shape[0]
