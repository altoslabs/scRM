"""One masked segment-attention pool, four instances."""

from __future__ import annotations

import math

import torch
from torch import nn


class SegmentAttentionPool(nn.Module):
    """Pool positions into their segments with one learned seed query."""

    def __init__(self, d_in: int, d_out: int, attention: bool = True) -> None:
        super().__init__()
        self.attention = attention
        if attention:
            self.wk = nn.Linear(d_in, d_in, bias=False)
        self.wv = nn.Linear(d_in, d_out, bias=False)
        if attention:
            self.seed_q = nn.Parameter(torch.randn(d_in) * d_in**-0.5)
        self.scale = 1.0 / math.sqrt(d_in)

    def forward(
        self,
        x: torch.Tensor,
        seg: torch.Tensor,
        member: torch.Tensor,
        n_seg: int,
    ) -> torch.Tensor:
        w_n, l_n, _ = x.shape
        if self.attention:
            score = (self.wk(x) * self.seed_q).sum(-1) * self.scale
        else:
            score = x.new_zeros(w_n, l_n)
        score = score.masked_fill(~member, float("-inf"))
        seg = torch.where(member, seg, torch.zeros_like(seg))
        flat_seg = (seg + n_seg * torch.arange(w_n, device=seg.device)[:, None]).reshape(-1)

        seg_max = x.new_full((w_n * n_seg,), float("-inf"))
        seg_max.scatter_reduce_(0, flat_seg, score.reshape(-1), reduce="amax")
        gathered = torch.where(torch.isinf(seg_max), torch.zeros_like(seg_max), seg_max)[flat_seg]
        weight = torch.exp(score.reshape(-1) - gathered)

        den = x.new_zeros(w_n * n_seg)
        den.index_add_(0, flat_seg, weight)
        num = x.new_zeros(w_n * n_seg, self.wv.out_features)
        num.index_add_(0, flat_seg, weight[:, None] * self.wv(x).reshape(w_n * l_n, -1))
        out = num / den.clamp_min(torch.finfo(x.dtype).tiny)[:, None]
        return out.reshape(w_n, n_seg, -1)


def first_position_of_segment(
    seg: torch.Tensor,
    member: torch.Tensor,
    n_seg: int,
) -> torch.Tensor:
    """Each segment's first member position — how per-contig RoPE positions and the contig→UMI
    grouping are derived on device from the ids the packer already ships.
    """
    w_n, l_n = seg.shape
    pos = torch.arange(l_n, device=seg.device).expand(w_n, l_n)
    seg = torch.where(member, seg, torch.zeros_like(seg))
    flat_seg = (seg + n_seg * torch.arange(w_n, device=seg.device)[:, None]).reshape(-1)
    first = seg.new_full((w_n * n_seg,), l_n)
    first.scatter_reduce_(0, flat_seg, torch.where(member, pos, torch.full_like(pos, l_n)).reshape(-1), reduce="amin")
    return first.reshape(w_n, n_seg)
