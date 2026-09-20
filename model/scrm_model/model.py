"""The backbone: bytes ⊂ contig ⊂ UMI ⊂ window ⊂ cell, assembled from `trunk.py`
and the segment pools.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import N_BASES, VOCAB, ArchConfig
from .pooling import SegmentAttentionPool, first_position_of_segment


def drop_molecules(slot_real: torch.Tensor, rho: float) -> torch.Tensor:
    """Keep a uniformly random (1 - rho) of each window's REAL molecule slots."""
    if rho <= 0.0:
        return slot_real
    w_n, p_n = slot_real.shape
    noise = torch.rand(w_n, p_n, device=slot_real.device, dtype=torch.float32)
    noise = noise.masked_fill(~slot_real, 2.0)
    order = noise.argsort(dim=1)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(p_n, device=slot_real.device).expand(w_n, p_n))
    n_real = slot_real.sum(1)
    n_keep = torch.ceil(n_real.float() * (1.0 - rho)).clamp(min=1).long()
    return rank < n_keep[:, None]
from .trunk import BaseTransformer, BaseTransformerArgs


@dataclass
class BackboneOut:
    h_bytes: torch.Tensor
    umi: torch.Tensor
    window: torch.Tensor


_BLOCK_MASK_BUILDER = None


def _compiled_block_mask():
    """`torch.compile(create_block_mask)`, built once. Compiled, the constructor evaluates the
    mask predicate blockwise; eager, it materialises the full dense grid first (64 GiB at the
    training shape).
    """
    global _BLOCK_MASK_BUILDER
    if _BLOCK_MASK_BUILDER is None:
        from torch.nn.attention.flex_attention import create_block_mask

        _BLOCK_MASK_BUILDER = torch.compile(create_block_mask)
    return _BLOCK_MASK_BUILDER


def _trunk_args(
    cfg: ArchConfig, *, dim: int, n_layers: int, n_heads: int, multiple_of: int, max_seqlen: int
) -> BaseTransformerArgs:
    return BaseTransformerArgs(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        multiple_of=multiple_of,
        norm_eps=cfg.norm_eps,
        rope_theta=cfg.rope_theta,
        max_seqlen=max_seqlen,
        attn_impl=cfg.attn_impl,
    )


class ScrmModel(nn.Module):
    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(VOCAB, cfg.dim_local)
        self.local = BaseTransformer(
            _trunk_args(
                cfg,
                dim=cfg.dim_local,
                n_layers=cfg.n_layers_local,
                n_heads=cfg.n_heads_local,
                multiple_of=cfg.multiple_of_local,
                max_seqlen=cfg.rope_max_pos,
            )
        )
        early_attn = cfg.pool_variant != "mean_all"
        late_attn = cfg.pool_variant == "attention"
        self.contig_pool = SegmentAttentionPool(cfg.dim_local, cfg.contig_out, attention=early_attn)
        self.umi_pool = SegmentAttentionPool(cfg.contig_out, cfg.dim_global, attention=early_attn)
        self.global_trunk = BaseTransformer(
            _trunk_args(
                cfg,
                dim=cfg.dim_global,
                n_layers=cfg.n_layers_global,
                n_heads=cfg.n_heads_global,
                multiple_of=cfg.multiple_of_global,
                max_seqlen=8,
            )
        )
        self.window_pool = SegmentAttentionPool(cfg.dim_global, cfg.dim_global, attention=late_attn)
        self.view_pool = SegmentAttentionPool(cfg.dim_global, cfg.dim_global, attention=late_attn)
        self.mol_drop = 0.0
        self.local.init_weights()
        self.global_trunk.init_weights()
        nn.init.trunc_normal_(self.embedding.weight, std=cfg.dim_local**-0.5)

    def _pair_mask(self, seg: torch.Tensor, member: torch.Tensor):
        """True where attention is allowed: same segment, both members — plus the diagonal, so a
        fully-masked (non-member) row returns something finite instead of an empty-softmax NaN.
        Its output is zeroed afterwards; the diagonal is NaN-hygiene, not information flow.
        """
        w_n, l_n = seg.shape
        if self.cfg.attn_impl == "flex_attention":
            def mask_mod(b, h, q, kv):
                same = (seg[b, q] == seg[b, kv]) & member[b, q] & member[b, kv]
                return same | (q == kv)

            return _compiled_block_mask()(mask_mod, w_n, None, l_n, l_n, device=seg.device)
        same = (seg.unsqueeze(2) == seg.unsqueeze(1)) & (member.unsqueeze(2) & member.unsqueeze(1))
        eye = torch.eye(l_n, dtype=torch.bool, device=seg.device)
        return (same | eye).unsqueeze(1)

    def _run_local(self, h: torch.Tensor, tok_idx: torch.Tensor, mask) -> torch.Tensor:
        """The local trunk, optionally under per-layer activation checkpointing."""
        if not self.cfg.checkpoint_local or not torch.is_grad_enabled():
            return self.local(h, tok_idx=tok_idx, mask=mask, attn_impl=self.cfg.attn_impl)
        freq_cis = self.local.rope_embeddings(seqlen=self.local.max_seqlen, tok_idx=tok_idx)
        for layer in self.local.layers:
            h = torch.utils.checkpoint.checkpoint(
                layer, h, freq_cis, None, mask, self.cfg.attn_impl, use_reentrant=False
            )
        return h

    def precompute(self, batch: dict[str, torch.Tensor]) -> dict:
        """Everything the forward derives from the BATCH alone — attention masks, per-contig RoPE
        positions, membership masks. Pure function of the batch, independent of weights, so the
        student, the teacher, and the checkpoint recompute SHARE one copy.
        """
        valid = batch["valid"]
        cseg = batch["contig_segment_id"]
        n_patches = batch["n_patches"]
        w_n, l_n = valid.shape
        device = valid.device

        n_contigs = torch.where(valid, cseg, cseg.new_full((), -1)).amax(dim=1) + 1
        c_n = max(int(n_contigs.max()), 1)
        b = self.cfg.p_bucket
        p_n = ((max(int(n_patches.max()), 1) + b - 1) // b) * b

        first_c = first_position_of_segment(cseg, valid, c_n)
        seg_start = first_c.gather(1, torch.where(valid, cseg, torch.zeros_like(cseg)))
        pos = torch.arange(l_n, device=device).expand(w_n, l_n)
        tok_idx = torch.where(valid, pos - seg_start, torch.zeros_like(pos))
        tok_idx = tok_idx.clamp_(0, self.cfg.rope_max_pos - 1)

        slot_real = torch.arange(p_n, device=device).expand(w_n, p_n) < n_patches[:, None]
        return {
            "c_n": c_n,
            "p_n": p_n,
            "n_contigs": n_contigs,
            "first_c": first_c,
            "tok_idx": tok_idx,
            "slot_real": slot_real,
            "local_mask": self._pair_mask(cseg, valid),
            "global_mask": self._pair_mask(torch.zeros_like(slot_real, dtype=torch.long), slot_real),
            "zero_pos": torch.zeros(p_n, dtype=torch.long, device=device),
        }

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        tokens: torch.Tensor | None = None,
        pre: dict | None = None,
    ) -> BackboneOut:
        tokens = batch["tokens"] if tokens is None else tokens
        valid = batch["valid"]
        cseg = batch["contig_segment_id"]
        useg = batch["umi_segment_id"]
        w_n, l_n = tokens.shape
        device = tokens.device
        pre = self.precompute(batch) if pre is None else pre
        c_n, p_n, slot_real = pre["c_n"], pre["p_n"], pre["slot_real"]

        h = self.embedding(tokens)
        h = self._run_local(h, pre["tok_idx"], pre["local_mask"])
        h = h * valid.unsqueeze(-1)

        contig = self.contig_pool(h, cseg, valid, c_n)
        contig_member = torch.arange(c_n, device=device).expand(w_n, c_n) < pre["n_contigs"][:, None]
        umi_of_contig = useg.gather(1, pre["first_c"].clamp(max=l_n - 1))
        umi = self.umi_pool(contig, umi_of_contig, contig_member, p_n)

        u = self.global_trunk(umi, tok_idx=pre["zero_pos"], mask=pre["global_mask"], attn_impl=self.cfg.attn_impl)
        u = u * slot_real.unsqueeze(-1)

        keep = drop_molecules(slot_real, self.mol_drop) if self.training else slot_real
        window = self.window_pool(u, torch.zeros_like(keep, dtype=torch.long), keep, 1).squeeze(1)
        return BackboneOut(h_bytes=h, umi=u, window=window)

    def compile_blocks(self) -> None:
        """Compile every TransformerBlock in place (state_dict keys unchanged — `nn.Module.compile`,
        not wrapping). Blocks are the dense, static-shaped 95%+ of the FLOPs; the pools' scatter
        ops and the mask build stay eager. With `p_bucket`, each trunk sees a handful of shapes,
        compiled once.
        Composes with the local trunk's activation checkpointing (`use_reentrant=False`).
        """
        for layer in list(self.local.layers) + list(self.global_trunk.layers):
            layer.compile(dynamic=False)

    def mlm_logits(self, h_bytes: torch.Tensor) -> torch.Tensor:
        """4-way logits, weight-tied to embedding rows 0..3 (A, C, G, T) — zero parameters (13.8)."""
        return h_bytes @ self.embedding.weight[:N_BASES].t()

    def pool_views(self, window: torch.Tensor, v: int) -> torch.Tensor:
        """(B*v, D) cell-major windows → (B, D) cells. The view pool's only caller (13.7)."""
        b_n = window.shape[0] // v
        x = window.view(b_n, v, -1)
        seg = torch.zeros(b_n, v, dtype=torch.long, device=window.device)
        member = torch.ones(b_n, v, dtype=torch.bool, device=window.device)
        return self.view_pool(x, seg, member, 1).squeeze(1)
