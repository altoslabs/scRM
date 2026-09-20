"""The byte-level and patch-level transformer trunk.

The hierarchical byte-to-patch design is credited to BLT (Byte Latent Transformer, Pagnoni
et al., arXiv:2412.09871); this implements that interface rather than importing it, and no
upstream code is vendored here.

ATTRIBUTION OWED, AND IT IS NOT "written from the papers alone". This file shares distinctive
expression with facebookresearch/lingua's `lingua/transformer.py` (BSD-3-Clause, Copyright 2024
Meta): the 2x2 real rotation packing in `_rope_tables` below, which the llama reference instead
represents with a complex `torch.polar`, so that form is lingua's rather than the RoPE paper's;
the SwiGLU hidden-width rule in `_hidden_dim`; and the `precompute_freqs_cis` / `apply_rotary_emb`
/ `reset_parameters(init_std, factor)` naming and signatures. The notice BSD-3-Clause requires is
in THIRD_PARTY_NOTICES.md.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

RMSNorm = nn.RMSNorm

_FLEX = None


def _flex():
    global _FLEX
    if _FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention

        _FLEX = torch.compile(flex_attention, dynamic=False)
    return _FLEX


@dataclass
class BaseTransformerArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_seqlen: int = 1024
    attn_impl: str = "sdpa"


def precompute_freqs_cis(head_dim, end, theta=10000.0, device=None):
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim
    inv_freq = 1.0 / (theta**exponents)
    pos = torch.arange(end, dtype=torch.float32, device=device)
    angles = torch.outer(pos, inv_freq)
    cos, sin = torch.cos(angles), torch.sin(angles)
    return torch.stack((cos, -sin, sin, cos), dim=-1).view(end, head_dim // 2, 2, 2)


def _broadcast_over_heads(freqs_cis):
    if freqs_cis.ndim == 4:
        return freqs_cis[None, :, None]
    if freqs_cis.ndim == 5:
        return freqs_cis[:, :, None]
    raise ValueError(f"freqs_cis must be 4-D or 5-D, got shape {tuple(freqs_cis.shape)}")


def _rotate(x, rotations):
    pairs = x.float().reshape(*x.shape[:-1], -1, 1, 2)
    return (pairs * rotations).sum(-1).reshape(x.shape).to(x.dtype)


def apply_rotary_emb(xq, xk, seq_dim, freqs_cis):
    if seq_dim != 1:
        raise NotImplementedError(f"seq_dim must be 1, got {seq_dim}")
    rotations = _broadcast_over_heads(freqs_cis)
    return _rotate(xq, rotations), _rotate(xk, rotations)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seqlen, theta=10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen
        self.theta = theta
        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(head_dim, max_seqlen, theta), persistent=False
        )

    def reset_parameters(self):
        self.freqs_cis = precompute_freqs_cis(
            self.head_dim, self.max_seqlen, self.theta, device=self.freqs_cis.device
        )

    def forward(self, seqlen=None, tok_idx=None):
        if tok_idx is not None:
            return self.freqs_cis[tok_idx]
        if seqlen is not None:
            return self.freqs_cis[:seqlen]
        raise ValueError("one of seqlen or tok_idx is required")


def _init_linear(layer):
    std = layer.in_features**-0.5
    nn.init.trunc_normal_(layer.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)


class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"dim {dim} is not divisible by n_heads {n_heads}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freq_cis, *, mask=None, attn_impl="sdpa"):
        bsz, seqlen, _ = x.shape
        shape = (bsz, seqlen, self.n_heads, self.head_dim)
        xq = self.wq(x).view(shape)
        xk = self.wk(x).view(shape)
        xv = self.wv(x).view(shape)

        xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis)
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)

        if attn_impl == "sdpa":
            out = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask)
        elif attn_impl == "flex_attention":
            out = _flex()(xq, xk, xv, block_mask=mask)
        else:
            raise NotImplementedError(f"unknown attn_impl {attn_impl!r}")

        return self.wo(out.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim))

    def reset_parameters(self):
        for layer in (self.wq, self.wk, self.wv, self.wo):
            _init_linear(layer)


def _hidden_dim(dim, multiple_of, ffn_dim_multiplier=None):
    hidden = int(2 * (4 * dim) / 3)
    if ffn_dim_multiplier is not None:
        hidden = int(ffn_dim_multiplier * hidden)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def reset_parameters(self):
        for layer in (self.w1, self.w2, self.w3):
            _init_linear(layer)


class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention = Attention(args.dim, args.n_heads)
        self.feed_forward = FeedForward(
            args.dim, _hidden_dim(args.dim, args.multiple_of, args.ffn_dim_multiplier)
        )
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, h, freq_cis, tok_idx=None, mask=None, attn_impl="sdpa"):
        h = h + self.attention(self.attention_norm(h), freq_cis, mask=mask, attn_impl=attn_impl)
        return h + self.feed_forward(self.ffn_norm(h))

    def init_weights(self):
        self.attention.reset_parameters()
        self.attention_norm.reset_parameters()
        self.feed_forward.reset_parameters()
        self.ffn_norm.reset_parameters()


class BaseTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args.dim
        self.n_layers = args.n_layers
        self.max_seqlen = args.max_seqlen
        self.attn_impl = args.attn_impl
        self.rope_embeddings = RotaryEmbedding(
            head_dim=args.dim // args.n_heads, max_seqlen=args.max_seqlen, theta=args.rope_theta
        )
        self.layers = nn.ModuleList(TransformerBlock(args) for _ in range(args.n_layers))

    def forward(self, h, *, tok_idx=None, mask=None, attn_impl=None):
        impl = self.attn_impl if attn_impl is None else attn_impl
        freq_cis = self.rope_embeddings(seqlen=h.shape[1], tok_idx=tok_idx)
        for layer in self.layers:
            h = layer(h, freq_cis, tok_idx, mask, impl)
        return h

    def init_weights(self):
        self.rope_embeddings.reset_parameters()
        for layer in self.layers:
            layer.init_weights()
