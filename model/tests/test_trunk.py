"""`scrm_model.trunk`'s own obligations."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from scrm_model import trunk as T
from scrm_model.config import arms

FLAGSHIP = arms()["S96"]


def _args(**kw):
    base = dict(dim=96, n_layers=2, n_heads=4, multiple_of=8, norm_eps=FLAGSHIP.norm_eps,
                rope_theta=FLAGSHIP.rope_theta, max_seqlen=64, attn_impl="sdpa")
    return T.BaseTransformerArgs(**{**base, **kw})


def test_flex_attention_is_compiled_and_not_the_raw_kernel():
    """`trunk._flex()` must return a `torch.compile`-wrapped kernel, not the raw `flex_attention`,
    which materialises the full score matrix.
    """
    from torch.nn.attention.flex_attention import flex_attention as raw

    got = T._flex()
    assert got is not raw, (
        "trunk._flex() returned the raw flex_attention: eager flex_attention materialises the full "
        "score matrix and OOMs at eval shapes. It must be torch.compile-wrapped."
    )
    assert callable(got)
    assert hasattr(got, "_torchdynamo_orig_callable") or type(got).__name__ != "function", type(got)
    assert T._flex() is got, "the compiled kernel must be built once, not per call"


def test_the_norm_is_the_framework_primitive_so_weight_decay_excludes_it():
    """`optim._NORM_TYPES` is `(nn.RMSNorm, ...)` and the decay split is an `isinstance` test."""
    from scrm_model.optim import _NORM_TYPES

    assert T.RMSNorm is nn.RMSNorm
    block = T.TransformerBlock(_args())
    assert isinstance(block.attention_norm, _NORM_TYPES)
    assert isinstance(block.ffn_norm, _NORM_TYPES)


def test_apply_rotary_emb_takes_seq_dim_third():
    """The signature is `(xq, xk, seq_dim, freqs_cis)` and `test_layers.py` calls it positionally."""
    rope = T.RotaryEmbedding(head_dim=8, max_seqlen=4, theta=10000.0)
    xq, xk = torch.randn(1, 5, 2, 8), torch.randn(1, 5, 2, 8)
    freqs = rope(tok_idx=torch.zeros(5, dtype=torch.long))
    q2, k2 = T.apply_rotary_emb(xq, xk, 1, freqs)
    assert torch.equal(q2, xq) and torch.equal(k2, xk), "position 0 must be a bitwise no-op"
    with pytest.raises(TypeError):
        T.apply_rotary_emb(xq, xk, freqs)
    with pytest.raises(NotImplementedError):
        T.apply_rotary_emb(xq, xk, 2, freqs)


def test_init_weights_is_per_parameter_fan_in_with_no_depth_factor():
    """`std = fan_in ** -0.5` per matrix, truncated at 3 sigma, and NO `1/sqrt(2*(l+1))`."""
    for n_layers in (1, 4, 6):
        torch.manual_seed(1234)
        tr = T.BaseTransformer(_args(n_layers=n_layers, max_seqlen=1600))
        tr.init_weights()
        for name, p in tr.named_parameters():
            if p.dim() != 2:
                continue
            ratio = p.std().item() / (p.shape[1] ** -0.5)
            assert 0.95 < ratio < 1.02, (
                f"{n_layers}L {name}: std/fan_in^-0.5 = {ratio:.4f}. A depth factor would make "
                f"this fall with layer index (1/sqrt(2) = 0.707 at layer 0)."
            )


def test_the_ffn_declares_w1_w3_w2_and_initialises_w1_w2_w3():
    """The FFN must declare w1, w3, w2 and must initialise in w1, w2, w3 order."""
    block = T.TransformerBlock(_args())
    assert [n for n, _ in block.feed_forward.named_parameters()] == \
           ["w1.weight", "w3.weight", "w2.weight"]
    src = __import__("inspect").getsource(T.FeedForward.reset_parameters)
    assert src.index("self.w1") < src.index("self.w2") < src.index("self.w3"), (
        "reset_parameters must draw in w1, w2, w3 order; see the comment above it"
    )


def test_position_zero_is_exactly_the_identity_rotation():
    """Position 0 must be exactly the identity rotation, so the global trunk is positionless."""
    tr = T.BaseTransformer(_args(n_layers=1, max_seqlen=8))
    row0 = tr.rope_embeddings(seqlen=1)[0]
    assert torch.equal(row0, torch.eye(2).expand_as(row0).contiguous()), row0


def test_the_rotary_frequency_uses_the_reciprocal_form():
    """`1 / theta**e`, NOT `theta ** -e`. In fp32 they are different numbers."""
    head_dim, theta, end = 24, 10000.0, 64
    e = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    reciprocal = 1.0 / (theta ** e)
    negative_exp = theta ** (-e)
    assert not torch.equal(reciprocal, negative_exp), (
        "the code must state which form it uses, even where the two are bitwise equal on this "
        "build"
    )
    pos = torch.arange(end, dtype=torch.float32)
    built = T.precompute_freqs_cis(head_dim, end, theta)
    want_cos = torch.cos(torch.outer(pos, reciprocal))
    assert torch.equal(built[..., 0, 0], want_cos), "precompute_freqs_cis must use 1/(theta**e)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex_attention needs a GPU")
def test_flex_attention_survives_a_long_sequence():
    """Flex attention must survive a long, realistic sequence shape."""
    from torch.nn.attention.flex_attention import create_block_mask

    dev = "cuda"
    tr = T.BaseTransformer(_args(n_layers=1, max_seqlen=4096, attn_impl="flex_attention")).to(dev).eval()
    tr.init_weights()
    b, s = 1, 4096
    h = torch.randn(b, s, 96, device=dev)
    tok = torch.arange(s, device=dev).unsqueeze(0)
    seg = (torch.arange(s, device=dev) // 512).unsqueeze(0).expand(b, s)

    def same_segment(bi, hi, q, k):
        return seg[bi, q] == seg[bi, k]

    bm = create_block_mask(same_segment, B=b, H=None, Q_LEN=s, KV_LEN=s, device=dev)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = tr(h, tok_idx=tok, mask=bm, attn_impl="flex_attention")
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    assert out.shape == (b, s, 96)
    assert torch.isfinite(out).all()
    assert peak_gb < 1.0, f"peak {peak_gb:.2f} GiB suggests the dense fallback, not the fused kernel"
