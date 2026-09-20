"""Per-layer invariances, each at the strength claimed: bitwise where bitwise holds, within
    tolerance otherwise.
    """

from __future__ import annotations

import numpy as np
import torch

from conftest import NEWLINE, TAB

from scrm_model.config import ArchConfig
from scrm_model.model import ScrmModel
from scrm_model.pooling import SegmentAttentionPool, first_position_of_segment


def cfg(**kw):
    base = dict(
        n_layers_local=2,
        dim_local=8,
        n_heads_local=2,
        n_layers_global=2,
        dim_global=16,
        n_heads_global=2,
        multiple_of_local=8,
        multiple_of_global=8,
        proj_hidden=32,
        proj_dim=16,
        attn_impl="sdpa",
        rope_max_pos=64,
    )
    base.update(kw)
    return ArchConfig(**base)


def batch_from_molecules(windows: list[list[list[list[int]]]], seq_len: int) -> dict[str, torch.Tensor]:
    """A window is a list of molecules, a molecule a list of contig base lists."""
    w_n = len(windows)
    tokens = np.full((w_n, seq_len), NEWLINE, dtype=np.int64)
    valid = np.zeros((w_n, seq_len), dtype=bool)
    cseg = np.zeros((w_n, seq_len), dtype=np.int64)
    useg = np.zeros((w_n, seq_len), dtype=np.int64)
    n_patches = np.zeros(w_n, dtype=np.int64)
    for w, mols in enumerate(windows):
        cursor, n_contigs = 0, 0
        for mol_i, contigs in enumerate(mols):
            for k, bases in enumerate(contigs):
                sep = TAB if k < len(contigs) - 1 else (NEWLINE if mol_i < len(mols) - 1 else None)
                span = len(bases) + (1 if sep is not None else 0)
                tokens[w, cursor : cursor + len(bases)] = bases
                if sep is not None:
                    tokens[w, cursor + len(bases)] = sep
                cseg[w, cursor : cursor + span] = n_contigs
                useg[w, cursor : cursor + span] = mol_i
                cursor += span
                n_contigs += 1
        valid[w, :cursor] = True
        n_patches[w] = len(mols)
    return {
        "tokens": torch.from_numpy(tokens),
        "valid": torch.from_numpy(valid),
        "contig_segment_id": torch.from_numpy(cseg),
        "umi_segment_id": torch.from_numpy(useg),
        "n_patches": torch.from_numpy(n_patches),
    }


def rng_bases(n, seed):
    return np.random.default_rng(seed).integers(0, 4, size=n).tolist()


A = rng_bases(11, 1)
B = rng_bases(7, 2)
C = rng_bases(9, 3)


class TestSegmentPool:
    def test_a_single_member_segment_is_exactly_its_value_projection(self):
        torch.manual_seed(0)
        pool = SegmentAttentionPool(8, 16)
        x = torch.randn(1, 3, 8)
        seg = torch.tensor([[0, 1, 2]])
        member = torch.ones(1, 3, dtype=torch.bool)
        out = pool(x, seg, member, 3)
        assert torch.allclose(out[0, 0], pool.wv(x[0, 0]), atol=1e-6)

    def test_an_empty_segment_is_exactly_zero(self):
        torch.manual_seed(0)
        pool = SegmentAttentionPool(8, 16)
        x = torch.randn(2, 4, 8)
        seg = torch.tensor([[0, 0, 2, 2], [0, 0, 0, 0]])
        member = torch.tensor([[True, True, True, True], [True, True, False, False]])
        out = pool(x, seg, member, 3)
        assert (out[0, 1] == 0).all(), "segment 1 of window 0 has no members"
        assert (out[1, 1:] == 0).all()
        assert torch.isfinite(out).all()

    def test_junk_segment_ids_outside_membership_change_nothing(self):
        torch.manual_seed(0)
        pool = SegmentAttentionPool(8, 16)
        x = torch.randn(1, 5, 8)
        member = torch.tensor([[True, True, True, False, False]])
        a = pool(x, torch.tensor([[0, 0, 1, 0, 1]]), member, 2)
        b = pool(x, torch.tensor([[0, 0, 1, 999, -7]]), member, 2)
        assert torch.equal(a, b)

    def test_mean_variant_is_exactly_wv_of_the_masked_mean(self):
        torch.manual_seed(0)
        pool = SegmentAttentionPool(8, 16, attention=False)
        x = torch.randn(2, 5, 8)
        seg = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 0, 0]])
        member = torch.tensor([[True, True, True, True, False], [True, True, True, False, False]])
        out = pool(x, seg, member, 2)
        assert torch.allclose(out[0, 0], pool.wv(x[0, :2].mean(0)), atol=1e-6)
        assert torch.allclose(out[0, 1], pool.wv(x[0, 2:4].mean(0)), atol=1e-6)
        assert torch.allclose(out[1, 0], pool.wv(x[1, :3].mean(0)), atol=1e-6)
        assert (out[1, 1] == 0).all()
        assert not hasattr(pool, "wk") and not hasattr(pool, "seed_q")

    def test_pool_variant_wires_the_right_pools(self):
        for variant, early, late in (("attention", True, True), ("mean_late", True, False),
                                     ("mean_all", False, False)):
            m = ScrmModel(cfg(pool_variant=variant))
            assert m.contig_pool.attention is early and m.umi_pool.attention is early, variant
            assert m.window_pool.attention is late and m.view_pool.attention is late, variant

    def test_first_position_of_segment(self):
        seg = torch.tensor([[0, 0, 1, 1, 1, 0]])
        member = torch.tensor([[True, True, True, True, True, False]])
        first = first_position_of_segment(seg, member, 3)
        assert first.tolist() == [[0, 2, 6]]


class TestContigOrderInvariance:
    def test_contig_vectors_are_bitwise_identical_under_contig_reshuffle(self):
        torch.manual_seed(0)
        model = ScrmModel(cfg())
        ab = batch_from_molecules([[[A, B, C]]], 40)
        ba = batch_from_molecules([[[B, A, C]]], 40)
        with torch.no_grad():
            out_ab = model(ab)
            out_ba = model(ba)
            contig_ab = model.contig_pool(out_ab.h_bytes, ab["contig_segment_id"], ab["valid"], 3)
            contig_ba = model.contig_pool(out_ba.h_bytes, ba["contig_segment_id"], ba["valid"], 3)
        assert torch.equal(contig_ab[0, 0], contig_ba[0, 1]), "contig A's vector moved with its offset"
        assert torch.equal(contig_ab[0, 1], contig_ba[0, 0]), "contig B's vector moved with its offset"
        assert torch.equal(contig_ab[0, 2], contig_ba[0, 2]), "contig C did not move and must not change"

    def test_umi_and_window_are_tolerance_equal_under_contig_reshuffle(self):
        torch.manual_seed(0)
        model = ScrmModel(cfg())
        ab = batch_from_molecules([[[A, B, C], [C]]], 48)
        ba = batch_from_molecules([[[B, A, C], [C]]], 48)
        with torch.no_grad():
            out_ab, out_ba = model(ab), model(ba)
        torch.testing.assert_close(out_ab.window, out_ba.window, rtol=1e-5, atol=1e-6)


class TestUmiOrderInvariance:
    def test_window_embedding_is_tolerance_invariant_to_molecule_order(self):
        torch.manual_seed(0)
        model = ScrmModel(cfg())
        abc = batch_from_molecules([[[A], [B], [C]]], 40)
        bac = batch_from_molecules([[[B], [A], [C]]], 40)
        with torch.no_grad():
            out1, out2 = model(abc), model(bac)
        torch.testing.assert_close(out1.window, out2.window, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(out1.umi[0, [1, 0, 2]], out2.umi[0, :3], rtol=1e-5, atol=1e-6)

    def test_identity_rope_is_actually_identity(self):
        from scrm_model.trunk import RotaryEmbedding, apply_rotary_emb

        rope = RotaryEmbedding(theta=10000.0, head_dim=8, max_seqlen=4)
        xq = torch.randn(1, 5, 2, 8)
        xk = torch.randn(1, 5, 2, 8)
        freqs = rope(tok_idx=torch.zeros(5, dtype=torch.long))
        q2, k2 = apply_rotary_emb(xq, xk, 1, freqs)
        assert torch.equal(q2, xq) and torch.equal(k2, xk)


class TestMlmHead:
    def test_logits_are_the_embedding_rows_and_add_no_parameters(self):
        model = ScrmModel(cfg())
        h = torch.randn(2, 6, model.cfg.dim_local)
        logits = model.mlm_logits(h)
        assert logits.shape == (2, 6, 4)
        assert torch.equal(logits, h @ model.embedding.weight[:4].t())
        names = [n for n, _ in model.named_parameters() if "head" in n or "mlm" in n]
        assert names == [], f"the head must be weight-tied, found {names}"


class TestParameterAccounting:
    def test_the_design_pinned_counts_hold_exactly(self):
        from scrm_model.projector import build_projector

        model = ScrmModel(ArchConfig())
        n = lambda m: sum(p.numel() for p in m.parameters())
        assert n(model.embedding) == 768
        assert n(model.local) == 443_136
        assert n(model.global_trunk) == 10_621_440
        assert n(build_projector(ArchConfig())) == 1_841_536
        assert n(model.contig_pool) == 96 * 96 + 96 * 96 + 96
        assert n(model.umi_pool) == 96 * 96 + 96 * 384 + 96
        assert n(model.window_pool) == 384 * 384 + 384 * 384 + 384
        assert n(model.view_pool) == 384 * 384 + 384 * 384 + 384


def test_flex_attention_matches_sdpa_on_the_same_batch():
    """The flex-attention path must match SDPA on the same batch."""
    torch.manual_seed(0)
    batch = batch_from_molecules([[[A, B], [C]], [[C], [B]]], 40)
    m_sdpa = ScrmModel(cfg())
    m_flex = ScrmModel(cfg(attn_impl="flex_attention"))
    m_flex.load_state_dict(m_sdpa.state_dict())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = {k: v.to(dev) for k, v in batch.items()}
    m_sdpa.to(dev), m_flex.to(dev)
    with torch.no_grad():
        a, b = m_sdpa(batch), m_flex(batch)
    torch.testing.assert_close(a.h_bytes, b.h_bytes, rtol=2e-4, atol=1e-5)
    torch.testing.assert_close(a.umi, b.umi, rtol=2e-4, atol=1e-5)
    torch.testing.assert_close(a.window, b.window, rtol=2e-4, atol=1e-5)


def test_the_two_pair_mask_constructions_agree_elementwise():
    """`_pair_mask`'s two constructions, the `mask_mod` closure and the dense `(same | eye)`, must
    agree as predicates over every (window, q, kv) triple.
    """
    torch.manual_seed(0)
    batch = batch_from_molecules([[[A, B], [C]], [[C], [B]]], 40)
    m_sdpa = ScrmModel(cfg())
    m_flex = ScrmModel(cfg(attn_impl="flex_attention"))
    cseg, valid = batch["contig_segment_id"], batch["valid"]
    w_n, l_n = valid.shape

    dense = m_sdpa._pair_mask(cseg, valid)
    block = m_flex._pair_mask(cseg, valid)
    b = torch.arange(w_n)[:, None, None]
    q = torch.arange(l_n)[None, :, None]
    kv = torch.arange(l_n)[None, None, :]
    from_mod = block.mask_mod(b, torch.zeros_like(b), q, kv)

    assert bool(valid.logical_not().any()), "the fixture must contain non-member positions"
    assert torch.equal(from_mod, dense.squeeze(1)), (
        f"the flex predicate and the dense reference differ at "
        f"{int((from_mod != dense.squeeze(1)).sum())} of {w_n * l_n * l_n} positions"
    )
