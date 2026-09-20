"""The checkpoint -> eval bridge: contract fields round-trip, and pair-mode rows are the view
    pool's output over adjacent real windows.
    """

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import make_batch

from scrm_model.config import arch_dict
from scrm_model.evalio import ScrmWindowEncoder, model_factory
from scrm_model.model import ScrmModel

from scrm_eval import embed as eval_embed  # noqa: E402  (path set in conftest)


def checkpoint_dict(cfg, model, views=None):
    """A checkpoint dict; `views=None` gives a readout with no view count at all."""
    arch = arch_dict(cfg, "S96")
    readout = {"window_to_cell": "pair_view_pool_then_mean", "pairing": "adjacent_v1"}
    if views is not None:
        readout = {"window_to_cell": "group_view_pool_then_mean",
                   "views": views, "grouping": f"adjacent_v{views}"}
    return {
        "arch": arch,
        "readout": readout,
        "arch_sha": eval_embed.arch_digest(arch),
        "state_dict": model.state_dict(),
    }


class FakeCorpus:
    """Four cells, two windows each, in eval's Batch shape — enough to drive embed() for real."""

    def __init__(self, specs, seq_len):
        self.specs = specs
        self.seq_len = seq_len

    def n_cells(self):
        return len(self.specs) // 2

    def cell_names(self):
        return [f"CELL{i}" for i in range(self.n_cells())]

    def build_batch(self, idx, *, epoch_seed, v, allow_short):
        assert v == 2 and allow_short
        wanted = [w for c in idx for w in (2 * c, 2 * c + 1)]
        b = make_batch([self.specs[w] for w in wanted], self.seq_len, seed=9)
        arrays = {k: t.numpy() for k, t in b.items()}

        class Batch:
            tokens = arrays["tokens"].astype(np.uint8)
            valid = arrays["valid"]
            contig_segment_id = arrays["contig_segment_id"].astype(np.uint16)
            umi_segment_id = arrays["umi_segment_id"].astype(np.uint16)
            n_patches = arrays["n_patches"].astype(np.uint32)
            patch_start_ids = np.zeros((len(wanted), 8), dtype=np.uint32)
            cell_index = np.repeat(np.arange(len(idx), dtype=np.uint32), 2)
            n_windows = len(wanted)

        return Batch


SPECS = [[[30], [22]], [[40]], [[22], [28]], [[35], [19]], [[55]], [[26], [30]], [[45]], [[24, 20]]]


def test_checkpoint_roundtrip_and_pair_rows(tiny_cfg, tmp_path, monkeypatch):
    torch.manual_seed(0)
    model = ScrmModel(tiny_cfg)
    path = tmp_path / "ckpt.pt"
    torch.save(checkpoint_dict(tiny_cfg, model), path)

    ckpt = eval_embed.Checkpoint.load(path, model_factory=model_factory)
    assert isinstance(ckpt.model, ScrmWindowEncoder)

    corpus = FakeCorpus(SPECS, seq_len=64)
    monkeypatch.setattr(
        eval_embed.controls, "corpus_facts",
        lambda ds, corpus=None, root=None: ({"k": 19}, {ds: "b" * 64}, list(corpus.cell_names())),
    )
    cells, windows, matrix, stamp = eval_embed.embed(
        ckpt, corpus, dataset="mouse_brain_5k", w_eval=2, eval_seed=1, seq_len=64, batch_cells=3
    )
    assert cells == ["CELL0", "CELL1", "CELL2", "CELL3"]
    assert windows.tolist() == [0, 0, 0, 0]
    assert matrix.shape == (4, tiny_cfg.dim_global)
    assert stamp.notes["rows"] == "pairs" and stamp.notes["pairing"] == "adjacent_v1"

    b = corpus.build_batch([0], epoch_seed=1, v=2, allow_short=True)
    win = ckpt.model.embed_windows(b.tokens, b.valid, b.contig_segment_id, b.umi_segment_id, b.patch_start_ids)
    by_hand = ckpt.model.pool_pairs(win[:2])
    np.testing.assert_allclose(matrix[0], by_hand[0], rtol=1e-5, atol=1e-6)


def test_pair_mode_rejects_odd_w_eval(tiny_cfg, tmp_path):
    torch.manual_seed(0)
    model = ScrmModel(tiny_cfg)
    path = tmp_path / "ckpt.pt"
    torch.save(checkpoint_dict(tiny_cfg, model), path)
    ckpt = eval_embed.Checkpoint.load(path, model_factory=model_factory)
    with pytest.raises(eval_embed.EmbedError, match="multiple of V=2"):
        eval_embed.embed(
            ckpt, FakeCorpus(SPECS, 64), dataset="mouse_brain_5k", w_eval=3, eval_seed=1, seq_len=64
        )


def test_a_pre_views_checkpoint_reads_as_v2(tiny_cfg, tmp_path):
    """A readout with no `views` field must read as V=2."""
    torch.manual_seed(0)
    model = ScrmModel(tiny_cfg)
    path = tmp_path / "ckpt.pt"
    torch.save(checkpoint_dict(tiny_cfg, model), path)
    ckpt = eval_embed.Checkpoint.load(path, model_factory=model_factory)
    assert ckpt.model.views == 2


def test_the_group_size_comes_from_the_checkpoint_not_the_literal_two(tiny_cfg, tmp_path):
    """A V=4 checkpoint must group FOUR windows per row, and reject a w_eval that is not a multiple."""
    torch.manual_seed(0)
    model = ScrmModel(tiny_cfg)
    path = tmp_path / "ckpt4.pt"
    torch.save(checkpoint_dict(tiny_cfg, model, views=4), path)
    ckpt = eval_embed.Checkpoint.load(path, model_factory=model_factory)
    assert ckpt.model.views == 4, "the readout's view count must reach the encoder"
    with pytest.raises(eval_embed.EmbedError, match="multiple of V=4"):
        eval_embed.embed(
            ckpt, FakeCorpus(SPECS, 64), dataset="mouse_brain_5k", w_eval=6, eval_seed=1, seq_len=64
        )


def test_factory_rejects_a_tampered_arch(tiny_cfg, tmp_path):
    torch.manual_seed(0)
    model = ScrmModel(tiny_cfg)
    raw = checkpoint_dict(tiny_cfg, model)
    raw["arch"]["dim_global"] = raw["arch"]["dim_global"] * 2
    path = tmp_path / "ckpt.pt"
    torch.save(raw, path)
    with pytest.raises(eval_embed.EmbedError, match="arch_sha"):
        eval_embed.Checkpoint.load(path, model_factory=model_factory)


def test_the_backfilled_readout_uses_train_pys_keys(tmp_path, tiny_cfg):
    """`run_eval.eval_ready`'s backfilled readout must be the shape `train.py` writes."""
    from scrm_model.run_eval import eval_ready

    path = tmp_path / "pre_contract.pt"
    torch.save({"student": ScrmModel(tiny_cfg).state_dict(), "step": 1}, path)
    out = eval_ready(path, "S96", tmp_path)
    readout = torch.load(out, map_location="cpu", weights_only=False)["readout"]

    assert set(readout) == {"levels", "pool", "window_to_cell", "views", "grouping"}, readout
    assert readout["views"] == 2, "a readout with no view count is V=2, named rather than guessed"
    assert readout["grouping"] == "adjacent_v2" and "pairing" not in readout
    assert readout["window_to_cell"] == "group_view_pool_then_mean"


def test_run_eval_refuses_a_training_dataset():
    """`run_eval` must refuse a training dataset rather than stamp a valid artifact for one."""
    from scrm_model.run_eval import main

    with pytest.raises(SystemExit) as e:
        main(["--checkpoint", "x.pt", "--arm", "S96", "--tag", "t", "--out-dir", "/tmp/x",
              "--datasets", "neurons_5k", "--w-eval", "972"])
    assert "neurons_5k" in str(e.value)


def test_run_eval_requires_the_coverage_and_the_datasets_explicitly():
    """`run_eval` must require the coverage and the datasets explicitly: full coverage is per
    dataset, so any fixed default is a cap.
    """
    from scrm_model.run_eval import main

    for missing in (["--datasets", "sc_mixology_5cl"], ["--w-eval", "972"]):
        with pytest.raises(SystemExit):
            main(["--checkpoint", "x.pt", "--arm", "S96", "--tag", "t", "--out-dir", "/tmp/x",
                  *missing])
