"""`scrm-embed`'s contracts, with a stub model and with the real corpus."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import os
import time as _t
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "target" / "py"))

from scrm_eval import artifact, embed, reduce, registry  # noqa: E402
from scrm_eval.embed import Checkpoint, EmbedError, arch_digest  # noqa: E402

REAL_CORPUS = registry.corpus_root()
BIN = Path(__file__).resolve().parents[2] / "target/release/scrm-corpus"

torch = pytest.importorskip("torch")
scrm_py = pytest.importorskip("scrm_py")

_REAL_CANDIDATES = registry.datasets("train")

needs_real = pytest.mark.skipif(
    not any((REAL_CORPUS / d / "manifest.json").exists() for d in _REAL_CANDIDATES),
    reason=f"no registered corpus under {REAL_CORPUS}",
)

ARCH = {"dim_global": 16, "n_layers": 2, "patch_len": 4, "seq_len": 256}
READOUT = {"levels": ["patch->umi", "umi->window"], "pool": "pma", "window_to_cell": "mean"}


class StubEncoder:
    """A deterministic `WindowEncoder`: mean of a fixed random token embedding over valid positions."""

    def __init__(self, dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.table = rng.normal(size=(7, dim)).astype(np.float32)
        self.dim = dim

    def embed_windows(self, tokens, valid, contig_segment_id, umi_segment_id, patch_start_ids):
        tok = np.asarray(tokens, dtype=np.int64)
        msk = np.asarray(valid, dtype=np.float32)
        emb = self.table[np.clip(tok, 0, 6)]
        num = (emb * msk[:, :, None]).sum(axis=1)
        den = np.maximum(msk.sum(axis=1, keepdims=True), 1.0)
        return (num / den).astype(np.float32)


def write_checkpoint(tmp_path, *, arch=ARCH, readout=READOUT, arch_sha=None, drop=None) -> Path:
    payload = {
        "arch": arch,
        "readout": readout,
        "arch_sha": arch_sha if arch_sha is not None else arch_digest(arch),
        "state_dict": {"table_seed": 0},
        "torch_version": torch.__version__,
    }
    if drop:
        payload.pop(drop)
    p = tmp_path / "ckpt.pt"
    torch.save(payload, p)
    return p


def factory(arch, state_dict):
    return StubEncoder(arch["dim_global"], seed=state_dict.get("table_seed", 0))


def test_arch_digest_is_canonical_in_key_order():
    """`arch_digest` must be invariant to key order."""
    assert arch_digest({"a": 1, "b": 2}) == arch_digest({"b": 2, "a": 1})


def test_arch_digest_changes_when_any_value_changes():
    base = dict(ARCH)
    for k in base:
        edited = dict(base)
        edited[k] = 999 if isinstance(base[k], int) else "x"
        assert arch_digest(edited) != arch_digest(base), k


@pytest.mark.parametrize("missing", ["arch", "readout", "arch_sha", "state_dict"])
def test_a_checkpoint_missing_a_required_field_is_refused(tmp_path, missing):
    p = write_checkpoint(tmp_path, drop=missing)
    with pytest.raises(EmbedError, match=missing):
        Checkpoint.load(p, model_factory=factory)


def test_an_empty_readout_spec_is_refused(tmp_path):
    """A checkpoint carrying an empty readout spec must be refused."""
    p = write_checkpoint(tmp_path, readout={})
    with pytest.raises(EmbedError, match="empty readout spec"):
        Checkpoint.load(p, model_factory=factory)


def test_a_tampered_arch_sha_is_refused(tmp_path):
    """A checkpoint whose `arch_sha` does not match its own arch config must be refused."""
    p = write_checkpoint(tmp_path, arch_sha="0" * 64)
    with pytest.raises(EmbedError, match="does not match its own arch config"):
        Checkpoint.load(p, model_factory=factory)


def test_an_arch_edited_after_stamping_is_refused(tmp_path):
    """A checkpoint whose arch config was edited after stamping must be refused."""
    good = arch_digest(ARCH)
    p = write_checkpoint(tmp_path, arch={**ARCH, "dim_global": 384}, arch_sha=good)
    with pytest.raises(EmbedError, match="architecture was changed after training"):
        Checkpoint.load(p, model_factory=factory)


def test_embed_takes_no_architecture_arguments():
    """`embed.embed`'s signature must expose no architecture parameter."""
    import inspect

    params = set(inspect.signature(embed.embed).parameters)
    forbidden = {"dim", "arch", "readout", "n_layers", "dim_global", "pool", "window_to_cell"}
    assert not (params & forbidden), f"embed() exposes architecture: {params & forbidden}"


def build_sim_corpus(tmp_path, n_cells=6, mols_per_cell=40) -> Path:
    """A corpus from hand-written FASTQ via the real binary, so its content is known by construction."""
    if not BIN.exists():
        pytest.skip(f"no {BIN}; cargo build --release -p scrm-cli")
    cdna = "ACGTTGCATTGACCGATCAGTTGCAATCGGATCAGTTACGGATCCAGTTAACGGTTCAGC"
    r1, r2, bcs = [], [], []
    i = 0
    def base4(value: int, width: int) -> str:
        """`value` as `width` base-4 digits in ACGT."""
        out = ""
        for _ in range(width):
            out = "ACGT"[value % 4] + out
            value //= 4
        return out

    for c in range(n_cells):
        cb = base4(c, 16)
        bcs.append(cb)
        for m in range(mols_per_cell):
            umi = base4(m, 12)
            rot = m % len(cdna)
            seq = cdna[rot:] + cdna[:rot]
            r1.append(f"@R{i} 1:N:0:A\n{cb}{umi}\n+\n{'I' * 28}\n")
            r2.append(f"@R{i} 2:N:0:A\n{seq}\n+\n{'I' * len(seq)}\n")
            i += 1
    d = Path(tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    (d / "r1.fastq").write_text("".join(r1))
    (d / "r2.fastq").write_text("".join(r2))
    (d / "bc.tsv").write_text("".join(f"{b}-1\n" for b in dict.fromkeys(bcs)))
    (d / "sheet.csv").write_text(
        "experiment,r1,r2,barcodes,geometry\n"
        f"sim,{d/'r1.fastq'},{d/'r2.fastq'},{d/'bc.tsv'},1{{b[16]u[12]}}2{{r:}}\n"
    )
    subprocess.run(
        [str(BIN), "--samplesheet", str(d / "sheet.csv"), "--out", str(d), "--threads", "2",
         "--min-umi-len", "1", "--min-cell-molecules", "1"],
        check=True, capture_output=True,
    )
    return d / "sim"


def test_the_whole_chain_on_a_simulated_corpus(tmp_path, monkeypatch):
    """checkpoint -> model -> corpus -> artifact -> read back must round-trip on a simulated corpus."""
    corpus_dir = build_sim_corpus(tmp_path / "sim")
    ck = Checkpoint.load(write_checkpoint(tmp_path, arch={**ARCH, "dim_global": 16}), model_factory=factory)
    corpus = scrm_py.Corpus(path=str(corpus_dir), seq_len=16384, max_umi_len=1600, min_v=1)

    _corpus = corpus
    monkeypatch.setattr(
        embed.controls, "corpus_facts",
        lambda ds, corpus=None, root=None: ({"k": 19}, {ds: "b" * 64}, list(_corpus.cell_names())),
    )

    out = embed.embed_to_file(
        ck, corpus, tmp_path / "emb.arrow", dataset="sim", w_eval=2, eval_seed=7, seq_len=16384
    )
    got = artifact.read(out)

    assert got.stamp.producer in artifact.MODEL_PRODUCERS
    assert got.stamp.readout == READOUT, "the readout spec must reach the artifact verbatim"
    assert got.stamp.arch_sha == ck.arch_sha
    assert got.stamp.w_eval == 2 and got.stamp.eval_seed == 7
    assert got.matrix.shape[1] == 16
    assert len(got) == got.stamp.n_windows > 0
    cells, reduced = reduce.reduce_windows(got.cell_ids, got.matrix, "mean")
    assert reduced.shape[0] == got.stamp.n_cells


def test_two_runs_at_one_seed_are_bitwise_identical(tmp_path, monkeypatch):
    """Two runs at one `eval_seed` must produce bitwise identical artifacts."""
    corpus_dir = build_sim_corpus(tmp_path / "sim")
    ck = Checkpoint.load(write_checkpoint(tmp_path), model_factory=factory)
    corpus = scrm_py.Corpus(path=str(corpus_dir), seq_len=16384, max_umi_len=1600, min_v=1)
    _corpus = corpus
    monkeypatch.setattr(
        embed.controls, "corpus_facts",
        lambda ds, corpus=None, root=None: ({"k": 19}, {ds: "b" * 64}, list(_corpus.cell_names())),
    )
    a = embed.embed(ck, corpus, dataset="sim", w_eval=2, eval_seed=11)
    b = embed.embed(ck, corpus, dataset="sim", w_eval=2, eval_seed=11)
    assert np.array_equal(a[2], b[2])
    assert a[0] == b[0] and np.array_equal(a[1], b[1])
    assert a[3].cell_order_sha256 == b[3].cell_order_sha256


def test_the_seed_is_irrelevant_when_a_cell_fits_in_its_windows(tmp_path, monkeypatch):
    """`eval_seed` must not change the embedding beyond rounding when a cell fits in its windows."""
    corpus_dir = build_sim_corpus(tmp_path / "small", n_cells=6, mols_per_cell=40)
    ck = Checkpoint.load(write_checkpoint(tmp_path), model_factory=factory)
    corpus = scrm_py.Corpus(path=str(corpus_dir), seq_len=16384, max_umi_len=1600, min_v=1)
    _corpus = corpus
    monkeypatch.setattr(
        embed.controls, "corpus_facts",
        lambda ds, corpus=None, root=None: ({"k": 19}, {ds: "b" * 64}, list(_corpus.cell_names())),
    )
    a = embed.embed(ck, corpus, dataset="sim", w_eval=1, eval_seed=1)[2]
    b = embed.embed(ck, corpus, dataset="sim", w_eval=1, eval_seed=2)[2]
    assert np.allclose(a, b, rtol=1e-5, atol=1e-6), "order must not change the embedding beyond rounding"
    assert not np.array_equal(a, b), (
        "and it is NOT bitwise identical: the arithmetic does not support that claim"
    )


def test_a_different_seed_changes_the_embedding_when_it_subsamples(tmp_path, monkeypatch):
    """`eval_seed` must change the embedding when a cell is deeper than its window budget."""
    corpus_dir = build_sim_corpus(tmp_path / "deep", n_cells=4, mols_per_cell=600)
    ck = Checkpoint.load(write_checkpoint(tmp_path), model_factory=factory)
    corpus = scrm_py.Corpus(path=str(corpus_dir), seq_len=16384, max_umi_len=1600, min_v=1)
    _corpus = corpus
    monkeypatch.setattr(
        embed.controls, "corpus_facts",
        lambda ds, corpus=None, root=None: ({"k": 19}, {ds: "b" * 64}, list(_corpus.cell_names())),
    )
    a = embed.embed(ck, corpus, dataset="sim", w_eval=1, eval_seed=1)[2]
    b = embed.embed(ck, corpus, dataset="sim", w_eval=1, eval_seed=2)[2]
    assert a.shape == b.shape
    assert not np.array_equal(a, b), "subsampling must depend on the stamped seed"


def test_w_eval_bounds_the_windows_per_cell_and_is_stamped(tmp_path, monkeypatch):
    """`w_eval` must bound the windows per cell and must be recorded in the stamp."""
    corpus_dir = build_sim_corpus(tmp_path / "deep", n_cells=6, mols_per_cell=600)
    ck = Checkpoint.load(write_checkpoint(tmp_path), model_factory=factory)
    corpus = scrm_py.Corpus(path=str(corpus_dir), seq_len=16384, max_umi_len=1600, min_v=1)
    _corpus = corpus
    monkeypatch.setattr(
        embed.controls, "corpus_facts",
        lambda ds, corpus=None, root=None: ({"k": 19}, {ds: "b" * 64}, list(_corpus.cell_names())),
    )
    rows = {}
    for w in (1, 2, 4):
        cells, windows, mat, stamp = embed.embed(ck, corpus, dataset="sim", w_eval=w, eval_seed=3)
        assert stamp.w_eval == w
        per_cell = {}
        for c in cells:
            per_cell[c] = per_cell.get(c, 0) + 1
        assert max(per_cell.values()) <= w, f"w_eval={w} exceeded: {max(per_cell.values())}"
        if w <= 2:
            assert max(per_cell.values()) == w, (
                f"w_eval={w} did not bind -- the deepest cell produced "
                f"{max(per_cell.values())} windows, so this fixture cannot see w_eval being ignored"
            )
        rows[w] = mat.shape[0]
    assert rows[1] < rows[2] < rows[4], (
        f"the artifact must grow with the window budget, got {rows}"
    )


def test_w_eval_below_one_is_refused(tmp_path):
    ck = Checkpoint.load(write_checkpoint(tmp_path), model_factory=factory)
    with pytest.raises(EmbedError, match="at least 1"):
        embed.embed(ck, object(), dataset="sim", w_eval=0, eval_seed=0)


def test_a_model_returning_the_wrong_number_of_vectors_is_refused(tmp_path, monkeypatch):
    """A model returning a row count other than one vector per window must be refused."""
    corpus_dir = build_sim_corpus(tmp_path / "sim")
    ck = Checkpoint.load(write_checkpoint(tmp_path), model_factory=factory)
    corpus = scrm_py.Corpus(path=str(corpus_dir), seq_len=16384, max_umi_len=1600, min_v=1)
    _corpus = corpus
    monkeypatch.setattr(
        embed.controls, "corpus_facts",
        lambda ds, corpus=None, root=None: ({"k": 19}, {ds: "b" * 64}, list(_corpus.cell_names())),
    )

    class Pooling(StubEncoder):
        def embed_windows(self, *a, **k):
            return super().embed_windows(*a, **k)[:1]

    object.__setattr__(ck, "model", Pooling(16))
    with pytest.raises(EmbedError, match="per window"):
        embed.embed(ck, corpus, dataset="sim", w_eval=2, eval_seed=0)


@needs_real
def test_the_whole_chain_on_the_real_corpus(tmp_path):
    """The same chain must round-trip on a real corpus, with the stamp cross-checked against the
    pinned digest fixture.
    """
    from scrm_eval import controls

    pinned = controls._pinned()
    corpora = pinned.tree_for(pinned.declared(REAL_CORPUS / next(
        d.name for d in REAL_CORPUS.iterdir() if (d / "manifest.json").exists()
    )))
    assert corpora, f"{REAL_CORPUS} is built under a molecule definition nothing has pinned"
    available = {d: v for d, v in corpora.items() if (REAL_CORPUS / d / "manifest.json").is_file()}
    assert available, (
        f"the pinned fixture names {len(corpora)} corpora for this molecule definition and none of "
        f"them is present under {REAL_CORPUS}"
    )
    dataset = min(available, key=lambda d: available[d]["n_rows"])
    ck = Checkpoint.load(write_checkpoint(tmp_path), model_factory=factory)
    corpus = scrm_py.Corpus(path=str(REAL_CORPUS / dataset), seq_len=16384, max_umi_len=1600, min_v=1)

    cells, windows, mat, stamp = embed.embed(
        ck, corpus, dataset=dataset, w_eval=1, eval_seed=5, batch_cells=64
    )
    assert stamp.n_cells == corpus.n_cells()
    assert stamp.corpus_digests[dataset] == corpora[dataset]["stream_sha256"], (
        "the stamp must carry the corpus's pinned content digest, so an embedding names the exact data "
        "it was computed on"
    )
    assert mat.shape == (len(cells), 16)
    out = tmp_path / "real.arrow"
    artifact.write(out, cells, windows, mat, stamp)
    back = artifact.read(out)
    assert np.array_equal(back.matrix, mat)
    _, reduced = reduce.reduce_windows(back.cell_ids, back.matrix, "mean")
    assert reduced.shape[0] == stamp.n_cells


def _sim(tmp_path, monkeypatch, dim=16, mols_per_cell=600):
    """The resume fixture: deep enough that `w_eval` changes the answer."""
    corpus_dir = build_sim_corpus(tmp_path / "sim", n_cells=12, mols_per_cell=mols_per_cell)
    ck = Checkpoint.load(write_checkpoint(tmp_path, arch={**ARCH, "dim_global": dim}), model_factory=factory)
    corpus = scrm_py.Corpus(path=str(corpus_dir), seq_len=16384, max_umi_len=1600, min_v=1)
    _corpus = corpus
    monkeypatch.setattr(
        embed.controls, "corpus_facts",
        lambda ds, corpus=None, root=None: ({"k": 19}, {ds: "b" * 64}, list(_corpus.cell_names())),
    )
    return ck, corpus


def test_a_resumed_embedding_equals_one_that_was_never_interrupted(tmp_path, monkeypatch):
    ck, corpus = _sim(tmp_path, monkeypatch)
    kw = dict(dataset="sim", w_eval=2, eval_seed=11, window_batch=4)
    whole = embed.embed(ck, corpus, **kw)

    rd = tmp_path / "resume"
    calls = {"n": 0}
    real = ck.model.embed_windows

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("preempted")
        return real(*a, **k)

    monkeypatch.setattr(ck.model, "embed_windows", flaky, raising=False)
    with pytest.raises(RuntimeError):
        embed.embed(ck, corpus, resume_dir=rd, flush_interval_s=0.0, **kw)
    assert list(rd.glob("chunk-*.npz")), "nothing was checkpointed, so a requeue would restart from zero"

    monkeypatch.setattr(ck.model, "embed_windows", real, raising=False)
    resumed = embed.embed(ck, corpus, resume_dir=rd, flush_interval_s=0.0, **kw)

    assert resumed[0] == whole[0], "cell order changed across resume"
    assert np.array_equal(resumed[1], whole[1]), "per-cell window ordinals changed across resume"
    assert np.array_equal(resumed[2], whole[2]), "the matrix is not bitwise identical across resume"
    assert resumed[3].cell_order_sha256 == whole[3].cell_order_sha256
    assert resumed[3].n_windows == whole[3].n_windows


def test_resume_state_from_a_different_computation_is_discarded(tmp_path, monkeypatch):
    """Resume state recorded under a different `w_eval` must be discarded, not replayed."""
    ck, corpus = _sim(tmp_path, monkeypatch)
    rd = tmp_path / "resume"
    embed.embed(ck, corpus, dataset="sim", w_eval=2, eval_seed=11, batch_cells=2,
                resume_dir=rd, flush_interval_s=0.0)
    assert list(rd.glob("chunk-*.npz"))

    at2 = embed.embed(ck, corpus, dataset="sim", w_eval=2, eval_seed=11, batch_cells=2)
    at4 = embed.embed(ck, corpus, dataset="sim", w_eval=4, eval_seed=11, batch_cells=2)
    assert at2[2].shape != at4[2].shape, (
        f"the fixture makes w_eval immaterial ({at2[2].shape} at 2 vs {at4[2].shape} at 4), so "
        f"replaying the stale chunks would produce the expected answer"
    )
    reused = embed.embed(ck, corpus, dataset="sim", w_eval=4, eval_seed=11, batch_cells=2,
                         resume_dir=rd, flush_interval_s=0.0)
    assert np.array_equal(reused[2], at4[2]), "stale w_eval=2 chunks leaked into a w_eval=4 embedding"
    assert reused[3].w_eval == 4


def test_shards_merge_into_exactly_the_unsharded_artifact(tmp_path, monkeypatch):
    """Three cell-range shards, concatenated, must equal the one-shot embedding BITWISE."""
    ck, corpus = _sim(tmp_path, monkeypatch)
    kw = dict(dataset="sim", w_eval=2, eval_seed=11, batch_cells=2)
    whole = embed.embed_to_file(ck, corpus, tmp_path / "whole.arrow", **kw)
    n = corpus.n_cells()

    bounds = [(0, 4), (4, 8), (8, n)]
    shards = []
    for lo, hi in bounds:
        shards.append(embed.embed_to_file(ck, corpus, tmp_path / f"s{lo}.arrow", cell_range=(lo, hi), **kw))
    merged = embed.merge_shards(shards, tmp_path / "merged.arrow", dataset="sim")

    a, b = artifact.read(whole), artifact.read(merged)
    assert list(a.cell_ids) == list(b.cell_ids), "merged cell order differs from the unsharded run"
    assert np.array_equal(a.windows, b.windows), "merged per-cell window ordinals differ"
    assert np.array_equal(a.matrix, b.matrix), "merged matrix is not bitwise identical"
    assert a.stamp.cell_order_sha256 == b.stamp.cell_order_sha256
    assert a.stamp.n_cells == b.stamp.n_cells and a.stamp.n_windows == b.stamp.n_windows
    assert a.stamp.w_eval == b.stamp.w_eval
    # The merge must carry the ARITHMETIC too: forward_precision is identity, not metadata.
    assert a.stamp.forward_precision == b.stamp.forward_precision, (
        "merged artifact lost forward_precision; it is identity, not metadata"
    )


def test_a_merge_that_does_not_cover_every_cell_is_refused(tmp_path, monkeypatch):
    """`merge_shards` must refuse a gap or a cover short of every cell."""
    ck, corpus = _sim(tmp_path, monkeypatch)
    kw = dict(dataset="sim", w_eval=2, eval_seed=11, batch_cells=2)
    s0 = embed.embed_to_file(ck, corpus, tmp_path / "s0.arrow", cell_range=(0, 4), **kw)
    s2 = embed.embed_to_file(ck, corpus, tmp_path / "s2.arrow", cell_range=(8, corpus.n_cells()), **kw)

    with pytest.raises(embed.EmbedError, match="contiguous"):
        embed.merge_shards([s0, s2], tmp_path / "gap.arrow", dataset="sim")
    with pytest.raises(embed.EmbedError, match="cells; refusing"):
        embed.merge_shards([s0], tmp_path / "short.arrow", dataset="sim")


def test_a_merge_of_shards_from_two_different_computations_is_refused(tmp_path, monkeypatch):
    """Shards that tile the cells perfectly but were not computed by one run must not merge."""
    ck, corpus = _sim(tmp_path, monkeypatch)
    n = corpus.n_cells()
    base = dict(dataset="sim", w_eval=2, batch_cells=2, eval_seed=11)

    def shard(lo, hi, **over):
        return embed.embed_to_file(ck, corpus, tmp_path / f"s{lo}_{hi}_{over.get('eval_seed', 11)}.arrow",
                                   cell_range=(lo, hi), **(base | over))

    good = [shard(0, 4), shard(4, 8), shard(8, n)]
    embed.merge_shards(good, tmp_path / "ok.arrow", dataset="sim")

    odd = shard(4, 8, eval_seed=99)
    with pytest.raises(embed.EmbedError, match="disagrees with the first on eval_seed"):
        embed.merge_shards([good[0], odd, good[2]], tmp_path / "bad.arrow", dataset="sim")

    with pytest.raises(embed.EmbedError, match="contiguous"):
        embed.merge_shards([shard(0, 8), shard(4, n)], tmp_path / "dup.arrow", dataset="sim")


def test_any_cell_is_a_legal_shard_boundary(tmp_path, monkeypatch):
    """Any cell must be a legal shard start, with the vectors identical to the unsharded run."""
    ck, corpus = _sim(tmp_path, monkeypatch)
    kw = dict(dataset="sim", w_eval=2, eval_seed=11, window_batch=4)
    whole = embed.embed(ck, corpus, **kw)
    shard = embed.embed(ck, corpus, cell_range=(3, 8), **kw)

    lo = whole[0].index(shard[0][0])
    assert shard[0] == whole[0][lo:lo + len(shard[0])], "unaligned shard changed the cell order"
    assert np.array_equal(shard[1], whole[1][lo:lo + len(shard[0])]), "window ordinals changed"
    assert np.array_equal(shard[2], whole[2][lo:lo + len(shard[0])]), "vectors are not identical"


def test_a_second_writer_does_not_interleave_its_chunks(tmp_path, monkeypatch):
    """A second live writer sharing a resume dir must start clean rather than interleave its chunks
    with the first writer's.
    """
    ck, corpus = _sim(tmp_path, monkeypatch)
    kw = dict(dataset="sim", w_eval=2, eval_seed=11, window_batch=4)
    real = ck.model.embed_windows

    clean = {"n": 0}

    def counting(*a, **k):
        clean["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(ck.model, "embed_windows", counting, raising=False)
    whole = embed.embed(ck, corpus, **kw)
    monkeypatch.setattr(ck.model, "embed_windows", real, raising=False)
    full_forwards = clean["n"]
    assert full_forwards > 2, (
        f"a clean run takes only {full_forwards} forwards, so 'B did less work than a clean run' "
        f"is not observable; the fixture needs more windows or a smaller window_batch"
    )
    rd = tmp_path / "shared"

    monkeypatch.setenv("SLURM_JOB_ID", "111")
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("preempted")
        return real(*a, **k)

    monkeypatch.setattr(ck.model, "embed_windows", flaky, raising=False)
    with pytest.raises(RuntimeError):
        embed.embed(ck, corpus, resume_dir=rd, flush_interval_s=0.0, **kw)
    assert list(rd.glob("chunk-*.npz")), "writer A left no chunks"
    assert (rd / "owner").read_text().strip() == "111"

    forwards = {"n": 0}

    def counted(*a, **k):
        forwards["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(ck.model, "embed_windows", counted, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "222")
    b = embed.embed(ck, corpus, resume_dir=rd, flush_interval_s=0.0, **kw)
    assert (rd / "owner").read_text().strip() == "222"
    assert list(b[0]) == list(whole[0])
    assert np.array_equal(b[2], whole[2]), "second writer's result diverged from the unsharded run"
    assert forwards["n"] == full_forwards, (
        f"writer B ran {forwards['n']} forwards where a clean run takes {full_forwards} -- it "
        f"replayed the LIVE writer A's chunks instead of discarding them, so two writers are "
        f"interleaving into one directory"
    )

    monkeypatch.setenv("SLURM_JOB_ID", "222")
    c = embed.embed(ck, corpus, resume_dir=rd, flush_interval_s=0.0, **kw)
    assert np.array_equal(c[2], whole[2])


def test_a_chunk_from_a_different_computation_is_refused_even_if_key_json_agrees(tmp_path, monkeypatch):
    """A chunk from a different computation must be refused on its own fingerprint, even where
    `key.json` agrees.
    """
    ck, corpus = _sim(tmp_path, monkeypatch)
    rd = tmp_path / "r"
    kw = dict(dataset="sim", eval_seed=11, batch_cells=2)
    embed.embed(ck, corpus, w_eval=2, resume_dir=rd, flush_interval_s=0.0, **kw)
    chunks = sorted(rd.glob("chunk-*.npz"))
    assert chunks

    at2 = embed.embed(ck, corpus, w_eval=2, **kw)
    at4 = embed.embed(ck, corpus, w_eval=4, **kw)
    assert at2[2].shape != at4[2].shape, (
        f"the fixture makes w_eval immaterial ({at2[2].shape} at 2 vs {at4[2].shape} at 4)"
    )
    import json as _json
    key = _json.loads((rd / "key.json").read_text())
    key["w_eval"] = 4
    (rd / "key.json").write_text(_json.dumps(key, sort_keys=True) + "\n")
    got = embed.embed(ck, corpus, w_eval=4, resume_dir=rd, flush_interval_s=0.0, **kw)
    assert np.array_equal(got[2], at4[2]), "w_eval=2 chunks were replayed into a w_eval=4 embedding"


def test_a_chunk_whose_matrix_has_the_wrong_row_count_stops_replay(tmp_path, monkeypatch):
    """A chunk whose matrix row count disagrees with its labels must stop replay."""
    ck, corpus = _sim(tmp_path, monkeypatch)
    rd = tmp_path / "r"
    kw = dict(dataset="sim", eval_seed=11, batch_cells=2)
    clean = embed.embed(ck, corpus, w_eval=2, **kw)
    embed.embed(ck, corpus, w_eval=2, resume_dir=rd, flush_interval_s=0.0, **kw)
    chunks = sorted(rd.glob("chunk-*.npz"))
    assert chunks, "no chunks to tamper with"

    z = dict(np.load(chunks[0], allow_pickle=False))
    rows = np.asarray(z["matrix"])
    assert rows.shape[0] >= 2, (
        f"the chunk holds {rows.shape[0]} row(s), so halving it cannot produce a label/row mismatch"
    )
    z["matrix"] = rows[: rows.shape[0] // 2]
    np.savez(chunks[0], **z)

    got = embed.embed(ck, corpus, w_eval=2, resume_dir=rd, flush_interval_s=0.0, **kw)
    assert np.array_equal(got[2], clean[2]), (
        "a chunk with fewer matrix rows than labels was replayed; the embedding is now a mix of "
        "rows nobody computed together"
    )


def test_padding_rows_are_not_forwarded_and_real_rows_are_unchanged(tmp_path, monkeypatch):
    """Padding rows must not be forwarded, and real rows' vectors must be unchanged."""
    ck, corpus = _sim(tmp_path, monkeypatch, dim=16)
    inner = ck.model.embed_windows

    w_eval = 32
    forwarded: list[int] = []

    def counting(tokens, valid, cseg, useg, psi):
        forwarded.append(int(np.asarray(tokens).shape[0]))
        return inner(tokens, valid, cseg, useg, psi)

    monkeypatch.setattr(ck.model, "embed_windows", counting, raising=False)
    cells, windows, matrix, _ = embed.embed(ck, corpus, dataset="sim", w_eval=w_eval, eval_seed=7)
    n_rows = matrix.shape[0]

    assert sum(forwarded) == n_rows, f"forwarded {sum(forwarded)} rows for {n_rows} real windows"
    n_cells = len(set(cells))
    assert n_rows < n_cells * w_eval, "the fixture must be padding-heavy or it tests nothing"

    def unfiltered():
        out_rows = []
        for start in range(0, n_cells, max(2, min(64, 16384 // w_eval))):
            idx = list(range(start, min(start + max(2, min(64, 16384 // w_eval)), n_cells)))
            b = corpus.build_batch(idx, epoch_seed=7, v=w_eval, allow_short=True)
            tok, val = np.asarray(b.tokens), np.asarray(b.valid)
            cseg, useg = np.asarray(b.contig_segment_id), np.asarray(b.umi_segment_id)
            psi = np.asarray(b.patch_start_ids)
            vecs = np.asarray(inner(tok, val, cseg, useg, psi), dtype=np.float32)
            out_rows.append(vecs[np.asarray(b.n_patches) > 0])
        return np.concatenate(out_rows, axis=0)

    assert np.array_equal(matrix, unfiltered()), "skipping padding changed a real row's vector"


def test_window_streaming_is_bitwise_identical(tmp_path, monkeypatch):
    """Window grouping must change how windows are batched, never the vectors they produce."""
    ck, corpus = _sim(tmp_path, monkeypatch, dim=16)
    w_eval, seed = 4, 11
    cells, windows, matrix, stamp = embed.embed(
        ck, corpus, dataset="sim", w_eval=w_eval, eval_seed=seed, window_batch=8)

    n_cells = corpus.n_cells()
    exp_cells, exp_windows, exp_rows = [], [], []
    for start in range(0, n_cells, 2):
        idx = list(range(start, min(start + 2, n_cells)))
        b = corpus.build_batch(idx, epoch_seed=seed, v=w_eval, allow_short=True)
        real = np.asarray(b.n_patches) > 0
        vecs = np.asarray(ck.model.embed_windows(
            np.asarray(b.tokens), np.asarray(b.valid), np.asarray(b.contig_segment_id),
            np.asarray(b.umi_segment_id), np.asarray(b.patch_start_ids)), dtype=np.float32)
        ci = np.asarray(b.cell_index)
        seen: dict[int, int] = {}
        for row, c in enumerate(ci.tolist()):
            if not real[row]:
                continue
            exp_cells.append(list(corpus.cell_names())[idx[c]])
            exp_windows.append(seen.get(c, 0))
            seen[c] = seen.get(c, 0) + 1
        exp_rows.append(vecs[real])
    expected = np.vstack(exp_rows)

    assert cells == exp_cells, "streaming changed the cell order"
    assert np.array_equal(windows, np.asarray(exp_windows, dtype=np.uint16)), "ordinals changed"
    assert matrix.shape == expected.shape, f"{matrix.shape} vs {expected.shape}"
    assert np.array_equal(matrix, expected), "streaming changed a vector; it must only regroup them"
    assert stamp.n_windows == expected.shape[0]


def test_a_dead_writer_is_inherited_and_a_live_one_is_not(tmp_path, monkeypatch):
    """A dead writer's chunks must be inherited and a live writer's must not."""
    ck, corpus = _sim(tmp_path, monkeypatch)
    kw = dict(dataset="sim", w_eval=2, eval_seed=11, window_batch=4)
    rd = tmp_path / "resume"

    monkeypatch.setenv("SLURM_JOB_ID", "111")
    embed.embed(ck, corpus, resume_dir=rd, flush_interval_s=0.0, **kw)
    chunks = sorted(rd.glob("chunk-*.npz"))
    assert chunks, "writer A left no chunks"
    n_before = len(chunks)

    monkeypatch.setenv("SLURM_JOB_ID", "222")
    old = _t.time() - (embed._Resume.STALE_AFTER_S + 60)
    os.utime(rd / "owner", (old, old))
    key = json.loads((rd / "key.json").read_text())
    r = embed._Resume(rd, key, 0.0)
    r.load([], [], [])
    assert len(sorted(rd.glob("chunk-*.npz"))) == n_before, \
        "a re-submission deleted its predecessor's chunks"

    monkeypatch.setenv("SLURM_JOB_ID", "333")
    r2 = embed._Resume(rd, key, 0.0)
    r2.load([], [], [])
    assert not sorted(rd.glob("chunk-*.npz")), \
        "two live writers must not interleave; the newcomer starts clean"


def test_a_batching_change_does_not_invalidate_a_checkpoint(tmp_path, monkeypatch):
    """A key differing only in batching fields must still match a stored checkpoint."""
    ck, corpus = _sim(tmp_path, monkeypatch)
    kw = dict(dataset="sim", w_eval=2, eval_seed=11, window_batch=4)
    rd = tmp_path / "resume"
    monkeypatch.setenv("SLURM_JOB_ID", "111")
    embed.embed(ck, corpus, resume_dir=rd, flush_interval_s=0.0, **kw)
    n_before = len(sorted(rd.glob("chunk-*.npz")))
    assert n_before, "nothing checkpointed"
    key = json.loads((rd / "key.json").read_text())
    assert not (set(key) & set(embed.BATCHING_FIELDS)), "batching must not be written into the key"

    for legacy in ({**key, "batch_cells": 42}, {**key, "window_batch": 512}):
        (rd / "key.json").write_text(json.dumps(legacy, sort_keys=True))
        old = _t.time() - (embed._Resume.STALE_AFTER_S + 60)
        os.utime(rd / "owner", (old, old))
        monkeypatch.setenv("SLURM_JOB_ID", "222")
        r = embed._Resume(rd, key, 0.0)
        replayed = r.load([], [], [])
        assert len(sorted(rd.glob("chunk-*.npz"))) == n_before, \
            f"a {sorted(set(legacy) & set(embed.BATCHING_FIELDS))} key discarded the chunks"
        assert replayed > 0, "chunks survived but were not replayed"

    (rd / "key.json").write_text(json.dumps({**key, "w_eval": 999}, sort_keys=True))
    r = embed._Resume(rd, key, 0.0)
    r.load([], [], [])
    assert not sorted(rd.glob("chunk-*.npz")), "a w_eval change must still invalidate the checkpoint"


def test_group_readout_labels_are_not_hardcoded_to_pairs():
    """`embed.py` must label one row per row `pool_pairs` RETURNS, not per assumed pair of two."""
    src = (Path(__file__).resolve().parents[2] / "eval/scrm_eval/embed.py").read_text()
    body = src[src.index("if pair_mode:"):]
    body = body[:body.index("staged = staged[emitted:]")]
    code = "\n".join(ln.split("#", 1)[0] for ln in body.splitlines())
    assert "k // 2" not in code, (
        "embed.py's group-readout branch contains `k // 2`. The number of labels must come from the "
        "rows pool_pairs returned (V is the checkpoint's, not 2), or a V!=2 arm writes a ragged "
        "artifact."
    )
    assert "rows.shape[0]" in code, (
        "the label loop should iterate the returned array's own row count, so a shape read off the "
        "data cannot drift from the data"
    )


def test_shards_under_different_arithmetic_are_refused(tmp_path, monkeypatch):
    """Two halves computed under different forward arithmetic are not one artifact.

    `forward_precision` is identity: bf16+eager and fp32+eager embeddings of the same cell by the
    same checkpoint are not interchangeable. A merge that accepted both would manufacture an
    artifact that never existed, and `refuse_mixed_precision` downstream could not see it because
    the merged stamp reports a single value.
    """
    import dataclasses

    ck, corpus = _sim(tmp_path, monkeypatch)
    n = corpus.n_cells()
    kw = dict(dataset="sim", w_eval=2, eval_seed=11, batch_cells=2)
    s0 = embed.embed_to_file(ck, corpus, tmp_path / "a0.arrow", cell_range=(0, 6), **kw)
    s1 = embed.embed_to_file(ck, corpus, tmp_path / "a1.arrow", cell_range=(6, n), **kw)

    # Rewrite one shard's stamp to claim different arithmetic, leaving the data untouched.
    emb = artifact.read(s1)
    other = "bf16+eager" if emb.stamp.forward_precision != "bf16+eager" else "fp32+eager"
    artifact.write(tmp_path / "a1_odd.arrow", list(emb.cell_ids), emb.windows, emb.matrix,
                   dataclasses.replace(emb.stamp, forward_precision=other))

    with pytest.raises(embed.EmbedError, match="disagrees with the first on forward_precision"):
        embed.merge_shards([s0, tmp_path / "a1_odd.arrow"], tmp_path / "mixed.arrow", dataset="sim")
