#!/usr/bin/env python3
"""The packer through its Python API: unit, integration, simulated data, real data."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from pyarrow import ipc

import scrm_py

REPO = Path(__file__).resolve().parents[3]

_ROOT_ENV = ("SCRM_CORPUS_ROOT", "CORPORA_ROOT", "CORPUS_ROOT")


def _corpus_root() -> Path | None:
    """The corpus tree, or None when no environment variable names one."""
    env = next((v for k in _ROOT_ENV if (v := os.environ.get(k))), None)
    if env:
        return Path(env)
    proj = os.environ.get("SCRM_PROJ")
    return Path(proj) / "corpus" if proj else None


REAL_CORPUS = _corpus_root()
SEQ_LEN = 16384
TAB, NEWLINE = 5, 6

def _real_corpora() -> list[Path]:
    """Every dataset present under the corpus root."""
    if REAL_CORPUS is None or not REAL_CORPUS.is_dir():
        return []
    return sorted(p for p in REAL_CORPUS.iterdir() if (p / "manifest.json").exists())


needs_real = pytest.mark.skipif(
    not _real_corpora(),
    reason=f"no corpus with a manifest.json under {REAL_CORPUS}" if REAL_CORPUS
    else f"no corpus root: set one of {', '.join(_ROOT_ENV)} or SCRM_PROJ",
)


def _smallest_corpus() -> Path:
    """The corpus with the fewest rows, as the standing subject for real-data tests."""
    found = _real_corpora()
    if not found:
        return (REAL_CORPUS or Path("no-corpus-root")) / "none"
    return min(found, key=lambda d: json.loads((d / "manifest.json").read_text())["n_rows"])


SMALL = _smallest_corpus()


def splitmix64(state):
    """splitmix64, from the algorithm rather than from the Rust source."""
    M = (1 << 64) - 1
    state = (state + 0x9E3779B97F4A7C15) & M
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
    return state, z ^ (z >> 31)


def derive_seed(epoch_seed, a, b):
    """`scrm_pack::derive_seed`, from its documented construction."""
    M = (1 << 64) - 1
    s = epoch_seed ^ 0xA0761D6478BD642F
    s = (s + (a * 0x9E3779B97F4A7C15)) & M
    s, _ = splitmix64(s)
    s = (s + (b * 0xC2B2AE3D27D4EB4F)) & M
    _, out = splitmix64(s)
    return out


def draw_without_replacement(n, want, seed):
    """Partial Fisher-Yates, matching the Rust draw exactly."""
    scratch = list(range(n))
    want = min(want, n)
    state = seed
    out = []
    for i in range(want):
        state, r = splitmix64(state)
        j = i + r % (n - i)
        scratch[i], scratch[j] = scratch[j], scratch[i]
        out.append(scratch[i])
    return out


def reference_tile_first(lengths, order, seq_len, want):
    """The tile: whole molecules, at most seq_len, one separator between each pair."""
    windows, current, used = [], [], 0
    for idx in order:
        ln = lengths[idx]
        cost = ln if not current else ln + 1
        if used + cost > seq_len:
            windows.append(current)
            if len(windows) == want:
                return windows
            current, used = [idx], ln
        else:
            current.append(idx)
            used += cost
    if current and len(windows) < want:
        windows.append(current)
    return windows


def reference_pack(window, rows, seq_len, epoch_seed, cell_i, shuffle_contigs=True):
    """The pack, from the spec: six arrays, contig-level boundaries, terminators inside spans."""
    tokens = np.full(seq_len, NEWLINE, dtype=np.uint8)
    cseg = np.zeros(seq_len, dtype=np.uint16)
    useg = np.zeros(seq_len, dtype=np.uint16)
    patches = []
    cursor = 0
    n_contigs = 0
    last_mol = len(window) - 1

    for mol_i, row in enumerate(window):
        raw = rows[row]
        patches.append(cursor)
        pieces = []
        start = 0
        for pos, b in enumerate(raw):
            if b == TAB:
                pieces.append(raw[start:pos])
                start = pos + 1
        pieces.append(raw[start:])

        if shuffle_contigs:
            s = derive_seed(epoch_seed, cell_i, row + 1)
            perm = draw_without_replacement(len(pieces), len(pieces), s)
        else:
            perm = list(range(len(pieces)))

        last_contig = len(perm) - 1
        for k, pi in enumerate(perm):
            piece = pieces[pi]
            ln = len(piece)
            if k < last_contig:
                sep = TAB
            elif mol_i < last_mol:
                sep = NEWLINE
            else:
                sep = None
            span = ln + (1 if sep is not None else 0)
            tokens[cursor : cursor + ln] = np.frombuffer(bytes(piece), dtype=np.uint8)
            if sep is not None:
                tokens[cursor + ln] = sep
            cseg[cursor : cursor + span] = n_contigs
            useg[cursor : cursor + span] = mol_i
            cursor += span
            n_contigs += 1

    valid = np.zeros(seq_len, dtype=bool)
    valid[:cursor] = True
    return tokens, valid, cseg, useg, np.array(patches, dtype=np.uint32), cursor


def read_cell_rows(corpus_dir, cell_index):
    """A cell's rows, straight from the corpus with pyarrow -- no Rust involved."""
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    rows, names = [], []
    for entry in manifest["shards"]:
        path = corpus_dir / entry["file"]
        with pa.memory_map(str(path), "rb") as src:
            r = ipc.open_file(src)
            index = json.loads(r.metadata[b"scrm_cell_rows"])
            flat = []
            for b in range(r.num_record_batches):
                t = r.get_batch(b).column("tokens")
                offs = np.asarray(t.offsets)
                vals = np.asarray(t.values)
                for i in range(len(offs) - 1):
                    flat.append(bytes(vals[offs[i] : offs[i + 1]]))
            for run in index:
                names.append(run["cell"])
                rows.append(flat[run["start"] : run["start"] + run["len"]])
    return names[cell_index], rows[cell_index]


CDNA = "ACGTTGCATTGACCGATCAGTTGCAATCGGATCAGTTACGGATCCAGTTAACGGTTCAGC"
BIN = REPO / "target" / "release" / "scrm-corpus"


def build_sim_corpus(tmp_path, cells_spec, *, min_umi_len=1, min_cell_molecules=1, max_umi_len=None,
                     sample="sim"):
    """Build a corpus from synthetic FASTQ with the real binary."""
    if not BIN.exists():
        pytest.skip(f"no release binary at {BIN}; run cargo build --release -p scrm-cli")
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    r1, r2 = [], []
    i = 0
    umi_n = 0
    for cb, seqs in cells_spec.items():
        for seq in seqs:
            u, umi = umi_n, ""
            for _ in range(12):
                umi = "ACGT"[u % 4] + umi
                u //= 4
            umi_n += 1
            r1.append(f"@R{i} 1:N:0:A\n{cb}{umi}\n+\n{'I' * 28}\n")
            r2.append(f"@R{i} 2:N:0:A\n{seq}\n+\n{'I' * len(seq)}\n")
            i += 1
    f1 = tmp_path / f"{sample}_S1_L001_R1_001.fastq"
    f2 = tmp_path / f"{sample}_S1_L001_R2_001.fastq"
    f1.write_text("".join(r1))
    f2.write_text("".join(r2))
    bc = tmp_path / "barcodes.tsv"
    bc.write_text("".join(f"{c}-1\n" for c in cells_spec))
    sheet = tmp_path / "samplesheet.csv"
    sheet.write_text(
        "experiment,r1,r2,barcodes,geometry\n"
        f"corpus,{f1},{f2},{bc},1{{b[16]u[12]}}2{{r:}}\n"
    )
    out = tmp_path / "corpus"
    subprocess.run(
        [
            str(BIN),            "--samplesheet", str(sheet), "--out", str(tmp_path), "--threads", "2",
            "--min-umi-len", str(min_umi_len), "--min-cell-molecules", str(min_cell_molecules),
            *(["--max-umi-len", str(max_umi_len)] if max_umi_len is not None else []),
        ],
        check=True,
        capture_output=True,
    )
    return out


def test_module_constants_come_from_rust():
    assert scrm_py.MIN_UMI_LEN_DEFAULT == 60
    assert scrm_py.MAX_UMI_LEN_DEFAULT == 1600
    assert scrm_py.MIN_CELL_MOLECULES_DEFAULT == 500
    assert scrm_py.max_patches(SEQ_LEN) == 268
    assert scrm_py.max_patches(SEQ_LEN) == SEQ_LEN // (scrm_py.MIN_UMI_LEN_DEFAULT + 1)
    assert scrm_py.MAX_UMI_LEN_DEFAULT * 10 <= SEQ_LEN, "the geometry margin must hold"


def test_geometry_is_refused_when_a_window_could_hold_one_molecule(tmp_path):
    corpus = build_sim_corpus(tmp_path, {"ACGTACGTACGTACGT": [CDNA[:40], CDNA[:45]]})
    with pytest.raises(RuntimeError, match="tenth of seq_len"):
        scrm_py.Corpus(str(corpus), seq_len=1024, max_umi_len=1024)
    with pytest.raises(RuntimeError, match="fit u16"):
        scrm_py.Corpus(str(corpus), seq_len=1 << 18, max_umi_len=1600)
    with pytest.raises(ValueError, match="min_v"):
        scrm_py.Corpus(str(corpus), min_v=0)


def test_the_geometry_is_asserted_against_the_bound_the_CORPUS_declares(tmp_path):
    """The geometry must be asserted against the `max_umi_len` bound the corpus declares."""
    corpus = build_sim_corpus(tmp_path, {"ACGTACGTACGTACGT": [CDNA[:40], CDNA[:45]]}, max_umi_len=2000)
    declared = json.loads((corpus / "manifest.json").read_text())["declared"]["config"]
    assert declared["max_umi_len"] == 2000, "the fixture must declare the wider bound"

    with pytest.raises(RuntimeError, match="tenth of seq_len"):
        scrm_py.Corpus(str(corpus), seq_len=SEQ_LEN)
    c = scrm_py.Corpus(str(corpus), seq_len=32768, min_v=1)
    assert c.compat_key()["max_umi_len"] == 2000


def test_opening_a_non_corpus_raises_rather_than_crashing():
    with pytest.raises(RuntimeError):
        scrm_py.Corpus("/tmp")
    with pytest.raises(RuntimeError):
        scrm_py.Corpus("/nonexistent/path/at/all")


def test_a_simulated_corpus_packs_to_exactly_the_expected_bytes(tmp_path):
    """A simulated corpus of hand-chosen content must pack to exactly the expected bytes."""
    cb = "ACGTACGTACGTACGT"
    seqs = [CDNA[:40], CDNA[:45], CDNA[:50]]
    corpus = build_sim_corpus(tmp_path, {cb: seqs})
    c = scrm_py.Corpus(str(corpus), seq_len=256, max_umi_len=25, min_v=1)
    assert c.n_cells() == 1
    assert c.cell_names() == [cb]
    assert list(c.molecules_per_cell()) == [3]

    b = c.build_batch([0], epoch_seed=1, v=1)
    used = int(b.valid[0].sum())
    assert used == sum(len(s) for s in seqs) + 2

    tok = b.tokens[0][:used]
    parts = bytes(tok).split(bytes([NEWLINE]))
    assert len(parts) == 3
    table = {0: "A", 1: "C", 2: "G", 3: "T"}
    decoded = sorted("".join(table[x] for x in p) for p in parts)
    assert decoded == sorted(seqs), "the packed bytes are not the corpus's sequences"

    assert np.array_equal(b.contig_segment_id[0][:used], b.umi_segment_id[0][:used])
    assert list(np.unique(b.umi_segment_id[0][:used])) == [0, 1, 2]


def test_a_shallow_cell_is_reported_by_cells_below_min_v(tmp_path):
    """A cell that cannot fill `min_v` windows must be reported by `cells_below_min_v`."""
    corpus = build_sim_corpus(
        tmp_path,
        {
            "ACGTACGTACGTACGT": [CDNA[:40]],
            "TTTTGGGGCCCCAAAA": [CDNA[:40], CDNA[:45], CDNA[:50], CDNA[:55]],
        },
    )
    c = scrm_py.Corpus(str(corpus), seq_len=64, max_umi_len=6, min_v=2)
    short = c.cells_below_min_v()
    names = c.cell_names()
    assert [names[i] for i in short] == ["ACGTACGTACGTACGT"], f"got {short} of {names}"

    bad = names.index("ACGTACGTACGTACGT")
    with pytest.raises(RuntimeError, match="could not supply"):
        c.build_batch([bad], epoch_seed=1, v=2)
    good = names.index("TTTTGGGGCCCCAAAA")
    assert c.build_batch([good], epoch_seed=1, v=2).n_windows == 2


def test_multi_contig_molecules_get_separate_attention_blocks(tmp_path):
    """A multi-contig molecule must get one attention block per contig."""
    cb = "ACGTACGTACGTACGT"
    corpus = build_sim_corpus(tmp_path, {cb: [CDNA[:40], CDNA[:45]]})
    c = scrm_py.Corpus(str(corpus), seq_len=256, max_umi_len=25, min_v=1)
    b = c.build_batch([0], epoch_seed=3, v=1)
    used = int(b.valid[0].sum())
    cseg = b.contig_segment_id[0][:used]
    useg = b.umi_segment_id[0][:used]
    assert np.all(np.diff(cseg.astype(np.int32)) >= 0)
    assert np.all(np.diff(useg.astype(np.int32)) >= 0)
    np_ = int(b.n_patches[0])
    assert np_ == int(useg[-1]) + 1, "n_patches must equal the window's molecule count"
    starts = b.patch_start_ids[0][:np_]
    for i in range(np_):
        s = int(starts[i])
        e = int(starts[i + 1]) if i + 1 < np_ else used
        assert e > s, f"patch {i} is empty"
        assert set(useg[s:e].tolist()) == {i}, f"patch {i} does not span exactly molecule {i}"


@needs_real
def test_the_rust_packer_matches_an_independent_python_implementation():
    """Every array, element-wise, against a reimplementation sharing no code with Rust."""
    dir_ = SMALL
    c = scrm_py.Corpus(str(dir_), min_v=2)
    cells = [0, 1, 2]
    seed = 20260816
    b = c.build_batch(cells, epoch_seed=seed, v=2)

    for slot, cell_i in enumerate(cells):
        _name, rows = read_cell_rows(dir_, cell_i)
        lengths = [len(r) for r in rows]

        mean = max(sum(lengths) // len(lengths), 1)
        per_window = SEQ_LEN // (mean + 1) + 1
        want = min(per_window * 2, len(lengths))
        order = draw_without_replacement(len(lengths), want, derive_seed(seed, cell_i, 0))
        windows = reference_tile_first(lengths, order, SEQ_LEN, 2)
        assert len(windows) >= 2, "fixture cell cannot supply 2 windows"

        for v in range(2):
            w = slot * 2 + v
            tok, valid, cseg, useg, patches, used = reference_pack(
                windows[v], rows, SEQ_LEN, seed, cell_i
            )
            assert used == int(b.valid[w].sum()), f"window {w}: occupancy differs"
            assert np.array_equal(tok, b.tokens[w]), f"window {w}: tokens differ"
            assert np.array_equal(valid, b.valid[w]), f"window {w}: valid differs"
            assert np.array_equal(cseg, b.contig_segment_id[w]), f"window {w}: contig ids differ"
            assert np.array_equal(useg, b.umi_segment_id[w]), f"window {w}: umi ids differ"
            n = int(b.n_patches[w])
            assert n == len(patches), f"window {w}: patch count differs"
            assert np.array_equal(patches, b.patch_start_ids[w][:n]), f"window {w}: patch starts differ"


@needs_real
def test_real_batch_shapes_dtypes_and_grouping():
    c = scrm_py.Corpus(str(SMALL))
    B, V = 32, 2
    b = c.build_batch(list(range(B)), epoch_seed=5, v=V)
    nw = B * V
    assert b.n_windows == nw and b.seq_len == SEQ_LEN and b.short_cells == 0

    for name, arr, dtype, shape in [
        ("tokens", b.tokens, np.uint8, (nw, SEQ_LEN)),
        ("valid", b.valid, np.bool_, (nw, SEQ_LEN)),
        ("contig_segment_id", b.contig_segment_id, np.uint16, (nw, SEQ_LEN)),
        ("umi_segment_id", b.umi_segment_id, np.uint16, (nw, SEQ_LEN)),
        ("cell_index", b.cell_index, np.uint32, (nw,)),
        ("n_patches", b.n_patches, np.uint32, (nw,)),
    ]:
        assert arr.shape == shape, f"{name}: shape {arr.shape} != {shape}"
        assert arr.dtype == dtype, f"{name}: dtype {arr.dtype} != {dtype}"
    assert b.patch_start_ids.shape == (nw, scrm_py.max_patches(SEQ_LEN))

    assert np.array_equal(b.cell_index, np.repeat(np.arange(B, dtype=np.uint32), V))

    assert int(b.tokens[b.valid].max()) < 7
    for w in range(nw):
        used = int(b.valid[w].sum())
        assert b.valid[w][:used].all() and not b.valid[w][used:].any()
    occupancy = b.valid.sum() / b.valid.size
    assert occupancy > 0.95, f"windows only {occupancy:.1%} full"


@needs_real
def test_real_batch_is_reproducible_and_the_seed_rotates_the_molecules():
    c = scrm_py.Corpus(str(SMALL))
    cells = list(range(8))
    a = c.build_batch(cells, epoch_seed=42, v=2)
    b = c.build_batch(cells, epoch_seed=42, v=2)
    assert np.array_equal(a.tokens, b.tokens), "same seed must give the same batch"
    assert np.array_equal(a.umi_segment_id, b.umi_segment_id)
    assert np.array_equal(a.patch_start_ids, b.patch_start_ids)

    d = c.build_batch(cells, epoch_seed=43, v=2)

    def molecules(batch, cell):
        out = set()
        for w in (cell * 2, cell * 2 + 1):
            used = int(batch.valid[w].sum())
            tok = batch.tokens[w][:used]
            useg = batch.umi_segment_id[w][:used]
            at = 0
            while at < used:
                end = at
                while end < used and useg[end] == useg[at]:
                    end += 1
                span = bytes(tok[at:end]).rstrip(bytes([NEWLINE]))
                out.add(bytes(sorted(span.split(bytes([TAB])))[0]) + b"|" + str(len(span)).encode())
                at = end
        return out

    for cell in range(8):
        x, y = molecules(a, cell), molecules(d, cell)
        assert x and y
        overlap = len(x & y) / len(x)
        assert overlap < 0.5, f"cell {cell}: two seeds drew {overlap:.0%} of the same molecules"


@needs_real
def test_real_batch_views_draw_no_molecule_twice():
    c = scrm_py.Corpus(str(SMALL))
    dir_ = SMALL
    n = 4
    b = c.build_batch(list(range(n)), epoch_seed=11, v=2)

    def canon(m):
        return tuple(sorted(bytes(m).split(bytes([TAB]))))

    for cell in range(n):
        _name, rows = read_cell_rows(dir_, cell)
        have = {}
        for r in rows:
            have[canon(r)] = have.get(canon(r), 0) + 1
        drawn = {}
        for w in (cell * 2, cell * 2 + 1):
            used = int(b.valid[w].sum())
            tok, useg = b.tokens[w][:used], b.umi_segment_id[w][:used]
            at = 0
            while at < used:
                end = at
                while end < used and useg[end] == useg[at]:
                    end += 1
                span = bytes(tok[at:end]).rstrip(bytes([NEWLINE]))
                k = canon(span)
                drawn[k] = drawn.get(k, 0) + 1
                at = end
        for k, cnt in drawn.items():
            assert k in have, f"cell {cell}: a window holds a molecule the cell does not"
            assert cnt <= have[k], (
                f"cell {cell}: a molecule was drawn {cnt} times but exists {have[k]} times -- the "
                "draw is not without replacement"
            )


@needs_real
def test_a_bad_cell_index_is_refused():
    c = scrm_py.Corpus(str(SMALL))
    with pytest.raises(ValueError, match="outside the corpus"):
        c.build_batch([0, 10**9], epoch_seed=1, v=2)


@needs_real
def test_the_gil_is_released_during_build_batch():
    """A Python thread must make progress while `build_batch` runs."""
    import threading

    c = scrm_py.Corpus(str(SMALL))
    counter = {"n": 0}
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            counter["n"] += 1

    t = threading.Thread(target=spin, daemon=True)
    t.start()
    before = counter["n"]
    c.build_batch(list(range(64)), epoch_seed=1, v=2)
    during = counter["n"] - before
    stop.set()
    t.join(timeout=5)
    assert during > 1000, (
        f"the Python thread advanced only {during} times during build_batch, so the GIL was held "
        "and prefetching from a thread would not overlap"
    )


def test_corpora_with_different_bounds_cannot_be_mixed(tmp_path):
    """Corpora declaring different bounds must not mix."""
    spec = {"ACGTACGTACGTACGT": [CDNA[:40], CDNA[:45], CDNA[:50]]}
    a = build_sim_corpus(tmp_path / "a", spec, min_umi_len=1, sample="sampA")
    b = build_sim_corpus(tmp_path / "b", spec, min_umi_len=30, sample="sampB")

    ka = json.loads((a / "manifest.json").read_text())["declared"]["config"]
    kb = json.loads((b / "manifest.json").read_text())["declared"]["config"]
    assert ka["min_umi_len"] != kb["min_umi_len"], "the fixture must actually differ"

    scrm_py.Corpus(str(a), seq_len=256, max_umi_len=25, min_v=1)
    scrm_py.Corpus(str(b), seq_len=256, max_umi_len=25, min_v=1)
    with pytest.raises(RuntimeError, match="compat_key differs"):
        scrm_py.Corpus([str(a), str(b)], seq_len=256, max_umi_len=25, min_v=1)


def test_corpora_with_equal_keys_mix_into_one_population(tmp_path):
    """Corpora with equal compat keys must mix into one population."""
    a = build_sim_corpus(
        tmp_path / "a", {"ACGTACGTACGTACGT": [CDNA[:40], CDNA[:45]]}, sample="sampA"
    )
    b = build_sim_corpus(
        tmp_path / "b", {"TTTTGGGGCCCCAAAA": [CDNA[:50], CDNA[:55]]}, sample="sampB"
    )
    one = scrm_py.Corpus([str(a), str(b)], seq_len=256, max_umi_len=25, min_v=1)
    assert one.n_cells() == 2
    assert sorted(one.cell_names()) == ["ACGTACGTACGTACGT", "TTTTGGGGCCCCAAAA"]
    got = one.build_batch([0, 1], epoch_seed=1, v=1)
    assert got.n_windows == 2
    assert int(got.valid.sum()) > 0


@needs_real
def test_every_real_corpus_mixes_despite_differing_umi_length():
    """umi_len 10 vs 12 must NOT block mixing -- 10 bp is Chromium V2, 12 bp is V3/V4."""
    dirs = sorted(str(p) for p in REAL_CORPUS.iterdir() if (p / "manifest.json").exists())
    import importlib.util

    _f = Path(__file__).parents[2] / "scrm-corpus/tests/pinned_digests.py"
    _spec = importlib.util.spec_from_file_location("scrm_pinned_digests", _f)
    _pd = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_pd)
    pinned = _pd.tree_for(_pd.declared(Path(dirs[0])))
    assert pinned, f"{REAL_CORPUS} is built under a molecule definition nothing has pinned"
    assert {Path(d).name for d in dirs} == set(pinned), (
        "the corpus on disk and the pinned fixture disagree about which datasets exist -- "
        "run pinned_digests.py <root> --write and say in the commit why the set changed"
    )
    lens = {
        json.loads((Path(d) / "manifest.json").read_text())["declared"]["config"]["umi_len"]
        for d in dirs
    }
    assert lens == {10, 12}, f"fixture must span both UMI lengths, got {lens}"
    one = scrm_py.Corpus(dirs, min_v=2)
    want = sum(c["cells"] for c in pinned.values())
    assert one.n_cells() == want, (
        f"the packer indexes {one.n_cells():,} cells across every corpus; the fixture pins {want:,}"
    )


def _budget_corpus(tmp_path):
    """One deep cell (12 distinct molecules) and one modest cell, at seq_len 64. Distinct slices of
    CDNA, so which molecules a window holds is readable from its bytes.
    """
    return build_sim_corpus(
        tmp_path,
        {
            "ACGTACGTACGTACGT": [CDNA[k : k + 40] for k in range(12)],
            "TTTTGGGGCCCCAAAA": [CDNA[:40], CDNA[:45], CDNA[:50], CDNA[:55]],
        },
    )


def test_full_budget_is_bit_identical_to_no_budget(tmp_path):
    """`frac=1.0` must be bitwise identical to no budget, whatever `frac_seed` says."""
    c = scrm_py.Corpus(str(_budget_corpus(tmp_path)), seq_len=64, max_umi_len=6, min_v=2)
    cells = list(range(c.n_cells()))
    a = c.build_batch(cells, epoch_seed=7, v=2)
    b = c.build_batch(cells, epoch_seed=7, v=2, frac=1.0, frac_seed=987654321)
    for field in ("tokens", "valid", "contig_segment_id", "umi_segment_id", "cell_index",
                  "patch_start_ids", "n_patches"):
        assert np.array_equal(np.asarray(getattr(a, field)), np.asarray(getattr(b, field))), field


def test_the_budget_subset_is_fixed_across_epochs_and_moves_with_its_seed(tmp_path):
    """At frac = 1/12 the deep cell keeps exactly one molecule: the valid token content must be
    identical across epoch seeds, and a different `frac_seed` must be able to keep a different
    molecule.
    """
    c = scrm_py.Corpus(str(_budget_corpus(tmp_path)), seq_len=64, max_umi_len=6, min_v=2)
    deep = c.cell_names().index("ACGTACGTACGTACGT")

    def content(epoch_seed, frac_seed):
        b = c.build_batch([deep], epoch_seed=epoch_seed, v=2, allow_short=True,
                          frac=1 / 12, frac_seed=frac_seed)
        tok, val = np.asarray(b.tokens), np.asarray(b.valid)
        return tuple(sorted(tuple(row[v]) for row, v in zip(tok, val) if v.any()))

    assert content(1, 42) == content(2, 42) == content(3, 42), \
        "the kept subset leaked epoch dependence -- the run-level budget guarantee is broken"
    seen = {content(1, fs) for fs in (1, 2, 3, 4, 5)}
    assert len(seen) > 1, "five frac_seeds kept the same single molecule of twelve; the subset is not seeded"


def test_cells_below_v_replays_the_runs_own_draws(tmp_path):
    """`cells_below_v` must replay the run's own draws: a cell it passes must never fail
    `build_batch` under any of the run's epoch seeds, and a cell it reports must fail under at
    least one.
    """
    c = scrm_py.Corpus(str(_budget_corpus(tmp_path)), seq_len=64, max_umi_len=6, min_v=2)
    seeds = [(20260817 * 0x9E3779B97F4A7C15 + e) % (1 << 64) for e in range(6)]

    below = set(c.cells_below_v(seeds, v=2))
    for i in range(c.n_cells()):
        failures = 0
        for s in seeds:
            try:
                c.build_batch([i], epoch_seed=s, v=2)
            except RuntimeError:
                failures += 1
        if i in below:
            assert failures > 0, f"cell {i} flagged but never fails"
        else:
            assert failures == 0, f"cell {i} passed the guard yet failed {failures} epochs"

    deep = c.cell_names().index("ACGTACGTACGTACGT")
    assert deep in set(c.cells_below_v(seeds, v=2, frac=1 / 12, frac_seed=42))
    with pytest.raises(ValueError, match="epoch seeds"):
        c.cells_below_v([], v=2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-q"]))


def test_the_corpus_root_env_names_match_the_registry():
    """`_ROOT_ENV` above must name exactly what `registry.py` honours, in the same order."""
    import re

    src = (REPO / "eval" / "scrm_eval" / "registry.py").read_text()
    m = re.search(r"^_ROOT_ENV = \(([^)]*)\)", src, re.M)
    assert m, "_ROOT_ENV not found in registry.py -- if it was renamed, this test must follow it"
    theirs = tuple(re.findall(r'"([^"]+)"', m.group(1)))
    assert theirs, "parsed no names out of registry._ROOT_ENV"
    assert theirs == _ROOT_ENV, (
        f"registry.py honours {theirs} but this file honours {_ROOT_ENV}. A corpus-root variable the "
        f"registry reads and this suite does not means the real-data tests -- the independent oracle "
        f"among them -- run against a different tree from the one under test, and pass."
    )
