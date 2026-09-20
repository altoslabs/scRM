#!/usr/bin/env python3
"""Verify a corpus from Python, independently of the Rust reader."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
from pyarrow import ipc

EXPECTED = pa.schema(
    [
        pa.field("cell_id", pa.dictionary(pa.int32(), pa.string()), nullable=False),
        pa.field("n_reads", pa.uint32(), nullable=False),
        pa.field("n_contigs", pa.uint16(), nullable=False),
        pa.field("tokens", pa.list_(pa.field("item", pa.uint8(), nullable=False)), nullable=False),
    ]
)

ALPHABET_SIZE = 7
TAB, NEWLINE = 5, 6


class Failed(Exception):
    pass


def check(condition, message):
    """Raises rather than `assert`, so `python -O` cannot turn this file into a no-op."""
    if not condition:
        raise Failed(message)


def check_tokens(where, offsets, values, n_contigs):
    """Row-level invariants for one batch, vectorised."""
    lengths = np.diff(offsets)
    check(np.all(lengths > 0), f"{where}: an empty token row")

    span = values[offsets[0] : offsets[-1]]
    check(int(span.max(initial=0)) < ALPHABET_SIZE, f"{where}: token outside the alphabet")
    check(not np.any(span == NEWLINE), f"{where}: Token::Newline is never stored")

    check(not np.any(values[offsets[:-1]] == TAB), f"{where}: leading separator")
    check(not np.any(values[offsets[1:] - 1] == TAB), f"{where}: trailing separator")

    tabs = span == TAB
    doubled = tabs[:-1] & tabs[1:]
    boundaries = offsets[1:-1] - offsets[0] - 1
    if boundaries.size:
        doubled[boundaries[boundaries < doubled.size]] = False
    check(not np.any(doubled), f"{where}: doubled separator")

    tab_cumsum = np.concatenate(([0], np.cumsum(tabs)))
    per_row = np.diff(tab_cumsum[offsets - offsets[0]]) + 1
    check(np.array_equal(per_row, n_contigs), f"{where}: n_contigs disagrees with the recount")


def compat_key_line(cfg):
    """`Stamp::compat_key()` in full, printed so a mismatch is visible rather than inferred."""
    lo = cfg.get("min_umi_len", "ABSENT")
    hi = cfg.get("max_umi_len", "ABSENT")
    return (
        f"  compat_key     alphabet {cfg['alphabet_size']} k {cfg['k']} umi_len {cfg['umi_len']} "
        f"{cfg['dedup_rule']} {cfg['bubble_resolution']} umi_tokens [{lo}, {hi}]"
    )


def cell_runs(dict_array):
    """Contiguous `(cell_id, length)` runs in one batch's dictionary column."""
    idx = np.asarray(dict_array.indices)
    check(len(idx) > 0, "empty batch")
    values = dict_array.dictionary
    change = np.flatnonzero(idx[1:] != idx[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(idx)]))
    return [(values[int(idx[a])].as_py(), int(b - a)) for a, b in zip(bounds[:-1], bounds[1:])]


def verify_shard(path: Path):
    """One shard, streamed. Returns (n_rows, n_tokens, sha256, first_cell, last_cell,
    declared, n_batches, per-row-token-digest-updater already applied).
    """
    with pa.memory_map(str(path), "rb") as source:
        reader = ipc.open_file(source)
        check(reader.schema.equals(EXPECTED), f"{path.name}: schema mismatch:\n{reader.schema}\nvs\n{EXPECTED}")
        declared = json.loads(reader.schema.metadata[b"scrm_declared"])
        footer = reader.metadata

        hasher = hashlib.sha256()
        n_rows = 0
        n_tokens = 0
        runs = []

        for b in range(reader.num_record_batches):
            batch = reader.get_batch(b)
            where = f"{path.name} batch {b}"
            check(batch.num_rows > 0, f"{where}: empty batch")
            for col in batch.columns:
                check(col.null_count == 0, f"{where}: nulls in a non-nullable column")

            tokens = batch.column("tokens")
            offsets = np.asarray(tokens.offsets)
            values = np.asarray(tokens.values)
            n_contigs = np.asarray(batch.column("n_contigs"))
            check_tokens(where, offsets, values, n_contigs)

            hasher.update(memoryview(values[offsets[0] : offsets[-1]]))
            n_tokens += int(offsets[-1] - offsets[0])
            n_rows += batch.num_rows

            for cell, length in cell_runs(batch.column("cell_id")):
                if runs and runs[-1][0] == cell:
                    runs[-1][1] += length
                else:
                    runs.append([cell, length])

        n_batches = reader.num_record_batches

    names = [c for c, _ in runs]
    check(len(set(names)) == len(names), f"{path.name}: a cell_id appears in more than one run")
    check(names == sorted(names), f"{path.name}: cell_id is not sorted")

    check(int(footer[b"scrm_n_rows"]) == n_rows, f"{path.name}: recorded row count disagrees")
    check(int(footer[b"scrm_n_tokens"]) == n_tokens, f"{path.name}: recorded token count disagrees")
    sha = hasher.hexdigest()
    recorded = footer[b"scrm_tokens_sha256"].decode()
    check(sha == recorded, f"{path.name}: token sha256 {sha} disagrees with recorded {recorded}")
    index = json.loads(footer[b"scrm_cell_rows"])
    check(
        [(r["cell"], r["len"]) for r in index] == [(c, n) for c, n in runs],
        f"{path.name}: cell index disagrees with the data",
    )

    return n_rows, n_tokens, sha, names[0], names[-1], declared, n_batches


def root_digest(shards):
    """Hand-reimplementation of `manifest.rs::root_digest`."""
    h = hashlib.sha256()
    h.update(len(shards).to_bytes(8, "little"))
    for s in shards:
        for field in (s["file"], s["tokens_sha256"], s["first_cell"], s["last_cell"]):
            raw = field.encode()
            h.update(len(raw).to_bytes(8, "little"))
            h.update(raw)
        h.update(s["n_rows"].to_bytes(8, "little"))
        h.update(s["n_tokens"].to_bytes(8, "little"))
    return h.hexdigest()


def verify_corpus(root: Path) -> int:
    manifest = json.loads((root / "manifest.json").read_text())
    shards = manifest["shards"]
    check(bool(shards), "manifest lists no shards")

    total_rows = total_tokens = total_batches = 0
    prev_last = None
    stream = hashlib.sha256()

    for k, entry in enumerate(shards):
        path = root / entry["file"]
        rows, ntok, sha, first, last, _declared, nb = verify_shard(path)

        check(rows == entry["n_rows"], f"shard {k}: manifest says {entry['n_rows']} rows, file has {rows}")
        check(ntok == entry["n_tokens"], f"shard {k}: manifest token count disagrees")
        check(sha == entry["tokens_sha256"], f"shard {k}: manifest digest disagrees with the data")
        check(
            (first, last) == (entry["first_cell"], entry["last_cell"]),
            f"shard {k}: manifest span {entry['first_cell']}..{entry['last_cell']} != file {first}..{last}",
        )
        check(
            prev_last is None or prev_last < first,
            f"shard {k}: starts at {first} but shard {k - 1} ended at {prev_last}",
        )
        prev_last = last

        with pa.memory_map(str(path), "rb") as source:
            r = ipc.open_file(source)
            for b in range(r.num_record_batches):
                t = r.get_batch(b).column("tokens")
                offs = np.asarray(t.offsets)
                stream.update(memoryview(np.asarray(t.values)[offs[0] : offs[-1]]))

        total_rows += rows
        total_tokens += ntok
        total_batches += nb

    check(total_rows == manifest["n_rows"], "manifest row total disagrees with the shards")
    check(total_tokens == manifest["n_tokens"], "manifest token total disagrees with the shards")

    root_hex = root_digest(shards)
    check(
        root_hex == manifest["tokens_sha256_root"],
        f"root recomputed as {root_hex}, manifest records {manifest['tokens_sha256_root']}",
    )

    cfg = manifest["declared"]["config"]
    counts = manifest["measured"]["counts"]
    umi = manifest["measured"]["umi_stats"]
    print(f"pyarrow {pa.__version__}  {len(shards)} shards  {total_batches} batches  "
          f"{total_rows} rows  {total_tokens} tokens")
    print(compat_key_line(cfg))
    print(f"  sha256 root    {root_hex}   (recomputed, matches manifest)")
    print(f"  sha256 stream  {stream.hexdigest()}")
    print(f"  cells          {shards[0]['first_cell']}..{shards[-1]['last_cell']}")
    print(f"  molecules      {counts['molecules_out']}  pairs_in {counts['pairs_in']}")
    print(f"  umi guard      mean H1 degree {umi['mean_hamming1_degree']:.4f}  "
          f"max group {umi['max_group_fraction'] * 100:.4f}%  chain depth {umi['max_chain_depth']}")
    print("OK: memory_map + ipc.open_file, schema, declared stamp, per-shard tallies + sha256,")
    print("    cross-shard ordering, manifest root, every row's alphabet/separators/n_contigs")
    return 0


def verify_single(path: Path) -> int:
    rows, ntok, sha, first, last, declared, nb = verify_shard(path)
    cfg = declared["config"]
    print(f"pyarrow {pa.__version__} read {rows} rows in {nb} batches")
    print(compat_key_line(cfg))
    print(f"  tokens         {ntok}")
    print(f"  sha256         {sha}   (recomputed, matches footer)")
    print(f"  cells          {first}..{last}")
    print("OK: memory_map + ipc.open_file, schema, declared stamp, footer tallies + sha256,")
    print("    sort order, contiguity, alphabet, separators, n_contigs")
    return 0


def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    target = Path(argv[1])
    return verify_corpus(target) if target.is_dir() else verify_single(target)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Failed as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
