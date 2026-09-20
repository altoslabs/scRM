#!/usr/bin/env python3
"""Check a corpus tree against the stream digests pinned in `expected_stream_digests.json`."""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
from pyarrow import ipc

PINNED = Path(__file__).with_name("expected_stream_digests.json")
MOLECULE_KEYS = (
    "alphabet_size", "k", "dedup_rule", "bubble_resolution",
    "min_umi_len", "max_umi_len", "min_cell_molecules",
)


ENTRY_KEYS = ("stream_sha256", "n_rows", "n_tokens", "cells")


class Failed(Exception):
    pass


def check(condition, message):
    if not condition:
        raise Failed(message)


def stream_digest(corpus: Path):
    """sha256 over every row's tokens in cell order, plus the tallies, streamed."""
    mpath = corpus / "manifest.json"
    check(mpath.exists(), f"{corpus} has no manifest.json -- incomplete or not a corpus")
    manifest = json.loads(mpath.read_text())

    h = hashlib.sha256()
    rows = tokens = 0
    cells = set()
    for entry in manifest["shards"]:
        path = corpus / entry["file"]
        check(path.exists(), f"manifest names {entry['file']} but it does not exist")
        with pa.memory_map(str(path), "rb") as src:
            reader = ipc.open_file(src)
            for b in range(reader.num_record_batches):
                batch = reader.get_batch(b)
                tok = batch.column("tokens")
                offs = np.asarray(tok.offsets)
                vals = np.asarray(tok.values)
                h.update(memoryview(vals[offs[0] : offs[-1]]))
                tokens += int(offs[-1] - offs[0])
                rows += batch.num_rows
        with pa.memory_map(str(path), "rb") as src:
            footer = ipc.open_file(src).metadata
            cells.update(r["cell"] for r in json.loads(footer[b"scrm_cell_rows"]))
    return h.hexdigest(), rows, tokens, len(cells)


def _cpus() -> int:
    """Cores this PROCESS may use, not cores the machine has."""
    n = os.environ.get("SLURM_CPUS_PER_TASK")
    if n and n.isdigit():
        return max(1, int(n))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def corpora_on_disk(root: Path) -> list[Path]:
    check(root.is_dir(), f"{root} is not a directory")
    found = sorted(p for p in root.iterdir() if (p / "manifest.json").exists())
    check(bool(found), f"no corpus with a manifest.json under {root}")
    return found


_MEASURED: dict[tuple[str, str], dict] = {}


def measured_entry(corpus: Path) -> dict:
    """The four pinned fields, measured from the corpus's own bytes."""
    mpath = corpus / "manifest.json"
    check(mpath.exists(), f"{corpus} has no manifest.json -- incomplete or not a corpus")
    key = (str(corpus.resolve()), hashlib.sha256(mpath.read_bytes()).hexdigest())
    if key not in _MEASURED:
        digest, rows, tokens, cells = stream_digest(corpus)
        _MEASURED[key] = {"stream_sha256": digest, "n_rows": rows, "n_tokens": tokens, "cells": cells}
    return _MEASURED[key]


def _one(corpus: Path) -> tuple[str, dict, dict]:
    cfg = json.loads((corpus / "manifest.json").read_text())["declared"]["config"]
    bounds = {k: cfg.get(k) for k in MOLECULE_KEYS}
    return corpus.name, measured_entry(corpus), bounds


def measure(root: Path, jobs: int | None = None):
    """Measure every corpus present, returning `(entries, bounds)`."""
    corpora = corpora_on_disk(root)
    jobs = jobs or min(len(corpora), _cpus())
    entries, bounds, first = {}, None, None
    if jobs <= 1:
        results = [_one(c) for c in corpora]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(_one, corpora))
    for name, entry, got in results:
        check(
            bounds is None or got == bounds,
            f"{name} declares bounds {got}, {first} declares {bounds} -- "
            "these corpora cannot be mixed, so they cannot be pinned together",
        )
        bounds, first = got, first or name
        entries[name] = entry
    return entries, bounds


def compare(measured: dict, pinned: dict):
    """Pure comparison: `(ok, mismatched, extra, missing)`, where `mismatched` names the bad fields."""
    extra = sorted(set(measured) - set(pinned))
    missing = sorted(set(pinned) - set(measured))
    mismatched = {}
    ok = []
    for name in sorted(set(measured) & set(pinned)):
        got, exp = measured[name], pinned[name]
        bad = [k for k in sorted(set(exp) | set(got)) if got.get(k) != exp.get(k)]
        (mismatched.setdefault(name, bad) if bad else ok.append(name))
    return ok, mismatched, extra, missing


def load_pinned() -> dict:
    """The fixture, schema-checked. `{"_why": [...], "trees": [{"bounds": {...}, "corpora": {...}}]}`."""
    d = json.loads(PINNED.read_text())
    check(set(d) == {"_why", "trees"}, f"{PINNED.name}: expected keys _why and trees, found {sorted(d)}")
    seen = []
    for i, t in enumerate(d["trees"]):
        check(set(t) == {"bounds", "corpora"}, f"{PINNED.name}: tree {i} has keys {sorted(t)}")
        check(
            set(t["bounds"]) == set(MOLECULE_KEYS),
            f"{PINNED.name}: tree {i} pins bounds {sorted(t['bounds'])}, "
            f"the molecule definition is {sorted(MOLECULE_KEYS)}",
        )
        for name, entry in t["corpora"].items():
            check(
                set(entry) == set(ENTRY_KEYS),
                f"{PINNED.name}: tree {i} entry {name} records {sorted(entry)}, "
                f"a pinned corpus is {sorted(ENTRY_KEYS)}",
            )
        key = _bounds_key(t["bounds"])
        check(key not in seen, f"{PINNED.name}: two trees share the molecule definition {t['bounds']}")
        seen.append(key)
    return d


def _bounds_key(bounds: dict):
    return tuple(bounds[k] for k in MOLECULE_KEYS)


def tree_for(bounds: dict) -> dict | None:
    """The corpora pinned for this molecule definition, or `None` if it has never been pinned."""
    want = _bounds_key(bounds)
    for t in load_pinned()["trees"]:
        if _bounds_key(t["bounds"]) == want:
            return t["corpora"]
    return None


def known_definitions() -> list[dict]:
    return [t["bounds"] for t in load_pinned()["trees"]]


def declared(corpus: Path) -> dict:
    """A corpus's own molecule definition, from its manifest."""
    mpath = corpus / "manifest.json"
    check(mpath.exists(), f"{corpus} has no manifest.json -- incomplete or not a corpus")
    cfg = json.loads(mpath.read_text())["declared"]["config"]
    missing = [k for k in MOLECULE_KEYS if cfg.get(k) is None]
    check(not missing, f"{corpus}/manifest.json declares no {missing}")
    return {k: cfg[k] for k in MOLECULE_KEYS}


def tallies(corpus: Path) -> dict:
    """`n_rows` and `n_tokens` as the BUILDER recorded them, for an O(1) content check."""
    man = json.loads((corpus / "manifest.json").read_text())
    return {"n_rows": man["n_rows"], "n_tokens": man["n_tokens"]}


def pinned_entry(corpus: Path, dataset: str | None = None) -> dict:
    """The pinned entry for a corpus on disk, or `Failed`."""
    dataset = dataset or corpus.name
    bounds = declared(corpus)
    corpora = tree_for(bounds)
    check(
        corpora is not None,
        f"{corpus} declares the molecule definition {bounds}, which is not pinned. "
        f"Pinned definitions: {known_definitions()}. Pin it first "
        f"(pinned_digests.py <root> --write) -- an unpinned corpus cannot be stamped.",
    )
    check(
        dataset in corpora,
        f"{dataset} is not pinned for its own molecule definition "
        f"({bounds['bubble_resolution']}); {len(corpora)} datasets are. "
        "Pin the tree before embedding from it.",
    )
    entry, got = corpora[dataset], tallies(corpus)
    bad = {k: (v, entry[k]) for k, v in got.items() if entry.get(k) != v}
    check(
        not bad,
        f"{corpus} records {got} but the pin for {dataset} says "
        f"{ {k: entry[k] for k in got} } -- differs on { {k: v for k, v in bad.items()} }. "
        "Same molecule definition, different content: diagnose it, do not re-pin.",
    )

    measured = measured_entry(corpus)
    bad = {
        k: {"measured": measured.get(k), "pinned": entry.get(k)}
        for k in sorted(set(entry) | set(measured))
        if measured.get(k) != entry.get(k)
    }
    check(
        not bad,
        f"{corpus} is not the content pinned for {dataset}: differs on {bad}. The molecule "
        "definition and the manifest's tallies match, so this is the same rule applied to different "
        "bytes -- a cross-cell permutation holds n_rows and n_tokens invariant by construction, and "
        "so does a re-run of one under a name already pinned. Diagnose it; do NOT re-pin. (If the "
        "assembler's contig EMISSION ORDER moved, every digest moves while no molecule does, because "
        "contig order is inside the digest and semantically empty: re-pin, and say that in the "
        "commit.)",
    )
    return entry


def verify(root: Path) -> int:
    measured, got_bounds = measure(root)
    want = tree_for(got_bounds)

    if want is None:
        print(f"FAILED: {root} declares the molecule definition {got_bounds},")
        print("        which is not pinned. Pinned definitions:")
        for b in known_definitions():
            print(f"          {b}")
        print("        Pin it (--write) and say in the commit which part of the definition moved.")
        return 1
    print(f"molecule definition: {got_bounds}")

    ok, mismatched, extra, missing = compare(measured, want)
    for name in ok:
        got = measured[name]
        print(
            f"  OK   {name}: {got['n_rows']:,} rows / {got['n_tokens']:,} tokens / "
            f"{got['cells']:,} cells  {got['stream_sha256'][:16]}…"
        )
    for name, bad in mismatched.items():
        print(f"  FAIL {name}:")
        for k in bad:
            print(f"         {k}: expected {want[name][k]}, got {measured[name].get(k)}")
        if "stream_sha256" in bad:
            print("         The data changed. Diagnose the pipeline before re-pinning.")
            print("         Contig emission order is inside the digest, so a change to it moves")
            print("         every digest while no molecule moves: re-pin and say so in the commit.")
    print()
    for name in extra:
        print(f"  FAIL {name}: present under {root} but not pinned -- unverified content")
    for name in missing:
        print(f"  FAIL {name}: pinned but not present under {root}")

    if mismatched or extra or missing:
        print(f"FAILED: {len(mismatched) + len(extra) + len(missing)} problem(s) above")
        return 1
    print(f"OK: all {len(measured)} corpora match the digests pinned in {PINNED.name}")
    print("    (invariant to sharding, to filenames, and to every column except `tokens`)")
    return 0


def report(root: Path) -> int:
    """Read-only: what would change if this tree replaced the pinned one?"""
    measured, bounds = measure(root)
    pinned = tree_for(bounds) or {}
    ok, mismatched, extra, missing = compare(measured, pinned)
    marks = [(n, "unchanged") for n in ok] + [(n, "new") for n in extra]
    marks += [(n, "CHANGED") for n in mismatched]
    for name, mark in sorted(marks):
        got = measured[name]
        print(f"  {mark:9} {name}: {got['n_rows']:,} rows / {got['cells']:,} cells  {got['stream_sha256'][:16]}…")
        if mark == "CHANGED":
            was = pinned[name]
            print(f"            pinned {was['stream_sha256'][:16]}…  rows {was['n_rows']:,}  fields {mismatched[name]}")
    for name in missing:
        print(f"  absent    {name}: pinned, not in this tree")

    print()
    print(f"bounds {bounds}")
    if mismatched:
        print(f"FAILED: {len(mismatched)} corpus/corpora changed content: {sorted(mismatched)}")
        print("        a rebuild that alters molecules is a finding, not a re-pin")
        return 1
    print(f"OK: nothing that was pinned changed content ({len(ok)} unchanged, {len(extra)} new)")
    return 0


def current_molecule_rule() -> str:
    """The consensus rule the current code produces, asked of that code."""
    import sys

    tp = str(Path(__file__).resolve().parents[3] / "target" / "py")
    if tp not in sys.path:
        sys.path.insert(0, tp)
    import scrm_py

    fn = getattr(scrm_py, "resolution_name", None)
    check(
        fn is not None,
        "target/py/scrm_py.so predates scrm_py.resolution_name(), so the molecule definition the "
        "current code produces cannot be established, and a tree must not be pinned without it. "
        "Rebuild the extension:\n"
        "    cargo build --release -p scrm-py && cp target/release/libscrm_py.so target/py/scrm_py.so",
    )
    return fn()


def write(root: Path) -> int:
    was = load_pinned()
    measured, bounds = measure(root)
    current = current_molecule_rule()
    check(
        bounds["bubble_resolution"] == current,
        f"{root} declares bubble_resolution {bounds['bubble_resolution']!r} but the code in this "
        f"tree produces {current!r}. These are different molecule definitions: the corpus was built "
        "by a different consensus rule, or by different consensus constants, which the name "
        "carries. Refusing to pin it, because pinning would make it embeddable against artifacts "
        "of another definition. Rebuild the corpus with this tree's code, or check out the code "
        "that built it.",
    )
    key = _bounds_key(bounds)
    trees = [t for t in was["trees"] if _bounds_key(t["bounds"]) != key]
    prior = next((t for t in was["trees"] if _bounds_key(t["bounds"]) == key), None)
    if prior is None:
        print(f"  NEW molecule definition: {bounds}")
        print(f"  {len(trees)} existing tree(s) carried through untouched")
    for name, got in measured.items():
        before = (prior or {}).get("corpora", {}).get(name, {}).get("stream_sha256")
        mark = "new" if before is None else ("unchanged" if before == got["stream_sha256"] else "CHANGED")
        print(
            f"  {name}: {got['stream_sha256'][:16]}…  {got['n_rows']:,} rows  "
            f"{got['cells']:,} cells  [{mark}]"
        )
    for name in sorted(set((prior or {}).get("corpora", {})) - set(measured)):
        print(f"  {name}: DROPPED -- was pinned for this definition, is not under {root}")

    out = {"_why": was["_why"], "trees": trees + [{"bounds": bounds, "corpora": measured}]}
    tmp = PINNED.with_name(PINNED.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=2) + "\n")
    os.replace(tmp, PINNED)

    back = load_pinned()
    check(back == out, f"{PINNED.name} does not read back as written -- serialisation bug")
    print(f"  pinned {len(measured)} corpora at bounds {bounds}, and it reads back identical")
    print(f"  the file now holds {len(back['trees'])} molecule definition(s)")
    print("  say in the commit message WHY these values moved")
    return 0


def main(argv) -> int:
    args = [a for a in argv[1:] if not a.startswith("-")]
    flags = {a for a in argv[1:] if a.startswith("-")}
    if len(args) != 1 or flags - {"--write", "--report"} or flags == {"--write", "--report"}:
        print(__doc__)
        return 2
    root = Path(args[0])
    if "--write" in flags:
        return write(root)
    if "--report" in flags:
        return report(root)
    return verify(root)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Failed as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
