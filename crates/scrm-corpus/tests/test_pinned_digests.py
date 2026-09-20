#!/usr/bin/env python3
"""Unit tests for the pinned-digest checker's comparison logic."""

import importlib.util
import json
import re
from pathlib import Path

import pyarrow as pa
import pytest
from pyarrow import ipc

_spec = importlib.util.spec_from_file_location(
    "pinned_digests", Path(__file__).with_name("pinned_digests.py")
)
pinned_digests = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pinned_digests)
compare = pinned_digests.compare

REPO = Path(__file__).resolve().parents[3]


def entry(digest="a" * 64, rows=10, tokens=100, cells=2):
    return {"stream_sha256": digest, "n_rows": rows, "n_tokens": tokens, "cells": cells}


TOKENS = pa.list_(pa.field("item", pa.uint8(), False))

def _current_rule() -> str:
    try:
        return pinned_digests.current_molecule_rule()
    except Exception:
        return "likelihood-q10-pre45-post40-omega7"


BOUNDS = {
    "alphabet_size": 7,
    "k": 19,
    "dedup_rule": "1MM_CR_flow",
    "bubble_resolution": _current_rule(),
    "min_umi_len": 60,
    "max_umi_len": 1600,
    "min_cell_molecules": 500,
}


def tiny_corpus(path: Path, payloads: dict[str, list[list[int]]]) -> Path:
    """A one-shard corpus holding `payloads` in cell order, written the way the writer writes one."""
    path.mkdir(parents=True, exist_ok=True)
    rows = [(cell, toks) for cell, molecules in payloads.items() for toks in molecules]
    table = pa.table(
        {
            "cell_id": pa.array([c for c, _ in rows]).dictionary_encode(),
            "n_reads": pa.array([1] * len(rows), type=pa.uint32()),
            "n_contigs": pa.array([1] * len(rows), type=pa.uint16()),
            "tokens": pa.array([t for _, t in rows], type=TOKENS),
        }
    )
    cell_rows, start = [], 0
    for cell, molecules in payloads.items():
        cell_rows.append({"cell": cell, "start": start, "n_rows": len(molecules)})
        start += len(molecules)
    shard = "shard-000.arrow"
    with ipc.new_file(
        pa.OSFile(str(path / shard), "wb"),
        table.schema,
        metadata={"scrm_cell_rows": json.dumps(cell_rows)},
    ) as w:
        w.write_table(table)
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "declared": {"config": dict(BOUNDS)},
                "measured": {},
                "shards": [{"file": shard, "n_rows": len(rows)}],
                "n_rows": len(rows),
                "n_tokens": sum(len(t) for _, t in rows),
                "tokens_sha256_root": "",
            }
        )
    )
    return path


def pin(tmp_path: Path, monkeypatch, *corpora: Path) -> None:
    """Pin exactly these corpora, in a fixture only this test can see."""
    fixture = tmp_path / "pinned.json"
    fixture.write_text(
        json.dumps(
            {
                "_why": ["a test fixture"],
                "trees": [
                    {
                        "bounds": dict(BOUNDS),
                        "corpora": {c.name: pinned_digests.measured_entry(c) for c in corpora},
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(pinned_digests, "PINNED", fixture)


ORIGINAL = {"AAAA": [[1, 2, 3], [4, 5]], "CCCC": [[6, 7], [1, 1, 1, 1]]}
PERMUTED = {"AAAA": [[6, 7], [1, 1, 1, 1]], "CCCC": [[1, 2, 3], [4, 5]]}


def test_the_gate_accepts_the_corpus_whose_bytes_were_pinned(tmp_path, monkeypatch):
    c = tiny_corpus(tmp_path / "a" / "ds", ORIGINAL)
    pin(tmp_path, monkeypatch, c)
    assert pinned_digests.pinned_entry(c)["n_rows"] == 4


def test_a_corpus_with_the_pinned_tallies_but_other_bytes_is_refused(tmp_path, monkeypatch):
    """The gate must discriminate on CONTENT, not on a key a permutation holds invariant."""
    original = tiny_corpus(tmp_path / "a" / "ds", ORIGINAL)
    permuted = tiny_corpus(tmp_path / "b" / "ds", PERMUTED)
    pin(tmp_path, monkeypatch, original)

    assert pinned_digests.declared(permuted) == pinned_digests.declared(original)
    assert pinned_digests.tallies(permuted) == pinned_digests.tallies(original)
    assert pinned_digests.measured_entry(permuted)["cells"] == pinned_digests.measured_entry(original)["cells"]

    with pytest.raises(pinned_digests.Failed, match="stream_sha256"):
        pinned_digests.pinned_entry(permuted)


def test_a_directory_holding_only_a_manifest_is_not_a_pinned_corpus(tmp_path, monkeypatch):
    """The gate must attest the shards an embedding will read, not the manifest describing them."""
    original = tiny_corpus(tmp_path / "a" / "ds", ORIGINAL)
    pin(tmp_path, monkeypatch, original)

    hollow = tmp_path / "hollow" / "ds"
    hollow.mkdir(parents=True)
    (hollow / "manifest.json").write_bytes((original / "manifest.json").read_bytes())

    with pytest.raises(pinned_digests.Failed, match="does not exist"):
        pinned_digests.pinned_entry(hollow)


def test_identical_sets_and_values_are_all_ok():
    m = {"one": entry(), "two": entry(digest="b" * 64)}
    ok, mismatched, extra, missing = compare(m, dict(m))
    assert ok == ["one", "two"]
    assert not mismatched and not extra and not missing


def test_a_corpus_present_but_unpinned_is_reported_not_ignored():
    """A corpus present under the root but unpinned must be reported, not ignored."""
    pinned = {"one": entry()}
    measured = {"one": entry(), "unpinned": entry(digest="c" * 64)}
    ok, mismatched, extra, missing = compare(measured, pinned)
    assert ok == ["one"]
    assert extra == ["unpinned"], "an unpinned corpus must surface, not pass silently"
    assert not missing and not mismatched


def test_a_corpus_pinned_but_absent_is_reported():
    ok, mismatched, extra, missing = compare({"one": entry()}, {"one": entry(), "gone": entry()})
    assert missing == ["gone"]
    assert ok == ["one"]
    assert not extra


def test_an_entry_short_of_a_field_is_refused_by_the_fixture_reader(tmp_path, monkeypatch):
    """A pinned entry that omits a field would be a gate on fewer fields, not a weaker pin."""
    fixture = tmp_path / "pinned.json"
    full = entry()
    for missing in sorted(pinned_digests.ENTRY_KEYS):
        short = {k: v for k, v in full.items() if k != missing}
        fixture.write_text(
            json.dumps({"_why": [], "trees": [{"bounds": dict(BOUNDS), "corpora": {"ds": short}}]})
        )
        monkeypatch.setattr(pinned_digests, "PINNED", fixture)
        with pytest.raises(pinned_digests.Failed, match="a pinned corpus is"):
            pinned_digests.load_pinned()

    fixture.write_text(
        json.dumps({"_why": [], "trees": [{"bounds": dict(BOUNDS), "corpora": {"ds": full}}]})
    )
    pinned_digests.load_pinned()


def test_the_measured_entry_is_exactly_what_a_pin_records(tmp_path):
    """The writer's shape and the schema check must be one definition, or a pin fails its own reader."""
    c = tiny_corpus(tmp_path / "a" / "ds", ORIGINAL)
    assert set(pinned_digests.measured_entry(c)) == set(pinned_digests.ENTRY_KEYS)


def test_a_field_only_the_measurement_has_is_still_compared():
    """A field present in the measurement and absent from the pinned entry must still be compared."""
    ok, mismatched, extra, missing = compare(
        {"one": entry()}, {"one": {k: v for k, v in entry().items() if k != "stream_sha256"}}
    )
    assert ok == [] and mismatched == {"one": ["stream_sha256"]}


def test_a_changed_digest_names_the_field():
    pinned = {"one": entry(digest="a" * 64)}
    measured = {"one": entry(digest="f" * 64)}
    ok, mismatched, extra, missing = compare(measured, pinned)
    assert not ok
    assert mismatched == {"one": ["stream_sha256"]}


@pytest.mark.parametrize("field", ["n_rows", "n_tokens", "cells"])
def test_every_tallied_field_is_compared_not_just_the_digest(field):
    """A digest match with a wrong row count would mean the tallies are decorative."""
    pinned = {"one": entry()}
    measured = {"one": dict(entry(), **{field: 999})}
    _, mismatched, _, _ = compare(measured, pinned)
    assert mismatched == {"one": [field]}


def test_several_problems_at_once_are_all_reported():
    """Reporting only the first would make a second run necessary to see the second problem."""
    pinned = {"same": entry(), "changed": entry(digest="a" * 64), "gone": entry()}
    measured = {"same": entry(), "changed": entry(digest="d" * 64), "new": entry()}
    ok, mismatched, extra, missing = compare(measured, pinned)
    assert ok == ["same"]
    assert list(mismatched) == ["changed"]
    assert extra == ["new"]
    assert missing == ["gone"]


def test_an_empty_tree_against_a_populated_fixture_reports_everything_missing():
    _, mismatched, extra, missing = compare({}, {"one": entry(), "two": entry()})
    assert missing == ["one", "two"]
    assert not extra and not mismatched


def test_molecule_keys_match_the_rust_compat_key():
    """`MOLECULE_KEYS` must name every field of `OwnedCompatKey`, in both directions."""
    src = (REPO / "crates" / "scrm-pack" / "src" / "reader.rs").read_text()
    body = re.search(r"pub struct OwnedCompatKey \{(.*?)\n\}", src, re.S)
    assert body, "OwnedCompatKey not found -- if it was renamed, this test must follow it"
    rust = tuple(re.findall(r"pub (\w+):", body.group(1)))
    assert rust, "parsed no fields out of OwnedCompatKey"
    assert set(rust) == set(pinned_digests.MOLECULE_KEYS), (
        f"the fixture pins {sorted(pinned_digests.MOLECULE_KEYS)} but the Rust compat key is "
        f"{sorted(rust)}. A field the loader compares and the fixture does not is a corpus that can "
        f"be pinned and then refused; a field the fixture pins and the loader ignores refuses "
        f"rebuilds that are compatible. Keep them equal."
    )


RETIRED_BOUNDS = dict(BOUNDS, bubble_resolution="weighted_split")


def _tree_declaring(path: Path, bounds: dict, payloads: dict) -> Path:
    """`tiny_corpus` but with the manifest's declared config overridden."""
    c = tiny_corpus(path, payloads)
    man = json.loads((c / "manifest.json").read_text())
    man["declared"]["config"] = dict(bounds)
    (c / "manifest.json").write_text(json.dumps(man))
    return c


def test_write_refuses_a_tree_the_current_code_could_not_have_built(tmp_path, monkeypatch):
    """`--write` must refuse a tree whose declared rule the current code could not have produced."""
    c = _tree_declaring(tmp_path / "root" / "ds", RETIRED_BOUNDS, ORIGINAL)
    monkeypatch.setattr(pinned_digests, "PINNED", tmp_path / "pinned.json")
    (tmp_path / "pinned.json").write_text(json.dumps({"_why": ["a test fixture"], "trees": []}))
    monkeypatch.setattr(pinned_digests, "current_molecule_rule",
                        lambda: BOUNDS["bubble_resolution"])
    with pytest.raises(pinned_digests.Failed) as e:
        pinned_digests.write(c.parent)
    msg = str(e.value)
    assert "weighted_split" in msg and BOUNDS["bubble_resolution"] in msg, (
        "the refusal must name BOTH definitions; a message that names neither sends the operator to "
        "diff two manifests by hand"
    )
    assert "Refusing to pin" in msg


def test_write_accepts_a_tree_this_code_did_build(tmp_path, monkeypatch):
    """The gate must not be a blanket refusal -- it has to pass the tree it is meant to pass."""
    c = tiny_corpus(tmp_path / "root" / "ds", ORIGINAL)
    monkeypatch.setattr(pinned_digests, "PINNED", tmp_path / "pinned.json")
    (tmp_path / "pinned.json").write_text(json.dumps({"_why": ["a test fixture"], "trees": []}))
    monkeypatch.setattr(pinned_digests, "current_molecule_rule",
                        lambda: BOUNDS["bubble_resolution"])
    assert pinned_digests.write(c.parent) == 0
    back = json.loads((tmp_path / "pinned.json").read_text())
    assert [t["bounds"]["bubble_resolution"] for t in back["trees"]] == [BOUNDS["bubble_resolution"]]


def test_a_stale_extension_RAISES_rather_than_skipping_the_check():
    """`current_molecule_rule` must never degrade to "cannot tell"."""
    import sys

    sys.path.insert(0, str(Path(pinned_digests.__file__).resolve().parents[3] / "target" / "py"))
    import scrm_py

    if hasattr(scrm_py, "resolution_name"):
        rule = pinned_digests.current_molecule_rule()
        assert rule == BOUNDS["bubble_resolution"], (
            f"the code produces {rule!r} but this test's BOUNDS say "
            f"{BOUNDS['bubble_resolution']!r} -- one of them is stale"
        )
        return
    with pytest.raises(pinned_digests.Failed) as e:
        pinned_digests.current_molecule_rule()
    assert "cargo build --release -p scrm-py" in str(e.value), (
        "the error must name the command that fixes it"
    )
