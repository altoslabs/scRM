"""The reuse contract: what a cached score's identity is, and what must invalidate it."""

from __future__ import annotations

import ast
import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scrm_eval import artifact, controls, scoring  # noqa: E402

A_MODEL_PRODUCER = sorted(artifact.MODEL_PRODUCERS)[0]
A_COUNT_PRODUCER = controls.PRODUCERS[0]

PKG = Path(scoring.__file__).parent


def _stamp(**over) -> SimpleNamespace:
    """A stand-in stamp: `reuse_key` reads only `REUSE_FIELDS` off it, plus `notes`."""
    base = {"producer": A_MODEL_PRODUCER, "checkpoint_sha256": "c" * 64, "arch_sha": "a" * 16,
            "cell_order_sha256": "o" * 64, "forward_precision": "bf16+eager", "w_eval": 512,
            "n_cells": 1000, "dim": 128, "readout": {"pool": "segment_attention"}, "notes": None}
    return SimpleNamespace(**(base | over))


def _emb(**over):
    return SimpleNamespace(stamp=_stamp(**over))


def _different(value):
    """Some other value of the same shape, so a field's contribution can be tested generically."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, str):
        return value + "-changed"
    if isinstance(value, dict):
        return value | {"changed": True}
    return "changed"


KEY = dict(reduction_rule="mean_standardised", context={"population": "p" * 32})


MUST_INVALIDATE = {
    "producer": "a count baseline's row and a byte-level model's row are not interchangeable",
    "checkpoint_sha256": "a re-trained arm at the same path is a different computation wearing the "
                         "same label -- the shape that publishes 0.9167 for a 0.3068 embedding",
    "arch_sha": "the same weights read through a different architecture are a different tensor",
    "cell_order_sha256": "hashed in ARTIFACT order, so two differently-ordered embeddings cannot be "
                         "called equal and let a report compare row i against row j",
    "forward_precision": "bf16+compiled and fp32+eager embeddings of one cell by one checkpoint are "
                         "not interchangeable",
    "w_eval": "a capped row and an all-reads row describe two different coverages of the same cell",
    "n_cells": "a salvaged prefix and a finished artifact are not the same population",
    "dim": "a different width is a different readout, whatever the label says",
    "readout": "an embedding that does not name its readout is what makes eleven aggregator "
               "comparisons unfalsifiable",
}


@pytest.mark.parametrize("field", sorted(MUST_INVALIDATE))
def test_changing_any_part_of_an_artifacts_identity_changes_the_key(field):
    """Changing any field of an artifact's identity must change its reuse key."""
    base = _emb()
    changed = _emb(**{field: _different(getattr(base.stamp, field))})
    assert scoring.reuse_key(base, **KEY) != scoring.reuse_key(changed, **KEY), (
        f"a change to {field} does not change the reuse key, so a stale row would be carried "
        f"forward: {MUST_INVALIDATE[field]}"
    )


def test_every_reuse_field_is_a_real_stamp_field():
    """Every name in `REUSE_FIELDS` must be a real stamp field."""
    real = {f.name for f in dataclasses.fields(artifact.Stamp)}
    assert set(scoring.REUSE_FIELDS) <= real, sorted(set(scoring.REUSE_FIELDS) - real)


def test_the_notes_are_in_the_key_because_the_population_floor_acts_on_them():
    """`notes` must be in the reuse key: `notes.salvaged` lowers the population floor from 95% to
    85%, so it changes what a row means with every hash on the stamp identical.
    """
    assert scoring.reuse_key(_emb(notes=None), **KEY) != scoring.reuse_key(
        _emb(notes={"salvaged": True, "cells_reached": 7140}), **KEY)


def test_the_reduction_rule_and_the_context_are_both_in_the_key():
    """The reduction rule and the scoring context must both be in the reuse key: neither scoring
    surface computes a function of one artifact alone.
    """
    base = _emb()
    assert scoring.reuse_key(base, reduction_rule="mean", context=KEY["context"]) != \
        scoring.reuse_key(base, reduction_rule="mean_standardised", context=KEY["context"])
    assert scoring.reuse_key(base, reduction_rule="mean", context={"population": "a"}) != \
        scoring.reuse_key(base, reduction_rule="mean", context={"population": "b"})


def _package_imports(name: str) -> set[str]:
    """Sibling modules of this package that `name` imports, at any nesting depth in its source."""
    tree = ast.parse((PKG / name).read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            if node.module:
                found.add(f"{node.module}.py")
            else:
                found.update(f"{a.name}.py" for a in node.names)
    return {f for f in found if (PKG / f).exists()}


def test_scoring_version_hashes_every_module_the_scorers_import():
    """`scoring_version()` must hash the whole import closure of the two scoring entrypoints."""
    seen: set[str] = set()
    frontier = {"run_probes.py", "run_annotation.py"}
    while frontier:
        mod = frontier.pop()
        if mod in seen:
            continue
        seen.add(mod)
        frontier |= _package_imports(mod)
    missing = sorted(seen - set(scoring.SCORING_MODULES))
    assert not missing, (
        f"these modules can change a score but are not hashed by scoring_version(): {missing}. "
        f"Add them to SCORING_MODULES."
    )


def _with_modules(mods, monkeypatch):
    monkeypatch.setattr(scoring, "SCORING_MODULES", mods)
    return scoring.scoring_version()


def test_a_module_that_cannot_be_hashed_raises_instead_of_being_skipped(monkeypatch):
    """A module `scoring_version()` cannot read must raise rather than drop out of the digest."""
    with pytest.raises(FileNotFoundError, match="SCORING_MODULES"):
        _with_modules(("probes.py", "no_such_module.py"), monkeypatch)


def test_the_scoring_version_reads_file_content_not_module_names(monkeypatch):
    """`scoring_version()` must hash file content: two different modules must not hash alike."""
    assert scoring.scoring_version() == scoring.scoring_version()
    assert _with_modules(("probes.py",), monkeypatch) != _with_modules(("metrics.py",), monkeypatch)


def test_probes_and_reduce_are_hashed():
    """`probes.py` and `reduce.py` must both be covered by `scoring_version()`."""
    for mod in ("probes.py", "reduce.py"):
        assert mod in scoring.SCORING_MODULES, mod


def test_a_mixed_forward_arithmetic_is_refused_and_a_baseline_never_blocks():
    """A mixed forward arithmetic must be refused by one function shared by both scoring surfaces,
    and a baseline must never block.
    """
    old = _emb(forward_precision=None)
    new = _emb(forward_precision="bf16+compiled")
    with pytest.raises(artifact.StampError, match="different forward arithmetic"):
        scoring.refuse_mixed_precision({"old": old, "new": new})

    base = _emb(producer=A_COUNT_PRODUCER, forward_precision=None)
    prec = scoring.refuse_mixed_precision({"a": new, "b": _emb(forward_precision="bf16+compiled"),
                                           "c": base})
    assert prec == {"a": "bf16+compiled", "b": "bf16+compiled"}, "a baseline is not a model artifact"

    assert scoring.refuse_mixed_precision({"legacy": old}) == {"legacy": "fp32+eager"}
