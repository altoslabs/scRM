"""Facts a shipped corpus records but nothing acts on."""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pytest

_ROOT_ENV = ("SCRM_CORPUS_ROOT", "CORPORA_ROOT", "CORPUS_ROOT")
_NO_ROOT = f"no corpus root: set one of {', '.join(_ROOT_ENV)} or SCRM_PROJ"


def corpus_root() -> Path | None:
    """The corpus tree, or None when no environment variable names one."""
    env = next((v for k in _ROOT_ENV if (v := os.environ.get(k))), None)
    if env:
        return Path(env)
    proj = os.environ.get("SCRM_PROJ")
    return Path(proj) / "corpus" if proj else None


ROOT = corpus_root()
MANIFESTS = sorted(glob.glob(str(ROOT / "*" / "manifest.json"))) if ROOT else []

PERCOLATION_DEGREE = 1.0

pytestmark = pytest.mark.skipif(
    not MANIFESTS,
    reason=f"no corpora with a manifest.json under {ROOT}" if ROOT else _NO_ROOT,
)


def _load():
    out = {}
    for m in MANIFESTS:
        out[m.split("/")[-2]] = json.load(open(m))
    return out


def test_percolating_corpora_are_named_not_merely_warned_about():
    """Corpora percolating above the threshold must be named, not merely warned about."""
    over = []
    for ds, d in _load().items():
        deg = d["measured"]["umi_stats"].get("mean_hamming1_degree")
        if deg is not None and deg >= PERCOLATION_DEGREE:
            over.append((ds, deg))
    print("\n  corpora at or above the percolation threshold (chimeric-contig risk):")
    for ds, deg in sorted(over, key=lambda x: -x[1]):
        print(f"    {ds:44s} mean_hamming1_degree {deg:.3f}")
    if not over:
        print("    none")
    missing = [ds for ds, d in _load().items()
               if d["measured"]["umi_stats"].get("mean_hamming1_degree") is None]
    assert not missing, f"corpora with no percolation measurement at all: {missing}"


def test_shallow_cell_loss_is_recorded_for_every_corpus():
    """Every corpus must record the fraction of called cells lost to `min_cell_molecules`."""
    rows = []
    for ds, d in _load().items():
        c = d["measured"]["counts"]
        called, shallow = c.get("cells_called"), c.get("dropped_cells_shallow")
        assert called is not None and shallow is not None, f"{ds} does not record its shallow loss"
        rows.append((ds, shallow, called))
    tot_s, tot_c = sum(r[1] for r in rows), sum(r[2] for r in rows)
    print(f"\n  cells dropped below min_cell_molecules, fleet: {tot_s:,}/{tot_c:,} "
          f"= {tot_s / tot_c * 100:.2f}% over {len(rows)} corpora")
    for ds, s, c in sorted(rows, key=lambda r: -r[1] / r[2])[:4]:
        if s:
            print(f"    {ds:44s} {s:6,}/{c:7,} = {s / c * 100:5.2f}%")


def test_trim_configuration_is_actually_stamped():
    """What this binary ACTS ON must reach the stamp, and it trims the reads itself.
    The builder applies all of them, so all of them are stamped. They were dropped from the
    stamp while trimming was external, because recording a parameter nothing applied reads as
    evidence that it was; the same rule puts them back.
    """
    want = {"decode_backend", "phred_offset", "quality_cutoff", "error_rate", "stringency",
            "adapters", "tso", "tso_min_score", "homopolymer_min_run", "min_len",
            "trimmed_here"}
    old, new, bad = [], [], {}
    for ds, d in _load().items():
        trim = d["declared"]["config"].get("trim") or {}
        if set(trim.keys()) <= {"decode_backend"}:
            old.append(ds)
            continue
        new.append(ds)
        gap = want - set(trim.keys())
        if gap:
            bad[ds] = sorted(gap)
    print(f"\n  trim stamp: {len(new)} corpora carry the full config, {len(old)} on the "
          f"decode_backend-only stamp")
    assert new, (
        f"no corpus carries the full trim config: all {len(old)} are on the decode_backend-only "
        f"stamp, so the gap check below runs against zero corpora. Nothing then records the "
        f"the TSO thresholds, the length floor or the fact that this binary trimmed. Rebuild "
        f"the corpora, or drop the claim in stamp.rs and TrimSettings."
    )
    assert not bad, (
        "these corpora carry a trim stamp that is neither the decode_backend-only shape nor the "
        f"full one, so stamp.rs's claim does not describe the artifact: {bad}"
    )


def test_the_corpus_root_env_names_match_the_registry():
    """The restated `_ROOT_ENV` above must name exactly what `registry.py` honours, in order."""
    import re

    src = (Path(__file__).resolve().parents[3] / "eval" / "scrm_eval" / "registry.py").read_text()
    m = re.search(r"^_ROOT_ENV = \(([^)]*)\)", src, re.M)
    assert m, "_ROOT_ENV not found in registry.py -- if it was renamed, this test must follow it"
    theirs = tuple(re.findall(r'"([^"]+)"', m.group(1)))
    assert theirs, "parsed no names out of registry._ROOT_ENV"
    assert theirs == _ROOT_ENV, (
        f"registry.py honours {theirs} but this file honours {_ROOT_ENV}. A corpus-root variable "
        f"the registry reads and these checkers do not is a checker auditing a different tree from "
        f"the one everything else uses, and reporting success about it."
    )
