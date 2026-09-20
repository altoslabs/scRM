"""`model/jobs/corpora.tsv` is the one declaration of how every corpus is acquired.

Fourteen per-dataset launchers used to hold this between them, and they disagreed in ways that
were invisible to anything but reading each file to the end: two built the same corpus from two
different samplesheets, one hardcoded a coverage constant the report generator refuses, one
validated its whitelist against the corpus it was fetching reads to build, and whether the reads
were released depended on which launcher you ran. These tests are what makes the replacement a
declaration rather than another claim: the shape is checked with no data, and where a corpus
exists on disk its manifest is the oracle.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TABLE = REPO / "model" / "jobs" / "corpora.tsv"

ROLES = {"train", "eval", "screen"}
KINDS = {"tenx-tar", "sra-reads", "sra-bam", "ebi-files", "czb-s3", "permuted-from"}
WHITELIST_SCHEMES = ("h5", "h5-multi", "tar:", "repo:", "geo:", "json:", "matrix:",
                     "derive:", "derived", "UNRESOLVED")


def rows() -> list[dict]:
    lines = [ln for ln in TABLE.read_text().splitlines() if ln and not ln.startswith("#")]
    header = lines[0].split("\t")
    out = []
    for ln in lines[1:]:
        cells = ln.split("\t")
        assert len(cells) == len(header), (
            f"ragged row {cells[0]!r}: {len(cells)} cells against {len(header)} columns. The "
            f"source column is k=v;k=v precisely so a kind needing extra fields cannot widen "
            f"the table."
        )
        out.append(dict(zip(header, cells)))
    return out


def source(row: dict) -> dict:
    return dict(kv.split("=", 1) for kv in row["source"].split(";") if "=" in kv)


def test_the_table_is_readable_and_not_empty():
    r = rows()
    assert r, "no dataset rows; a declaration nothing reads is not a declaration"
    assert len({x["key"] for x in r}) == len(r), "duplicate key"


@pytest.mark.parametrize("row", rows(), ids=lambda r: r["key"])
def test_every_row_declares_a_known_role_kind_and_whitelist_scheme(row):
    assert row["role"] in ROLES, f"{row['key']}: role {row['role']!r} not in {sorted(ROLES)}"
    assert row["kind"] in KINDS, f"{row['key']}: kind {row['kind']!r} not in {sorted(KINDS)}"
    assert row["whitelist"].startswith(WHITELIST_SCHEMES), (
        f"{row['key']}: whitelist {row['whitelist']!r} names no scheme. A path with no scheme is "
        f"how a dataset ends up buildable only on the machine that already had the file."
    )
    assert row["geometry"].startswith("1{b[") and row["geometry"].endswith("}2{r:}"), (
        f"{row['key']}: geometry {row['geometry']!r} is not an FGDL string. Chemistry is declared "
        f"here and never auto-detected, so a malformed one has no fallback."
    )


@pytest.mark.parametrize("row", rows(), ids=lambda r: r["key"])
def test_each_kind_carries_the_locator_fields_it_needs(row):
    need = {
        "tenx-tar": {"host", "ver", "sample"},
        "sra-reads": {"srr"},
        "sra-bam": {"srr", "tag", "gsm_matrix", "gsm_genotype"},
        "ebi-files": {"accession", "sample", "lanes", "lane_suffix", "cells_called"},
        "czb-s3": {"bucket", "prefix", "channel"},
        "permuted-from": {"src", "seed"},
    }[row["kind"]]
    have = set(source(row))
    assert need <= have, f"{row['key']} ({row['kind']}): missing {sorted(need - have)}"


@pytest.mark.parametrize("row", [r for r in rows() if r["whitelist"].startswith("repo:")],
                         ids=lambda r: r["key"])
def test_a_repo_whitelist_is_tracked_by_git_not_merely_on_disk(row):
    rel = row["whitelist"][len("repo:"):]
    r = subprocess.run(["git", "-C", str(REPO), "ls-files", "--", rel],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("not a git checkout")
    on_disk = (REPO / rel).is_file()
    assert r.stdout.strip(), (
        f"{row['key']} declares repo:{rel}, which git does not track"
        + (" -- it is present on disk, which is exactly how this goes unnoticed: the machine that "
           "built the corpora has it and a clone does not." if on_disk else " and is not on disk.")
    )


def _corpus_root() -> Path | None:
    for var in ("SCRM_CORPUS_ROOT", "CORPORA_ROOT", "CORPUS_ROOT"):
        if os.environ.get(var):
            return Path(os.environ[var])
    proj = os.environ.get("SCRM_PROJ")
    return Path(proj) / "corpus" if proj else None


@pytest.mark.parametrize("row", rows(), ids=lambda r: r["key"])
def test_the_declared_geometry_matches_the_corpus_that_was_actually_built(row):
    root = _corpus_root()
    if root is None:
        pytest.skip("no corpus root: set SCRM_CORPUS_ROOT, CORPORA_ROOT, CORPUS_ROOT or SCRM_PROJ")
    m = root / row["key"] / "manifest.json"
    if not m.is_file():
        pytest.skip(f"{row['key']} is not built under {root}")
    built = json.loads(m.read_text())["declared"]["config"]["geometry"]
    assert built == row["geometry"], (
        f"{row['key']}: the table declares {row['geometry']!r} and the corpus on disk was built "
        f"with {built!r}. The geometry decides where the barcode and UMI are, so these cannot "
        f"differ -- one of them is wrong and rebuilding against the table would change the data."
    )


def test_every_dataset_the_page_evaluates_is_declared_here():
    import sys
    sys.path.insert(0, str(REPO / "measurements"))
    try:
        import results_page as rp
    except Exception as exc:                                    # pragma: no cover
        pytest.skip(f"results_page not importable: {exc}")
    declared = {r["key"] for r in rows()}
    missing = sorted(set(rp.DS) - declared)
    assert not missing, (
        f"{len(missing)} dataset(s) the page evaluates have no acquisition row: {missing}. A "
        f"dataset that cannot be acquired cannot be reproduced, and a name with no row is how "
        f"that goes unnoticed until someone tries."
    )


@pytest.mark.parametrize("row", [r for r in rows() if r["whitelist"].startswith("derive:")],
                         ids=lambda r: r["key"])
def test_a_derived_whitelist_names_a_derivation_that_exists(row):
    """`derive:<dataset>` is executed by model/jobs/derive_labels.py, which must implement it.

    A scheme naming a derivation nobody wrote would refuse at build time on the cluster, after the
    reads have been fetched, rather than here.
    """
    sys.path.insert(0, str(REPO / "model" / "jobs"))
    try:
        import derive_labels
    finally:
        sys.path.pop(0)
    ds = row["whitelist"][len("derive:"):]
    assert ds in derive_labels.DERIVATIONS, (
        f"{row['key']} declares derive:{ds}, which derive_labels.py does not implement; it knows "
        f"{sorted(derive_labels.DERIVATIONS)}")


#: `extra` is a locator under the row's OWN source, so only the kinds that can resolve one may
#: declare it. corpus.sbatch refuses any other kind rather than guessing a base.
EXTRA_KINDS = {"tenx-tar", "czb-s3"}


@pytest.mark.parametrize("row", [r for r in rows() if "extra" in source(r)], ids=lambda r: r["key"])
def test_an_extra_locator_is_resolvable_by_the_kind_that_declares_it(row):
    extra = source(row)["extra"]
    assert row["kind"] in EXTRA_KINDS, (
        f"{row['key']}: kind {row['kind']!r} declares extra={extra!r} but resolves no locator for "
        f"it; only {sorted(EXTRA_KINDS)} do")
    assert not extra.startswith(("/", "http", "s3:")), (
        f"{row['key']}: extra {extra!r} must be relative to the kind's own source, not absolute")
    if row["kind"] == "tenx-tar":
        assert extra.endswith((".tar", ".tar.gz", ".tgz")), (
            f"{row['key']}: a tenx-tar extra is extracted, so {extra!r} must be an archive")
