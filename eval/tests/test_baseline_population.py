"""The discriminator that tells a stale baseline from a cell no count model could ever embed."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "measurements"))
import baseline_population as bp  # noqa: E402
from scrm_eval import controls as _controls  # noqa: E402


def _a_producer_that_does_not_filter_cells() -> str:
    """SOME panel model that embeds every cell it is handed -- asked for, never named."""
    for p in _controls.PANEL:
        if p not in bp.FILTERS_CELLS:
            return p
    pytest.skip("every panel member declares a cell filter, so there is no unfiltered case to test")


def _a_producer_that_filters_cells() -> str:
    """SOME producer whose own published rule drops cells, with its rule declared."""
    for p in _controls.PANEL:
        if p in bp.FILTERS_CELLS:
            return p
    pytest.skip("no producer declares a cell filter, so there is no downgrade case to test")


def _a_read_derived_producer() -> str:
    """SOME producer whose input is the corpus rather than the count matrix -- asked for, not named."""
    for p in bp.READS_THE_CORPUS:
        return p
    pytest.skip("no read-derived producer is declared, so there is no wider universe to test")


def _case(monkeypatch, tmp_path, *, corpus, in_matrix, in_artifact, producer, dataset="a_dataset"):
    """Wire one (corpus, matrix, artifact) shape under a throwaway dataset name and grade it."""
    ds = dataset
    bar = [f"CELL{i:06d}" for i in range(corpus)]
    monkeypatch.setattr(bp.registry, "PROJ", tmp_path)
    monkeypatch.setattr(bp.controls, "corpus_facts", lambda d: (None, None, list(bar)))
    monkeypatch.setattr(bp.targets, "load_counts", lambda d:
                        types.SimpleNamespace(obs_names=[f"{b}-1" for b in bar[:in_matrix]]))
    kept = bar[:in_artifact]
    monkeypatch.setattr(bp.artifact, "read_stamp", lambda p: types.SimpleNamespace(
        n_cells=len(kept), cell_order_sha256=bp.artifact.cell_order_digest(kept)))
    monkeypatch.setattr(bp.artifact, "read", lambda p:
                        types.SimpleNamespace(cell_ids=np.array(kept)))
    bp.owed.cache_clear()
    bp.owed_ordered.cache_clear()
    art = tmp_path / "eval_artifacts" / f"{ds}.{producer}.arrow"
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_bytes(b"not read: artifact.read is patched")
    return bp.check(ds, producer)


def test_a_baseline_short_of_cells_the_matrix_has_is_stale(monkeypatch, tmp_path):
    """A baseline short of cells the matrix has must be graded stale."""
    verdict, detail = _case(monkeypatch, tmp_path, corpus=6343, in_matrix=6343, in_artifact=6315,
                            producer=_a_producer_that_does_not_filter_cells())
    assert verdict == "STALE", detail
    assert "28" in detail, f"the size of the gap must be reported, got: {detail}"


def test_a_baseline_short_only_of_cells_the_matrix_lacks_is_not_stale(monkeypatch, tmp_path):
    """A baseline short only of cells the matrix lacks must not be graded stale: a cell with no
    counts cannot be embedded by a count model.
    """
    verdict, detail = _case(monkeypatch, tmp_path, corpus=5255, in_matrix=5249, in_artifact=5249,
                            producer=_a_producer_that_does_not_filter_cells())
    assert verdict == "OK", detail


def _a_declared_filtered_pair():
    """SOME (dataset, producer) whose drop has been MEASURED, with its size -- asked for, not named."""
    for (ds, prod), n in bp.KNOWN_FILTERED.items():
        return ds, prod, n
    pytest.skip("no measured filtered drop is declared, so there is no exemption to test")


def test_a_measured_drop_is_attributed_to_its_declared_rule_and_still_reported(monkeypatch,
                                                                              tmp_path):
    """A gap matching a measured drop must be attributed to the producer's declared rule and must
    still be reported.
    """
    ds, prod, n = _a_declared_filtered_pair()
    corpus = 2474
    verdict, detail = _case(monkeypatch, tmp_path, corpus=corpus, in_matrix=corpus,
                            in_artifact=corpus - n, producer=prod, dataset=ds)
    assert verdict == "WARN", detail
    assert bp.FILTERS_CELLS[prod] in detail, "the exemption must name its own cause"
    assert str(n) in detail, f"a downgraded gap is still reported, got: {detail}"


def test_a_filtering_producer_with_no_measured_drop_for_the_pair_is_stale(monkeypatch, tmp_path):
    """A filtering producer with no measured drop for the pair must be graded stale: the exemption
    is per (dataset, producer) pair, not per producer.
    """
    ds, prod, _ = _a_declared_filtered_pair()
    verdict, detail = _case(monkeypatch, tmp_path, corpus=6343, in_matrix=6343, in_artifact=6315,
                            producer=prod, dataset=ds + "_a_different_dataset")
    assert verdict == "STALE", detail
    assert "KNOWN_FILTERED" in detail, detail


def test_a_declared_drop_that_changed_size_is_stale_and_says_both_numbers(monkeypatch, tmp_path):
    """A declared drop that changed size must be graded stale and must report both numbers."""
    ds, prod, n = _a_declared_filtered_pair()
    corpus = 2474
    verdict, detail = _case(monkeypatch, tmp_path, corpus=corpus, in_matrix=corpus,
                            in_artifact=corpus - (n + 7), producer=prod, dataset=ds)
    assert verdict == "STALE", detail
    assert str(n) in detail and str(n + 7) in detail, detail


def test_a_read_derived_control_is_graded_against_the_CORPUS_and_not_the_matrix(monkeypatch,
                                                                                tmp_path):
    """A read-derived control must be graded against the corpus, not the count matrix, so holding
    more cells than the matrix-backed set is correct.
    """
    prod = _a_read_derived_producer()
    verdict, detail = _case(monkeypatch, tmp_path, corpus=5255, in_matrix=5249, in_artifact=5255,
                            producer=prod)
    assert verdict == "OK", detail
    assert "corpus" in detail, detail
    verdict, detail = _case(monkeypatch, tmp_path, corpus=5255, in_matrix=5249, in_artifact=5249,
                            producer=prod)
    assert verdict == "STALE", f"a read-derived control owes every corpus cell: {verdict} {detail}"


def test_the_input_matrix_is_current_when_it_holds_exactly_the_matrix_backed_corpus(monkeypatch,
                                                                                    tmp_path):
    """`fm/cells_<ds>` must be current only when it holds exactly the corpus intersected with the
    registered matrix: a missing cell and an extra cell are both wrong.
    """
    ds = "a_dataset"
    bar = [f"CELL{i:06d}" for i in range(5255)]
    monkeypatch.setattr(bp.registry, "PROJ", tmp_path)
    monkeypatch.setattr(bp.controls, "corpus_facts", lambda d: (None, None, list(bar)))
    monkeypatch.setattr(bp.targets, "load_counts", lambda d:
                        types.SimpleNamespace(obs_names=[f"{b}-1" for b in bar[:5249]]))
    bp.owed.cache_clear()
    bp.owed_ordered.cache_clear()

    monkeypatch.setattr(bp, "_handed", lambda d: set(bar[:5249]))
    assert bp.matrix_dir(ds)[0] == "OK"
    monkeypatch.setattr(bp, "_handed", lambda d: set(bar[:5249]) - {bar[0]})
    assert bp.matrix_dir(ds)[0] == "STALE", "a missing cell must be caught"
    monkeypatch.setattr(bp, "_handed", lambda d: set(bar))
    assert bp.matrix_dir(ds)[0] == "STALE", "an extra cell must be caught too"


def test_the_full_population_in_the_wrong_order_is_not_OK(monkeypatch, tmp_path):
    """A complete population in the wrong order must get its own verdict rather than reading as OK,
    because `cell_order_sha256` hashes cells in artifact order.
    """
    ds = "a_dataset"
    bar = [f"CELL{i:06d}" for i in range(300)]
    monkeypatch.setattr(bp.registry, "PROJ", tmp_path)
    monkeypatch.setattr(bp.controls, "corpus_facts", lambda d: (None, None, list(bar)))
    monkeypatch.setattr(bp.targets, "load_counts", lambda d:
                        types.SimpleNamespace(obs_names=[f"{b}-1" for b in bar]))
    bp.owed.cache_clear()
    bp.owed_ordered.cache_clear()
    shuffled = list(reversed(bar))
    monkeypatch.setattr(bp.artifact, "read_stamp", lambda p: types.SimpleNamespace(
        n_cells=len(shuffled), cell_order_sha256=bp.artifact.cell_order_digest(shuffled)))
    monkeypatch.setattr(bp.artifact, "read", lambda p:
                        types.SimpleNamespace(cell_ids=np.array(shuffled)))
    prod = _a_producer_that_does_not_filter_cells()
    art = tmp_path / "eval_artifacts" / f"{ds}.{prod}.arrow"
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_bytes(b"not read: artifact.read is patched")
    verdict, detail = bp.check(ds, prod)
    assert verdict == "WARN", f"a reordered artifact must not read as OK: {verdict} {detail}"
    assert "order" in detail.lower(), detail


def test_the_launcher_reads_the_shared_predicate_instead_of_keeping_its_own(monkeypatch):
    """The launcher must read this predicate rather than restating it inline."""
    repo = Path(__file__).resolve().parents[2]
    src = (repo / "model/jobs/baselines_matrix.sbatch").read_text()
    assert "baseline_population" in src, (
        "baselines_matrix.sbatch no longer consults measurements/baseline_population.py, so an "
        "artifact or an input matrix built for an older corpus can be skipped as though current."
    )
    assert "matrix_dir" in src, "the input-matrix currency test must be the shared one"
