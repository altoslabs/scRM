"""The evaluation assembly end to end, on synthetic embeddings with known answers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from proj_guard import needs_proj

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scrm_eval import artifact, cli, controls  # noqa: E402


def fake_embeddings(mat: np.ndarray, cells=None) -> artifact.Embeddings:
    """An `Embeddings` with one window per cell, bypassing disk."""
    n = mat.shape[0]
    ids = np.asarray(cells if cells is not None else [f"C{i:04d}" for i in range(n)])
    stamp = artifact.Stamp(
        producer="test",
        dim=mat.shape[1],
        n_cells=n,
        n_windows=n,
        cell_order_sha256=artifact.cell_order_digest(ids.tolist()),
        corpus_compat_key={"k": 19},
        corpus_digests={"ds": "a" * 64},
        code_rev="test",
        w_eval=1,
        eval_seed=0,
        seq_len=16384,
    )
    return artifact.Embeddings(
        cell_ids=ids, windows=np.zeros(n, dtype=np.uint16), matrix=mat.astype(np.float32), stamp=stamp
    )


def clustered(n_per=60, k=6, dim=24, sep=6.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(size=(n_per, dim)) + sep * np.eye(dim)[c] for c in range(k)])


def test_the_clustering_protocol_is_deterministic():
    """The clustering protocol must return identical labels across runs at one seed."""
    x = clustered()
    assert np.array_equal(cli.protocol_labels(x), cli.protocol_labels(x))


def test_the_protocol_produces_exactly_the_requested_number_of_clusters():
    """The clustering protocol must produce exactly the requested number of clusters."""
    x = clustered()
    for k in (2, 5, 10):
        assert len(set(cli.protocol_labels(x, k=k).tolist())) == k


def test_a_rotated_copy_of_the_reference_scores_perfectly_on_every_symmetric_metric():
    """A rotated copy of the reference must score perfectly on every symmetric metric."""
    rng = np.random.default_rng(1)
    ref = clustered(dim=24)
    rot = np.linalg.qr(rng.normal(size=(24, 24)))[0]
    emb = fake_embeddings(ref @ rot)
    cells = [f"C{i:04d}" for i in range(ref.shape[0])]

    got = cli.evaluate(emb, "ds", rule="mean", reference=(cells, ref))
    assert got["ari_symmetric_ari"] == pytest.approx(1.0, abs=1e-6)
    assert got["cka_vs_count_scanpy"] == pytest.approx(1.0, abs=1e-6)
    assert got["neighbor_overlap_vs_count_scanpy"] == pytest.approx(1.0, abs=1e-6)
    assert got["pc_r2_variance_weighted"] > 0.999


def test_an_unrelated_embedding_scores_at_the_floor_on_every_symmetric_metric():
    rng = np.random.default_rng(2)
    ref = clustered(dim=24)
    emb = fake_embeddings(rng.normal(size=(ref.shape[0], 24)))
    cells = [f"C{i:04d}" for i in range(ref.shape[0])]

    got = cli.evaluate(emb, "ds", rule="mean", reference=(cells, ref))
    assert got["ari_symmetric_ari"] < 0.05, got["ari_symmetric_ari"]
    assert got["cka_vs_count_scanpy"] < 0.2, got["cka_vs_count_scanpy"]
    assert got["neighbor_overlap_vs_count_scanpy"] < 0.1
    assert got["pc_r2_variance_weighted"] < 0.05


def test_depth_residualisation_is_reported_when_a_nuisance_is_supplied():
    """Raw R2 and depth-residualised R2 must both be reported when a nuisance is supplied."""
    rng = np.random.default_rng(3)
    n = 400
    depth = rng.normal(size=(n, 2))
    ref = np.column_stack([depth @ np.array([[2.0], [1.5]]), rng.normal(size=(n, 5))])
    cells = [f"C{i:04d}" for i in range(n)]
    emb = fake_embeddings(np.column_stack([depth, np.zeros((n, 6))]))

    got = cli.evaluate(emb, "ds", rule="mean", reference=(cells, ref), depth=(cells, depth))
    assert got["pc_r2_variance_weighted"] > 0.2, "depth alone should score on a depth-heavy PC space"
    assert got["pc_r2_depth_residualised"] < 0.05, "and near zero once depth is removed"


def test_independent_labels_add_the_label_metrics_and_their_absence_omits_them():
    """Independent labels must add the label metrics, and their absence must omit those keys."""
    x = clustered()
    cells = [f"C{i:04d}" for i in range(x.shape[0])]
    labels = {c: i // 60 for i, c in enumerate(cells)}
    emb = fake_embeddings(x)

    with_labels = cli.evaluate(emb, "ds", rule="mean", independent_labels=labels)
    assert with_labels["purity_purity_adj"] > 0.9
    assert with_labels["reference_clusters"] == 6

    without = cli.evaluate(emb, "ds", rule="mean")
    assert "purity_purity_adj" not in without
    assert without["effective_rank"] > 1.0


def test_a_collapsed_embedding_produces_numbers_rather_than_exceptions():
    """A collapsed embedding must produce numbers rather than raise."""
    n = 200
    cells = [f"C{i:04d}" for i in range(n)]
    emb = fake_embeddings(np.tile(np.arange(8, dtype=np.float32), (n, 1)))
    got = cli.evaluate(emb, "ds", rule="mean", reference=(cells, clustered(n_per=n // 4, k=4, dim=8)))
    assert got["effective_rank"] == 1.0
    assert got["cka_vs_count_scanpy"] == 0.0


def test_every_reduction_rule_is_accepted_by_the_assembly():
    """Every reduction rule must run through one assembly."""
    from scrm_eval import reduce

    rng = np.random.default_rng(4)
    ids = [f"C{i:04d}" for i in range(20) for _ in range(3)]
    mat = rng.normal(size=(60, 12)).astype(np.float32)
    emb = artifact.Embeddings(
        cell_ids=np.asarray(ids),
        windows=np.asarray([j for _ in range(20) for j in range(3)], dtype=np.uint16),
        matrix=mat,
        stamp=fake_embeddings(mat[:20]).stamp,
    )
    for rule in reduce.RULES:
        got = cli.evaluate(emb, "ds", rule=rule)
        assert got["n_cells"] == 20, rule
        assert got["rule"] == rule


def test_the_control_registry_matches_what_is_advertised():
    """Every control named in `PRODUCERS` must have a builder in `BUILDERS`."""
    assert set(controls.PRODUCERS) == set(controls.BUILDERS)


def test_an_unknown_control_is_refused_by_name():
    from scrm_eval import registry

    with pytest.raises(ValueError, match="unknown control"):
        controls.build(registry.datasets("train")[0], "counts_std", Path("/tmp"))


@needs_proj
def test_every_control_has_a_count_matrix_recorded():
    """Every dataset a control can be built for must have a registered, openable count matrix."""
    from scrm_eval import registry

    need = [d for d in registry.datasets("eval") + registry.datasets("eval-pending")
            + registry.datasets("train") if not d.endswith("_permuted")]
    missing = [d for d in need if d not in registry.COUNT_MATRICES]
    assert not missing, (
        f"these datasets have no registered count matrix: {missing}. Register each in "
        "registry.COUNT_MATRICES, the one table read by both targets.load_counts and "
        "controls._read_counts."
    )
    for name, path in registry.count_matrices().items():
        pth = Path(path)
        if str(pth).endswith((".h5ad", ".h5")):
            assert pth.is_file(), f"{name}: {path} is not a file"
        else:
            assert pth.is_dir(), f"{name}: {path} is not a directory"


def test_digest_file_detects_a_single_changed_byte(tmp_path):
    """`digest_file` must detect a single changed byte."""
    p = tmp_path / "a.bin"
    p.write_bytes(b"\x00" * 4096)
    before = controls.digest_file(p)
    data = bytearray(p.read_bytes())
    data[2048] = 1
    p.write_bytes(bytes(data))
    assert controls.digest_file(p) != before


def test_the_count_fm_panel_is_a_well_formed_set_of_distinct_comparators():
    """`controls.PANEL` must be a non-empty set of distinct models disjoint from the producers."""
    assert controls.PANEL, "controls.PANEL must not be empty"
    assert len(set(controls.PANEL)) == len(controls.PANEL), (
        f"a model appears twice in controls.PANEL: {controls.PANEL}"
    )
    overlap = set(controls.PANEL) & set(controls.PRODUCERS)
    assert not overlap, (
        f"{sorted(overlap)} is declared both a count-FM panel member and a control producer"
    )


def test_an_npz_baseline_becomes_a_stamped_artifact(tmp_path, monkeypatch):
    """An npz baseline must convert into a stamped artifact without being scored."""
    cells = [f"C{i:04d}" for i in range(10)]
    monkeypatch.setattr(controls, "corpus_facts", lambda ds: ({"k": 19}, {ds: "a" * 64}, cells))
    npz = tmp_path / "gf.npz"
    np.savez(npz, emb=np.arange(10 * 4, dtype=np.float32).reshape(10, 4),
             barcode=np.array([f"{c}-1" for c in cells]))

    panel_model = controls.PANEL[0]
    out = controls.from_npz("ds", panel_model, npz, tmp_path)
    got = artifact.read(out)
    assert got.stamp.producer == panel_model
    assert got.stamp.readout is None, "a baseline is a different producer, not a model"
    assert got.cell_ids.tolist() == cells
    assert got.stamp.notes["cells_kept"] == 10


def test_a_panel_model_that_dropped_cells_is_realigned_and_the_count_recorded(tmp_path, monkeypatch):
    """A panel model that dropped cells must be realigned into corpus order, with the size of the
    intersection recorded.
    """
    cells = [f"C{i:04d}" for i in range(10)]
    monkeypatch.setattr(controls, "corpus_facts", lambda ds: ({"k": 19}, {ds: "a" * 64}, cells))
    kept = cells[::2]
    npz = tmp_path / "gf.npz"
    np.savez(npz, emb=np.ones((5, 3), dtype=np.float32), barcode=np.array(list(reversed(kept))))

    got = artifact.read(controls.from_npz("ds", controls.PANEL[0], npz, tmp_path))
    assert got.cell_ids.tolist() == kept, "rows must come back in CORPUS order, not npz order"
    assert got.stamp.n_cells == 5
    assert got.stamp.notes["cells_in_npz"] == 5 and got.stamp.notes["cells_kept"] == 5


def test_an_npz_missing_its_keys_is_refused(tmp_path):
    npz = tmp_path / "bad.npz"
    np.savez(npz, embedding=np.ones((3, 2)))
    with pytest.raises(ValueError, match="has no 'emb'"):
        controls.from_npz("ds", controls.PANEL[0], npz, tmp_path)


def test_an_npz_sharing_no_cells_with_the_corpus_is_refused(tmp_path, monkeypatch):
    """An npz sharing no cells with the corpus must be refused."""
    monkeypatch.setattr(controls, "corpus_facts", lambda ds: ({"k": 19}, {ds: "a" * 64}, ["A", "B"]))
    npz = tmp_path / "x.npz"
    np.savez(npz, emb=np.ones((2, 3), dtype=np.float32), barcode=np.array(["Z", "Y"]))
    with pytest.raises(ValueError, match="shares no cells"):
        controls.from_npz("ds", controls.PANEL[0], npz, tmp_path)


def _scrm(mat, cells):
    """An `Embeddings` stamped as ours, so `reduce.rule_for` resolves to SCRM_RULE."""
    import dataclasses

    e = fake_embeddings(mat.astype(np.float32), cells=cells)
    return dataclasses.replace(e, stamp=dataclasses.replace(
        e.stamp, producer=sorted(artifact.MODEL_PRODUCERS)[0], checkpoint_sha256="a" * 64,
        arch_sha="arch0",
        readout={"pool": "segment_attention"}, forward_precision="bf16+eager"))


def test_the_reduction_an_scrm_artifact_is_read_with_is_material_and_is_not_a_default():
    """The reduction an scRM artifact is read with must change the numbers and must not come from a
    literal default.
    """
    from scrm_eval import reduce

    x = clustered(n_per=50, k=4, dim=16, sep=3.0) + 300.0
    cells = [f"C{i:04d}" for i in range(x.shape[0])]
    labels = {c: i // 50 for i, c in enumerate(cells)}
    emb = _scrm(x, cells)

    assert reduce.rule_for(emb.stamp) == reduce.SCRM_RULE, "ours must not resolve to a bare mean"
    raw = cli.evaluate(emb, "ds", rule="mean", independent_labels=labels)
    ours = cli.evaluate(emb, "ds", rule=reduce.rule_for(emb.stamp), independent_labels=labels)
    assert raw["silhouette_within_dataset_only"] != ours["silhouette_within_dataset_only"], (
        "the reduction does not reach the angular readouts, so this CLI cannot be reading one"
    )
    assert ours["rule"] == reduce.SCRM_RULE, "the row must record the rule that produced it"

    base = fake_embeddings(x.astype(np.float32), cells=cells)
    assert reduce.rule_for(base.stamp) == "mean"


def test_the_report_does_not_claim_a_clustering_it_never_runs():
    """The report must not stamp Leiden provenance for a reference protocol that is k-means."""
    src = Path(cli.__file__).read_text()
    assert "cluster_provenance()" not in src, (
        "cli.py emits graph-clustering provenance for a protocol that is k-means"
    )


def test_build_controls_only_iterates_the_producers_something_can_build():
    """`build_controls` must iterate only the producers something can build."""
    src = Path(cli.__file__).read_text()
    assert "controls.BUILT_PRODUCERS" in src
    assert "controls.PRODUCERS" not in src, "the build loop must not include an unbuildable producer"
    assert set(controls.BUILT_PRODUCERS) <= set(controls.PRODUCERS), (
        "BUILT_PRODUCERS names a producer PRODUCERS does not declare"
    )


def test_the_dataset_argument_must_name_a_corpus_the_artifacts_cover():
    """`--dataset` must name a corpus the artifacts' `corpus_digests` cover."""
    cells = [f"C{i:04d}" for i in range(20)]
    emb = fake_embeddings(np.zeros((20, 4), dtype=np.float32), cells=cells)

    cli.check_dataset("ds", [emb])
    with pytest.raises(artifact.StampError, match="names no corpus"):
        cli.check_dataset("some_other_tissue", [emb])


def test_the_per_pc_r2_list_is_not_silently_truncated():
    """`pc_r2_per_pc` must hold one entry per reference dimension, never a truncated list."""
    ref = clustered(n_per=40, k=4, dim=20)
    cells = [f"C{i:04d}" for i in range(ref.shape[0])]
    emb = fake_embeddings(ref.astype(np.float32), cells=cells)
    got = cli.evaluate(emb, "ds", rule="mean", reference=(cells, ref))
    assert len(got["pc_r2_per_pc"]) == got["reference_dim"] == 20


def test_the_rules_flag_does_not_default_to_a_literal_reduction():
    """`--rules` must not default to a literal reduction."""
    src = Path(cli.__file__).read_text()
    assert '"--rules", nargs="*", default=None' in src, (
        "--rules has a literal default again, so the default invocation reads an scrm artifact raw"
    )


def test_a_multi_well_10x_bundle_is_refused_rather_than_joined_on_a_colliding_key(tmp_path):
    """A multi-well 10x bundle must be refused rather than joined on a colliding barcode key."""
    from scrm_eval import tenx

    single = tmp_path / "single.csv"
    single.write_text("Barcode,Cluster\nAAAACCCGGGTTTAAA-1,3\nCCCCAAAGGGTTTAAA-1,4\n")
    assert tenx._read_two_column(single) == {"AAAACCCGGGTTTAAA": "3", "CCCCAAAGGGTTTAAA": "4"}

    multi = tmp_path / "multi.csv"
    multi.write_text("Barcode,Cluster\nAAAACCCGGGTTTAAA-1,3\nAAAACCCGGGTTTAAA-2,7\n")
    with pytest.raises(ValueError, match="aggregated multi-well bundle"):
        tenx._read_two_column(multi)


def test_the_pca10_controls_width_is_checked_against_its_own_name(tmp_path):
    """`counts_10x_pca10` must be checked to carry the ten components its name states."""
    from scrm_eval import tenx

    pcadir = tmp_path / "ds" / "analysis" / "pca" / "gene_expression_10_components"
    pcadir.mkdir(parents=True)
    wide = ",".join(f"PC-{i + 1}" for i in range(3))
    (pcadir / "projection.csv").write_text(f"Barcode,{wide}\nAAAACCCGGGTTTAAA-1,1,2,3\n")
    with pytest.raises(ValueError, match="components, but this is the"):
        tenx.pca("ds", root=tmp_path)


def test_a_dataset_without_a_cellranger_bundle_is_scored_rather_than_crashing():
    """The entrypoint must score a dataset that ships no 10x analysis bundle and record that it did."""
    src = Path(cli.__file__).read_text()
    assert "except FileNotFoundError:" in src, (
        "cli.main must tolerate a missing CellRanger bundle; an unconditional tenx.clusters() call "
        "raises for every evaluation dataset that ships none"
    )
    assert '"10x graph clustering" if independent is not None else None' in src, (
        "the report's independent_labels field must go null when the labels are absent, or it "
        "records a provenance that did not happen"
    )
