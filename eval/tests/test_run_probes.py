"""The E6/E9 runner is a join: targets from counts, embeddings from artifacts, probes over both.
These tests check the join — the probes' own statistics are covered in test_targets_probes.
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scrm_eval import artifact, controls, run_probes  # noqa: E402
from test_cli import fake_embeddings  # noqa: E402

A_MODEL_PRODUCER = sorted(artifact.MODEL_PRODUCERS)[0]
A_COUNT_PRODUCER = controls.PRODUCERS[0]


@pytest.fixture(scope="module")
def counts_dir(tmp_path_factory):
    """A tiny 10x v3 matrix dir with structured counts: two blocks of cells expressing two
    disjoint gene blocks, so panels and probes have real signal to find.
    """
    import scipy.io
    import scipy.sparse as sp

    rng = np.random.default_rng(7)
    n, g = 80, 300
    x = rng.poisson(0.5, size=(n, g)).astype(np.float32)
    x[:40, :10] += rng.poisson(6.0, size=(40, 10))
    x[40:, 10:20] += rng.poisson(6.0, size=(40, 10))
    d = tmp_path_factory.mktemp("mtx")
    with gzip.open(d / "matrix.mtx.gz", "wb") as f:
        scipy.io.mmwrite(f, sp.csr_matrix(x.T))
    barcodes = ["".join(rng.choice(list("ACGT"), 16)) + "-1" for _ in range(n)]
    with gzip.open(d / "barcodes.tsv.gz", "wt") as f:
        f.write("\n".join(barcodes) + "\n")
    with gzip.open(d / "features.tsv.gz", "wt") as f:
        f.write("".join(f"ENS{i}\tGene{i}\tGene Expression\n" for i in range(g)))
    return d, [b.split("-")[0] for b in barcodes], x


def test_report_joins_targets_probes_and_every_artifact(counts_dir, tmp_path):
    d, cells, x = counts_dir
    detected = (x > run_probes.DETECTION_THRESHOLD).astype(np.float32)
    informed = fake_embeddings(detected + np.random.default_rng(0).normal(0, 0.01, x.shape), cells=cells)
    noise = fake_embeddings(np.random.default_rng(1).normal(size=x.shape), cells=cells)

    gmt = tmp_path / "m.gmt"
    gmt.write_text("MOD_A\turl\t" + "\t".join(f"Gene{i}" for i in range(20)) + "\n")

    report = run_probes.probe_report({"informed": informed, "noise": noise}, str(d), gmt, "mean")

    assert set(report["artifacts"]) == {"informed", "noise"}
    for tag in ("informed", "noise"):
        e = report["artifacts"][tag]
        assert set(e["binary_expression"]) == {
            "hvg200", "hvg500", "hvg2000", "leiden_markers",
            "pearson2000",
            "legacy_hvg500", "legacy_hvg1000", "legacy_hvg2000", "legacy_markers100",
            "markers_wilcoxon_gt0", "markers_wilcoxon_gt1", "markers_fstat_gt0",
        }
        assert "count_likelihood" not in e
        assert set(e["graded_expression"]) == {"raw", "depth_residualised"}
        assert set(e["module_activity"]) == {"raw", "depth_residualised"}
        assert e["binary_expression"]["hvg200"]["baseline"] == 0.5
        assert e["module_activity"]["raw"]["baseline"] == 0.0
    hvg = report["artifacts"]
    assert (hvg["informed"]["binary_expression"]["hvg200"]["macro"]
            > hvg["noise"]["binary_expression"]["hvg200"]["macro"])
    assert hvg["informed"]["binary_expression"]["hvg200"]["macro"] > 0.7


def test_e9_is_skipped_without_a_gmt_never_faked(counts_dir):
    d, cells, x = counts_dir
    emb = fake_embeddings(np.random.default_rng(2).normal(size=(len(cells), 8)), cells=cells)
    report = run_probes.probe_report({"e": emb}, str(d), None, "mean")
    assert "module_activity" not in report["artifacts"]["e"]
    assert "modules" not in report["panels"]


def test_a_skipped_task_says_so_in_the_report_rather_than_just_being_absent(counts_dir):
    """A skipped task must be recorded in the report rather than merely absent."""
    d, cells, _ = counts_dir
    emb = fake_embeddings(np.random.default_rng(21).normal(size=(len(cells), 8)), cells=cells)
    report = run_probes.probe_report({"e": emb}, str(d), None, "mean")
    assert "module_activity" in report["not_scored"], "E9's absence must be explained, not implied"
    assert "E9 skipped" in report["not_scored"]["module_activity"]


def test_the_headline_marker_panel_records_its_own_detection_threshold(counts_dir):
    """Every marker panel must record the detection threshold that produced it."""
    d, cells, _ = counts_dir
    emb = fake_embeddings(np.random.default_rng(22).normal(size=(len(cells), 8)), cells=cells)
    report = run_probes.probe_report({"e": emb}, str(d), None, "mean")
    for name in report["artifacts"]["e"]["binary_expression"]:
        assert "detection_threshold" in report["panels"][name]["provenance"], name
    assert report["panels"]["leiden_markers"]["provenance"]["detection_threshold"] == \
        run_probes.DETECTION_THRESHOLD
    grid = ("markers_wilcoxon_gt0", "markers_wilcoxon_gt1", "markers_fstat_gt0", "legacy_markers100")
    widths = {report["panels"][g]["provenance"]["top_k"] for g in grid}
    assert widths == {100}, f"the marker grid must be crossed at one panel width, got {widths}"
    assert report["panels"]["markers_wilcoxon_gt0"]["provenance"]["detection_threshold"] == 0
    assert report["panels"]["markers_wilcoxon_gt1"]["provenance"]["detection_threshold"] == 1


def test_the_depth_nuisance_is_on_the_targets_log1p_scale(counts_dir, monkeypatch):
    """The depth nuisance must be on the targets' log1p scale, so `depth_residualised` removes
    library size.
    """
    d, cells, x = counts_dir
    seen: dict[str, np.ndarray] = {}
    real = run_probes.expression_regression_probe

    def capture(*a, depth=None, **k):
        if depth is not None:
            seen["depth"] = np.asarray(depth).copy()
        return real(*a, depth=depth, **k)

    monkeypatch.setattr(run_probes, "expression_regression_probe", capture)
    emb = fake_embeddings(np.random.default_rng(41).normal(size=(len(cells), 6)), cells=cells)
    run_probes.probe_report({"e": emb}, str(d), None, "mean")

    assert "depth" in seen, "no depth nuisance reached the probe, so there is no residualised variant"
    got = seen["depth"]
    assert got.shape == (len(cells), 2), f"the nuisance must be controls.depth2's shape, got {got.shape}"

    from scrm_eval import targets as T

    ad = T.load_counts(str(d))
    rows, has = T.align_to_cells(ad, cells)
    sub = ad.X[rows[has]]
    want = np.zeros((len(cells), 2))
    want[has, 0] = np.log1p(np.asarray(sub.sum(axis=1)).ravel())
    want[has, 1] = np.log1p(np.asarray((sub > 0).sum(axis=1)).ravel())
    assert np.allclose(got, want), (
        "the depth nuisance is not log1p of the two count summaries, so `depth_residualised` is "
        "removing something other than library size"
    )
    assert got.max() < 20.0, (
        f"the nuisance's largest value is {got.max():.1f}. log1p of any real library size is under "
        f"20, so this is a raw count column, not the log1p scale the targets are on"
    )


def test_population_mismatch_reconciles_a_gap_but_refuses_a_chasm(counts_dir):
    """One missing cell is reconciled; most of them missing still aborts."""
    d, cells, _ = counts_dir
    a = fake_embeddings(np.zeros((len(cells), 4), dtype=np.float32), cells=cells)
    b = fake_embeddings(np.zeros((len(cells) - 1, 4), dtype=np.float32), cells=cells[:-1])
    report = run_probes.probe_report({"a": a, "b": b}, str(d), None, "mean")
    assert report["population"]["n_cells_scored"] == len(cells) - 1
    assert report["population"]["cells_dropped_to_intersect"] == {"a": 1, "b": 0}

    half = fake_embeddings(np.zeros((len(cells) // 2, 4), dtype=np.float32), cells=cells[: len(cells) // 2])
    with pytest.raises(Exception, match="not the same population"):
        run_probes.probe_report({"a": a, "b": half}, str(d), None, "mean")


def test_both_scoring_surfaces_pin_one_blas_thread_in_the_serial_path(counts_dir, monkeypatch):
    """Both scoring surfaces must pin one BLAS thread in the serial path."""
    from threadpoolctl import threadpool_info

    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("SCRM_PROBE_JOBS", raising=False)
    assert any(i["num_threads"] > 1 for i in threadpool_info()), (
        "this process already has every thread pool at 1, so the assertions below would hold "
        "whether the scorers pinned anything or not; run without OMP_NUM_THREADS=1"
    )

    def watcher(real, seen):
        def wrapped(*a, **k):
            seen.append({(i["user_api"], i["prefix"]): i["num_threads"] for i in threadpool_info()})
            return real(*a, **k)
        return wrapped

    d, cells, x = counts_dir

    seen_probes: list[dict] = []
    monkeypatch.setattr(run_probes, "binary_expression_probe",
                        watcher(run_probes.binary_expression_probe, seen_probes))
    emb = fake_embeddings(np.random.default_rng(51).normal(size=(len(cells), 6)), cells=cells)
    run_probes.probe_report({"e": emb}, str(d), None, "mean")
    monkeypatch.undo()

    from scrm_eval import run_annotation

    seen_ann: list[dict] = []
    monkeypatch.setattr(run_annotation, "annotation_probe",
                        watcher(run_annotation.annotation_probe, seen_ann))
    rng = np.random.default_rng(52)
    acells = ["".join(rng.choice(list("ACGT"), 16)) for _ in range(60)]
    classes = np.array(["L1"] * 30 + ["L2"] * 30)
    ax = rng.normal(size=(60, 5)) + 4.0 * np.eye(5)[(classes == "L2").astype(int)]
    aemb = fake_embeddings(ax.astype(np.float32), cells=acells)
    run_annotation.annotation_report({"e": aemb}, dict(zip(acells, classes)), None)

    for surface, seen in (("run_probes", seen_probes), ("run_annotation", seen_ann)):
        assert seen, f"{surface}'s probe was never called, so nothing was observed"
        for info in seen:
            over = {k: v for k, v in info.items() if v != 1}
            assert not over, (
                f"{surface} entered its probe with thread pools above 1: {over}, so the probe's "
                f"value depends on the core count through summation order"
            )


def test_a_report_killed_mid_write_does_not_replace_the_published_one(tmp_path, monkeypatch):
    """A report killed mid-write must leave the published one intact."""
    import json as _j
    import pathlib

    from scrm_eval import run_annotation

    out = tmp_path / "e2.json"
    out.write_text(_j.dumps({"artifacts": {"previous": "run"}}, indent=2))
    before = out.read_text()

    real_write = pathlib.Path.write_text

    def killed_half_way(self, data, *a, **k):
        real_write(self, data[: len(data) // 2])
        raise OSError("killed mid-write")

    monkeypatch.setattr(pathlib.Path, "write_text", killed_half_way)
    with pytest.raises(OSError, match="killed mid-write"):
        run_annotation._write(out, {"artifacts": {"fresh": [1, 2, 3], "more": "padding" * 20}})
    monkeypatch.undo()

    assert out.read_text() == before, (
        "the published report was replaced by a half-written one; a tmp-then-os.replace write "
        "must leave the previous report intact"
    )
    _j.loads(out.read_text())

    run_annotation._write(out, {"artifacts": {"fresh": 1}})
    assert _j.loads(out.read_text())["artifacts"] == {"fresh": 1}


def test_annotation_report_joins_labels_by_bare_barcode():
    from scrm_eval import run_annotation

    rng = np.random.default_rng(3)
    cells = ["".join(rng.choice(list("ACGT"), 16)) for _ in range(90)]
    classes = np.array(["L1"] * 30 + ["L2"] * 30 + ["L3"] * 30)
    x = rng.normal(size=(90, 6)) + 5.0 * np.eye(6)[np.searchsorted(["L1", "L2", "L3"], classes)]
    emb = fake_embeddings(x.astype(np.float32), cells=cells)
    labels = {c: cls for c, cls in zip(cells[:80], classes[:80])}

    report = run_annotation.annotation_report({"e": emb}, labels, "mean")
    e = report["artifacts"]["e"]
    assert e["linear"]["n_cells"] == 80, "unlabelled cells must be excluded, never guessed"
    assert e["linear"]["macro"] > 0.9 and e["knn_cosine"]["macro"] > 0.9
    assert e["linear"]["baseline"] < 0.35


def _annotation_arms():
    """One separable embedding and one pure-noise embedding of the same cells, both tagged `armX`."""
    import dataclasses

    rng = np.random.default_rng(17)
    cells = ["".join(rng.choice(list("ACGT"), 16)) for _ in range(90)]
    classes = np.array(["L1"] * 30 + ["L2"] * 30 + ["L3"] * 30)
    sep = rng.normal(size=(90, 6)) + 5.0 * np.eye(6)[np.searchsorted(["L1", "L2", "L3"], classes)]

    def arm(mat, ckpt):
        e = fake_embeddings(mat.astype(np.float32), cells=cells)
        return dataclasses.replace(e, stamp=dataclasses.replace(
            e.stamp, producer=A_MODEL_PRODUCER, checkpoint_sha256=ckpt, arch_sha="arch0",
            readout={"pool": "segment_attention"}, forward_precision="bf16+eager"))

    labels = {c: cls for c, cls in zip(cells, classes)}
    return arm(sep, "1" * 64), arm(rng.normal(size=(90, 6)), "2" * 64), labels, cells


def test_a_re_embedded_arm_does_not_inherit_its_annotation_row(tmp_path):
    """E2/E3's resume must key on the artifact's identity, not on its name."""
    from scrm_eval import run_annotation

    good, noise, labels, _ = _annotation_arms()
    out = tmp_path / "e2.json"

    first = run_annotation.annotation_report({"armX": good}, labels, None, out=out)
    second = run_annotation.annotation_report({"armX": noise}, labels, None, out=out)

    m1 = first["artifacts"]["armX"]["linear"]["macro"]
    m2 = second["artifacts"]["armX"]["linear"]["macro"]
    assert m1 > 0.9, "the separable arm must score well"
    assert m2 < 0.6, "the noise arm's own score must be visibly different"
    assert m2 != m1, "a re-embedded arm inherited the previous checkpoint's macro-F1"

    again = run_annotation.annotation_report({"armX": noise}, labels, None, out=out)
    import json as _j
    assert _j.loads(_j.dumps(again["artifacts"]["armX"], default=str)) == \
        _j.loads(_j.dumps(second["artifacts"]["armX"], default=str)), (
        "an unchanged artifact must be carried forward, or every restart recomputes from scratch"
    )


def test_a_relabelled_ground_truth_does_not_inherit_the_old_annotation_row(tmp_path):
    """A relabelled ground truth must not inherit the previous annotation row."""
    from scrm_eval import run_annotation

    good, _, labels, cells = _annotation_arms()
    out = tmp_path / "e2.json"
    first = run_annotation.annotation_report({"armX": good}, labels, None, out=out)

    shuffled = dict(zip(cells, np.roll(np.asarray([labels[c] for c in cells]), 17)))
    assert len(shuffled) == len(labels), "the relabelling must keep n_labels"
    second = run_annotation.annotation_report({"armX": good}, shuffled, None, out=out)

    assert second["artifacts"]["armX"]["linear"]["macro"] != \
        first["artifacts"]["armX"]["linear"]["macro"], "a row survived a change of ground truth"


def test_annotation_refuses_a_mixed_forward_arithmetic_and_records_the_field(tmp_path):
    """`run_annotation` must refuse a mixed forward arithmetic and record the field."""
    import dataclasses

    from scrm_eval import run_annotation

    good, noise, labels, _ = _annotation_arms()
    legacy = dataclasses.replace(noise, stamp=dataclasses.replace(
        noise.stamp, forward_precision=None))

    with pytest.raises(Exception, match="different forward arithmetic"):
        run_annotation.annotation_report({"a": good, "b": legacy}, labels, None)

    report = run_annotation.annotation_report({"a": good}, labels, None)
    assert report["artifacts"]["a"]["forward_precision"] == "bf16+eager"


def test_common_population_intersects_and_refuses_a_real_mismatch():
    """`common_population` must intersect a small gap and refuse a real mismatch."""
    rng = np.random.default_rng(11)
    ids = [f"C{i:04d}" for i in range(100)]
    a = fake_embeddings(rng.normal(size=(100, 4)).astype(np.float32), cells=ids)
    rev = list(reversed(ids[:98]))
    b = fake_embeddings(rng.normal(size=(98, 4)).astype(np.float32), cells=rev)

    tags = (A_MODEL_PRODUCER, A_COUNT_PRODUCER)
    cells, dropped = run_probes.common_population(dict(zip(tags, (a, b))))
    assert cells == ids[:98], "canonical order is the first artifact's, restricted to the intersection"
    assert dropped == dict(zip(tags, (2, 0)))

    pos = {str(c): i for i, c in enumerate(b.cell_ids)}
    assert pos["C0003"] == 94
    take = [pos[c] for c in cells]
    assert np.allclose(b.matrix[take][3], b.matrix[94])

    c = fake_embeddings(rng.normal(size=(50, 4)).astype(np.float32), cells=ids[:50])
    with pytest.raises(Exception, match="not the same population"):
        run_probes.common_population(dict(zip(tags, (a, c))))


def test_mixed_forward_arithmetic_is_refused(counts_dir):
    """fp32-eager and bf16-compiled embeddings of the same cells are not comparable."""
    import dataclasses

    d, cells, _ = counts_dir
    old = fake_embeddings(np.zeros((len(cells), 4), dtype=np.float32), cells=cells)
    new = fake_embeddings(np.zeros((len(cells), 4), dtype=np.float32), cells=cells)
    old = dataclasses.replace(old, stamp=dataclasses.replace(
        old.stamp, producer=A_MODEL_PRODUCER, forward_precision=None))
    new = dataclasses.replace(new, stamp=dataclasses.replace(
        new.stamp, producer=A_MODEL_PRODUCER, forward_precision="bf16+compiled"))

    with pytest.raises(Exception, match="different forward arithmetic"):
        run_probes.probe_report({"old": old, "new": new}, str(d), None, "mean")

    same = dataclasses.replace(new, stamp=dataclasses.replace(new.stamp, forward_precision="bf16+compiled"))
    base = dataclasses.replace(old, stamp=dataclasses.replace(
        old.stamp, producer=A_COUNT_PRODUCER, forward_precision=None))
    rep = run_probes.probe_report({"a": new, "b": same, "c": base}, str(d), None, "mean")
    assert rep["population"]["n_cells_scored"] == len(cells)


def test_a_salvaged_shortfall_is_allowed_but_an_unexplained_one_is_not(counts_dir):
    """A shortfall must be allowed when every short artifact is stamped `notes.salvaged`, and refused otherwise."""
    import dataclasses

    d, cells, _ = counts_dir
    n = len(cells)
    keep = int(n * 0.90)

    full = fake_embeddings(np.zeros((n, 4), dtype=np.float32), cells=cells)
    full = dataclasses.replace(full, stamp=dataclasses.replace(full.stamp, producer=A_MODEL_PRODUCER))
    part = fake_embeddings(np.zeros((keep, 4), dtype=np.float32), cells=cells[:keep])

    plain = dataclasses.replace(part, stamp=dataclasses.replace(part.stamp, producer=A_MODEL_PRODUCER))
    with pytest.raises(Exception, match="unexplained shortfall"):
        run_probes.common_population({"full": full, "short": plain})

    salv = dataclasses.replace(part, stamp=dataclasses.replace(
        part.stamp, producer=A_MODEL_PRODUCER, notes={"salvaged": True, "cells_reached": keep}))
    got, dropped = run_probes.common_population({"full": full, "short": salv})
    assert len(got) == keep
    assert dropped["full"] == n - keep and dropped["short"] == 0

    tiny = fake_embeddings(np.zeros((n // 2, 4), dtype=np.float32), cells=cells[: n // 2])
    tiny = dataclasses.replace(tiny, stamp=dataclasses.replace(
        tiny.stamp, producer=A_MODEL_PRODUCER, notes={"salvaged": True}))
    with pytest.raises(Exception, match="salvaged-artifact floor"):
        run_probes.common_population({"full": full, "short": tiny})


def test_the_depth_nuisance_is_the_depth2_control_columns(counts_dir):
    """`depth_nuisance` is what `controls.depth2` builds, so the gate and production share one."""
    from scrm_eval import targets

    d, _, _ = counts_dir
    ad = targets.load_counts(str(d))
    ref_cells, ref = controls.depth2(str(d), adata=ad)
    got = run_probes.depth_nuisance(ad, [str(c).split("-")[0] for c in ref_cells])
    assert got.shape == (len(ref_cells), 2), "two columns: log1p total counts, log1p genes detected"
    assert np.allclose(got, ref, atol=1e-5)


def test_equal_length_artifacts_that_disagree_on_cells_are_refused_at_the_strict_floor():
    """Equal-length artifacts have no short artifact to explain, so the strict floor applies."""
    import dataclasses

    rng = np.random.default_rng(5)
    shared = [f"C{i:04d}" for i in range(880)]
    a_ids = shared + [f"A{i:04d}" for i in range(120)]
    b_ids = shared + [f"B{i:04d}" for i in range(120)]
    a = fake_embeddings(rng.normal(size=(1000, 4)).astype(np.float32), cells=a_ids)
    a = dataclasses.replace(a, stamp=dataclasses.replace(a.stamp, producer=A_MODEL_PRODUCER))
    b = fake_embeddings(rng.normal(size=(1000, 4)).astype(np.float32), cells=b_ids)
    b = dataclasses.replace(b, stamp=dataclasses.replace(b.stamp, producer=A_MODEL_PRODUCER))

    assert len(a.cell_ids) == len(b.cell_ids), "the case is equal-length, so nothing is short"
    with pytest.raises(Exception, match="not the same population"):
        run_probes.common_population({"a": a, "b": b})

    wide = [f"C{i:04d}" for i in range(1000)]
    c = fake_embeddings(rng.normal(size=(1000, 4)).astype(np.float32), cells=wide)
    c = dataclasses.replace(c, stamp=dataclasses.replace(c.stamp, producer=A_MODEL_PRODUCER))
    d = fake_embeddings(rng.normal(size=(1000, 4)).astype(np.float32),
                        cells=wide[:960] + [f"D{i:04d}" for i in range(40)])
    d = dataclasses.replace(d, stamp=dataclasses.replace(d.stamp, producer=A_MODEL_PRODUCER))
    cells, dropped = run_probes.common_population({"c": c, "d": d})
    assert len(cells) == 960 and dropped == {"c": 40, "d": 40}


def test_a_salvage_stamp_on_an_equal_length_artifact_does_not_lower_the_floor():
    """`notes.salvaged` on an artifact that is not short explains no shortfall."""
    import dataclasses

    rng = np.random.default_rng(6)
    shared = [f"C{i:04d}" for i in range(880)]
    a = fake_embeddings(rng.normal(size=(1000, 4)).astype(np.float32),
                        cells=shared + [f"A{i:04d}" for i in range(120)])
    a = dataclasses.replace(a, stamp=dataclasses.replace(
        a.stamp, producer=A_MODEL_PRODUCER, notes={"salvaged": True}))
    b = fake_embeddings(rng.normal(size=(1000, 4)).astype(np.float32),
                        cells=shared + [f"B{i:04d}" for i in range(120)])
    b = dataclasses.replace(b, stamp=dataclasses.replace(
        b.stamp, producer=A_MODEL_PRODUCER, notes={"salvaged": True}))

    with pytest.raises(Exception, match="not the same population"):
        run_probes.common_population({"a": a, "b": b})


def _report_to(path, arts, dataset, gmt, reuse=True):
    import json as _j
    r = run_probes.probe_report(arts, dataset, gmt, "mean", reuse_from=path if reuse else None)
    path.write_text(_j.dumps(r, indent=2))
    return r


def test_an_unchanged_artifact_is_carried_forward_bit_identically(counts_dir, tmp_path):
    """An unchanged artifact's row must be carried forward bit-identically."""
    d, cells, x = counts_dir
    rng = np.random.default_rng(0)
    a = fake_embeddings((x > run_probes.DETECTION_THRESHOLD) + rng.normal(0, 0.01, x.shape), cells=cells)
    b = fake_embeddings(rng.normal(size=x.shape), cells=cells)
    out = tmp_path / "r.json"

    first = _report_to(out, {"a": a}, str(d), None)
    fresh = run_probes.probe_report({"a": a, "b": b}, str(d), None, "mean", reuse_from=None)
    reused = _report_to(out, {"a": a, "b": b}, str(d), None)

    assert set(reused["artifacts"]) == {"a", "b"}

    def norm(e):
        """A row in its on-disk form, so a tuple and a list of the same values compare equal."""
        import json as _j
        return _j.loads(_j.dumps({k: v for k, v in e.items() if k != "_reuse_key"}, default=str))

    assert norm(reused["artifacts"]["a"]) == norm(first["artifacts"]["a"]), "carried row was altered"
    assert norm(reused["artifacts"]["a"]) == norm(fresh["artifacts"]["a"]), (
        "the carried row differs from a freshly computed one -- reuse is not value-preserving"
    )


def test_a_re_embedded_arm_is_recomputed_not_carried(counts_dir, tmp_path):
    """A re-trained checkpoint under the same arm name must not inherit the previous row."""
    import dataclasses

    d, cells, x = counts_dir
    rng = np.random.default_rng(2)

    def arm(mat, ckpt):
        e = fake_embeddings(mat, cells=cells)
        return dataclasses.replace(e, stamp=dataclasses.replace(
            e.stamp, producer=A_MODEL_PRODUCER, checkpoint_sha256=ckpt, arch_sha="arch0",
            readout={"pool": "segment_attention"}, forward_precision="bf16+eager"))

    v1 = arm(rng.normal(size=x.shape), "1" * 64)
    v2 = arm((x > run_probes.DETECTION_THRESHOLD) + rng.normal(0, 0.01, x.shape), "2" * 64)
    out = tmp_path / "r.json"

    first = _report_to(out, {"arm": v1}, str(d), None)
    second = _report_to(out, {"arm": v2}, str(d), None)

    assert second["artifacts"]["arm"]["_reuse_key"] != first["artifacts"]["arm"]["_reuse_key"]
    m1 = first["artifacts"]["arm"]["binary_expression"]["hvg200"]["macro"]
    m2 = second["artifacts"]["arm"]["binary_expression"]["hvg200"]["macro"]
    assert m1 != m2, "a re-embedded arm inherited the previous checkpoint's row"


def test_adding_an_artifact_that_shrinks_the_population_recomputes_every_row(counts_dir, tmp_path):
    """Every probe value is a function of the artifact set, not of one artifact."""
    d, cells, x = counts_dir
    rng = np.random.default_rng(31)
    a = fake_embeddings((x > run_probes.DETECTION_THRESHOLD) + rng.normal(0, 0.01, x.shape), cells=cells)
    b = fake_embeddings(rng.normal(size=x.shape), cells=cells)
    out = tmp_path / "r.json"

    first = _report_to(out, {"a": a, "b": b}, str(d), None)
    assert first["population"]["n_cells_scored"] == len(cells)
    assert first["artifacts"]["a"]["binary_expression"]["hvg200"]["n_cells"] == len(cells)

    keep = int(len(cells) * 0.95)
    c = fake_embeddings(rng.normal(size=(keep, x.shape[1])), cells=cells[:keep])
    second = _report_to(out, {"a": a, "b": b, "c": c}, str(d), None)

    assert second["population"]["n_cells_scored"] == keep
    for tag in ("a", "b", "c"):
        got = second["artifacts"][tag]["binary_expression"]["hvg200"]["n_cells"]
        assert got == keep, (
            f"{tag} reports a macro fit on {got} cells inside a report scored on {keep} -- a row "
            f"was carried forward across a change in the intersected population"
        )


def test_reuse_off_recomputes_everything(counts_dir, tmp_path):
    """`--no-reuse` must recompute every row and still stamp the key."""
    d, cells, x = counts_dir
    a = fake_embeddings(np.random.default_rng(3).normal(size=x.shape), cells=cells)
    out = tmp_path / "r.json"
    _report_to(out, {"a": a}, str(d), None)
    again = run_probes.probe_report({"a": a}, str(d), None, "mean", reuse_from=None)
    assert "_reuse_key" in again["artifacts"]["a"], "the key must still be stamped when reuse is off"


def test_a_corrupt_prior_report_costs_a_slow_rescore_not_a_wrong_one(counts_dir, tmp_path):
    d, cells, x = counts_dir
    a = fake_embeddings(np.random.default_rng(4).normal(size=x.shape), cells=cells)
    out = tmp_path / "r.json"
    out.write_text("{ this is not json")
    r = run_probes.probe_report({"a": a}, str(d), None, "mean", reuse_from=out)
    assert "a" in r["artifacts"], "a malformed prior report must not lose the artifact"


def test_a_change_to_the_scoring_code_invalidates_every_carried_row(counts_dir, tmp_path, monkeypatch):
    """A change to the scoring code must invalidate every carried row."""
    d, cells, x = counts_dir
    a = fake_embeddings(np.random.default_rng(5).normal(size=x.shape), cells=cells)
    out = tmp_path / "r.json"
    first = _report_to(out, {"a": a}, str(d), None)
    k1 = first["artifacts"]["a"]["_reuse_key"]

    monkeypatch.setattr(run_probes.scoring, "scoring_version", lambda: "pretend-probes-changed")
    second = run_probes.probe_report({"a": a}, str(d), None, "mean", reuse_from=out)
    assert second["artifacts"]["a"]["_reuse_key"] != k1, (
        "the key ignored a change to the scoring code, so a stale row would be carried forward"
    )
