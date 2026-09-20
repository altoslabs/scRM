"""E6/E9 machinery on synthetic data: alignment never fabricates, panels are the stated
definitions, and each probe detects a planted signal while its baseline stays put.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scrm_eval import probes, registry, targets  # noqa: E402


def fake_adata(n_cells=60, n_genes=40, seed=0):
    import anndata
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    x = sp.csr_matrix(rng.poisson(0.8, size=(n_cells, n_genes)).astype(np.float32))
    a = anndata.AnnData(x)
    a.obs_names = [f"{''.join(rng.choice(list('ACGT'), 16))}-1" for _ in range(n_cells)]
    a.var_names = [f"Gene{i}" for i in range(n_genes)]
    return a


def _tenx_dir(root, x, genes=None):
    """`x` (cells x genes) written as the 10x v3 mtx triple `targets.load_counts` reads."""
    import gzip

    import scipy.io
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    n, g = x.shape
    root.mkdir(parents=True, exist_ok=True)
    with gzip.open(root / "matrix.mtx.gz", "wb") as f:
        scipy.io.mmwrite(f, sp.csr_matrix(np.asarray(x, dtype=np.float32).T))
    cells = ["".join(rng.choice(list("ACGT"), 16)) for _ in range(n)]
    with gzip.open(root / "barcodes.tsv.gz", "wt") as f:
        f.write("\n".join(c + "-1" for c in cells) + "\n")
    names = genes or [f"Gene{i}" for i in range(g)]
    with gzip.open(root / "features.tsv.gz", "wt") as f:
        f.write("".join(f"ENS{i}\t{names[i]}\tGene Expression\n" for i in range(g)))
    return cells


def test_a_gene_detected_in_too_few_cells_is_filtered_before_every_selection(tmp_path, monkeypatch):
    """`MIN_CELLS = 20` must be applied in `load_counts`, so HVG selection, the Leiden graph and the
    gene universe all see one gene set.
    """
    n, g = 80, 60
    x = np.random.default_rng(1).poisson(3.0, size=(n, g)).astype(np.float32)
    rare = list(range(10))
    x[:, :5] = 0.0
    for j in range(5):
        x[j, j] = 7.0
    x[:, 5:10] = 0.0
    x[:3, 5:10] = 4.0
    monkeypatch.setattr(targets, "CACHE_DIR", tmp_path / "tcache")
    d = tmp_path / "mtx"
    _tenx_dir(d, x)

    ad = targets.load_counts(str(d))
    rare_names = {f"Gene{j}" for j in rare}
    kept = rare_names & set(map(str, ad.var_names))
    assert not kept, (
        f"genes detected in 1-3 cells survived a MIN_CELLS={targets.MIN_CELLS} filter: {sorted(kept)}. "
        f"HVG selection, the Leiden graph and the gene universe would each see a different gene set"
    )
    assert ad.n_vars == g - len(rare), f"expected {g - len(rare)} genes after the filter, got {ad.n_vars}"
    idx = targets.hvg_panel(ad, n_top=10)
    assert idx.max() < ad.n_vars
    assert not rare_names & {str(ad.var_names[i]) for i in idx}


def test_a_species_with_no_hallmark_collection_gets_None_and_an_unknown_dataset_raises():
    """A species MSigDB does not publish must yield no target set, never a borrowed one."""
    species = registry.species()
    scored = registry.datasets("eval") + registry.datasets("eval-pending")
    served = [d for d in scored if species[d] in targets.GENE_SETS]
    unserved = [d for d in scored if species[d] not in targets.GENE_SETS]
    assert served and unserved, (
        "this test needs at least one scored dataset on each side of the partition; with all of them "
        f"on one side it asserts nothing. served={served} unserved={unserved}"
    )
    for ds in unserved:
        assert targets.gene_set_path(ds) is None, (
            f"{ds} is {species[ds]}, a species with no hallmark collection, and a GMT was returned "
            "anyway -- so it is another species' collection being borrowed"
        )
    for ds in served:
        assert targets.gene_set_path(ds), (
            f"{ds} is {species[ds]}, which GENE_SETS covers, so the None above is a fact about the "
            "species rather than the function being broken for everything"
        )
    with pytest.raises(targets.TargetError, match="unknown species"):
        targets.gene_set_path("no_such_dataset_anywhere")


class TestAlignment:
    def test_absent_cells_are_masked_never_fabricated(self):
        a = fake_adata()
        cells = [str(n).split("-")[0] for n in a.obs_names[:5]] + ["A" * 16]
        rows, has = targets.align_to_cells(a, cells)
        assert has.tolist() == [True] * 5 + [False]
        ts = targets.binary_expression_targets(a, cells, np.arange(4), panel="hvg")
        assert not ts.has_counts[-1]
        assert not ts.matrix[-1].any(), "an absent cell's targets must be all-zero AND masked"

    def test_bare_and_suffixed_barcodes_match(self):
        a = fake_adata()
        bare = str(a.obs_names[0]).split("-")[0]
        _, has = targets.align_to_cells(a, [bare, str(a.obs_names[1])])
        assert has.all()


class TestPanels:
    def test_hvg_panel_size_and_indices(self):
        a = fake_adata()
        idx = targets.hvg_panel(a, n_top=10)
        assert len(idx) == 10 and idx.max() < a.n_vars

    def test_leiden_marker_panel_is_a_union_of_per_cluster_topk(self):
        a = fake_adata(n_cells=80)
        idx, labels, prov = targets.leiden_marker_panel(a, top_k=5)
        assert len(labels) == a.n_obs and prov["n_clusters"] >= 1
        assert len(idx) <= prov["n_clusters"] * 5 and len(idx) > 0

    def test_wilcoxon_panel_cache_round_trips_and_key_covers_every_parameter(self, tmp_path,
                                                                             monkeypatch):
        """The disk cache must return the same panel, and must not be able to serve a stale one."""
        monkeypatch.setattr(targets, "CACHE_DIR", tmp_path / "tcache")
        a = fake_adata(n_cells=80)
        i1, l1, p1 = targets.leiden_marker_panel_wilcoxon(a, top_k=5)
        assert not p1.get("from_cache"), "first call must compute"
        assert list(tmp_path.glob("tcache/*.npz")), "first call must leave a cache entry"

        i2, l2, p2 = targets.leiden_marker_panel_wilcoxon(a, top_k=5)
        assert p2.get("from_cache") is True, "second call must be served from disk"
        assert np.array_equal(i1, i2) and np.array_equal(l1, l2)
        assert p1["n_clusters"] == p2["n_clusters"]

        i3, _, _ = targets.leiden_marker_panel_wilcoxon(a, top_k=2)
        assert len(list(tmp_path.glob("tcache/*.npz"))) == 2
        assert len(i3) <= len(i1)

        with pytest.raises(TypeError, match="not in the cache key"):
            targets._panel_key(panel="x", digest="d", some_new_knob=1)

    def test_gmt_roundtrip_and_module_scores_align(self, tmp_path):
        a = fake_adata()
        gmt = tmp_path / "m.gmt"
        gmt.write_text("MOD_A\turl\t" + "\t".join(f"Gene{i}" for i in range(20)) + "\n"
                       "MOD_TINY\turl\tGene0\tGene1\tGene2\n")
        sets = targets.read_gmt(gmt)
        assert set(sets) == {"MOD_A", "MOD_TINY"}
        cells = [str(n).split("-")[0] for n in a.obs_names[:10]]
        ts = targets.module_scores(a, sets, cells)
        assert ts.gene_names == ["MOD_A"], "a module below tmin present genes is skipped"
        assert ts.provenance["tmin"] == 15, "the gene-set floor is 15"
        assert ts.provenance["scorer"] == "decoupler.mt.aucell"
        assert ts.matrix.shape == (10, 1) and ts.has_counts.all()


class TestProbes:
    def test_binary_probe_detects_planted_signal_and_baseline_is_chance(self):
        rng = np.random.default_rng(0)
        n, d, g = 200, 8, 6
        emb = rng.normal(size=(n, d))
        y = np.zeros((n, g), dtype=bool)
        for j in range(3):
            y[:, j] = emb[:, j] > 0
        y[:, 3:] = rng.random((n, 3)) < 0.5
        res = probes.binary_expression_probe(emb, y, np.ones(n, bool), folds=4, seeds=2)
        assert res.baseline == 0.5
        assert res.per_target[:3].mean() > 0.9, "planted genes must be recovered"
        assert abs(res.per_target[3:].mean() - 0.5) < 0.12, "noise genes must sit at chance"

    def test_module_probe_depth_residualisation_kills_a_depth_only_embedding(self):
        rng = np.random.default_rng(1)
        n = 300
        depth = rng.gamma(4, 1, size=n)
        emb = np.stack([depth, depth**2, rng.normal(size=n)], axis=1)
        scores = np.stack([2 * depth + rng.normal(0, 0.1, n)], axis=1)
        raw = probes.module_activity_probe(emb, scores, np.ones(n, bool), folds=4, seeds=1)
        resid = probes.module_activity_probe(emb, scores, np.ones(n, bool), depth=depth, folds=4, seeds=1)
        assert raw.macro > 0.9, "without residualisation depth masquerades as signal"
        assert abs(resid.macro) < 0.25, "residualised, a depth-only embedding must score ~0"
        assert resid.metric.endswith("depth_residualised")

    def test_masked_cells_are_excluded_from_fitting(self):
        rng = np.random.default_rng(2)
        emb = rng.normal(size=(50, 4))
        y = rng.random((50, 3)) < 0.5
        has = np.ones(50, bool)
        has[40:] = False
        res = probes.binary_expression_probe(emb, y, has, folds=3, seeds=1)
        assert res.n_cells == 40


class TestAnnotationProbe:
    def test_annotation_probe_parallel_matches_sequential(self):
        """Parallelising the fold fits must not move a single number."""
        rng = np.random.default_rng(3)
        y = np.array(["A"] * 40 + ["B"] * 35 + ["C"] * 25)
        x = rng.normal(size=(100, 6)) + 4.0 * np.eye(6)[np.searchsorted(["A", "B", "C"], y)]
        one = probes.annotation_probe(x, y, seeds=2, n_jobs=1)
        many = probes.annotation_probe(x, y, seeds=2, n_jobs=-1)
        assert set(one) == set(many)
        for name in one:
            assert one[name].macro == pytest.approx(many[name].macro, abs=1e-12), name
            assert one[name].baseline == pytest.approx(many[name].baseline, abs=1e-12), name
            assert one[name].seed_sd == pytest.approx(many[name].seed_sd, abs=1e-12), name

    def test_separable_embedding_aces_and_noise_sits_at_baseline(self):
        rng = np.random.default_rng(0)
        y = np.array(["A"] * 60 + ["B"] * 40 + ["C"] * 20)
        x = rng.normal(size=(120, 8)) + 6.0 * np.eye(8)[np.searchsorted(["A", "B", "C"], y)]
        res = probes.annotation_probe(x, y, seeds=2)
        assert set(res) == {"linear", "knn_cosine", "mlp"}
        for name, r in res.items():
            assert r.macro > 0.95, f"{name} must ace a separable embedding"
            assert r.n_targets == 3
            assert r.baseline < 0.35
            assert r.seed_sd >= 0.0 and r.seed_sd < 0.2
        noise = probes.annotation_probe(rng.normal(size=(120, 8)), y, seeds=2)
        assert noise["linear"].macro < 0.55
        assert noise["mlp"].macro < 0.60, "the MLP must not manufacture signal from noise"

    def test_unlabelled_cells_are_excluded_never_guessed(self):
        rng = np.random.default_rng(1)
        y = np.array(["A"] * 30 + ["B"] * 30 + [""] * 40)
        x = rng.normal(size=(100, 4))
        res = probes.annotation_probe(x, y, seeds=1)
        assert res["linear"].n_cells == 60


def test_the_linear_probe_is_scale_robust():
    """The linear probe must read an embedding whose dimensions have wildly different scales."""
    rng = np.random.default_rng(5)
    y = np.array(["A"] * 40 + ["B"] * 40 + ["C"] * 40)
    base = rng.normal(size=(120, 6)) + 5.0 * np.eye(6)[np.searchsorted(["A", "B", "C"], y)]
    scaled = base * np.array([1e-4, 1e4, 1.0, 1e-3, 1e3, 1.0])
    res = probes.annotation_probe(scaled, y, seeds=1)
    assert res["linear"].macro > 0.9, f"scale-sensitive linear probe: {res['linear'].macro:.3f}"


class TestGradedExpression:
    def test_graded_targets_are_log_normalised_not_raw_counts(self):
        a = fake_adata(n_cells=40, n_genes=30)
        cells = [str(n).split("-")[0] for n in a.obs_names]
        ts = targets.expression_targets(a, cells, np.arange(5), panel="g")
        assert ts.matrix.dtype == np.float32 and ts.matrix.shape == (40, 5)
        assert ts.matrix.max() < 12.0, "log1p CP10K values, not raw counts"
        assert ts.provenance["transform"].startswith("normalize_total")

    def test_regression_probe_recovers_a_planted_signal_and_depth_variant_kills_a_depth_only_embedding(self):
        rng = np.random.default_rng(11)
        n = 120
        depth = rng.uniform(1000, 20000, size=n)
        signal = rng.normal(size=(n, 3))
        y = signal * 2.0 + np.log1p(depth)[:, None] * 0.5
        has = np.ones(n, dtype=bool)
        informed = probes.expression_regression_probe(signal, y, has, seeds=1)
        assert informed.macro > 0.7, informed.macro
        depth_only = np.log1p(depth).reshape(-1, 1)
        raw = probes.expression_regression_probe(depth_only, y, has, seeds=1)
        residualised = probes.expression_regression_probe(depth_only, y, has, depth=depth, seeds=1)
        assert raw.macro > residualised.macro + 0.05, f"raw {raw.macro:.3f} vs res {residualised.macro:.3f}"
        assert residualised.macro < 0.15, f"depth-residualised must kill it: {residualised.macro:.3f}"


def test_count_probe_recovers_planted_signal():
    """Counts generated from the embedding with a depth offset must be predictable from it, and an
    unrelated embedding must not be.
    """
    rng = np.random.default_rng(0)
    n, g, d = 500, 12, 8
    z = rng.normal(size=(n, d))
    w = rng.normal(scale=0.6, size=(d, g))
    depth = rng.integers(2000, 20000, n).astype(float)
    expo = depth / depth.mean()
    counts = rng.poisson(np.exp(z @ w - 1.5) * expo[:, None]).astype(float)

    planted = probes.count_regression_probe(z, counts, np.ones(n, bool), depth)
    assert planted.macro > 0.3, f"probe cannot see a planted count signal: {planted.macro:.4f}"

    noise = probes.count_regression_probe(
        rng.normal(size=(n, d)), counts, np.ones(n, bool), depth
    )
    assert noise.macro < planted.macro, "an unrelated embedding must not score like the real one"


def test_nb_probe_recovers_planted_signal_in_overdispersed_counts():
    """The NB family must recover a planted signal in genuinely overdispersed counts."""
    rng = np.random.default_rng(1)
    n, g, d = 300, 6, 6
    z = rng.normal(size=(n, d))
    w = rng.normal(scale=0.6, size=(d, g))
    depth = rng.integers(3000, 15000, n).astype(float)
    expo = depth / depth.mean()
    rate = np.exp(z @ w - 1.0) * expo[:, None]
    counts = rng.poisson(rate * rng.gamma(shape=2.0, scale=0.5, size=rate.shape)).astype(float)
    assert counts.var(axis=0).mean() > counts.mean(axis=0).mean(), "fixture is not overdispersed"

    res = probes.count_regression_probe(z, counts, np.ones(n, bool), depth, family="nb")
    assert res.macro > 0.15, f"NB probe cannot see a planted signal: {res.macro:.4f}"


def test_seed_sd_measures_the_macro_not_the_targets():
    """The scale bar must describe the number that is published."""
    import numpy as np
    from scrm_eval.probes import _seed_sd

    rng = np.random.default_rng(0)
    base = rng.normal(0.7, 0.20, size=400)
    per_seed = np.stack([base + rng.normal(0, 0.001, size=400) for _ in range(3)])
    sd = _seed_sd(per_seed)
    assert sd < 0.01, "the macro is stable across seeds even though targets vary widely"
    assert sd < per_seed.std(axis=0).mean() * 5 or sd < 0.01

    shifted = np.stack([np.full(400, 0.5), np.full(400, 0.6), np.full(400, 0.7)])
    assert abs(_seed_sd(shifted) - 0.1) < 1e-9

    assert np.isnan(_seed_sd(np.full((1, 10), 0.5)))
    assert np.isnan(_seed_sd(np.full((3, 10), np.nan)))


def test_binary_probe_reports_a_seed_sd():
    import numpy as np
    from scrm_eval.probes import binary_expression_probe

    rng = np.random.default_rng(5)
    n = 240
    x = rng.normal(size=(n, 6))
    y = (x[:, [0, 1, 2]] + rng.normal(0, 0.5, size=(n, 3)) > 0).astype(np.int8)
    r = binary_expression_probe(x, y, np.ones(n, dtype=bool))
    assert np.isfinite(r.seed_sd), "3 seeds must yield a spread"
    assert 0 <= r.seed_sd < 0.1, f"implausible spread {r.seed_sd}"
    assert r.n_scored == 3


def test_a_target_that_cannot_be_scored_is_not_averaged_in_as_a_zero():
    """A refusal to measure must not read as a measurement of zero, and its denominator must show."""
    rng = np.random.default_rng(3)
    n = 150
    sig = rng.normal(size=(n, 2))
    x = np.column_stack([sig, rng.normal(0, 0.05, size=(n, 2))])

    scores = np.column_stack([sig[:, 0], sig[:, 1], np.full(n, 2.5)])
    has = np.ones(n, dtype=bool)
    res = probes.module_activity_probe(x, scores, has)

    assert res.n_targets == 3, "the panel size is still the panel size"
    assert res.n_scored == 2, "the denominator of the macro must be reported, not implied"
    assert not np.isfinite(res.per_target[2]), "an unscorable module is NaN, never a real 0.0"
    assert res.macro == pytest.approx(float(np.nanmean(res.per_target[:2])))
    assert res.macro > 0.8, (
        f"macro {res.macro:.4f} -- an unscorable module was averaged in as a measured zero, "
        f"dragging the headline below the mean over the modules that were actually scored"
    )

    graded = np.column_stack([sig[:, 0], np.full(n, 1.0)])
    e6b = probes.expression_regression_probe(x, graded, has)
    assert e6b.n_targets == 2 and e6b.n_scored == 1
    assert not np.isfinite(e6b.per_target[1])


def test_the_panel_cache_cannot_serve_a_panel_built_on_a_different_leiden_graph():
    """The clustering the marker panels are DERIVED from is part of what defines them."""
    a = fake_adata(n_cells=80)
    _, _, first = targets.leiden_marker_panel_wilcoxon(a, top_k=5)
    assert not first.get("from_cache")

    real = dict(targets.CLUSTER_GRAPH)
    try:
        targets.CLUSTER_GRAPH["n_neighbors"] = 10
        _, _, second = targets.leiden_marker_panel_wilcoxon(a, top_k=5)
    finally:
        targets.CLUSTER_GRAPH.clear()
        targets.CLUSTER_GRAPH.update(real)
    assert not second.get("from_cache"), (
        "a changed Leiden graph was served the panel built by the previous one"
    )

    assert "graph" in targets._PANEL_KEY_FIELDS
    with pytest.raises(TypeError, match="not in the cache key"):
        targets._panel_key(panel="x", digest="d", graph=targets.CLUSTER_GRAPH, another_knob=1)
