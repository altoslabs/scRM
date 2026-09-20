"""The metrics against embeddings whose answers are known by construction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scrm_eval import metrics  # noqa: E402


def planted_subspace(n: int, dim: int, rank: int, *, seed: int = 0) -> np.ndarray:
    """`n` points spanning exactly `rank` directions of `dim`-space, isotropic within the subspace."""
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.normal(size=(dim, rank)))[0]
    coeff = rng.normal(size=(n, rank))
    return coeff @ basis.T


def blobs(n_per: int, k: int, dim: int, sep: float, *, seed: int = 0):
    """`k` unit-variance Gaussian blobs whose centres are `sep` apart along distinct axes."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(k):
        centre = np.zeros(dim)
        centre[c % dim] = sep * (c + 1)
        xs.append(rng.normal(size=(n_per, dim)) + centre)
        ys.append(np.full(n_per, c))
    return np.vstack(xs), np.concatenate(ys)


@pytest.mark.parametrize("rank", [1, 2, 5, 17, 64])
def test_effective_rank_recovers_a_planted_subspace(rank):
    """`effective_rank` must equal the planted rank on points isotropic within a rank-k subspace,
    where every retained direction carries equal weight and the entropy is log(k).
    """
    x = planted_subspace(2000, 384, rank)
    got = metrics.effective_rank(x)
    assert got == pytest.approx(rank, rel=0.06), f"rank {rank} planted, measured {got:.2f}"


def test_effective_rank_is_monotone_in_planted_rank():
    """`effective_rank` must increase strictly with the planted rank across a series of subspaces."""
    got = [metrics.effective_rank(planted_subspace(2000, 384, r)) for r in (2, 4, 8, 16, 32, 64)]
    assert all(a < b for a, b in zip(got, got[1:])), got


def test_isotropic_noise_uses_every_dimension():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 384))
    got = metrics.effective_rank(x)
    assert got > 0.9 * 384, f"isotropic noise should approach min(n, d)=384, got {got:.1f}"


def test_total_collapse_is_exactly_one():
    """`effective_rank` must be exactly 1.0, not NaN, when every cell maps to the same point."""
    x = np.tile(np.arange(384, dtype=float), (500, 1))
    assert metrics.effective_rank(x) == 1.0


def test_near_collapse_approaches_one_from_above():
    """`effective_rank` must approach 1 from above on one strong direction plus faint noise."""
    rng = np.random.default_rng(1)
    direction = rng.normal(size=384)
    x = np.outer(rng.normal(size=800), direction) + 1e-3 * rng.normal(size=(800, 384))
    got = metrics.effective_rank(x)
    assert 1.0 < got < 1.5, got


def test_effective_rank_ignores_a_translation():
    """`effective_rank` must be unchanged by a large mean vector added to a planted subspace."""
    x = planted_subspace(1500, 384, 8)
    shifted = x + 50.0 * np.ones(384)
    assert metrics.effective_rank(shifted) == pytest.approx(metrics.effective_rank(x), rel=1e-6)


def test_effective_rank_is_invariant_to_rotation_and_scale():
    """`effective_rank` must be invariant to rotation and to uniform scaling of a planted subspace."""
    rng = np.random.default_rng(2)
    x = planted_subspace(1200, 64, 9)
    q = np.linalg.qr(rng.normal(size=(64, 64)))[0]
    assert metrics.effective_rank(x @ q) == pytest.approx(metrics.effective_rank(x), rel=1e-6)
    assert metrics.effective_rank(7.5 * x) == pytest.approx(metrics.effective_rank(x), rel=1e-6)


def test_silhouette_rises_with_separation():
    xs = []
    for sep in (0.0, 1.0, 3.0, 10.0):
        x, y = blobs(120, 4, 32, sep)
        xs.append(metrics.silhouette(x, y))
    assert all(a < b for a, b in zip(xs, xs[1:])), xs
    assert xs[-1] > 0.5, f"well-separated blobs should score high, got {xs[-1]:.3f}"


def test_silhouette_of_labels_unrelated_to_the_geometry_is_near_zero():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(400, 32))
    y = rng.integers(0, 5, size=400)
    assert abs(metrics.silhouette(x, y)) < 0.1


def test_silhouette_is_nan_when_the_reference_has_one_cluster():
    """`silhouette` must return NaN, not raise, when the reference labels name a single cluster."""
    x, _ = blobs(50, 2, 16, 5.0)
    assert np.isnan(metrics.silhouette(x, np.zeros(100, dtype=int)))


def test_silhouette_rejects_a_label_length_mismatch():
    x, y = blobs(10, 2, 8, 1.0)
    with pytest.raises(ValueError, match="labels"):
        metrics.silhouette(x, y[:-1])


def test_identical_clusterings_score_one():
    y = np.repeat(np.arange(6), 40)
    got = metrics.cluster_agreement(y, y)
    assert got["ari"] == pytest.approx(1.0) and got["nmi"] == pytest.approx(1.0)


def test_agreement_decays_monotonically_as_labels_are_corrupted():
    """`cluster_agreement`'s ARI must fall at every step as a rising fraction of labels is corrupted."""
    rng = np.random.default_rng(4)
    ref = np.repeat(np.arange(8), 60)
    aris = []
    for frac in (0.0, 0.1, 0.25, 0.5, 0.9):
        found = ref.copy()
        idx = rng.choice(len(ref), size=int(frac * len(ref)), replace=False)
        found[idx] = rng.integers(0, 8, size=idx.size)
        aris.append(metrics.cluster_agreement(found, ref)["ari"])
    assert all(a > b for a, b in zip(aris, aris[1:])), aris
    assert aris[-1] < 0.1, f"near-total corruption should approach chance, got {aris[-1]:.3f}"


def test_a_single_cluster_for_everything_scores_zero_not_high():
    """`cluster_agreement`'s ARI must be 0 on a clustering that gives every cell one label."""
    ref = np.repeat(np.arange(5), 50)
    got = metrics.cluster_agreement(np.zeros_like(ref), ref)
    assert abs(got["ari"]) < 1e-9


def test_shattering_the_data_is_punished_far_harder_by_ari_than_by_nmi():
    """ARI must collapse to ~0 while NMI stays at least 10x larger on a clustering that gives every
    cell its own label: ARI is chance-corrected and NMI, under sklearn's
    `average_method="arithmetic"`, is not.
    """
    ref = np.repeat(np.arange(5), 50)
    got = metrics.cluster_agreement(np.arange(ref.size), ref)
    assert got["ari"] < 0.05, got
    assert got["nmi"] > 10 * max(got["ari"], 1e-3), got


def test_the_gate_refuses_identical_embeddings():
    x = planted_subspace(300, 64, 5)
    ok, cos = metrics.sensitive(x, x.copy())
    assert not ok and cos == pytest.approx(1.0)


def test_the_gate_refuses_a_rescaled_embedding():
    """`sensitive` must refuse a comparison of an embedding against a uniformly rescaled copy."""
    x = planted_subspace(300, 64, 5)
    ok, cos = metrics.sensitive(x, 2.0 * x)
    assert not ok and cos == pytest.approx(1.0)


def test_the_gate_reports_no_cosine_when_no_cell_has_a_centred_norm():
    """`sensitive` must refuse and report no cosine when centring leaves every cell at zero norm."""
    collapsed = np.tile(np.arange(16, dtype=float), (200, 1))
    ok, cos = metrics.sensitive(collapsed, collapsed * 3.0 + 1.0)
    assert not ok, "the gate must still refuse the comparison"
    assert cos is None, "and must not invent a cosine where none exists"

    x = planted_subspace(300, 64, 5)
    assert metrics.sensitive(x, x.copy())[1] == pytest.approx(1.0)


def test_the_gate_admits_genuinely_different_embeddings():
    ok, cos = metrics.sensitive(planted_subspace(300, 64, 5, seed=0), planted_subspace(300, 64, 5, seed=1))
    assert ok and abs(cos) < 0.5


def test_the_archive_defect_would_have_been_caught():
    """`sensitive` must admit the comparison and report a near-zero cosine on two unrelated tensors,
    registering that the two are not the same tensor.
    """
    x = planted_subspace(500, 384, 20)
    rng = np.random.default_rng(5)
    y = rng.normal(size=x.shape)
    ok, cos = metrics.sensitive(x, y)
    assert ok, "two unrelated tensors must register as different"
    assert abs(cos) < 0.2, cos


def test_the_gate_sees_through_a_shared_mean_because_it_centres():
    """`sensitive` must admit a comparison of two embeddings sharing a large mean and differing only
    in a small residual, where both flattened and uncentred per-cell cosine read ~1.0.
    """
    base = np.ones((400, 64)) * 10.0
    a = base + planted_subspace(400, 64, 6, seed=0) * 0.05
    b = base + planted_subspace(400, 64, 6, seed=1) * 0.05

    flat = float((a.ravel() @ b.ravel()) / (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel())))
    raw_per_cell = float(
        (
            (a * b).sum(axis=1)
            / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        ).mean()
    )
    assert flat > 0.999, "flattened cosine is blind here"
    assert raw_per_cell > 0.999, "and so is UNCENTRED per-cell cosine -- the reason for centring"

    ok, cos = metrics.sensitive(a, b)
    assert ok, "after centring the gate must admit this comparison"
    assert abs(cos) < 0.2, cos


@pytest.mark.parametrize(
    "bad,match",
    [
        (np.zeros((3, 3, 3)), "2-D"),
        (np.zeros((1, 8)), "at least 2 cells"),
        (np.array([[1.0, np.nan], [0.0, 1.0]]), "NaN"),
        (np.array([[1.0, np.inf], [0.0, 1.0]]), "NaN or inf"),
    ],
)
def test_malformed_input_is_refused_with_a_reason(bad, match):
    with pytest.raises(ValueError, match=match):
        metrics.effective_rank(bad)


def test_purity_is_one_for_perfectly_separated_labels():
    x, y = blobs(60, 4, 32, 25.0)
    got = metrics.knn_label_purity(x, y, k=10)
    assert got["purity"] == pytest.approx(1.0)
    assert got["purity_adj"] == pytest.approx(1.0)


def test_purity_falls_to_chance_for_labels_unrelated_to_geometry():
    """`knn_label_purity` must give purity ~= chance and purity_adj ~= 0 on labels unrelated to
    the geometry.
    """
    rng = np.random.default_rng(11)
    x = rng.normal(size=(600, 32))
    y = rng.integers(0, 4, size=600)
    got = metrics.knn_label_purity(x, y, k=15)
    assert got["purity"] == pytest.approx(got["chance"], abs=0.06)
    assert abs(got["purity_adj"]) < 0.1


def test_purity_rises_monotonically_with_separation():
    got = [metrics.knn_label_purity(*blobs(80, 4, 32, sep), k=10)["purity_adj"] for sep in (0.0, 1.0, 3.0, 10.0)]
    assert all(a < b for a, b in zip(got, got[1:])), got


def test_the_chance_correction_makes_imbalanced_datasets_comparable():
    """`purity_adj` must stay ~0 where uncorrected purity exceeds 0.85, on an imbalanced label set
    carrying no geometric information.
    """
    rng = np.random.default_rng(12)
    x = rng.normal(size=(500, 16))
    y = np.zeros(500, dtype=int)
    y[:25] = 1
    got = metrics.knn_label_purity(x, y, k=10)
    assert got["purity"] > 0.85, got
    assert abs(got["purity_adj"]) < 0.15, "the corrected value must not inherit the imbalance"


def test_purity_needs_no_clustering_and_so_has_no_protocol_to_mismatch():
    """`knn_label_purity` must depend only on the embedding and the labels, returning the same value
    for two callers, on a construction where ARI moves materially with the clustering resolution.
    """
    x, y = blobs(50, 3, 16, 4.0)
    a = metrics.knn_label_purity(x, y, k=7)
    b = metrics.knn_label_purity(x, y, k=7)
    assert a == b
    coarse = metrics.cluster_agreement(metrics.cluster_labels(x, resolution=0.1, seed=0), y)["ari"]
    fine = metrics.cluster_agreement(metrics.cluster_labels(x, resolution=3.0, seed=0), y)["ari"]
    assert abs(coarse - fine) > 0.05, "resolution must move ARI materially"


def test_purity_rejects_a_label_length_mismatch():
    x, y = blobs(10, 2, 8, 1.0)
    with pytest.raises(ValueError, match="labels"):
        metrics.knn_label_purity(x, y[:-1])


def test_silhouette_is_circular_on_kmeans_labels_even_with_no_structure():
    """`silhouette` must score k-means labels above random labels on structureless noise."""
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(21)
    x = rng.normal(size=(600, 16))

    random_labels = rng.integers(0, 5, size=600)
    kmeans_labels = KMeans(n_clusters=5, n_init=10, random_state=0).fit_predict(x)

    s_random = metrics.silhouette(x, random_labels)
    s_kmeans = metrics.silhouette(x, kmeans_labels)
    assert abs(s_random) < 0.05, f"random labels on noise must score ~0, got {s_random:.3f}"
    assert s_kmeans > s_random + 0.05, (
        f"k-means labels on the SAME noise scored {s_kmeans:.3f} vs {s_random:.3f} for random"
    )


def test_purity_degrades_when_the_reference_splits_a_real_cluster():
    """`purity_adj` must fall when the reference over-splits every true cluster in two."""
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(22)
    true_k = 6
    x = np.vstack([rng.normal(size=(150, 24)) + 6.0 * np.eye(24)[c] for c in range(true_k)])

    at_truth = KMeans(n_clusters=true_k, n_init=10, random_state=0).fit_predict(x)
    over_split = KMeans(n_clusters=2 * true_k, n_init=10, random_state=0).fit_predict(x)

    p_truth = metrics.knn_label_purity(x, at_truth, k=10)["purity_adj"]
    p_split = metrics.knn_label_purity(x, over_split, k=10)["purity_adj"]
    assert p_truth > 0.9, f"at the true k purity should be near 1, got {p_truth:.3f}"
    assert p_split < p_truth - 0.1, (
        f"splitting every true cluster in two must cost purity: {p_split:.3f} vs {p_truth:.3f}"
    )


def test_projecting_an_embedding_that_contains_the_pcs_recovers_them():
    """`pc_regression` must recover the PCs from a 384-d embedding holding them under a random
    linear mix plus noise.
    """
    rng = np.random.default_rng(30)
    pcs = rng.normal(size=(600, 10))
    emb = pcs @ rng.normal(size=(10, 384)) + 0.05 * rng.normal(size=(600, 384))
    got = metrics.pc_regression(emb, pcs, folds=5)
    assert got["r2_variance_weighted"] > 0.98, got["r2_variance_weighted"]
    assert min(got["r2_per_pc"]) > 0.95


def test_projecting_an_unrelated_embedding_gives_zero_or_negative():
    """`pc_regression` must score ~0 out of sample for an unrelated embedding, where 384 features
    onto 10 targets would fit noise in sample.
    """
    rng = np.random.default_rng(31)
    got = metrics.pc_regression(rng.normal(size=(600, 384)), rng.normal(size=(600, 10)))
    assert got["r2_variance_weighted"] < 0.05, got


def test_the_per_pc_breakdown_says_WHICH_structure_was_captured():
    """`pc_regression`'s `r2_per_pc` must be high on the PCs an embedding carries and low on the
    PCs it does not.
    """
    rng = np.random.default_rng(32)
    pcs = rng.normal(size=(700, 10))
    emb = pcs[:, :3] @ rng.normal(size=(3, 384)) + 0.05 * rng.normal(size=(700, 384))
    r2 = metrics.pc_regression(emb, pcs)["r2_per_pc"]
    assert min(r2[:3]) > 0.95, r2
    assert max(r2[3:]) < 0.1, r2


def test_cka_is_invariant_to_the_transformations_that_carry_no_meaning():
    rng = np.random.default_rng(33)
    a = rng.normal(size=(300, 10))
    rot = np.linalg.qr(rng.normal(size=(10, 10)))[0]
    assert metrics.linear_cka(a, a) == pytest.approx(1.0)
    assert metrics.linear_cka(a, a @ rot) == pytest.approx(1.0, abs=1e-9)
    assert metrics.linear_cka(a, 5.0 * a) == pytest.approx(1.0, abs=1e-9)
    assert metrics.linear_cka(a, a + 100.0) == pytest.approx(1.0, abs=1e-9)


def test_cka_of_a_constant_representation_is_zero_not_nan():
    rng = np.random.default_rng(34)
    assert metrics.linear_cka(rng.normal(size=(200, 8)), np.ones((200, 8))) == 0.0


def test_neighbour_overlap_is_one_for_identical_and_chance_for_unrelated():
    rng = np.random.default_rng(35)
    a = rng.normal(size=(400, 16))
    assert metrics.neighbor_overlap(a, a, k=10) == pytest.approx(1.0)
    assert metrics.neighbor_overlap(a, rng.normal(size=(400, 16)), k=10) < 0.1


def test_overlap_and_cka_can_disagree_and_that_is_informative():
    """`neighbor_overlap` must stay high while `linear_cka` falls under a monotone non-linear
    reparameterisation, which preserves neighbourhoods and destroys the linear map.
    """
    rng = np.random.default_rng(36)
    base = rng.normal(size=(500, 6))
    warped = np.sign(base) * np.abs(base) ** 3
    assert metrics.neighbor_overlap(base, warped, k=15) > 0.35
    assert metrics.linear_cka(base, warped) < 0.9


def test_residualising_removes_exactly_the_nuisance_and_nothing_else():
    """`residualise` must make a target that is the nuisance unpredictable from it and leave a
    target independent of it unchanged.
    """
    rng = np.random.default_rng(40)
    depth = rng.normal(size=(500, 2))
    signal = rng.normal(size=(500, 3))
    targets = np.column_stack([depth @ rng.normal(size=(2, 2)), signal])

    before = metrics.pc_regression(depth, targets, folds=5)["r2_per_pc"]
    clean = metrics.residualise(targets, depth)
    after = metrics.pc_regression(depth, clean, folds=5)["r2_per_pc"]
    assert min(before[:2]) > 0.9, f"depth-driven targets should start predictable: {before[:2]}"
    assert max(after[:2]) < 0.1, f"depth-driven targets must collapse: {after[:2]}"
    for j in (2, 3, 4):
        keep = np.corrcoef(clean[:, j], targets[:, j])[0, 1]
        assert keep > 0.99, f"column {j} was altered: r={keep:.3f}"


def test_depth_encoding_alone_scores_well_before_residualising_and_not_after():
    """`pc_regression` must score a depth-only embedding materially against a PC space whose leading
    component is depth, and near zero once depth is residualised out.
    """
    rng = np.random.default_rng(41)
    depth = rng.normal(size=(600, 2))
    pcs = np.column_stack([
        depth @ np.array([[2.0], [1.5]]) + 0.3 * rng.normal(size=(600, 1)),
        rng.normal(size=(600, 4)),
    ])
    naive = metrics.pc_regression(depth, pcs)["r2_variance_weighted"]
    residual = metrics.pc_regression(depth, metrics.residualise(pcs, depth))["r2_variance_weighted"]
    assert naive > 0.2, f"depth should score materially before residualising, got {naive:.3f}"
    assert residual < 0.05, f"and near zero after, got {residual:.3f}"


def test_a_comparison_with_no_cosine_returns_None_not_nan():
    """`sensitive` must return `None`, not NaN, for the cosine between embeddings of different width."""
    ok, cos = metrics.sensitive(planted_subspace(200, 64, 5), planted_subspace(200, 16, 5))
    assert ok is True
    assert cos is None, "a cosine between different dimensionalities does not exist"
