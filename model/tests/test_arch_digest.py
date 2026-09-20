"""The arch digests of published arms are pinned."""

from __future__ import annotations

from scrm_model.config import arch_dict, arms

from scrm_eval.embed import arch_digest  # noqa: E402  (path set in conftest)

PINNED = {
    "S96": "7c4e4ffc9d304902eaaaeacc4ebac858edb48d5f5078d7f7245d71e7f7359bba",
}


def test_published_arm_digests_do_not_move():
    for arm, want in PINNED.items():
        got = arch_digest(arch_dict(arms()[arm], arm))
        assert got == want, f"{arm}: arch digest moved {want} -> {got}"


def test_default_pool_variant_is_stripped_and_nondefault_is_not():
    assert "pool_variant" not in arch_dict(arms()["S96"], "S96")
    for arm in ("S96ML", "S96MA"):
        assert arch_dict(arms()[arm], arm)["pool_variant"] == arms()[arm].pool_variant


def test_the_three_pool_arms_are_three_distinct_architectures():
    """A mean-pool arm must not hash to the flagship's, or C7 would compare a model with itself."""
    digests = {arm: arch_digest(arch_dict(arms()[arm], arm)) for arm in ("S96", "S96ML", "S96MA")}
    assert len(set(digests.values())) == 3, digests


PROJECTOR_CONVENTION_EXCEPTIONS: set[str] = set()


def test_the_projector_convention_names_its_exceptions():
    """`proj_dim` must equal `dim_global` for every live arm, with no exceptions."""
    from scrm_model.config import arms

    offenders = {name: (cfg.proj_dim, cfg.dim_global)
                 for name, cfg in arms().items() if cfg.proj_dim != cfg.dim_global}
    assert set(offenders) == PROJECTOR_CONVENTION_EXCEPTIONS, (
        f"projector geometry off convention: {offenders}. Set proj_dim = dim_global, and "
        "proj_hidden to the 8/3 ratio, ceil(4*dim_global*2/3) at multiple_of 8."
    )


def test_every_arm_satisfies_the_regularisers_sketch_width_band():
    """Every arm must satisfy `2*D <= K <= 4*D` on both tensors `reg` is applied to: the projector
    output and the cell embedding.
    """
    from scrm_model.config import arms

    for name, cfg in arms().items():
        ratio = 4 * cfg.proj_dim / cfg.dim_global
        assert 2.0 <= ratio <= 4.0, f"arm {name}: K/D = {ratio:.2f} on the cell embedding"
