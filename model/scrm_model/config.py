"""`ArchConfig` — the architecture's numbers, with their invariants as assertions."""

from dataclasses import asdict, dataclass

SEQ_LEN = 16384

SEQ_LEN_OPTIONS = (16384, 32768, 49152)
MAX_UMI_LEN = 1600
MIN_UMI_LEN = 60
MAX_PATCHES = (SEQ_LEN + 1) // (MIN_UMI_LEN + 1)

VOCAB = 8
MASK_ID = 7
N_ID = 4
TAB_ID = 5
NEWLINE_ID = 6
N_BASES = 4


@dataclass(frozen=True)
class ArchConfig:
    """One arm of `arms()`. The dataclass defaults are the 13.6 M shape, and every live arm
    overrides the global trunk down from there. A default's VALUE is inside `arch_sha`, so changing
    one moves every published arm's digest.
    """

    n_layers_local: int = 4
    dim_local: int = 96
    n_heads_local: int = 4
    n_layers_global: int = 6
    dim_global: int = 384
    n_heads_global: int = 6
    dim_contig: int | None = None
    multiple_of_local: int = 256
    multiple_of_global: int = 256
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    proj_hidden: int = 1024
    proj_dim: int = 384
    attn_impl: str = "flex_attention"
    rope_max_pos: int = MAX_UMI_LEN
    checkpoint_local: bool = True
    pool_variant: str = "attention"
    p_bucket: int = 32

    def __post_init__(self) -> None:
        assert self.pool_variant in ("attention", "mean_late", "mean_all"), self.pool_variant
        assert self.dim_local % self.n_heads_local == 0
        assert self.dim_global % self.n_heads_global == 0
        assert self.head_dim_local % 2 == 0, "RoPE needs an even local head_dim"
        assert self.head_dim_global % 2 == 0, "RoPE needs an even global head_dim"
        assert MAX_UMI_LEN * 10 <= SEQ_LEN, "the packer's geometry margin (11.1)"
        assert 2 * self.dim_global <= 4 * self.proj_dim <= 4 * self.dim_global, (
            f"the regulariser's sketch width is out of band (invariant 14): K = 4*proj_dim = "
            f"{4 * self.proj_dim} against the cell embedding's D = dim_global = {self.dim_global}, "
            f"i.e. K/D = {4 * self.proj_dim / self.dim_global:.2f}, outside 2 <= K/D <= 4. "
            f"proj_dim must be in [{self.dim_global // 2}, {self.dim_global}] "
            "(the S-ladder's convention is proj_dim == dim_global)."
        )

    @property
    def contig_out(self) -> int:
        """Width the contig pool emits, i.e. the width the UMI pool aggregates over."""
        return self.dim_contig if self.dim_contig is not None else self.dim_local

    @property
    def head_dim_local(self) -> int:
        return self.dim_local // self.n_heads_local

    @property
    def head_dim_global(self) -> int:
        return self.dim_global // self.n_heads_global


def arch_dict(cfg: ArchConfig, arm: str) -> dict:
    """The checkpoint's `arch` dict — THE ONE construction, used by train.py and run_eval.py."""
    d = {**asdict(cfg), "arm": arm}
    if cfg.pool_variant == "attention":
        del d["pool_variant"]
    return d


def arms() -> dict[str, ArchConfig]:
    """THE ARCHITECTURES THIS PROJECT MAINTAINS, and nothing else.

    Every entry must have a run declared on it in `registry.ABLATION_DELTAS`, and
    `test_no_hardcoded_lists` asserts the two sets are equal in BOTH directions: an architecture
    nobody trains is one `train.py --arm` will still offer. `S96ML`/`S96MA` are S96 with mean
    pooling in the two late pools and in all four; they exist for C7, and they were removed once
    already when their runs were -- which is why they come back only alongside those runs.
    """
    return {
        "S96": ArchConfig(
            n_layers_global=6,
            dim_global=96,
            n_heads_global=4,
            multiple_of_global=8,
            proj_hidden=256,
            proj_dim=96,
        ),
        "S96ML": ArchConfig(
            n_layers_global=6,
            dim_global=96,
            n_heads_global=4,
            multiple_of_global=8,
            proj_hidden=256,
            proj_dim=96,
            pool_variant="mean_late",
        ),
        "S96MA": ArchConfig(
            n_layers_global=6,
            dim_global=96,
            n_heads_global=4,
            multiple_of_global=8,
            proj_hidden=256,
            proj_dim=96,
            pool_variant="mean_all",
        ),
    }


FLAGSHIP_RECIPE: dict = {
    "arm": "S96",
    "regularizer": "sigreg",
    "lamb": 0.0196,
    "seed": 20260817,
    "train_cells": 3000,
    "train_frac": 1.0,
    "steps": 8947,
    "corpora": ("mouse_brain_5k", "neuron_1k_v3", "neurons_5k", "sod1_spinal_cord"),
}

#: Recipe fields that are RECORDED but are not arguments. "regularizer" joined them when the
#: --regularizer flag went: a flag with one choice tunes nothing.
FLAGSHIP_RECORDED_ONLY = ("steps", "corpora", "regularizer")

assert FLAGSHIP_RECIPE["arm"] in arms(), FLAGSHIP_RECIPE["arm"]
assert all(k in FLAGSHIP_RECIPE for k in FLAGSHIP_RECORDED_ONLY), FLAGSHIP_RECORDED_ONLY
