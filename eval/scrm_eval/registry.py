"""THE registry of trained runs and evaluation datasets. One declaration, many consumers."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

ROOT_ENV = "SCRM_PROJ"

#: The value `PROJ` takes when the data root is not named. It is deliberately the UNEXPANDED shell
#: literal, so any path derived from it is recognisable on sight as "nobody named a data root"
#: rather than looking like a real directory that happens to be empty. Import stays harmless --
#: `eval/tests` runs with no data root and must keep doing so -- but "no data root" and "no data"
#: are two answers, and `root_unset()` / `require_root()` below are how a reader tells them apart.
UNSET_ROOT = "$SCRM_PROJ"

PROJ = Path(os.environ.get(ROOT_ENV, UNSET_ROOT))

#: The one sentence every refusal about a missing data root prints. It lives here, beside the
#: declaration, so no consumer has to restate the variable's name.
ROOT_HINT = (
    f"{ROOT_ENV} names no data root, so every path below it is a path under a directory literally "
    f"called {UNSET_ROOT!r}, which never exists. Nothing was missing: nothing was looked at. Set "
    f"{ROOT_ENV} to the data root (corpora, runs, eval_artifacts, eval_reports)."
)


#: The paper's LaTeX macro namespace. `math_commands.tex` declares `\\scrmInput` and
#: `\\scrmMissing`, and every section resolves `\\scrm...` against them. Declared here because
#: three producers emit macros and a prefix repeated per producer is a prefix that can disagree.
TEX_NS = "scrm"


def root_unset() -> bool:
    """Is the data root unnamed, read at CALL time? Then an empty read means nothing was looked at."""
    return not os.environ.get(ROOT_ENV)


def require_root(what: str) -> Path:
    """The data root, or a refusal naming `what` needed it. For readers whose emptiness is a FAILURE."""
    if root_unset():
        raise FileNotFoundError(f"{what}: {ROOT_HINT}")
    return Path(os.environ[ROOT_ENV])


_ROOT_ENV = ("SCRM_CORPUS_ROOT", "CORPORA_ROOT", "CORPUS_ROOT")
CORPUS_ROOT = Path(next((v for k in _ROOT_ENV if (v := os.environ.get(k))), "") or PROJ / "corpus")


def corpus_root() -> Path:
    """The corpus root, read at CALL time rather than import time."""
    env = next((v for k in _ROOT_ENV if (v := os.environ.get(k))), None)
    return Path(env) if env else Path(os.environ.get(ROOT_ENV, PROJ)) / "corpus"

REQUIRED_RUNS: tuple[tuple[str, str, str], ...] = (
    ('armS96-c3k-sigreg-lam002-s20260903', 'S96',
     'scrm S-96 (reported) · 3k mouse cells · SIGReg λ=0.0196'),
    ('armS96-sc_mixology_5cl', 'S96',
     'scrm S-96 · sc_mixology_5cl only · in-domain, not a held-out result'),
    ('armS96-c3k-sigreg-lam002', 'S96',
     'scrm S-96 · SIGReg λ=0.0196 · reseed 20260817 (seed floor)'),
)

OPTIONAL_RUNS: tuple[tuple[str, str, str], ...] = (
    ('armS96-c3k-sigreg-lam002-s20260902', 'S96',
     'scrm S-96 · reseed 20260902 (seed floor)'),
    ('armS96-untrained-i1', 'S96',
     'scrm S-96 · UNTRAINED, random init 1 (C2 control)'),
    ('armS96-untrained-i2', 'S96',
     'scrm S-96 · UNTRAINED, random init 2 (C2 control)'),
    ('armS96-untrained-i3', 'S96',
     'scrm S-96 · UNTRAINED, random init 3 (C2 control)'),
    ('armS96-c3k-sigreg-lam002-permuted', 'S96',
     'scrm S-96 · TRAINED ON PERMUTED MOLECULES, depth held (C3 control)'),
    ('armS96ML-c3k-sigreg-lam002', 'S96ML',
     'scrm S-96 · window+view pools are means (C7)'),
    ('armS96MA-c3k-sigreg-lam002', 'S96MA',
     'scrm S-96 · ALL FOUR pools are means (C7)'),
)

STEPS_PER_PASS = 8_947

SIGREG_TOKEN_PASSES = 3

FLAGSHIP = "armS96-c3k-sigreg-lam002-s20260903"


def seed_base(name: str) -> str:
    """A run name with its reseed suffix removed, i.e. the recipe the seeds are seeds OF."""
    base, sep, tail = name.rpartition("-s")
    return base if sep and tail.isdigit() and len(tail) == 8 else name


def reported_arm(available) -> str | None:
    """THE one arm every published number comes from, or None if it is not present."""
    return FLAGSHIP if FLAGSHIP in available else None

#: The seeds the floor is measured over. Every member must be trained by the same code at the
#: same world size: a set that mixes either measures that difference, not seed spread, and this is
#: the threshold every ablation verdict is judged against. `measurements/seed_floor.py --check`
#: prints 0.0023 over these three, which the rule rounds up to the pin below.
SEED_FLOOR_SET = ("armS96-c3k-sigreg-lam002",
                  "armS96-c3k-sigreg-lam002-s20260902",
                  "armS96-c3k-sigreg-lam002-s20260903")
#: MEASURED, and RATIFIED by `measurements/seed_floor.py --check`: the range of E6 detection AUROC
#: (markers_wilcoxon_gt1, mean over the 9 eval datasets) across the three members of
#: SEED_FLOOR_SET, which the rule rounds up. Do not narrow the set to tighten the floor: a floor
#: over two seeds is not a tighter floor, it is a less-informed one, and a two-seed value let two
#: ablation rows read as effects.
SEED_FLOOR = 0.0024

FLAGSHIP_RECIPE: dict = {
    "arm": "S96",
    "regularizer": "sigreg",
    "lamb": 0.0196,
    "seed": 20260903,
    "steps": 8947,
    "train_cells": 3000,
    "train_frac": 1.0,
    "corpora": ("mouse_brain_5k", "neuron_1k_v3", "neurons_5k", "sod1_spinal_cord"),
}

ABLATION_DELTAS: dict[str, dict] = {
    FLAGSHIP: {},
    "armS96-c3k-sigreg-lam002": {"seed": 20260817},

    "armS96-c3k-sigreg-lam002-s20260902": {"seed": 20260902},

    "armS96-untrained-i1": {"untrained": 1, "seed": 20260901},
    "armS96-untrained-i2": {"untrained": 2, "seed": 20260902},
    "armS96-untrained-i3": {"untrained": 3, "seed": 20260903},
    "armS96-c3k-sigreg-lam002-permuted": {
        "corpora": tuple(f"{d}_permuted" for d in FLAGSHIP_RECIPE["corpora"]),
        "seed": 20260817},

    "armS96ML-c3k-sigreg-lam002": {"arm": "S96ML", "seed": 20260817},
    "armS96MA-c3k-sigreg-lam002": {"arm": "S96MA", "seed": 20260817},

    "armS96-sc_mixology_5cl": {"corpora": ("sc_mixology_5cl",), "seed": 20260817},
}


def recipe(run: str) -> dict:
    """The full training recipe for `run`: the flagship's, with that run's declared delta applied."""
    if run not in ABLATION_DELTAS:
        raise KeyError(
            f"{run!r} has no declared recipe. Add it to ABLATION_DELTAS saying what it changes about "
            f"the flagship, or it cannot be produced. Known: {sorted(ABLATION_DELTAS)}"
        )
    recipe_ = {**FLAGSHIP_RECIPE, **ABLATION_DELTAS[run]}
    if "untrained" in recipe_:
        for k in ("regularizer", "lamb", "train_cells", "train_frac", "corpora"):
            recipe_.pop(k, None)
        recipe_["steps"] = 0
    return recipe_


TRAINING: dict[str, tuple[int, int, str, str]] = {
    "armS96-c3k-sigreg-lam002": (
        1_182_720, 8_947, "3k cells, SIGReg lambda=0.0196, dim_global 96 (no expansion)", "none"),
    "armS96-c3k-sigreg-lam002-s20260903": ( 1_182_720, 8_947, "flagship recipe, seed 20260903 (reseed 2 of 2)", "none"),
    "armS96-sc_mixology_5cl": (
        1_182_720, 8_947,
        "sc_mixology_5cl ONLY — AN EVALUATION DATASET; the flagship's recipe with the corpus AND "
        "the seed changed (20260817, not the flagship's 20260903) — see this run's ABLATION_DELTAS "
        "entry, which is what results_page publishes as \\scrmSpecDelta",
        "IN-DOMAIN on sc_mixology_5cl; not a held-out result there"),

    "armS96-c3k-sigreg-lam002-s20260902": (
        1_182_720, 8_947, "flagship recipe, seed 20260902 (reseed 1 of 2)", "none"),

    "armS96-untrained-i1": (
        1_182_720, 0,
        "NONE — random init 1, the flagship's arch/readout/arch_sha verbatim with regenerated weights", "none"),
    "armS96-untrained-i2": ( 1_182_720, 0, "NONE — random init 2", "none"),
    "armS96-untrained-i3": ( 1_182_720, 0, "NONE — random init 3", "none"),
    "armS96-c3k-sigreg-lam002-permuted": (
        1_182_720, 8_947,
        "the four training corpora PERMUTED across cells, per-cell molecule counts held exactly (C3)", "none"),
    "armS96ML-c3k-sigreg-lam002": (
        1_164_096, 8_947,
        "flagship recipe, S96ML: the window and view pools are MEANS (C7) — 18,624 fewer parameters, "
        "which a mean pool cannot avoid", "none"),
    "armS96MA-c3k-sigreg-lam002": (
        1_145_472, 8_947,
        "flagship recipe, S96MA: ALL FOUR pools are means (C7) — 37,248 fewer parameters", "none"),
}

NOT_A_RESULT: frozenset[str] = frozenset({
    "smoke_armA",
    "armS192-mixedfull-v4",
    "armS128-fly_antenna",
})


def is_result(name: str) -> bool:
    """Is this run directory a result the pipeline should act on?"""
    return name not in NOT_A_RESULT and "." not in name


CONTROL_ARMS: tuple[str, ...] = (
    "armS96-untrained-i1",
    "armS96-untrained-i2",
    "armS96-untrained-i3",
    "armS96-c3k-sigreg-lam002-permuted",
)


def is_control(name: str) -> bool:
    """Is this arm a negative control, whose inputs or weights are deliberately wrong?"""
    if name in CONTROL_ARMS:
        return True
    rec = TRAINING.get(name)
    return bool(rec) and rec[1] == 0

DATASETS: tuple[tuple[str, str, str], ...] = (

    ("pbmc_1k_v2", "human", "eval"),
    ("nuclei_2k", "mouse", "eval"),
    ("tm_10X_P7_2", "mouse", "eval"),
    ("heart_10k_v3", "mouse", "eval"),
    ("sc_mixology_5cl", "human", "eval"),
    ("neuron_10k_v3", "mouse", "eval"),
    ("pbmc_10k_v3", "human", "eval"),
    ("got_ET01", "human", "eval"),
    ("fly_antenna", "fly", "eval"),

    ("mouse_brain_5k", "mouse", "train"),
    ("neuron_1k_v3", "mouse", "train"),
    ("neurons_5k", "mouse", "train"),
    ("sod1_spinal_cord", "mouse", "train"),

    ("mouse_brain_5k_permuted", "mouse", "train"),
    ("neuron_1k_v3_permuted", "mouse", "train"),
    ("neurons_5k_permuted", "mouse", "train"),
    ("sod1_spinal_cord_permuted", "mouse", "train"),

    #: MACAQUE AND RAT ARE NOT TRAINED ON. No arm's recipe names either corpus -- every arm
    #: trains on the four mouse corpora, on sc_mixology_5cl, or on the permuted set -- so
    #: role="train" asserted something false, and because datasets("eval") excluded them the
    #: embed fleet never covered them: Figure 2 plotted scRM on both from artifacts that were
    #: later deleted, and figure_species.py could not regenerate it (KeyError, no scRM row).
    #: "eval-pending" is the truthful role -- registered and scored, not an evaluation COLUMN --
    #: which keeps datasets("eval") at nine so no table gains a dataset.
    ("2500_rhesus_monkey_pbmcs_singleplex_3p_gem_x", "macaque", "eval-pending"),
    ("2500_wistar_rat_pbmcs_singleplex_3p_gem_x", "rat", "eval-pending"),

    ("crispr_a549_5k", "human", "eval-pending"),
)


COUNT_MATRICES: dict[str, str] = {
    "2500_rhesus_monkey_pbmcs_singleplex_3p_gem_x":
        "raw/2500_rhesus_monkey_pbmcs_singleplex_3p_gem_x/filtered.h5",
    "2500_wistar_rat_pbmcs_singleplex_3p_gem_x":
        "raw/2500_wistar_rat_pbmcs_singleplex_3p_gem_x/filtered.h5",
    "fly_antenna": "raw/fly_antenna/antenna_raw_counts_FCA43_sample1.h5ad",
    "got_ET01": "raw/got_mpn/ET01/matrix",
    "crispr_a549_5k": "raw/crispr_a549_5k/sample_feature_bc_matrix",
    "mouse_brain_5k": "whitelists/mouse_brain_5k/filtered_feature_bc_matrix",
    "neurons_5k": "whitelists/neurons_5k/sample_feature_bc_matrix",
    "sod1_spinal_cord": "whitelists/sod1_spinal_cord/sample_filtered_feature_bc_matrix",
    "neuron_1k_v3": "raw/neuron_1k_v3_matrix/filtered_feature_bc_matrix",
    "nuclei_2k": "raw/nuclei_2k_barcodes/mm10",
    "sc_mixology_5cl": "raw/sc_mixology_meta/sc_mixology_5cl_matrix",
    "tm_10X_P7_2": "raw/tabula_muris/10X_P7_2_matrix",
    "heart_10k_v3": "raw/heart_10k_v3/tenx_matrix",
    "neuron_10k_v3": "raw/neuron_10k_v3/tenx_matrix",
    "pbmc_1k_v2": "raw/pbmc_1k_v2/tenx_matrix",
    "pbmc_10k_v3": "raw/pbmc_10k_v3/tenx_matrix",
}


def _proj() -> Path:
    """`PROJ` resolved at CALL time rather than import time."""
    return Path(os.environ.get(ROOT_ENV, PROJ))


def count_matrices() -> dict[str, Path]:
    """Every registered count matrix as an absolute path, rooted at `PROJ` as it is right now."""
    root = _proj()
    return {ds: root / rel for ds, rel in COUNT_MATRICES.items()}


def count_matrix(dataset: str) -> Path:
    """One dataset's count matrix, or `KeyError` naming what is registered."""
    if dataset not in COUNT_MATRICES:
        raise KeyError(
            f"{dataset} has no registered count matrix; {len(COUNT_MATRICES)} datasets do "
            f"({', '.join(sorted(COUNT_MATRICES))}). Register it in registry.COUNT_MATRICES -- "
            "one table, read by both targets.load_counts and controls._read_counts."
        )
    return _proj() / COUNT_MATRICES[dataset]


def runs(required_only: bool = False) -> tuple[str, ...]:
    src = REQUIRED_RUNS if required_only else REQUIRED_RUNS + OPTIONAL_RUNS
    return tuple(r for r, _, _ in src)


def arms_for(names=None) -> tuple[str, ...]:
    """The ArchConfig arm per run, positionally aligned with `runs()`."""
    table = {r: a for r, a, _ in REQUIRED_RUNS + OPTIONAL_RUNS}
    return tuple(table[n] for n in (names or runs()))


def labels() -> dict[str, str]:
    return {r: lab for r, _, lab in REQUIRED_RUNS + OPTIONAL_RUNS}


def datasets(role: str = "eval") -> tuple[str, ...]:
    return tuple(d for d, _, r in DATASETS if r == role)


def species() -> dict[str, str]:
    return {d: s for d, s, _ in DATASETS}


def discover_runs() -> set[str]:
    """Run directories on disk that hold a checkpoint, minus the ones that are not results."""
    root = PROJ / "runs"
    if not root.is_dir():
        return set()
    return {p.name for p in root.iterdir()
            if (p / "latest.pt").is_file() and is_result(p.name)}


_MIXED_SPECIES_ARMS: tuple[str, ...] = tuple(
    r[0] for r in REQUIRED_RUNS + OPTIONAL_RUNS if "mixed56" in r[0] or "mixedfull" in r[0]
)


IN_DOMAIN: dict[str, tuple[str, ...]] = {
    "sc_mixology_5cl": ("armS96-sc_mixology_5cl",),
}


def in_domain_arms(dataset: str) -> tuple[str, ...]:
    """Arms that trained on `dataset` and therefore cannot be a held-out result on it."""
    return IN_DOMAIN.get(dataset, ())


REPORT_PANEL = "markers_wilcoxon_gt1"

SPEC_PANEL = REPORT_PANEL

REPORT_PANELS: dict[str, tuple[str, ...]] = {
    "binary_expression": (REPORT_PANEL,),
    "module_activity": ("raw", "depth_residualised"),
}


def spec_dataset() -> str:
    """The dataset the specialisation table reports, derived from `IN_DOMAIN`."""
    ds = [d for d, arms in IN_DOMAIN.items() if arms]
    if len(ds) != 1:
        raise KeyError(f"the specialisation table reports ONE in-domain dataset; IN_DOMAIN has {ds}. "
                       "Name it explicitly at every call site before registering a second.")
    return ds[0]


REPORTED_CKPT: dict[str, str] = {
    "armS96-c3k-sigreg-lam002": "reported-step8947.pt",
    "armS96-c3k-sigreg-lam002-s20260903": "reported-step8947.pt",
    "armS96-sc_mixology_5cl": "reported-step8947.pt",
}


def reported_ckpt(run: str) -> str:
    """The checkpoint filename `run` is reported at. `latest.pt` unless declared otherwise."""
    return REPORTED_CKPT.get(run, "latest.pt")


CKPT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "armS96-sc_mixology_5cl": ("pass1.pt",),
}


def ckpt_candidates(run: str) -> tuple[str, ...]:
    """Candidate checkpoints that must be measured against `run`'s reported one. Usually none."""
    return CKPT_CANDIDATES.get(run, ())


class StaleRequirement(Exception):
    """A cached `w_eval` was measured on a molecule definition the corpus no longer declares."""


LABEL_FILES: dict[str, tuple[str, bool]] = {
    "sc_mixology_5cl": ("raw/sc_mixology_meta/sc_mixology_5cl_labels_sng.json", False),
    "tm_10X_P7_2": ("raw/tabula_muris/10X_P7_2_labels.json", False),
    "fly_antenna": ("raw/fly_antenna_meta/fly_antenna_labels_broad.json", False),
    "2500_rhesus_monkey_pbmcs_singleplex_3p_gem_x":
        ("raw/2500_rhesus_monkey_pbmcs_singleplex_3p_gem_x/marker_labels.json", True),
    "2500_wistar_rat_pbmcs_singleplex_3p_gem_x":
        ("raw/2500_wistar_rat_pbmcs_singleplex_3p_gem_x/marker_labels.json", True),
    "pbmc_10k_v3": ("raw/pbmc_10k_v3/marker_labels_shared4.json", True),
}


def label_file(dataset: str):
    """The independent-label file for `dataset`, or None when it has none."""
    e = LABEL_FILES.get(dataset)
    return None if e is None else PROJ / e[0]


def labelled_datasets() -> tuple[str, ...]:
    """Every dataset with an independent label file, in declaration order."""
    return tuple(LABEL_FILES)


def marker_derived() -> tuple[str, ...]:
    """The datasets whose labels are Leiden-on-counts rather than an author annotation."""
    return tuple(d for d, (_, m) in LABEL_FILES.items() if m)


WINDOWS_REQUIRED = "eval_artifacts/windows_required.json"


def windows_required() -> dict:
    """The whole requirement cache, read at call time so `SCRM_PROJ` moves it."""
    import json as _json

    path = require_root(WINDOWS_REQUIRED) / WINDOWS_REQUIRED
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. It is written by `measurements/windows_required.py --write` "
            f"(model/jobs/windows_required.sbatch) and every coverage decision reads it."
        )
    return _json.loads(path.read_text())


def full_coverage(dataset: str) -> int:
    """`w_eval` for reading ALL of `dataset`'s reads, REFUSED unless it names the corpus on disk."""
    req = windows_required()
    e = req.get(dataset)
    if not e or "w_eval" not in e:
        raise KeyError(
            f"{dataset} has no windows_required entry, so its all-reads requirement is unknown. "
            f"Run measurements/windows_required.py --write against {corpus_root()}."
        )
    man = corpus_root() / dataset / "manifest.json"
    if man.is_file():
        import json as _json

        on_disk = _json.loads(man.read_text())["declared"]["config"]["bubble_resolution"]
        cached = e.get("corpus_molecule_rule")
        if cached != on_disk:
            raise StaleRequirement(
                f"{dataset}'s cached w_eval={e['w_eval']} was measured on molecule definition "
                f"{cached!r}, but the corpus on disk declares {on_disk!r}. The requirement is a "
                f"function of tokens per cell, so it does not survive that change -- and because "
                f"the coverage gate compares an artifact against this same file, a stale entry "
                f"passes by agreeing with itself. Re-measure with "
                f"`REPIN=1 sbatch model/jobs/windows_required.sbatch` (fleet drained), then re-embed."
            )
    return int(e["w_eval"])


def arms_with_full_coverage(dataset: str) -> tuple[str, ...]:
    """Every scrm arm holding a FULL-COVERAGE artifact for `dataset`, found on disk."""
    import json as _json

    import pyarrow as pa
    from pyarrow import ipc

    need = full_coverage(dataset)
    out: set[str] = set()
    for f in sorted((PROJ / "eval_runs").glob(f"*/{dataset}.*.arrow")):
        # THE STAMP KEY AND THE PRODUCER SET COME FROM artifact, NOT FROM LITERALS HERE. Written
        # as `md[b"scrm_eval_stamp"]` inside a bare `except Exception: continue`, a rename made
        # the KeyError indistinguishable from "not a model artifact": this returned () for every
        # dataset, silently, against 147 artifacts that all carried the previous key. An artifact
        # whose stamp cannot be parsed is reported; one that is simply not ours is skipped.
        from scrm_eval import artifact as _art
        with pa.memory_map(str(f)) as src:
            md = ipc.open_file(src).schema.metadata or {}
        raw = md.get(_art.STAMP_KEY)
        if raw is None:
            continue
        try:
            st = _json.loads(raw)
        except ValueError as exc:
            raise ValueError(f"{f} carries an unparseable {_art.STAMP_KEY.decode()}: {exc}") from exc
        if st.get("producer") not in _art.MODEL_PRODUCERS or (st.get("w_eval") or 0) < need:
            continue
        out.add(f.parent.name)
    return tuple(sorted(out))


def discover_corpora() -> set[str]:
    root = corpus_root()
    if not root.is_dir():
        return set()
    return {p.name for p in root.iterdir() if (p / "manifest.json").is_file()}


def baselines() -> tuple[str, ...]:
    """Every count-side representation: the producers, then the frozen count-FM panel."""
    from . import controls

    return tuple(controls.PRODUCERS) + tuple(controls.PANEL)


def main(argv=None) -> int:
    """Emit one name per line, so a bash consumer can `mapfile` it instead of keeping a copy."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", action="store_true", help="all runs, required then optional")
    p.add_argument("--required", action="store_true", help="runs a scorer must wait for")
    p.add_argument("--optional", action="store_true", help="runs included only if present")
    p.add_argument("--arms", action="store_true", help="ArchConfig arm per run, aligned with --runs")
    p.add_argument("--datasets", action="store_true", help="evaluation datasets, cheapest first")
    p.add_argument("--all-datasets", action="store_true", help="every registered dataset, any role")
    p.add_argument("--panel", action="store_true", help="count foundation models (controls.PANEL)")
    p.add_argument("--producers", action="store_true",
                   help="count control producers (controls.PRODUCERS)")
    p.add_argument("--baselines", action="store_true",
                   help="every count-side model: producers then panel")
    p.add_argument("--flagship", action="store_true",
                   help="the declared FLAGSHIP arm; for launchers that would otherwise name one")
    p.add_argument("--label-file", metavar="DATASET",
                   help="path to DATASET's independent cell labels; exits 1 when it has none, so a "
                        "launcher can tell 'no ground truth independent of the counts' from a typo")
    p.add_argument("--labelled-datasets", action="store_true",
                   help="every dataset with an independent label file")
    p.add_argument("--recipe", metavar="RUN",
                   help="shell KEY=VALUE lines giving RUN's full training recipe")
    p.add_argument("--ckpt", metavar="RUN",
                   help="the checkpoint filename RUN is REPORTED at (latest.pt unless measured "
                        "otherwise); this is what eval_matrix treats as publishable")
    p.add_argument("--role", default="eval")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    if a.label_file:
        if a.label_file not in LABEL_FILES:
            if a.label_file not in {d for d, _, _ in DATASETS}:
                print(f"{a.label_file} is not a registered dataset", file=sys.stderr)
            return 1
        print(label_file(a.label_file))
        return 0
    if a.labelled_datasets:
        print("\n".join(labelled_datasets()))
        return 0
    if a.ckpt:
        if a.ckpt not in set(runs()):
            print(f"{a.ckpt} is not a registered run", file=sys.stderr)
            return 1
        print(reported_ckpt(a.ckpt))
        return 0
    if a.recipe:
        r = recipe(a.recipe)
        for k, v in sorted(r.items()):
            raw = " ".join(map(str, v)) if isinstance(v, tuple) else str(v)
            print(f"{k.upper()}={shlex.quote(raw)}")
        return 0
    if a.json:
        print(json.dumps({"required": list(runs(True)), "optional": [r for r, _, _ in OPTIONAL_RUNS],
                          "labels": labels(), "datasets": {r: list(datasets(r))
                                                           for r in ("eval", "train", "finetune")}},
                         indent=2))
        return 0
    out: tuple[str, ...] = ()
    if a.flagship:
        out = (FLAGSHIP,)
    elif a.panel or a.producers or a.baselines:
        from . import controls
        if a.panel:
            out = tuple(controls.PANEL)
        elif a.producers:
            out = tuple(controls.PRODUCERS)
        else:
            out = tuple(controls.PRODUCERS) + tuple(controls.PANEL)
    elif a.all_datasets:
        out = tuple(d for d, _, _ in DATASETS)
    elif a.required:
        out = runs(True)
    elif a.optional:
        out = tuple(r for r, _, _ in OPTIONAL_RUNS)
    elif a.arms:
        out = arms_for()
    elif a.datasets:
        out = datasets(a.role)
    elif a.runs:
        out = runs()
    else:
        p.error("pick one of --runs/--required/--optional/--arms/--datasets/--panel/--producers/"
                "--baselines/--flagship/--json")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
