"""The registry is the only list of runs and datasets; no consumer may keep its own copy."""

from __future__ import annotations

import contextlib
import io
import json
import re
import sys
from pathlib import Path

import pytest

from proj_guard import needs_proj

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "eval"), str(REPO / "measurements")]

from scrm_eval import registry  # noqa: E402

PROJ = registry.PROJ


def _bash_array(text: str, name: str) -> list[str] | None:
    """Extract `NAME=(a b c)` from a bash script, or None if it does not declare one."""
    m = re.search(rf"^{name}=\((.*?)\)\s*$", text, re.M | re.S)
    if not m:
        return None
    body = m.group(1)
    if "$" in body or "@" in body:
        return None
    return body.split()


FROZEN = {
    "coverage_sweep.sbatch", "heldout_baselines.sbatch", "merge_shards.sbatch",
}


def _active_scripts() -> list[Path]:
    return sorted(p for p in (REPO / "model" / "jobs").glob("*.sbatch") if p.name not in FROZEN)


def test_every_trained_run_on_disk_is_registered():
    """Every trained run on disk must be named by the registry."""
    on_disk = registry.discover_runs()
    if not on_disk:
        pytest.skip("no runs/ on this machine")
    missing = sorted(on_disk - set(registry.runs()))
    assert not missing, (
        f"{len(missing)} trained runs on disk that the registry does not name: {missing[:6]}"
        f"{' ...' if len(missing) > 6 else ''}. Either add each to REQUIRED_RUNS/OPTIONAL_RUNS "
        f"(OPTIONAL while it is still training or embedding, since a REQUIRED run with no artifact "
        f"blocks every scorer that waits for it), or delete the run directory."
    )


def test_every_corpus_on_disk_is_registered():
    """Every corpus on disk must be named by the registry, so it can be pinned."""
    root = registry.corpus_root()
    on_disk = registry.discover_corpora()
    if not on_disk:
        pytest.skip(f"no corpus tree under {root} on this machine")
    missing = sorted(on_disk - {d for d, _, _ in registry.DATASETS})
    assert not missing, (
        f"{len(missing)} corpora under {root} that the registry does not name: {missing}.\n"
        f"Either add each to registry.DATASETS with its species and role ('eval' puts it on the "
        f"page, 'train' keeps it off), or delete the directory. A corpus the registry does not "
        f"name cannot be pinned, so nothing can embed it."
    )


def test_the_corpus_guard_fires_on_an_unregistered_corpus_and_follows_the_resolved_root(monkeypatch,
                                                                                        tmp_path):
    """The corpus guard must fire on an unregistered corpus and must read the resolved root."""
    planted = "a_corpus_no_registry_names"
    assert planted not in {d for d, _, _ in registry.DATASETS}
    (tmp_path / planted / "shards").mkdir(parents=True)
    (tmp_path / planted / "manifest.json").write_text("{}")

    for var in ("SCRM_CORPUS_ROOT", "CORPORA_ROOT", "CORPUS_ROOT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SCRM_CORPUS_ROOT", str(tmp_path))

    assert registry.corpus_root() == tmp_path, "the resolver did not follow SCRM_CORPUS_ROOT"
    assert registry.discover_corpora() == {planted}, (
        "discover_corpora() did not read the resolved root; bound to a fixed path it would audit "
        "one tree while every other stage read another"
    )
    with pytest.raises(AssertionError, match=planted):
        test_every_corpus_on_disk_is_registered()


def test_registry_names_nothing_that_is_not_on_disk():
    """A registered run with no checkpoint on disk must be OPTIONAL."""
    on_disk = registry.discover_runs()
    if not on_disk:
        pytest.skip("no runs/ on this machine")
    ghosts = sorted(set(registry.runs(required_only=True)) - on_disk)
    assert not ghosts, (
        f"REQUIRED runs with no checkpoint on disk: {ghosts}. A scorer waits for every REQUIRED run, "
        f"so this configuration deadlocks -- move them to OPTIONAL until they have trained."
    )


@pytest.mark.parametrize("script", _active_scripts(), ids=lambda p: p.name)
def test_active_job_scripts_do_not_keep_their_own_list(script: Path):
    """An active job script must take its runs and datasets from the registry, not restate them."""
    text = script.read_text()
    for name in ("RUNS", "REQUIRED", "OPTIONAL", "DSETS"):
        got = _bash_array(text, name)
        if got is None:
            continue
        if name == "DSETS":
            expect = list(registry.datasets("eval"))
            hint = "$PY -m scrm_eval.registry --datasets"
        elif name == "REQUIRED":
            expect = list(registry.runs(required_only=True))
            hint = "$PY -m scrm_eval.registry --required"
        elif name == "OPTIONAL":
            expect = [r for r, _, _ in registry.OPTIONAL_RUNS]
            hint = "$PY -m scrm_eval.registry --optional"
        else:
            expect = list(registry.runs())
            hint = "$PY -m scrm_eval.registry --runs"
        assert got == expect, (
            f"{script.name} declares {name}= with its own copy of the list, and it has drifted.\n"
            f"  script:   {got}\n  registry: {expect}\n"
            f"Read it from the registry instead:  mapfile -t {name} < <({hint})"
        )


def test_every_dataset_has_a_recorded_source():
    """`model/jobs/corpora.tsv` and `registry.DATASETS` must name the same datasets, both ways."""
    tsv = REPO / "model" / "jobs" / "corpora.tsv"
    declared = {ln.split("\t")[0] for ln in tsv.read_text().splitlines()
                if ln and not ln.startswith("#") and "\t" in ln and not ln.startswith("key\t")}
    known = {d for d, _, _ in registry.DATASETS}
    assert declared == known, (
        f"corpora.tsv and registry.DATASETS disagree.\n"
        f"  acquirable but unknown: {sorted(declared - known)}\n"
        f"  known but unacquirable: {sorted(known - declared)}\n"
        f"A dataset in one and not the other is either a corpus nothing can describe or a name "
        f"nothing can build."
    )


def test_every_registered_run_has_a_training_entry():
    """Every registered run with a checkpoint must have a TRAINING entry."""
    trained = registry.discover_runs()
    missing = [r for r in registry.runs() if r in trained and r not in registry.TRAINING]
    assert not missing, (
        f"runs with a checkpoint on disk but no TRAINING entry: {missing}. Add (params, steps, data, "
        f"label exposure) for each -- params from the checkpoint's own tensors, steps from its `step` "
        f"field, never estimated."
    )


def test_results_page_datasets_and_arm_labels_come_from_the_registry():
    """`results_page.DS` and `results_page.ALL_ARMS` must come from the registry."""
    import results_page as R

    assert set(R.DS) == set(registry.datasets("eval")), (
        f"results_page.DS has drifted from the registry.\n"
        f"  page only:     {sorted(set(R.DS) - set(registry.datasets('eval')))}\n"
        f"  registry only: {sorted(set(registry.datasets('eval')) - set(R.DS))}"
    )
    page_runs = {r for r, _ in R.ALL_ARMS}
    assert page_runs == set(registry.runs()), (
        f"results_page.ALL_ARMS has drifted from the registry.\n"
        f"  page only:     {sorted(page_runs - set(registry.runs()))}\n"
        f"  registry only: {sorted(set(registry.runs()) - page_runs)}"
    )


@needs_proj
def test_every_eval_dataset_has_a_window_requirement():
    """Every evaluation dataset must have a pinned all-reads window requirement, or
    `full_coverage()` raises and the coverage gate cannot run.
    """
    bad = {}
    for d in registry.datasets("eval"):
        try:
            registry.full_coverage(d)
        except (KeyError, FileNotFoundError, registry.StaleRequirement) as exc:
            bad[d] = f"{type(exc).__name__}: {exc}"
    assert not bad, (
        "these evaluation datasets have no usable all-reads requirement, so the coverage gate "
        "cannot run on them and a capped embedding could reach a published table. Re-measure with "
        "`sbatch model/jobs/windows_required.sbatch` (it refuses to move an existing w_eval without "
        "REPIN=1):\n  " + "\n  ".join(f"{d}: {m}" for d, m in sorted(bad.items()))
    )


def test_registry_cli_returns_a_nonempty_list_for_every_flag():
    """`registry --<flag>` must emit a non-empty list for every flag."""
    for flag in ("--runs", "--required", "--optional", "--arms", "--datasets"):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = registry.main([flag])
        lines = [x for x in buf.getvalue().splitlines() if x.strip()]
        assert rc == 0 and lines, f"{flag} produced nothing"
    assert len(registry.arms_for()) == len(registry.runs())


def _cli_count(*flags: str) -> int:
    """How many names `registry --<flag>` emits, through the CLI the launchers actually call."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        registry.main(list(flags))
    return len([ln for ln in buf.getvalue().splitlines() if ln.strip()])


_MATRIX_BOUNDS = {
    "eval_matrix.sbatch": (("--runs",), ("--datasets",)),
    "baselines_matrix.sbatch": (("--panel", "+1"), ("--datasets",)),
}


def test_array_bounds_cover_the_registry_matrix():
    """Every launcher's `#SBATCH --array` bound must cover the registry matrix."""
    for name, dims in _MATRIX_BOUNDS.items():
        text = (REPO / "model" / "jobs" / name).read_text()
        m = re.search(r"^#SBATCH --array=0-(\d+)", text, re.M)
        assert m, f"{name} has no `#SBATCH --array=0-N` line to check"
        covers = int(m.group(1)) + 1
        need, shape = 1, []
        for dim in dims:
            n = _cli_count(dim[0]) + (1 if "+1" in dim else 0)
            need *= n
            shape.append(f"{n}({dim[0]}{'+1' if '+1' in dim else ''})")
        assert covers >= need, (
            f"{name}: #SBATCH --array=0-{m.group(1)} covers {covers} tasks but the registry needs "
            f"{' x '.join(shape)} = {need}. The runs past the end would be silently never "
            f"embedded. Set it to 0-{need - 1} or wider."
        )


def test_flagship_still_satisfies_its_rule():
    """The declared FLAGSHIP must still be the smallest arm within tolerance of the best."""
    reports = sorted((PROJ / "eval_artifacts").glob("*.allmodels.probes.json")) if PROJ.exists() else []
    if not reports:
        pytest.skip("no probes reports on this machine")
    per: dict[str, list[float]] = {}
    for f in reports:
        d = json.loads(f.read_text())
        ds = d["dataset"]
        for key, art in d.get("artifacts", {}).items():
            if not key.startswith(f"{ds}.") or not key.endswith(".allreads"):
                continue
            run = key[len(ds) + 1: -len(".allreads")]
            macro = art.get("binary_expression", {}).get("hvg2000", {}).get("macro")
            if macro is not None:
                per.setdefault(run, []).append(float(macro))
    common = None
    scored: dict[str, set[str]] = {}
    for f in reports:
        d = json.loads(f.read_text())
        ds = d["dataset"]
        for key in d.get("artifacts", {}):
            if key.startswith(f"{ds}.") and key.endswith(".allreads"):
                scored.setdefault(key[len(ds) + 1: -len(".allreads")], set()).add(ds)
    flag = registry.FLAGSHIP
    if flag not in scored:
        pytest.skip(f"{flag} has no probes rows on this machine")
    common = scored[flag]
    in_domain = {a for arms_ in registry.IN_DOMAIN.values() for a in arms_}
    ref_params = registry.TRAINING[flag][0]
    arms = {r: ds for r, ds in scored.items()
            if r in registry.TRAINING and common <= ds and r not in in_domain
            and not registry.is_control(r) and registry.TRAINING[r][0] == ref_params}
    excluded_arch = {r for r in scored
                     if r in registry.TRAINING and common <= scored[r]
                     and registry.TRAINING[r][0] != ref_params}

    def mean_on_common(run: str) -> float:
        vals = []
        for f in reports:
            d = json.loads(f.read_text())
            ds = d["dataset"]
            if ds not in common:
                continue
            art = d.get("artifacts", {}).get(f"{ds}.{run}.allreads")
            if art:
                m = art.get("binary_expression", {}).get("hvg2000", {}).get("macro")
                if m is not None:
                    vals.append(float(m))
        return sum(vals) / len(vals) if vals else float("nan")

    means = {r: mean_on_common(r) for r in arms}
    means = {r: v for r, v in means.items() if v == v}
    best = max(means.values())

    def beats(run: str, other: str) -> bool:
        """Whether `run` beats `other` by more than the seed floor."""
        per_ds = []
        for f_ in reports:
            d_ = json.loads(f_.read_text())
            ds_ = d_["dataset"]
            if ds_ not in common:
                continue
            a_ = d_.get("artifacts", {}).get(f"{ds_}.{run}.allreads")
            b_ = d_.get("artifacts", {}).get(f"{ds_}.{other}.allreads")
            if not a_ or not b_:
                continue
            x = a_.get("binary_expression", {}).get("hvg2000", {}).get("macro")
            y = b_.get("binary_expression", {}).get("hvg2000", {}).get("macro")
            if x is not None and y is not None:
                per_ds.append(x - y)
        if not per_ds:
            return False
        delta = sum(per_ds) / len(per_ds)
        npos = sum(1 for v in per_ds if v > 0)
        clears = delta > registry.SEED_FLOOR
        unanimous = npos in (0, len(per_ds))
        if clears and not unanimous:
            print(f"[flagship rule] {run} leads {other} by {delta:+.4f} (floor "
                  f"{registry.SEED_FLOOR}) on {npos}/{len(per_ds)} datasets -- clears the floor but "
                  "the sign is not unanimous, so the study's rule does not call it a real difference")
        return clears and unanimous

    best_run = max(means, key=means.get)
    within = {r for r in means if not beats(best_run, r)}
    params = {r: registry.TRAINING[r][0] for r in within if registry.TRAINING[r][0]}
    assert flag in within, (
        f"FLAGSHIP {flag} scores {means.get(flag):.4f} against the best {best:.4f} on "
        f"{len(common)} shared datasets -- outside the {registry.SEED_FLOOR} seed floor. The rule is "
        f"'smallest arm within tolerance of the best'; re-ratify or fix the declaration."
    )
    for r in sorted(excluded_arch):
        n = registry.TRAINING[r][0]
        real = beats(flag, r) or beats(r, flag)
        print(f"[flagship rule] {r} is NOT a flagship candidate: {n:,} params "
              f"({(n - ref_params) / ref_params:+.1%}) is a different architecture, not a capacity "
              f"variant. Against {flag} the study's rule calls the difference "
              f"{'REAL' if real else 'not real'}.")
    smallest = min(params, key=params.get)
    assert params[flag] <= params[smallest], (
        f"FLAGSHIP {flag} has {params[flag]:,} params but {smallest} is within tolerance at "
        f"{params[smallest]:,}. The rule selects the smallest arm within tolerance."
    )
    assert set(params.values()) == {ref_params}, (
        f"the flagship-candidate pool now holds more than one parameter count ({sorted(set(params.values()))}), "
        f"so the size half of the rule is live again. The parameter-match filter above must "
        f"select the same design architecture at any width, not the same parameter count."
    )


def test_no_measurement_keeps_its_own_copy_of_the_seed_set():
    """No measurement may keep its own copy of the seed set."""
    import re

    repo = Path(__file__).resolve().parents[2]
    reseed = re.compile(r"[\"'](arm[A-Za-z0-9_.-]+-s\d{8})[\"']")
    offenders = []
    for f in sorted((repo / "measurements").glob("*.py")):
        for line_no, line in enumerate(f.read_text().splitlines(), 1):
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            names = reseed.findall(line)
            if len(names) >= 2:
                offenders.append(f"{f.name}:{line_no} assigns {names}")
    assert not offenders, (
        "a measurement keeps its own copy of a reseed set instead of reading "
        "registry.SEED_FLOOR_SET:\n  " + "\n  ".join(offenders))


def test_seed_floor_set_is_the_flagship_s_own_reseeds():
    """The floor's attribution must name the arm every delta is measured against."""
    assert len(registry.SEED_FLOOR_SET) >= 2, "a spread needs at least two seeds"
    assert registry.FLAGSHIP in registry.SEED_FLOOR_SET, (
        f"SEED_FLOOR_SET {registry.SEED_FLOOR_SET} does not contain the flagship "
        f"{registry.FLAGSHIP}, so the noise bar is attributed to an arm no delta uses")
    base = registry.seed_base(registry.FLAGSHIP)
    for name in registry.SEED_FLOOR_SET:
        assert registry.seed_base(name) == base, (
            f"{name} is not a reseed of {base}, the flagship's recipe; a spread across "
            "different recipes is not a seed spread")
    for name in registry.SEED_FLOOR_SET:
        assert name in set(registry.runs()), f"{name} is not a registered run"


def test_full_coverage_refuses_a_requirement_measured_on_another_molecule_definition(tmp_path,
                                                                                     monkeypatch):
    """A cached `w_eval` whose corpus is not the corpus on disk must RAISE, not be returned."""
    import json

    from scrm_eval import registry

    ds = registry.datasets("eval")[0]
    root = tmp_path / "corpus"
    (root / ds).mkdir(parents=True)
    proj = tmp_path / "proj"
    (proj / "eval_artifacts").mkdir(parents=True)
    # The data root is named through the ENVIRONMENT, the same channel this test already uses for
    # the corpus root two lines down. `registry.PROJ` is the import-time snapshot of that variable;
    # patching the attribute said "the root is here" through a channel the call-time resolvers do
    # not read, so this test passed only while SCRM_PROJ happened to be unset -- and `check.sh`
    # runs with it SET, against the real requirement cache, where the stale entry it plants was
    # never read at all and the refusal it asserts could not fire.
    monkeypatch.setenv(registry.ROOT_ENV, str(proj))
    monkeypatch.setenv("SCRM_CORPUS_ROOT", str(root))

    def write(rule_on_disk, rule_cached):
        (root / ds / "manifest.json").write_text(
            json.dumps({"declared": {"config": {"bubble_resolution": rule_on_disk}}}))
        (proj / registry.WINDOWS_REQUIRED).write_text(json.dumps(
            {ds: {"w_eval": 1736, "corpus_molecule_rule": rule_cached}}))

    current = "likelihood-q10-pre45-post40-omega7-walk2-consume-tip-anchored-rejoin-flank-seeded"

    write(current, "weighted_split")
    with pytest.raises(registry.StaleRequirement) as e:
        registry.full_coverage(ds)
    msg = str(e.value)
    assert "weighted_split" in msg and current in msg, (
        "the refusal must name BOTH definitions -- the one measured and the one on disk -- or the "
        "operator has to diff a manifest against a cache by hand"
    )
    assert "1736" in msg, "and the value it is refusing, so the log says what was nearly used"

    write(current, None)
    with pytest.raises(registry.StaleRequirement):
        registry.full_coverage(ds)

    write(current, current)
    assert registry.full_coverage(ds) == 1736

    (proj / registry.WINDOWS_REQUIRED).write_text(json.dumps({}))
    with pytest.raises(KeyError):
        registry.full_coverage(ds)


@needs_proj
def test_the_label_table_is_declared_once():
    """No consumer keeps its own dataset -> label-file table."""
    import sys

    sys.path.insert(0, str(REPO / "measurements"))
    from scrm_eval import registry

    declared = {d: registry.label_file(d) for d in registry.labelled_datasets()}
    assert declared, "the registry declares no label files at all"

    import uce_protocols

    assert uce_protocols.LABELS == declared, (
        "uce_protocols.LABELS has drifted from registry.LABEL_FILES; derive it rather than listing it"
    )
    assert tuple(uce_protocols.MARKER_DERIVED) == registry.marker_derived(), (
        "uce_protocols.MARKER_DERIVED has drifted; the authority is declared beside the path"
    )

    launcher = (REPO / "model" / "jobs" / "annotate_one.sbatch").read_text()
    assert "--label-file" in launcher, (
        "annotate_one.sbatch must resolve labels through `registry --label-file`, not a `case`"
    )
    for path, _ in registry.LABEL_FILES.values():
        assert path not in launcher, f"annotate_one.sbatch still hardcodes the label path {path}"

    missing = [d for d, p in declared.items() if not p.is_file()]
    assert not missing, f"declared label files that are not on disk: {missing}"


def test_the_reported_checkpoint_is_declared_once_and_the_launcher_reads_it():
    """`latest.pt` is the reporting RULE, and it must live in exactly one place."""
    for run in registry.runs():
        assert registry.reported_ckpt(run).endswith(".pt"), run
    unknown = set(registry.REPORTED_CKPT) - set(registry.runs())
    assert not unknown, f"REPORTED_CKPT names runs the registry does not declare: {sorted(unknown)}"

    raw = (REPO / "model" / "jobs" / "eval_matrix.sbatch").read_text()
    assert "registry --ckpt" in raw, (
        "eval_matrix.sbatch must ask the registry which checkpoint is publishable for a run"
    )
    code = "\n".join(l for l in raw.splitlines()
                     if not l.lstrip().startswith("#") or l.lstrip().startswith("#SBATCH"))
    assert "CKPT:-latest.pt" not in code, (
        "eval_matrix.sbatch still defaults CKPT to a literal, so the reporting rule is stated twice"
    )


def test_the_specialisation_panel_is_the_one_the_scorer_builds():
    """`SPEC_PANEL` names a panel `run_probes` actually produces, and one in-domain dataset."""
    scorer = (REPO / "eval" / "scrm_eval" / "run_probes.py").read_text()
    assert f'panels["{registry.SPEC_PANEL}"]' in scorer, (
        f"run_probes does not build a panel named {registry.SPEC_PANEL}"
    )
    assert registry.spec_dataset() in registry.datasets("eval")
    assert registry.in_domain_arms(registry.spec_dataset()), (
        "the specialisation dataset must have a declared in-domain arm"
    )


def test_a_declared_candidate_checkpoint_exists_and_is_not_the_reported_one():
    """Every declared candidate checkpoint must exist and must not be the reported one."""
    for run, cands in registry.CKPT_CANDIDATES.items():
        assert run in registry.runs(), f"CKPT_CANDIDATES names an undeclared run: {run}"
        assert cands, f"{run} declares an empty candidate tuple; drop the key instead"
        for c in cands:
            assert c.endswith(".pt"), f"{run}: {c} is not a checkpoint filename"
            assert c != registry.reported_ckpt(run), (
                f"{run} declares its own reported checkpoint {c} as a candidate"
            )
    assert registry.ckpt_candidates(registry.FLAGSHIP) == registry.CKPT_CANDIDATES.get(
        registry.FLAGSHIP, ()
    )


def test_a_reported_checkpoint_is_never_a_filename_training_can_move():
    """`latest.pt` is not a checkpoint identity for an arm that has been resumed."""
    runs = registry.PROJ / "runs"
    if not runs.is_dir():
        pytest.skip("no runs directory on this machine")
    bad = []
    for run in registry.runs():
        d = runs / run
        if not d.is_dir():
            continue
        resumed = any(f.name.startswith("reported-step") for f in d.glob("*.pt"))
        if resumed and registry.reported_ckpt(run) == "latest.pt":
            bad.append(f"{run}: has a preserved reported-step*.pt but is still reported at latest.pt")
    assert not bad, (
        "these arms were resumed and their reported checkpoint is a moving filename: "
        + "; ".join(bad)
        + ". Declare the stable filename in REPORTED_CKPT."
    )


def test_the_reported_arm_is_a_name_and_never_the_best_of_several_seeds():
    """`reported_arm` must not improve when a better-scoring sibling appears beside the flagship."""
    flag = registry.FLAGSHIP
    reseeds = [a for a in registry.SEED_FLOOR_SET if a != flag]
    assert reseeds, "the flagship has no reseed, so this test asserts nothing"
    assert registry.reported_arm({flag: 0.10}) == flag
    beside = {flag: 0.10, **{r: 0.99 for r in reseeds}}
    assert registry.reported_arm(beside) == flag, (
        "reported_arm picked a reseed that scored higher; the rule is a name, not a maximum"
    )
    assert registry.reported_arm({r: 0.99 for r in reseeds}) is None
