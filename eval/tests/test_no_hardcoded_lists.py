"""No consumer may keep its own copy of a list the registry or controls already declares."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: Where a list is DECLARED, not copied. `panel.py` holds the baseline panel and imports
#: nothing, so `eval/countfm/embed_count_fm.py` -- which runs under the foundation models' own
#: interpreter and cannot import `controls` -- can read the declaration instead of keeping a
#: second copy, which is what this test refuses.
DECLARATION_SITES = {
    "eval/scrm_eval/controls.py",
    "eval/scrm_eval/panel.py",
    "eval/scrm_eval/registry.py",
    "measurements/ablation.py",
}

SCAN_SUFFIXES = (".py", ".sbatch", ".sh")

_DOCSTRING = re.compile(r'("""|\'\'\')(?:(?!\1).)*?\1', re.S)


def _blank(match: str) -> str:
    """`match` with every character replaced by a space, newlines kept, so line numbers survive."""
    return re.sub(r"[^\n]", " ", match)


def _scan_text(text: str, suffix: str) -> str:
    """The CODE of a source file: comments and triple-quoted spans blanked, everything else intact."""
    if suffix != ".py":
        return text
    try:
        import io
        import tokenize

        out = list(text)
        lines = text.splitlines(keepends=True)
        starts = [0]
        for ln in lines[:-1]:
            starts.append(starts[-1] + len(ln))
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            body = tok.string.lstrip("rRbBuUfF")
            triple = tok.type == tokenize.STRING and body[:3] in ('"""', "\'\'\'")
            if tok.type != tokenize.COMMENT and not triple:
                continue
            a = starts[tok.start[0] - 1] + tok.start[1]
            b = starts[tok.end[0] - 1] + tok.end[1]
            out[a:b] = _blank(text[a:b])
        return "".join(out)
    except (tokenize.TokenError, SyntaxError, IndentationError, IndexError):
        return _DOCSTRING.sub(lambda m: _blank(m.group(0)), text)


def _names() -> set[str]:
    import sys

    sys.path.insert(0, str(REPO / "eval"))
    from scrm_eval import controls

    return set(controls.PANEL) | set(controls.PRODUCERS)


def _dataset_names() -> set[str]:
    """Every dataset the registry declares, whatever its role."""
    import sys

    sys.path.insert(0, str(REPO / "eval"))
    from scrm_eval import registry

    return {d for d, _, _ in registry.DATASETS}


def _sources(bases=("eval", "model", "measurements", "crates")):
    for base in bases:
        for p in (REPO / base).rglob("*"):
            if p.suffix in SCAN_SUFFIXES and "target" not in p.parts:
                rel = str(p.relative_to(REPO))
                if rel not in DECLARATION_SITES:
                    yield rel, _scan_text(p.read_text(errors="replace"), p.suffix)


def _offending_literals(text: str, names: set[str]) -> list[str]:
    """Literal collections quoting two or more declared model names, comments excluded."""
    out = []
    for m in re.finditer(r'[\(\[\{]([^()\[\]{}]{0,400})[\)\]\}]', text, re.S):
        body = m.group(1)
        quoted = set(re.findall(r'["\']([A-Za-z0-9_.\-]+)["\']\s*(?!:)', body))
        hits = quoted & names
        if len(hits) >= 2:
            out.append(", ".join(sorted(hits)))
    return out


def _every_declared_name() -> set[str]:
    """Every dataset, run, panel member and producer any declaration names."""
    import sys

    sys.path.insert(0, str(REPO / "eval"))
    from scrm_eval import registry

    return _dataset_names() | set(registry.runs()) | _names()


_QUOTED = re.compile(r'["\']([A-Za-z0-9_.\-]+)["\']')


def _declared_names_in_code(text: str, names: set[str]):
    """`(line number, line, hits)` for every CODE line quoting a declared name -- ONE name is enough."""
    for n, line in enumerate(_scan_text(text, ".py").splitlines(), 1):
        hits = sorted({m for m in _QUOTED.findall(line) if m in names})
        if hits:
            yield n, line, hits


def test_no_consumer_keeps_its_own_copy_of_the_baseline_panel():
    names = _names()
    bad: dict[str, list[str]] = {}
    for rel, text in _sources():
        hits = _offending_literals(text, names)
        if hits:
            bad[rel] = hits
    assert not bad, (
        "these files keep their own copy of a list controls/registry already declares -- read the "
        "declaration instead (controls.PANEL / controls.PRODUCERS, or "
        "`python -m scrm_eval.registry --panel/--producers/--baselines`):\n"
        + "\n".join(f"  {f}: {h}" for f, hh in sorted(bad.items()) for h in hh)
    )


def test_no_consumer_keeps_its_own_list_of_arms():
    """No consumer may keep its own list of arm names."""
    import sys

    sys.path.insert(0, str(REPO / "eval"))
    from scrm_eval import registry as R

    known = set(R.runs())
    bad: dict[str, list[str]] = {}
    for rel, text in _sources():
        if "registry.runs()" in text or "arms_with_full_coverage" in text:
            continue
        for hits in _offending_literals(text, known):
            if len(hits.split(", ")) >= 3:
                bad.setdefault(rel, []).append(hits)
    assert not bad, (
        "these files keep their own list of arm names; derive it from "
        "registry.arms_with_full_coverage(dataset) or registry.runs() instead:\n"
        + "\n".join(f"  {f}: {h}" for f, hh in sorted(bad.items()) for h in hh)
    )


def test_the_collection_scan_fires_on_a_planted_list_and_not_on_prose():
    """Two declared names in a literal tuple must be found; the same two in prose must not."""
    names = _names()
    two = sorted(names)[:2]
    assert len(two) == 2, "this control needs at least two declared model names to plant"
    planted = f'MODELS = ("{two[0]}", "{two[1]}")\n'
    assert _offending_literals(_scan_text(planted, ".py"), names), (
        f"the collection scan did not see a planted tuple of {two} -- it would find nothing, ever"
    )
    in_comment = f'# the panel used to be ("{two[0]}", "{two[1]}") before it grew\n'
    in_docstring = f'"""It once held ("{two[0]}", "{two[1]}"), which is the point."""\n'
    for prose in (in_comment, in_docstring):
        assert not _offending_literals(_scan_text(prose, ".py"), names), (
            f"a name in PROSE was reported as a hardcoded list: {prose!r}"
        )


def test_the_suites_strict_scan_fires_on_a_single_planted_name():
    """One name is the suite's threshold, so the control plants exactly one."""
    names = _every_declared_name()
    one = sorted(names)[0]
    found = list(_declared_names_in_code(f'ds = "{one}"\n', names))
    assert found and found[0][2] == [one], (
        f"the suite's strict scan did not see a planted {one!r}"
    )
    assert not list(_declared_names_in_code(f'#: measured on {one}, which is why\n', names)), (
        "a declared name in a comment was reported as a hardcoded literal"
    )
    for prefix in ("r", "f", "rb", "R", "F"):
        assert not list(_declared_names_in_code(f'x = {prefix}"""measured on {one}"""\n', names)), (
            f"a declared name inside a {prefix}-prefixed triple-quoted string was reported as code"
        )
    assert not list(_declared_names_in_code(f'ds = pick()  # was {one}\n', names)), (
        "a declared name in a TRAILING comment was reported as a hardcoded literal"
    )
    assert list(_declared_names_in_code(f'x = ("#", "{one}")\n', names)), (
        "a '#' inside a string swallowed the rest of the line, hiding a real hardcoded name"
    )
    for fixture in ('cells = ["A", "B", "C"]\n', 'bc = "A" * 16\n', 'digest = "a" * 64\n',
                    'emb = fake_embeddings(mat, cells=cells)  # dataset "ds"\n'):
        assert not list(_declared_names_in_code(fixture, names)), (
            f"synthetic fixture data was reported as a declared name: {fixture!r}"
        )


_DATASET_SCAN_BASES = ("eval",)


def test_no_consumer_keeps_its_own_list_of_datasets():
    """No consumer may keep its own list of dataset names."""
    names = _dataset_names()
    bad: dict[str, list[str]] = {}
    for rel, text in _sources(_DATASET_SCAN_BASES):
        hits = _offending_literals(text, names)
        if hits:
            bad[rel] = hits
    assert not bad, (
        "these files keep their own list of dataset names; derive it from registry.datasets(role) "
        "or `$PY -m scrm_eval.registry --datasets` instead:\n"
        + "\n".join(f"  {f}: {h}" for f, hh in sorted(bad.items()) for h in hh)
    )


def test_no_test_in_this_suite_names_a_dataset_or_an_arm_IN_CODE():
    """No test in this suite may name a dataset or an arm in code; one name is too many."""
    names = _every_declared_name()
    bad: list[str] = []
    for path in sorted((REPO / "eval" / "tests").glob("*.py")):
        rel = str(path.relative_to(REPO))
        if rel in DECLARATION_SITES:
            continue
        for n, line, hits in _declared_names_in_code(path.read_text(), names):
            bad.append(f"{rel}:{n} names {hits} in code: {line.strip()[:100]}")
    assert not bad, (
        "these tests hardcode a name the registry or controls already declares. Ask the declaration "
        "for an example instead of typing one -- registry.datasets(role), registry.runs(), "
        "registry.FLAGSHIP, registry.baselines(), controls.PANEL/PRODUCERS -- or, if the test needs "
        "SOME member with a property, add an `_a_something()` helper in the style of "
        "test_results_page._an_arm(). A name that belongs in prose belongs in the docstring, where "
        "this scan does not look:\n  " + "\n  ".join(bad)
    )


@pytest.mark.parametrize("flag,attr", [("--panel", "PANEL"), ("--producers", "PRODUCERS")])
def test_registry_cli_agrees_with_controls(flag, attr):
    """The bash-facing accessor and the Python declaration must be the same list, in the same order."""
    import subprocess
    import sys

    sys.path.insert(0, str(REPO / "eval"))
    from scrm_eval import controls

    out = subprocess.run([sys.executable, "-m", "scrm_eval.registry", flag],
                         capture_output=True, text=True, cwd=REPO / "eval", check=True)
    assert out.stdout.split() == list(getattr(controls, attr))


def test_every_declared_model_has_its_per_model_mappings():
    """A dict keyed by model name is allowed, but it must cover every declared model."""
    import sys

    sys.path.insert(0, str(REPO / "eval"))
    sys.path.insert(0, str(REPO / "measurements"))
    from scrm_eval import controls
    import results_page as rp

    declared = set(controls.PANEL) | set(controls.PRODUCERS)
    missing = sorted(declared - set(rp.NICE))
    assert not missing, (
        f"results_page.NICE has no display label for {missing}; a declared model with no label "
        "renders as its bare producer key"
    )
    orphans = sorted(set(rp.NICE) - declared)
    assert not orphans, (
        f"results_page.NICE labels {orphans}, which controls.PANEL/PRODUCERS no longer declare. "
        "Drop its NICE label too."
    )


def test_every_registered_run_can_actually_be_produced():
    """A registered arm with no recipe, or a recipe naming an architecture that does not exist."""
    import sys

    sys.path.insert(0, str(REPO / "eval"))
    sys.path.insert(0, str(REPO / "model"))
    from scrm_eval import registry
    from scrm_model.config import arms

    known_arms = set(arms())
    declared = set()
    for run in registry.runs():
        r = registry.recipe(run)
        assert r["arm"] in known_arms, (
            f"registry declares {run!r} on ArchConfig arm {r['arm']!r}, which config.arms() does not "
            f"define ({sorted(known_arms)}). The arm cannot be trained, and the ablation gate would "
            "report it INCOMPLETE as though it were merely unscheduled."
        )
        if "untrained" not in r:
            assert r["corpora"], f"{run!r} has an empty corpora list and cannot be trained"
        declared.add(r["arm"])

    assert known_arms == declared, (
        f"config.arms() defines {sorted(known_arms - declared)} which no registered run is declared "
        f"on. An architecture nobody trains is trainable, and `train.py --arm` will offer it: delete "
        f"the entry, or declare a run on it in registry.ABLATION_DELTAS. (Registry arms: "
        f"{sorted(declared)}.)"
    )


def test_every_arm_the_ablation_declares_is_registered():
    """A member of the ablation study that the registry does not name is never scored."""
    import sys

    sys.path.insert(0, str(REPO / "eval"))
    sys.path.insert(0, str(REPO / "measurements"))
    from scrm_eval import controls, registry
    import ablation

    declared = {m for c in ablation.CONDITIONS for m in c.members} | set(ablation.GRID_2x2.values())
    computed_controls = set(controls.PRODUCERS) | set(controls.PANEL)
    missing = sorted(declared - computed_controls - set(registry.runs()))
    assert not missing, (
        f"measurements/ablation.py declares {missing}, which registry.REQUIRED_RUNS/OPTIONAL_RUNS "
        "do not name. score_one.sbatch would never score them and the study would read INCOMPLETE "
        "for a reason that is a missing name, not missing compute. Register them as OPTIONAL."
    )


def test_eta_reads_the_registry_rather_than_hardcoding_the_matrix_shape():
    """`eta.py` must derive the matrix shape from the registry rather than restate it."""
    src = (REPO / "measurements/eta.py").read_text()
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))

    assert "arm_count()" in code and "pending_datasets()" in code, (
        "eta.py must derive the matrix shape, not restate it"
    )
    assert "registry" in code, "eta.py must read the registry"

    import re
    bad = re.findall(r"^\s*ARMS\s*=\s*\d+", code, re.M)
    assert not bad, f"eta.py hardcodes an arm count again: {bad}"
    bad = re.findall(r"^\s*PENDING\s*=\s*\(", code, re.M)
    assert not bad, f"eta.py hardcodes a dataset list again: {bad}"


ROOT_DECLARATION = "eval/scrm_eval/registry.py"

PROJ_LITERAL = "$SCRM_PROJ"

def _code_only(text: str) -> list[tuple[int, str]]:
    """`(line number, line)` for CODE lines holding the project-root literal."""
    return [
        (n, ln) for n, ln in enumerate(_scan_text(text, ".py").splitlines(), 1)
        if PROJ_LITERAL in ln and not ln.lstrip().startswith(("#", "#:", "*"))
    ]


PROSE_ALLOWED: set[str] = set()


def test_no_python_module_outside_the_registry_restates_the_project_root():
    """No module in `measurements/` or `model/scrm_model/` may restate the project root."""
    bad = {}
    for base in ("measurements", "model/scrm_model"):
        for path in sorted((REPO / base).rglob("*.py")):
            rel = str(path.relative_to(REPO))
            if rel in PROSE_ALLOWED or "vendor" in path.parts:
                continue
            hits = [f"{n}: {ln.strip()[:90]}" for n, ln in _code_only(path.read_text())]
            if hits:
                bad[rel] = hits
    assert not bad, (
        "these modules restate the project root instead of reading registry.PROJ / "
        "registry.corpus_root():\n"
        + "\n".join(f"  {f}: {h}" for f, hh in sorted(bad.items()) for h in hh)
    )


def test_the_project_root_is_declared_once_in_the_eval_package():
    """No module in `scrm_eval` may restate the project root."""
    bad = {}
    for path in sorted((REPO / "eval" / "scrm_eval").glob("*.py")):
        rel = str(path.relative_to(REPO))
        if rel == ROOT_DECLARATION:
            continue
        hits = [f"{n}: {ln.strip()}" for n, ln in _code_only(path.read_text())]
        if hits:
            bad[rel] = hits
    assert not bad, (
        "these modules restate the project root instead of reading registry.PROJ "
        f"(declared once, in {ROOT_DECLARATION}, with an SCRM_PROJ override):\n"
        + "\n".join(f"  {f}: {h}" for f, hh in sorted(bad.items()) for h in hh)
    )


def test_neither_count_matrix_reader_keeps_its_own_table():
    """`targets.load_counts` and `controls._read_counts` must read ONE table."""
    import sys

    sys.path.insert(0, str(REPO / "eval"))
    from scrm_eval import controls, registry, targets

    for mod, gone in ((targets, "MATRIX_DIRS"), (controls, "MATRICES")):
        assert not hasattr(mod, gone), (
            f"{mod.__name__}.{gone} is back. The count-matrix paths are declared once, in "
            "registry.COUNT_MATRICES; read them with registry.count_matrix(ds)."
        )
    missing = [d for d in registry.datasets("eval") if d not in registry.COUNT_MATRICES]
    assert not missing, (
        f"these scored datasets have no registered count matrix: {missing}. Every eval dataset needs "
        "one, or it is published without its mandatory count controls."
    )


def test_the_count_matrix_table_follows_SCRM_PROJ():
    """The paths are RELATIVE to `PROJ` and resolved at call time, so one variable moves them all."""
    import os
    import sys

    sys.path.insert(0, str(REPO / "eval"))
    from scrm_eval import registry

    ds = registry.datasets("eval")[0]
    before = registry.count_matrix(ds)
    old = os.environ.get("SCRM_PROJ")
    os.environ["SCRM_PROJ"] = "/tmp/fake_proj_for_this_test"
    try:
        after = registry.count_matrix(ds)
    finally:
        if old is None:
            del os.environ["SCRM_PROJ"]
        else:
            os.environ["SCRM_PROJ"] = old
    assert str(after).startswith("/tmp/fake_proj_for_this_test"), (
        f"SCRM_PROJ did not move the count-matrix root: {after}"
    )
    assert after != before and after.name == before.name, (
        "the root moved but the relative path did not survive it"
    )


def test_the_train_only_guard_is_the_same_list_everywhere():
    """The train-only guard must be the same list everywhere it is used."""
    import sys

    sys.path.insert(0, str(REPO / "eval"))
    sys.path.insert(0, str(REPO / "measurements"))
    sys.path.insert(0, str(REPO / "model"))
    from scrm_eval import registry
    from scrm_model import run_eval
    import results_page as rp

    declared = set(registry.datasets("train"))
    assert set(run_eval.TRAIN_ONLY) == declared, (
        "run_eval.TRAIN_ONLY has drifted from registry.datasets('train'); derive it rather than "
        f"listing it. Missing: {sorted(declared - set(run_eval.TRAIN_ONLY))}"
    )
    assert set(rp.TRAIN_ONLY) == declared, (
        "results_page.TRAIN_ONLY has drifted from registry.datasets('train')"
    )
    assert not declared & set(registry.datasets("eval")), (
        f"these datasets are declared BOTH train and eval: "
        f"{sorted(declared & set(registry.datasets('eval')))}"
    )


_RUN_SHAPED = re.compile(r'["\'](arm[A-Z0-9][A-Za-z0-9]*(?:[-.][A-Za-z0-9_.]+)+)["\']')


def test_no_module_names_a_run_the_registry_does_not_declare():
    """A run name in code must name a run the registry declares."""
    import sys

    sys.path.insert(0, str(REPO / "eval"))
    from scrm_eval import registry

    known = set(registry.runs()) | set(registry.ABLATION_DELTAS) | {registry.FLAGSHIP}
    bad: list[str] = []
    for rel, text in _sources():
        for n, line in enumerate(text.splitlines(), 1):
            for name in sorted(set(_RUN_SHAPED.findall(line))):
                if name not in known:
                    bad.append(f"{rel}:{n} names {name!r}: {line.strip()[:100]}")
    assert not bad, (
        "these modules name a run the registry does not declare. A dangling run name does not "
        "raise -- the code that reads it finds no checkpoint and no artifact, skips, and reports "
        "success on an empty result. Derive the runs from registry.runs() / ABLATION_DELTAS / "
        "FLAGSHIP, or delete the measurement with the arm:\n  " + "\n  ".join(bad)
    )


def test_the_dangling_run_scan_fires_on_a_run_the_registry_does_not_declare():
    """A permanent positive control: the scan must reject a run-shaped name that is not declared."""
    import sys

    sys.path.insert(0, str(REPO / "eval"))
    from scrm_eval import registry

    known = set(registry.runs()) | set(registry.ABLATION_DELTAS) | {registry.FLAGSHIP}
    dead = f"{registry.FLAGSHIP}-retired"
    assert dead not in known
    line = f'RUNS = [("{dead}", "{registry.FLAGSHIP_RECIPE["arm"]}", "a label")]'
    assert {m for m in _RUN_SHAPED.findall(line) if m not in known} == {dead}
    tags = " ".join(f'"{t}"' for t in ("S96", "--arm", "arm", "args.arm"))
    assert not {m for m in _RUN_SHAPED.findall(tags) if m not in known}
    for run in registry.runs():
        assert not [m for m in _RUN_SHAPED.findall(f'"{run}"') if m not in known], run


def test_the_acquisition_verifier_carries_no_dataset_list_of_its_own():
    """It must verify what `corpora.tsv` declares, not a second list that can drift from it.

    The predecessor carried its own 19-entry table: it named five datasets the paper does not use
    and omitted six it does, so stage 1 verified the wrong set while reading like the authority.
    """
    import ast
    import sys

    sys.path.insert(0, str(REPO / "eval"))
    sys.path.insert(0, str(REPO / "measurements"))
    from scrm_eval import registry

    import verify_acquisition as va

    covered = {r["key"] for r in va.rows()}
    known = {d for d, _, _ in registry.DATASETS}
    assert covered == known, (
        f"the verifier covers a different set than the registry declares.\n"
        f"  verified but unregistered: {sorted(covered - known)}\n"
        f"  registered but unverified: {sorted(known - covered)}"
    )

    # No module-level container may enumerate dataset names: that is the drift the rewrite removed.
    tree = ast.parse((REPO / "measurements" / "verify_acquisition.py").read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        literals = {n.value for n in ast.walk(node.value)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        hits = sorted(literals & known)
        name = getattr(node.targets[0], "id", "?")
        assert len(hits) <= 3, (
            f"module-level {name} enumerates {len(hits)} dataset names ({hits[:4]}), which is a "
            f"second declaration that can disagree with corpora.tsv"
        )


def test_no_arm_selection_picks_a_winner_without_excluding_the_controls():
    """`startswith("arm")` may style a row; it may not choose one."""
    bad = []
    for rel, text in _sources():
        if rel.startswith("eval/tests/") or rel.endswith("test_no_hardcoded_lists.py"):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if 'startswith("arm")' not in line:
                continue
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "is_control" in line:
                continue
            if 'class="ours"' in line or "ours = label.startswith" in line:
                continue
            if "not k.startswith" in line or "not m.startswith" in line:
                continue
            bad.append(f"{rel}:{n}  {stripped[:110]}")
    assert not bad, (
        "these lines select over arms by name prefix without excluding the negative controls. A "
        "maximum or a win taken over `startswith('arm')` can be carried by an untrained init or "
        "the permuted arm. Use `registry.is_control`:\n  " + "\n  ".join(bad)
    )


_PRODUCER_READ = re.compile(r'''\[\s*["']producer["']\s*\]|get\(\s*["']producer["']''')


def test_the_page_compares_a_producer_against_the_declaration_and_not_a_literal():
    """A stamp's `producer` may be tested against `artifact.MODEL_PRODUCERS`, never against a name.

    The stamp key was renamed, so pre-rename artifacts are unreadable to this tree BY DESIGN. A
    producer literal in the page is therefore only ever safe by accident: it goes on matching one
    of the two spellings, and a report keyed by the other renders as a blank row instead of being
    refused. `results_page.OURS` is the derived name for the places that need to print it.
    """
    import sys

    sys.path.insert(0, str(REPO / "eval"))
    sys.path.insert(0, str(REPO / "measurements"))
    from scrm_eval import artifact
    import results_page as rp

    assert set(rp._OURS) == set(artifact.MODEL_PRODUCERS), (
        "results_page.OURS is no longer derived from artifact.MODEL_PRODUCERS; it is the one place "
        "the page is allowed to know our producer's name, and it must be read, not typed."
    )
    names = set(artifact.MODEL_PRODUCERS)
    rel = "measurements/results_page.py"
    bad = []
    for n, line in enumerate(_scan_text((REPO / rel).read_text(), ".py").splitlines(), 1):
        if not _PRODUCER_READ.search(line):
            continue
        hits = sorted({m for m in _QUOTED.findall(line) if m in names})
        if hits:
            bad.append(f"{rel}:{n} compares a producer against {hits}: {line.strip()[:100]}")
    assert not bad, (
        "these lines compare a stamp's producer against a literal model name. Test membership in "
        "artifact.MODEL_PRODUCERS instead, or use results_page.OURS where the name is printed:\n  "
        + "\n  ".join(bad)
    )


def test_the_producer_literal_scan_fires_on_a_planted_comparison():
    """A permanent positive control: the scan must reject a planted producer comparison."""
    import sys

    sys.path.insert(0, str(REPO / "eval"))
    from scrm_eval import artifact

    names = set(artifact.MODEL_PRODUCERS)
    one = sorted(names)[0]
    planted = f'if e.get("producer") == "{one}":\n'
    assert _PRODUCER_READ.search(planted)
    assert {m for m in _QUOTED.findall(planted) if m in names} == {one}, (
        "the producer scan would find nothing, ever"
    )
    ok = 'if e.get("producer") in _artifact.MODEL_PRODUCERS:\n'
    assert not {m for m in _QUOTED.findall(ok) if m in names}, (
        "the derived form was reported as a hardcoded literal"
    )


def test_an_unset_project_root_is_detectable_rather_than_a_path_that_never_exists():
    """"No data root" and "no data" must be two answers, not one empty read.

    With the variable unset, `PROJ` is an unexpanded literal and every path below it is a path
    under a directory that never exists. Each reader then guarded with `if f.is_file()` and
    returned empty, so the page reported 63 unproducible cells against a data root that held every
    report they needed. The predicate below is what lets a reader say which it is.
    """
    import os
    import sys

    sys.path.insert(0, str(REPO / "eval"))
    from scrm_eval import registry

    old = os.environ.get(registry.ROOT_ENV)
    try:
        os.environ.pop(registry.ROOT_ENV, None)
        assert registry.root_unset(), "an unset root did not read as unset"
        with pytest.raises(FileNotFoundError, match=registry.ROOT_ENV):
            registry.require_root("this test")
        with pytest.raises(FileNotFoundError, match=registry.ROOT_ENV):
            registry.windows_required()
        os.environ[registry.ROOT_ENV] = "/tmp/a_root_that_is_named_but_empty"
        assert not registry.root_unset(), "a named root read as unset"
        assert registry.require_root("this test") == Path("/tmp/a_root_that_is_named_but_empty")
        with pytest.raises(FileNotFoundError) as exc:
            registry.windows_required()
        assert registry.ROOT_ENV not in str(exc.value), (
            "a NAMED root with no requirement cache was reported as a missing root; the two "
            "diagnoses must not collapse back into one"
        )
    finally:
        if old is None:
            os.environ.pop(registry.ROOT_ENV, None)
        else:
            os.environ[registry.ROOT_ENV] = old
    assert registry.ROOT_ENV in registry.ROOT_HINT, (
        "the refusal must name the variable the reader has to set"
    )
