"""The published page's renderer."""
import json
import sys
from pathlib import Path


from proj_guard import needs_proj

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "measurements"))
import results_page as rp  # noqa: E402
from scrm_eval import artifact, controls as _controls, registry as _registry  # noqa: E402


def _arm_keys(n: int = 1, exclude=()) -> list[str]:
    """`n` DISTINCT selected scrm run keys, excluding any named."""
    out = [r for r, _ in rp.ARMS if r.startswith("arm") and r not in exclude]
    assert len(out) >= n, f"need {n} selected scrm arms, have {len(out)}"
    return out[:n]


def _a_baseline() -> str:
    """SOME count-side baseline the page will render -- the first the declaration lists."""
    for key in _registry.baselines():
        if key in rp.NICE and key not in rp.DROPPED_PRODUCERS:
            return key
    raise AssertionError("no declared baseline has a display label the page would render")


def _model_producer() -> str:
    """The producer string that marks an artifact as OURS, from the set that enforces it."""
    return sorted(artifact.MODEL_PRODUCERS)[0]


def _e2_dataset() -> str:
    """The dataset the page reads its E2 table for, from the page's own declaration."""
    ds = rp.E2_DS
    assert ds in _registry.datasets("eval"), (
        f"results_page reads its E2 table for {ds!r}, which registry.datasets('eval') does not "
        "name -- the E2 table would render no rows at all"
    )
    assert _registry.label_file(ds), f"E2's dataset {ds} has no independent label file"
    assert rp.E3_DS != ds, "E2 and E3 must not be the same dataset"
    return ds


def test_every_scrm_arm_label_maps_back_to_its_key():
    """Every scRM arm display label must map back to exactly one run key."""
    assert len(rp._ARM_OF) == len(rp.ALL_ARMS), "two arms share a display label"
    for key, label in rp.ARMS:
        assert rp._ARM_OF[label] == key


def test_in_domain_arms_are_real_arms():
    """Every name in `IN_DOMAIN` must be a scored dataset and a known arm."""
    known = {k for k, _ in rp.ARMS}
    for ds, arms in rp.IN_DOMAIN.items():
        assert ds in rp.DS, f"{ds} is not a scored dataset"
        for a in arms:
            assert a in known, f"{a} is not a known arm, so it would never be marked"


@needs_proj
def test_e2_reads_scrm_rows_from_the_combined_annotation_report(tmp_path, monkeypatch):
    """E2 must read its scRM rows from the combined annotation report."""

    ds = _e2_dataset()
    labels = dict(rp.ARMS)
    hs = next((a for a in _registry.in_domain_arms(ds) if a in labels), None)
    assert hs, (
        f"no selected arm is declared in-domain on {ds}, so the in-domain third of this fixture "
        "cannot be built; registry.IN_DOMAIN is where that is declared"
    )
    need = rp.full_coverage(ds)
    run, low = _arm_keys(2, exclude=(hs,))
    model, base = _model_producer(), _a_baseline()
    art = {
        f"{ds}.{run}.allreads": {
            "producer": model, "w_eval": need,
            "linear": {"macro": 0.91, "baseline": 0.098, "n_cells": 3822},
            "knn_cosine": {"macro": 0.88, "baseline": 0.098, "n_cells": 3822}},
        f"{ds}.{low}.allreads": {
            "producer": model, "w_eval": need - 1,
            "linear": {"macro": 0.99, "baseline": 0.098, "n_cells": 3822},
            "knn_cosine": {"macro": 0.99, "baseline": 0.098, "n_cells": 3822}},
        f"{ds}.{hs}.allreads": {
            "producer": model, "w_eval": need,
            "linear": {"macro": 0.999, "baseline": 0.098, "n_cells": 3822},
            "knn_cosine": {"macro": 0.999, "baseline": 0.098, "n_cells": 3822}},
        f"{ds}.{base}": {
            "producer": base, "w_eval": 1,
            "linear": {"macro": 1.0, "baseline": 0.098, "n_cells": 3822},
            "knn_cosine": {"macro": 1.0, "baseline": 0.098, "n_cells": 3822}},
    }
    fake = tmp_path / "eval_artifacts"
    fake.mkdir()
    (fake / f"{ds}.allmodels.annotation.json").write_text(json.dumps({"artifacts": art}))
    (fake / f"{ds}.E2_standardised.json").write_text(json.dumps({rp.NICE[base]: {
        "linear": [1.0, 0.098, 3822], "knn_cosine": [1.0, 0.098, 3822]}}))
    monkeypatch.setattr(rp, "P", tmp_path)

    _, e2, _, _ = rp.collect()
    rendered = list(e2)
    at_coverage = labels[run]
    assert at_coverage in rendered, "an at-coverage arm must reach the E2 table"
    assert abs(e2[at_coverage]["linear"][0] - 0.91) < 1e-9
    assert labels[low] not in rendered, "a below-coverage arm must be withheld"
    indom = [k for k in rendered if "IN-DOMAIN" in k]
    assert len(indom) == 1 and labels[hs] in indom[0]
    assert sum(1 for k in rendered if rp.NICE[base] in k) == 1, "a baseline must not be listed twice"


def test_the_e8_table_reports_the_flagship_and_a_control_cannot_win_its_column():
    """E8's scRM column must report the flagship, never the best-scoring arm."""
    flag = dict(rp.ALL_ARMS)[_registry.FLAGSHIP]
    controls_ = [lab for run, lab in rp.ALL_ARMS
                 if "untrained" in run or run.endswith("_permuted")]
    assert controls_, "no control arm is declared, so this test asserts nothing"

    fms = [rp.NICE[k] for k in _controls.PANEL if k in rp.NICE][:2]
    assert len(fms) == 2, "need two declared count foundation models to bracket the range"

    def _e3_row(returned):
        """E8's E3 line ALONE. Asserting on the whole body would pass on another task's 0.500.

        `contrast_table` returns `(body, macros)`: the table rows and the caption macros the main
        text reads. Unpacking here rather than indexing at each call site means a third element
        would fail loudly instead of silently changing which value this test inspects.
        """
        body, macros = returned
        assert macros.startswith("\\newcommand"), f"E8's macros fragment is malformed: {macros!r}"
        got = [ln for ln in body.splitlines() if ln.startswith("cell-type annotation")]
        assert len(got) == 1, f"expected exactly one E3 row in E8's body, got {got}"
        return got[0]

    e3 = {flag: 0.500, controls_[0]: 0.900, fms[0]: 0.700, fms[1]: 0.600}
    row = _e3_row(rp.contrast_table(_registry.PROJ, {}, {}, e3))
    assert "$0.500$" in row, (
        f"E8 did not report the flagship's 0.500; a control at 0.900 or a count model won "
        f"instead:\n{row}"
    )
    assert "$0.900$" not in row, "E8 reported the C2/C3 control's value in the scrm column"
    assert "$0.600$" in row and "$0.700$" in row, (
        "E8 must report BOTH ends of the count-FM range, not just the best"
    )
    one = {flag: 0.500, fms[0]: 0.700}
    row1 = _e3_row(rp.contrast_table(_registry.PROJ, {}, {}, one))
    assert "$0.700$" not in row1, f"E8 built a count-FM range from a single model:\n{row1}"
    assert rp.TEX_MISSING in row1, (
        f"a refused E3 row must be MARKED missing, never dotted or left blank:\n{row1}"
    )


def test_no_depth_threshold_auroc_literal_is_in_the_source():
    """`results_page.py` must carry no depth-only AUROC literal in its code."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_no_hardcoded_lists import _scan_text

    code = _scan_text(Path(rp.__file__).read_text(), ".py")
    for gone in ("0.628", "0.719", "0.702", "0.770", "0.695", "0.751"):
        assert gone not in code, (
            f"{gone} is a depth-only AUROC transcribed into the page as code; "
            f"depth_threshold_sentence derives the current pair instead"
        )
