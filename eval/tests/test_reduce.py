"""The reduction contract: which rule an artifact is read with, and who decides."""

from pathlib import Path

from scrm_eval import artifact, reduce


def test_no_scorer_hardcodes_a_reduction_for_an_scrm_artifact():
    """No scorer may name a reduction rule itself instead of asking `reduce.rule_for`."""
    import re

    repo = Path(__file__).resolve().parents[2]

    EXEMPT = {
        "reduce.py": "defines the rules",
        "coverage_sweep.py": "varies w_eval, holds the rule fixed so the sweep is the only variable",
    }
    SCORERS = [
        repo / "eval/scrm_eval/run_probes.py",
        repo / "eval/scrm_eval/run_annotation.py",
        repo / "eval/scrm_eval/cli.py",
        repo / "measurements/arm_geometry.py",
        repo / "measurements/uce_protocols.py",
        repo / "measurements/e7_vs_control.py",
    ]
    _RULES = {"mean", "median", "max", "l2_mean", "first", "mean_centred", "mean_standardised"}

    def hardcoded_rules(src: str):
        """(line, rule) for every reduce_windows call passing a rule as a string literal."""
        for m in re.finditer(r"reduce_windows\s*\(", src):
            i, depth = m.end(), 1
            while i < len(src) and depth:
                depth += (src[i] == "(") - (src[i] == ")")
                i += 1
            args = src[m.end():i - 1]
            for lit in re.findall(r"[\"\']([a-z_]+)[\"\']", args):
                if lit in _RULES:
                    yield src[:m.start()].count("\n") + 1, lit

    offenders = []
    for f in SCORERS:
        if not f.exists() or f.name in EXEMPT:
            continue
        src = f.read_text()
        for line, lit in hardcoded_rules(src):
            offenders.append(f"{f.name}:{line} passes the literal {lit!r}")
        assert "rule_for" in src, f"{f.name} does not consult reduce.rule_for"
    assert not offenders, (
        "a scorer names a reduction itself instead of asking reduce.rule_for(stamp):\n  "
        + "\n  ".join(offenders))


def test_rule_for_never_standardises_a_baseline():
    """`rule_for` must never return a standardising rule for a baseline artifact."""
    class Stamp:
        def __init__(self, producer):
            self.producer = producer

    from scrm_eval import controls

    assert reduce.rule_for(Stamp(sorted(artifact.MODEL_PRODUCERS)[0])) == reduce.SCRM_RULE
    baselines = set(controls.PANEL) | set(controls.PRODUCERS)
    assert baselines, "the panel is empty, so this test would assert nothing"
    for other in sorted(baselines):
        assert reduce.rule_for(Stamp(other)) == "mean", other
    assert reduce.rule_for(None) == "mean", "an absent stamp must not get our rule"
