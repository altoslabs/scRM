"""Read a per-target number out of a perturbation report, whatever shape it stores them in."""
from __future__ import annotations

BY_NAME = "per_target_by_name"
ALIGNED = "per_target"


def per_target(report: dict, entry: dict, target: str, field: str = "auroc"):
    """`entry`'s value for `target`, or None if that pair was not scored."""
    by = entry.get(BY_NAME)
    if not isinstance(by, dict):
        raise TypeError(
            f"this entry has no {BY_NAME!r} mapping, so a per-target number cannot be located by "
            f"name. Regenerate the report with the current producer -- do not index "
            f"{ALIGNED!r} positionally at the call site."
        )
    return _pick(by.get(target) or {}, field)


def _pick(rec, field: str):
    """One field out of a per-target record. "fp:<key>" reaches into its `false_positive` block."""
    if not isinstance(rec, dict):
        return None
    if field.startswith("fp:"):
        return (rec.get("false_positive") or {}).get(field[3:])
    return rec.get(field)


def targets_of(report: dict, entry: dict) -> list[str]:
    """Every target `entry` holds a number for, by name, in the report's declared order where it has one."""
    by = entry.get(BY_NAME)
    if not isinstance(by, dict):
        raise TypeError(f"this entry has no {BY_NAME!r} mapping")
    declared = list(report.get("targets") or [])
    return [t for t in declared if t in by] + sorted(set(by) - set(declared))
