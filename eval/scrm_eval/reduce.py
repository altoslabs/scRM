"""Window-to-cell reductions."""

from __future__ import annotations

import numpy as np

from .artifact import MODEL_PRODUCERS

RULES = ("mean", "median", "max", "l2_mean", "first", "mean_centred", "mean_standardised")


SCRM_RULE = "mean_standardised"

BASELINE_RULE = "mean"


def rule_for(stamp, *, override: str | None = None) -> str:
    """The rule an artifact must be reduced with, read from its own stamp."""
    if override is not None:
        return _rule(override)
    producer = getattr(stamp, "producer", None) if stamp is not None else None
    return SCRM_RULE if producer in MODEL_PRODUCERS else BASELINE_RULE


def _rule(name: str):
    if name not in RULES:
        raise ValueError(f"unknown reduction {name!r}; known: {RULES}")
    return name


def reduce_windows(cell_ids, matrix, rule: str = "mean"):
    """Collapse per-window rows to one vector per cell."""
    _rule(rule)
    ids = np.asarray(cell_ids)
    m = np.asarray(matrix, dtype=np.float64)
    if ids.shape[0] != m.shape[0]:
        raise ValueError(f"{ids.shape[0]} cell ids but {m.shape[0]} rows")
    if m.ndim != 2:
        raise ValueError(f"expected 2-D rows, got {m.shape}")

    change = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [ids.shape[0]]))
    unique = ids[starts]
    if len(set(unique.tolist())) != unique.shape[0]:
        raise ValueError(
            "a cell's windows are not contiguous in this artifact -- reducing by runs would merge "
            "two separate appearances of the same cell"
        )

    out = np.empty((unique.shape[0], m.shape[1]), dtype=np.float64)
    for i, (a, b) in enumerate(zip(starts, stops)):
        block = m[a:b]
        if rule in ("mean", "mean_centred", "mean_standardised"):
            out[i] = block.mean(axis=0)
        elif rule == "median":
            out[i] = np.median(block, axis=0)
        elif rule == "max":
            out[i] = block.max(axis=0)
        elif rule == "l2_mean":
            norms = np.linalg.norm(block, axis=1, keepdims=True)
            out[i] = np.where(norms > 0, block / norms, block).mean(axis=0)
        else:
            out[i] = block[0]
    if rule == "mean_standardised":
        mu = out.mean(axis=0, keepdims=True)
        sd = out.std(axis=0, keepdims=True)
        out = (out - mu) / (sd + 1e-8)
    if rule == "mean_centred":
        out = out - out.mean(axis=0, keepdims=True)
    return unique, out


def all_rules(cell_ids, matrix) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Every rule from one pass, which is the purpose of storing per-window vectors."""
    return {r: reduce_windows(cell_ids, matrix, r) for r in RULES}
