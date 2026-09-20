"""The identity of a cached score: what may be carried forward, and what must invalidate it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import artifact

REUSE_FIELDS = ("producer", "checkpoint_sha256", "arch_sha", "cell_order_sha256",
                "forward_precision", "w_eval", "n_cells", "dim", "readout")

SCORING_MODULES = ("scoring.py", "run_probes.py", "run_annotation.py", "probes.py", "metrics.py",
                   "targets.py", "controls.py", "tenx.py", "reduce.py", "artifact.py",
                   "registry.py")


def scoring_version() -> str:
    """A digest of the scoring code itself: a code change invalidates every carried row."""
    h = hashlib.sha256()
    for mod in SCORING_MODULES:
        f = Path(__file__).with_name(mod)
        if not f.exists():
            raise FileNotFoundError(
                f"{mod} is named in SCORING_MODULES but is not in {f.parent}. A module that cannot "
                f"be hashed cannot invalidate a cached score, so this must be fixed rather than "
                f"skipped."
            )
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def digest(obj) -> str:
    """A stable digest of any JSON-able object, for putting a target set into a reuse context."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:32]


def reuse_key(emb: artifact.Embeddings, *, reduction_rule: str, context: dict) -> str:
    """The identity of one row: this artifact, this code, this context."""
    st = emb.stamp
    payload = {f: getattr(st, f, None) for f in REUSE_FIELDS}
    payload["reduction_rule"] = reduction_rule
    payload["_scoring_version"] = scoring_version()
    payload["notes"] = st.notes or {}
    payload["context"] = context
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def refuse_mixed_precision(arts: dict[str, artifact.Embeddings]) -> dict[str, str]:
    """Raise if the model artifacts were produced by different forward arithmetic. Returns the map."""
    prec = {t: (e.stamp.forward_precision or "fp32+eager")
            for t, e in arts.items() if e.stamp.producer in artifact.MODEL_PRODUCERS}
    if len(set(prec.values())) > 1:
        raise artifact.StampError(
            "model artifacts were produced by different forward arithmetic and cannot be compared: "
            + ", ".join(f"{t}={v}" for t, v in sorted(prec.items(), key=lambda kv: kv[1]))
            + ". Re-embed the minority group rather than mixing them."
        )
    return prec
