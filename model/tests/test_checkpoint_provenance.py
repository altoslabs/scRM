"""Every checkpoint must be one this tree's `losses.SIGReg` could have written.

Nothing in an artifact's stamp records which code produced the weights, so the only evidence is
the shape of the saved regulariser state. `losses.SIGReg` registers its quadrature buffers with
`persistent=False`, so its `state_dict()` is exactly `{"global_step"}` and a checkpoint it wrote
carries exactly that under `reg`.

These tests are a WHITELIST, not a blacklist: they assert the state is what this implementation
produces, rather than looking for the fingerprint of some particular other one. That is both
stronger -- it refuses every implementation that is not this one, including ones nobody has
written yet -- and self-contained, since it needs no reference to anyone else's class or
submodule names.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from scrm_model import losses

#: The complete persistent state of this tree's regulariser. Anything else under a checkpoint's
#: `reg` key was written by something that is not `losses.SIGReg`.
PERSISTENT_STATE = ["global_step"]


def _sigreg() -> torch.nn.Module:
    """A `losses.SIGReg` built with whatever its required arguments are, for state inspection."""
    import inspect

    kwargs = {}
    for name, p in list(inspect.signature(losses.SIGReg.__init__).parameters.items())[1:]:
        if p.default is inspect.Parameter.empty and name not in ("self", "args", "kwargs"):
            kwargs[name] = 96 if "dim" in name else 256
    return losses.SIGReg(**kwargs)


def test_sigreg_persists_only_its_step_counter():
    """The whitelist is derived from the class, so it cannot drift from what the class does."""
    assert sorted(_sigreg().state_dict()) == PERSISTENT_STATE, (
        "losses.SIGReg's persistent state changed. If a buffer became persistent on purpose, "
        "PERSISTENT_STATE moves with it -- but every checkpoint already on disk then fails the "
        "whitelist below, which is the correct outcome and not a reason to weaken the check."
    )


def test_no_checkpoint_on_disk_has_foreign_regulariser_state():
    """Every checkpoint under `runs/` must carry this implementation's state, or none at all."""
    proj = os.environ.get("SCRM_PROJ")
    if not proj:
        pytest.skip("SCRM_PROJ is unset, so there is no checkpoint tree to scan; the structural "
                    "test above runs unconditionally and is the guard")
    root = Path(proj) / "runs"
    if not root.is_dir():
        pytest.skip(f"{root} does not exist, so SCRM_PROJ does not name a data root")
    bad = []
    for ckpt in sorted(root.glob("*/*.pt")):
        try:
            reg = torch.load(ckpt, map_location="cpu", weights_only=False).get("reg", {})
        except Exception as exc:  # a corrupt checkpoint is a different failure, reported as itself
            bad.append(f"{ckpt}: unreadable ({exc})")
            continue
        keys = sorted(str(k) for k in reg)
        # An untrained run has no regulariser state at all, which is not foreign state.
        if keys and keys != PERSISTENT_STATE:
            bad.append(f"{ckpt}: reg state is {keys}, not {PERSISTENT_STATE}")
    assert not bad, ("checkpoints whose regulariser state this tree could not have written:\n  "
                     + "\n  ".join(bad))
