"""Load GeneJEPA's EMA-teacher encoder from its published checkpoint.

Their code is not in this repository: eval/countfm/fetch.py clones BiostateAI/GeneJEPA at a
pinned commit into the weights root and this module imports configs, models and tokenizer
from there.

`load_teacher_encoder` puts the EMA-teacher weights from `genejepa-epoch=49.ckpt` into a plain
`GenePerceiverEncoder`, which is numerically what upstream's
`GenePerceiverJEPA.get_embedding(..., use_teacher=True)` does without needing `ema_pytorch`.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Upstream, pinned. `fetch.py` clones exactly this, and the checkpoint's pickled config
#: dataclasses must match the code that wrote them.
UPSTREAM = "https://github.com/BiostateAI/GeneJEPA"
UPSTREAM_COMMIT = "a2f4d7218b17f2f52cc5f1cc94420c8ef1ae3265"

#: Where `fetch.py` puts the clone, relative to `--weights-root`. `fetch.py` reads this and the
#: commit above from here rather than keeping its own copy.
SRC_DIR = "genejepa_src"

#: The package inside the clone, and the modules `_their_modules` imports from it. `fetch.py
#: --check` verifies exactly these files, so what the check requires is what this module imports.
SRC_PACKAGE = "genejepa"
THEIR_MODULES = ("configs", "models", "tokenizer")

TEACHER_PREFIX = "model.teacher_encoder.ema_model."


def source_files() -> tuple[str, ...]:
    """The clone-relative paths this module needs, for `fetch.py --check` to verify."""
    return tuple(f"{SRC_PACKAGE}/{m}.py" for m in THEIR_MODULES)


def _their_modules(weights_root: Path):
    """Import their three modules from the fetched clone, or say how to get it."""
    src = Path(weights_root) / SRC_DIR
    # PER FILE, not `is_dir`. A clone interrupted partway leaves the directory behind.
    absent = [f for f in source_files() if not (src / f).is_file()]
    if absent:
        raise SystemExit(
            f"GeneJEPA's own code is not at {src} ({', '.join(absent)} missing). It is not "
            "redistributed here.\n"
            f"  python eval/countfm/fetch.py --weights-root {weights_root} --only genejepa\n"
            f"clones {UPSTREAM} at {UPSTREAM_COMMIT[:12]}."
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    # THEIR forward CALLS `torch.utils.checkpoint.checkpoint_sequential` WITHOUT IMPORTING THAT
    # SUBMODULE. `torch.utils` does not expose it on attribute access, so importing it here is
    # what makes their code run unmodified.
    import torch.utils.checkpoint  # noqa: F401, PLC0415

    from genejepa import configs, models, tokenizer  # noqa: PLC0415

    return configs, models, tokenizer


def load_teacher_encoder(ckpt_path: str | Path, weights_root: str | Path):
    """The EMA-teacher encoder from the published Lightning checkpoint, on CPU, in eval mode."""
    import torch

    import __main__

    configs, models, _ = _their_modules(Path(weights_root))

    # THE CHECKPOINT PICKLES ITS CONFIG DATACLASSES UNDER `__main__`. Registering their classes
    # there is what lets torch.load unpickle `hyper_parameters`.
    for name in ("ModelConfig", "TrainingConfig", "DataConfig", "ExperimentConfig"):
        cls = getattr(configs, name, None)
        if cls is not None and not hasattr(__main__, name):
            setattr(__main__, name, cls)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    config = ckpt["hyper_parameters"]["model_config"]
    teacher = {
        k[len(TEACHER_PREFIX):]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith(TEACHER_PREFIX)
    }
    if not teacher:
        raise SystemExit(
            f"{ckpt_path} has no weights under {TEACHER_PREFIX!r}: this is not the published "
            "teacher checkpoint, and the student is not the inference path its README names."
        )
    model = models.GenePerceiverEncoder(config)
    model.load_state_dict(teacher, strict=True)
    model.eval()
    return model


def tokenizer_for(weights_root: str | Path):
    """Their `scRNATokenizer`, so preprocessing is theirs and not a reimplementation."""
    _, _, tok = _their_modules(Path(weights_root))
    return tok.scRNATokenizer
