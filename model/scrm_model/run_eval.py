"""Embed a training checkpoint over one or more corpora and score against controls + the count-FM
panel, one stamped report per dataset.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
for extra in (REPO / "model", REPO / "eval", REPO / "target" / "py"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scrm_model.config import arch_dict, arms  # noqa: E402
from scrm_eval.embed import arch_digest  # noqa: E402
from scrm_model.evalio import model_factory  # noqa: E402
from scrm_eval import registry  # noqa: E402

PANEL_DIR = registry.PROJ / "eval_artifacts"
TRAIN_ONLY = registry.datasets("train")


def eval_ready(path: Path, arm: str, work: Path) -> Path:
    """Backfill `arch`, `readout` and `arch_sha` into a copy of a checkpoint that lacks them, from
    the arm's known config. The training checkpoint is left untouched.
    """
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if "arch" in raw:
        return path
    arch = arch_dict(arms()[arm], arm)
    raw["arch"] = arch
    raw["arch_sha"] = arch_digest(arch)
    raw["readout"] = {
        "levels": ["bytes->contig", "contig->umi", "umi->window", "views->group2"],
        "pool": "seed_attention",
        "window_to_cell": "group_view_pool_then_mean",
        "views": 2,
        "grouping": "adjacent_v2",
    }
    raw["state_dict"] = raw["student"]
    out = work / f"ckpt_{arm}_evalready.pt"
    torch.save(raw, out)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--arm", choices=tuple(arms()), required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--corpus-root", type=Path, default=registry.corpus_root())
    p.add_argument("--datasets", nargs="+", required=True,
                   help="held-out datasets to embed and score; a TRAIN_ONLY name is refused")
    p.add_argument("--w-eval", type=int, required=True,
                   help="windows per cell; use the dataset's value from windows_required.json")
    p.add_argument("--eval-seed", type=int, default=20260818)
    args = p.parse_args(argv)
    named = [ds for ds in args.datasets if ds in TRAIN_ONLY]
    if named:
        raise SystemExit(
            f"--datasets names training corpora {named}: an evaluation table may not contain a "
            "training dataset (results_page.py asserts "
            "the disjointness). Score held-out datasets here."
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import scrm_py
    from scrm_eval import embed as ev

    ckpt = ev.Checkpoint.load(eval_ready(args.checkpoint, args.arm, args.out_dir), model_factory=model_factory)
    py = sys.executable
    for ds in args.datasets:
        corpus = scrm_py.Corpus([str(args.corpus_root / ds)])
        art = args.out_dir / f"{ds}.{args.tag}.arrow"
        if not art.exists():
            ev.embed_to_file(ckpt, corpus, art, dataset=ds, w_eval=args.w_eval, eval_seed=args.eval_seed)
            print(f"embedded {ds} -> {art}", flush=True)
        others = sorted(PANEL_DIR.glob(f"{ds}.*.arrow"))
        report = args.out_dir / f"{ds}.{args.tag}.report.json"
        subprocess.run(
            [py, "-m", "scrm_eval.cli", "--artifacts", str(art), *map(str, others),
             "--dataset", ds, "-o", str(report)],
            check=True,
            env={"PYTHONPATH": f"{REPO / 'eval'}:{REPO / 'target' / 'py'}", "PATH": "/usr/bin:/bin"},
        )
        print(f"report {report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
