"""Max affordable window size at a fixed 256-window batch."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for extra in (REPO / "model", REPO / "target" / "py", REPO / "eval"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scrm_model import optim  # noqa: E402
from scrm_model.config import FLAGSHIP_RECIPE, arms  # noqa: E402
from scrm_model.losses import training_step  # noqa: E402
from scrm_model.masking import block_mask  # noqa: E402
from scrm_model.model import ScrmModel  # noqa: E402
from scrm_model.projector import build_projector  # noqa: E402
from scrm_model.train import to_device  # noqa: E402
from scrm_model.losses import SIGReg  # noqa: E402

SIZES = [16384, 24576, 32768, 49152, 65536, 98304, 131008]


def one_size(corpus_dirs, cfg, seq_len: int, device) -> dict:
    import scrm_py

    corpus = scrm_py.Corpus(corpus_dirs, seq_len=seq_len, min_v=2)
    below = set(corpus.cells_below_min_v())
    cells = [i for i in range(corpus.n_cells()) if i not in below][:128]
    raw = corpus.build_batch(cells, epoch_seed=7, v=2)
    nb = {
        k: np.asarray(getattr(raw, k))
        for k in ("tokens", "valid", "contig_segment_id", "umi_segment_id", "n_patches", "cell_index")
    }
    m = block_mask(nb, epoch_seed=7, v=2, min_ratio=0.2, max_ratio=0.4, compute_bins=False)
    host = {
        "tokens": torch.from_numpy(nb["tokens"]),
        "tokens_student": torch.from_numpy(m.tokens_student),
        "mlm_keep": torch.from_numpy(m.mlm_keep),
        "valid": torch.from_numpy(nb["valid"]),
        "contig_segment_id": torch.from_numpy(nb["contig_segment_id"].view(np.int16)),
        "umi_segment_id": torch.from_numpy(nb["umi_segment_id"].view(np.int16)),
        "n_patches": torch.from_numpy(nb["n_patches"].astype(np.int64)),
    }

    torch.manual_seed(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    student = ScrmModel(cfg).to(device)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    projector = build_projector(cfg).to(device)
    reg = SIGReg(num_slices=4 * cfg.proj_dim)
    opt = optim.build_optimizer(student, projector, fused=True)
    grad_params = [p for g in opt.param_groups for p in g["params"]]
    batch = to_device(host, device)

    times = []
    for it in range(2):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = training_step(student, teacher, projector, reg, batch, v=2)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(grad_params, optim.GRAD_CLIP)
        opt.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return {
        "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
        "reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        "step_s": times[-1],
        "compile_s": times[0],
        "tok_s": 256 * seq_len / times[-1],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", nargs="+", required=True)
    p.add_argument("--arm", choices=sorted(arms()), default=FLAGSHIP_RECIPE["arm"])
    p.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    args = p.parse_args(argv)
    device = torch.device("cuda")
    name = torch.cuda.get_device_name()
    total_gb = torch.cuda.get_device_properties(device).total_memory / 2**30
    cfg = arms()[args.arm]
    print(f"{name} ({total_gb:.0f} GB), arm {args.arm}, 256 windows (B=128, v=2), bf16 + flex_attention")
    print(f"{'seq_len':>9} {'peak GB':>9} {'reserved':>9} {'step s':>8} {'M tok/s':>9}")
    for s in args.sizes:
        try:
            r = one_size([str(c) for c in args.corpus], cfg, s, device)
            print(
                f"{s:>9} {r['peak_gb']:>9.1f} {r['reserved_gb']:>9.1f} {r['step_s']:>8.2f} "
                f"{r['tok_s'] / 1e6:>9.1f}"
            )
        except torch.cuda.OutOfMemoryError:
            print(f"{s:>9}       OOM — this GPU's ceiling is the previous row")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
