"""Where the step time goes, and what torch.compile recovers."""

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


def make_batch(corpus_dirs, seq_len, device, batch_cells: int = 128):
    import scrm_py

    corpus = scrm_py.Corpus(corpus_dirs, seq_len=seq_len, min_v=2)
    raw = corpus.build_batch(list(range(batch_cells)), epoch_seed=7, v=2)
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
    return to_device(host, device)


def run(student, teacher, projector, reg, opt, grad_params, batch, iters):
    times = []
    for _ in range(iters):
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
    return times


def sweep(args, cfg, device) -> int:
    """Throughput and peak memory against B, compiled, on one GPU."""
    print(f"  {'B':>6s} {'windows':>8s} {'tokens/step':>13s} {'s/step':>8s} {'M tok/s':>9s} {'peak GiB':>9s}")
    for b in args.batch_cells:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            batch = make_batch([str(c) for c in args.corpus], args.seq_len, device, b)
            torch.manual_seed(0)
            student = ScrmModel(cfg).to(device)
            teacher = copy.deepcopy(student)
            for q in teacher.parameters():
                q.requires_grad_(False)
            projector = build_projector(cfg).to(device)
            reg = SIGReg(num_slices=4 * cfg.proj_dim)
            opt = optim.build_optimizer(student, projector, fused=True)
            gp = [g for grp in opt.param_groups for g in grp["params"]]
            student.compile_blocks()
            teacher.compile_blocks()
            ts = run(student, teacher, projector, reg, opt, gp, batch, 5)
            best = min(ts[2:])
            tok = b * 2 * args.seq_len
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"  {b:6d} {b*2:8d} {tok:13,} {best:8.3f} {tok/best/1e6:9.2f} {peak:9.1f}")
        except torch.cuda.OutOfMemoryError:
            print(f"  {b:6d} {b*2:8d} {'':13s} {'OOM':>8s}")
            torch.cuda.empty_cache()
        finally:
            for name in ("student", "teacher", "projector", "opt", "batch"):
                if name in dir():
                    pass
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", nargs="+", required=True)
    p.add_argument("--arm", choices=sorted(arms()), default=FLAGSHIP_RECIPE["arm"])
    p.add_argument("--seq-len", type=int, default=16384)
    p.add_argument("--batch-cells", type=int, nargs="+", default=[128],
                   help="sweep B; each value is benchmarked separately")
    args = p.parse_args(argv)
    device = torch.device("cuda")
    cfg = arms()[args.arm]
    print(f"{torch.cuda.get_device_name()}, arm {args.arm}, S={args.seq_len}")
    if len(args.batch_cells) > 1:
        return sweep(args, cfg, device)

    b = args.batch_cells[0]
    batch = make_batch([str(c) for c in args.corpus], args.seq_len, device, b)
    torch.manual_seed(0)
    student = ScrmModel(cfg).to(device)
    teacher = copy.deepcopy(student)
    for q in teacher.parameters():
        q.requires_grad_(False)
    projector = build_projector(cfg).to(device)
    reg = SIGReg(num_slices=4 * cfg.proj_dim)
    opt = optim.build_optimizer(student, projector, fused=True)
    grad_params = [g for grp in opt.param_groups for g in grp["params"]]

    ts = run(student, teacher, projector, reg, opt, grad_params, batch, 4)
    eager = min(ts[1:])
    print(f"eager: first {ts[0]:.2f}s (flex compile), "
          f"steady {eager:.2f}s  ({b * 2 * args.seq_len / eager / 1e6:.1f} M tok/s)")

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        run(student, teacher, projector, reg, opt, grad_params, batch, 1)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))

    student.compile_blocks()
    teacher.compile_blocks()
    ts = run(student, teacher, projector, reg, opt, grad_params, batch, 5)
    best = min(ts[2:])
    print(f"compiled blocks: first {ts[0]:.1f}s, steady {best:.2f}s  ({b * 2 * args.seq_len / best / 1e6:.1f} M tok/s)")
    print(f"speedup vs eager: {eager / best:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
