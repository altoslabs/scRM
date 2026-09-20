"""The loader gate: measure the FULL per-step data cost —
`build_batch` + block masking + tensor conversion — at the training shape, warm, and report the
margin against the GPU step it must hide behind.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for extra in (REPO / "model", REPO / "target" / "py"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scrm_model.config import SEQ_LEN  # noqa: E402
from scrm_model.masking import block_mask  # noqa: E402

GPU_STEP_MS = 150.0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", nargs="+", required=True)
    p.add_argument("--batch-cells", type=int, default=128)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--no-warm", action="store_true", help="skip the shard read that fills the page cache")
    p.add_argument("--workers", type=int, default=3, help="the loop's prefetch pool size")
    args = p.parse_args(argv)

    import torch

    import scrm_py

    corpus = scrm_py.Corpus([str(c) for c in args.corpus], min_v=2)
    n = corpus.n_cells()
    rng = np.random.default_rng(0)

    def one(step: int) -> tuple[float, float, int]:
        cells = rng.choice(n, size=args.batch_cells, replace=False)
        t0 = time.perf_counter()
        raw = corpus.build_batch([int(c) for c in cells], epoch_seed=step, v=2)
        t1 = time.perf_counter()
        np_batch = {
            "tokens": np.asarray(raw.tokens),
            "valid": np.asarray(raw.valid),
            "contig_segment_id": np.asarray(raw.contig_segment_id),
            "umi_segment_id": np.asarray(raw.umi_segment_id),
            "n_patches": np.asarray(raw.n_patches),
            "cell_index": np.asarray(raw.cell_index),
        }
        m = block_mask(np_batch, epoch_seed=step, v=2, min_ratio=0.2, max_ratio=0.4)
        tensors = [
            torch.from_numpy(np_batch["tokens"]),
            torch.from_numpy(m.tokens_student),
            torch.from_numpy(m.mlm_keep),
            torch.from_numpy(np_batch["valid"]),
            torch.from_numpy(np_batch["contig_segment_id"].view(np.int16)),
            torch.from_numpy(np_batch["umi_segment_id"].view(np.int16)),
        ]
        if torch.cuda.is_available():
            tensors = [t.pin_memory() for t in tensors]
        t2 = time.perf_counter()
        return (t1 - t0) * 1e3, (t2 - t1) * 1e3, sum(t.numel() for t in tensors[:1])

    print(f"{n} cells across {len(args.corpus)} datasets; shape B={args.batch_cells} V=2 S={SEQ_LEN}")
    cold = [one(i) for i in range(args.warmup)]
    for i, (b, m, _) in enumerate(cold):
        print(f"  warmup {i}: build {b:7.1f} ms  mask+tensors {m:6.1f} ms   (cold numbers, reported, not averaged)")

    if not args.no_warm:
        from scrm_model.train import warm_page_cache

        warm_page_cache([str(c) for c in args.corpus], print)

    rows = [one(100 + i) for i in range(args.iters)]
    build = [r[0] for r in rows]
    mask = [r[1] for r in rows]
    total = [b + m for b, m in zip(build, mask)]
    tok_s = args.batch_cells * 2 * SEQ_LEN / (statistics.median(total) / 1e3)
    print(
        f"warm, single-thread, {args.iters} iters: build_batch {statistics.median(build):.1f} ms "
        f"(p90 {sorted(build)[int(len(build) * 0.9)]:.1f})  mask+tensors {statistics.median(mask):.1f} ms  "
        f"TOTAL {statistics.median(total):.1f} ms/batch  {tok_s / 1e6:.1f} M tokens/s"
    )

    from concurrent.futures import ThreadPoolExecutor

    depth = 4
    pool = ThreadPoolExecutor(max_workers=args.workers)
    window: dict[int, object] = {}
    waits: list[float] = []
    steps = args.iters
    for step in range(steps):
        for s in range(step, min(step + depth, steps)):
            if s not in window:
                window[s] = pool.submit(one, 300 + s)
        t0 = time.perf_counter()
        window.pop(step).result()
        waits.append((time.perf_counter() - t0) * 1e3)
        time.sleep(GPU_STEP_MS / 1e3)
    pool.shutdown(wait=False, cancel_futures=True)

    steady = waits[5:]
    med = statistics.median(steady)
    p90 = sorted(steady)[int(len(steady) * 0.9)]
    print(
        f"deployed shape ({args.workers} workers, depth {depth}, {GPU_STEP_MS:.0f} ms step): "
        f"data_wait median {med:.2f} ms  p90 {p90:.2f} ms  max {max(steady):.1f} ms  "
        f"(fill wait, excluded: {waits[0]:.0f} ms)"
    )
    ok = med < 0.01 * GPU_STEP_MS and p90 < 0.05 * GPU_STEP_MS
    print(
        ("PASS" if ok else "FAIL")
        + ": gate is data_wait median < 1% and p90 < 5% of the GPU step -- the hard"
        " requirement, measured in the shape the loop runs"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
