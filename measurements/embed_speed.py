"""Is the eval forward leaving most of the GPU on the table? Measured, not assumed."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "model"), str(REPO / "eval"), str(REPO / "target" / "py")]
import scrm_py  # noqa: E402
from scrm_eval import embed as ev, registry  # noqa: E402
from scrm_model.evalio import model_factory  # noqa: E402
from scrm_model.run_eval import eval_ready  # noqa: E402

PROJ = registry.PROJ


def timed(fn, *a, warmup: int, reps: int):
    for _ in range(warmup):
        fn(*a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(*a)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps, out


def main() -> int:
    run, arm, ds = registry.FLAGSHIP, registry.FLAGSHIP_RECIPE["arm"], "heart_10k_v3"
    ckpt = ev.Checkpoint.load(eval_ready(PROJ / "runs" / run / "latest.pt", arm, Path("/tmp")),
                              model_factory=model_factory)
    corpus = scrm_py.Corpus([str(registry.corpus_root() / ds)])
    w = 1528
    batch = corpus.build_batch(list(range(0, 2)), epoch_seed=20260818, v=w, allow_short=True)
    tok, val = np.asarray(batch.tokens), np.asarray(batch.valid)
    cseg, useg = np.asarray(batch.contig_segment_id), np.asarray(batch.umi_segment_id)
    psi = np.asarray(batch.patch_start_ids)
    n = min(64, tok.shape[0])
    args = (tok[:n], val[:n], cseg[:n], useg[:n], psi[:n])
    print(f"{ds}/{run}: timing {n} windows per call\n")

    m = ckpt.model

    def eager():
        return m.embed_windows(*args)

    def amp():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return m.embed_windows(*args)

    base_t, base_out = timed(eager, warmup=1, reps=3)
    rows = [("eager fp32 (what runs today)", base_t, 0.0)]

    amp_t, amp_out = timed(amp, warmup=1, reps=3)
    rows.append(("autocast bf16", amp_t, float(np.abs(amp_out - base_out).max())))

    try:
        m.model.compile_blocks()
        c_t, c_out = timed(eager, warmup=2, reps=3)
        rows.append(("compiled fp32", c_t, float(np.abs(c_out - base_out).max())))
        ca_t, ca_out = timed(amp, warmup=2, reps=3)
        rows.append(("compiled + bf16", ca_t, float(np.abs(ca_out - base_out).max())))
    except Exception as exc:
        print(f"compile unavailable: {exc}")

    print(f"{'config':32s} {'s/call':>9s} {'speedup':>8s} {'max|delta| vs today':>21s}")
    for name, t, d in rows:
        print(f"{name:32s} {t:9.4f} {base_t / t:7.2f}x {d:21.3e}")
    print("\nprojected heart_10k_v3 hours per arm (1,024,284 windows):")
    for name, t, _ in rows:
        print(f"  {name:32s} {1_024_284 / n * t / 3600:5.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
