"""When will the eval matrix finish? Measured from the checkpoints, not estimated from wishes."""
from __future__ import annotations

import functools
import re
import sys
from pathlib import Path

import numpy as np

def _registry():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    from scrm_eval import registry
    return registry


P = _registry().PROJ


def arm_count() -> int:
    """How many runs the eval matrix is actually expected to embed."""
    return len(_registry().runs())


def pending_datasets() -> tuple[str, ...]:
    """Evaluation datasets that do NOT yet hold one artifact per registered run."""
    reg = _registry()
    want = arm_count()
    out = []
    for ds in reg.datasets("eval"):
        have = len(list((P / "eval_runs").glob(f"*/{ds}.*.allreads.arrow")))
        if have < want:
            out.append(ds)
    return tuple(out)


@functools.lru_cache(maxsize=None)
def _req() -> dict:
    return _registry().windows_required()


def shard_lo_hi(name: str, total: int) -> tuple[int, int]:
    """The absolute cell range a shard covers, matching embed.py's np.array_split."""
    m = re.search(r"shard(\d+)-of-(\d+)", name)
    if not m:
        return 0, total
    i, n = int(m.group(1)), int(m.group(2))
    part = np.array_split(np.arange(total), n)[i]
    return int(part[0]), int(part[-1]) + 1


def task_progress():
    """(dataset, done_cells, span_cells, rate_cells_per_s or None) for every checkpointed task."""
    out = []
    for rd in sorted(P.glob("eval_runs/*/*.resume")):
        chunks = sorted(rd.glob("chunk-*.npz"))
        if not chunks:
            continue
        ds = rd.name.split(".")[0]
        total = _req().get(ds, {}).get("cells")
        if not total:
            continue
        lo, hi = shard_lo_hi(rd.name, total)
        try:
            with np.load(chunks[-1]) as z:
                reached = int(z["next_start"])
        except Exception:
            continue
        done = max(0, min(reached, hi) - lo)
        rate = None
        if len(chunks) >= 3:
            try:
                with np.load(chunks[1]) as z:
                    start_cell = int(z["next_start"])
                dt = chunks[-1].stat().st_mtime - chunks[1].stat().st_mtime
                dc = min(reached, hi) - start_cell
                if dt > 0 and dc > 0:
                    rate = dc / dt
            except Exception:
                pass
        out.append((ds, done, hi - lo, rate))
    return out


def median_of(rates) -> float:
    """Median of a dataset's own live task rates -- the honest basis for projecting THAT dataset."""
    return float(np.median(list(rates)))


def main() -> int:
    ARMS = arm_count()
    PENDING = pending_datasets()
    print(f"  registry: {ARMS} runs x {len(_registry().datasets('eval'))} eval datasets; "
          f"{len(PENDING)} dataset(s) not yet complete")
    if not PENDING:
        print("  MATRIX COMPLETE -- every registered run has an artifact for every eval dataset")
        return 0
    tasks = task_progress()
    rates = [r for _, _, _, r in tasks if r]
    if not rates:
        print("no live checkpoint rates yet -- no ETA can be measured")
        return 1
    fleet = float(np.sum(rates))
    median = float(np.median(rates))

    print(f"  fleet throughput  {fleet*3600:,.0f} cell-arms/h from {len(rates)} timed tasks"
          f"  (median {median*3600:,.0f}/h each)")
    total_remaining = 0
    per_dataset_floor: list[float] = []
    for ds in PENDING:
        n = _req().get(ds, {}).get("cells")
        if not n:
            continue
        target = n * ARMS
        done_arms = len(list((P / "eval_runs").glob(f"*/{ds}.*.allreads.arrow")))
        done = done_arms * n + sum(d for t, d, _, _ in tasks if t == ds)
        live = [r for t, _, _, r in tasks if t == ds and r]
        rem = max(0, target - done)
        total_remaining += rem
        started = len({rd.name.split(".")[1] for rd in (P / "eval_runs").glob(f"*/{ds}.*.resume")})
        unstarted = max(0, ARMS - done_arms - started)
        own = median_of(live) if live else median
        max_slots = max(len(live), ARMS - done_arms, 1)
        rate_now = sum(live) if live else 0.0
        best_rate = max(rate_now, min(fleet, max_slots * own))
        if live:
            eta_h = rem / (rate_now * 3600)
            best_h = rem / (best_rate * 3600)
            tag = f"{len(live)} timed, {done_arms} done"
            if unstarted and best_h < eta_h - 0.05:
                tag += (f", {unstarted} arms not started: {eta_h:.1f} h at today's allocation, "
                        f"{best_h:.1f} h across at most {max_slots} slots")
                eta_h = best_h
        else:
            eta_h = rem / (best_rate * 3600)
            tag = (f"not started: {rem / (median * 3600):.0f} h on one task, {eta_h:.1f} h across its "
                   f"{max_slots} remaining arms -- it cannot use more slots than that")
        per_dataset_floor.append(rem / (best_rate * 3600))
        print(f"  {ds:17s} {done:7,}/{target:,} cell-arms ({100*done/target:4.1f}%)"
              f"   +{eta_h:5.1f} h  [{tag}]")
    fleet_h = total_remaining / fleet / 3600
    bound_h = max(per_dataset_floor) if per_dataset_floor else 0.0
    print(f"\n  MATRIX ETA {max(fleet_h, bound_h):.1f} h  ({total_remaining:,} cell-arms remaining;"
          f" {fleet_h:.1f} h if work were freely reallocatable, but the slowest single dataset"
          f" needs {bound_h:.1f} h across its own remaining arms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
