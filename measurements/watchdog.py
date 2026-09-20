"""Watch this project's Slurm jobs, record what went wrong, and auto-retry the mechanical failures."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from scrm_eval import registry  # noqa: E402

P = registry.PROJ
STATE = P / "eval_reports" / "watchdog.json"
SLURM_LOGS = P.parent / "logs"
LOG = SLURM_LOGS / "watchdog.log"
BAD = ("FAILED", "OUT_OF_MEMORY", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "CANCELLED")
MAX_RETRIES = 1
RETRYABLE = {"scrm-score1": "model/jobs/score_one.sbatch"}


def sacct(since_min: int) -> list[dict]:
    start = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - since_min * 60))
    cmd = ["sacct", "-S", start, "-X", "--parsable2", "--noheader",
           "--format=JobID,JobName,State,ExitCode,Elapsed,ReqMem,MaxRSS,Reason,WorkDir"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    except Exception as exc:
        print(f"[watchdog] sacct failed: {exc}", flush=True)
        return []
    rows = []
    for line in out.splitlines():
        f = line.split("|")
        if len(f) < 8:
            continue
        rows.append(dict(zip(("jobid", "name", "state", "exit", "elapsed", "reqmem", "maxrss",
                              "reason", "workdir"), f)))
    return rows


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            pass
    return {"seen": {}, "retries": {}, "events": []}


def save(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".json.tmp")
    state["events"] = state["events"][-500:]
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(STATE)


def note(state: dict, msg: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    state["events"].append({"at": stamp, "msg": msg})
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(f"{stamp}  {msg}\n")
    print(f"[watchdog] {msg}", flush=True)


def double_mem(reqmem: str) -> str:
    m = re.match(r"(\d+)([GM])", reqmem or "")
    if not m:
        return "192G"
    v, unit = int(m.group(1)), m.group(2)
    v = v * 2 if unit == "G" else max(1, v * 2 // 1024)
    return f"{min(v, 768)}G"


def dataset_of(jobid: str, name: str) -> str | None:
    """The dataset a scoring job was for, read from its own log rather than guessed."""
    for pat in (f"scrm-score1-{jobid}.out", f"scrm-e4score-{jobid}.out",
                f"scrm-annot-fly-{jobid}.out"):
        f = SLURM_LOGS / pat
        if f.exists():
            for line in f.read_text().splitlines()[:40]:
                m = re.search(r"\b([a-z][a-z0-9_]{3,})\s*:\s*(scoring|waiting|annotation)", line)
                if m:
                    return m.group(1)
                m = re.search(r"dataset[= ]([a-z][a-z0-9_]+)", line)
                if m:
                    return m.group(1)
    return None


def one_pass(since_min: int, do_retry: bool, rank_floor: float = 8.0) -> int:
    state = load_state()
    rows = sacct(since_min)
    n_new = 0
    for r in rows:
        st = r["state"].split()[0]
        if st not in BAD or r["jobid"] in state["seen"]:
            continue
        state["seen"][r["jobid"]] = st
        n_new += 1
        if st == "CANCELLED":
            note(state, f"{r['jobid']} {r['name']} CANCELLED (recorded, not retried)")
            continue
        ds = dataset_of(r["jobid"], r["name"])
        detail = (f"{r['jobid']} {r['name']} {st} exit={r['exit']} elapsed={r['elapsed']} "
                  f"reqmem={r['reqmem']} maxrss={r['maxrss']}" + (f" dataset={ds}" if ds else ""))
        key = f"{r['name']}:{ds or r['jobid']}"
        if st == "OUT_OF_MEMORY" and do_retry and r["name"] in RETRYABLE and ds:
            used = state["retries"].get(key, 0)
            if used >= MAX_RETRIES:
                note(state, f"{detail} -- OOM again after {used} retry; NOT retrying, needs a person")
                continue
            mem = double_mem(r["reqmem"])
            cmd = ["sbatch", f"--mem={mem}", RETRYABLE[r["name"]], ds]
            env = {"FORCE": "1"}
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                                     cwd=str(Path(__file__).resolve().parents[1]),
                                     env={**__import__("os").environ, **env})
                state["retries"][key] = used + 1
                note(state, f"{detail} -- resubmitted at {mem}: {out.stdout.strip() or out.stderr.strip()}")
            except Exception as exc:
                note(state, f"{detail} -- retry FAILED to submit: {exc}")
        else:
            why = ("not a retryable job" if r["name"] not in RETRYABLE else
                   "dataset unknown" if not ds else "not an OOM")
            note(state, f"{detail} -- recorded, not retried ({why})")

    for logf in (P / "runs").glob("*/log.jsonl"):
        try:
            if time.time() - logf.stat().st_mtime > 900:
                continue
            with open(logf, "rb") as f:
                f.seek(max(f.seek(0, 2) - 65536, 0))
                tail = f.read().decode(errors="replace").splitlines()
            ranks = []
            for line in reversed(tail):
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if "eff_rank_cell" in rec:
                    ranks.append((rec["step"], rec["eff_rank_cell"]))
                if len(ranks) >= 10:
                    break
            if len(ranks) < 10:
                continue
            mean = sum(r for _, r in ranks) / len(ranks)
            key = f"rank-collapse:{logf.parent.name}"
            if mean < rank_floor and key not in state["seen"]:
                state["seen"][key] = f"{mean:.2f}"
                note(state, f"CELL-RANK COLLAPSE in live run {logf.parent.name}: tail-10 mean "
                            f"eff_rank_cell {mean:.2f} < floor {rank_floor} at step {ranks[0][0]} "
                            f"(flagged, not killed)")
                n_new += 1
        except Exception:
            continue

    try:
        q = subprocess.run(["squeue", "--me", "-h", "-o", "%i %j %r"],
                           capture_output=True, text=True, timeout=60).stdout
        for line in q.splitlines():
            if "DependencyNeverSatisfied" in line and line.split()[0] not in state["seen"]:
                state["seen"][line.split()[0]] = "DependencyNeverSatisfied"
                note(state, f"DEPENDENCY BROKEN, job will never run: {line.strip()}")
                n_new += 1
    except Exception:
        pass

    state["last_check"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    state["queue_depth"] = len([l for l in (q or "").splitlines() if l.strip()]) if 'q' in dir() else None
    save(state)
    if not n_new:
        print(f"[watchdog] {state['last_check']}: nothing new", flush=True)
    return n_new


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", type=int, default=30, help="minutes of sacct history per pass")
    ap.add_argument("--loop", type=int, default=0, help="seconds between passes; 0 = one pass")
    ap.add_argument("--no-retry", action="store_true", help="record only, never resubmit")
    ap.add_argument("--rank-floor", type=float, default=8.0,
                    help="flag a live training run whose tail-10 mean eff_rank_cell drops below this")
    a = ap.parse_args(argv)
    if not a.loop:
        one_pass(a.since, not a.no_retry, a.rank_floor)
        return 0
    while True:
        try:
            one_pass(a.since, not a.no_retry, a.rank_floor)
        except Exception as exc:
            print(f"[watchdog] pass raised {exc!r}; continuing", flush=True)
        time.sleep(a.loop)


if __name__ == "__main__":
    raise SystemExit(main())
