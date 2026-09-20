"""Derive the three training corpus sets, and write them where a launcher can read them."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "eval")]
from scrm_eval import registry  # noqa: E402
from scrm_eval.registry import DATASETS  # noqa: E402

P = registry.PROJ
TABLE = REPO / "model" / "jobs" / "tenx_pretrain_datasets.tsv"
PINNED = REPO / "crates" / "scrm-corpus" / "tests" / "expected_stream_digests.json"
DEST = REPO / "model" / "jobs" / "training_sets.json"

EXISTING_MOUSE = {"mouse_brain_5k": "brain", "neuron_1k_v3": "neuron",
                  "neurons_5k": "neuron", "sod1_spinal_cord": "spinal_cord"}

FORBIDDEN = {d for d, _, role in DATASETS if role.startswith("eval")} | {"crispr_a549_5k"}


def tokens(key: str) -> int | None:
    f = registry.corpus_root() / key / "manifest.json"
    if not f.exists():
        return None
    return int(json.loads(f.read_text())["n_tokens"])


def inventory() -> list[dict]:
    """Every buildable corpus, with the cell count MEASURED from the corpus, not the whitelist."""
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "scrm_pinned_digests", REPO / "crates/scrm-corpus/tests/pinned_digests.py")
    _pd = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_pd)
    trees = _pd.load_pinned()["trees"]
    pin = {}
    for _t in trees:
        pin.update(_t["corpora"])
    out = []
    for r in csv.DictReader(TABLE.open(), delimiter="\t"):
        t = tokens(r["key"])
        if t:
            measured = r["key"] in pin
            out.append(dict(key=r["key"], species=r["species"], tissue=r["tissue"],
                            cells=int(pin[r["key"]]["cells"]) if measured else int(r["cells"]),
                            cells_measured=measured, tokens=t))
    for k, tis in EXISTING_MOUSE.items():
        t = tokens(k)
        if t:
            out.append(dict(key=k, species="mouse", tissue=tis,
                            cells=int(pin[k]["cells"]) if k in pin else 0, tokens=t))
    return out


def diversity_first(rows: list[dict], budget: int, tol: float = 1.02) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["species"], r["tissue"])].append(r)
    for g in groups.values():
        g.sort(key=lambda r: r["tokens"])
    order = sorted(groups)
    sel: list[dict] = []
    total = rnd = 0
    while total < budget:
        added = False
        for g in order:
            if rnd >= len(groups[g]):
                continue
            r = groups[g][rnd]
            if total + r["tokens"] > budget * tol:
                continue
            sel.append(r)
            total += r["tokens"]
            added = True
            if total >= budget:
                break
        if not added:
            break
        rnd += 1
    return sel


def summarise(name: str, sel: list[dict]) -> dict:
    sp = sorted({r["species"] for r in sel})
    ti = sorted({r["tissue"] for r in sel})
    tot = sum(r["tokens"] for r in sel)
    return {"name": name,
            "corpora": [r["key"] for r in sorted(sel, key=lambda r: r["key"])],
            "n_corpora": len(sel),
            "cells": sum(r["cells"] for r in sel),
            "tokens": tot,
            "steps_one_pass": round(tot / 4_194_304),
            "species": sp, "tissues": ti}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)

    rows = inventory()
    mouse = [r for r in rows if r["species"] == "mouse"]
    budget = sum(r["tokens"] for r in mouse)

    human = [r for r in rows if r["species"] == "human"]
    sets = {"mouse56": summarise("mouse56", mouse),
            "mixed56": summarise("mixed56", diversity_first(rows, budget)),
            "mixedfull": summarise("mixedfull", rows),
            "human": summarise("human", human)}

    for name, s in sets.items():
        bad = sorted(set(s["corpora"]) & FORBIDDEN)
        if bad:
            raise SystemExit(f"{name} contains scored dataset(s) {bad}: that is a leak, not a corpus")
        print(f"{name:11s} {s['n_corpora']:2d} corpora  {s['cells']:>7,} cells  "
              f"{s['tokens']/1e9:7.2f} B tokens  {s['steps_one_pass']:>6,} steps/pass")
        print(f"            species {','.join(s['species'])}")
        print(f"            tissues {','.join(s['tissues'])}")
    m56, x56 = sets["mouse56"]["tokens"], sets["mixed56"]["tokens"]
    print(f"\ntoken match: mixed56 is {100*x56/m56:.1f}% of mouse56 "
          f"({sets['mixed56']['cells']:,} vs {sets['mouse56']['cells']:,} cells)")
    print(f"scaling    : mixedfull is {sets['mixedfull']['tokens']/m56:.2f}x mouse56")

    if a.write:
        payload = {"_why": "Derived by measurements/training_sets.py from "
                           "model/jobs/tenx_pretrain_datasets.tsv and the corpora on disk. Do not "
                           "hand-edit: re-run with --write so the sets stay re-derivable.",
                   "sets": sets}
        tmp = DEST.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(tmp, DEST)
        print(f"\nwrote {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
