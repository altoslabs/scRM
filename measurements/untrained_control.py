#!/usr/bin/env python3
"""Write random-init checkpoints at the flagship's architecture: the untrained control."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
from scrm_eval import registry  # noqa: E402

PROJ = registry.PROJ

INIT_SEEDS = (20260901, 20260902, 20260903)

def run_name(arm: str, n: int) -> str:
    return f"arm{arm}-untrained-i{n}"


def template_arm(raw: dict, run: str, registry) -> str:
    """The ArchConfig arm this control IS, taken from the template and cross-checked, not assumed."""
    arm = raw.get("arm")
    if not arm:
        raise SystemExit(
            f"the template checkpoint for {run!r} records no 'arm', so the control's own "
            "architecture cannot be named from it. Refusing to guess (a typed arm is what this "
            "function exists to remove)."
        )
    if run in registry.runs():
        declared = registry.arms_for([run])[0]
        if declared != arm:
            raise SystemExit(
                f"{run!r}: the checkpoint says arm {arm!r} but the registry declares {declared!r}. "
                "One of the two is wrong; a control copied from this template would inherit the "
                "disagreement."
            )
    return str(arm)


def assert_registered(names: list[str], registry) -> None:
    """Every control name must be a NEGATIVE CONTROL to the registry before it can be written."""
    unknown = [n for n in names if not registry.is_control(n)]
    if unknown:
        raise SystemExit(
            "registry.is_control() is False for " + ", ".join(unknown) + ".\n"
            "These would be scored as legitimate trained arms and could win a 'our best' maximum.\n"
            "Add each to registry.CONTROL_ARMS and give it a registry.TRAINING record with steps=0 "
            "and an OPTIONAL_RUNS entry, then re-run. (The registry currently enumerates the S-128 "
            "names.)"
        )


def _tensor_differs(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.numel() <= 1 or not torch.equal(a, b)


def build(flagship: Path, seed: int):
    """`(payload, flagship_raw)` for one init seed, with the architecture copied verbatim."""
    sys.path[:0] = [str(REPO / "model"), str(REPO / "eval"), str(REPO / "target" / "py")]
    from scrm_eval.embed import arch_digest
    from scrm_model.config import ArchConfig
    from scrm_model.model import ScrmModel

    raw = torch.load(flagship, map_location="cpu", weights_only=False)
    for field in ("arch", "readout", "arch_sha", "state_dict"):
        if field not in raw:
            raise SystemExit(f"{flagship} has no {field!r}; it cannot be a control's template")

    fields = {k: v for k, v in raw["arch"].items() if k in ArchConfig.__dataclass_fields__}
    torch.manual_seed(seed)
    model = ScrmModel(ArchConfig(**fields))
    sd = model.state_dict()

    payload = {
        "arch": raw["arch"],
        "readout": raw["readout"],
        "arch_sha": raw["arch_sha"],
        "state_dict": sd,
        "step": 0,
        "total_steps": 0,
        "arm": raw.get("arm"),
        "seed": seed,
        "untrained": True,
        "derived_from": str(flagship),
        "torch_version": torch.__version__,
    }
    if arch_digest(payload["arch"]) != payload["arch_sha"]:
        raise SystemExit("arch_sha does not recompute from the copied arch; refusing to write")
    return payload, raw


def check(payload: dict, flagship_raw: dict, others: list[dict]) -> None:
    fsd, sd = flagship_raw["state_dict"], payload["state_dict"]
    if set(sd) != set(fsd):
        raise SystemExit(f"state_dict keys differ from the flagship's: {set(sd) ^ set(fsd)}")
    for k in sd:
        if sd[k].shape != fsd[k].shape:
            raise SystemExit(f"{k}: shape {tuple(sd[k].shape)} != flagship {tuple(fsd[k].shape)}")
    same = [k for k in sd if not _tensor_differs(sd[k], fsd[k])]
    if same:
        raise SystemExit(
            f"{len(same)} tensor(s) are IDENTICAL to the flagship's trained weights, e.g. {same[:5]}. "
            "This control would score like the flagship and be read as 'pretraining does nothing'."
        )
    const = [k for k in sd if sd[k].numel() > 1 and sd[k].min() == sd[k].max()]
    for other in others:
        dup = [k for k in sd if k not in const and not _tensor_differs(sd[k], other["state_dict"][k])]
        if dup:
            raise SystemExit(
                f"this init shares {len(dup)} NON-constant tensor(s) with init seed {other['seed']}, "
                f"e.g. {dup[:5]} -- three identical files are one init reported as three"
            )
    if others:
        print(f"      cross-init: {len(sd) - len(const)} random tensors all differ; "
              f"{len(const)} constant-initialised tensors exempt (all-ones norms)")


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="write the checkpoints (default: check only)")
    ap.add_argument("--flagship", default=None, help="run name to copy the architecture from")
    args = ap.parse_args(argv[1:])

    sys.path[:0] = [str(REPO / "model"), str(REPO / "eval"), str(REPO / "target" / "py")]
    from scrm_eval import registry

    run = args.flagship or registry.FLAGSHIP
    flagship = PROJ / "runs" / run / registry.reported_ckpt(run)
    if not flagship.is_file():
        raise SystemExit(f"{flagship} does not exist")
    print(f"template: {flagship}")

    built = []
    arm = None
    for n, seed in enumerate(INIT_SEEDS, start=1):
        payload, raw = build(flagship, seed)
        check(payload, raw, [b for b, _ in built])
        if arm is None:
            arm = template_arm(raw, run, registry)
            print(f"arm:      {arm} (from the template checkpoint, cross-checked against the registry)")
        name = run_name(arm, n)
        built.append((payload, name))
        n_params = sum(t.numel() for t in payload["state_dict"].values())
        print(f"  {name}: seed {seed}  {len(payload['state_dict'])} tensors  {n_params:,} params  "
              f"arch_sha {payload['arch_sha'][:16]}…  all tensors differ from the flagship's")

    assert_registered([nm for _, nm in built], registry)

    if not args.write:
        print("\ncheck only; pass --write to create the run directories")
        return 0

    for payload, name in built:
        d = PROJ / "runs" / name
        d.mkdir(parents=True, exist_ok=True)
        out = d / "latest.pt"
        tmp = out.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(out)
        sha = hashlib.sha256(out.read_bytes()).hexdigest()
        print(f"  wrote {out}  {out.stat().st_size / 2**20:.1f} MiB  sha256 {sha[:16]}…")

    from scrm_eval import embed as ev
    from scrm_model.evalio import model_factory

    for _, name in built:
        ck = ev.Checkpoint.load(PROJ / "runs" / name / registry.reported_ckpt(name),
                                model_factory=model_factory)
        print(f"  Checkpoint.load OK: {name}  dim={ck.arch.get('dim_global')}  "
              f"readout.views={ck.readout.get('views')}  ckpt_sha={ck.checkpoint_sha256[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
