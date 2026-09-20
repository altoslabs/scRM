"""The step-7 training loop."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for extra in (REPO / "model", REPO / "target" / "py", REPO / "eval"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scrm_model import optim  # noqa: E402
from scrm_model.config import (  # noqa: E402
    FLAGSHIP_RECIPE,
    SEQ_LEN,
    SEQ_LEN_OPTIONS,
    arch_dict,
    arms,
)
from scrm_model.losses import training_step  # noqa: E402
from scrm_model import masking  # noqa: E402
from scrm_model.masking import block_mask  # noqa: E402
from scrm_model.model import ScrmModel  # noqa: E402
from scrm_model.projector import build_projector  # noqa: E402
from dataclasses import replace as _dc_replace  # noqa: E402
from scrm_model.teacher import make_teacher  # noqa: E402

V_DEFAULT = 2
_STOP = {"flag": False}


def _sigterm(_sig, _frm):
    _STOP["flag"] = True


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arm", choices=tuple(arms().keys()), default=FLAGSHIP_RECIPE["arm"])
    p.add_argument("--token-passes", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=0, help="override total steps (0 = from token passes)")
    p.add_argument("--batch-cells", type=int, default=128, help="B; a tuning knob, stamped (13.0)")
    p.add_argument("--views", type=int, default=V_DEFAULT, help="V; part of the readout, not just a knob")
    p.add_argument("--w-nb", type=float, default=0.0,
                   help="weight on the NB reconstruction term over the learned molecule codebook")
    p.add_argument("--nb-codes", type=int, default=512, help="codebook size for --w-nb")
    p.add_argument("--nb-warmup", type=int, default=0,
                   help="steps to fit the codebook with w_nb=0, then freeze it (0 = never freeze)")
    p.add_argument("--w-mol", type=float, default=0.0,
                   help="weight on the per-molecule capture contrast (0 = off)")
    p.add_argument("--mol-temp", type=float, default=0.07, help="initial temperature for --w-mol")
    p.add_argument("--mol-queue", type=int, default=0,
                   help="extra negatives for --w-mol from a queue of past cell vectors (0 = off, "
                        "which is in-batch only and saturates by ~step 3,000)")

    p.add_argument("--grad-clip", type=float, default=optim.GRAD_CLIP,
                   help="gradient-norm clip; binds on ~99%% of warmup steps at the default")
    p.add_argument("--warmup-steps", type=int, default=optim.WARMUP_STEPS,
                   help="LR warmup length. ABSOLUTE, so it is 11.2%% of a 1-pass arm and 3.7%% of a "
                        "3-pass arm -- a confound in every cross-budget comparison")
    p.add_argument("--lr-start", type=float, default=optim.LR_START, help="LR at step 0")
    p.add_argument("--lr-end", type=float, default=optim.LR_END,
                   help="LR floor. Does NOT scale with --lr-max, so a peak sweep also sweeps the "
                        "peak:floor ratio -- a second variable inside the 'one-lever' LR sweep")
    p.add_argument("--wd-start", type=float, default=optim.WD_START,
                   help="weight decay at step 0 (--wd-end sets the other end)")
    p.add_argument("--beta2", type=float, default=optim.BETAS[1],
                   help="AdamW beta2. 0.95 is a documented DEPARTURE from JEPA-DNA's 0.999, whose "
                        "LR schedule and warmup this model did adopt")
    p.add_argument("--ema-m-start", type=float, default=0.996, help="EMA momentum at step 0")
    p.add_argument("--ema-m-end", type=float, default=1.0,
                   help="EMA momentum at the end; 1.0 freezes the teacher exactly at the last step")
    p.add_argument("--smooth-l1-beta", type=float, default=1.0,
                   help="L2->L1 transition for BOTH EMA terms; was torch's implicit default")
    p.add_argument("--w-commit", type=float, default=1.0,
                   help="VQ commitment weight; was an implicit 1.0 hidden inside the NB term, and "
                        "VQ-VAE's published convention is 0.25")
    p.add_argument("--nb-zero-inflated", action="store_true",
                   help="ZINB instead of NB over the codebook; over a codebook a zero has a referent")
    p.add_argument("--w-reg", type=float, default=1.0,
                   help="weight on the regulariser mix (lamb*reg + (1-lamb)*inv); the dominant trunk term")
    p.add_argument("--lamb", type=float, default=FLAGSHIP_RECIPE["lamb"],
                   help="convex mix inside the regulariser: lamb*reg + (1-lamb)*inv. "
                        "0.0196 = LeJEPA's published lambda=0.02 in this convention, "
                        "and is the flagship's")
    p.add_argument("--mol-drop", type=float, default=0.0,
                   help="uniform-random molecule dropout at the window pool, train only (0 = off)")
    p.add_argument("--w-mlm", type=float, default=1.0, help="weight on the masked-byte term")
    p.add_argument("--w-ema-umi", type=float, default=1.0, help="weight on the per-UMI EMA term")
    p.add_argument("--w-reg-cell", type=float, default=0.0,
                   help="weight on the regulariser applied to the cell embedding (0 = off, the published arms)")
    p.add_argument("--w-ema-cell", type=float, default=1.0,
                   help="weight on the cell-level EMA term (1.0 = the published arms)")
    p.add_argument("--train-frac", type=float, default=FLAGSHIP_RECIPE["train_frac"],
                   help="fraction of each cell's molecules the whole run may draw from (X1 ladder)")
    p.add_argument("--train-cells", type=int, default=FLAGSHIP_RECIPE["train_cells"],
                   help="train on this many seeded-random cells (0 = every cell; X2 ladder)")
    p.add_argument("--population-ladder", default=None,
                   help="comma-separated fracs; exclude the union of their shortfall cells (X1)")
    p.add_argument("--lr-max", type=float, default=optim.LR_MAX,
                   help="peak learning rate after warmup (default: the published 3e-4)")
    p.add_argument("--wd-end", type=float, default=optim.WD_END,
                   help="weight-decay cosine endpoint (0.4 = published; 0.04 = flat, the H4 twin)")
    p.add_argument("--seed", type=int, default=FLAGSHIP_RECIPE["seed"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--attn", default=None, choices=(None, "flex_attention", "sdpa"))
    p.add_argument("--prefetch-workers", type=int, default=3)
    p.add_argument("--prefetch-depth", type=int, default=4)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--no-warm", action="store_true", help="skip the page-cache warm-up (trap 8)")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--gradnorm-every", type=int, default=200,
                   help="per-term gradient-norm instrument cadence (13.12); 0 disables")
    p.add_argument("--seq-len", type=int, default=SEQ_LEN, choices=SEQ_LEN_OPTIONS)
    p.add_argument("--init-from", type=Path, default=None,
                   help="checkpoint to initialise weights from (fresh optimizer and schedule)")
    return p.parse_args(argv)


def run_frac_seed(seed: int) -> int:
    """The read-budget subset's stream, derived from the run seed under its own domain ("frac") so
    it can never collide with an epoch's molecule draw. ONE definition, used by the eligibility
    guard and the batch producer alike.
    """
    return (seed ^ 0x66726163) % (1 << 64)


def epoch_seed_at(seed: int, epoch: int) -> int:
    """The per-epoch stream. ONE definition, so the eligibility guard replays exactly the seeds the
    run will use.
    """
    return (seed * 0x9E3779B97F4A7C15 + epoch) % (1 << 64)


def eligible_cells(corpus, args, total_steps: int, log) -> np.ndarray:
    """Which cells this run may draw from: the --train-cells subset minus every cell that cannot
    supply V windows under ANY epoch draw the run will actually perform, over the read-budget
    subset(s) of --population-ladder (default: this run's own --train-frac).
    """
    pool = np.arange(corpus.n_cells(), dtype=np.int64)
    if args.train_cells:
        rng = np.random.default_rng((args.seed, 0xCE115))
        pool = np.sort(rng.choice(pool, size=min(args.train_cells, len(pool)), replace=False))
        log(f"train-cells: {len(pool)} of {corpus.n_cells()} cells (seeded subset)")
    fracs = ([float(x) for x in args.population_ladder.split(",")] if args.population_ladder
             else [args.train_frac])
    if args.population_ladder and args.train_frac not in fracs:
        raise SystemExit(f"--population-ladder {fracs} does not contain --train-frac "
                         f"{args.train_frac}: this run would train on a rung its own population "
                         "matching does not cover")
    fs = run_frac_seed(args.seed)
    while True:
        per_epoch = max(len(pool) // args.batch_cells, 1)
        seeds = [epoch_seed_at(args.seed, e) for e in range((total_steps - 1) // per_epoch + 2)]
        below: set[int] = set()
        for f in fracs:
            below |= set(int(c) for c in corpus.cells_below_v(seeds, v=args.views, frac=f, frac_seed=fs))
        keep = pool[~np.isin(pool, np.fromiter(below, dtype=np.int64, count=len(below)))] if below else pool
        if len(keep) == len(pool):
            return keep
        log(f"eligibility: excluding {len(pool) - len(keep)} cells that cannot supply v={args.views} "
            f"windows under the run's own draws (ladder {fracs}); re-deriving the epoch count")
        pool = keep


def warm_page_cache(corpus_dirs: list[str], log, threads: int = 8) -> None:
    """Read every shard once so a cold page cache cannot stall early steps."""
    t0 = time.perf_counter()
    shards = [s for d in corpus_dirs for s in sorted(Path(d).glob("*.arrow"))]

    def read(shard: Path) -> None:
        with open(shard, "rb", buffering=0) as f:
            while f.read(1 << 24):
                pass

    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(read, shards))
    log(f"warm: {len(shards)} shards in {time.perf_counter() - t0:.1f}s")


def make_batch_producer(corpus, cells_eligible: np.ndarray, args, total_steps: int, *, rank: int = 0, world: int = 1):
    """step -> pinned torch batch, pure function of (seed, step): safe to run out of order in
    worker threads and consumed strictly in order by the loop.
    """
    b_n = args.batch_cells
    assert b_n % world == 0, "global batch must split evenly across ranks"
    b_local = b_n // world
    if len(cells_eligible) < b_n:
        raise SystemExit(
            f"{len(cells_eligible)} eligible cells is fewer than --batch-cells {b_n}: an epoch "
            "cannot be formed. Lower --batch-cells, or train on a corpus with more cells that can "
            f"supply v={args.views} windows."
        )
    per_epoch = len(cells_eligible) // b_n
    pin = args.device.startswith("cuda")

    def produce(step: int) -> dict[str, torch.Tensor]:
        epoch, slot = divmod(step, per_epoch)
        rng = np.random.default_rng((args.seed, epoch))
        order = rng.permutation(cells_eligible)
        cells = order[slot * b_n : (slot + 1) * b_n][rank * b_local : (rank + 1) * b_local]
        epoch_seed = epoch_seed_at(args.seed, epoch)
        raw = corpus.build_batch([int(c) for c in cells], epoch_seed=epoch_seed, v=args.views,
                                 frac=args.train_frac, frac_seed=run_frac_seed(args.seed))
        np_batch = {
            "tokens": np.asarray(raw.tokens),
            "valid": np.asarray(raw.valid),
            "contig_segment_id": np.asarray(raw.contig_segment_id),
            "umi_segment_id": np.asarray(raw.umi_segment_id),
            "n_patches": np.asarray(raw.n_patches),
            "cell_index": np.asarray(raw.cell_index),
        }
        lo, hi = optim.mask_ratio_at(step, total_steps)
        m = block_mask(
            np_batch, epoch_seed=epoch_seed, v=args.views, min_ratio=lo, max_ratio=hi,
            compute_bins=step % 100 == 0
        )
        out = {
            "tokens": torch.from_numpy(np_batch["tokens"]),
            "tokens_student": torch.from_numpy(m.tokens_student),
            "mlm_keep": torch.from_numpy(m.mlm_keep),
            "valid": torch.from_numpy(np_batch["valid"]),
            "contig_segment_id": torch.from_numpy(np_batch["contig_segment_id"].view(np.int16)),
            "umi_segment_id": torch.from_numpy(np_batch["umi_segment_id"].view(np.int16)),
            "n_patches": torch.from_numpy(np_batch["n_patches"].astype(np.int64)),
        }
        if pin:
            out = {k: v.pin_memory() for k, v in out.items()}
        out["_realised_ratio"] = torch.from_numpy(m.realised_ratio)
        out["_realised_by_length"] = torch.from_numpy(m.realised_by_length)
        return out

    return produce


def assert_schedule_fits(total_steps: int, warmup: int, steps_per_pass: int, token_passes: float,
                         *, explicit_steps: bool) -> None:
    """TOTAL_STEPS MUST CLEAR THE WARMUP THIS RUN ACTUALLY USES, asserted rather than left to the
    caller.
    """
    if explicit_steps or total_steps >= 2 * warmup:
        return
    raise SystemExit(
        f"total_steps {total_steps} is below 2x this run's LR warmup ({warmup}): this corpus "
        f"gives {steps_per_pass} steps per token pass, so --token-passes {token_passes} "
        f"cannot complete the schedule. Raise --token-passes to at least "
        f"{max(1, -(-2 * warmup // max(steps_per_pass, 1)))}, lower --warmup-steps, or pass "
        "--steps explicitly if a short run is deliberate."
    )


def assert_startup_couplings(args, world: int) -> None:
    """Assert every cross-parameter coupling at startup."""
    if args.mol_drop > 0.0 and world > 1:
        raise SystemExit(
            f"--mol-drop {args.mol_drop} with world_size {world}: the dropout draw's size comes "
            "from the rank-local molecule maximum, so the ranks' global RNG streams diverge -- the "
            "/world gradient average is then not demonstrably the single-process gradient. Run "
            "mol-drop on one rank (every trained mol-drop arm did), or make the draw rank-invariant "
            "deliberately."
        )


def checkpoint_payload(step, *, student, projector, teacher, opt, reg, nb_modules, mol_module,
                       args, cfg, total_steps, world, corpus_compat_key=None) -> dict:
    """Everything a checkpoint must carry, as a dict -- ONE construction."""
    import hashlib

    sd = getattr(student, "_orig_mod", student).state_dict()
    arch = arch_dict(cfg, args.arm)
    readout = {
        "levels": ["bytes->contig", "contig->umi", "umi->window",
                   f"views->group{args.views}"],
        "pool": {"attention": "seed_attention", "mean_late": "seed_attention_early_mean_late",
                 "mean_all": "mean_all"}[cfg.pool_variant],
        "window_to_cell": "group_view_pool_then_mean",
        "views": args.views,
        "grouping": f"adjacent_v{args.views}",
    }
    payload = {
            "arch": arch,
            "readout": readout,
            "arch_sha": hashlib.sha256(
                __import__("json").dumps(arch, sort_keys=True).encode()
            ).hexdigest(),
            "state_dict": sd,
            "torch_version": torch.__version__,
            "student": sd,
            "projector": projector.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": opt.state_dict(),
            "nb_codebook": nb_modules[0].state_dict() if nb_modules is not None else None,
            "nb_recon": nb_modules[1].state_dict() if nb_modules is not None else None,
            "step": step,
            "arm": args.arm,
            "regularizer": REGULARIZER,
            "w_reg_cell": args.w_reg_cell,
            "w_reg": args.w_reg,
            "w_mlm": args.w_mlm,
            "mol_drop": args.mol_drop,
            "lamb": args.lamb,
            "w_ema_umi": args.w_ema_umi,
            "w_nb": args.w_nb,
            "nb_codes": args.nb_codes if args.w_nb else None,
            "nb_warmup": args.nb_warmup,
            "w_mol": args.w_mol,
            "mol_temp": args.mol_temp,
            "mol_queue": args.mol_queue,
            "grad_clip": args.grad_clip,
            "warmup_steps": args.warmup_steps,
            "lr_start": args.lr_start,
            "lr_end": args.lr_end,
            "wd_start": args.wd_start,
            "beta2": args.beta2,
            "ema_m_start": args.ema_m_start,
            "ema_m_end": args.ema_m_end,
            "smooth_l1_beta": args.smooth_l1_beta,
            "w_commit": args.w_commit,
            "mol_contrast": mol_module.state_dict() if mol_module is not None else None,
            "nb_zero_inflated": bool(args.nb_zero_inflated) if args.w_nb else None,
            "w_ema_cell": args.w_ema_cell,
            "train_frac": args.train_frac,
            "train_cells": args.train_cells,
            "population_ladder": args.population_ladder,
            "wd_end": args.wd_end,
            "lr_max": args.lr_max,
            "train_seq_len": args.seq_len,
            "mask_block_floor": masking._BLOCK_FLOOR,
            "mask_max_regions": masking._MAX_REGIONS,
            "mask_ratio_start": list(optim.mask_ratio_at(0, total_steps)),
            "mask_ratio_end": list(optim.mask_ratio_at(total_steps, total_steps)),
            "seed": args.seed,
            "total_steps": total_steps,
            "batch_cells": args.batch_cells,
            "world": world,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "reg": reg.state_dict(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "corpus_compat_key": corpus_compat_key,
    }
    return payload

def save_checkpoint(path: Path, step: int, **kw) -> None:
    """`checkpoint_payload` to disk, tmp-then-rename."""
    tmp = path.with_suffix(".tmp")
    torch.save(checkpoint_payload(step, **kw), tmp)
    tmp.rename(path)


def restore_checkpoint(state: dict, *, student, projector, teacher, opt, reg, nb_modules,
                       mol_module, args, source: str = "the checkpoint", log=print) -> int:
    """The READ side of `checkpoint_payload`. Returns the step to resume at."""
    student.load_state_dict(state["student"])
    projector.load_state_dict(state["projector"])
    teacher.load_state_dict(state["teacher"])
    optim.load_optimizer_state(opt, state["optimizer"])
    if state.get("regularizer") not in (None, REGULARIZER):
        raise SystemExit(
            f"{source} was trained with the {state['regularizer']} objective and this tree trains "
            f"{REGULARIZER}: resuming would continue one objective under another. Start a new run "
            "directory."
        )
    if state.get("reg"):
        reg.load_state_dict(state["reg"])
    if nb_modules is not None:
        if state.get("nb_codebook") is None:
            raise SystemExit(
                f"{source} has no nb_codebook but this run sets --w-nb {args.w_nb}: resuming would "
                "restart the codebook from random init against a trained encoder. Delete the run "
                "directory to start clean, or resume a checkpoint written after the nb_codebook fix."
            )
        nb_modules[0].load_state_dict(state["nb_codebook"])
        nb_modules[1].load_state_dict(state["nb_recon"])
    if mol_module is not None:
        if state.get("mol_contrast") is None:
            raise SystemExit(
                f"{source} has no mol_contrast but this run sets --w-mol {args.w_mol}: resuming "
                "would restart the learned temperature at log(--mol-temp) and the negative queue "
                "empty, against Adam moments carried over from the old ones. Delete the run "
                "directory to start clean, or resume a checkpoint written after this fix."
            )
        mol_module.load_state_dict(state["mol_contrast"])
    if state.get("rng_state") is not None:
        torch.set_rng_state(state["rng_state"].cpu())
        cuda_state = state.get("cuda_rng_state")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_state.cpu())
    return state["step"]


def snapshot_at_pass_boundary(step_done: int, steps_per_pass: int, save, latest: Path, out: Path):
    """At a token-pass boundary, checkpoint AT the boundary and file the copy as `pass{n}.pt`."""
    if not steps_per_pass or step_done % steps_per_pass:
        return None
    import shutil

    save(step_done)
    dest = out / f"pass{step_done // steps_per_pass}.pt"
    shutil.copy2(latest, dest)
    return dest


def allreduce_grads(grad_params: list[torch.Tensor], world: int) -> None:
    """Average gradients across ranks, flattened into one bucket (13.6 M params ≈ 27 MB bf16)."""
    with_grad = [p.grad for p in grad_params if p.grad is not None]
    flat = torch._utils._flatten_dense_tensors(with_grad)
    torch.distributed.all_reduce(flat)
    flat /= world
    for g, synced in zip(with_grad, torch._utils._unflatten_dense_tensors(flat, with_grad)):
        g.copy_(synced)


_LONG_KEYS = ("tokens", "tokens_student", "contig_segment_id", "umi_segment_id")


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """H2D at wire width, widen to long on device (embedding and gather want int64)."""
    out = {}
    for k, v in batch.items():
        t = v.to(device, non_blocking=True)
        out[k] = t.long() if k in _LONG_KEYS else t
    return out


#: The objective. NOT a flag: invariant 5 says flags expose tuning parameters, and one with a single
#: choice tunes nothing -- it only invites a reader to think the alternative is in the tree, which is
#: how a reviewer read `choices=("sigreg",)` as a missing capability. Recorded in every checkpoint,
#: and the resume guard below refuses a payload trained under anything else.
REGULARIZER = FLAGSHIP_RECIPE["regularizer"]


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "log.jsonl"

    import os

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        import torch.distributed as dist

        dist.init_process_group("nccl" if args.device.startswith("cuda") else "gloo")
        if args.device.startswith("cuda"):
            local = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local)
            args.device = f"cuda:{local}"

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    import scrm_py

    signal.signal(signal.SIGTERM, _sigterm)
    device = torch.device(args.device)

    cfg = arms()[args.arm]
    if args.attn:
        cfg = type(cfg)(**{**cfg.__dict__, "attn_impl": args.attn})
    elif not device.type.startswith("cuda"):
        raise SystemExit("training at the real shape needs flex_attention on GPU; pass --attn to override")

    corpus = scrm_py.Corpus([str(c) for c in args.corpus], seq_len=args.seq_len, min_v=args.views)
    n_tokens = sum(json.loads((Path(d) / "manifest.json").read_text())["n_tokens"] for d in args.corpus)
    tokens_per_step = args.batch_cells * args.views * args.seq_len
    steps_per_pass = max(1, round(n_tokens / tokens_per_step))
    total_steps = args.steps or max(1, round(args.token_passes * steps_per_pass))
    assert_schedule_fits(total_steps, args.warmup_steps, steps_per_pass, args.token_passes,
                         explicit_steps=bool(args.steps))
    assert_startup_couplings(args, world)
    cells_eligible = eligible_cells(corpus, args, total_steps, log)
    log(
        f"arm {args.arm}: {len(cells_eligible)} eligible of {corpus.n_cells()} cells "
        f"(subset and guard exclusions logged above; train_frac={args.train_frac}), "
        f"{n_tokens:,} corpus tokens, {tokens_per_step:,} grad tokens/step, "
        f"{steps_per_pass} steps/token-pass, total_steps {total_steps}"
    )

    torch.manual_seed(args.seed)
    student = ScrmModel(cfg).to(device)
    projector = build_projector(cfg).to(device)
    from scrm_model.losses import SIGReg

    reg = SIGReg(num_slices=4 * cfg.proj_dim)
    reg = reg.to(device)
    from scrm_model.losses import Weights as _Weights
    loss_weights = _Weights(w_reg_cell=args.w_reg_cell, w_ema_cell=args.w_ema_cell,
                            w_nb=args.w_nb, w_reg=args.w_reg, w_mlm=args.w_mlm,
                            w_ema_umi=args.w_ema_umi, lamb=args.lamb, w_mol=args.w_mol,
                            smooth_l1_beta=args.smooth_l1_beta, w_commit=args.w_commit)
    nb_modules = None
    if args.w_nb:
        from scrm_model.nbhead import MoleculeCodebook, NBReconstruction

        nb_modules = (
            MoleculeCodebook(cfg.dim_global, codes=args.nb_codes).to(device),
            NBReconstruction(cfg.dim_global, args.nb_codes,
                             zero_inflated=args.nb_zero_inflated).to(device),
        )
    mol_module = None
    if args.w_mol:
        from scrm_model.molcontrast import MoleculeCellContrast

        mol_module = MoleculeCellContrast(cfg.dim_global, temp=args.mol_temp, queue=args.mol_queue).to(device)
    student.mol_drop = args.mol_drop
    teacher = make_teacher(student, total_steps=total_steps,
                           m_start=args.ema_m_start, m_end=args.ema_m_end)
    teacher.averaged.to(device)
    teacher.model.mol_drop = 0.0
    opt = optim.build_optimizer(student, projector, fused=device.type == "cuda",
                                wd_start=args.wd_start, lr_start=args.lr_start,
                                betas=(optim.BETAS[0], args.beta2))
    grad_params = [p for g in opt.param_groups for p in g["params"]]
    if mol_module is not None:
        mol_extra = [p for p in mol_module.parameters() if p.requires_grad]
        optim.add_undecayed_group(opt, mol_extra)
        grad_params += mol_extra
    if nb_modules is not None:
        extra = [p for p in nb_modules[1].parameters() if p.requires_grad]
        optim.add_undecayed_group(opt, extra)
        grad_params += extra

    start_step = 0
    latest = args.out / "latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        start_step = restore_checkpoint(
            state, student=student, projector=projector, teacher=teacher, opt=opt, reg=reg,
            nb_modules=nb_modules, mol_module=mol_module, args=args, source=str(latest), log=log)
        log(f"resumed at step {start_step} from {latest}")
    elif args.init_from is not None:
        init = torch.load(args.init_from, map_location=device, weights_only=False)
        if init.get("arch", {}).get("arm") not in (None, args.arm):
            raise SystemExit(
                f"--init-from holds arm {init['arch']['arm']}, this run is arm {args.arm}: "
                "loading mismatched weights would silently reshape the model"
            )
        student.load_state_dict(init["student"])
        projector.load_state_dict(init["projector"])
        teacher.load_state_dict(init["teacher"])
        log(f"initialised weights from {args.init_from} (step {init.get('step')}); fresh schedule")
        if not args.no_warm:
            warm_page_cache([str(c) for c in args.corpus], log)
    else:
        teacher.update(student, step=0)
        if world > 1:
            import torch.distributed as dist

            probe = torch.cat([p.detach().flatten()[:1] for p in student.parameters()])[:64].clone()
            ref = probe.clone()
            dist.broadcast(ref, src=0)
            assert torch.equal(probe, ref), "rank model inits diverged; seeding is broken"
        if not args.no_warm and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            warm_page_cache([str(c) for c in args.corpus], log)
    if world > 1:
        torch.distributed.barrier()

    # THE INSTRUMENT DECIDES THIS, NOT --no-compile. The gradient-norm instrument calls
    # autograd.grad(retain_graph=True) before the combined backward, and a compiled backward with
    # donated buffers forbids any retain_graph. `trunk._FLEX` and `model._BLOCK_MASK_BUILDER` are
    # compiled at import, so a backward is compiled even under --no-compile: gating the mitigation
    # on the flag made `--no-compile` with the default --gradnorm-every crash at the first
    # instrumented step. Disabling donation costs a little activation memory and nothing else.
    if args.gradnorm_every and device.type == "cuda":
        torch._functorch.config.donated_buffer = False
    if not args.no_compile and device.type == "cuda":
        student.compile_blocks()
        teacher.model.compile_blocks()

    def save(step):
        save_checkpoint(latest, step, student=student, projector=projector, teacher=teacher,
                        opt=opt, reg=reg, nb_modules=nb_modules, mol_module=mol_module,
                        args=args, cfg=cfg, total_steps=total_steps, world=world,
                        corpus_compat_key=dict(corpus.compat_key()))

    produce = make_batch_producer(corpus, cells_eligible, args, total_steps, rank=rank, world=world)
    pool = ThreadPoolExecutor(max_workers=args.prefetch_workers)
    window: dict[int, object] = {}

    def ensure_submitted(frm: int, upto: int):
        for s in range(frm, min(upto, total_steps)):
            if s not in window:
                window[s] = pool.submit(produce, s)

    log_f = open(log_path, "a") if rank == 0 else None
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    step = start_step
    try:
        while step < total_steps:
            if world > 1:
                flag = torch.tensor([int(_STOP["flag"])], device=device)
                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
                if bool(flag.item()):
                    break
            elif _STOP["flag"]:
                break
            ensure_submitted(step, step + args.prefetch_depth)
            t0 = time.perf_counter()
            batch = window.pop(step).result()
            data_wait_ms = (time.perf_counter() - t0) * 1e3

            realised = batch.pop("_realised_ratio")
            by_length = batch.pop("_realised_by_length")
            dev_batch = to_device(batch, device)

            lr, wd = optim.apply_schedules(
                opt, step, total_steps, args.wd_end, args.lr_max,
                warmup=args.warmup_steps, lr_start=args.lr_start,
                lr_end=args.lr_end, wd_start=args.wd_start)
            step_weights = loss_weights
            if args.w_nb and args.nb_warmup:
                if step < args.nb_warmup:
                    step_weights = _dc_replace(loss_weights, w_nb=0.0)
                elif nb_modules is not None and not bool(nb_modules[0].frozen):
                    nb_modules[0].frozen.fill_(True)
                    log(f"codebook frozen at step {step} after {args.nb_warmup} warmup steps; "
                        f"w_nb -> {args.w_nb}")
            with autocast:
                out = training_step(student, teacher.model, projector, reg, dev_batch,
                                    v=args.views, weights=step_weights, nb=nb_modules, mol=mol_module)
            gnorms = {}
            if args.gradnorm_every and step % args.gradnorm_every == 0:
                from scrm_model import guards as _guards

                groups = _guards.parameter_groups(getattr(student, "_orig_mod", student), projector,
                                                  nb=nb_modules, mol=mol_module)
                gnorms = _guards.per_term_gradient_norms(out.terms, groups)
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            if world > 1:
                allreduce_grads(grad_params, world)
            grad_norm = torch.nn.utils.clip_grad_norm_(grad_params, args.grad_clip)
            opt.step()
            momentum = teacher.update(getattr(student, "_orig_mod", student), step + 1)

            step_ms = (time.perf_counter() - t0) * 1e3
            logs = {**out.logs, "loss": float(out.loss.detach())}
            if world > 1:
                keys = sorted(k for k, v in logs.items() if isinstance(v, float))
                vals = torch.tensor([logs[k] for k in keys], device=device, dtype=torch.float64)
                torch.distributed.all_reduce(vals)
                logs = {**logs, **dict(zip(keys, (vals / world).tolist()))}
            rec = {
                "step": step,
                **logs,
                "lr": lr,
                "wd": wd,
                "ema_momentum": momentum,
                "grad_norm": float(grad_norm),
                "realised_mask_ratio": float(realised.mean()),
                "data_wait_ms": round(data_wait_ms, 3),
                "host_rss_gb": round(int(open("/proc/self/statm").read().split()[1]) * 4096 / 2**30, 2),
                "step_ms": round(step_ms, 2),
            }
            if step % 100 == 0:
                rec["realised_by_length"] = by_length.tolist()
                log(
                    f"step {step}/{total_steps} loss {rec['loss']:.4f} mlm {rec['loss_mlm']:.4f} "
                    f"rank_w {rec['eff_rank_window']:.1f} rank_c {rec['eff_rank_cell']:.1f} "
                    f"wait {data_wait_ms:.1f}ms step {step_ms:.0f}ms"
                )
            rec.update(gnorms)
            if log_f is not None:
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()

            if (step + 1) % args.ckpt_every == 0 and rank == 0:
                save(step + 1)
            if rank == 0:
                snapshot_at_pass_boundary(step + 1, steps_per_pass, save, latest, args.out)
            step += 1
    finally:
        if rank == 0:
            save(step)
            log_f.close()
        pool.shutdown(wait=False, cancel_futures=True)
        if world > 1:
            torch.distributed.destroy_process_group()

    log(f"stopped at step {step}/{total_steps}" + (" (SIGTERM)" if _STOP["flag"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
