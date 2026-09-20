"""`scrm-embed` — checkpoint in, stamped per-window artifact out."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from . import artifact, controls


class EmbedError(Exception):
    """A checkpoint that cannot be embedded from."""


BATCHING_FIELDS = ("batch_cells", "window_batch")


def _identity(key: dict) -> dict:
    """The part of a resume key that says which computation this is."""
    return {k: v for k, v in key.items() if k not in BATCHING_FIELDS}


class _Resume:
    """Partial results on disk, so a preempted embedding restarts where it stopped."""

    def __init__(self, root, key: dict, interval_s: float = 120.0):
        self.root = Path(root)
        self.key = key
        self.interval = float(interval_s)
        self._t = time.monotonic()
        self._t0 = self._t
        self._cells = 0
        self._blocks = 0
        self._next = 0
        self._flushes = 0

    STALE_AFTER_S: float = 900.0

    def _touch_owner(self, me: str | None = None) -> None:
        """Refresh the heartbeat. Written atomically so a reader never sees an empty owner file."""
        f = getattr(self, "_owner_file", None) or (self.root / "owner")
        tmp = f.with_name("owner.tmp")
        tmp.write_text((me or os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}") + "\n")
        os.replace(tmp, f)

    def _fingerprint(self) -> str:
        """A digest over `_identity`, not over the whole key."""
        return hashlib.sha256(json.dumps(_identity(self.key), sort_keys=True).encode()).hexdigest()[:16]

    def load(self, out_cells: list, out_windows: list, blocks: list) -> int:
        """Replay the contiguous prefix already on disk; return the cell index to continue from."""
        self.root.mkdir(parents=True, exist_ok=True)

        me = os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}"
        owner_file = self.root / "owner"
        prior = owner_file.read_text().strip() if owner_file.exists() else None
        if prior is not None and prior != me:
            age = time.time() - owner_file.stat().st_mtime
            if age < self.STALE_AFTER_S:
                print(f"[resume] {self.root.name} has a LIVE writer, job {prior} "
                      f"(heartbeat {age:.0f}s old); this is {me} -- discarding its chunks rather than "
                      f"interleaving two writers", flush=True)
                for old in self.root.glob("chunk-*.npz"):
                    old.unlink()
                (self.root / "key.json").unlink(missing_ok=True)
            else:
                print(f"[resume] {self.root.name} was left by job {prior} "
                      f"(heartbeat {age:.0f}s old, stale); inheriting its chunks", flush=True)
        self._owner_file = owner_file
        self._touch_owner(me)

        kf = self.root / "key.json"
        if kf.exists():
            try:
                same = _identity(json.loads(kf.read_text())) == _identity(self.key)
            except json.JSONDecodeError:
                same = False
            if not same:
                print(f"[resume] discarding stale state in {self.root}: key changed", flush=True)
                for old in self.root.glob("chunk-*.npz"):
                    old.unlink()
                kf.unlink()
        if not kf.exists():
            kf.write_text(json.dumps(self.key, sort_keys=True) + "\n")
            return 0

        nxt = 0
        for chunk in sorted(self.root.glob("chunk-*.npz")):
            try:
                z = np.load(chunk, allow_pickle=False)
                start, nx = int(z["start"]), int(z["next_start"])
                sha = str(z["key_sha"]) if "key_sha" in z.files else None
                if sha is not None and sha != self._fingerprint():
                    raise ValueError(f"chunk key {sha} != current {self._fingerprint()}")
                blob = str(z["cells"])
                cells_here = blob.split("\n") if blob else []
                windows_here = [int(w) for w in z["windows"]]
                m = np.asarray(z["matrix"])
            except Exception as exc:
                print(f"[resume] {chunk.name} unreadable ({exc}); replay stops here", flush=True)
                break
            if start != nxt:
                print(f"[resume] gap at {chunk.name}: expected start {nxt}, got {start}; replay stops here",
                      flush=True)
                break
            if len(cells_here) != len(windows_here):
                print(f"[resume] {chunk.name}: {len(cells_here)} cells vs {len(windows_here)} "
                      f"ordinals; replay stops here", flush=True)
                break
            if m.size and m.shape[0] != len(cells_here):
                print(f"[resume] {chunk.name}: matrix has {m.shape[0]} rows for "
                      f"{len(cells_here)} labels; chunks predate a change to the row count. "
                      f"Replay stops here -- delete this directory to re-embed clean.", flush=True)
                break
            out_cells.extend(cells_here)
            out_windows.extend(windows_here)
            if m.size:
                blocks.append(m)
            nxt = nx
        self._cells, self._blocks, self._next = len(out_cells), len(blocks), nxt
        if nxt:
            print(f"[resume] {nxt} cells already embedded; continuing from there "
                  f"({len(out_cells)} rows replayed)", flush=True)
        return nxt

    def flush(self, next_start: int, out_cells: list, out_windows: list, blocks: list, *,
              force=False, total: int | None = None):
        """Write a durable chunk, then REPORT it.

        An embedding is hours of one opaque call, so silence is indistinguishable from a hang: a
        job at 100% GPU and a wedged one looked identical for 58 minutes, and the only way to tell
        them apart was sstat on the node. The line is emitted after os.replace, so it reports work
        that survived, never work in progress, and its cadence is the flush interval rather than
        per cell.
        """
        if not force and (time.monotonic() - self._t) < self.interval:
            return
        new_blocks = blocks[self._blocks:]
        matrix = np.vstack(new_blocks) if new_blocks else np.zeros((0, 0), dtype=np.float32)
        chunk = self.root / f"chunk-{self._next:09d}.npz"
        tmp = chunk.with_name(chunk.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez(
                fh,
                start=np.int64(self._next),
                next_start=np.int64(next_start),
                key_sha=np.asarray(self._fingerprint()),
                cells=np.asarray("\n".join(out_cells[self._cells:])),
                windows=np.asarray(out_windows[self._cells:], dtype=np.uint16),
                matrix=matrix,
            )
        os.replace(tmp, chunk)
        self._touch_owner()
        self._cells, self._blocks, self._next = len(out_cells), len(blocks), next_start
        self._t = time.monotonic()
        self._flushes += 1
        el = self._t - self._t0
        rate = next_start / el if el > 0 else 0.0
        where = f"{next_start:,}" if total is None else f"{next_start:,}/{total:,} ({100 * next_start / max(total, 1):4.1f}%)"
        eta = ""
        if total and rate > 0 and next_start < total:
            eta = f"  eta {(total - next_start) / rate / 60:5.1f}m"
        print(f"[embed] cells {where}  rows {len(out_cells):,}  "
              f"{rate:6.2f} cell/s  elapsed {el / 60:5.1f}m{eta}  chunk {self._flushes}", flush=True)


class WindowEncoder(Protocol):
    """What `embed` needs from a model, and nothing more."""

    def embed_windows(
        self,
        tokens: np.ndarray,
        valid: np.ndarray,
        contig_segment_id: np.ndarray,
        umi_segment_id: np.ndarray,
        patch_start_ids: np.ndarray,
    ) -> np.ndarray:
        ...


def arch_digest(arch: dict) -> str:
    """sha256 over the architecture config, canonically serialised."""
    return hashlib.sha256(json.dumps(arch, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Checkpoint:
    """The parts of a checkpoint `embed` is allowed to see."""

    arch: dict
    readout: dict
    arch_sha: str
    checkpoint_sha256: str
    model: WindowEncoder
    torch_version: str | None = None
    corpus_compat_key: dict | None = None

    @classmethod
    def load(cls, path, *, model_factory) -> Checkpoint:
        """Read a checkpoint and verify it describes itself consistently."""
        import torch

        path = Path(path)
        raw = torch.load(path, map_location="cpu", weights_only=False)
        for field in ("arch", "readout", "arch_sha", "state_dict"):
            if field not in raw:
                raise EmbedError(
                    f"{path} has no {field!r}. A checkpoint that does not declare its architecture and "
                    "readout cannot be embedded from"
                )
        if not raw["readout"]:
            raise EmbedError(
                f"{path} declares an empty readout spec. A model embedding that does not name the readout "
                "that produced it cannot be compared with anything, so no aggregation choice made "
                "over it is falsifiable"
            )
        recomputed = arch_digest(raw["arch"])
        if recomputed != raw["arch_sha"]:
            raise EmbedError(
                f"{path}: arch_sha does not match its own arch config.\n"
                f"  recorded   {raw['arch_sha']}\n  recomputed {recomputed}\n"
                "The architecture was changed after training, so this checkpoint's weights and its "
                "declared shape disagree."
            )
        return cls(
            arch=raw["arch"],
            readout=raw["readout"],
            arch_sha=raw["arch_sha"],
            checkpoint_sha256=controls.digest_file(path),
            model=_build_model(model_factory, raw["arch"], raw["state_dict"], raw["readout"]),
            torch_version=raw.get("torch_version") or torch.__version__,
            corpus_compat_key=raw.get("corpus_compat_key"),
        )


def _build_model(model_factory, arch, state_dict, readout):
    """Call `model_factory` with the readout when it accepts one, without it otherwise."""
    import inspect

    try:
        sig = inspect.signature(model_factory)
        if len(sig.parameters) >= 3:
            return model_factory(arch, state_dict, readout)
    except (TypeError, ValueError):
        pass
    return model_factory(arch, state_dict)


def embed(
    checkpoint: Checkpoint,
    corpus,
    *,
    dataset: str,
    w_eval: int,
    eval_seed: int,
    seq_len: int = 16384,
    window_batch: int = 512,
    batch_cells: int | None = None,
    resume_dir=None,
    flush_interval_s: float = 120.0,
    cell_range: tuple[int, int] | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, artifact.Stamp]:
    """Run the model over a corpus and return `(cell_ids, windows, matrix, stamp)`."""
    if w_eval < 1:
        raise EmbedError(f"w_eval must be at least 1, got {w_eval}")
    _assert_trained_and_embedded_corpora_agree(checkpoint, corpus)
    pair_mode = hasattr(checkpoint.model, "pool_pairs")
    pair_v = int(getattr(checkpoint.model, "views", 2)) if pair_mode else 1
    if pair_mode and w_eval % pair_v:
        raise EmbedError(
            f"w_eval must be a multiple of V={pair_v} under the group readout (design 13.10), "
            f"got {w_eval}"
        )

    n_cells = corpus.n_cells()
    cell_names = corpus.cell_names()
    out_cells: list[str] = []
    out_windows: list[int] = []
    blocks: list[np.ndarray] = []

    cell_lo, cell_hi = (0, n_cells) if cell_range is None else cell_range
    if not (0 <= cell_lo < cell_hi <= n_cells):
        raise EmbedError(f"cell_range {(cell_lo, cell_hi)} out of bounds for {n_cells} cells")

    resume = None
    start_at = cell_lo
    if resume_dir is not None:
        _corpus_def = corpus.compat_key() if hasattr(corpus, "compat_key") else None
        resume = _Resume(
            resume_dir,
            {
                "dataset": dataset, "w_eval": w_eval, "eval_seed": eval_seed, "seq_len": seq_len,
                "corpus_def": _corpus_def,
                "n_cells": int(n_cells),
                "pair_mode": bool(pair_mode),
                "pair_v": int(pair_v),
                "checkpoint_sha256": checkpoint.checkpoint_sha256, "arch_sha": checkpoint.arch_sha,
                "forward_precision": getattr(checkpoint.model, "forward_precision", None),
                "cell_id_digest": hashlib.sha256("\x00".join(cell_names).encode()).hexdigest(),
                "cell_range": [int(cell_lo), int(cell_hi)],
            },
            flush_interval_s,
        )
        replayed = resume.load(out_cells, out_windows, blocks)
        start_at = max(cell_lo, replayed)

    staged: list[tuple[int, int]] = []
    stage_rows: list[np.ndarray] = []
    stage_val: list[np.ndarray] = []
    stage_cseg: list[np.ndarray] = []
    stage_useg: list[np.ndarray] = []
    stage_psi: list[np.ndarray] = []
    staged_n = 0

    def _forward(rows_t, rows_v, rows_c, rows_u, rows_p, *, allow_ragged: bool) -> np.ndarray:
        """Forward a window stack in EXACTLY `window_batch`-row slices, returning one vector per row."""
        n = rows_t.shape[0]
        end = n if allow_ragged else (n // window_batch) * window_batch
        out = []
        for lo in range(0, end, window_batch):
            hi = min(lo + window_batch, end)
            out.append(np.asarray(
                checkpoint.model.embed_windows(rows_t[lo:hi], rows_v[lo:hi], rows_c[lo:hi],
                                               rows_u[lo:hi], rows_p[lo:hi]), dtype=np.float32))
        return (np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)), end

    def _drain(upto_cell: int, *, final: bool = False) -> None:
        """Forward whole `window_batch` slices, emit every cell they complete, carry the rest."""
        nonlocal staged, stage_rows, stage_val, stage_cseg, stage_useg, stage_psi, staged_n
        if not staged:
            return
        tok = np.concatenate(stage_rows); val = np.concatenate(stage_val)
        cseg = np.concatenate(stage_cseg); useg = np.concatenate(stage_useg)
        psi = np.concatenate(stage_psi)
        vecs, done = _forward(tok, val, cseg, useg, psi, allow_ragged=final)
        if vecs.shape[0] != done:
            raise EmbedError(
                f"model returned {vecs.shape[0]} vectors for {done} windows -- the artifact is "
                "per window, so these must agree exactly")
        off = 0
        emitted = 0
        for cell, n in staged:
            if off + n > done:
                break
            v = vecs[off:off + n]
            off += n
            emitted += 1
            if pair_mode:
                k = (n // pair_v) * pair_v
                if k:
                    rows = np.asarray(checkpoint.model.pool_pairs(v[:k]), dtype=np.float32)
                    blocks.append(rows)
                    for j in range(rows.shape[0]):
                        out_cells.append(cell_names[cell])
                        out_windows.append(j)
                    assert rows.shape[0] == k // pair_v, (
                        f"pool_pairs at V={pair_v} returned {rows.shape[0]} rows for {k} windows, "
                        f"expected {k // pair_v}"
                    )
            else:
                blocks.append(v)
                for j in range(n):
                    out_cells.append(cell_names[cell])
                    out_windows.append(j)
        staged = staged[emitted:]
        if off < tok.shape[0]:
            stage_rows = [tok[off:]]; stage_val = [val[off:]]
            stage_cseg = [cseg[off:]]; stage_useg = [useg[off:]]; stage_psi = [psi[off:]]
        else:
            stage_rows, stage_val, stage_cseg, stage_useg, stage_psi = [], [], [], [], []
        staged_n = int(tok.shape[0] - off)
        if resume is not None:
            resume.flush(staged[0][0] if staged else upto_cell, out_cells, out_windows,
                         blocks, total=cell_hi)

    for cell in range(start_at, cell_hi):
        batch, real, n_real = cell_windows(corpus, cell, w_cap=w_eval, eval_seed=eval_seed)
        if n_real:
            sel = np.flatnonzero(real)
            stage_rows.append(np.asarray(batch.tokens)[sel])
            stage_val.append(np.asarray(batch.valid)[sel])
            stage_cseg.append(np.asarray(batch.contig_segment_id)[sel])
            stage_useg.append(np.asarray(batch.umi_segment_id)[sel])
            stage_psi.append(np.asarray(batch.patch_start_ids)[sel])
            staged.append((cell, n_real))
            staged_n += n_real
        if staged_n >= window_batch:
            _drain(cell + 1)
    _drain(cell_hi, final=True)

    matrix = np.vstack(blocks) if blocks else np.zeros((0, 0), dtype=np.float32)
    compat, digests, _ = controls.corpus_facts(dataset, corpus=corpus)
    stamp = artifact.Stamp(
        producer="scrm",
        dim=int(matrix.shape[1]),
        n_cells=len(dict.fromkeys(out_cells)),
        n_windows=int(matrix.shape[0]),
        cell_order_sha256=artifact.cell_order_digest(dict.fromkeys(out_cells)),
        corpus_compat_key=dict(compat) if isinstance(compat, dict) else {"compat_key": list(compat)},
        corpus_digests=digests,
        code_rev=controls.code_rev(),
        w_eval=w_eval,
        eval_seed=eval_seed,
        seq_len=seq_len,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        arch_sha=checkpoint.arch_sha,
        readout=checkpoint.readout,
        torch_version=checkpoint.torch_version,
        forward_precision=getattr(checkpoint.model, "forward_precision", None),
        notes={"dataset": dataset, "rows": "pairs" if pair_mode else "windows",
               **({"pairing": "adjacent_v1"} if pair_mode else {}),
               **({} if cell_range is None else {"cell_range": f"{cell_lo}:{cell_hi}", "shard": "partial"})},
    )
    return out_cells, np.asarray(out_windows, dtype=np.uint16), matrix, stamp


def merge_shards(shard_paths, out_path, *, dataset: str) -> Path:
    """Concatenate cell-range shards into the one artifact they would have been."""
    parts = []
    for path in shard_paths:
        emb = artifact.read(path)
        rng = emb.stamp.notes.get("cell_range")
        if rng is None:
            raise EmbedError(f"{path} has no cell_range in its stamp; it is not a shard")
        lo, hi = (int(v) for v in str(rng).split(":"))
        parts.append((lo, hi, emb))
    parts.sort(key=lambda t: t[0])

    ref = parts[0][2].stamp
    for lo, hi, emb in parts[1:]:
        # forward_precision is IDENTITY, not metadata: two shards of one dataset embedded under
        # different arithmetic are not two halves of one artifact, so they must not merge.
        for field in ("checkpoint_sha256", "arch_sha", "readout", "w_eval", "eval_seed", "seq_len",
                      "corpus_digests", "corpus_compat_key", "dim", "forward_precision"):
            if getattr(emb.stamp, field) != getattr(ref, field):
                raise EmbedError(f"shard {lo}:{hi} disagrees with the first on {field}")
        if emb.stamp.notes.get("rows") != ref.notes.get("rows"):
            raise EmbedError(f"shard {lo}:{hi} has row meaning {emb.stamp.notes.get('rows')!r}")

    _, _, all_cells = controls.corpus_facts(dataset)
    expect = 0
    for lo, hi, _ in parts:
        if lo != expect:
            raise EmbedError(f"shards are not contiguous: expected next range to start at {expect}, got {lo}")
        expect = hi
    if expect != len(all_cells):
        raise EmbedError(f"shards cover {expect} of {len(all_cells)} cells; refusing a partial artifact")

    cells: list[str] = []
    windows: list[int] = []
    blocks = []
    for _, _, emb in parts:
        cells.extend(str(c) for c in emb.cell_ids)
        windows.extend(int(w) for w in emb.windows)
        blocks.append(emb.matrix)
    matrix = np.vstack(blocks)

    compat, digests = ref.corpus_compat_key, ref.corpus_digests
    stamp = artifact.Stamp(
        producer=ref.producer, dim=int(matrix.shape[1]), n_cells=len(dict.fromkeys(cells)),
        n_windows=int(matrix.shape[0]),
        cell_order_sha256=artifact.cell_order_digest(dict.fromkeys(cells)),
        corpus_compat_key=dict(compat) if isinstance(compat, dict) else {"compat_key": list(compat)},
        corpus_digests=digests, code_rev=controls.code_rev(), w_eval=ref.w_eval,
        eval_seed=ref.eval_seed, seq_len=ref.seq_len, checkpoint_sha256=ref.checkpoint_sha256,
        arch_sha=ref.arch_sha, readout=ref.readout, torch_version=ref.torch_version,
        # Carry the arithmetic through the merge. Omitted, the merged stamp reads
        # forward_precision=None, which `scoring.refuse_mixed_precision` treats as a pre-field
        # artifact (fp32+eager), and a sharded dataset then cannot be scored beside an unsharded
        # one.
        forward_precision=ref.forward_precision,
        notes={k: v for k, v in ref.notes.items() if k not in ("cell_range", "shard")}
        | {"merged_from": str(len(parts))},
    )
    artifact.write(out_path, cells, np.asarray(windows, dtype=np.uint16), matrix, stamp)
    print(f"[merge] {len(parts)} shards -> {out_path} ({stamp.n_cells:,} cells, {stamp.n_windows:,} rows)",
          flush=True)
    return Path(out_path)


def _assert_trained_and_embedded_corpora_agree(checkpoint, corpus) -> None:
    """REFUSE to embed a checkpoint with a corpus whose molecule definition it never trained on."""
    trained = getattr(checkpoint, "corpus_compat_key", None)
    if not trained:
        print("[embed] checkpoint records no corpus_compat_key, so it predates the field; the "
              "trained-vs-embedded molecule definition CANNOT be checked for this artifact",
              flush=True)
        return
    getter = getattr(corpus, "compat_key", None)
    if getter is None:
        print("[embed] the checkpoint states its molecule definition but this corpus object cannot "
              "state its own (no compat_key); the coupling CANNOT be checked for this artifact",
              flush=True)
        return
    embed_compat = getter()
    got = dict(embed_compat) if isinstance(embed_compat, dict) else {"compat_key": list(embed_compat)}
    if dict(trained) != got:
        differs = sorted(set(dict(trained)) | set(got))
        detail = "\n".join(
            f"    {k}: trained={dict(trained).get(k)!r}  embedding={got.get(k)!r}"
            for k in differs if dict(trained).get(k) != got.get(k)
        )
        raise EmbedError(
            "REFUSING: this checkpoint trained on a different molecule definition than the corpus "
            "it is being embedded from. The vectors would describe one definition while every "
            "downstream comparison treated them as another, which makes every such artifact "
            f"unattributable.\n{detail}\n"
            "  Embed from the corpus root the run trained on (SCRM_CORPUS_ROOT), or retrain."
        )


def cell_windows(corpus, cell: int, w_cap: int, eval_seed: int, v_start: int = 128):
    """Every window of ONE cell, with no padding left in the result."""
    v = min(int(v_start), int(w_cap))
    while True:
        b = corpus.build_batch([int(cell)], epoch_seed=int(eval_seed), v=v, allow_short=True)
        n_patches = np.asarray(b.n_patches)
        real = n_patches > 0
        n_real = int(real.sum())
        if n_real < v or v >= w_cap:
            return b, real, n_real
        v = min(v * 2, int(w_cap))


def embed_to_file(checkpoint: Checkpoint, corpus, out_path, **kw) -> Path:
    """`embed` plus the write, so the stamp and the data cannot be separated by a caller."""
    out_path = Path(out_path)
    kw.setdefault("resume_dir", out_path.with_name(out_path.name + ".resume"))
    cells, windows, matrix, stamp = embed(checkpoint, corpus, **kw)
    artifact.write(out_path, cells, windows, matrix, stamp)
    if kw.get("resume_dir") is not None:
        shutil.rmtree(kw["resume_dir"], ignore_errors=True)
    return out_path
