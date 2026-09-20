"""The embeddings artifact: one row per window, and a stamp that cannot be omitted."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import MISSING, asdict, dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
from pyarrow import ipc

STAMP_KEY = b"scrm_eval_stamp"

MODEL_PRODUCERS = frozenset({"scrm"})

_PRECISION = re.compile(r"(fp32|bf16|fp16)\+(eager|compiled)")


class StampError(Exception):
    """A missing, incomplete or contradictory stamp."""


@dataclass(frozen=True)
class Stamp:
    """Everything needed to say what an embedding is, and to reproduce it."""

    producer: str
    dim: int
    n_cells: int
    n_windows: int
    cell_order_sha256: str
    corpus_compat_key: dict
    corpus_digests: dict
    code_rev: str
    w_eval: int
    eval_seed: int
    seq_len: int
    checkpoint_sha256: str | None = None
    arch_sha: str | None = None
    readout: dict | None = None
    #: RESERVED AND NOT POPULATED. Nothing writes this and nothing compares it: every artifact
    #: on disk carries None. It is declared because invariant 3's determinism check belongs here,
    #: but that check is NOT implemented, and a reader must not mistake the field's presence for
    #: the check's existence.
    #:
    #: There is no RNG state at embed time that is not already covered: `losses.py` seeds a
    #: private generator from `global_step`, so `W` is a pure function of the step number and
    #: touches no global stream; the training stream, the batch size and `global_step` are
    #: byte-pinned inside `checkpoint_sha256`; the eval forward draws nothing; and the only
    #: eval-side randomness is `eval_seed`, which is its own stamp field and is recomputable
    #: rather than a hash of opaque state.
    #:
    #: It is not simply deleted because `read()` below REFUSES a stamp carrying a field this
    #: reader does not know, so dropping the name would make every artifact already written
    #: unreadable. Populating it is the fix; removing it is a migration.
    rng_state_sha256: str | None = None
    torch_version: str | None = None
    forward_precision: str | None = None
    notes: dict = field(default_factory=dict)

    def validate(self) -> None:
        if not self.producer:
            raise StampError("stamp has no producer")
        for name in ("dim", "n_cells", "n_windows", "seq_len", "w_eval"):
            if getattr(self, name) <= 0:
                raise StampError(f"stamp field {name} must be positive, got {getattr(self, name)}")
        if len(self.cell_order_sha256) != 64:
            raise StampError(f"cell_order_sha256 is not a sha256: {self.cell_order_sha256!r}")
        if not self.corpus_digests:
            raise StampError("stamp records no corpus digests -- the embedding names no source data")
        if not self.corpus_compat_key:
            raise StampError("stamp records no corpus compat_key")
        if self.forward_precision is not None and not _PRECISION.fullmatch(self.forward_precision):
            raise StampError(
                f"forward_precision {self.forward_precision!r} is not <dtype>+<mode> with dtype in "
                "fp32/bf16/fp16 and mode in eager/compiled -- it is identity, and identity compared "
                "by string equality cannot be free-form"
            )
        if self.producer in MODEL_PRODUCERS:
            for name in ("checkpoint_sha256", "arch_sha", "readout"):
                if not getattr(self, name):
                    raise StampError(
                        f"producer {self.producer!r} must record {name}: a model embedding without "
                        "its readout spec cannot be compared to anything (design 12.3)"
                    )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Stamp:
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as e:
            raise StampError(f"stamp is not valid JSON: {e}") from e
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise StampError(f"stamp has fields this reader does not know: {unknown}")
        missing = sorted(
            f
            for f, spec in cls.__dataclass_fields__.items()
            if f not in raw and spec.default is MISSING and spec.default_factory is MISSING
        )
        if missing:
            raise StampError(f"stamp is missing required fields: {missing}")
        try:
            stamp = cls(**raw)
        except TypeError as e:
            raise StampError(f"stamp does not match this reader's schema: {e}") from e
        stamp.validate()
        return stamp


def cell_order_digest(cell_ids) -> str:
    """sha256 over the cell ids in order, newline-joined."""
    h = hashlib.sha256()
    for cid in cell_ids:
        h.update(str(cid).encode())
        h.update(b"\n")
    return h.hexdigest()


def _staging_path(path: Path) -> Path:
    """A staging name for `path` that no other writer can be inside."""
    return path.with_name(f".{path.name}.{os.getpid()}.{os.urandom(6).hex()}.tmp")


def write(path, cell_ids, windows, embeddings, stamp: Stamp) -> None:
    """Write a per-window artifact, checking that the stamp describes the data it is attached to."""
    cell_ids = [str(c) for c in cell_ids]
    windows = np.asarray(windows, dtype=np.uint16)
    emb = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
    if emb.ndim != 2:
        raise ValueError(f"embeddings must be 2-D (rows, dim), got {emb.shape}")
    if not (len(cell_ids) == windows.size == emb.shape[0]):
        raise ValueError(
            f"ragged inputs: {len(cell_ids)} cell_ids, {windows.size} windows, {emb.shape[0]} rows"
        )
    if not np.isfinite(emb).all():
        raise ValueError("embeddings contain NaN or inf")

    stamp.validate()
    if emb.shape[1] != stamp.dim:
        raise StampError(f"stamp says dim={stamp.dim} but the data is {emb.shape[1]}-dimensional")
    if emb.shape[0] != stamp.n_windows:
        raise StampError(f"stamp says n_windows={stamp.n_windows} but the data has {emb.shape[0]} rows")
    unique_cells = len(dict.fromkeys(cell_ids))
    if unique_cells != stamp.n_cells:
        raise StampError(f"stamp says n_cells={stamp.n_cells} but the data holds {unique_cells}")
    want = cell_order_digest(dict.fromkeys(cell_ids))
    if want != stamp.cell_order_sha256:
        raise StampError("cell_order_sha256 does not match the cell ids being written")

    table = pa.table(
        {
            "cell_id": pa.array(cell_ids, type=pa.string()),
            "window": pa.array(windows, type=pa.uint16()),
            "embedding": pa.FixedSizeListArray.from_arrays(
                pa.array(emb.reshape(-1), type=pa.float32()), emb.shape[1]
            ),
        },
        metadata={STAMP_KEY: stamp.to_json().encode()},
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _staging_path(path)
    try:
        with ipc.new_file(pa.OSFile(str(tmp), "wb"), table.schema) as w:
            w.write_table(table)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class Embeddings:
    """A loaded artifact: the per-window rows, plus the stamp they came with."""

    cell_ids: np.ndarray
    windows: np.ndarray
    matrix: np.ndarray
    stamp: Stamp

    def __len__(self) -> int:
        return self.matrix.shape[0]


def read_stamp(path) -> Stamp:
    """The stamp alone, without loading the embeddings."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with pa.memory_map(str(path), "rb") as src:
        meta = ipc.open_file(src).schema.metadata or {}
        if STAMP_KEY not in meta:
            raise StampError(
                f"{path} carries no {STAMP_KEY.decode()}. An unstamped embedding is unattributable: "
                "it cannot say which readout, corpus or seed produced it (design 12.2)"
            )
        return Stamp.from_json(meta[STAMP_KEY].decode())


def read(path) -> Embeddings:
    """Load an artifact, refusing anything whose stamp is absent, malformed or contradicted."""
    path = Path(path)
    stamp = read_stamp(path)
    with pa.memory_map(str(path), "rb") as src:
        reader = ipc.open_file(src)
        table = reader.read_all()

    cell_ids = np.asarray(table.column("cell_id").to_pylist())
    windows = np.asarray(table.column("window").to_numpy(zero_copy_only=False), dtype=np.uint16)
    flat = table.column("embedding").combine_chunks()
    matrix = np.asarray(flat.values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(
        len(table), -1
    )

    if matrix.shape[1] != stamp.dim:
        raise StampError(f"stamp says dim={stamp.dim}, file holds {matrix.shape[1]}")
    if matrix.shape[0] != stamp.n_windows:
        raise StampError(f"stamp says n_windows={stamp.n_windows}, file holds {matrix.shape[0]}")
    got = cell_order_digest(dict.fromkeys(cell_ids.tolist()))
    if got != stamp.cell_order_sha256:
        raise StampError(
            "cell order does not match the stamp -- the file's rows are not the cells it claims"
        )
    return Embeddings(cell_ids=cell_ids, windows=windows, matrix=matrix, stamp=stamp)


def same_population(a: Stamp, b: Stamp) -> None:
    """Raise unless two stamps describe the same cells of the same corpus content."""
    if a.cell_order_sha256 != b.cell_order_sha256:
        raise StampError(
            f"{a.producer} and {b.producer} cover different cells or a different order "
            f"({a.n_cells} vs {b.n_cells} cells)"
        )
    same_corpus(a, b)


def same_corpus(a: Stamp, b: Stamp) -> None:
    """Raise unless two stamps describe the same corpus CONTENT, saying nothing about which cells."""
    if a.corpus_digests != b.corpus_digests:
        only_a = sorted(set(a.corpus_digests) - set(b.corpus_digests))
        only_b = sorted(set(b.corpus_digests) - set(a.corpus_digests))
        changed = sorted(
            d for d in set(a.corpus_digests) & set(b.corpus_digests)
            if a.corpus_digests[d] != b.corpus_digests[d]
        )
        raise StampError(
            f"{a.producer} and {b.producer} were computed on different corpus content: "
            f"only in {a.producer}={only_a}, only in {b.producer}={only_b}, differing={changed}"
        )
    if a.corpus_compat_key != b.corpus_compat_key:
        raise StampError(
            f"{a.producer} and {b.producer} declare different compat_keys: "
            f"{a.corpus_compat_key} vs {b.corpus_compat_key}"
        )
