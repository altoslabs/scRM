"""Recover a dataset's called-cell whitelist from its own pinned corpus."""

import glob
import gzip
import hashlib
import json
import pathlib
import sys

import pyarrow as pa
import pyarrow.ipc as ipc

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "eval"))
from scrm_eval import registry  # noqa: E402

CORPUS = registry.corpus_root()
RAW = registry.PROJ / "raw"
CALIBRATION = ("pbmc_1k_v2", RAW / "pbmc_1k_v2/matrix/barcodes.tsv.gz")


def corpus_cells(ds: str) -> set[str]:
    """Every distinct `cell_id` in a corpus, read from the dictionary rather than the indices."""
    cells: set[str] = set()
    shards = sorted(glob.glob(str(CORPUS / ds / "*.arrow")))
    if not shards:
        raise SystemExit(f"no shards under {CORPUS / ds}")
    for f in shards:
        with pa.memory_map(f, "rb") as src:
            reader = ipc.open_file(src)
            for i in range(reader.num_record_batches):
                cells.update(reader.get_batch(i).column(0).dictionary.to_pylist())
    return cells


def calibrate() -> str:
    """Prove the suffix rule against a dataset whose original whitelist is on disk."""
    ds, path = CALIBRATION
    if not path.exists():
        raise SystemExit(f"calibration whitelist missing: {path}")
    lines = [l for l in gzip.decompress(path.read_bytes()).decode().split("\n") if l]
    suffixes = sorted({b.split("-", 1)[1] for b in lines if "-" in b})
    if len(suffixes) != 1:
        raise SystemExit(f"{ds} whitelist mixes suffixes {suffixes}; the rule is not one suffix")
    stripped = {b.rsplit("-", 1)[0] for b in lines}
    got = corpus_cells(ds)
    if got != stripped:
        raise SystemExit(
            f"CALIBRATION FAILED on {ds}: corpus has {len(got)} cells, whitelist-stripped has "
            f"{len(stripped)}, and they are not the same set. The suffix rule does not hold, so no "
            f"derivation should be trusted."
        )
    print(f"calibration OK on {ds}: {len(got)} cells, corpus == whitelist minus '-{suffixes[0]}'")
    return suffixes[0]


def derive(ds: str, suffix: str) -> None:
    manifest = json.loads((CORPUS / ds / "manifest.json").read_text())
    called = manifest["measured"]["counts"]["cells_called"]
    cells = corpus_cells(ds)
    if len(cells) != called:
        raise SystemExit(
            f"REFUSING to derive {ds}: corpus holds {len(cells)} cells but {called} were called, so "
            f"{called - len(cells)} were dropped by min_cell_molecules. The corpus is a SUBSET of the "
            f"whitelist, not the whitelist, and writing it would build against a smaller population."
        )
    out = RAW / ds / "barcodes.tsv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{b}-{suffix}\n" for b in sorted(cells))
    with gzip.open(out, "wt") as fh:
        fh.write(body)
    recorded = manifest["declared"]["config"].get("whitelist_sha256", "")
    print(f"wrote {out}: {len(cells)} barcodes")
    print(f"  content sha256 {hashlib.sha256(body.encode()).hexdigest()[:32]}")
    print(f"  recorded whitelist_sha256 {recorded[:32]} -- hashes the GZIP BYTES, so not comparable")


if __name__ == "__main__":
    suffix = calibrate()
    for ds in sys.argv[1:]:
        derive(ds, suffix)
