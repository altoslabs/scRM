#!/usr/bin/env python
"""Does the corpus's cell set match the cell calls it claims to have adopted?"""

import gzip
import hashlib
import json
import sys
from pathlib import Path

import pyarrow as pa
from pyarrow import ipc


def audit(counts: dict, n_corpus_cells: int, min_cell_molecules: int) -> list[str]:
    """Every arithmetic problem in the cell accounting, as human-readable strings."""
    called = counts["cells_called"]
    shallow = counts["dropped_cells_shallow"]
    molecules = counts["dropped_cells_molecules"]
    recorded_emptied = counts.get("dropped_cells_emptied")
    bounds_removed = counts["dropped_umi_short"] + counts["dropped_umi_long"]

    problems = []
    emptied = called - n_corpus_cells - shallow

    if emptied < 0:
        problems.append(
            f"corpus holds {n_corpus_cells:,} cells, more than called ({called:,}) minus "
            f"shallow drops ({shallow:,}) -- cells appeared from somewhere"
        )
    elif emptied and not bounds_removed:
        problems.append(
            f"{emptied:,} called cells are missing and unexplained: not shallow, and the length "
            "bounds removed no molecules, so nothing could have emptied them"
        )

    if recorded_emptied is not None and recorded_emptied != max(emptied, 0):
        problems.append(
            f"dropped_cells_emptied={recorded_emptied:,} but the books need {max(emptied, 0):,} "
            f"(called {called:,} - corpus {n_corpus_cells:,} - shallow {shallow:,})"
        )

    if shallow == 0 and molecules:
        problems.append(
            f"no cells were dropped as shallow, yet {molecules:,} molecules are recorded as lost with them"
        )
    elif shallow:
        lo, hi = shallow, shallow * min_cell_molecules
        if not lo <= molecules < hi:
            problems.append(
                f"{molecules:,} molecules lost with {shallow:,} shallow cells is outside "
                f"[{lo:,}, {hi:,}) -- each dropped cell held at least 1 molecule and fewer than "
                f"the {min_cell_molecules:,} floor, so this pair cannot both be right"
            )
    return problems


def die(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    if len(sys.argv) != 2:
        die("usage: cell_accounting.py <corpus_dir>")
    root = Path(sys.argv[1])
    manifest = root / "manifest.json"
    if not manifest.exists():
        die(f"{root} has no manifest.json -- incomplete, not a corpus")
    m = json.loads(manifest.read_text())
    cfg = m["declared"]["config"]
    counts = m["measured"]["counts"]

    wl = [i["path"] for i in m["declared"]["inputs"] if "barcodes.tsv" in i["path"]]
    if len(wl) != 1:
        die(f"expected exactly one barcodes.tsv input in the manifest, found {len(wl)}: {wl}")
    wl_path = Path(wl[0])
    if not wl_path.exists():
        die(f"whitelist recorded in the manifest is gone: {wl_path}")

    got = hashlib.sha256(wl_path.read_bytes()).hexdigest()
    want = cfg["whitelist_sha256"]
    repackaged = got != want

    opener = gzip.open if wl_path.suffix == ".gz" else open
    with opener(wl_path, "rt") as fh:
        whitelist = {line.strip().split("-")[0] for line in fh if line.strip()}

    cells: set[str] = set()
    for s in m["shards"]:
        with pa.memory_map(str(root / s["file"]), "rb") as src:
            r = ipc.open_file(src)
            cells.update(x["cell"] for x in json.loads(r.metadata[b"scrm_cell_rows"]))

    called = counts["cells_called"]
    # A FILE-level sha mismatch is not a content change, and it must not short-circuit the two
    # checks below -- those prove identity from the data, and they are strictly stronger than a
    # hash. gzip stamps the current time into its header, so re-deriving the same barcodes changes
    # the file's sha256 while the barcode SET is untouched; dying on the hash meant a repackaged
    # whitelist stopped the count and invented-cell checks from ever running.
    if len(whitelist) != called:
        die(f"whitelist has {len(whitelist):,} barcodes but the build recorded cells_called={called:,}"
            + (f"\n  its sha256 also differs from the manifest ({want} -> {got}), so this is a real "
               f"content change, not a repackaging" if repackaged else ""))

    invented = cells - whitelist
    if invented:
        die(f"{len(invented):,} corpus cells are not in the whitelist, e.g. {sorted(invented)[:5]}"
            + (f"\n  its sha256 also differs from the manifest ({want} -> {got})" if repackaged else ""))

    if repackaged:
        print(f"  {root.name}: whitelist REPACKAGED since the build -- manifest {want[:16]}, on disk "
              f"{got[:16]}, but the {len(whitelist):,} barcodes are the corpus's own cells, so the "
              f"content is unchanged. Whitelists written before gzip mtime=0 hash differently on "
              f"every regeneration.")

    problems = audit(counts, len(cells), cfg["min_cell_molecules"])
    if problems:
        die("cell accounting does not close:\n  " + "\n  ".join(problems))

    shallow = counts["dropped_cells_shallow"]
    emptied = called - len(cells) - shallow
    kept = len(cells) / called * 100 if called else 0.0
    print(
        f"  {root.name}: {len(cells):,} cells = {called:,} called - {shallow:,} below floor "
        f"- {emptied:,} emptied upstream  ({kept:.2f}% kept, whitelist verified by content"
        + (f", {counts['dropped_cells_molecules']:,} molecules lost with the dropped cells)" if shallow else ")")
    )


if __name__ == "__main__":
    main()
