"""Derive an evaluation dataset's ground-truth inputs from its published files.

A derivation returns a bundle mapping output filename to content. `-o` writes the bundle into a
directory; `--check` asserts every member reproduces the file already there. A `.json` member holds
JSON, a `.tsv.gz` member one item per gzipped line.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from pathlib import Path

#: A target counts as knocked down at this effect size and significance. Targets below it carry no
#: transcriptional signal, so scoring them measures the experiment and not the embedding.
MIN_LOG2FC = -1.0
MAX_P = 0.05


def mixology_5cl(raw: Path) -> dict[str, object]:
    """The called-cell whitelist, and barcode -> cell line for the demuxlet singlets.

    Joins GSE126906_barcodes.csv.gz (cell_name -> barcode_sequence) with
    sc_10x_5cl.metadata.csv.gz (cell_name -> cell_line_demuxlet, demuxlet_cls).

    The whitelist is every cell the metadata carries; the labels are the SINGLETS only. A doublet is
    a called cell with an ambiguous cell line, so it belongs in the corpus and out of the scored
    population -- the same treatment a QC mask gets, reported rather than filtered.
    """
    bc: dict[str, str] = {}
    with gzip.open(raw / "GSE126906_barcodes.csv.gz", "rt") as fh:
        for row in csv.DictReader(fh):
            bc[row["cell_name"]] = row["barcode_sequence"]

    out: dict[str, str] = {}
    called: list[str] = []
    with gzip.open(raw / "sc_10x_5cl.metadata.csv.gz", "rt") as fh:
        rdr = csv.reader(io.TextIOWrapper(io.BytesIO(fh.read().encode()), newline=""))
        header = next(rdr)
        # R's write.csv leaves the rowname column unnamed, so the data columns are offset by one.
        i_line = header.index("cell_line_demuxlet") + 1
        i_cls = header.index("demuxlet_cls") + 1
        for row in rdr:
            if not row:
                continue
            name = row[0].strip('"')
            b = bc.get(name)
            if b is None:
                continue
            called.append(b)
            if row[i_cls].strip('"') == "SNG":
                out[b] = row[i_line].strip('"')
    return {"labels.json": out,
            "barcodes.tsv.gz": sorted(f"{b}-1" for b in set(called))}


def tm_10x_p7_2(raw: Path) -> dict[str, object]:
    """barcode -> cell ontology class for channel 10X_P7_2.

    Reads Metadata/tabula-muris-senis-full-metadata.csv from the czb-tabula-muris-senis bucket.
    Not annotations_droplet.csv from the original Tabula Muris repository: it covers the same
    channel but agrees on only 734 of these 1,645 barcodes, with 14 ontology classes against 16.
    """
    want_channel = "10X_P7_2"
    out: dict[str, str] = {}
    with (raw / "tabula-muris-senis-full-metadata.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            cell = r.get("cell") or ""
            if want_channel not in cell and want_channel not in (r.get("batch") or ""):
                continue
            bc = cell.split("_")[-1].split("-")[0]
            cls = (r.get("cell_ontology_class") or "").strip()
            if bc and cls:
                out[bc] = cls
    # The called cells ARE the annotated cells: the channel's whitelist is the label key set, with
    # the Cell Ranger suffix the corpus was built against.
    return {"labels.json": out,
            "barcodes.tsv.gz": sorted(f"{bc}-1" for bc in out)}


def crispr_a549_5k(raw: Path) -> dict[str, object]:
    """The guide-called whitelist, the single-guide labels, and the targets that were knocked down.

    Reads crispr_analysis/protospacer_calls_per_cell.csv, crispr_analysis/feature_reference.csv and
    crispr_analysis/perturbation_efficiencies_by_target.csv, all members of the published
    count_crispr_analysis tarball.

    The whitelist is every barcode carrying a guide call. A cell with more than one guide stays in
    the corpus but is absent from the labels: its label is ambiguous, so it belongs in a sensitivity
    axis rather than the scored population.
    """
    with (raw / "crispr_analysis" / "protospacer_calls_per_cell.csv").open(newline="") as fh:
        calls = list(csv.DictReader(fh))
    with (raw / "crispr_analysis" / "feature_reference.csv").open(newline="") as fh:
        gene = {r["id"]: r["target_gene_name"] for r in csv.DictReader(fh)}
    with (raw / "crispr_analysis" / "perturbation_efficiencies_by_target.csv").open(newline="") as fh:
        eff = list(csv.DictReader(fh))

    bcs = [r["cell_barcode"] for r in calls]
    widths = {len(b.split("-")[0]) for b in bcs}
    if widths != {16}:
        raise SystemExit(f"guide-called barcodes are not 16 bp: {sorted(widths)}")

    single = [r for r in calls if r["num_features"] == "1"]
    unknown = sorted({r["feature_call"] for r in single} - set(gene))
    if unknown:
        raise SystemExit(f"{len(unknown)} guide(s) have no target_gene_name: {unknown[:4]}")

    strong = sorted({r["Target Gene"] for r in eff
                     if float(r["Log2 Fold Change"]) <= MIN_LOG2FC and float(r["p Value"]) < MAX_P})
    return {
        "barcodes.tsv.gz": bcs,
        "labels_guide.json": {r["cell_barcode"]: r["feature_call"] for r in single},
        "labels_target.json": {r["cell_barcode"]: gene[r["feature_call"]] for r in single},
        "effective_targets.json": strong,
    }


DERIVATIONS = {"sc_mixology_5cl": mixology_5cl, "tm_10X_P7_2": tm_10x_p7_2,
               "crispr_a549_5k": crispr_a549_5k}


def _load(path: Path) -> object:
    if path.name.endswith(".tsv.gz"):
        with gzip.open(path, "rt") as fh:
            return [ln for ln in fh.read().splitlines() if ln]
    return json.loads(path.read_text())


def _dump(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.name.endswith(".tsv.gz"):
        # mtime=0 so the same barcodes always hash the same: the whitelist's sha256 is recorded in
        # the corpus manifest as its identity, and gzip otherwise stamps the current time.
        with gzip.GzipFile(tmp, "wb", mtime=0) as fh:
            fh.write("".join(f"{x}\n" for x in content).encode())
    else:
        tmp.write_text(json.dumps(content, sort_keys=True))
    tmp.replace(path)


def _describe(name: str, content: object) -> str:
    if isinstance(content, dict):
        return f"{len(content)} entries, {len(set(content.values()))} distinct values"
    return f"{len(content)} items"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=sorted(DERIVATIONS))
    ap.add_argument("--raw", type=Path, required=True, help="directory holding the published inputs")
    ap.add_argument("-o", "--out", type=Path,
                    help="write the bundle into this directory (default: --raw)")
    ap.add_argument("--check", action="store_true",
                    help="assert the derivation reproduces every member already in --out")
    a = ap.parse_args()

    bundle = DERIVATIONS[a.dataset](a.raw)
    out = a.out or a.raw
    for name, content in bundle.items():
        print(f"{a.dataset}: {name}: {_describe(name, content)}")

    if a.check:
        for name, content in bundle.items():
            f = out / name
            if not f.is_file():
                raise SystemExit(f"  cannot check {f}: it is not on disk")
            have = _load(f)
            if have == content:
                print(f"  reproduces {f} exactly")
                continue
            if isinstance(content, dict) and isinstance(have, dict):
                detail = (f"only-on-disk {sorted(set(have) - set(content))[:4]}, "
                          f"only-derived {sorted(set(content) - set(have))[:4]}, disagree "
                          f"{[k for k in set(have) & set(content) if have[k] != content[k]][:4]}")
            else:
                detail = (f"only-on-disk {sorted(set(have) - set(content))[:4]}, "
                          f"only-derived {sorted(set(content) - set(have))[:4]}")
            raise SystemExit(
                f"  DOES NOT reproduce {f}: {len(have)} on disk vs {len(content)} derived; {detail}. "
                f"These are ground truth, so a mismatch changes what every number over "
                f"{a.dataset} means.")
        return 0

    for name, content in bundle.items():
        _dump(out / name, content)
        print(f"  wrote {out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
