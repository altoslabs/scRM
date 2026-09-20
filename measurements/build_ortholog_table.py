#!/usr/bin/env python3
"""Build a strict-1:1 ortholog bridge `<species> -> human` for the count-FM baselines."""
from __future__ import annotations

import argparse
import collections
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
from scrm_eval import registry  # noqa: E402

WEIGHTS = registry.PROJ / "countfm" / "scfm_weights"
MART = "https://may2025.archive.ensembl.org/biomart/martservice"

SPECIES = {
    "rat": ("rnorvegicus_gene_ensembl", "ensrnog", "2500_wistar_rat_pbmcs_singleplex_3p_gem_x"),
    "mouse": ("mmusculus_gene_ensembl", "ensmusg", "mouse_brain_5k"),
    "fly": ("dmelanogaster_gene_ensembl", "fbgn", "fly_antenna"),
    "macaque": (None, None, "2500_rhesus_monkey_pbmcs_singleplex_3p_gem_x"),
}

QUERY = """<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" datasets="{ds}">
<Dataset name="{ds}" interface="default">
<Attribute name="ensembl_gene_id"/><Attribute name="external_gene_name"/>
<Attribute name="hsapiens_homolog_ensembl_gene"/>
<Attribute name="hsapiens_homolog_associated_gene_name"/>
<Attribute name="hsapiens_homolog_orthology_type"/>
</Dataset></Query>"""


def fetch(mart_dataset: str, timeout: int = 600) -> list[list[str]]:
    body = urllib.parse.urlencode({"query": QUERY.format(ds=mart_dataset)}).encode()
    with urllib.request.urlopen(MART, data=body, timeout=timeout) as fh:  # noqa: S310 (pinned host)
        text = fh.read().decode()
    return [ln.split("\t") for ln in text.rstrip("\n").split("\n")][1:]


def strict_one2one(rows: list[list[str]]) -> list[tuple[str, str, str, str]]:
    """`ortholog_one2one` rows that are also unique in both directions."""
    o2o = [r for r in rows if len(r) >= 5 and r[2] and r[4].strip() == "ortholog_one2one"]
    src = collections.Counter(r[0] for r in o2o)
    dst = collections.Counter(r[2] for r in o2o)
    keep = [(r[0], r[1], r[2], r[3]) for r in o2o if src[r[0]] == 1 and dst[r[2]] == 1]
    print(f"[mart] {len(rows)} rows -> {len(o2o)} one2one -> {len(keep)} strict-unique pairs")
    return sorted(keep)


def coverage(dataset: str, ids: set[str] | None) -> None:
    """Report how much of the REGISTERED matrix the bridge reaches. The number that bounds an FM."""
    from scrm_eval import targets

    var = targets.load_counts(dataset).var
    gene_ids = [str(g) for g in var["gene_ids"]]
    if ids is None:
        n_ensg = sum(1 for g in gene_ids if g.startswith("ENSG"))
        print(f"[coverage] {dataset}: {n_ensg}/{len(gene_ids)} genes are already human ENSG ids "
              f"({n_ensg / len(gene_ids):.1%}) -- no bridge needed"
              if n_ensg else f"[coverage] {dataset}: no human ENSG ids; a bridge IS needed")
        return
    hit = sum(1 for g in gene_ids if g in ids)
    print(f"[coverage] {dataset}: {hit}/{len(gene_ids)} matrix genes map 1:1 to human "
          f"({hit / len(gene_ids):.1%})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--species", required=True, choices=sorted(SPECIES))
    ap.add_argument("--check", action="store_true", help="report coverage only; write nothing")
    ap.add_argument("--out-dir", type=Path, default=WEIGHTS)
    a = ap.parse_args(argv)

    mart_ds, id_col, registered = SPECIES[a.species]
    if mart_ds is None:
        coverage(registered, None)
        if not a.check:
            print(f"refusing to write a bridge for {a.species}: its matrix is already human-annotated, "
                  "and passing a table would disable species auto-detection (see module docstring)")
            return 1
        return 0

    out = a.out_dir / f"ortholog_{a.species}_human.tsv"
    if a.check and out.exists():
        ids = {ln.split("\t")[0] for ln in out.read_text().splitlines()[1:]}
        print(f"[existing] {out} holds {len(ids)} pairs")
        coverage(registered, ids)
        return 0

    pairs = strict_one2one(fetch(mart_ds))
    if len(pairs) < 1000:
        print(f"only {len(pairs)} pairs -- refusing to write a bridge this thin")
        return 1
    coverage(registered, {p[0] for p in pairs})
    if a.check:
        return 0
    tmp = out.with_suffix(".tsv.tmp")
    with tmp.open("w") as fh:
        fh.write(f"{id_col}\t{a.species}_symbol\tensg\thuman_symbol\n")
        for p in pairs:
            fh.write("\t".join(p) + "\n")
    tmp.replace(out)
    print(f"wrote {out} ({len(pairs)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
