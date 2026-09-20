"""Add a species to UCE at inference time, transcribing `data_proc/Create New Species Files.ipynb`."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from scrm_eval import registry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--species", required=True, help="short name used in --species at inference")
    ap.add_argument("--taxonomy-id", required=True, type=int, help="NCBI taxid; seeds the chrom tokens")
    ap.add_argument("--fasta", required=True, type=Path, help="Ensembl pep.all.fa for the SAME assembly")
    ap.add_argument("--embeddings", required=True, type=Path, help="gene_symbol_to_embedding_ESM2.pt")
    ap.add_argument("--assembly", required=True, help="primary_assembly name as it appears in the FASTA")
    ap.add_argument("--all-tokens", type=Path,
                    default=registry.PROJ / "fm" / "uce" / "model_files" / "all_tokens.torch")
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args()

    import pandas as pd
    import torch

    a.out_dir.mkdir(parents=True, exist_ok=True)

    gene_symbol_to_location: dict[str, str] = {}
    gene_symbol_to_chrom: dict[str, str] = {}
    missing = 0
    with open(a.fasta) as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            parts = line.split()
            sym = [t for t in parts if t.startswith("gene_symbol")]
            if not sym:
                missing += 1
                continue
            bits_sym = sym[0].split(":", 1)
            if len(bits_sym) != 2 or not bits_sym[1]:
                missing += 1
                continue
            symbol = bits_sym[1]
            loc = [t for t in parts if t.startswith("primary_assembly") or t.startswith("chromosome")]
            if not loc:
                missing += 1
                continue
            bits = loc[0].split(":")
            if len(bits) < 5 or bits[1] != a.assembly:
                missing += 1
                continue
            gene_symbol_to_chrom[symbol] = bits[2]
            gene_symbol_to_location[symbol] = bits[3]

    df = pd.DataFrame({
        "gene_symbol": list(gene_symbol_to_chrom),
        "chromosome": [gene_symbol_to_chrom[g] for g in gene_symbol_to_chrom],
        "start": [gene_symbol_to_location[g] for g in gene_symbol_to_chrom],
    })
    df["species"] = a.species
    df["gene_symbol"] = df["gene_symbol"].str.upper()
    before = len(df)
    df = df.drop_duplicates(subset="gene_symbol", keep="first")
    if len(df) != before:
        print(f"[species] {before - len(df)} symbols collided when uppercased; kept the first of each",
              flush=True)
    unreadable = {"NA", "NAN", "NULL", "NONE", "N/A", "NAT", "-NAN", "INF", "-INF", ""}
    hit = df["gene_symbol"].isin(unreadable)
    if hit.any():
        print(f"[species] dropping {int(hit.sum())} gene(s) whose uppercased symbol pandas reads as "
              f"a null: {sorted(df.loc[hit, 'gene_symbol'])} -- their loader cannot index these",
              flush=True)
        df = df.loc[~hit]
    chrom_csv = a.out_dir / f"{a.species}_to_chrom_pos.csv"
    df.to_csv(chrom_csv, index=False)
    n_chrom = df["chromosome"].nunique()
    print(f"[species] {len(df):,} genes mapped, {n_chrom} unique chromosomes "
          f"({missing:,} FASTA headers without a usable symbol/assembly) -> {chrom_csv}", flush=True)

    all_pe = torch.load(a.all_tokens, weights_only=False)[0:4]
    pe = torch.load(a.embeddings, weights_only=False)
    offset = len(all_pe)
    pe_stacked = torch.stack(list(pe.values()))
    all_pe = torch.vstack((all_pe, pe_stacked))
    chrom_token_offset = all_pe.shape[0]
    torch.manual_seed(a.taxonomy_id)
    all_pe = torch.vstack((all_pe, torch.normal(mean=0, std=1, size=(n_chrom, all_pe.shape[1]))))
    all_pe.requires_grad = False
    tokens_path = a.out_dir / f"{a.species}_pe_tokens.torch"
    torch.save(all_pe, tokens_path)
    with open(a.out_dir / f"{a.species}_offsets.pkl", "wb") as fh:
        pickle.dump({a.species: offset}, fh)

    meta = {"species": a.species, "taxonomy_id": a.taxonomy_id, "assembly": a.assembly,
            "genes_in_embeddings": len(pe), "genes_mapped_from_fasta": len(df),
            "n_uniq_chrom": int(n_chrom), "CHROM_TOKEN_OFFSET": int(chrom_token_offset),
            "token_rows": int(all_pe.shape[0]), "token_dim": int(all_pe.shape[1]),
            "protein_embeddings_file": a.embeddings.name, "fasta": a.fasta.name,
            "source": "data_proc/Create New Species Files.ipynb, cells 6-17"}
    (a.out_dir / f"{a.species}_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    print(f"[species] CHROM_TOKEN_OFFSET={chrom_token_offset} -- the caller MUST pass this; the "
          "default 143574 belongs to the training species", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
