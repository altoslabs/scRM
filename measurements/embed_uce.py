"""Embed one dataset's filtered count matrix with UCE (Rosen et al., 33-layer), emitting the panel npz."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from scrm_eval import registry  # noqa: E402

PROJ = registry.PROJ
UCE_ROOT = PROJ / "vendor" / "UCE"
ASSETS = PROJ / "fm" / "uce"


def _species_pe_path(species: str) -> Path | None:
    """The protein-embedding .pt this species will be matched against, from the same CSV UCE's own
    `gene_embeddings.py` reads for species outside its training set. None for a training species,
    whose embeddings UCE resolves internally.
    """
    csv = ASSETS / "model_files" / "new_species_protein_embeddings.csv"
    if not csv.exists():
        return None
    for line in csv.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        name, _, path = line.partition(",")
        if name.strip() == species and Path(path.strip()).exists():
            return Path(path.strip())
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--matrix-dir", type=Path)
    ap.add_argument("--adata", type=Path, help="an h5ad with raw counts in .X (UCE requires raw)")
    ap.add_argument("--species", required=True, choices=("human", "mouse", "fly", "rat"))
    ap.add_argument("--out-npz", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=25)
    a = ap.parse_args()

    import scanpy as sc

    work = ASSETS / "work" / a.dataset
    work.mkdir(parents=True, exist_ok=True)
    h5ad = work / f"{a.dataset}.h5ad"
    if not (a.matrix_dir or a.adata):
        raise SystemExit("pass --matrix-dir (our filtered 10x matrix) or --adata (an h5ad)")
    def _stale_work() -> str | None:
        """Why the cached work directory does not describe the input we were handed, or None."""
        if not h5ad.exists():
            return None
        want = None
        if a.matrix_dir:
            bc = Path(a.matrix_dir) / "barcodes.tsv.gz"
            if bc.exists():
                import gzip
                with gzip.open(bc, "rt") as fh:
                    want = [l.strip() for l in fh if l.strip()]
        elif a.adata:
            want = [str(o) for o in sc.read(a.adata).obs_names]
        if want is None:
            return None
        cached = [str(o) for o in sc.read_h5ad(h5ad, backed="r").obs_names]
        if cached == want:
            return None
        return (f"cached input has {len(cached):,} cells, the matrix now has {len(want):,}"
                if len(cached) != len(want) else
                f"cached input has the same {len(cached):,} cells in a different order")

    why = _stale_work()
    if why is not None:
        print(f"[uce-embed] DISCARDING {work}: {why}", flush=True)
        for f in sorted(work.iterdir()):
            if f.is_file():
                f.rename(f.with_name(f.name + ".stale"))
    if not h5ad.exists():
        if a.adata:
            ad = sc.read(a.adata)
            for k in list(ad.obsm):
                del ad.obsm[k]
        else:
            ad = sc.read_10x_mtx(a.matrix_dir)
        ad.var_names = [str(v) for v in ad.var_names]
        ad.obs_names = [str(o) for o in ad.obs_names]
        ad.var_names_make_unique()
        chrom_csv = ASSETS / "species_build" / f"{a.species}_to_chrom_pos.csv"
        if chrom_csv.exists():
            import pandas as pd

            known = set(pd.read_csv(chrom_csv)["gene_symbol"].dropna().astype(str))
            keep = [str(v).upper() in known for v in ad.var_names]
            if not all(keep):
                print(f"[uce-embed] {sum(keep):,} of {ad.n_vars:,} genes are in the {a.species} "
                      f"chromosome table; dropping {ad.n_vars - sum(keep):,} UCE cannot tokenize "
                      f"(e.g. {[str(v) for v, k in zip(ad.var_names, keep) if not k][:6]})", flush=True)
                ad = ad[:, keep].copy()
        ad.write(h5ad)
        print(f"[uce-embed] wrote {h5ad} ({ad.n_obs} cells x {ad.n_vars} genes)", flush=True)

    out_adata = work / f"{a.dataset}_uce_adata.h5ad"
    if not out_adata.exists():
        cmd = [sys.executable, str(UCE_ROOT / "eval_single_anndata.py"),
               "--adata_path", str(h5ad),
               "--dir", str(work) + "/",
               "--species", a.species,
               "--model_loc", str(ASSETS / "model_files" / "33l_8ep_1024t_1280.torch"),
               "--nlayers", "33",
               "--batch_size", str(a.batch_size),
               "--protein_embeddings_dir", str(ASSETS / "model_files" / "protein_embeddings") + "/"]
        build = ASSETS / "species_build"
        meta = build / f"{a.species}_meta.json"
        if meta.exists():
            off = json.loads(meta.read_text())["CHROM_TOKEN_OFFSET"]
            cmd += ["--spec_chrom_csv_path", str(build / f"{a.species}_to_chrom_pos.csv"),
                    "--token_file", str(build / f"{a.species}_pe_tokens.torch"),
                    "--offset_pkl_path", str(build / f"{a.species}_offsets.pkl"),
                    "--CHROM_TOKEN_OFFSET", str(off)]
        else:
            cmd += ["--spec_chrom_csv_path", str(ASSETS / "model_files" / "species_chrom.csv"),
                    "--token_file", str(ASSETS / "model_files" / "all_tokens.torch"),
                    "--offset_pkl_path", str(ASSETS / "model_files" / "species_offsets.pkl")]
        env = dict(__import__("os").environ)
        env["PYTHONPATH"] = f"{UCE_ROOT}:{env.get('PYTHONPATH', '')}"
        print("[uce-embed] running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=str(ASSETS), env=env)

    ad = sc.read(out_adata)
    emb = np.asarray(ad.obsm["X_uce"], dtype=np.float32)
    a.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out_npz, emb=emb, barcode=np.asarray(ad.obs_names, dtype=object))
    print(f"[uce-embed] {a.dataset}: {emb.shape} -> {a.out_npz}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
