"""Zero-shot cell embeddings from count-based deep-learning foundation models.

Produces `{emb:[N,D], barcode:[N]}` npz files, which `eval/scrm_eval/controls.from_npz`
wraps as stamped artifacts, from the pretrained count-FM baselines `eval/scrm_eval/panel.py`
declares EMBEDDABLE, with ZERO training on our data:

  * ``--model scgpt``            scGPT whole-human checkpoint (wanglab/scGPT-human on HF,
                                 pretrained on 33M normal human cells from CELLxGENE census).
                                 Embedding = the official ``scgpt.tasks.embed_data`` zero-shot
                                 path (CLS cell embedding, L2-normalized), run WITHOUT
                                 flash-attn (``use_fast_transformer=False``).
  * ``--model geneformer``       Geneformer V2 (ctheodoris/Geneformer, Geneformer-V2-104M,
                                 2024). Rank-value tokenization + CLS embedding from the
                                 second-to-last layer, replicating the official
                                 TranscriptomeTokenizer/EmbExtractor defaults (see
                                 GENEFORMER PROTOCOL below).
  * ``--model genejepa``         GeneJEPA (Litman et al. 2025, bioRxiv 10.1101/2025.10.14.682378;
                                 Biostate AI). Count-space JEPA trained on Tahoe-100M: Perceiver
                                 encoder (d=768, 512 latents, 24 blocks) over unordered expressed-
                                 gene tokens (gene-identity embedding + Fourier expression features).
                                 Weights = huggingface ``elonlit/GeneJEPA`` Lightning ckpt (epoch 49);
                                 we load the EMA-TEACHER encoder (the README-recommended inference
                                 path) through ``eval/countfm/genejepa_loader.py``, which imports
                                 THEIR code from the clone ``eval/countfm/fetch.py`` makes at a
                                 pinned commit -- nothing is vendored into this repository.
                                 Preprocessing is THEIRS exactly: their 62,710-gene
                                 vocab (``gene_metadata.parquet``, matrix genes mapped by Ensembl ID,
                                 unmapped DROPPED -- coverage printed) and their GLOBAL log1p stats
                                 (``global_stats.json``: values = (log1p(count) - mean)/(std + 1e-6),
                                 NO per-cell depth normalization). Cell embedding = mean over the 512
                                 processed latents (their ``get_embedding``), dim 768. A built-in
                                 bit-identity double-forward on the first ~100 cells runs every time.
  * ``--model scvi_pretrained``  Pretrained scVI reference model from the scvi-tools hub
                                 (default: scvi-tools/tabula-sapiens-lung-scvi, scvi 1.4.2,
                                 n_latent=20). STRICT no-training inference: weights are
                                 loaded frozen via ``SCVI.load_query_data`` (scArches-style
                                 query init: the only new parameters are the zero-initialized
                                 one-hot inputs for the query batch category, which contribute
                                 nothing) and latents come from ``get_latent_representation``
                                 (posterior mean -> deterministic). ``train()`` is NEVER called;
                                 determinism is asserted at runtime.

Accepted count-matrix inputs (``--matrix``):
  * 10x HDF5 (``*.h5``) or AnnData (``*.h5ad``)
  * 10x mtx directory (``matrix.mtx[.gz]`` + ``features.tsv[.gz]``/``genes.tsv``)
  * alevin-fry quant dir (``.../af_quant`` or its parent; ``alevin/quants_mat.mtx``)

GENE-ID HANDLING (documented choice):
  alevin-fry columns are versioned Ensembl IDs; scGPT's vocab is gene symbols. We bridge
  ENSG <-> symbol with the static GENCODE-derived mapping that ships with the Geneformer
  gc104M release (``gene_name_id_dict_gc104M.pkl``, symbol -> ENSG; inverted for ENSG ->
  symbol). One canonical offline mapping for every model -- no GTF parse, no pybiomart,
  no network at runtime.

MOUSE INPUT (documented asymmetry): our byte-level model is mouse-trained and embeds
human data zero-shot natively; the human-trained count FMs cannot -- mouse matrices are
first converted to the human gene space via a STATIC strict one-to-one ortholog table
(Ensembl BioMart may2025 archive export, ``ortholog_one2one`` rows only, many-to-many
dropped; ``scfm_weights/ortholog_mouse_human.tsv``). Species is auto-detected (ENSMUSG
IDs, or title-case symbols when IDs are absent) and mapping coverage is printed so eval
tables can footnote it.

OTHER SPECIES (``--ortholog-table``): the same strict-1:1-ortholog bridge generalizes to
any species with a 4-column TSV shaped like the mouse table (``<src_id> <src_symbol>
<ensg> <human_symbol>``, header row, tab-separated -- id/symbol columns matched against
whichever of gene_ids/gene_symbols the input matrix has). Pass ``--ortholog-table PATH``
to use it explicitly; this OVERRIDES species auto-detection (which only distinguishes
mouse/human and would misclassify e.g. fly gene symbols). Mouse auto-detect (no flag
needed) is unchanged. Example: ``scfm_weights/ortholog_fly_human.tsv`` (Ensembl BioMart
may2025 archive, ``dmelanogaster_gene_ensembl`` -> ``hsapiens_homolog_*``, one2one only).

CELL SUBSETTING (``--cell-filter-suffix``): keep only obs (barcodes) whose string ends
with the given suffix, applied right after loading the matrix -- e.g. to pull one 10x
library's cells out of a multi-library h5ad by its ``__<library>`` suffix.

GENEFORMER PROTOCOL (replicated from the official repo @ HF main, files
``geneformer/tokenizer.py`` + ``geneformer/emb_extractor.py``):
  1. strip Ensembl versions; collapse IDs via ``ensembl_mapping_dict_gc104M.pkl``; sum
     duplicate columns; keep genes present in the gc104M token dictionary.
  2. per cell: ``x_norm = counts / n_counts_total * 10_000 / gene_median`` (n_counts =
     total counts over ALL matrix genes, medians from
     ``gene_median_dictionary_gc104M.pkl``); rank nonzero genes by descending x_norm.
  3. truncate to 4094 tokens, wrap as ``[<cls>] + ranks + [<eos>]`` (V2 special tokens).
  4. forward through Geneformer-V2-104M (BertForMaskedLM, eval mode) and take the
     ``<cls>`` position of hidden_states[num_layers - 1] -- the second-to-last layer,
     matching EmbExtractor defaults ``emb_mode="cls", emb_layer=-1``.

Validated end-to-end on the sc_mixology_5cl alevin-fry quant directory (3,918 cells, 5 lung
cell lines) -- see ``--sanity-labels``.

Reproduce (substitute your own paths; nothing below is a path this repository can assume):
    <WEIGHTS_ROOT>  the directory `eval/countfm/fetch.py --weights-root` was given
    <MATRIX>        a count matrix in one of the forms above, e.g. an `af_quant` directory
    <OUT_NPZ>       where to write the embedding
    <LABELS_JSON>   optional; bare barcode -> class, for the sanity probe only

    micromamba run -n scfm python eval/countfm/embed_count_fm.py \
        --model scgpt \
        --weights-root <WEIGHTS_ROOT> \
        --matrix <MATRIX> \
        --out-npz <OUT_NPZ> \
        --sanity-labels <LABELS_JSON>

`model/jobs/baselines_matrix.sbatch` is the same call with the paths resolved from
`$SCRM_PROJ` -- read it rather than retyping one.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# THIS SCRIPT RUNS UNDER THE MODELS' OWN INTERPRETER (the `scfm` env). It puts its own directory
# and `eval/` on the path itself and imports only modules with no third-party dependencies:
# `provenance` (numpy) and `scrm_eval.panel` (nothing at all).
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provenance import savez_stamped, stamp  # noqa: E402
# The panel is declared in one place. `panel.py` has no imports: this script runs under the
# models' own interpreter, which does not have `controls`'.
from scrm_eval.panel import EMBEDDABLE  # noqa: E402

#: No default. The weights are not in this repository: `eval/countfm/fetch.py` pulls them from
#: their publishers into a directory you name, and `--weights-root` is that directory.
WEIGHTS_ROOT: Path | None = None
SCGPT_DIR = "scgpt_human"  # wanglab/scGPT-human (whole-human checkpoint)
GENEFORMER_MODEL_DIR = "geneformer/Geneformer-V2-104M"  # ctheodoris/Geneformer
GENEFORMER_DICT_DIR = "geneformer/dictionaries"  # gc104M dictionaries from the same repo
SCVI_DEFAULT_DIR = "scvi_ts_lung"  # scvi-tools/tabula-sapiens-lung-scvi (scvi-hub reference)
GENEJEPA_DIR = "genejepa"  # elonlit/GeneJEPA (ckpt + gene_metadata.parquet + global_stats.json)
GENEJEPA_CKPT = "genejepa-epoch=49.ckpt"
ORTHOLOG_TSV = "ortholog_mouse_human.tsv"  # Ensembl BioMart may2025, strict 1:1

# The FILES each protocol opens, named once and opened through these names below. `fetch.py
# --check` asks this module for them, so what the fetch verifies is what the embedder reads.
SCGPT_VOCAB = "vocab.json"
SCGPT_ARGS = "args.json"  # scgpt/tasks/cell_emb.py:201-203 opens these three from model_dir
SCGPT_CKPT = "best_model.pt"
GENEFORMER_CONFIG = "config.json"  # transformers' from_pretrained(GENEFORMER_MODEL_DIR)
GENEFORMER_WEIGHTS = "model.safetensors"
GENEFORMER_TOKEN_DICT = "token_dictionary_gc104M.pkl"
GENEFORMER_MEDIAN_DICT = "gene_median_dictionary_gc104M.pkl"
GENEFORMER_ENSEMBL_DICT = "ensembl_mapping_dict_gc104M.pkl"
GENEFORMER_NAME_ID_DICT = "gene_name_id_dict_gc104M.pkl"  # also the ENSG<->symbol bridge
GENEJEPA_META = "gene_metadata.parquet"
GENEJEPA_STATS = "global_stats.json"
SCVI_CKPT = "model.pt"

#: Per model, the paths under `--weights-root` without which its embedding cannot run. Keyed by
#: the same model names `--model` takes; the values are built from the constants above rather
#: than retyped.
REQUIRED_FILES: dict[str, tuple[str, ...]] = {
    "scgpt": tuple(f"{SCGPT_DIR}/{f}" for f in (SCGPT_ARGS, SCGPT_CKPT, SCGPT_VOCAB)),
    "geneformer": (
        f"{GENEFORMER_MODEL_DIR}/{GENEFORMER_CONFIG}",
        f"{GENEFORMER_MODEL_DIR}/{GENEFORMER_WEIGHTS}",
        *(f"{GENEFORMER_DICT_DIR}/{f}" for f in (GENEFORMER_ENSEMBL_DICT, GENEFORMER_MEDIAN_DICT,
                                                 GENEFORMER_NAME_ID_DICT, GENEFORMER_TOKEN_DICT)),
    ),
    "genejepa": tuple(f"{GENEJEPA_DIR}/{f}" for f in (GENEJEPA_CKPT, GENEJEPA_META, GENEJEPA_STATS)),
    "scvi_pretrained": (f"{SCVI_DEFAULT_DIR}/{SCVI_CKPT}",),
}

# ASSERTED, NOT DOCUMENTED. A model `--model` accepts but this table does not name would be
# fetched unverified.
assert set(REQUIRED_FILES) == set(EMBEDDABLE), (
    f"REQUIRED_FILES and panel.EMBEDDABLE disagree: {sorted(set(EMBEDDABLE) ^ set(REQUIRED_FILES))}"
)

GENEFORMER_INPUT_SIZE = 4096  # V2 model input size (with <cls>/<eos>)
GENEFORMER_TARGET_SUM = 10_000


# --------------------------------------------------------------------------------------
# count-matrix loading
# --------------------------------------------------------------------------------------
def _strip_version(gene_id: str) -> str:
    return gene_id.split(".", 1)[0]


def load_matrix(path: Path) -> tuple[sp.csr_matrix, list[str], list[str] | None, list[str] | None]:
    """Load a count matrix -> (X cells x genes CSR, barcodes, gene_ids | None, gene_symbols | None).

    gene_ids are UNVERSIONED Ensembl IDs when available.
    """
    import anndata as ad
    import scanpy as sc

    if path.is_file() and path.suffix == ".h5":
        adata = sc.read_10x_h5(path)
        adata.var_names_make_unique()
        ids = [_strip_version(g) for g in adata.var["gene_ids"]] if "gene_ids" in adata.var else None
        return sp.csr_matrix(adata.X), list(adata.obs_names), ids, list(adata.var_names)

    if path.is_file() and path.suffix == ".h5ad":
        adata = ad.read_h5ad(path)
        ids_raw = adata.var.get("gene_ids", adata.var_names)
        looks_ens = sum(str(g).startswith("ENS") for g in ids_raw[:100]) > 50
        ids = [_strip_version(str(g)) for g in ids_raw] if looks_ens else None
        syms = None if (ids is not None and ids_raw is adata.var_names) else list(adata.var_names)
        return sp.csr_matrix(adata.X), list(adata.obs_names), ids, syms

    if not path.is_dir():
        raise SystemExit(f"--matrix {path}: not a .h5/.h5ad file or a directory")

    # alevin-fry quant dir (af_quant, its parent, or the alevin dir itself)
    for alevin in (path / "af_quant" / "alevin", path / "alevin", path):
        if (alevin / "quants_mat.mtx").exists() or (alevin / "quants_mat.mtx.gz").exists():
            from scipy.io import mmread

            mtx = alevin / "quants_mat.mtx"
            if not mtx.exists():
                mtx = alevin / "quants_mat.mtx.gz"
            m = sp.csr_matrix(mmread(str(mtx)))  # alevin-fry: cells (rows) x genes (cols)
            barcodes = (alevin / "quants_mat_rows.txt").read_text().split()
            genes = (alevin / "quants_mat_cols.txt").read_text().split()
            assert m.shape == (len(barcodes), len(genes)), f"mtx {m.shape} vs {len(barcodes)}x{len(genes)}"
            return m, barcodes, [_strip_version(g) for g in genes], None

    # 10x mtx dir
    for mtx_name in ("matrix.mtx.gz", "matrix.mtx"):
        if (path / mtx_name).exists():
            adata = sc.read_10x_mtx(path)
            adata.var_names_make_unique()
            ids = [_strip_version(g) for g in adata.var["gene_ids"]] if "gene_ids" in adata.var else None
            return sp.csr_matrix(adata.X), list(adata.obs_names), ids, list(adata.var_names)

    raise SystemExit(f"--matrix {path}: no quants_mat.mtx / matrix.mtx found")


# --------------------------------------------------------------------------------------
# species detection + mouse -> human ortholog conversion
# --------------------------------------------------------------------------------------
def detect_species(gene_ids: list[str] | None, gene_symbols: list[str] | None) -> str:
    if gene_ids is not None:
        n_mouse = sum(g.startswith("ENSMUSG") for g in gene_ids)
        n_human = sum(g.startswith("ENSG") for g in gene_ids)
        if n_mouse + n_human > 0:
            return "mouse" if n_mouse > n_human else "human"
    if gene_symbols is not None:
        sample = [s for s in gene_symbols[:2000] if re.fullmatch(r"[A-Za-z][A-Za-z0-9-]+", s or "")]
        if sample:
            title_case = sum(bool(re.fullmatch(r"[A-Z][a-z][a-z0-9-]*", s)) for s in sample)
            return "mouse" if title_case / len(sample) > 0.5 else "human"
    raise SystemExit("could not detect species from gene ids/symbols")


def convert_to_human(
    x: sp.csr_matrix,
    gene_ids: list[str] | None,
    gene_symbols: list[str] | None,
    tsv: Path,
) -> tuple[sp.csr_matrix, list[str], list[str]]:
    """Map a non-human matrix into the human gene space via a static strict-1:1 ortholog table.

    ``tsv`` is a 4-column TSV (header + ``<src_id> <src_symbol> <ensg> <human_symbol>``
    rows) -- e.g. ``ortholog_mouse_human.tsv`` (ENSMUSG) or ``ortholog_fly_human.tsv``
    (FBgn). Returns (X cells x human_genes, human ENSG ids, human symbols). Unmapped
    source genes are DROPPED (their counts do not exist in any human FM vocab). Mapping
    coverage is printed so eval tables can footnote it.
    """
    by_id: dict[str, tuple[str, str]] = {}
    by_sym: dict[str, tuple[str, str]] = {}
    with tsv.open() as fh:
        next(fh)  # header: ensmusg  mouse_symbol  ensg  human_symbol
        for line in fh:
            ensmusg, msym, ensg, hsym = line.rstrip("\n").split("\t")
            by_id[ensmusg] = (ensg, hsym)
            if msym:
                by_sym[msym] = (ensg, hsym)

    n_genes = x.shape[1]
    if gene_ids is not None:
        hits = [(j, *by_id[g]) for j, g in enumerate(gene_ids) if g in by_id]
    else:
        assert gene_symbols is not None
        hits = [(j, *by_sym[s]) for j, s in enumerate(gene_symbols) if s in by_sym]

    cols = [j for j, _, _ in hits]
    h_ids = [e for _, e, _ in hits]
    h_syms = [s for _, _, s in hits]
    print(
        f"[ortholog] input: {len(hits)}/{n_genes} matrix genes map 1:1 to human "
        f"({len(hits) / n_genes:.1%} of matrix; table={tsv.name}, {len(by_id)} pairs)",
        flush=True,
    )
    if len(hits) < 1000:
        raise SystemExit(f"only {len(hits)} genes mapped -- wrong species detection, ortholog table, or gene naming?")
    return sp.csr_matrix(x[:, cols]), h_ids, h_syms


# --------------------------------------------------------------------------------------
# ENSG <-> symbol bridging (static Geneformer gc104M gene_name_id dictionary)
# --------------------------------------------------------------------------------------
def _load_name_id_dict(weights_root: Path) -> dict[str, str]:
    with (weights_root / GENEFORMER_DICT_DIR / GENEFORMER_NAME_ID_DICT).open("rb") as fh:
        return pickle.load(fh)  # symbol -> ENSG


def ensure_symbols(gene_ids: list[str] | None, gene_symbols: list[str] | None, weights_root: Path) -> list[str | None]:
    """Return one symbol (or None) per gene, mapping ENSG -> symbol where symbols are missing."""
    if gene_symbols is not None:
        return list(gene_symbols)
    assert gene_ids is not None
    name_id = _load_name_id_dict(weights_root)
    id_name: dict[str, str] = {}
    for name, gid in name_id.items():
        id_name.setdefault(gid, name)
    syms = [id_name.get(g) for g in gene_ids]
    n = sum(s is not None for s in syms)
    print(f"[genes] ENSG->symbol via gene_name_id_dict_gc104M: {n}/{len(syms)} mapped", flush=True)
    return syms


def ensure_ids(gene_ids: list[str] | None, gene_symbols: list[str] | None, weights_root: Path) -> list[str | None]:
    """Return one unversioned ENSG (or None) per gene, mapping symbol -> ENSG where IDs are missing."""
    if gene_ids is not None:
        return list(gene_ids)
    assert gene_symbols is not None
    name_id = _load_name_id_dict(weights_root)
    ids = [name_id.get(s) for s in gene_symbols]
    n = sum(i is not None for i in ids)
    print(f"[genes] symbol->ENSG via gene_name_id_dict_gc104M: {n}/{len(ids)} mapped", flush=True)
    return ids


# --------------------------------------------------------------------------------------
# embedders
# --------------------------------------------------------------------------------------
def embed_scgpt(
    x: sp.csr_matrix,
    barcodes: list[str],
    symbols: list[str | None],
    weights_root: Path,
    device: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    import anndata as ad
    from scgpt.tasks import embed_data

    keep = [j for j, s in enumerate(symbols) if s]
    adata = ad.AnnData(X=sp.csr_matrix(x[:, keep]).astype(np.float32))
    adata.obs_names = barcodes
    adata.var_names = [f"g{j}" for j in keep]  # unique placeholders; real names in gene_symbol col
    adata.var["gene_symbol"] = [symbols[j] for j in keep]

    model_dir = weights_root / SCGPT_DIR
    vocab = set(json.loads((model_dir / SCGPT_VOCAB).read_text()).keys())
    in_vocab = sum(s in vocab for s in adata.var["gene_symbol"])
    print(
        f"[scgpt] vocab hit: {in_vocab}/{x.shape[1]} matrix genes in scGPT vocab "
        f"({in_vocab / x.shape[1]:.1%}); vocab size {len(vocab)}",
        flush=True,
    )

    # max_length=0 => cover every expressed gene, so the collator's `sampling=True`
    # randperm[:max_length] is a no-op.
    nnz_max = int(np.diff(adata.X.indptr).max()) if adata.X.nnz else 1
    eff_len = int(max_length) if max_length and max_length > 0 else nnz_max + 1
    n_trunc = int((np.diff(adata.X.indptr) > eff_len).sum())
    print(
        f"[scgpt] max_length={eff_len} (nnz/cell: max {nnz_max}, "
        f"median {int(np.median(np.diff(adata.X.indptr)))}); "
        f"cells truncated by random gene subsampling: {n_trunc}/{adata.n_obs}",
        flush=True,
    )
    out = embed_data(
        adata,
        model_dir,
        gene_col="gene_symbol",
        max_length=eff_len,
        batch_size=batch_size,
        device=device,
        use_fast_transformer=False,  # no flash-attn on this stack; falls back to sdpa attention
        return_new_adata=True,
    )
    return np.asarray(out.X, dtype=np.float32)


def embed_geneformer(
    x: sp.csr_matrix,
    barcodes: list[str],
    gene_ids: list[str | None],
    weights_root: Path,
    device: str,
    batch_size: int,
    emb_mode: str = "cls",
) -> np.ndarray:
    import torch
    from transformers import BertModel

    dict_dir = weights_root / GENEFORMER_DICT_DIR
    with (dict_dir / GENEFORMER_TOKEN_DICT).open("rb") as fh:
        token_dict: dict[str, int] = pickle.load(fh)
    with (dict_dir / GENEFORMER_MEDIAN_DICT).open("rb") as fh:
        median_dict: dict[str, float] = pickle.load(fh)
    with (dict_dir / GENEFORMER_ENSEMBL_DICT).open("rb") as fh:
        ens_map: dict[str, str] = pickle.load(fh)

    # collapse IDs (official sum_ensembl_ids), keep tokenizable genes, sum duplicates
    collapsed = [ens_map.get(g, g) if g else None for g in gene_ids]
    col_by_gene: dict[str, list[int]] = {}
    for j, g in enumerate(collapsed):
        if g is not None and g in token_dict and g in median_dict:
            col_by_gene.setdefault(g, []).append(j)
    genes = sorted(col_by_gene)
    print(
        f"[geneformer] vocab hit: {len(genes)}/{x.shape[1]} matrix genes in gc104M token dict "
        f"({len(genes) / x.shape[1]:.1%}); vocab size {len(token_dict)}",
        flush=True,
    )
    if not genes:
        raise SystemExit("no matrix genes found in the Geneformer token dictionary")

    n_counts = np.asarray(x.sum(axis=1), dtype=np.float64).ravel()  # over ALL matrix genes
    sel = sp.lil_matrix((x.shape[1], len(genes)), dtype=np.float64)
    for k, g in enumerate(genes):
        for j in col_by_gene[g]:
            sel[j, k] = 1.0
    xg = sp.csr_matrix(x.astype(np.float64) @ sp.csr_matrix(sel))  # cells x kept genes (dups summed)

    tokens = np.array([token_dict[g] for g in genes], dtype=np.int64)
    medians = np.array([median_dict[g] for g in genes], dtype=np.float64)
    cls_id, eos_id, pad_id = token_dict["<cls>"], token_dict["<eos>"], token_dict["<pad>"]

    input_rows: list[np.ndarray] = []
    for i in range(xg.shape[0]):
        row = xg[i]
        vals = row.data / max(n_counts[i], 1.0) * GENEFORMER_TARGET_SUM / medians[row.indices]
        ranked = tokens[row.indices][np.argsort(-vals)][: GENEFORMER_INPUT_SIZE - 2]
        input_rows.append(np.concatenate(([cls_id], ranked, [eos_id])))

    # BertModel = the checkpoint's bert trunk WITHOUT the MLM head (hidden states identical;
    # the head would materialize a [B, 4096, 20275] logit tensor)
    model = BertModel.from_pretrained(weights_root / GENEFORMER_MODEL_DIR, output_hidden_states=True)
    model.eval().to(device)
    layer = model.config.num_hidden_layers - 1  # second-to-last (EmbExtractor emb_layer=-1)

    embs = np.empty((len(input_rows), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(input_rows), batch_size):
            batch = input_rows[i : i + batch_size]
            maxlen = max(len(r) for r in batch)
            ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
            mask = torch.zeros((len(batch), maxlen), dtype=torch.long)
            for k, r in enumerate(batch):
                ids[k, : len(r)] = torch.from_numpy(r)
                mask[k, : len(r)] = 1
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device))
            hs = out.hidden_states[layer]
            if emb_mode == "cls":
                emb_i = hs[:, 0, :]
            else:  # "cell": mean over GENE positions only (official mean_nonpadding_embs; drop <cls>/<eos>)
                lens = torch.tensor([len(r) for r in batch], device=hs.device)
                gene_mask = (torch.arange(maxlen, device=hs.device)[None, :] >= 1) & (
                    torch.arange(maxlen, device=hs.device)[None, :] < (lens - 1)[:, None]
                )
                emb_i = (hs * gene_mask[:, :, None]).sum(1) / (lens - 2).clamp(min=1)[:, None]
            embs[i : i + len(batch)] = emb_i.float().cpu().numpy()
            if i == 0:
                print(f"[geneformer] layer {layer} (2nd-to-last), emb_mode={emb_mode}, batch len {maxlen}", flush=True)
    return embs


def embed_genejepa(
    x: sp.csr_matrix,
    barcodes: list[str],
    gene_ids: list[str | None],
    gene_symbols: list[str] | None,
    weights_root: Path,
    device: str,
    batch_size: int,
) -> np.ndarray:
    import sys

    import pandas as pd
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from genejepa_loader import load_teacher_encoder

    mdir = weights_root / GENEJEPA_DIR
    # THEIR gene vocab: rows sorted by token_id; embedding index = row rank (upstream gene_map).
    meta = pd.read_parquet(mdir / GENEJEPA_META).sort_values("token_id").reset_index(drop=True)
    stats = json.loads((mdir / GENEJEPA_STATS).read_text())
    mean, std = float(stats["mean"]), float(stats["std"])

    by_ens: dict[str, int] = {}
    by_sym: dict[str, int] = {}
    for i, (sym, ens) in enumerate(zip(meta["gene_symbol"], meta["ensembl_id"], strict=True)):
        if ens:
            by_ens.setdefault(str(ens), i)
        if sym:
            by_sym.setdefault(str(sym).upper(), i)

    n_genes = x.shape[1]
    if gene_ids is not None and sum(g is not None for g in gene_ids) > 0:
        vocab_idx = np.array([by_ens.get(g, -1) if g else -1 for g in gene_ids], dtype=np.int64)
        key = "ensembl_id"
    else:
        assert gene_symbols is not None
        vocab_idx = np.array([by_sym.get(s.upper(), -1) if s else -1 for s in gene_symbols], dtype=np.int64)
        key = "gene_symbol"
    keep = np.where(vocab_idx >= 0)[0]
    xk = sp.csr_matrix(x[:, keep]).astype(np.float32)
    tok = vocab_idx[keep]
    total_counts = float(x.sum())
    print(
        f"[genejepa] vocab hit ({key}): {len(keep)}/{n_genes} matrix genes in GeneJEPA vocab "
        f"({len(keep) / n_genes:.1%} of matrix genes, {float(xk.sum()) / max(total_counts, 1.0):.1%} of counts; "
        f"vocab size {len(meta)}); unmapped genes dropped",
        flush=True,
    )
    if len(keep) < 1000:
        raise SystemExit(f"only {len(keep)} genes mapped into the GeneJEPA vocab -- wrong gene naming?")

    uniq, inv = np.unique(tok, return_inverse=True)
    if len(uniq) != len(tok):  # duplicate matrix columns hitting one vocab gene: sum them
        sel = sp.csr_matrix((np.ones(len(tok), dtype=np.float32), (np.arange(len(tok)), inv)), (len(tok), len(uniq)))
        xk = sp.csr_matrix(xk @ sel)
        tok = uniq
        print(f"[genejepa] summed duplicate columns -> {len(tok)} unique vocab genes", flush=True)

    n_empty = int((np.diff(xk.indptr) == 0).sum())
    if n_empty:
        print(f"[genejepa] WARNING: {n_empty} cells have ZERO in-vocab counts (constant embedding)", flush=True)

    model = load_teacher_encoder(mdir / GENEJEPA_CKPT, weights_root).to(device)
    d = model.config.d
    print(
        f"[genejepa] EMA-teacher encoder loaded: d={d}, latents={model.config.latents_L}, "
        f"blocks={model.config.blocks_D}, gene_vocab={model.config.gene_vocab_size}",
        flush=True,
    )
    tok_t = torch.from_numpy(tok)

    def forward_batch(rows: sp.csr_matrix) -> np.ndarray:
        # THEIR collate: tokens = expressed genes; values = (log1p(count) - mean) / (std + 1e-6)
        lengths = np.diff(rows.indptr)
        indices = tok_t[torch.from_numpy(rows.indices.astype(np.int64))]
        values = (torch.log1p(torch.from_numpy(rows.data).float()) - mean) / (std + 1e-6)
        offsets = torch.from_numpy(np.concatenate(([0], np.cumsum(lengths))).astype(np.int64))
        out = model(indices.to(device), values.to(device), offsets.to(device))
        return out.float().cpu().numpy()

    embs = np.empty((x.shape[0], d), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            embs[i : i + batch_size] = forward_batch(xk[i : i + batch_size])
            if i == 0:
                print(f"[genejepa] first batch ok ({min(batch_size, x.shape[0])} cells)", flush=True)

        # determinism gate: re-run the first ~100 cells with IDENTICAL batching; must be bit-identical
        n_check = min(100, x.shape[0])
        redo = np.concatenate([forward_batch(xk[i : i + batch_size]) for i in range(0, n_check, batch_size)])
        n_redo = len(redo)
        bit_identical = np.array_equal(redo.view(np.uint8), embs[:n_redo].view(np.uint8))
        print(f"[genejepa] determinism check on {n_redo} cells: bit-identical={bit_identical}", flush=True)
        if not bit_identical:
            raise SystemExit("GeneJEPA forward is not deterministic -- protocol violation")
    return embs


def embed_scvi_pretrained(
    x: sp.csr_matrix,
    barcodes: list[str],
    gene_ids: list[str | None],
    model_dir: Path,
    batch_size: int,
) -> np.ndarray:
    import anndata as ad
    import scvi
    import torch

    ckpt = torch.load(model_dir / SCVI_CKPT, map_location="cpu", weights_only=False)
    ref_vars = list(ckpt["var_names"])
    reg = ckpt["attr_dict"]["registry_"]
    setup = reg["setup_args"]
    fields = reg["field_registries"]
    print(f"[scvi] reference: {model_dir.name}, {len(ref_vars)} genes, setup={setup}", flush=True)

    by_id = {g: j for j, g in enumerate(gene_ids) if g}
    hit_cols = {v: by_id[v] for v in ref_vars if v in by_id}
    print(
        f"[scvi] vocab hit: {len(hit_cols)}/{len(ref_vars)} reference genes present in matrix "
        f"({len(hit_cols) / len(ref_vars):.1%} of reference space; "
        f"{len(hit_cols)}/{x.shape[1]} = {len(hit_cols) / x.shape[1]:.1%} of matrix genes); "
        f"missing reference genes are zero-filled",
        flush=True,
    )

    # build query matrix in the exact reference gene order (missing genes = zero columns)
    xq = sp.lil_matrix((x.shape[0], len(ref_vars)), dtype=np.float32)
    xc = sp.csc_matrix(x)
    for k, v in enumerate(ref_vars):
        if v in hit_cols:
            xq[:, k] = xc[:, hit_cols[v]]
    q = ad.AnnData(X=sp.csr_matrix(xq))
    q.obs_names = barcodes
    q.var_names = ref_vars
    if setup.get("layer"):
        q.layers[setup["layer"]] = q.X.copy()
    if setup.get("batch_key"):
        q.obs[setup["batch_key"]] = "scrm_query"  # new category; its one-hot weights init to zero
    if setup.get("labels_key"):
        ref_labels = fields["labels"]["state_registry"]["categorical_mapping"]
        q.obs[setup["labels_key"]] = str(ref_labels[0])  # labels are not used by the SCVI encoder

    model = scvi.model.SCVI.load_query_data(q, str(model_dir))
    model.module.eval()
    model.is_trained = True  # bookkeeping flag ONLY -- no train() call anywhere; weights stay frozen
    z = model.get_latent_representation(q, give_mean=True, batch_size=batch_size).astype(np.float32)

    # strict-zero-shot guarantee: posterior mean must be deterministic
    z2 = model.get_latent_representation(q[: min(64, q.n_obs)], give_mean=True, batch_size=batch_size)
    assert np.allclose(z[: len(z2)], z2, atol=1e-5), "scVI latents are not deterministic -- protocol violation"
    print(f"[scvi] latent {z.shape}, deterministic check passed (no training performed)", flush=True)
    return z


# --------------------------------------------------------------------------------------
# sanity probe
# --------------------------------------------------------------------------------------
def sanity_logreg(emb: np.ndarray, barcodes: list[str], labels_json: Path) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

    labels = json.loads(labels_json.read_text())  # bare-barcode keys
    bare = [re.sub(r"-\d+$", "", b) for b in barcodes]
    keep = [i for i, b in enumerate(bare) if b in labels]
    if not keep:
        print("[sanity] NO barcodes matched the labels json -- check barcode formats", flush=True)
        return
    # Stripping the "-N" suffix maps AAA-1 and AAA-2 (two libraries, two DISTINCT cells)
    # onto one bare barcode, so both get the same label and can land in different CV
    # folds. Flag it rather than silently scoring it.
    n_dup = len(keep) - len({bare[i] for i in keep})
    if n_dup:
        print(
            f"[sanity] WARNING: {n_dup} of {len(keep)} cells share a bare barcode with "
            f"another cell (suffix stripped). Distinct cells collapsed onto one label can "
            f"split across folds -- pass --cell-filter-suffix to scope to one library.",
            flush=True,
        )
    x = emb[keep]
    xl2 = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)  # L2, matches harness raw tier
    y = np.array([labels[bare[i]] for i in keep])
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    # The harness probe: LogisticRegression(C=1.0, class_weight="balanced"), reporting pooled
    # out-of-fold accuracy.
    clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced")
    pred = cross_val_predict(clf, xl2, y, cv=cv)
    acc = float((pred == y).mean())
    accs = cross_val_score(clf, xl2, y, cv=cv, scoring="accuracy")
    print(
        f"[sanity] 5-fold LogReg on {len(keep)} labeled cells, {len(set(y))} classes: "
        f"pooled-OOF acc {acc:.4f}  (harness protocol: C=1.0, class_weight=balanced) | "
        f"mean-of-folds {accs.mean():.4f} +/- {accs.std():.4f} "
        f"(fold spread, NOT a seed spread)",
        flush=True,
    )
    # Second, geometry-robust probe (StandardScaler centers per-dim -- the protocol the task evals
    # use, e.g. eval_cancer_task/eval_fly_fm_probe), for embeddings with a large constant offset.
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    accs2 = cross_val_score(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), x, y, cv=cv, scoring="accuracy"
    )
    print(f"[sanity] scaler+LogReg (centered) variant: acc {accs2.mean():.4f} +/- {accs2.std():.4f}", flush=True)


# --------------------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(EMBEDDABLE))
    p.add_argument("--matrix", type=Path, required=True, help="10x .h5 / .h5ad / mtx dir / af_quant dir")
    p.add_argument("--out-npz", type=Path, required=True)
    p.add_argument("--weights-root", type=Path, required=True,
                   help="where eval/countfm/fetch.py put the publishers' weights")
    p.add_argument(
        "--scvi-model-dir",
        type=Path,
        default=None,
        help=f"scvi-hub model dir ({SCVI_CKPT}). Default: <weights-root>/{SCVI_DEFAULT_DIR}",
    )
    p.add_argument("--device", default="cuda", help="cuda|cpu (scgpt/geneformer)")
    p.add_argument("--batch-size", type=int, default=64)
    # 0 = no truncation (cover every expressed gene). `max_length` reaches scGPT's collator,
    # which is constructed with `sampling=True` (scgpt/tasks/cell_emb.py) and then does
    # `torch.randperm(len(genes))[:max_length]`, so a nonzero value embeds each cell from a
    # random, unseeded subset of its expressed genes. Pass 1200 explicitly to reproduce an old run.
    p.add_argument(
        "--max-length",
        type=int,
        default=0,
        help="scGPT input length. 0 = cover every expressed gene (no random "
        "gene subsampling). scGPT's own training value was 1200, but using "
        "it here truncates ~71%% of each cell at random.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seeds torch/numpy before every baseline embed. scGPT's gene "
        "subsampling was the only unseeded path in the panel (GeneJEPA and "
        "scVI both have explicit determinism gates), so its rows were not "
        "reproducible run-to-run.",
    )
    p.add_argument(
        "--gf-emb-mode",
        choices=["cls", "cell"],
        default="cls",
        help="Geneformer pooling: 'cls' (official V2 EmbExtractor default) or 'cell' "
        "(mean over gene positions, the mode most zero-shot benchmarks use)",
    )
    p.add_argument(
        "--sanity-labels",
        type=Path,
        default=None,
        help="optional labels json (bare barcode -> class) for a 5-fold LogReg accuracy print",
    )
    p.add_argument(
        "--ortholog-table",
        type=Path,
        default=None,
        help="strict-1:1 ortholog TSV (src_id, src_symbol, ensg, human_symbol) to bridge a "
        "non-mouse species into the human gene space; overrides species auto-detection. "
        "Mouse auto-detect (no flag) is unchanged.",
    )
    p.add_argument(
        "--cell-filter-suffix",
        default=None,
        help="keep only barcodes (obs names) ending with this suffix, e.g. '__<library>' to pull "
        "one library's cells out of a multi-library matrix",
    )
    args = p.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Determinism for the whole panel. scGPT's gene subsampling is the only unseeded path here;
    # GeneJEPA and scVI both carry explicit determinism gates.
    import random as _random

    import torch as _torch

    _random.seed(args.seed)
    np.random.seed(args.seed)
    _torch.manual_seed(args.seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(args.seed)
    print(f"[seed] {args.seed} (torch/numpy/random)", flush=True)

    x, barcodes, gene_ids, gene_symbols = load_matrix(args.matrix)
    print(f"[matrix] {x.shape[0]} cells x {x.shape[1]} genes from {args.matrix}", flush=True)

    if args.cell_filter_suffix is not None:
        keep = [i for i, b in enumerate(barcodes) if b.endswith(args.cell_filter_suffix)]
        print(
            f"[cell-filter] suffix={args.cell_filter_suffix!r}: keeping {len(keep)}/{len(barcodes)} cells",
            flush=True,
        )
        if not keep:
            raise SystemExit("--cell-filter-suffix matched zero cells")
        x = sp.csr_matrix(x[keep])
        barcodes = [barcodes[i] for i in keep]

    if args.ortholog_table is not None:
        print(f"[ortholog] using explicit table {args.ortholog_table} (species auto-detection bypassed)", flush=True)
        x, gene_ids, gene_symbols = convert_to_human(x, gene_ids, gene_symbols, args.ortholog_table)
    else:
        species = detect_species(gene_ids, gene_symbols)
        print(f"[species] detected: {species}", flush=True)
        if species == "mouse":
            x, gene_ids, gene_symbols = convert_to_human(x, gene_ids, gene_symbols, args.weights_root / ORTHOLOG_TSV)

    if args.model == "scgpt":
        symbols = ensure_symbols(gene_ids, gene_symbols, args.weights_root)
        emb = embed_scgpt(x, barcodes, symbols, args.weights_root, args.device, args.batch_size, args.max_length)
    elif args.model == "geneformer":
        ids = ensure_ids(gene_ids, gene_symbols, args.weights_root)
        emb = embed_geneformer(x, barcodes, ids, args.weights_root, args.device, args.batch_size, args.gf_emb_mode)
    elif args.model == "genejepa":
        ids = ensure_ids(gene_ids, gene_symbols, args.weights_root)
        emb = embed_genejepa(x, barcodes, ids, gene_symbols, args.weights_root, args.device, args.batch_size)
    else:
        ids = ensure_ids(gene_ids, gene_symbols, args.weights_root)
        model_dir = args.scvi_model_dir or (args.weights_root / SCVI_DEFAULT_DIR)
        emb = embed_scvi_pretrained(x, barcodes, ids, model_dir, args.batch_size)

    assert emb.shape[0] == len(barcodes) and np.isfinite(emb).all(), "bad embedding matrix"
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    bc = np.array(barcodes)
    savez_stamped(
        args.out_npz,
        {
            "emb": emb.astype(np.float32),
            "barcode": bc,
            "barcodes": bc,
            "model": np.array(args.model),
            "source": np.array(str(args.matrix)),
        },
        stamp(args, corpus=args.matrix, extra={"n_cells": len(barcodes), "fm_model": args.model}),
    )
    print(f"[out] wrote {args.out_npz}  (emb {emb.shape}, {len(barcodes)} barcodes)", flush=True)

    if args.sanity_labels is not None:
        sanity_logreg(emb, barcodes, args.sanity_labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
