# scRM — the single-cell read model

Cells are embedded directly from **raw sequencing bytes** (A/C/G/T): no reference
genome, no gene annotation, no count matrix, no gene vocabulary, no ortholog
bridge.

> **GitHub:** [altoslabs/scRM](https://github.com/altoslabs/scRM) ·
> **Python packages:** `scrm_model` (model + training), `scrm_eval` (evaluation),
> `scrm_py` (the Rust packer's bindings) ·

**Licensed [MIT](LICENSE).**

Count matrices are permitted on the **evaluation side only**, as prediction
targets — predicting them from byte-derived embeddings is the cross-modality
claim (`eval/scrm_eval/targets.py`).

## Architecture

Two halves that meet at one artifact: the Rust pipeline writes an Arrow corpus,
and the Python model trains on windows packed out of it.

1. **Chemistry (`crates/scrm-chem`)** — an FGDL geometry string in, `(cb, umi,
   read)` byte slices out, executed through `seq_geom_parser`. Chemistry is
   *declared, never auto-detected*: the UMI length comes from the geometry string
   and is never inferred from a chemistry name. Scope is 10x Chromium
   single-cell polyA assays; probe-based (Flex) and spatial are out of scope.

2. **Paired FASTQ front end (`crates/scrm-fastq`)** — two gzipped files in,
   `ParsedPair`s out, over a `paraseq` paired stream with `rapidgzip-core` for
   parallel decode. Three enforced rules: R1/R2 record ids must match for every
   pair (a mismatch aborts the run); R1 is byte-identical throughout, contributing
   CB + UMI only from raw bytes; an `N` in a retained region **discards** the pair —
   no base is substituted.

3. **R2 trimming (`crates/scrm-trim`)** — **this crate trims R2, in process.**
   It measures terminal homopolymers and TSO hits into the manifest on the *raw*
   read first, so the reported rates describe the data and not the trimming, then
   trims R2 -- the 5' TSO, the 3' adapter, the quality tail and the anchored
   homopolymer runs -- and enforces the corpus-validity length floor coupled to
   `k`. R1 never appears in the crate's API. All of it is first-party:
   `quality.rs` and `adapter.rs` are written from cutadapt's documented rules, so
   no trimmer is invoked and none is a dependency. `crates/scrm-trim/src/` gives
   each step and its parameters.

4. **Cell-barcode correction (`crates/scrm-correct`)** — Hamming-1 correction
   against an **explicit list** of CellRanger-called cells (alevin-fry
   `ExplicitList` semantics, via `libradicl`'s `BarcodeLookupMap`). No knee
   finder and no 10x master whitelist: the corpus cell set is the CellRanger
   filtered list, which is what the evaluation labels are keyed by.

5. **UMI deduplication (`crates/scrm-umi`)** — `1MM_CR_flow`, per cell, with
   **no gene partition**: there is no reference, so there are no genes. Hamming-1
   candidate pairs via STARsolo's sort/swap kernel, each pair directed only by a
   *strict* count inequality (equal counts yield no edge, which makes the relation
   acyclic by construction), chains followed to their local maximum.

6. **Per-molecule assembly (`crates/scrm-asm`)** — one UMI is one captured mRNA:
   a molecule has exactly one true sequence, and every read disagreement is an
   error. Single-sequence molecules pass straight through; the rest go through
   `debruijn`'s `filter_kmers` + `compress_kmers` at **k = 19** (odd, 9..=31,
   `--k`) and a consensus walk. Bubbles are the corpus's only source of `N`, and
   bubble resolution has no flag: each disputed column is called by quality-weighted
   likelihood over the reads' Phreds (fgbio's `ConsensusCaller` model, reimplemented in
   Rust), an exact tie or a call below the confidence floor becomes `N`, and a run of two
   or more unresolved columns splits the molecule instead of emitting an `N` run. The
   rule is stamped as `bubble_resolution` and is part of `compat_key`.

7. **Corpus (`crates/scrm-corpus`)** — Arrow IPC (Feather v2), one row per
   molecule, **sorted by `cell_id` with one contiguous run per cell** — the
   dataloader reads a cell as one row range.
   Columns: `cell_id` `Dictionary<Int32,Utf8>`, `umi` `UInt64`, `n_reads`
   `UInt32`, `n_contigs` `UInt16`, `tokens` `List<UInt8>`. Provenance is split —
   the *declared* stamp goes in the schema's `custom_metadata` (header, before any
   batch), the *measured* stamp (row count, token count, token sha256, per-cell
   row counts) into the IPC footer. `verify` walks the shards on disk and checks them against the manifest.

8. **Window packing (`crates/scrm-pack`, bound by `crates/scrm-py`)** — a
   corpus becomes training windows: `seq_len = 16384` tokens, molecules tiled in
   with `NEWLINE` separators, contigs as local-attention blocks and UMIs as
   patches. All the logic is in `scrm-pack` (pure Rust, mmap + rayon, exercised
   by `cargo test` with no Python in the loop); `scrm-py` is bindings and no
   logic, running `build_batch` under `Python::detach` so rayon works with the
   GIL released. Six numpy arrays per batch cross the boundary once.
   `max_umi_len = 1600` is a tenth of `seq_len`, so no molecule can fail to fit a
   window; `assert_geometry` checks the ratio at open time.

9. **The model (`model/scrm_model/model.py`)** — the hierarchy is
   **bytes ⊂ contig ⊂ UMI ⊂ window ⊂ cell**. A byte-level local trunk, then one
   masked `SegmentAttentionPool` per level (`pooling.py`): `contig_pool`
   (`dim_local → contig_out`), `umi_pool` (`contig_out → dim_global`, *the*
   patching step, where the width change lives), a global trunk over UMI slots
   with positionless attention (`tok_idx = 0`, the identity rotation), then
   `window_pool` and `view_pool` for the cell. Empty segments pool to exactly
   zero by construction; invalid byte rows are zeroed after the local trunk and
   padding UMI slots after the global trunk. Both trunks are
   `scrm_model.trunk.BaseTransformer`, which is **not vendored**: a LLaMA-style
   pre-norm stack, substantially all PyTorch calls — `nn.RMSNorm`, `F.silu` over
   three `nn.Linear` for SwiGLU, and `F.scaled_dot_product_attention` or a
   `torch.compile`-wrapped `flex_attention` for the attention itself.

10. **The objective (`model/scrm_model/losses.py`)** — four forwards, one EMA
    teacher read at two levels, four terms: `mlm` (block-masked bases,
    `masking.py`), `ema_umi`, `ema_cell`, and a regularizer. The regularizer is
    **first-party** `losses.SIGReg` — the Epps–Pulley normality statistic on random
    1-D slices, 17 quadrature points over `[-3, 3]` weighted by `exp(-t^2/2)`,
    which is the standard normal characteristic function. `losses.py`'s module
    docstring itemises which parts of it come from LeJEPA's published description
    and which are this tree's. Mixed as
    `lamb * reg + (1 - lamb) * inv` by the loss assembly. SIGReg is the objective,
    not a choice: `train.py` builds it unconditionally and no flag selects a
    regularizer. `lamb` is a flag.
    Masking is data, computed in the prefetch path, not on the GPU-critical path.

11. **Evaluation (`eval/scrm_eval/`)** — the embedding artifact is **per
    window** and carries a stamp that cannot be omitted. Every window-to-cell rule
    is an offline reduction of that file.

## Training arms and the regularizer

Arms are declared in `model/scrm_model/config.py` (`arms()`) and selected with
`--arm`. There are **three**, and `train.py --help` prints exactly that set —
`--arm {S96,S96ML,S96MA}`. **S96** is the flagship (`dim_global = dim_local =
96`, so no pool widens, `pool_variant = attention`). **S96ML** and **S96MA** are
that same architecture with mean pooling instead — `mean_late` replaces the
window and view pools, `mean_all` replaces all four — and they exist for the C7
ablation rows, which is why each carries a declared run:
`armS96ML-c3k-sigreg-lam002` and `armS96MA-c3k-sigreg-lam002` in
`registry.ABLATION_DELTAS`. `eval/tests/test_no_hardcoded_lists.py` asserts that
`arms()` and the registry's arms are the *same* set in **both** directions, so an
architecture with no run declared on it is a test failure rather than a latent
`--arm` option — these two were removed once already when their runs were, and
they come back only alongside them. `pool_variant` itself takes `attention`,
`mean_late` or `mean_all`, and `model/tests/test_layers.py` exercises all three.
An arm is architecture; the
regularizer is neither architecture nor a flag. SIGReg is a constant in the code
(`train.REGULARIZER`), recorded in every checkpoint, and `restore_checkpoint`
refuses to resume one trained under a different objective. A flag with one value
tunes nothing, so there is none.

**Every `train.py` default that names a training choice is the flagship's**, read
from one declaration (`config.FLAGSHIP_RECIPE`): `--arm`, `--lamb`, `--seed`,
`--train-cells` and `--train-frac`. A launcher that omits a flag gets the
flagship. The recipe's remaining fields — `steps`, `corpora` and `regularizer` —
are `config.FLAGSHIP_RECORDED_ONLY`: recorded, and not arguments.

`seq_len` moves only on a sanctioned ladder — `SEQ_LEN_OPTIONS = (16384, 32768,
49152)` — each rung recorded with its measurement in `config.py`. Runs at
different rungs are different experiments and their curves are not comparable.

## Install

Rust >= 1.85 (edition 2024) and Python >= 3.11.

```bash
pip install -r requirements.txt     # the Python side

cargo build --release               # target/release/scrm-corpus
cargo build --release -p scrm-py && mkdir -p target/py \
  && cp target/release/libscrm_py.so target/py/scrm_py.so   # the packer's bindings
```

Every Python entrypoint expects `PYTHONPATH=model:eval:target/py`. The two
packages are used from the tree; there is no install step for them.

`envs/` holds the exact environment the published runs used, as conda and pip
freeze files.

## Building the corpus

**R2 is trimmed by the builder.** Nothing external is needed: no Trim Galore, no
pre-step, and no trimmer dependency at all. `scrm-corpus` clips the 5' TSO, the 3' adapter, the
quality tail and the anchored homopolymer runs, then applies the length floor with the read
pair still intact. R1 is never touched -- it carries the barcode and UMI at fixed offsets.
`crates/scrm-trim/src/` gives each step, its parameters, and why a corpus built before trimming
moved into the builder cannot be reproduced from this tree.

Inputs then come from a **samplesheet CSV**, one row per read pair — nothing is
inferred from filenames:

```text
experiment,r1,r2,barcodes,geometry
mouse_brain_5k,fq/..._L001_R1_001.fastq.gz,fq/..._L001_R2_001.fastq.gz,wl/mb.tsv.gz,1{b[16]u[12]}2{r:}
```

`geometry` is an FGDL string, not a chemistry name, and is the one input allowed
to vary per experiment. Everything in `compat_key` — `(alphabet_size, k,
dedup_rule, bubble_resolution, min_umi_len, max_umi_len, min_cell_molecules)` — is
a build-wide flag: corpora that disagree on it cannot be mixed
(`Corpus::open_many` hard-fails).

```bash
target/release/scrm-corpus \
    --samplesheet <your samplesheet.csv> \
    --out $SCRM_PROJ/corpus_v3 \
    --threads 32
```

Each experiment is built into `<out>/<experiment>/`, and `manifest.json` is
written **last**: a directory of shards without one is an incomplete build, not a
corpus. Defaults: `--k 19`, `--min-umi-len 60`, `--max-umi-len 1600`,
`--min-cell-molecules 500`. There is no `--limit`: to build from part of a dataset,
truncate the FASTQ. Reads are spilled to disk in cell-rank order
(`crates/scrm-cli/src/collate.rs`) so peak memory is one shard, not one dataset.

`model/jobs/rebuild_corpus.sbatch` is the cluster version, restart-safe at dataset
granularity.

## Training, embedding, scoring

All three run from the repo with `PYTHONPATH` covering the three source roots:

```bash
export PYTHONPATH=model:eval:target/py
export SCRM_PROJ=/path/to/data-root   # registry.require_root raises without it, so the
                                      # full_coverage lookup below needs it, as does
                                      # run_eval's --corpus-root default

# train
python -m scrm_model.train \
    --corpus $SCRM_PROJ/corpus/{mouse_brain_5k,neuron_1k_v3,neurons_5k,sod1_spinal_cord} \
    --token-passes 1 --out runs/armS96-c3k-sigreg-lam002

# embed a checkpoint over ONE held-out dataset and score it against the controls + count-FM panel.
# --datasets and --w-eval are both required. --datasets refuses a TRAIN_ONLY name, and --w-eval is
# one integer applied to every name given, so group only datasets that share a requirement.
# The value is the dataset's own all-reads requirement from eval_artifacts/windows_required.json --
# do not transcribe it, read it (registry.full_coverage reads the same file, and
# measurements/windows_required.py --write --repin is what moves it):
W=$(python -c 'from scrm_eval.registry import full_coverage; print(full_coverage("nuclei_2k"))')

python -m scrm_model.run_eval \
    --checkpoint runs/armS96-c3k-sigreg-lam002/latest.pt --arm S96 \
    --tag armS96-c3k-sigreg-lam002 --out-dir eval_runs/armS96-c3k-sigreg-lam002 \
    --datasets nuclei_2k --w-eval "$W"

# probes on ours + every baseline artifact for one dataset (E6/E9)
python -m scrm_eval.run_probes \
    --artifacts eval_runs/<arm>/nuclei_2k.<arm>.arrow /path/to/eval_artifacts/nuclei_2k.*.arrow \
    --dataset nuclei_2k --gmt <hallmark>.gmt -o probes.json

# cell-line annotation macro-F1 from a barcode→label JSON (E2)
python -m scrm_eval.run_annotation \
    --artifacts eval_runs/<arm>/sc_mixology_5cl.<arm>.arrow \
    --labels sc_mixology_5cl_labels_sng.json -o e2.json
```

Three properties of the training loop are requirements: the data loader is never
the bottleneck (`data_wait_ms` is logged every step and must stay under 1% of
`step_ms`); SIGTERM checkpoints and exits 0 so a preempted job requeues cleanly
from an atomically written `latest.pt`; and `total_steps` is counted in *token
passes* over the corpus, which is what the LR and mask-ratio cosines are built on.
`--init-from` continues pretraining from another run's weights with a fresh
schedule, distinct from resuming into `--out`.

On the eval side, `embed()` takes **no architecture arguments** — they are read
from the checkpoint, whose `arch_sha` is recomputed and asserted — and a model
artifact without a readout spec is unrepresentable on disk. Controls are a
different `producer` writing the identical schema, never a separate code path:
`count_scanpy` (the count-space baseline), `counts_10x_pca10` (10x's own PCA — a
ceiling, and partly circular for label metrics) and `depth2` (log total counts,
log genes detected — the floor a representation clears by carrying library size
alone). Window→cell reductions are named in every report: `mean`, `median`,
`max`, `l2_mean`, `first`.

`scrm_eval.cli` is the single entrypoint that consumes N artifacts at once and
refuses to compare artifacts over different cell populations or corpus content,
or to credit an A/B difference without reporting whether the embeddings differ at
all.

## Slurm

Launchers live in `model/jobs/`, 40 of them. Submit every one with `sbatch`,
never `./` — but only **10 carry the guard that enforces it**
(`if [ -z "${SLURM_JOB_ID:-}" ]`, in `baseline_probes`, `e2_protocol_match`,
`memsweep`, `mixology_baselines`, `mixology_fm`, `panel`, `rebuild_corpus`,
`restamp_baselines`, `smoke` and `speed`; list them
with `grep -lF 'if [ -z "${SLURM_JOB_ID' model/jobs/*.sbatch`). The other 30
have no such check at all: 8 of them open with `${1:?usage: sbatch $0 …}`, which
refuses a *missing argument*, not a login-node run, and the remaining 22 refuse
nothing. A bare `./` on any of the 30 starts executing on the login node.

Account, partition and log directory come from the environment
(`SBATCH_ACCOUNT`, `SBATCH_PARTITION`, `SBATCH_OUTPUT`) — no launcher carries
`#SBATCH --output`, `--partition` or `--account`, so every `sbatch` needs the
first two from the environment or the command line. `watchdog.sbatch` reads its
own placement from `scontrol` when it resubmits itself, so the chain follows
whatever you submitted the first hop with.

## Repository layout

| Path | Role |
|---|---|
| `crates/scrm-core/` | The shared contract: token alphabet, the single definition of `k` (`K19`), `Molecule`, `LenFilter`, the provenance `Stamp`. Does no work. |
| `crates/scrm-chem/` | FGDL geometry execution → `(cb, umi, read)` slices |
| `crates/scrm-fastq/` | Paired gzipped FASTQ stream, id assertion, CB/UMI extraction, `N` discard |
| `crates/scrm-trim/` | R2 artifact measurement (terminal homopolymers, TSO) and the `k`-coupled length floor. Trims R2 in process: the 5' TSO, the 3' adapter, the quality tail and the anchored homopolymer runs |
| `crates/scrm-correct/` | CB Hamming-1 correction against the CellRanger called-cell list |
| `crates/scrm-umi/` | `1MM_CR_flow` UMI dedup, no gene partition; carries the dedup criterion bench |
| `crates/scrm-asm/` | Per-(CB, UMI) cDBG assembly at k = 19 + bubble resolution; carries the assembly bench |
| `crates/scrm-corpus/` | The Arrow corpus format: writer, reader, manifest, and the corpus checkers in its `tests/` |
| `crates/scrm-pack/` | Window tiling and batch packing — all the packing logic, pure Rust |
| `crates/scrm-py/` | PyO3 `cdylib` for `scrm-pack`. Bindings and no logic. |
| `crates/scrm-cli/` | `scrm-corpus`, the binary where the nine library crates meet; samplesheet parsing and the cell-rank spill |
| `model/scrm_model/` | The model (`model.py`, `pooling.py`, `projector.py`), objective (`losses.py`, `masking.py`, `teacher.py`), `config.py` (`ArchConfig` + `arms()`), `optim.py`, `guards.py`, the `train.py` loop, `run_eval.py`, `evalio.py`, and the `bench_*` gates. Nothing is vendored; `trunk.py` and `losses.SIGReg` each state in their header what they take from published work, and the notices owed are in `THIRD_PARTY_NOTICES.md` |
| `model/jobs/` | Slurm launchers; account, partition and log directory come from the environment |
| `model/tests/` | Guards, layer invariances, masker obligations, DDP parity, training machinery — CPU-only via the `sdpa` path |
| `eval/scrm_eval/` | The harness: `embed.py`, `artifact.py` (the stamped per-window format), `controls.py`, `metrics.py`, `probes.py`, `targets.py`, `reduce.py`, `tenx.py`, `run_probes.py`, `run_annotation.py`, `cli.py` |
| `eval/tests/` | Metrics against embeddings whose answers are known by construction; the artifact contract; the paper's generated fragments |
| `measurements/` | Analysis and measurement scripts (coverage sweeps, assembly/dedup/homopolymer measurements, ETA and preemption cost) — including `results_page.py`, which derives the paper's generated table fragments from the stamped reports on disk, transcribing nothing |
| `check.sh` | Everything that can fail, in one command |

## Checks

```bash
./check.sh                              # fmt, clippy, cargo test, and the four pytest suites
./check.sh /path/to/corpus_dir ...      # the above, then the real-data checks per corpus
```

`SCRM_PY` names the interpreter and defaults to `python3`; set it to the one that
has `requirements.txt` installed. `SCRM_PROJ` adds the seed-floor check, and
`SCRM_CORPUS_ROOT` names the corpus tree.

The Python checks are separate from `cargo test`: they need a real corpus and
take minutes. They cover what Rust cannot — whether Arrow C++ reads what arrow-rs
wrote (`pyarrow_compat.py`), whether the cell set matches the CellRanger calls the
build promised to adopt (`cell_accounting.py`), and whether content matches the
pinned digests (`pinned_digests.py`, run once per tree). The model and eval suites
need no GPU and no corpus.

## License

**MIT** — the full text is in [`LICENSE`](LICENSE); the SPDX identifier is `MIT`, declared once in
`Cargo.toml`'s `[workspace.package]` and inherited by every crate. The copyright notice and the
permission notice travel with any copy or substantial portion.

Third-party obligations are separate and survive independently of this licence, and the notices they
require are in `THIRD_PARTY_NOTICES.md`. Not every module carries a header — 38 of the 46 tracked
files under `crates/*/src/` have none (`git ls-files 'crates/*/src/**/*.rs' 'crates/*/src/*.rs' |
sort -u | while read f; do head -1 "$f" | grep -q '^//!' || echo "$f"; done`), and most of those
implement nothing anyone else published. What is true is
narrower and is the part that matters: **every module that implements a published method carries a
`//!` header naming that method, its source project and its licence**, and points at
`THIRD_PARTY_NOTICES.md`. Those are `scrm-correct/src/lib.rs` (alevin-fry `ExplicitList`),
`scrm-umi/src/lib.rs` (STARsolo's sort/swap kernel, implemented in its `pairs.rs`),
`scrm-umi/src/flow.rs` (CellRanger `1MM_CR_flow`), `scrm-asm/src/lib.rs` (the `debruijn` crate),
`scrm-asm/src/consensus.rs` (fgbio's `ConsensusCaller`), `scrm-trim/src/lib.rs` with its
`quality.rs` and `adapter.rs` (BWA's and cutadapt's trimming rules), and on the Python side
`model/scrm_model/trunk.py` (BLT, and the expression shared with lingua) and
`model/scrm_model/losses.py` (LeJEPA and Epps & Pulley). `eval/countfm/fetch.py` records every baseline embedder
and weight file with its source, its size and its sha256. **Those weights are not redistributed
here**, and their terms are their publishers' own: this repository does not restate them, because a
licence named here and named wrongly is worse than a pointer to the source.
