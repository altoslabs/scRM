# Reproducing the paper

## Prerequisites

`reproduce.sh` reads its configuration from the environment and refuses to run when any of these
four is unset:

| variable | what it names |
| --- | --- |
| `SCRM_PROJ` | the data root — corpora, runs, `eval_artifacts` |
| `SCRM_PY` | the interpreter of the `scrm` environment, as an **absolute** path |
| `SCRM_SLURM_ACCOUNT` | the slurm account every job is charged to (exported as `SBATCH_ACCOUNT`) |
| `SCRM_SLURM_PARTITION` | the slurm partition every job is submitted to (exported as `SBATCH_PARTITION`) |

**`SCRM_PY` is not optional and must be absolute.** Every launcher, `check.sh` and `reproduce.sh`
derive `LD_LIBRARY_PATH` from it as `$(dirname $(dirname $SCRM_PY))/lib`. Left unset it defaulted to
the bare name `python3`, and that derivation yields the *relative* path `./lib`, i.e. an
`LD_LIBRARY_PATH` naming nothing. Measured: the model suite then fails two `test_ddp_parity` tests
with `ImportError: … CXXABI_1.3.15 not found` out of scipy — an environment failure that reads as a
code failure. `check.sh` and `reproduce.sh` now resolve the interpreter with `command -v` and
**refuse** a relative or nonexistent `lib/`, naming which of the two it is.

Optional: `SCRM_CORPUS_ROOT` (default `$SCRM_PROJ/corpus_rel` — see *What this release can reproduce
today*), `SCRM_LOG_DIR` (sets `SBATCH_OUTPUT` to `$SCRM_LOG_DIR/%x-%j.out`), `SCRM_PAPER` (default
`../scrm-paper`, stage 6 only) and `SCRM_TEX_BIN` (stage 6c only).

    export SCRM_PROJ=/path/to/data-root
    export SCRM_PY=/path/to/envs/scrm/bin/python
    export SCRM_SLURM_ACCOUNT=<account>
    export SCRM_SLURM_PARTITION=<partition>

    ./reproduce.sh --dry-run all     # print the plan, run nothing that submits or writes
    ./reproduce.sh 0                 # ONE stage at a time: 0, then 1, then 2 ...

## `reproduce.sh` prints a plan and submits one stage. It does not sequence the stages.

Read this before running it, because the script's name oversells it:

- **No stage waits for the one before it.** No stage passes `--dependency` to the next: a stage
  `sbatch`es its jobs and returns while they are still queued. Run one stage, watch it land with
  `squeue --me`, then run the next. `./reproduce.sh all` runs the stages back to back against inputs
  that do not exist yet, and is useful only with `--dry-run`.
- **Convergence is by repetition.** Each stage skips what is already on disk, so re-run a stage
  until it reports nothing left to submit.
- **The pin and the training are manual.** Stage 2 submits the per-dataset corpus builds but only
  *prints* the two `pin_corpus.sbatch` commands; stage 3 submits nothing at all and prints one
  `sbatch` per run. The printed lines are marked `MANUAL`, and *From a clean clone to the tables*
  below is the same path written out.
- **Stage 5 is the one real dependency graph.** `model/jobs/step2_suite.sh` chains
  embed → score → measure with `--dependency` inside itself, and checks every precondition before it
  submits anything.
- `--dry-run` executes only read-only introspection, announced with `?` on stderr; everything that
  submits or writes is announced with `$` and skipped.

## What this release can reproduce today

The paper's scope is twenty corpora. **This tree has built and pinned all twenty**, and
`crates/scrm-corpus/tests/expected_stream_digests.json` is the authority — it names exactly:

    2500_rhesus_monkey_pbmcs_singleplex_3p_gem_x  2500_wistar_rat_pbmcs_singleplex_3p_gem_x
    crispr_a549_5k  fly_antenna  got_ET01  heart_10k_v3  mouse_brain_5k
    mouse_brain_5k_permuted  neuron_10k_v3  neuron_1k_v3  neuron_1k_v3_permuted
    neurons_5k  neurons_5k_permuted  nuclei_2k  pbmc_10k_v3  pbmc_1k_v2
    sc_mixology_5cl  sod1_spinal_cord  sod1_spinal_cord_permuted  tm_10X_P7_2

All twenty sit under one `bounds` key, so they are one molecule definition and
`Corpus::open_many` accepts them together.

Everything a pin gates is therefore available for those six and for nothing else:

- **All three of `registry.REQUIRED_RUNS` are trainable now.** Their recipes need only the four
  pretraining corpora (`mouse_brain_5k`, `neuron_1k_v3`, `neurons_5k`, `sod1_spinal_cord`) and
  `sc_mixology_5cl`, all of which are pinned. Verify with
  `$SCRM_PY -m scrm_eval.registry --recipe <run>`.
- **All nine evaluation datasets (`registry --datasets`) are built and pinned**, measured against
  `corpus_rel`. A build that has landed in the root but is not in the fixture is still refused —
  being on disk is not being pinned. Note that `sc_mixology_5cl` is the **in-domain** dataset, so
  it is not a held-out result on its own.
- The C3 permuted control's four `*_permuted` corpora are built and pinned. Each carries exactly
  its source's row and cell counts, which is what permutation preserves.
- An unpinned corpus cannot be embedded at all: `controls.corpus_facts` reads the pinned stream
  digest, so it cannot be stamped. Pin before embedding, never the reverse.

**`$SCRM_PROJ/corpus_rel` is the release's corpus root, and it is `reproduce.sh`'s default.** The
predecessor root `corpus_v3` is not: its twenty corpora were built before trimming moved into the
builder, so their manifests carry no `trimmed_here` in the trim stamp and
`crates/scrm-corpus/tests/test_corpus_facts_visibility.py` fails on all twenty
(`test_trim_configuration_is_actually_stamped`, gap `['trimmed_here']` per corpus). `check.sh` runs
that test, so the old default made `./reproduce.sh all` die at its own stage 0. Against `corpus_rel`
the same test passes, and `pinned_digests.py $SCRM_PROJ/corpus_rel` verifies all twenty
pinned corpora `OK` — the pins in this tree are this tree's own corpora, re-derived from the shards
on disk rather than asserted.

Counts below that say "20" or "the 20" are the paper's scope, not what this tree can rebuild today.

| # | stage | what it does | driver |
| --- | --- | --- | --- |
| 0 | code | rustfmt, clippy, `cargo test`, four pytest suites | `check.sh` |
| 1 | acquire | fetch reads for the 20 datasets | `ACQUISITION` in `measurements/verify_acquisition.py` |

Stage 2's builder trims R2 itself (`crates/scrm-trim`, in process), so nothing external is needed and
there is no separate trimming stage. Corpora built before trimming moved into the builder
cannot be reproduced from this tree, because their manifests carry no `trimmed_here`;
`check.sh` refuses such a corpus root and says so by name.

| 2 | corpus | submit the per-dataset builds; **prints** the pin, does not submit it | per-dataset launchers + `pin_corpus.sbatch` |
| 3 | train | **prints** one `sbatch` per run; submits nothing | `registry.runs()` + `arm_ablation.sbatch` |
| 4 | baselines | count controls, 4 count FMs, UCE | `baselines_matrix.sbatch`, `uce_matrix.sbatch` |
| 5 | eval | embed → score → measure, one real `--dependency` DAG | `model/jobs/step2_suite.sh` |
| 6 | paper | write every generated table, then verify none drifted | `results_page.py --write-tex` / `--check-tex` |

Stage 2 covers the paper's twenty; six of them are pinned in this tree. Stage 3's list is the eleven
runs in `registry.runs()`, of which the three in `registry.REQUIRED_RUNS` are the ones every scorer
blocks on.

## From a clean clone to the tables

The path by hand, because stages 2, 3 and the render are manual. **No launcher carries `#SBATCH
--partition` or `--account`** — measured over the 40 files in `model/jobs/*.sbatch` with
`grep -l '^#SBATCH.*--partition\|^#SBATCH.*--account' model/jobs/*.sbatch`, which returns nothing.
So every `sbatch` below needs both, from the environment or on the command line. `watchdog.sbatch`
used to pin them to the cluster that produced the paper; it now reads its own placement from
`scontrol` when it resubmits itself, so the chain follows whatever you submitted the first hop with.
Export them once and slurm picks them up (`reproduce.sh` does exactly this), or pass
`--partition=… --account=…` on every command:

    export SCRM_PROJ=/path/to/data-root
    export SCRM_PY=/path/to/envs/scrm/bin/python
    export SCRM_CORPUS_ROOT=$SCRM_PROJ/corpus_rel      # the release's root; not $SCRM_PROJ/corpus
    export SCRM_REPO=$PWD
    export SBATCH_PARTITION=<partition>                # your Slurm partition
    export SBATCH_ACCOUNT=<account>                    # your Slurm account
    export PYTHONPATH=model:eval:target/py

All 40 do carry `--requeue` and their own `--mem` and `--time`, because the partition the paper's
runs used is a scavenger tier with `GraceTime=0`: a preemption arrives with no SIGTERM window, so
state has to be on disk before the kill, and resume is what carries a job across it.

`SCRM_CORPUS_ROOT` is not optional for the hand path: every launcher's own default is
`$SCRM_PROJ/corpus`, and `arm_ablation.sbatch` will refuse with "has no manifest" against it.

**1. The gate.** `./check.sh` must print `CODE CHECKS PASSED`. With no corpus named, the corpus half
does not run and `check.sh` says so in its verdict line; that is code coverage only. Naming corpora
on the command line — `./check.sh $SCRM_CORPUS_ROOT/*` — additionally runs `pyarrow_compat`,
`cell_accounting` and `pinned_digests` over them, and the verdict becomes `ALL CHECKS PASSED`.
`pinned_digests` in verify mode walks the **disk**, so it fails while a build that is not yet pinned
is sitting in the root: a corpus present but unpinned is a failure, not something invisible.

**2. The reads, then the corpus, one dataset at a time.** `./reproduce.sh --dry-run 1` and
`--dry-run 2` print the fetch and build launcher for every dataset; run them one at a time. The
corpus launchers **delete their FASTQ once `manifest.json` exists** — the manifest is written last,
so it is the only honest completion marker — because the reads are far larger than the corpus and
the filesystem is shared.

**3. Pin, then embed — never the reverse.**

    sbatch model/jobs/pin_corpus.sbatch $SCRM_CORPUS_ROOT --report   # read-only; non-zero only for CHANGED
    sbatch model/jobs/pin_corpus.sbatch $SCRM_CORPUS_ROOT --write    # adds new entries to the fixture

A **mismatch** on an existing entry is not a reason to run `--write`: it means the pipeline changed
the data or the corpus is corrupt. `--write` is for a new dataset, and the commit must say which
entries moved and why.

**4. The three arms every scorer blocks on.** `model/jobs/score_one.sbatch` reads
`registry --required` and waits until each of these has an artifact, so a missing one is a deadlock,
not a gap:

| run | what it is |
| --- | --- |
| `armS96-c3k-sigreg-lam002-s20260903` | the reported flagship (`registry.FLAGSHIP`), seed 20260903 |
| `armS96-sc_mixology_5cl` | trained on `sc_mixology_5cl` only — the in-domain arm |
| `armS96-c3k-sigreg-lam002` | the same recipe at seed 20260817 — the seed floor |

One launcher produces all three, and it takes the run **name**:

    sbatch model/jobs/arm_ablation.sbatch armS96-c3k-sigreg-lam002-s20260903
    sbatch model/jobs/arm_ablation.sbatch armS96-sc_mixology_5cl
    sbatch model/jobs/arm_ablation.sbatch armS96-c3k-sigreg-lam002

It resolves the recipe itself — `eval "$($SCRM_PY -m scrm_eval.registry --recipe "$RUN")"`, which is
`registry.FLAGSHIP_RECIPE` with that run's declared `ABLATION_DELTAS` entry applied — so there is no
flag that can disagree with the registry. Check what a name resolves to before submitting:

    $SCRM_PY -m scrm_eval.registry --recipe armS96-c3k-sigreg-lam002-s20260903
    # ARM=S96 CORPORA='mouse_brain_5k neuron_1k_v3 neurons_5k sod1_spinal_cord'
    # LAMB=0.0196 REGULARIZER=sigreg SEED=20260903 STEPS=8947 TRAIN_CELLS=3000 TRAIN_FRAC=1.0

It refuses, and each refusal is the point: a corpus not in `expected_stream_digests.json`
(*"run pinned_digests.py --report / --write and commit first"*); a shard directory with no
`manifest.json` (*"a directory of shards is not yet a corpus"*); a run that already has
`$SCRM_PROJ/runs/<run>/latest.pt` (pass `RESUME_LONGER=1` to train it further, which re-checks the
checkpoint's `corpus_compat_key` against the corpus's molecule definition); and a run declared
`untrained`, which is random init and comes from `measurements/untrained_control.py --write`
instead. That script runs **after** the flagship has trained, not in parallel with it: it copies the
architecture out of `$SCRM_PROJ/runs/$FLAGSHIP/<reported checkpoint>`, cross-checks the arm it finds
there against the registry, and exits with `<path> does not exist` when the checkpoint is not there.

The other eight runs in `registry.runs()` are `OPTIONAL_RUNS` — scored when present, never blocking
— and are produced the same way: `sbatch model/jobs/arm_ablation.sbatch <name>` for each of
`$SCRM_PY -m scrm_eval.registry --optional`, minus the three untrained ones.

**5. Baselines.** `./reproduce.sh 4` submits `embedder_equiv.sbatch` (first: it measures that the
count-FM embedder and weights are the ones that produced the panel), then `baselines_matrix.sbatch`
and `uce_matrix.sbatch`. These need the third-party weights, which are not redistributed here — see
*What this workflow cannot do*.

**6. Embed, score, measure.**

    ./model/jobs/step2_suite.sh        # DRY=1 to see the DAG without submitting it

This is the one stage that chains. It checks its preconditions first and submits nothing if any
fails: every evaluation dataset present **and** pinned, a usable `w_eval` for each, and a reported
checkpoint at its declared step count for **every** run in `registry.runs()` — not just the
required three. The corpus half of those preconditions is now met for all twenty; whether it
submits depends on the checkpoint half, which it reports by name.

**7. The tables.**

    PYTHONPATH=model:eval:target/py $SCRM_PY measurements/results_page.py --write-tex generated

That is the public terminus: it reads the stamped reports on disk and derives every published
number, transcribing nothing. `results_page.py --write-tex` and `./reproduce.sh 6` additionally need
the private paper tree and refuse without it.

## Every number in the paper is generated

A paper table's body is `\input{generated/<name>}`. The producer that DERIVES the numbers also WRITES
the fragment, so there is one derivation and no copy step. `--check-tex` re-derives and compares:

    ./reproduce.sh 6        # writes, then verifies

A fragment that has drifted from the reports is a **failed check**, not a wrong figure in a PDF. A
cell no report can produce is emitted as `\scrmMissing{<task>}`, which renders conspicuously as
`??` and makes the check fail, so a row cannot quietly vanish from a table.

**The guarantee is checked in the built PDF, not in the fragment.** Stage 6c compiles the paper
(tectonic 0.17.0 lives in the `tex` environment) and `tools/verify_generated.py` reads every number
out of the PDF's pages. `\input` inside a `tabular` fails with `! Misplaced \noalign` — LaTeX's
`\input` does file bookkeeping whose grouping breaks an alignment. `math_commands.tex`
defines `\scrmInput` over TeX's primitive `\@@input`, which does not. Currently: **56 of 56
generated numbers verified on the page**, and 3 `??` cells awaiting E7.

**Generated from this repository** — nine fragments, all derived, none transcribed:
`results_page.py --write-tex` writes `e8_body`, `c5_body`, `e7_body`, `e7_n`, `e7full_body`,
`specialize_body` and `specialize_n`; `ablation.py --write-tex` writes `ablation_body` and
`ablation_n`. Together they carry `tab:e8` (Table 1), `tab:c5`, `tab:e7`, `tab:e7full`,
`tab:specialize` and `tab:ablation`.

**Generated, but OUTSIDE the `--check-tex` guarantee:** `figures/species_data.tex` (Figure 2)
is derived by `measurements/figure_species.py -o`, which reads
`eval_reports/uce/b_logreg.<dataset>.json` and computes Wilson intervals itself. It does not go
through `emit_tex`/`write_tex`, so `--check-tex` never compares it and a stale copy cannot be
caught by stage 6. It is also absent from the generate driver, so it is the one published figure
that is regenerated only when someone remembers to. Two things about it are worth knowing: the
protocol-B reports it reads carry **no stamp on any row** — no `checkpoint_sha256`, no `arch_sha`,
no `forward_precision`, no `w_eval` — and `figure_species.py` REFUSES (`KeyError`) rather than
drop a model, so it does not run at all until every plotted model has a row in every column.

`tab:dataurls` is generated too, by `measurements/verify_acquisition.py --write-tex`, which
stage 6 now writes and stage 6b checks alongside the others.

**The channel that is NOT covered by the guarantee is the appendix prose.** The paper tree's
`tools/derive_appendix_numbers.py` reads reports this repository produces -- `arm_geometry`,
`seed_floor`, `paired_significance`, `e4_vocab_reach`, `e7_isoform_structure`,
`collapse_trajectory`, `e7_vs_control`, the probes and annotation reports -- prints them to
`.ablation-data/derived.txt`, and a human types the result into the appendix. Those numbers reach
the paper without an `\input`, so `--check-tex` cannot see them drift. That is why the producers
in that list are kept even though no fragment reads them.

## What this workflow cannot do

- **Stage 6 is not publicly reproducible, and it refuses instead of half-running.** It needs
  `$SCRM_PAPER` — a separate, **private** repository holding `main.tex`, the `generated/` fragment
  directory, `tools/derive_appendix_numbers.py` and `tools/verify_generated.py`. Nothing in this
  repository can supply it and no variable can stand in for a tree that is not there, so
  `./reproduce.sh 6` prints what is missing and **exits non-zero**. `--dry-run 6` still prints every
  command.

  **What a public reader loses is the typesetting half, not the numbers**: stages 0–5 need nothing
  private, and `measurements/results_page.py --write-tex` derives every published figure from
  the stamped reports on disk with no paper tree involved.

- **It is not cluster-independent.** It submits to Slurm — account and partition from
  `SCRM_SLURM_ACCOUNT` / `SCRM_SLURM_PARTITION` — and reads `$SCRM_PROJ`. External reproduction
  additionally needs the weights below and the four environments pinned in `envs/`.

- **~17 GB of third-party weights are not redistributed here.**
  `eval/countfm/fetch.py` records every file, its measured size and its source, and pins each
  by sha256. Their terms are their publishers' own and are not restated here: follow the recorded
  source for each.

- **`figures/framework.tex` and `figures/design/` are hand-drawn**, not derived.
  `figures/collapse.tex` has no producer either; `figure_species.py` is the only figure generator.

## The cross-species panels

`2500_rhesus_monkey_...` and `2500_wistar_rat_...` are two rows of `model/jobs/corpora.tsv` like any
other, built by `model/jobs/corpus.sbatch <key>`. They are `role=train`, so they never appear in an
evaluation table; they carry the cross-species figure.

- Each row DECLARES `1{b[16]u[12]}2{r:}`, and both corpora's `manifest.json` records exactly that
  geometry (`cb_len` 16, `umi_len` 12). Chemistry is declared, never auto-detected.
- Each declares `whitelist=h5-multi`, because 10x publishes these samples' filtered matrix in a
  doubled directory under the name with `_Multiplex` dropped. MEASURED: that URL answers 200 for
  these samples and the plain `<sample>/<sample>_filtered_feature_bc_matrix.h5` answers 403.
- `measured.counts.cells_called` is 2,714 and 2,474.
- Rhesus needs no ortholog bridge: 10x aligned to rheMac10 but annotated against human gene models,
  so every gene carries an ENSG id. Rat does need one.

Build them like any other row:

    sbatch model/jobs/corpus.sbatch 2500_rhesus_monkey_pbmcs_singleplex_3p_gem_x
    sbatch model/jobs/corpus.sbatch 2500_wistar_rat_pbmcs_singleplex_3p_gem_x

## The four environments, pinned in `envs/`

`envs/` holds the four environments the published runs used. Three are micromamba environments;
`uce-scfm-extra` is **not an environment** — it is a pip `--target` directory prepended to
`PYTHONPATH`, and `micromamba env export` does not cover it.

| environment | stages | what runs there | files |
| --- | --- | --- | --- |
| `scrm` | 0, 1, 2, 3, 5, 6 | every Python entrypoint (`check.sh`, `verify_acquisition`, `registry`, the corpus builders' helpers, `step2_suite.sh`, `results_page.py`) | `.conda-explicit.txt`, `.yml`, `.pip-freeze.txt` |
| `scfm` | 4 | the count-FM embedder (`$SCRM_PROJ/countfm/scripts/embed_count_fm.py`) and `measurements/embed_uce.py` | same three |
| `uce-scfm-extra` | 4 | prepended to `PYTHONPATH` for the UCE run only (`model/jobs/uce_matrix.sbatch`) | `.pip-freeze.txt` |
| `tex` | 6c | `tectonic 0.17.0` (verified in the pin), which builds the paper | `.conda-explicit.txt`, `.yml` |

Recreate them:

    MM=micromamba                     # whatever `command -v micromamba` prints; 2.5.0 was used
    E=$(micromamba info | awk '/envs directories/{print $NF}')   # `micromamba info` prints yours
    PROJ=$SCRM_PROJ
    for e in scrm scfm tex; do
        $MM create -y -n $e --file envs/$e.conda-explicit.txt
    done
    # THE PIP HALF IS NOT INSTALLABLE AS IT STANDS -- prune the lines each header names first
    # (the `@ file:///home/conda/...` conda re-reports, and scrm's two `-e` editable paths):
    $E/scrm/bin/pip install -r envs/scrm.pip-freeze.txt
    $E/scfm/bin/pip install      -r envs/scfm.pip-freeze.txt
    # uce-scfm-extra needs no pruning: 43 plain `name==version` lines, straight into a target dir
    $E/scfm/bin/pip install --target $PROJ/envs/uce-scfm-extra -r envs/uce-scfm-extra.pip-freeze.txt

Four things the files say in their own headers:

- **The explicit specs name `conda.anaconda.org`.** On a host that reaches conda-forge through a
  mirror, substitute the host before creating an environment:

      sed 's|https://conda.anaconda.org/|https://<mirror-host>/|' \
          envs/$e.conda-explicit.txt > /tmp/$e.spec.txt

  **The substitution is on the HOST, not the channel path**: two channel paths are in play, not one
  — `scrm` is 89 `conda-forge` lines **and 29 `bioconda` lines** (`scfm` 41 and `tex` 43 are
  conda-forge only), and a `/conda-forge/`-only substitution leaves those 29 pointing at a path the
  public host does not serve. Each line ends in the package's md5, so no line can have changed WHICH
  package it names. The URLs themselves are **UNVERIFIED**: no URL in these specs has been resolved
  from this tree.
- **`--explicit` is the conda half only.** It records no pip package, and `scrm` has 231 pip
  lines, `scfm` 144. The `.yml` carries both halves but flattens every pip install to
  `name==version`, which is a *listing*, not an install spec: `peft==0.8.2` there is in fact a fork
  at `git+https://github.com/lauradmartens/peft.git@f00283c5`, and `scooby==0.1.0` resolves to
  nothing on PyPI. The `.pip-freeze.txt` files keep the
  real markers and are uncurated: this is the environment that produced the numbers, leftovers
  included. `torch==2.11.0+cu128` (and `uce-scfm-extra`'s `torch==2.13.0`) carries a CUDA local
  version and comes from PyTorch's own wheel index, not PyPI.
- **`envs/scrm.yml` is the environment, not the import list.** It pins what the tree imports
  AND those packages' own dependencies, so it carries names nothing here imports directly --
  `matplotlib` and `seaborn` come in under `scanpy`, and `jinja2` under `torch` -- those three
  are not removable. The rest are leftovers, and an earlier version of this paragraph got
  their cause wrong: `fastapi`, `uvicorn` and `starlette` are pulled by `mlflow-skinny`, not
  by `huggingface_hub` or `transformers` (which need them only under extras this tree does
  not install), and `tensorboard` has no unconditional requirer at all. Nothing here imports
  `mlflow`, `lightning`, `tensorboard`, `fastapi`, `uvicorn` or `starlette`, so that chain is
  removable at its root.
  Three entries the environment on disk
  carried are not in it, because nothing here imports any of them: `unitig-dm` (the predecessor
  tree -- the import lives in the count-FM embedder at `$SCRM_PROJ/countfm/`, which runs in
  `scfm` and puts its own copy on `sys.path`), `blt` (`facebookresearch/blt`, which nothing
  here imports and which is not needed to reproduce this work).
  The `.pip-freeze.txt` files still record them because they record the environment that
  produced the numbers, leftovers included. When installing from a freeze, drop those two and
  drop every
  `<pkg> @ file:///home/conda/feedstock_root/...` line (those are conda packages that pip merely
  reports; the explicit spec installs them).
- **`uce-scfm-extra` SHADOWS `scfm`, and it is the environment the UCE column was produced under.**
  `PYTHONPATH` prepends, so measured under the launcher's exact invocation: `torch 2.13.0+cu130` and
  `numpy 2.4.6` from the target directory win over `scfm`'s own `2.3.1` and `1.26.4`. A different
  torch is a different forward, so the shadowing cannot be tidied without re-embedding UCE.

Two tools sit outside all four and are not pinned here: 10x's `bamtofastq` at
`$HOME/bin/bamtofastq` (stage 2, `got_ET01` only), and the `simpleaf` / `varcall` envs, which no
`reproduce.sh` stage invokes (`model/jobs/mouse_index.sbatch`).

## Third-party code

- No third-party code is in this tree. The count-FM baseline embedder that produced every count
  foundation model column lives at `$SCRM_PROJ/countfm/`; `eval/countfm/fetch.py` fetches it from
  its publishers and records every embedder and weight file with its source, its size and its
  sha256, and `model/jobs/embedder_equiv.sbatch` measures that it reproduces the panel (bitwise
  identical).
- Nothing under `model/` is third-party code. Each module's header records the
  citations owed.
- UCE's own code is at `$SCRM_PROJ/vendor/UCE`; `measurements/embed_uce.py` calls one
  entrypoint.

## Measured costs, for planning

| stage | cost |
| --- | --- |
| corpus | ~350 GB of reads → ~141 GB of corpus for the 20 |
| embed (stage 5) | 120 artifacts, ~4h50 wall at 8–9 concurrent H100s |
| score, per dataset | 6 min – 1h40 (scales with cells × artifacts) |
| annotation, per dataset | up to ~1h50 (`pbmc_10k_v3`, 21 artifacts × 3 readers) |
| baselines | minutes per (model, dataset); UCE ~6 min per dataset |

## Rules that outrank convenience

- Submit to one partition, named by `SCRM_SLURM_PARTITION`. On a preemptible partition, the
  `--requeue` flag and the checkpoint-resume paths are what carry a job across a kill.
- **Pin the corpus, then embed** — never the reverse. `controls.corpus_facts` reads the pinned
  digest, so an unpinned corpus cannot be stamped at all.
- A pinned-digest **mismatch is not a reason to run `--write`**: it means the pipeline changed the
  data or the corpus is corrupt, and both need diagnosing.
- **A corpus launcher deletes its own reads, gated on `manifest.json`.** The manifest is written
  last, so it is the only honest completion marker; the fetch stage's completion marker is deleted
  together with the reads, because leaving it behind makes a rebuild skip the download and then fail
  on an empty directory. `$SCRM_PROJ` is a shared filesystem and the reads are much larger than the
  corpus: download, build, delete, one dataset at a time. `got_ET01` keeps its source BAM, from
  which `model/jobs/corpus.sbatch` regenerates the FASTQ without a re-download, and it keeps every
  called-cell whitelist — building against a different cell list produces a different population.
- `check.sh` is the code gate; `reproduce.sh` is the plan and the data gate above it. Neither
  replaces the other, and neither sequences the stages for you.
