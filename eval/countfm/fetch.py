#!/usr/bin/env python3
"""Fetch the count-FM baselines' code and weights from their publishers.

**This repository hosts neither.** Each baseline's code is its authors' and each checkpoint is
theirs to distribute, so nothing here is redistributed: this script pulls them at evaluation
time, into a directory you name, and prints what it got and from where.

    python eval/countfm/fetch.py --weights-root $SCRM_PROJ/countfm/scfm_weights
    python eval/countfm/fetch.py --weights-root <dir> --only geneformer --dry-run

Run it with the interpreter that will do the embedding -- the models' dependencies are not
this package's, so they live in their own environment (`envs/scfm.yml`), and
`model/jobs/baselines_matrix.sbatch` finds it through `SCRM_PY_SCFM`.

WHERE EACH ARTIFACT COMES FROM, and which of it is CODE and which is WEIGHTS. The two are not
always the same publisher, which is why they are recorded separately: what runs and what is loaded
can be released by different people, from different places, under different terms that this file
does not restate.

| model             | code, i.e. what runs              | weights, i.e. what is loaded             |
| ----------------- | --------------------------------- | ---------------------------------------- |
| `scgpt`           | `scgpt==0.2.4`, PyPI, imported    | HF `wanglab/scGPT-human` (whole repo)    |
| `geneformer`      | none: the protocol is replicated  | HF `ctheodoris/Geneformer`, 2 patterns   |
|                   | on `transformers` in the embedder |                                          |
| `genejepa`        | GitHub `BiostateAI/GeneJEPA`,     | HF `elonlit/GeneJEPA`: ckpt + gene       |
|                   | cloned at a pinned commit and     | metadata + global stats. A DIFFERENT     |
|                   | imported (`genejepa_loader.py`)   | publisher from the code -- see below.    |
| `scvi_pretrained` | `scvi-tools`, PyPI, imported      | HF `scvi-tools/tabula-sapiens-lung-scvi` |
| `uce`             | fetched separately, then run by   | UCE's own release                       |
|                   | `measurements/uce_*.py`           |                                          |

GeneJEPA: the CODE is cloned from GitHub under the `BiostateAI` organisation; the WEIGHTS come
from the HuggingFace repo `elonlit/GeneJEPA`, whose model card states the repo holds
"artifacts only (no training code)". Code and weights are therefore different publishers, and the
table above keeps them in separate columns for that reason.

scGPT: this script fetches the weights from HuggingFace `wanglab/scGPT-human`. Whether those
bytes are identical to the Google-Drive `scGPT_human` bundle the authors also publish is
**UNVERIFIED** -- it is not established by anything in this tree. What IS pinned is the bytes:
`DIGESTS` below records the sha256 of every file the panel was produced from, and both `--check`
and a fetch verify them, so a stranger can tell whether they have the same checkpoint.

Only what the embedder reads is fetched, not whole repositories where a pattern can say so, so
this is tens of gigabytes smaller than a mirror. `--check` verifies an existing tree without downloading: every
FILE the embedder opens, by size and sha256, plus GeneJEPA's clone at its pinned commit.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import genejepa_loader  # noqa: E402  (same directory; imports nothing third-party)

#: What each model needs, as (HuggingFace repo, [allow_patterns], local directory). The patterns
#: are `huggingface_hub` globs, relative to the REPO; `None` means the whole repo. What the
#: embedder then OPENS is declared in `embed_count_fm.REQUIRED_FILES`, and `place()` below
#: reconciles the two.
#:
#: The V2-104M protocol needs the `geneformer/*_gc104M.pkl` dictionaries. The gc30M model's
#: `gene_dictionaries_30m` are a different token vocabulary with different gene medians.
WEIGHTS: dict[str, tuple[str, list[str] | None, str]] = {
    "scgpt": ("wanglab/scGPT-human", None, "scgpt_human"),
    "geneformer": (
        "ctheodoris/Geneformer",
        ["Geneformer-V2-104M/*", "geneformer/*gc104M.pkl"],
        "geneformer",
    ),
    "genejepa": ("elonlit/GeneJEPA", None, "genejepa"),
    "scvi_pretrained": ("scvi-tools/tabula-sapiens-lung-scvi", None, "scvi_ts_lung"),
}

#: The authors' code, which is imported rather than vendored. Versions are the ones the embedding
#: protocols in `embed_count_fm.py` were written against.
CODE: dict[str, list[str]] = {
    "scgpt": ["scgpt==0.2.4"],
    # Geneformer needs no package: `embed_count_fm.py` replicates its published protocol --
    # rank-value tokenization, CLS embedding from the second-to-last layer, EmbExtractor's
    # defaults -- on `transformers`, which loads the checkpoint directly.
    "geneformer": ["transformers>=4.40"],
    "scvi_pretrained": ["scvi-tools>=1.4"],
    # Their code is cloned, not pip-installed (GENEJEPA_SRC below), and is used UNMODIFIED, so what
    # it imports has to be installed: models.py imports ema_pytorch even on the inference path.
    "genejepa": ["ema_pytorch", "pyarrow"],
}


#: GeneJEPA publishes its CHECKPOINT on HuggingFace (`elonlit/GeneJEPA`: the ckpt, the gene
#: metadata and the global stats, and nothing else) and its CODE on GitHub, under a different
#: organisation. This clones the code. The commit is pinned because the checkpoint pickles
#: config dataclasses, which must match the code that wrote them. Read from `genejepa_loader`,
#: which is what imports the clone.
GENEJEPA_SRC = (genejepa_loader.UPSTREAM, genejepa_loader.UPSTREAM_COMMIT, genejepa_loader.SRC_DIR)

#: The HuggingFace revision each snapshot came from, where this tree can say. Only GeneJEPA's is
#: established, from the `huggingface_hub` download sidecar. The other three resolve to the repo's
#: default branch, and it is the DIGESTS below that pin their bytes.
REVISIONS: dict[str, str] = {
    "genejepa": "b51cac7a0f792d44853fa503f5ecd4bc89b4b348",
}

#: (size in bytes, sha256) of every file the embedder opens, as MEASURED on the tree that
#: produced the published panel. Keys are `--weights-root`-relative, and the set of keys is
#: checked against the embedder's own `REQUIRED_FILES`.
#:
#: A MISMATCH IS NOT A REASON TO EDIT THIS TABLE. It means the bytes you have are not the bytes
#: the reported numbers came from -- a truncated download, or a publisher who has moved the
#: checkpoint since. Diagnose which.
#:
#: For the two LFS files in the GeneJEPA snapshot these sha256 are the etags HuggingFace records
#: (HF stores an LFS object's sha256 as its etag), so those two are pinned against the publisher.
DIGESTS: dict[str, tuple[int, str]] = {
    "scgpt_human/args.json": (1300, "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"),
    "scgpt_human/best_model.pt": (205385258, "6cb5d451ab5c4b33eb673adbe4fddc61d2389df1b89b7651a9fe2e557572b922"),
    "scgpt_human/vocab.json": (1317639, "acca93d114ca62c3f0f50debbd23e8c87f0714f4737764454f6b2b13f2e8580f"),
    "geneformer/Geneformer-V2-104M/config.json": (
        590, "467d4492f0dd53b4d60afffe20812db484ca1cf9fdbeb6a6e060e93564f70859"),
    "geneformer/Geneformer-V2-104M/model.safetensors": (
        417571156, "fff5cba29ddd8792991fa77b4872246fbe548a178cebda3775cdc72b67780e7f"),
    "geneformer/dictionaries/ensembl_mapping_dict_gc104M.pkl": (
        3957652, "0819bcbd869cfa14279449b037eb9ed1d09a91310e77bd1a19d927465030e95c"),
    "geneformer/dictionaries/gene_median_dictionary_gc104M.pkl": (
        1512661, "a51c53f6a771d64508dfaf61529df70e394c53bd20856926117ae5d641a24bf5"),
    "geneformer/dictionaries/gene_name_id_dict_gc104M.pkl": (
        1660882, "fabfa0c2f49c598c59ae432a32c3499a5908c033756c663b5e0cddf58deea8e1"),
    "geneformer/dictionaries/token_dictionary_gc104M.pkl": (
        425590, "67c445f4385127adfc48dcc072320cd65d6822829bf27dd38070e6e787bc597f"),
    "genejepa/genejepa-epoch=49.ckpt": (
        3337799886, "5db5c5750aeecc09955fefcaf143b349d9d8e1e877fe4a1073400ba66ef56f05"),
    "genejepa/gene_metadata.parquet": (
        1326799, "d6104d1ca570d94832be0d27ffe8b7e4e54cfd366c9080239219f9c406eb751c"),
    "genejepa/global_stats.json": (55, "083f7aa484e9751f1b57e9285058f3895b1622b5b17e74255d61383dfb954f72"),
    "scvi_ts_lung/model.pt": (8763014, "c00d77db85f726f718a8fb615faafd16c169ba2586d385b3cfa5037ba2a13923"),
}


def embedder():
    """The embedder module, which is where the paths and filenames are DECLARED. Imported lazily
    because it pulls in numpy and scipy, which a `--dry-run` has no use for."""
    from importlib import import_module

    return import_module("embed_count_fm")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd: list[str], *, dry: bool) -> None:
    print("  $ " + " ".join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, check=True)


def fetch_weights(model: str, root: Path, *, dry: bool) -> None:
    """Through `huggingface_hub`'s Python API, not its CLI: the CLI's entry-point name has
    changed across releases and it is not always on PATH even where the library is importable,
    which made a code-only fetch fail on a tool it did not need."""
    repo, patterns, dest = WEIGHTS[model]
    out = root / dest
    print(f"\n== {model}: {repo} -> {out}")
    rev = REVISIONS.get(model)
    if dry:
        print(f"  snapshot_download({repo!r}, allow_patterns={patterns!r}, revision={rev!r})")
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("huggingface_hub is not importable: pip install huggingface-hub") from None
    snapshot_download(
        repo_id=repo,
        local_dir=str(out),
        allow_patterns=patterns,
        # PINNED WHERE THIS TREE CAN SAY WHICH REVISION THE PANEL USED, and the repo's default
        # branch otherwise.
        revision=rev,
    )
    print(f"  got {patterns or 'everything'}" + (f" at revision {rev[:12]}" if rev else ""))


def place(model: str, root: Path, *, dry: bool) -> int:
    """Move what was fetched to where the embedder opens it. Returns the number it could not.

    `snapshot_download` writes REPO-relative paths, and those are not always the paths the
    embedder reads: the gc104M dictionaries sit at `geneformer/<name>.pkl` in the repo and are
    read from `<root>/geneformer/dictionaries/<name>.pkl`. The EMBEDDER's layout is the contract
    -- it is the layout the published panel was produced from, and the one `DIGESTS` pins -- so
    the fetch adapts to the repo, matching by basename, and prints every move it makes.
    """
    if dry:
        return 0
    dest = root / WEIGHTS[model][2]
    unplaced = 0
    for rel in embedder().REQUIRED_FILES[model]:
        want = root / rel
        if want.is_file():
            continue
        found = [q for q in dest.rglob(want.name) if q.is_file()]
        if len(found) != 1:
            # Nothing to do, or a choice this script must not make silently. `check()` reports
            # the file as MISSING either way.
            unplaced += 1
            if len(found) > 1:
                print(f"  AMBIGUOUS {want.name}: {[str(q) for q in found]} -- not moving any")
            continue
        want.parent.mkdir(parents=True, exist_ok=True)
        found[0].replace(want)
        print(f"  placed {found[0].relative_to(root)} -> {rel}")
    return unplaced


def required_files(models: list[str] | None = None) -> list[tuple[str, str]]:
    """(model, `--weights-root`-relative path) for every file an embedding opens.

    Read from the embedder's own `REQUIRED_FILES` and from `genejepa_loader.source_files()`, so
    what this verifies is what those two actually open. A list typed out here would be a second
    declaration, and the one that drifts is always the one nobody runs.
    """
    ecf = embedder()
    chosen = sorted(set(models or ecf.REQUIRED_FILES) & set(ecf.REQUIRED_FILES))
    out = [(m, rel) for m in chosen for rel in ecf.REQUIRED_FILES[m]]
    if "genejepa" in chosen:
        src = genejepa_loader.SRC_DIR
        out += [("genejepa", f"{src}/{rel}") for rel in genejepa_loader.source_files()]
    return out


def _check_clone(root: Path) -> int:
    """GeneJEPA's clone must be AT the pinned commit, not merely present."""
    url, commit, dest = GENEJEPA_SRC
    src = root / dest
    if not (src / ".git").is_dir():
        print(f"  NO CLONE     {dest}: no .git, so its revision cannot be established ({url})")
        return 1
    got = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True)
    head = got.stdout.strip()
    if got.returncode != 0 or head != commit:
        print(f"  WRONG REV    {dest} is at {head or '(unreadable)'}, pinned at {commit}")
        print("               The checkpoint pickles config dataclasses, which must match the code")
        print("               that wrote them, so this is not a version preference.")
        return 1
    print(f"  ok           {dest} at {commit[:12]}  ({url})")
    return 0


def check(root: Path, models: list[str] | None = None) -> int:
    """Verify an existing tree FILE BY FILE. Returns the number of problems.

    THE FIRST VERSION OF THIS COULD NOT FAIL. It was `p.exists()` over five model DIRECTORIES,
    and `snapshot_download(local_dir=...)` creates the directory before it downloads anything --
    so an interrupted fetch still printed "every path the embedder reads is present". It also
    never looked at the GeneJEPA clone, without which `genejepa_loader` raises at embed time,
    after the allocation is spent. A check that cannot fail is worse than no check, because it
    reads as coverage.
    """
    src_prefix = genejepa_loader.SRC_DIR + "/"
    wanted = required_files(models)
    problems = 0
    total = 0
    for model, rel in wanted:
        path = root / rel
        if not path.is_file():
            print(f"  MISSING      {rel}   ({model})")
            problems += 1
            continue
        size = path.stat().st_size
        total += size
        pinned = DIGESTS.get(rel)
        if pinned is None:
            if rel.startswith(src_prefix):
                print(f"  ok           {rel}  {size:,} B  (pinned by the commit, not a digest)")
            else:
                print(f"  UNPINNED     {rel}  {size:,} B: no entry in DIGESTS, so these bytes are")
                print("               not established to be the ones the panel was produced from")
                problems += 1
            continue
        want_size, want_sha = pinned
        if size != want_size:
            print(f"  WRONG SIZE   {rel}  {size:,} B, pinned at {want_size:,} B -- an interrupted")
            print("               download, or a different release of the same file")
            problems += 1
            continue
        got = sha256(path)
        if got != want_sha:
            print(f"  WRONG BYTES  {rel}  sha256 {got}")
            print(f"               pinned           {want_sha}")
            problems += 1
            continue
        print(f"  ok           {rel}  {size:,} B  sha256 {got[:16]}")
    if any(rel.startswith(src_prefix) for _, rel in wanted):
        problems += _check_clone(root)
    print(f"\n  {total / 2**30:.2f} GiB read, {len(wanted)} files")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights-root", type=Path, required=True,
                    help="where to put them; pass the same path to embed_count_fm.py --weights-root")
    ap.add_argument("--only", action="append", choices=sorted(WEIGHTS), help="one model; repeatable")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and fetch nothing")
    ap.add_argument("--check", action="store_true", help="verify an existing tree, download nothing")
    ap.add_argument("--skip-code", action="store_true", help="weights only, do not pip install")
    args = ap.parse_args()

    models = args.only or sorted(WEIGHTS)

    if args.check:
        problems = check(args.weights_root, models)
        if problems:
            print(f"\n{problems} problem(s). This tree cannot produce the published numbers as it")
            print("stands; fetch again, or diagnose which artifact moved -- do NOT edit DIGESTS.")
            return 1
        print("\nevery file the embedder opens is present, the pinned size, and the pinned bytes")
        return 0

    args.weights_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_code:
        pkgs = [p for m in models for p in CODE.get(m, [])]
        if pkgs:
            print("== authors' code (imported, not vendored)")
            run([sys.executable, "-m", "pip", "install", *pkgs], dry=args.dry_run)
    for m in models:
        fetch_weights(m, args.weights_root, dry=args.dry_run)
        place(m, args.weights_root, dry=args.dry_run)
        if m == "genejepa":
            url, commit, dest = GENEJEPA_SRC
            out = args.weights_root / dest
            print(f"\n== genejepa code: {url} @ {commit[:12]} -> {out}")
            if not out.exists():
                run(["git", "clone", "--quiet", url, str(out)], dry=args.dry_run)
            run(["git", "-C", str(out), "checkout", "--quiet", commit], dry=args.dry_run)

    if not args.dry_run:
        # Verified here rather than left to the next reader.
        print("\n== verifying what was fetched")
        if check(args.weights_root, models):
            print("\nthe fetch did not produce the pinned bytes -- see the lines above")
            return 1

    print("\nGeneJEPA's code is cloned from GitHub at a pinned commit and imported from there;")
    print("nothing is copied into this tree.")
    print(f"\nVerify with:  python {Path(__file__).name} --weights-root {args.weights_root} --check")
    print("Terms for each are recorded in this file's header. UCE is fetched separately;")
    print("see measurements/uce_add_species.py and model/jobs/uce_matrix.sbatch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
