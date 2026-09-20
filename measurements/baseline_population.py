"""Which count-side baseline artifacts describe a population the corpus no longer has."""
from __future__ import annotations

import argparse
import functools
import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from scrm_eval import artifact, controls, registry, targets  # noqa: E402

FILTERS_CELLS = {
    "uce": "vendor/UCE/data_proc/data_utils.py: sc.pp.filter_cells(ad, min_genes=25)",
}

READS_THE_CORPUS = ("kmer_bag",)

KNOWN_FILTERED = {
    ("2500_wistar_rat_pbmcs_singleplex_3p_gem_x", "uce"): 56,
}


def _handed(dataset: str) -> set[str]:
    """The bare barcodes in `fm/cells_<ds>` -- the matrix the panel models were actually handed."""
    p = registry.PROJ / "fm" / f"cells_{dataset}" / "barcodes.tsv.gz"
    if not p.exists():
        return set()
    with gzip.open(p, "rt") as fh:
        return {line.strip().split("-")[0] for line in fh if line.strip()}


@functools.lru_cache(maxsize=None)
def owed(dataset: str) -> frozenset[str]:
    """The cells a count-side producer OWES: in the corpus AND in the REGISTERED count matrix."""
    return frozenset(owed_ordered(dataset))


@functools.lru_cache(maxsize=None)
def owed_ordered(dataset: str) -> tuple[str, ...]:
    """`owed`, IN CORPUS ORDER -- which is the order a complete count-side artifact must be written in."""
    _, _, cells = controls.corpus_facts(dataset)
    ad = targets.load_counts(dataset)
    obs = {str(o).split("-")[0] for o in ad.obs_names}
    return tuple(str(c) for c in cells if str(c) in obs)


def matrix_dir(dataset: str) -> tuple[str, str]:
    """`(verdict, detail)` for `fm/cells_<ds>` itself, the input every count model was handed."""
    handed = _handed(dataset)
    if not handed:
        return "ABSENT", f"no fm/cells_{dataset}/barcodes.tsv.gz"
    want = owed(dataset)
    if handed == want:
        return "OK", f"{len(handed)} cells, matches the matrix-backed corpus exactly"
    return "STALE", (f"handed {len(handed)} cells but the corpus and matrix now agree on "
                     f"{len(want)} ({len(want - handed)} missing, {len(handed - want)} extra)")


def check(dataset: str, producer: str) -> tuple[str, str]:
    """`(verdict, detail)` for one baseline artifact. Verdict is OK, STALE, WARN or ABSENT."""
    art = registry.PROJ / "eval_artifacts" / f"{dataset}.{producer}.arrow"
    if not art.exists():
        return "ABSENT", "no artifact"
    _, _, cells = controls.corpus_facts(dataset)
    corpus = {str(c) for c in cells}
    try:
        stamp = artifact.read_stamp(art)
    except Exception as e:
        return "STALE", f"unreadable: {type(e).__name__}: {e}"
    want_seq = (tuple(str(c) for c in cells) if producer in READS_THE_CORPUS
                else owed_ordered(dataset))
    universe = "the corpus" if producer in READS_THE_CORPUS else "the count matrix"
    if stamp.cell_order_sha256 == artifact.cell_order_digest(want_seq):
        return "OK", (f"{stamp.n_cells} cells in corpus order, covering all {len(want_seq)} "
                      f"{universe} has")
    try:
        emb = artifact.read(art)
    except Exception as e:
        return "STALE", f"unreadable: {type(e).__name__}: {e}"
    have = {str(c).split("-")[0] for c in emb.cell_ids.tolist()}
    want = frozenset(want_seq)
    if not want:
        return "WARN", f"no registered count matrix for {dataset} to compare against"
    missing = want - have
    if not missing:
        return "WARN", (f"{len(have)} cells, the full matrix-backed corpus, but NOT in corpus order "
                        f"-- `cell_order_sha256` disagrees, so row i is not the same cell as in the "
                        f"other artifacts of this dataset")
    frac = len(want & have) / len(want)
    detail = (f"{len(missing)} of {len(want)} cells {universe} has are missing "
              f"({frac:.2%} covered); corpus {len(corpus)}, artifact {len(have)}")
    if producer in FILTERS_CELLS:
        known = KNOWN_FILTERED.get((dataset, producer))
        if known == len(missing):
            return "WARN", (f"{detail} -- the declared drop for this pair, "
                            f"{FILTERS_CELLS[producer]}")
        if known is None:
            return "STALE", (f"{detail} -- {producer} does drop cells by its own rule "
                             f"({FILTERS_CELLS[producer]}) but this pair has no MEASURED drop "
                             f"declared in KNOWN_FILTERED, so the gap is unexplained")
        return "STALE", (f"{detail} -- {producer}'s declared drop for this pair is {known} cells, "
                         f"not {len(missing)}. Either the artifact is stale or the drop has moved, "
                         f"and a moved drop is a re-measurement rather than a new default")
    return "STALE", detail


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", help="one dataset; default every registered dataset")
    p.add_argument("--producer", help="one producer; default every count-side baseline")
    p.add_argument("--check", action="store_true",
                   help="exit non-zero if any pair is STALE, printing how to rebuild it")
    p.add_argument("-q", "--quiet", action="store_true", help="print only WARN and STALE")
    a = p.parse_args(argv)

    dsets = [a.dataset] if a.dataset else [d[0] for d in registry.DATASETS]
    prods = [a.producer] if a.producer else list(controls.BUILT_PRODUCERS) + list(controls.PANEL)
    stale, looked, stale_dirs = [], 0, []
    for ds in dsets:
        dverdict, ddetail = matrix_dir(ds)
        if dverdict != "ABSENT" and (dverdict != "OK" or not a.quiet):
            print(f"  {dverdict:5s} {ds:52s} {'fm/cells_':16s} {ddetail}")
        if dverdict == "STALE":
            stale_dirs.append(ds)
        for prod in prods:
            verdict, detail = check(ds, prod)
            if verdict == "ABSENT":
                continue
            looked += 1
            if verdict == "STALE":
                stale.append((ds, prod))
            if not a.quiet or verdict != "OK":
                print(f"  {verdict:5s} {ds:52s} {prod:16s} {detail}")
    print(f"\nchecked {looked} (dataset, producer) baseline artifacts across {len(dsets)} datasets; "
          f"{len(stale)} stale, {len(stale_dirs)} stale input matrices")
    if stale_dirs:
        print("\nthese input matrices no longer match their corpus. Rebuild them BEFORE the "
              "artifacts below -- every producer of the dataset reads them:")
        for ds in stale_dirs:
            print(f"    mv $PROJ/fm/cells_{ds}{{,.stale}}")
        if not stale:
            print("  ...but every artifact of these datasets covers the full matrix-backed corpus, "
                  "so no published number is on the wrong cells and this is not a failure.")
    if stale and a.check:
        print("\nrebuild each of these -- move the cached inputs aside first, or every existence "
              "guard in the launcher will skip them again:")
        for ds, prod in stale:
            print(f"  {ds}.{prod}:")
            if prod == "uce":
                print(f"    mv $PROJ/fm/uce/npz/{ds}.uce.npz{{,.stale}}")
                print(f"    ONE_DS={ds} sbatch --array=0 model/jobs/uce_matrix.sbatch")
            else:
                print(f"    mv $PROJ/fm/{ds}.{prod}.npz{{,.stale}}")
                print(f"    ONE_DS={ds} sbatch model/jobs/baselines_matrix.sbatch")
            print(f"    mv $PROJ/eval_artifacts/{ds}.{prod}.arrow{{,.stale}}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
