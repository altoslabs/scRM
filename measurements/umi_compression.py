#!/usr/bin/env python3
"""Read-to-corpus ratio and UMI duplication per corpus, and the macros `app:pipeline` states in prose.

`pipeline_detail.tex` argues that the corpus format buys space from DUPLICATION rather than from the
species or the depth, and it argued it with nine literals: the ratio range of the four corpora that
compress best, their singleton range, their merge-rate range, and the singleton and merge rates of
the two corpora whose corpus is larger than their reads. Every one of them is a manifest field, and
every one of them moves when a corpus is rebuilt -- so none of them belongs in prose.

Reads only. The source is each corpus's own `manifest.json`, which is written LAST and therefore
only exists for a finished corpus; `measured.umi_stats` is what the assembler measured about the UMI
graph it saw, and the shard and declared-input byte counts are the two sides of the ratio.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "model"), str(REPO / "eval"), str(REPO / "target" / "py")]

from scrm_eval import registry as R  # noqa: E402

#: The appendix's sentence says "the four that compress best", so FOUR is the sentence's own
#: definition. The fragment's header prints which four they are and the ratio of the fifth.
N_BEST = 4

#: A corpus whose reads are SMALLER than the corpus built from them, which is the appendix's "the
#: two corpora that do not compress". A rule, not a count: this file refuses rather than printing
#: two of three.
FLAT_RATIO = 1.0

TEX_FRAGMENT = "compression_n"


def corpora(root: Path | None = None) -> list[tuple[str, dict, int, int]]:
    """Every finished corpus under the root, as (name, umi_stats, fastq bytes, shard bytes).

    The scope is the DISK, not `registry.datasets()`: the roles cover thirteen names and the
    appendix's sentence is about the sixteen corpora this work builds, which includes the CRISPR
    screen and the two cross-species panels that carry no role. `_permuted` corpora are the same
    reads with the labels shuffled, so counting them would double four datasets.
    """
    root = root or R.corpus_root()
    if not root.is_dir():
        # Two diagnoses: nobody named a data root, or a data root is named and the corpora sit
        # somewhere else under it. Say which.
        raise SystemExit(f"[tex] REFUSED: no corpus root at {root}, so no manifest can be read and "
                         + (R.ROOT_HINT if R.root_unset() else
                            f"the paragraph would be derived from nothing. Name the corpora with "
                            f"one of {', '.join(R._ROOT_ENV)}."))
    out = []
    for d in sorted(root.iterdir()):
        mf = d / "manifest.json"
        if not mf.is_file() or d.name.endswith("_permuted") or d.name.startswith("_"):
            continue
        m = json.loads(mf.read_text())
        fq = sum(i["bytes"] for i in m["declared"]["inputs"]
                 if "fastq" in i["path"].lower() or i["path"].endswith((".fq.gz", ".fastq.gz")))
        sh = sum((d / s["file"]).stat().st_size for s in m["shards"] if (d / s["file"]).is_file())
        if fq and sh:
            out.append((d.name, m["measured"]["umi_stats"], fq, sh))
    return out


def ranked(rows: list[tuple[str, dict, int, int]]) -> tuple[list, list, float | None]:
    """The `N_BEST` best-compressing corpora, the flat ones, and the next ratio after the best."""
    by_ratio = sorted(rows, key=lambda r: r[2] / r[3], reverse=True)
    best = by_ratio[:N_BEST]
    nxt = (by_ratio[N_BEST][2] / by_ratio[N_BEST][3]) if len(by_ratio) > N_BEST else None
    flat = [r for r in by_ratio if r[2] / r[3] < FLAT_RATIO]
    return best, flat, nxt


def _pair(vals: list[float], fmt: str) -> tuple[str, str]:
    """A range as the appendix prints it: low then high, both in the prose's own format."""
    return format(min(vals), fmt), format(max(vals), fmt)


def macros(rows: list[tuple[str, dict, int, int]]) -> str:
    """The nine numbers `app:pipeline`'s duplication paragraph states, derived from the manifests."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from results_page import TEX_NS

    if len(rows) <= N_BEST:
        raise SystemExit(f"[tex] REFUSED: {len(rows)} corpora carry a manifest and a ratio, so "
                         f"'the {N_BEST} that compress best' is not a subset of them. "
                         f"{R.ROOT_HINT if R.root_unset() else f'Root: {R.corpus_root()}'}")
    best, flat, nxt = ranked(rows)
    if len(flat) != 2:
        # The prose prints a PAIR of singleton fractions and a PAIR of merge rates for these.
        # Three of them, or one, is not a pair.
        raise SystemExit(f"[tex] REFUSED: {len(flat)} corpora have a ratio below {FLAT_RATIO} "
                         f"({[n for n, *_ in flat]}), and app:pipeline states two. Re-word the "
                         f"paragraph before deriving it.")
    ratio_lo, ratio_hi = _pair([r[2] / r[3] for r in best], ".2f")
    sing_lo, sing_hi = _pair([r[1]["singleton_fraction"] * 100 for r in best], ".1f")
    merge_lo, merge_hi = _pair([r[1]["merge_rate"] for r in best], ".3f")
    fsing_lo, fsing_hi = _pair([r[1]["singleton_fraction"] * 100 for r in flat], ".1f")
    fmerge_lo, fmerge_hi = _pair([r[1]["merge_rate"] for r in flat], ".3f")
    out = [f"% GENERATED by measurements/{Path(__file__).name} -o -- do not edit by hand.",
           f"% {len(rows)} corpora under {R.corpus_root()}; ratio = declared FASTQ bytes / shard "
           f"bytes.",
           f"% Best {N_BEST}: {', '.join(n for n, *_ in best)}"
           + (f" (the next is {nxt:.2f})" if nxt is not None else ""),
           f"% Flat (ratio < {FLAT_RATIO:g}, the corpus is the larger of the two): "
           f"{', '.join(n for n, *_ in flat)}"]
    pre = "\\newcommand{\\" + TEX_NS + "Compress"
    for stem, val in (("BestRatioLo", f"${ratio_lo}$"), ("BestRatioHi", f"${ratio_hi}$"),
                      ("BestSingletonLo", f"${sing_lo}\\%$"), ("BestSingletonHi", f"${sing_hi}\\%$"),
                      ("BestMergeLo", f"${merge_lo}$"), ("BestMergeHi", f"${merge_hi}$"),
                      ("FlatSingletonLo", f"${fsing_lo}\\%$"),
                      ("FlatSingletonHi", f"${fsing_hi}\\%$"),
                      ("FlatMergeLo", f"${fmerge_lo}$"), ("FlatMergeHi", f"${fmerge_hi}$")):
        out.append(pre + stem + "}{" + val + "}")

    # The SECOND paragraph of app:pipeline_detail states six more figures over ALL the corpora --
    # the worst ratio and which corpus it is, the median, the mean, and the pooled read and corpus
    # sizes with their ratio.
    ratios = sorted(r[2] / r[3] for r in rows)
    worst = min(rows, key=lambda r: r[2] / r[3])
    mid = (ratios[len(ratios) // 2] if len(ratios) % 2
           else (ratios[len(ratios) // 2 - 1] + ratios[len(ratios) // 2]) / 2)
    fq_tot = sum(r[2] for r in rows)
    sh_tot = sum(r[3] for r in rows)
    GB = 1000 ** 3
    for stem, val in (("N", str(len(rows))),
                      ("WorstRatio", f"${worst[2] / worst[3]:.2f}$"),
                      ("WorstName", "\\texttt{" + worst[0].replace("_", "\\_") + "}"),
                      ("MedianRatio", f"${mid:.2f}$"),
                      ("MeanRatio", f"${sum(ratios) / len(ratios):.2f}$"),
                      ("PooledReadsGB", f"${fq_tot / GB:.0f}$"),
                      ("PooledCorpusGB", f"${sh_tot / GB:.0f}$"),
                      ("PooledRatio", f"${fq_tot / sh_tot:.2f}$"),
                      ("BelowOne", str(sum(1 for r in ratios if r < 1.0)))):
        out.append(pre + stem + "}{" + val + "}")
    return "\n".join(out)


def table(rows: list[tuple[str, dict, int, int]]) -> str:
    """Every corpus's ratio and UMI statistics, for checking the paragraph against the manifests."""
    out = [f"{'corpus':44}{'ratio':>7}{'h1_deg':>8}{'singleton':>10}{'merge':>8}{'chain':>7}"]
    for n, u, fq, sh in sorted(rows, key=lambda r: r[2] / r[3], reverse=True):
        out.append(f"{n:44}{fq / sh:7.2f}{u['mean_hamming1_degree']:8.3f}"
                   f"{u['singleton_fraction']:10.3f}{u['merge_rate']:8.3f}{u['max_chain_depth']:7d}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", type=Path, metavar="DIR",
                    help=f"write {TEX_FRAGMENT}.tex here instead of printing the fragment")
    ap.add_argument("--table", action="store_true",
                    help="print every corpus's ratio and UMI statistics instead of the fragment")
    a = ap.parse_args(argv)

    rows = corpora()
    if a.table:
        print(table(rows))
        return 0
    body = macros(rows) + "\n"
    print(body, end="")
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        dst = a.out / f"{TEX_FRAGMENT}.tex"
        tmp = dst.with_suffix(".tex.tmp")
        tmp.write_text(body)
        tmp.replace(dst)
        print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
