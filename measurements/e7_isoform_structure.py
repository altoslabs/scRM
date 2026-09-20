"""Is the perturbation signal scrm sees in ISOFORM CHOICE -- polyA site, 3'UTR end, junction usage?"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scrm_eval import controls, registry, reports  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
P = registry.PROJ
REFS = registry.PROJ / "refs" / "hg38"
GENOME = REFS / "GRCh38.primary_assembly.genome.fa"
GTF = REFS / "gencode.v44.basic.annotation.gtf.gz"
DS = "crispr_a549_5k"
WORK = P / "work" / "e7_isoform"

DECODE = np.frombuffer(b"ACGTN\t\n", dtype=np.uint8)

BIN = 10_000
DOWNSTREAM_SLOP = 500
MIN_CONTIG = 40
MIN_MOL_PER_GENE_CELL = 2
MIN_CELLS_PER_GENE = 300
MIN_CELLS_PER_JUNCTION = 50

#: The bar `app:perturbation` states when it says a probe "passes $0.60$" on a target. The
#: comparison is STRICT.
TEX_PASS = 0.60

#: One macro stem per probe family, because `app:perturbation` names every family in prose. An
#: UNKNOWN family RAISES rather than being skipped.
TEX_FAMILY = {"junction": "Junction", "apa_annot": "ApaAnnot", "apa_denovo": "ApaDenovo",
              "coverage": "Coverage", "apa_perm": "Perm"}
TEX_FRAGMENT = "e7iso_n"

_G: dict = {}


def load_annotation() -> dict:
    """Protein-coding gene bodies, a binned lookup, and each gene's annotated 3' ends."""
    bodies: dict[str, tuple[str, str, int, int]] = {}
    ends: dict[str, set[int]] = defaultdict(set)
    gname = re.compile(r'gene_name "([^"]+)"')
    gtype = re.compile(r'gene_type "([^"]+)"')
    with gzip.open(GTF, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.split("\t")
            if f[2] not in ("gene", "transcript"):
                continue
            m = gname.search(f[8])
            if m is None:
                continue
            g = m.group(1)
            if f[2] == "gene":
                t = gtype.search(f[8])
                if t is not None and t.group(1) == "protein_coding":
                    bodies[g] = (f[0], f[6], int(f[3]), int(f[4]))
            else:
                ends[g].add(int(f[4]) if f[6] == "+" else int(f[3]))
    names = sorted(bodies)
    chrom = np.array([bodies[g][0] for g in names])
    fwd = np.array([bodies[g][1] == "+" for g in names])
    start = np.array([bodies[g][2] for g in names], dtype=np.int64)
    stop = np.array([bodies[g][3] for g in names], dtype=np.int64)
    index: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i in range(len(names)):
        for b in range(start[i] // BIN, (stop[i] + DOWNSTREAM_SLOP) // BIN + 1):
            index[(chrom[i], b)].append(i)
    return {"names": names, "chrom": chrom, "fwd": fwd, "start": start, "stop": stop,
            "index": dict(index), "ends": {g: sorted(ends[g]) for g in names}}


def _init(cell_ix: dict[str, int], ann: dict) -> None:
    """Build the index ONCE, in the parent, and let fork share it."""
    import mappy

    _G["al"] = mappy.Aligner(str(GENOME), preset="splice", k=14, n_threads=8)
    if not _G["al"]:
        raise RuntimeError(f"could not build a minimap2 index over {GENOME}")
    _G["cell_ix"] = cell_ix
    _G["ann"] = ann


def _assign(ch: str, s: int, e: int) -> int:
    """Gene index by maximum overlap, or -1."""
    ann = _G["ann"]
    idx, chrom, start, stop = ann["index"], ann["chrom"], ann["start"], ann["stop"]
    best, bov = -1, 0
    for b in range(s // BIN, e // BIN + 1):
        for j in idx.get((ch, b), ()):
            if chrom[j] != ch:
                continue
            ov = min(e, stop[j] + DOWNSTREAM_SLOP) - max(s, start[j])
            if ov > bov:
                best, bov = j, ov
    return best


def align_shard(path: str) -> dict:
    """Align a shard's molecules and accumulate per (cell, gene). Nothing per-alignment escapes."""
    import pyarrow as pa
    from pyarrow import ipc

    al, cell_ix, ann = _G["al"], _G["cell_ix"], _G["ann"]
    n_c, n_g = len(cell_ix), len(ann["names"])
    fwd, start, stop = ann["fwd"], ann["start"], ann["stop"]
    relsum = np.zeros((n_c, n_g), dtype=np.float32)
    cnt = np.zeros((n_c, n_g), dtype=np.int32)
    jx: list[tuple[int, int, int, int]] = []
    n_mol = n_kept = n_unmapped = n_unassigned = n_span = 0
    n_eq90 = n_le90 = n_gt180 = n_multi = 0
    with pa.memory_map(path) as src:
        rd = ipc.open_file(src)
        for b in range(rd.num_record_batches):
            tb = rd.get_batch(b)
            cids = tb.column("cell_id").to_pylist()
            toks = tb.column("tokens")
            offs = np.asarray(toks.offsets)
            flat = np.asarray(toks.values)
            for i, cid in enumerate(cids):
                ci = cell_ix.get(str(cid).split("-")[0], -1)
                if ci < 0:
                    continue
                n_mol += 1
                ln_tok = int(offs[i + 1] - offs[i])
                n_eq90 += ln_tok == 90
                n_le90 += ln_tok <= 90
                n_gt180 += ln_tok > 180
                seq = DECODE[flat[offs[i]:offs[i + 1]]].tobytes().decode("ascii")
                n_multi += "\t" in seq
                best = None
                for part in seq.split("\t"):
                    if len(part) < MIN_CONTIG:
                        continue
                    for h in al.map(part, cs=False):
                        if h.mapq < 1 or h.mlen < 0.5 * len(part):
                            continue
                        if best is None or h.mlen > best.mlen:
                            best = h
                if best is None:
                    n_unmapped += 1
                    continue
                gj = _assign(best.ctg, best.r_st, best.r_en)
                if gj < 0:
                    n_unassigned += 1
                    continue
                site = best.r_en if fwd[gj] else best.r_st
                ln = max(1, stop[gj] - start[gj])
                rel = (site - start[gj]) / ln if fwd[gj] else (stop[gj] - site) / ln
                relsum[ci, gj] += min(1.5, max(-0.5, rel))
                cnt[ci, gj] += 1
                n_kept += 1
                pos = best.r_st
                for blen, op in best.cigar:
                    if op == 3:
                        jx.append((ci, gj, pos, pos + blen))
                        n_span += 1
                        pos += blen
                    elif op in (0, 2, 7, 8):
                        pos += blen
    j = np.array(jx, dtype=np.int64).reshape(-1, 4)
    return {"relsum": relsum, "cnt": cnt, "jx": j,
            "stats": np.array([n_mol, n_kept, n_unmapped, n_unassigned, n_span,
                               n_eq90, n_le90, n_gt180, n_multi], dtype=np.int64)}


def stage_align(jobs: int, limit: int = 0) -> None:
    import multiprocessing as mp

    lab = labels()
    cells = sorted(lab)
    cell_ix = {c: i for i, c in enumerate(cells)}
    ann = load_annotation()
    shards = sorted(glob.glob(str(registry.corpus_root() / DS / "*.arrow")))
    if limit:
        shards = shards[:limit]
    print(f"{len(shards)} shards | {len(cells):,} labelled cells | {len(ann['names']):,} "
          f"protein-coding genes | {jobs} workers", flush=True)
    _init(cell_ix, ann)
    print("splice index built", flush=True)
    ctx = mp.get_context("fork")
    relsum = np.zeros((len(cells), len(ann["names"])), dtype=np.float64)
    cnt = np.zeros((len(cells), len(ann["names"])), dtype=np.int32)
    jx: list[np.ndarray] = []
    stats = np.zeros(9, dtype=np.int64)
    with ctx.Pool(jobs) as pool:
        for k, r in enumerate(pool.imap_unordered(align_shard, shards)):
            relsum += r["relsum"]
            cnt += r["cnt"]
            if len(r["jx"]):
                jx.append(r["jx"])
            stats += r["stats"]
            print(f"  {k + 1}/{len(shards)}: {stats[1]:,} placed, {stats[2]:,} unmapped, "
                  f"{stats[3]:,} unassigned, {stats[4]:,} junction obs", flush=True)
    n_mol, n_kept, n_unmapped, n_unassigned, n_span, n_eq90, n_le90, n_gt180, n_multi = (
        int(v) for v in stats)
    print(f"\n{n_mol:,} molecules of labelled cells\n"
          f"  {n_kept:,} placed on a protein-coding gene ({n_kept / max(1, n_mol):.1%})\n"
          f"  {n_unmapped:,} unmapped ({n_unmapped / max(1, n_mol):.1%})\n"
          f"  {n_unassigned:,} mapped but not in a protein-coding gene "
          f"({n_unassigned / max(1, n_mol):.1%})\n"
          f"  {n_span:,} junction observations ({n_span / max(1, n_kept):.2%} of placed molecules)\n"
          f"length: {n_eq90 / max(1, n_mol):.1%} exactly 90 bp, {n_le90 / max(1, n_mol):.1%} <= 90, "
          f"{n_gt180 / max(1, n_mol):.1%} > 180, {n_multi / max(1, n_mol):.2%} multi-contig",
          flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    dst = WORK / (f"align.smoke{limit}.npz" if limit else "align.npz")
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, relsum=relsum, cnt=cnt, cells=np.array(cells),
                            genes=np.array(ann["names"]), stats=stats,
                            jx=np.concatenate(jx) if jx else np.zeros((0, 4), dtype=np.int64))
    tmp.replace(dst)
    print(f"wrote {dst}")


def _pca(x: np.ndarray, seed: int = 0) -> np.ndarray:
    from sklearn.decomposition import PCA

    k = min(50, x.shape[1], x.shape[0] - 1)
    return PCA(n_components=k, random_state=seed).fit_transform(x)


def _impute(vals: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Fill unobserved entries with the column mean over observed cells."""
    out = np.array(vals, dtype=np.float32)
    mu = np.array([out[obs[:, j], j].mean() if obs[:, j].any() else 0.0
                   for j in range(out.shape[1])], dtype=np.float32)
    out[~obs] = np.take(mu, np.nonzero(~obs)[1])
    return out


def stage_features() -> None:
    d = np.load(WORK / "align.npz", allow_pickle=False)
    relsum, cnt = d["relsum"], d["cnt"]
    cells = [str(c) for c in d["cells"]]
    names = [str(g) for g in d["genes"]]
    ann = load_annotation()
    obs = cnt >= MIN_MOL_PER_GENE_CELL
    keep = obs.sum(axis=0) >= MIN_CELLS_PER_GENE
    print(f"{len(cells):,} cells x {len(names):,} genes; {keep.sum():,} genes with "
          f">= {MIN_MOL_PER_GENE_CELL} molecules in >= {MIN_CELLS_PER_GENE} cells", flush=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_rel = np.where(obs, relsum / np.maximum(cnt, 1), np.nan)

    mats: dict[str, np.ndarray] = {}
    mats["apa_denovo"] = _impute(mean_rel[:, keep], obs[:, keep])
    mats["coverage"] = np.log1p(cnt[:, keep]).astype(np.float32)

    gi = {g: i for i, g in enumerate(ann["names"])}
    lo = np.full(len(names), np.nan)
    hi = np.full(len(names), np.nan)
    absmean = np.full_like(mean_rel, np.nan)
    for j, g in enumerate(names):
        k = gi.get(g)
        if k is None:
            continue
        ln = max(1, ann["stop"][k] - ann["start"][k])
        absmean[:, j] = (ann["start"][k] + mean_rel[:, j] * ln if ann["fwd"][k]
                         else ann["stop"][k] - mean_rel[:, j] * ln)
        e = ann["ends"].get(g, [])
        if len(e) >= 2:
            lo[j], hi[j] = (e[0], e[-1]) if ann["fwd"][k] else (e[-1], e[0])
    span = hi - lo
    ka = keep & np.isfinite(span) & (np.abs(span) >= 200)
    print(f"  {ka.sum():,} of those have an annotated 3'-end span >= 200 bp", flush=True)
    with np.errstate(invalid="ignore"):
        rel_annot = np.clip((absmean[:, ka] - lo[ka]) / span[ka], 0.0, 1.0)
    mats["apa_annot"] = _impute(rel_annot, obs[:, ka] & np.isfinite(rel_annot))

    rng = np.random.default_rng(20260826)
    perm = np.array(mats["apa_denovo"])
    for j in range(perm.shape[1]):
        perm[:, j] = perm[rng.permutation(len(cells)), j]
    mats["apa_perm"] = perm

    jx = d["jx"]
    if len(jx):
        key = np.stack([jx[:, 1], jx[:, 2], jx[:, 3]], axis=1)
        uj, jj = np.unique(key, axis=0, return_inverse=True)
        jj = jj.ravel()
        jci = jx[:, 0].astype(np.int64)
        pairs = np.unique(jj * np.int64(len(cells)) + jci)
        per_j = np.bincount(pairs // len(cells), minlength=len(uj))
        kj = per_j >= MIN_CELLS_PER_JUNCTION
        print(f"  {len(uj):,} distinct junctions, {int(kj.sum()):,} seen in "
              f">= {MIN_CELLS_PER_JUNCTION} cells", flush=True)
        if kj.sum() >= 2:
            remap = np.full(len(uj), -1, dtype=np.int64)
            remap[kj] = np.arange(int(kj.sum()))
            sel = remap[jj] >= 0
            jm = np.zeros((len(cells), int(kj.sum())), dtype=np.float32)
            np.add.at(jm, (jci[sel], remap[jj[sel]]), 1.0)
            mats["junction"] = (jm / np.maximum(jm.sum(axis=1, keepdims=True), 1)).astype(np.float32)

    red = {k: _pca(v) for k, v in mats.items()}
    for k, v in mats.items():
        print(f"  {k:12s} {v.shape[0]} cells x {v.shape[1]:,} features -> PCA-{red[k].shape[1]}",
              flush=True)
    tmp = WORK / "features.npz.tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, cells=np.array(cells),
                            **{f"X_{k}": v for k, v in red.items()},
                            **{f"n_{k}": np.array(v.shape[1]) for k, v in mats.items()})
    tmp.replace(WORK / "features.npz")
    print(f"wrote {WORK / 'features.npz'}")


def labels() -> dict[str, set[str]]:
    """cell -> the targets its guides hit. ONE definition, imported from the scorer."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from e7_vs_control import labels as _labels

    return _labels()[0]


def stage_score(out_path: Path) -> None:
    sys.path.insert(0, str(REPO / "eval"))

    d = np.load(WORK / "features.npz", allow_pickle=False)
    cells = [str(c) for c in d["cells"]]
    vsc = json.loads((P / "eval_reports" / f"e7_vs_control.{DS}.json").read_text())
    targets = vsc["targets"]
    ref = {m: {t: reports.per_target(vsc, vsc["results"][m], t) for t in targets}
           for m in vsc["results"]}

    lab = labels()
    missing = sum(1 for c in lab if c not in set(cells))
    if missing:
        print(f"WARNING: {missing:,} labelled cells have no aligned molecules and are dropped; "
              f"the population is then NOT the published one", flush=True)
    Y = np.zeros((len(cells), len(targets)), dtype=bool)
    for j, t in enumerate(targets):
        Y[:, j] = [t in lab.get(c, ()) for c in cells]
    has = np.array([c in lab for c in cells])
    print(f"{int(has.sum()):,} of {len(cells):,} cells carry an effective-target label\n", flush=True)

    keys = sorted(k[2:] for k in d.files if k.startswith("X_"))
    fms = list(controls.PANEL)
    rep = {"dataset": DS, "targets": targets, "n_cells": int(has.sum()), "results": {},
           "protocol": "e7_vs_control.target_vs_ntc (ridge AUROC vs the Non-Targeting controls, "
                       "5 folds x 3 seeds, measured null) "
                       "on PCA-50 of each isoform-structure matrix; molecules placed on hg38 with "
                       "minimap2 splice (mappy 2.31), genes and 3' ends from GENCODE v44 basic",
           "reference_note": "hg38/GENCODE are read on the ANALYSIS side only, to interpret a result; "
                             "nothing here reaches the corpus, the model or a training input"}
    al = np.load(WORK / "align.npz", allow_pickle=False)
    s = [int(v) for v in al["stats"]]
    rep["corpus"] = {"n_molecules": s[0], "placed": s[1], "unmapped": s[2], "unassigned": s[3],
                     "junction_obs": s[4], "len_eq_90": s[5], "len_le_90": s[6],
                     "len_gt_180": s[7], "multi_contig": s[8]}
    from e7_vs_control import target_vs_ntc

    _t = json.loads((P / "raw" / DS / "labels_target.json").read_text())
    ntc = [c.split("-")[0] for c, v in _t.items() if v == "Non-Targeting"]
    print(f"negative class = {len(ntc)} Non-Targeting cells (E7's own control)\n", flush=True)
    for k in keys:
        res = target_vs_ntc(np.asarray(d[f"X_{k}"], dtype=np.float32), lab, cells, targets,
                            ntc, null_reps=20)
        per = [res.get(t, {}).get("auroc") for t in targets]
        ok = [v for v in per if v is not None]
        macro = float(np.mean(ok)) if ok else float("nan")
        rep["results"][k] = {"macro": macro, "n_features": int(d[f"n_{k}"]),
                             "per_target": per, "per_target_by_name": res}
        print(f"{k:12s} macro AUROC = {macro:.4f}   ({int(d[f'n_{k}']):,} features)", flush=True)

    print(f"\n{'target':10s} " + " ".join(f"{k[:10]:>10s}" for k in keys) +
          f" {'scrm':>7s} {'cntPCA':>7s} {'bestFM':>7s}", flush=True)
    rep["reference"] = {}
    for j, t in enumerate(targets):
        _ours = registry.reported_arm(ref)
        if _ours is None:
            continue
        sb = ref[_ours][t]
        cs = ref["count_scanpy"][t]
        bf = max(ref[m][t] for m in fms if m in ref)
        rep["reference"][t] = {"scrm": sb, "count_pca": cs, "best_fm": bf}
        row = " ".join(f"{rep['results'][k]['per_target'][j]:10.3f}" for k in keys)
        print(f"{t:10s} {row} {sb:7.3f} {cs:7.3f} {bf:7.3f}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rep, indent=2))
    tmp.replace(out_path)
    print(f"\nwrote {out_path}")


def macros(rep: dict, src: Path) -> str:
    """Every isoform probe's macro AUROC and pass count, which `app:perturbation` states in prose.

    The appendix carried all five as literals matching a derivation run of 2026-09-02 digit for
    digit. `e7_vs_control.py` was then committed against three times (the last 2026-09-06) and this
    report re-scored on 2026-09-19, and nobody retyped the paragraph: it named a junction macro of
    0.5670 passing on five targets against a report that says 0.5994 passing on twelve. The feature
    matrices did not move -- `n_features` is identical in both -- so the SCORE moved and nothing was
    re-measured, which is exactly the drift a literal cannot show you. WHICH of those commits moved
    it is UNVERIFIED and does not need to be known for the number to be derived instead of typed.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from results_page import TEX_NS

    unknown = sorted(set(rep["results"]) - set(TEX_FAMILY))
    if unknown:
        raise KeyError(f"{src} scores {unknown}, which TEX_FAMILY does not name, so the appendix "
                       f"cannot state it. Name the family here, or retire the probe.")
    out = [f"% GENERATED by measurements/{Path(__file__).name} --write-tex -- do not edit by hand.",
           f"% {src.name}: {len(rep['targets'])} targets, {rep['n_cells']:,} labelled cells. "
           f"A family 'passes' a target when its AUROC > {TEX_PASS:.2f}."]
    pre = "\\newcommand{\\" + TEX_NS + "Eseven"
    for k in sorted(rep["results"]):
        r = rep["results"][k]
        per = [v for v in r["per_target"] if v is not None]
        if not per:
            raise SystemExit(f"[tex] REFUSED: {k} scored no target in {src}, so its macro is not a "
                             f"number and the appendix would state one. Re-run --stage score.")
        stem = "Iso" + TEX_FAMILY[k]
        out.append(pre + stem + "}{$" + f"{r['macro']:.4f}" + "$}")
        out.append(pre + stem + "Pass}{" + str(sum(v > TEX_PASS for v in per)) + "}")
    return "\n".join(out)


def stage_tex(report: Path, out_dir: Path) -> int:
    """Write the fragment the appendix inputs, from the report on disk and nothing else."""
    if not report.is_file():
        raise SystemExit(f"[tex] REFUSED: {report} does not exist, so there is nothing to derive. "
                         f"Run --stage score first; a fragment of guesses is worse than no paper.")
    body = macros(json.loads(report.read_text()), report) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{TEX_FRAGMENT}.tex"
    tmp = dst.with_suffix(".tex.tmp")
    tmp.write_text(body)
    tmp.replace(dst)
    print(body, end="")
    print(f"wrote {dst}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("align", "features", "score"))
    ap.add_argument("--jobs", type=int, default=max(1, len(os.sched_getaffinity(0)) - 1))
    ap.add_argument("--limit", type=int, default=0, help="align only the first N shards (smoke test)")
    ap.add_argument("-o", "--out", type=Path,
                    default=P / "eval_reports" / f"e7_isoform_structure.{DS}.json")
    ap.add_argument("--write-tex", type=Path, metavar="DIR",
                    help=f"derive {TEX_FRAGMENT}.tex from the report at --out into DIR and exit")
    a = ap.parse_args()
    if bool(a.stage) == bool(a.write_tex):
        # One flag, not a stage plus a disposition: the three stages MEASURE and --write-tex only
        # reads the report they wrote.
        ap.error("pass exactly one of --stage {align,features,score} or --write-tex DIR")
    if a.write_tex:
        return stage_tex(a.out, a.write_tex)
    if a.stage == "align":
        stage_align(a.jobs, a.limit)
    elif a.stage == "features":
        stage_features()
    else:
        stage_score(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
