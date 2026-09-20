"""UCE's evaluation, on UCE's data, with UCE's models -- and checked against UCE's own numbers."""

from __future__ import annotations

import argparse
import functools
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "eval"), str(REPO / "model"), str(REPO / "measurements")]

from scrm_eval import registry as _R  # noqa: E402

THEIR_NB = _R.PROJ / "vendor" / "uce_reproduce"
THEIR_DATA = _R.PROJ / "fm" / "uce" / "export_data"

TOL = {"b_logreg": 1e-3, "c_cosine": 1e-3, "a_scib": 5e-2}


def _nb(name: str) -> dict:
    p = THEIR_NB / name
    if not p.exists():
        raise SystemExit(f"their notebook is missing: {p} (clone of Rosen et al.'s repo)")
    return json.loads(p.read_text())


def _outputs(cell: dict) -> str:
    """Every non-stderr output of one cell, as text."""
    parts = []
    for o in cell.get("outputs", []):
        if o.get("name") == "stderr":
            continue
        t = "".join(o.get("text", []) or []) or "".join(o.get("data", {}).get("text/plain", []) or [])
        if t:
            parts.append(t)
    return "\n".join(parts)


def _find(nb: dict, src_needle: str, out_pattern: str) -> re.Match:
    """The output of the cell whose SOURCE contains `src_needle`, matched against `out_pattern`."""
    for cell in nb.get("cells", []):
        if src_needle not in "".join(cell.get("source", [])):
            continue
        m = re.search(out_pattern, _outputs(cell))
        if m:
            return m
    raise SystemExit(
        f"could not find {out_pattern!r} in the output of the cell containing {src_needle!r}. "
        f"Their notebook changed, or it was re-run with outputs cleared -- either way this is a "
        f"missing reference value, and the comparison must not proceed without it."
    )


def _read(path: Path):
    import anndata

    if not path.exists():
        raise SystemExit(f"their data is missing: {path}\n"
                         f"  run: sbatch model/jobs/uce_their_data.sbatch")
    return anndata.read_h5ad(path)


def _verdict(ours: float, theirs: float, tol: float) -> dict:
    d = float(ours) - float(theirs)
    return {"ours": float(ours), "theirs": float(theirs), "diff": d,
            "reproduced": bool(abs(d) <= tol), "tol": tol}


def protocol_b() -> dict:
    """fly_embedding_check.ipynb, on their fly atlas, every embedding it carries."""
    from uce_protocols import _logreg_folds

    nb = _nb("fly_embedding_check.ipynb")
    ref = _find(nb, "Mean over folds", r"Mean over folds: acc=([0-9.]+) ± ([0-9.]+)")
    ref_ctrl = _find(nb, "np.random.shuffle(y)", r"Mean over folds: acc=([0-9.]+) ± ([0-9.]+)")

    ad = _read(THEIR_DATA / "fly_cell_atlas_uce_adata_33layer.h5ad")
    vcs = ad.obs["broad_annotation"].value_counts()
    their_keep = list(vcs[vcs >= 100].index)[1:]
    ours_keep = [t for t in vcs[vcs >= 100].index if t not in ("unknown", "artefact")]
    sub = ad[(ad.obs["broad_annotation"].isin(their_keep))
             & (~ad.obs["broad_annotation"].isin(["unknown", "artefact"]))]
    y = sub.obs["broad_annotation"].values

    out = {"protocol": "fly_embedding_check.ipynb", "their_file": "fly_cell_atlas_uce_adata_33layer.h5ad",
           "n_cells": int(sub.n_obs),
           "selection_equivalence": {
               "their_positional_[1:]_then_name_filter": sorted(map(str, set(their_keep) - {"unknown", "artefact"})),
               "our_name_filter_only": sorted(map(str, ours_keep)),
               "identical": sorted(map(str, set(their_keep) - {"unknown", "artefact"})) == sorted(map(str, ours_keep)),
           },
           "embeddings": {}}
    for key in sub.obsm:
        if not key.startswith("X_") or key.endswith("umap"):
            continue
        X = np.asarray(sub.obsm[key])
        if X.ndim != 2 or X.shape[1] < 2:
            continue
        accs = _logreg_folds(X, y, shuffle_labels=False)
        np.random.seed(0)
        ctrl = _logreg_folds(X, y, shuffle_labels=True)
        row = {"acc_mean": float(np.mean(accs)), "acc_sd": float(np.std(accs)), "folds": accs,
               "shuffled_labels_mean": float(np.mean(ctrl))}
        if key == "X_uce":
            row["vs_theirs"] = _verdict(np.mean(accs), float(ref.group(1)), TOL["b_logreg"])
            row["vs_theirs_shuffled"] = _verdict(np.mean(ctrl), float(ref_ctrl.group(1)), TOL["b_logreg"])
        out["embeddings"][key] = row
    return out


def protocol_c() -> dict:
    """check_cosine_similarity.ipynb: their two runs of the same 100k cells, their three numbers."""
    nb = _nb("check_cosine_similarity.ipynb")
    ref_rand = _find(nb, "cs_random_pair.mean()", r"np\.float32\(([0-9.]+)\), np\.float32\(([0-9.]+)\)")
    ref_seed = _find(nb, "cs_different_seed.mean()", r"np\.float32\(([0-9.]+)\), np\.float32\(([0-9.]+)\)")

    a = _read(THEIR_DATA / "tabula_sapiens_subset_unshuffled_uce_adata.h5ad")
    b = _read(THEIR_DATA / "tabula_sapiens_subset_shuffled_uce_adata.h5ad")

    key = "idx" if "idx" in a.obs and "idx" in b.obs else None
    ia = a.obs[key].to_numpy() if key else a.obs_names.to_numpy()
    ib = b.obs[key].to_numpy() if key else b.obs_names.to_numpy()
    if set(ia.tolist()) != set(ib.tolist()):
        raise SystemExit(f"their two runs are not the same cells on {key or 'obs_names'}; "
                         f"|A\\B|={len(set(ia.tolist()) - set(ib.tolist()))} -- refusing to pair them")
    order = {v: i for i, v in enumerate(ib.tolist())}
    perm_b = np.array([order[v] for v in ia.tolist()])

    X = np.asarray(a.obsm["X_uce"], dtype=np.float64)
    Y = np.asarray(b.obsm["X_uce"], dtype=np.float64)[perm_b]
    if X.shape != Y.shape:
        raise SystemExit(f"their two runs disagree in shape: {X.shape} vs {Y.shape}")

    def cos(P, Q):
        num = (P * Q).sum(1)
        return num / (np.linalg.norm(P, axis=1) * np.linalg.norm(Q, axis=1) + 1e-12)

    same = cos(X, Y)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    rand = cos(X, Y[perm])
    return {"protocol": "check_cosine_similarity.ipynb", "n_cells": int(len(X)),
            "aligned_on": key or "obs_names",
            "shuffled_gene_order_same_cell": {
                **_verdict(same.mean(), float(ref_seed.group(1)), TOL["c_cosine"]),
                "sd_ours": float(same.std()), "sd_theirs": float(ref_seed.group(2))},
            "random_pairing_null": {
                **_verdict(rand.mean(), float(ref_rand.group(1)), TOL["c_cosine"]),
                "sd_ours": float(rand.std()), "sd_theirs": float(ref_rand.group(2)),
                "note": "their null pairs cells as the file happens to order them; ours permutes "
                        "with a stated seed, so agreement here is distributional, not exact"}}


def protocol_a(sample_size: int = 100_000, resamples: int = 2, *,
               only_embedding: str | None = None, only_resample: int | None = None) -> dict:
    """scib.ipynb, on their Tabula Sapiens file, over every embedding it carries."""
    from scib_metrics.benchmark import BatchCorrection, BioConservation, Benchmarker

    ad = _read(THEIR_DATA / "new_tabula_scib.h5ad")
    keys = [k for k in ad.obsm if k.startswith("X_") and not k.endswith("umap")]
    if only_embedding is not None:
        if only_embedding not in keys:
            raise SystemExit(f"{only_embedding!r} is not in the file; it carries {keys}")
        keys = [only_embedding]
    draws = range(resamples) if only_resample is None else [only_resample]
    label = next(c for c in ("cell_ontology_class", "cluster_id", "cell_type") if c in ad.obs)
    batch = next((c for c in ("sample_id", "donor_assay", "donor") if c in ad.obs), None)
    if batch is None:
        ad.obs["batch"] = "one"
        batch = "batch"

    out = {"protocol": "scib.ipynb", "their_file": "new_tabula_scib.h5ad",
           "label_key": label, "batch_key": batch, "n_cells_total": int(ad.n_obs),
           "sample_size": sample_size, "resamples": resamples,
           "shard": None if only_embedding is None and only_resample is None
                    else {"embedding": only_embedding, "resample": only_resample},
           "subsampling": "their own recipe -- see the docstring; their notebook subsamples and "
                          "aggregates mean/std across draws",
           "neighbors": "pynndescent (CPU). Theirs is faiss GPU HNSW; no faiss is installed here, "
                        "so the neighbour graph is approximate in a different way. Same metric "
                        "definitions, different backend.",
           "embeddings": {}}

    import numpy as _np
    import scanpy as sc

    per_draw: dict[str, list[dict]] = {k: [] for k in keys}
    for i in draws:
        sub = sc.pp.subsample(ad.to_memory() if hasattr(ad, "to_memory") else ad,
                              copy=True, n_obs=min(sample_size, ad.n_obs), random_state=i)
        bm = Benchmarker(
            sub, batch_key=batch, label_key=label, embedding_obsm_keys=keys,
            bio_conservation_metrics=BioConservation(),
            batch_correction_metrics=BatchCorrection(
                ilisi_knn=False, kbet_per_label=False, graph_connectivity=False, pcr_comparison=False),
            n_jobs=-1,
        )
        bm.benchmark()
        df = bm.get_results(min_max_scale=False)
        for k in keys:
            if k in df.index:
                row = {c: float(v) for c, v in df.loc[k].items()
                       if isinstance(v, (int, float, _np.floating)) and _np.isfinite(float(v))}
                per_draw[k].append(row)
        print(f"[their_eval] protocol A draw {i + 1}/{resamples} done "
                  f"(embeddings: {keys})", flush=True)

    nb = _nb("scib.ipynb")
    theirs: dict[str, dict[str, float]] = {}
    blob = _outputs(_cell_with(nb, "new_tabula_benchmark_results_df"))
    for m in re.finditer(r"^(X_\w+)\s+([\d.\s]+)$", blob, re.M):
        vals = [float(x) for x in m.group(2).split()]
        theirs[m.group(1).lower()] = {"printed": vals}

    for k in keys:
        draws = per_draw[k]
        if not draws:
            continue
        cols = sorted(set().union(*[set(d) for d in draws]))
        row = {c: {"mean": float(_np.mean([d[c] for d in draws if c in d])),
                   "sd": float(_np.std([d[c] for d in draws if c in d]))} for c in cols}
        t = theirs.get(k.lower())
        if t:
            row["theirs_printed_full_run"] = t["printed"]
        out["embeddings"][k] = row
    return out


def _cell_with(nb: dict, needle: str) -> dict:
    for cell in nb.get("cells", []):
        if needle in "".join(cell.get("source", [])):
            return cell
    raise SystemExit(f"no cell containing {needle!r} in their notebook")


def protocol_b_scrm() -> dict:
    """THE ONE THAT PUTS scrm ON THEIR DATA: their file, their labels, their protocol, our vectors."""

    sys.path.insert(0, str(REPO / "measurements"))
    from uce_protocols import _load, _logreg_folds, runs_for

    ad = _read(THEIR_DATA / "fly_cell_atlas_uce_adata_33layer.h5ad")
    bc = ad.obs["Barcode"].astype(str).str.slice(0, 16).to_numpy()
    theirs_ix = {b: i for i, b in enumerate(bc)}

    from scrm_eval import registry as _R

    vecs, orders = {}, {}
    for name in list(runs_for("fly_antenna")) + list(_R.baselines()):
        try:
            orders[name], vecs[name] = _load("fly_antenna", name)
        except (FileNotFoundError, Exception) as exc:  # noqa: BLE001 - a missing arm must not stop the rest
            print(f"[their_eval] no fly artifact for {name}: {type(exc).__name__}", flush=True)

    if not vecs:
        raise SystemExit("no scrm fly artifacts found -- nothing to compare")

    common = None
    for cells in orders.values():
        s = {c for c in cells if c in theirs_ix}
        common = s if common is None else (common & s)
    common = sorted(common)
    if len(common) < 200:
        raise SystemExit(f"only {len(common)} shared cells -- too few for their >=100-cell rule")

    rows = np.array([theirs_ix[c] for c in common])
    y_all = ad.obs["broad_annotation"].astype(str).to_numpy()[rows]

    uniq, cnt = np.unique(y_all, return_counts=True)
    keep = {t for t, n in zip(uniq, cnt) if n >= 100 and t not in ("unknown", "artefact")}
    mask = np.array([t in keep for t in y_all])
    y = y_all[mask]
    rows_kept = rows[mask]
    common_kept = [c for c, m in zip(common, mask) if m]

    out = {"protocol": "fly_embedding_check.ipynb, applied to scrm",
           "their_file": "fly_cell_atlas_uce_adata_33layer.h5ad",
           "their_labels": "broad_annotation",
           "their_cells_total": int(ad.n_obs),
           "cells_we_can_embed": len(common),
           "cells_scored_after_their_rule": int(mask.sum()),
           "classes": sorted(keep),
           "comparator": "X_uce recomputed on THESE cells in the same folds -- their published 0.8060 "
                         "is over 307,241 whole-atlas cells and is not comparable to this population",
           "embeddings": {}}

    def score(name, X):
        accs = _logreg_folds(X, y, shuffle_labels=False)
        np.random.seed(0)
        ctrl = _logreg_folds(X, y, shuffle_labels=True)
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
        std = _logreg_folds(Xs, y, shuffle_labels=False)
        out["embeddings"][name] = {
            "dim": int(X.shape[1]),
            "acc_mean": float(np.mean(accs)), "acc_sd": float(np.std(accs)),
            "shuffled_labels_mean": float(np.mean(ctrl)),
            "acc_mean_standardised_ADDITION": float(np.mean(std)),
        }
        print(f"[their_eval]   {name}: {np.mean(accs):.4f} +/- {np.std(accs):.4f} "
              f"(standardised {np.mean(std):.4f})", flush=True)

    score("X_uce (theirs, these cells)", np.asarray(ad.obsm["X_uce"])[rows_kept].astype(np.float64))
    for name, cells in orders.items():
        by = {c: i for i, c in enumerate(cells)}
        score(name, vecs[name][[by[c] for c in common_kept]].astype(np.float64))
    return out


def protocol_cert_fly() -> dict:
    """Certify OUR UCE invocation against THEIR published embedding, ON FLY -- with its null."""
    sys.path.insert(0, str(REPO / "measurements"))
    from uce_protocols import _load

    ad = _read(THEIR_DATA / "fly_cell_atlas_uce_adata_33layer.h5ad")
    bc = ad.obs["Barcode"].astype(str).str.slice(0, 16).to_numpy()
    ix = {b: i for i, b in enumerate(bc)}
    cells, ours = _load("fly_antenna", "uce")
    pairs = [(i, ix[c]) for i, c in enumerate(cells) if c in ix]
    if not pairs:
        raise SystemExit("no shared cells between our uce artifact and their fly file")
    MINE = np.asarray(ours)[np.array([a for a, _ in pairs])].astype(np.float64)
    THEIRS = np.asarray(ad.obsm["X_uce"])[np.array([b for _, b in pairs])].astype(np.float64)

    def cos(A, B):
        return (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(MINE))
    same, null = cos(MINE, THEIRS), cos(MINE, THEIRS[perm])

    def anisotropy(X, n=800):
        """Mean pairwise cosine on a subsample -- the null's origin, stated as a property of the space."""
        S = X[rng.choice(len(X), min(n, len(X)), replace=False)]
        S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
        G = S @ S.T
        iu = np.triu_indices(len(S), 1)
        return float(G[iu].mean())

    return {"protocol": "certify our UCE invocation on THEIR fly file",
            "their_file": "fly_cell_atlas_uce_adata_33layer.h5ad",
            "cells_matched": len(pairs), "dim": int(MINE.shape[1]),
            "per_cell_cosine_ours_vs_theirs": {
                "mean": float(same.mean()), "sd": float(same.std()),
                "min": float(same.min()), "p01": float(np.percentile(same, 1))},
            "shuffled_pairing_null": {"mean": float(null.mean()), "sd": float(null.std())},
            "margin_over_null": float(same.mean() - null.mean()),
            "anisotropy_mean_pairwise_cosine": {"ours": anisotropy(MINE), "theirs": anisotropy(THEIRS)},
            "why_this_exists": "the certification on record was measured on mouse_kidney_norn "
                               "(0.9779 against a 0.5011 null, margin 0.477); it does not transfer "
                               "to fly, where the null is 0.864 and the margin is 0.093"}


def protocol_d_scrm() -> dict:
    """PROTOCOL D ON THEIR DATA: their fly file, THEIR two label granularities, our vectors."""
    import scanpy as sc
    from scipy.cluster.hierarchy import fcluster
    from sklearn.metrics import adjusted_rand_score

    sys.path.insert(0, str(REPO / "measurements"))
    from uce_protocols import _load, runs_for

    ad = _read(THEIR_DATA / "fly_cell_atlas_uce_adata_33layer.h5ad")
    FINE, BROAD = "consensus_community_annotation", "broad_annotation"
    for c in (FINE, BROAD):
        if c not in ad.obs:
            raise SystemExit(f"their fly file has no {c!r} -- protocol D needs the nested pair")
    bc = ad.obs["Barcode"].astype(str).str.slice(0, 16).to_numpy()
    ix = {b: i for i, b in enumerate(bc)}

    vecs, orders = {}, {}
    from scrm_eval import registry as _R
    for name in list(runs_for("fly_antenna")) + list(_R.baselines()):
        try:
            orders[name], vecs[name] = _load("fly_antenna", name)
        except Exception as exc:  # noqa: BLE001 - a missing arm must not stop the rest
            print(f"[their_eval] no fly artifact for {name}: {type(exc).__name__}", flush=True)
    if not vecs:
        raise SystemExit("no fly artifacts -- nothing to score")

    common = None
    for cells in orders.values():
        sset = {c for c in cells if c in ix}
        common = sset if common is None else (common & sset)
    common = sorted(common)
    rows = np.array([ix[c] for c in common])
    fine = ad.obs[FINE].astype(str).to_numpy()[rows]
    broad = ad.obs[BROAD].astype(str).to_numpy()[rows]

    uniq, cnt = np.unique(fine, return_counts=True)
    keep_t = {t for t, n in zip(uniq, cnt) if n >= 100 and t not in ("unknown", "artefact")}
    mask = np.array([t in keep_t for t in fine])
    fine, broad, rows = fine[mask], broad[mask], rows[mask]
    common = [c for c, m in zip(common, mask) if m]
    types = sorted(set(fine))
    import collections
    parent = {t: collections.Counter(broad[fine == t]).most_common(1)[0][0] for t in types}
    n_broad = len({parent[t] for t in types})
    out = {"protocol": "fly_dendrogram.ipynb, applied to scrm",
           "their_file": "fly_cell_atlas_uce_adata_33layer.h5ad",
           "their_labels": {"fine": FINE, "broad": BROAD},
           "n_cells": len(common), "n_fine_types": len(types), "n_broad_classes": n_broad,
           "min_cells_per_type": 100, "results": {}}
    if len(types) < 3:
        out["skipped"] = (f"only {len(types)} of their fine types clear the >=100-cell threshold on "
                          f"the {len(common)} cells scrm can embed -- the protocol needs a hierarchy "
                          "to score, and this population does not carry one")
        return out

    import anndata
    truth = [parent[t] for t in types]
    reps = {"X_uce (theirs)": np.asarray(ad.obsm["X_uce"])[rows].astype(np.float32)}
    for name, cells in orders.items():
        by = {c: i for i, c in enumerate(cells)}
        reps[name] = vecs[name][[by[c] for c in common]].astype(np.float32)

    for rep, X in reps.items():
        a = anndata.AnnData(X=X.copy(), obs={"fine": fine})
        a.obs["fine"] = a.obs["fine"].astype("category")
        a.obsm["X_rep"] = X
        sc.tl.dendrogram(a, use_rep="X_rep", groupby="fine", key_added="d")
        d = a.uns["d"]
        order = list(d.get("categories_ordered", a.obs["fine"].cat.categories))
        cut = fcluster(np.asarray(d["linkage"]), t=n_broad, criterion="maxclust")
        lab = {order[i]: int(cut[i]) for i in range(len(order))}
        ari = adjusted_rand_score(truth, [lab[t] for t in types])
        cen = np.stack([X[fine == t].mean(axis=0) for t in types])
        cen = cen - cen.mean(axis=0, keepdims=True)
        cc = np.corrcoef(cen)
        np.fill_diagonal(cc, -np.inf)
        nn = [types[int(np.argmax(cc[i]))] for i in range(len(types))]
        purity = float(np.mean([parent[types[i]] == parent[nn[i]] for i in range(len(types))]))
        out["results"][rep] = {"ari_broad_ADDITION": float(ari), "nn_purity_ADDITION": purity}
        print(f"[their_eval]   {rep:34s} ARI(broad)={ari:.4f}  nn_purity={purity:.4f}", flush=True)
    return out


PROTOCOLS = {"b": protocol_b, "c": protocol_c, "a": protocol_a, "scrm": protocol_b_scrm,
             "cert_fly": protocol_cert_fly, "d_scrm": protocol_d_scrm}


def reduce_protocol_a(report_dir: Path | None = None) -> dict:
    """Recombine the (embedding, draw) shards into their notebook's mean/std aggregation."""
    import numpy as _np

    d = Path(report_dir) if report_dir else _R.PROJ / "eval_reports" / "uce"
    shards: dict[str, dict[int, dict]] = {}
    meta: dict = {}
    for f in sorted(d.glob("their_eval_a_X_*_d*.json")):
        blob = json.loads(f.read_text())
        a_blob = blob.get("protocols", {}).get("a", {})
        sh = a_blob.get("shard") or {}
        emb, draw = sh.get("embedding"), sh.get("resample")
        if emb is None or draw is None:
            continue
        row = a_blob.get("embeddings", {}).get(emb)
        if row:
            shards.setdefault(emb, {})[int(draw)] = row
        meta = meta or {k: v for k, v in a_blob.items() if k not in ("embeddings", "shard")}

    want = int(meta.get("resamples", 2))
    out, incomplete = {}, {}
    for emb, draws in sorted(shards.items()):
        if len(draws) < want:
            incomplete[emb] = sorted(draws)
            continue
        metrics = set.intersection(*[set(v) for v in draws.values()])

        def _val(row):
            """protocol_a already wraps each draw as {"mean": x, "sd": 0.0} -- one draw has no
            spread. Read through that wrapper rather than averaging dicts.
            """
            return float(row["mean"]) if isinstance(row, dict) else float(row)

        out[emb] = {m: {"mean": float(_np.mean([_val(draws[i][m]) for i in sorted(draws)])),
                        "std": float(_np.std([_val(draws[i][m]) for i in sorted(draws)])),
                        "draws": len(draws)}
                    for m in sorted(metrics)}
    return {**meta, "reduced_from_shards": True, "embeddings": out,
            "incomplete_embeddings": incomplete,
            "note": "mean/std across draws per embedding, their notebook's aggregation. An "
                    "embedding missing a draw is listed in incomplete_embeddings and excluded."}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--embedding", help="protocol A: score ONE obsm key (e.g. X_uce)")
    ap.add_argument("--resample", type=int, help="protocol A: run ONE draw index")
    ap.add_argument("--reduce-a", action="store_true",
                    help="combine the protocol A shards into their mean/std aggregation")
    a = ap.parse_args(argv)
    if a.reduce_a:
        red = reduce_protocol_a()
        print(json.dumps(red, indent=2)[:4000])
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            tmp = a.out.with_suffix(a.out.suffix + ".tmp")
            tmp.write_text(json.dumps(red, indent=2) + "\n")
            tmp.replace(a.out)
            print(f"wrote {a.out}")
        return 0 if red.get("embeddings") else 1
    if not a.all and not a.protocol:
        ap.error("pick --protocol {a,b,c} or --all")
    which = sorted(PROTOCOLS) if a.all else [a.protocol]

    report = {"what": "UCE's protocols on UCE's data with UCE's models, compared against the "
                      "numbers stored in UCE's own notebook outputs",
              "their_notebooks": str(THEIR_NB), "their_data": str(THEIR_DATA), "protocols": {}}
    failed = []
    for name in which:
        try:
            fn = PROTOCOLS[name]
            if name == "a" and (a.embedding or a.resample is not None):
                fn = functools.partial(fn, only_embedding=a.embedding, only_resample=a.resample)
            report["protocols"][name] = fn()
        except SystemExit as exc:
            report["protocols"][name] = {"error": str(exc)}
            failed.append(name)
            print(f"[their_eval] protocol {name}: {exc}", flush=True)
    print(json.dumps(report, indent=2)[:4000])
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = a.out.with_suffix(a.out.suffix + ".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(a.out)
        print(f"wrote {a.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
