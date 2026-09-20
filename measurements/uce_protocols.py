"""UCE's evaluation protocols, run on scrm artifacts.

UCE's code and weights live at $SCRM_PROJ/vendor/uce_reproduce, not in this repository.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO / "eval", REPO / "model"):
    sys.path.insert(0, str(extra))

from scrm_eval import controls, artifact, reduce as _reduce  # noqa: E402
from scrm_eval import registry as _registry  # noqa: E402

P = _registry.PROJ
OUT = P / "eval_reports" / "uce"

LABELS = {d: _registry.label_file(d) for d in _registry.labelled_datasets()}
MARKER_DERIVED = _registry.marker_derived()
SEED_SET = tuple(_registry.SEED_FLOOR_SET)


def runs_for(dataset: str) -> tuple[str, ...]:
    """EVERY full-coverage scrm arm on this dataset, minus the ones trained on it."""
    import json

    from scrm_eval import registry as R

    bad = set(R.in_domain_arms(dataset))
    sets = json.loads((REPO / "model/jobs/training_sets.json").read_text())["sets"]
    for name, members in sets.items():
        corpora = members["corpora"] if isinstance(members, dict) else members
        if dataset in corpora:
            bad |= {a for a in R._MIXED_SPECIES_ARMS if name in a} or set(R._MIXED_SPECIES_ARMS)
    return tuple(a for a in R.arms_with_full_coverage(dataset) if a not in bad)
BASELINES = ("count_scanpy",) + tuple(controls.PANEL)


CENTRE = bool(os.environ.get("CENTRE"))
STANDARDISE = bool(os.environ.get("STANDARDISE"))


def _load(dataset: str, name: str):
    """(bare barcodes, cell matrix) for one model, windows mean-reduced -- the published rule."""
    path = P / "eval_runs" / name / f"{dataset}.{name}.allreads.arrow"
    if not path.exists():
        path = P / "eval_artifacts" / f"{dataset}.{name}.arrow"
    emb = artifact.read(path)
    ours = emb.stamp.producer in artifact.MODEL_PRODUCERS
    rule = _reduce.rule_for(emb.stamp)
    if CENTRE and ours:
        rule = "mean_centred"
    cells, mat = _reduce.reduce_windows(emb.cell_ids, emb.matrix, rule)
    if STANDARDISE:
        mat = (mat - mat.mean(0)) / (mat.std(0) + 1e-8)
    return [str(c).split("-")[0] for c in cells], mat


def _labels(dataset: str) -> dict[str, str]:
    return {k.split("-")[0]: v for k, v in json.loads(LABELS[dataset].read_text()).items()}


def _panel(dataset: str):
    """One AnnData over the intersection of every model's cells with the labelled cells, each
    embedding in obsm -- the layout scib.ipynb's Benchmarker consumes.
    """
    import anndata

    labels = _labels(dataset)
    mats, orders = {}, {}
    for name in runs_for(dataset) + BASELINES:
        try:
            orders[name], mats[name] = _load(dataset, name)
        except FileNotFoundError:
            print(f"[uce] {dataset}: no artifact for {name}, skipped")
    common = None
    for cells in orders.values():
        s = {c for c in cells if c in labels}
        common = s if common is None else (common & s)
    common = sorted(common)
    by0 = {c: i for i, c in enumerate(orders["count_scanpy"])}
    ad = anndata.AnnData(
        X=mats["count_scanpy"][[by0[c] for c in common]].astype(np.float32),
        obs={"cell_type": [labels[c] for c in common], "batch": ["b0"] * len(common)},
    )
    ad.obs_names = common
    for name, cells in orders.items():
        by = {c: i for i, c in enumerate(cells)}
        ad.obsm[f"X_{name}"] = mats[name][[by[c] for c in common]]
    return ad


def protocol_a_scib(dataset: str) -> dict:
    """scib.ipynb's benchmark(), verbatim settings; batch metrics off (single-channel data)."""
    from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation

    ad = _panel(dataset)
    keys = [k for k in ad.obsm]
    bm = Benchmarker(
        ad,
        batch_key="batch",
        label_key="cell_type",
        embedding_obsm_keys=keys,
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(
            bras=False, ilisi_knn=False, kbet_per_label=False,
            graph_connectivity=False, pcr_comparison=False,
        ),
        n_jobs=min(64, len(__import__("os").sched_getaffinity(0))),
    )
    bm.prepare()
    bm.benchmark()
    try:
        df = bm.get_results(min_max_scale=False)
    except KeyError:
        df = bm._results.transpose()
        df = df.loc[df.index != "Metric Type"]
    out = {"dataset": dataset, "protocol": "scib.ipynb benchmark(), BioConservation defaults, "
                                           "min_max_scale=False; no batch axis in this data",
           "n_cells": int(ad.n_obs), "n_types": int(ad.obs['cell_type'].nunique()),
           "results": json.loads(df.to_json(orient="index"))}
    return out


def _logreg_folds(X, y, shuffle_labels: bool) -> list[float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedKFold

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    accs = []
    y = np.asarray(y).copy()
    for train_idx, val_idx in kf.split(X, y):
        if shuffle_labels:
            np.random.shuffle(y)
        model = LogisticRegression(n_jobs=1, max_iter=200)
        model.fit(X[train_idx], y[train_idx])
        accs.append(float(accuracy_score(y[val_idx], model.predict(X[val_idx]))))
    return accs


def _b_one_model(key, X, y):
    """One embedding's three fold-sets. Runs in a worker process, ONE BLAS thread."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        X = np.asarray(X)
        accs = _logreg_folds(X, y, shuffle_labels=False)
        np.random.seed(0)
        ctrl = _logreg_folds(X, y, shuffle_labels=True)
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
        accs_std = _logreg_folds(Xs, y, shuffle_labels=False)
    return key.removeprefix("X_"), {
        "acc_mean": float(np.mean(accs)), "acc_sd": float(np.std(accs)), "folds": accs,
        "shuffled_labels_mean": float(np.mean(ctrl)),
        "acc_mean_standardised_ADDITION": float(np.mean(accs_std)),
    }


def protocol_b_logreg(dataset: str) -> dict:
    """fly_embedding_check.ipynb: >=100-cell types, 10-fold logistic accuracy, shuffled control."""
    import os

    from joblib import Parallel, delayed

    ad = _panel(dataset)
    vcs = ad.obs["cell_type"].value_counts()
    keep = [t for t in vcs[vcs >= 100].index if t not in ("unknown", "artefact")]
    sub = ad[ad.obs["cell_type"].isin(keep)]
    out = {"dataset": dataset, "protocol": "fly_embedding_check.ipynb: types>=100 cells, "
                                           "StratifiedKFold(10, seed 0), LogisticRegression(max_iter=200)",
           "n_cells": int(sub.n_obs), "types": sorted(keep), "results": {}}
    y = sub.obs["cell_type"].values
    keys = list(sub.obsm)
    n_jobs = min(len(keys), len(os.sched_getaffinity(0)))
    print(f"[uce]   protocol b: {len(keys)} models x 30 fits, {n_jobs} workers, 1 BLAS thread each",
          flush=True)
    done = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_b_one_model)(k, np.asarray(sub.obsm[k]), y) for k in keys)
    out["results"] = dict(done)
    return out


def protocol_c_cosine(dataset: str) -> dict:
    """check_cosine_similarity.ipynb: same-cell cosine across TRAINING seeds of the flagship recipe,
    against the within-cell-type shuffled-pair reference (their >=5,000-cell threshold scaled to 100).
    """
    def cos(A, B):
        A = A.astype(np.float32)
        B = B.astype(np.float32)
        dot = np.einsum("ij,ij->i", A, B)
        return dot / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-8)

    labels = _labels(dataset)
    loaded, absent = {}, []
    for name in SEED_SET:
        try:
            loaded[name] = _load(dataset, name)
        except (FileNotFoundError, OSError):
            absent.append(name)
    if len(loaded) < 2:
        return {"dataset": dataset, "skipped": True, "seeds_absent": absent,
                "reason": f"protocol C needs two seeds of one recipe; {len(loaded)} of "
                          f"{len(SEED_SET)} members of SEED_FLOOR_SET are embedded for this dataset"}

    def one_pair(na, nb):
        a_cells, A = loaded[na]
        b_cells, B = loaded[nb]
        by = {c: i for i, c in enumerate(b_cells)}
        common = [c for c in a_cells if c in by and c in labels]
        ai = {c: i for i, c in enumerate(a_cells)}
        A = A[[ai[c] for c in common]]
        B = B[[by[c] for c in common]]
        same = cos(A, B)
        y = np.asarray([labels[c] for c in common])
        np.random.seed(0)
        refs = []
        for t, n in zip(*np.unique(y, return_counts=True)):
            if n < 100:
                continue
            X = A[y == t]
            refs.append(cos(X, X[np.random.choice(len(X), size=len(X), replace=False)]))
        ref = np.concatenate(refs) if refs else np.array([np.nan])
        np.random.seed(1)
        perm = np.random.permutation(len(A))
        cross = cos(A, A[perm])
        Ac = A - A.mean(0)
        Bc = B - B.mean(0)
        same_c = cos(Ac, Bc)
        cross_c = cos(Ac, Ac[perm])
        return {"seed_a": na, "seed_b": nb, "n_cells": len(common),
                "same_cell_across_seeds": {"mean": float(same.mean()), "sd": float(same.std())},
                "within_type_shuffled_pairs": {"mean": float(np.nanmean(ref)),
                                               "sd": float(np.nanstd(ref))},
                "anisotropy_ADDITION": {
                    "all_pairs_raw": {"mean": float(cross.mean()), "sd": float(cross.std())},
                    "same_cell_across_seeds_centred": {"mean": float(same_c.mean()),
                                                       "sd": float(same_c.std())},
                    "all_pairs_centred": {"mean": float(cross_c.mean()), "sd": float(cross_c.std())}}}

    names = [n for n in SEED_SET if n in loaded]
    pairs = [one_pair(names[a], names[b])
             for a in range(len(names)) for b in range(a + 1, len(names))]
    worst = min(pairs, key=lambda r: r["same_cell_across_seeds"]["mean"])
    return {"dataset": dataset,
            "protocol": "check_cosine_similarity.ipynb: per-cell cosine over the flagship's reseed "
                        f"set {tuple(names)}; the reported pair is the WORST of {len(pairs)} "
                        "(lowest same-cell cosine); reference = within-type shuffled pairs "
                        "(>=100 cells)",
            "protocol_caveat": "UCE's notebook compares INFERENCE seeds of one trained model; ours "
                               "are two independently TRAINED models -- a strictly harder test that "
                               "no unaligned pair passes in raw cosine",
            "seeds_absent": absent,
            "seed_pairs_ADDITION": pairs,
            "reported_pair": [worst["seed_a"], worst["seed_b"]],
            "anisotropy_ADDITION": worst["anisotropy_ADDITION"],
            "n_cells": worst["n_cells"],
            "same_cell_across_seeds": worst["same_cell_across_seeds"],
            "within_type_shuffled_pairs": worst["within_type_shuffled_pairs"]}


def protocol_d_dendrogram(dataset: str) -> dict:
    """fly_dendrogram.ipynb: sc.tl.dendrogram over cell types, computed FROM the embedding."""
    import scanpy as sc
    from scipy.cluster.hierarchy import fcluster
    from sklearn.metrics import adjusted_rand_score

    GRANULARITIES = {
        "fly_antenna": (P / "raw/fly_antenna_meta/fly_antenna_labels_fine.json",
                        P / "raw/fly_antenna_meta/fly_antenna_labels_broad.json"),
    }
    if dataset not in GRANULARITIES:
        return {"dataset": dataset,
                "skipped": "protocol D needs a NESTED pair of label granularities (fine types inside "
                           "broad classes) to score a hierarchy against a count-independent "
                           f"reference; {dataset} ships one flat label set, so the protocol does not "
                           "apply to it. Their notebook is fly-only for the same reason."}
    fine_p, broad_p = GRANULARITIES[dataset]
    if not (fine_p.exists() and broad_p.exists()):
        return {"dataset": dataset, "skipped": f"label files missing: {fine_p}, {broad_p}"}
    fine = {k.split("-")[0]: v for k, v in json.loads(fine_p.read_text()).items()}
    broad = {k.split("-")[0]: v for k, v in json.loads(broad_p.read_text()).items()}
    ad = _panel(dataset)
    keep = [c for c in ad.obs_names if c in fine and c in broad]
    ad = ad[keep].copy()
    ad.obs["fine"] = [fine[c] for c in ad.obs_names]
    ad.obs["broad"] = [broad[c] for c in ad.obs_names]
    vc = ad.obs["fine"].value_counts()
    keep_t = set(vc[vc >= 100].index)
    ad = ad[[t in keep_t for t in ad.obs["fine"]]].copy()
    ad.obs["fine"] = ad.obs["fine"].astype("category")
    parent = {t: ad.obs["broad"][ad.obs["fine"] == t].mode()[0] for t in ad.obs["fine"].cat.categories}
    types = list(ad.obs["fine"].cat.categories)
    n_broad = len({parent[t] for t in types})
    out = {"dataset": dataset, "n_cells": int(ad.n_obs), "n_fine_types": len(types),
           "n_broad_classes": n_broad, "min_cells_per_type": 100,
           "protocol": "fly_dendrogram.ipynb: sc.tl.dendrogram(use_rep=...) over fine cell types "
                       "with >=100 cells; ADDITION = nn_purity and ARI of the dendrogram cut at "
                       "n_broad against the authors' broad classes (count-independent reference)",
           "results": {}}
    if len(types) < 3:
        out["skipped"] = f"only {len(types)} fine types clear the >=100-cell threshold"
        return out
    truth = [parent[t] for t in types]
    for rep in [k[2:] for k in ad.obsm]:
        sc.tl.dendrogram(ad, use_rep=f"X_{rep}", groupby="fine", key_added=f"d_{rep}")
        d = ad.uns[f"d_{rep}"]
        Z = np.asarray(d["linkage"])
        order = list(d["categories_ordered"] if "categories_ordered" in d else types)
        cut = fcluster(Z, t=n_broad, criterion="maxclust")
        lab = {order[i]: int(cut[i]) for i in range(len(order))}
        ari = adjusted_rand_score(truth, [lab[t] for t in types])
        cen = np.stack([ad.obsm[f"X_{rep}"][np.asarray(ad.obs["fine"] == t)].mean(axis=0) for t in types])
        cen = cen - cen.mean(axis=0, keepdims=True)
        cc = np.corrcoef(cen)
        np.fill_diagonal(cc, -np.inf)
        nn = [types[int(np.argmax(cc[i]))] for i in range(len(types))]
        purity = float(np.mean([parent[types[i]] == parent[nn[i]] for i in range(len(types))]))
        out["results"][rep] = {"ari_broad_ADDITION": float(ari), "nn_purity_ADDITION": purity}
        print(f"  {rep:26s} ARI(broad)={ari:.4f}  nn_purity={purity:.4f}", flush=True)
    return out


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=sorted(LABELS))
    ap.add_argument("--all", action="store_true")
    #: DEFAULT IS b ALONE. The paper shows protocol B and nothing else: figures/species.tex
    #: and measurements/figure_species.py read b_logreg.<dataset>.json, and no number in the
    #: paper comes from a_scib, c_cosine or d_dendrogram. Defaulting to "abcd" recomputed
    #: three unread families on every run. Pass them explicitly if you want them.
    ap.add_argument("--protocols", default="b", help="subset of 'abcd'; the paper uses b")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    datasets = sorted(LABELS) if a.all else [a.dataset]
    for ds in datasets:
        for proto, fn in (("a_scib", protocol_a_scib), ("b_logreg", protocol_b_logreg),
                          ("c_cosine", protocol_c_cosine), ("d_dendrogram", protocol_d_dendrogram)):
            if proto[0] not in a.protocols:
                continue
            tag = ".standardised" if STANDARDISE else (".centred" if CENTRE else "")
            out = OUT / f"{proto}.{ds}{tag}.json"
            if out.exists() and not os.environ.get("FORCE"):
                print(f"[uce] {ds} · {proto}: report exists, skipped (FORCE=1 to redo)", flush=True)
                continue
            print(f"[uce] {ds} · {proto}", flush=True)
            rep = fn(ds)
            rep["centred"] = CENTRE
            rep["standardised"] = STANDARDISE
            if STANDARDISE:
                rep["centred_note"] = (
                    "EVERY embedding per-feature standardised ((x - mean) / sd) before the protocol, "
                    "scrm and baselines alike, which subtracts the mean as well as equalising the "
                    "scales. Their protocol does neither. Measured on protocol B, where both columns "
                    "are published: it moves scrm +0.25 to +0.48 and every count model by <=0.05, "
                    "because 86% of our cell vector's variance sits in 2 of 128 directions and a "
                    "probe that penalises coefficients equally can only use the largest ones."
                )
            else:
                rep["centred_note"] = (
                "EVERY embedding reduced with reduce rule `mean_centred` (pooled mean minus the "
                "mean over the cells being embedded), scrm and baselines alike, so the protocol "
                "applies one uniform step to all models. Centring a baseline is a no-op to 3-4 dp "
                "(their offsets are 0.0-1.0 against scrm's 262); it is decisive only for scrm"
                if CENTRE else "embeddings used as stamped, no centring"
            )
            if ds in MARKER_DERIVED:
                rep["labels_marker_derived"] = True
                rep["labels_note"] = (
                    "Cell types are MARKER-DERIVED, not an author annotation: Leiden on this "
                    "dataset's counts, each cluster assigned to the highest-scoring canonical "
                    "immune marker set in the species' OWN Ensembl symbol casing (no ortholog "
                    "bridge), clusters without a clear margin left unlabelled. See "
                    "raw/<dataset>/marker_labels.provenance.json for every cluster's score vector. "
                    "CONSEQUENCE: count_scanpy is recovering a partition its own pipeline drew, so "
                    "read it as an upper bound rather than a comparison. scrm (reads only) and the "
                    "count FMs (their own vocabularies) are NOT circular on these labels."
                )
            tmp = out.with_name(out.name + ".tmp")
            tmp.write_text(json.dumps(rep, indent=2))
            tmp.replace(out)
            print(f"[uce]   wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
