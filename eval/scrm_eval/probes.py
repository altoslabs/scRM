"""E6/E9 probes: frozen embeddings → count-derived targets, with the mandatory baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import residualise


@dataclass
class ProbeResult:
    macro: float
    baseline: float
    per_target: np.ndarray
    metric: str
    n_cells: int
    n_targets: int
    seeds: int
    n_scored: int = -1
    seed_sd: float = float("nan")
    ci95: tuple = ()


def _ci95(per_target: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple:
    """Percentile bootstrap over TARGETS for the macro average."""
    v = np.asarray(per_target, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return ()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _seed_sd(per_seed: np.ndarray) -> float:
    """SD across seeds of the macro, where `per_seed` is (seeds, targets)."""
    if per_seed.shape[0] < 2:
        return float("nan")
    usable = per_seed[np.isfinite(per_seed).any(axis=1)]
    if usable.shape[0] < 2:
        return float("nan")
    macros = np.nanmean(usable, axis=1)
    macros = macros[np.isfinite(macros)]
    return float(np.std(macros, ddof=1)) if macros.size >= 2 else float("nan")


def _over_scored(per_seed: np.ndarray) -> tuple[np.ndarray, float, int]:
    """`(per_target, macro, n_scored)` from a (seeds, targets) array where NaN means "not scorable"."""
    finite = np.isfinite(per_seed)
    counts = finite.sum(axis=0)
    sums = np.where(finite, per_seed, 0.0).sum(axis=0)
    per_target = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    scored = per_target[np.isfinite(per_target)]
    macro = float(scored.mean()) if scored.size else float("nan")
    return per_target, macro, int(scored.size)


def _folds(n: int, k: int, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return [(np.concatenate([f for j, f in enumerate(np.array_split(order, k)) if j != i]), f)
            for i, f in enumerate(np.array_split(order, k))]


RIDGE_ALPHAS = np.logspace(-2, 4, 13)


def _ridge():
    from sklearn.linear_model import RidgeCV

    return RidgeCV(alphas=RIDGE_ALPHAS, alpha_per_target=True)


def _nuisance(depth, has_counts) -> np.ndarray:
    """The nuisance design for residualisation, as (n_kept, k). Accepts one column or several."""
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim == 1:
        d = d.reshape(-1, 1)
    return d[has_counts]


def binary_expression_probe(
    embedding: np.ndarray,
    targets: np.ndarray,
    has_counts: np.ndarray,
    *,
    folds: int = 5,
    seeds: int = 3,
) -> ProbeResult:
    """E6: per-gene AUROC of a ridge score, macro-averaged; baseline is chance (0.5) by
    construction, so what is reported beside the macro is the per-gene BASE RATE's best
    constant-score AUROC — exactly 0.5 — making any macro above it attributable to the
    embedding. Genes constant within a test fold are skipped in that fold.
    """
    from scipy.stats import rankdata
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = embedding[has_counts]
    y = targets[has_counts].astype(np.float32)
    asum = np.zeros((seeds, y.shape[1]))
    acnt = np.zeros((seeds, y.shape[1]), dtype=np.int64)
    for s in range(seeds):
        for train, test in _folds(len(x), folds, seed=s):
            model = make_pipeline(StandardScaler(), _ridge()).fit(x[train], y[train])
            scores = np.asarray(model.predict(x[test]))
            if scores.ndim == 1:
                scores = scores[:, None]
            yt = y[test]
            ranks = rankdata(scores, axis=0)
            npos = yt.sum(axis=0)
            nneg = len(test) - npos
            scorable = (npos > 0) & (nneg > 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                auc = ((ranks * yt).sum(axis=0) - npos * (npos + 1) / 2) / (npos * nneg)
            asum[s, scorable] += auc[scorable]
            acnt[s, scorable] += 1
    with np.errstate(invalid="ignore"):
        per_seed = np.where(acnt > 0, asum / np.maximum(acnt, 1), np.nan)
    per_gene = np.nanmean(per_seed, axis=0)
    return ProbeResult(
        macro=float(np.nanmean(per_gene)),
        baseline=0.5,
        per_target=per_gene,
        metric="auroc",
        n_cells=int(has_counts.sum()),
        n_targets=targets.shape[1],
        seeds=seeds,
        n_scored=int(np.isfinite(per_gene).sum()),
        seed_sd=_seed_sd(per_seed),
        ci95=_ci95(per_gene),
    )


def expression_regression_probe(
    embedding: np.ndarray,
    targets: np.ndarray,
    has_counts: np.ndarray,
    *,
    depth: np.ndarray | None = None,
    folds: int = 5,
    seeds: int = 3,
) -> ProbeResult:
    """E6b: per-gene Pearson r of ridge predictions of GRADED expression, macro-averaged."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = embedding[has_counts]
    y = targets[has_counts].astype(np.float64)
    if depth is not None:
        y = np.asarray(residualise(y, _nuisance(depth, has_counts)))
        y = y.reshape(len(x), -1)
    rs = np.full((seeds, y.shape[1]), np.nan)
    for s in range(seeds):
        preds = np.zeros_like(y)
        for train, test in _folds(len(x), folds, seed=s):
            pred = make_pipeline(StandardScaler(), _ridge()).fit(x[train], y[train]).predict(x[test])
            preds[test] = np.asarray(pred).reshape(len(test), -1)
        for g in range(y.shape[1]):
            if y[:, g].std() > 0 and preds[:, g].std() > 0:
                rs[s, g] = float(np.corrcoef(y[:, g], preds[:, g])[0, 1])
    per_gene, macro, n_scored = _over_scored(rs)
    return ProbeResult(
        macro=macro,
        baseline=0.0,
        per_target=per_gene,
        metric="pearson_r" + ("_depth_residualised" if depth is not None else ""),
        n_cells=int(has_counts.sum()),
        n_targets=targets.shape[1],
        seeds=seeds,
        n_scored=n_scored,
        seed_sd=_seed_sd(rs),
        ci95=_ci95(per_gene),
    )


def count_regression_probe(
    embedding: np.ndarray,
    counts: np.ndarray,
    has_counts: np.ndarray,
    depth: np.ndarray,
    *,
    family: str = "poisson",
    folds: int = 5,
    seeds: int = 1,
    alpha: float | None = None,
) -> ProbeResult:
    """E6c: predict RAW counts under a count likelihood, scored by fraction of deviance explained."""
    from sklearn.linear_model import PoissonRegressor
    from sklearn.preprocessing import StandardScaler

    if family not in ("poisson", "nb"):
        raise ValueError(f"family must be 'poisson' or 'nb', got {family!r}")
    x = embedding[has_counts]
    y = counts[has_counts].astype(np.float64)
    expo = np.asarray(depth, dtype=np.float64)[has_counts]
    expo = np.maximum(expo, 1.0)
    expo = expo / expo.mean()

    def poisson_deviance(obs, mu):
        mu = np.maximum(mu, 1e-9)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(obs > 0, obs * np.log(obs / mu), 0.0)
        return float(2.0 * np.sum(term - (obs - mu)))

    if alpha is None:
        probe_genes = np.argsort(-y.sum(axis=0))[: min(12, y.shape[1])]
        best, alpha = -np.inf, 1e-4
        for cand in (1e-6, 1e-4, 1e-2, 1.0):
            got = count_regression_probe(
                x, y[:, probe_genes], np.ones(len(x), bool), expo,
                family=family, folds=folds, seeds=1, alpha=cand,
            ).macro
            if got > best:
                best, alpha = got, cand

    scores = np.full((seeds, y.shape[1]), np.nan)
    for s in range(seeds):
        splits = _folds(len(x), folds, seed=s)
        for g in range(y.shape[1]):
            yg = y[:, g]
            if yg.sum() == 0:
                continue
            mu_hat = np.zeros(len(x))
            mu_null = np.zeros(len(x))
            for train, test in splits:
                sc = StandardScaler().fit(x[train])
                xtr, xte = sc.transform(x[train]), sc.transform(x[test])
                rate_train = yg[train] / expo[train]
                if family == "poisson":
                    m = PoissonRegressor(alpha=alpha, max_iter=300)
                    m.fit(xtr, rate_train, sample_weight=expo[train])
                    mu_hat[test] = m.predict(xte) * expo[test]
                else:
                    import statsmodels.api as sm

                    mom = yg[train].var() - yg[train].mean()
                    disp = max(mom, 1e-6) / max(yg[train].mean() ** 2, 1e-9)
                    fam = sm.families.NegativeBinomial(alpha=max(disp, 1e-4))
                    fit = sm.GLM(yg[train], sm.add_constant(xtr), family=fam,
                                 offset=np.log(expo[train])).fit_regularized(alpha=alpha, L1_wt=0.0)
                    mu_hat[test] = fit.predict(sm.add_constant(xte), offset=np.log(expo[test]))
                mu_null[test] = expo[test] * (yg[train].sum() / expo[train].sum())
            d_model, d_null = poisson_deviance(yg, mu_hat), poisson_deviance(yg, mu_null)
            if d_null > 0:
                scores[s, g] = 1.0 - d_model / d_null
    per_gene = np.nanmean(scores, axis=0)
    return ProbeResult(
        macro=float(np.nanmean(per_gene)),
        baseline=0.0,
        per_target=per_gene,
        metric=f"deviance_explained_vs_depth_null_{family}",
        n_cells=int(has_counts.sum()),
        n_targets=counts.shape[1],
        seeds=seeds,
        seed_sd=_seed_sd(scores),
        ci95=_ci95(per_gene),
    )


def module_activity_probe(
    embedding: np.ndarray,
    scores: np.ndarray,
    has_counts: np.ndarray,
    *,
    depth: np.ndarray | None = None,
    folds: int = 5,
    seeds: int = 3,
) -> ProbeResult:
    """E9: per-module Pearson r of ridge predictions, macro-averaged. `depth` given makes it the
    depth-residualised variant: targets are residualised with `metrics.residualise` before fitting,
    so a depth-only embedding scores ~0. Baseline: predict the training mean — r = 0 by
    construction.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = embedding[has_counts]
    y = scores[has_counts].astype(np.float64)
    if depth is not None:
        y = residualise(y, _nuisance(depth, has_counts))
        y = np.asarray(y).reshape(len(y), -1)
    rs = np.full((seeds, y.shape[1]), np.nan)
    for s in range(seeds):
        preds = np.zeros_like(y)
        for train, test in _folds(len(x), folds, seed=s):
            pred = make_pipeline(StandardScaler(), _ridge()).fit(x[train], y[train]).predict(x[test])
            preds[test] = np.asarray(pred).reshape(len(test), -1)
        for m in range(y.shape[1]):
            sd_t, sd_p = y[:, m].std(), preds[:, m].std()
            if sd_t > 0 and sd_p > 0:
                rs[s, m] = float(np.corrcoef(y[:, m], preds[:, m])[0, 1])
    per_module, macro, n_scored = _over_scored(rs)
    return ProbeResult(
        macro=macro,
        baseline=0.0,
        per_target=per_module,
        metric="pearson_r" + ("_depth_residualised" if depth is not None else ""),
        n_cells=int(has_counts.sum()),
        n_targets=scores.shape[1],
        seeds=seeds,
        n_scored=n_scored,
        seed_sd=_seed_sd(rs),
    )


def _annotation_reader(name: str, dim: int, knn_k: int):
    """Build one E2 reader. Module-level and keyed by NAME so it survives pickling to a worker."""
    import numpy as np
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import f1_score, make_scorer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "linear":
        return make_pipeline(StandardScaler(), LogisticRegressionCV(
            Cs=np.logspace(-3, 3, 7), max_iter=5000, cv=3,
            scoring=make_scorer(f1_score, average="macro")))
    if name == "knn_cosine":
        return KNeighborsClassifier(n_neighbors=knn_k, metric="cosine")
    if name == "mlp":
        hidden = max(64, int(round(dim * 8 / 3)))
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(hidden, hidden), activation="relu", max_iter=600,
            early_stopping=False, random_state=0))
    raise KeyError(name)


def _annotation_fold(name: str, dim: int, knn_k: int, x, y, train, test):
    """One (reader, seed, fold) fit: the unit of work the probe parallelises over."""
    import numpy as np

    pred = _annotation_reader(name, dim, knn_k).fit(x[train], y[train]).predict(x[test])
    vals, counts = np.unique(y[train], return_counts=True)
    return test, pred, vals[counts.argmax()]


def annotation_probe(
    embedding: np.ndarray,
    labels: np.ndarray,
    *,
    folds: int = 5,
    seeds: int = 3,
    knn_k: int = 10,
    n_jobs: int = -1,
    readers: tuple[str, ...] | None = None,
) -> dict[str, ProbeResult]:
    """E2 (GeneJEPA T1): cell-type identification from the frozen embedding, macro-F1."""
    from sklearn.metrics import f1_score

    has = np.asarray([bool(l) for l in labels])
    x = embedding[has]
    y = np.asarray(labels)[has]
    out: dict[str, ProbeResult] = {}
    from joblib import Parallel, delayed

    dim = int(embedding.shape[1])
    baseline = np.zeros(seeds)
    for name in tuple(readers) if readers else ("linear", "knn_cosine", "mlp"):
        jobs = [(s, tr, te) for s in range(seeds) for tr, te in _folds(len(x), folds, seed=s)]
        done = Parallel(n_jobs=n_jobs, backend="loky", pre_dispatch="all")(
            delayed(_annotation_fold)(name, dim, knn_k, x, y, tr, te) for _, tr, te in jobs)
        f1s = np.zeros(seeds)
        for s in range(seeds):
            preds = np.empty(len(x), dtype=object)
            base = np.empty(len(x), dtype=object)
            for (js, _, _), (test, pred, maj) in zip(jobs, done):
                if js != s:
                    continue
                preds[test] = pred
                base[test] = maj
            f1s[s] = f1_score(y, preds.astype(str), average="macro")
            baseline[s] = f1_score(y, base.astype(str), average="macro")
        out[name] = ProbeResult(
            macro=float(f1s.mean()),
            baseline=float(baseline.mean()),
            per_target=f1s,
            metric="macro_f1",
            n_cells=int(has.sum()),
            n_targets=len(np.unique(y)),
            seeds=seeds,
            seed_sd=_seed_sd(f1s[:, None]),
            ci95=_ci95(f1s),
        )
    return out
