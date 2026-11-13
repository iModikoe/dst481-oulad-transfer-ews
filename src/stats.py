"""Statistical comparison for the pre-specified confirmatory family (§3.8.3).

Two things matter here and both are about honesty rather than technique.
First, the family is fixed at five comparisons in config.yaml before any result
is inspected; nothing else in the study is tested. Second, the resamples that
produce the Tier 3 estimates share training data, so an ordinary paired t-test
understates the variance. The Nadeau-Bengio correction adjusts for that.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def corrected_resampled_ttest(diffs, n_train: int, n_test: int):
    """Nadeau & Bengio corrected resampled t-test.

    `diffs` are per-split differences (e.g. internal minus transferred AUC).
    The variance is inflated by (1/k + n_test/n_train) instead of 1/k to account
    for the dependence induced by overlapping training partitions.
    """
    d = np.asarray([x for x in diffs if np.isfinite(x)], dtype=float)
    k = len(d)
    if k < 2:
        return {"mean": float(d.mean()) if k else np.nan, "t": np.nan, "p": np.nan, "df": k - 1}
    mean = float(d.mean())
    var = float(d.var(ddof=1))
    if var == 0:
        return {"mean": mean, "t": np.inf if mean != 0 else 0.0,
                "p": 0.0 if mean != 0 else 1.0, "df": k - 1}
    ratio = (n_test / n_train) if n_train > 0 else 0.0
    se = np.sqrt(var * (1.0 / k + ratio))
    t = mean / se if se > 0 else np.nan
    p = float(2 * sps.t.sf(abs(t), df=k - 1)) if np.isfinite(t) else np.nan
    return {"mean": mean, "t": float(t), "p": p, "df": k - 1}


def holm_bonferroni(pvals, alpha: float = 0.05):
    """Holm-Bonferroni step-down. Returns (adjusted p, reject) in input order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    adj = np.full(m, np.nan)
    running = 0.0
    for rank, i in enumerate(order):
        if not np.isfinite(p[i]):
            continue
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    reject = np.array([bool(np.isfinite(a) and a <= alpha) for a in adj])
    return adj, reject


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan
    r = sps.spearmanr(x[m], y[m])
    return float(r.statistic), float(r.pvalue)


def spearman_ci(x, y, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 4:
        r, p = spearman(x, y)
        return r, np.nan, np.nan, p
    rng = np.random.default_rng(seed)
    r, p = spearman(x, y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(x[idx])) < 3 or len(np.unique(y[idx])) < 3:
            continue
        rb, _ = spearman(x[idx], y[idx])
        if np.isfinite(rb):
            boots.append(rb)
    if len(boots) < 20:
        return r, np.nan, np.nan, p
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return r, float(lo), float(hi), p


def kendall_tau_b(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3:
        return np.nan
    return float(sps.kendalltau(a, b, variant="b").statistic)


def jaccard_topk(rank_a, rank_b, k: int = 10):
    """Overlap of the top-k sets of two importance rankings (lower rank = more important)."""
    a = set(np.argsort(rank_a)[:k])
    b = set(np.argsort(rank_b)[:k])
    if not a and not b:
        return np.nan
    return float(len(a & b) / len(a | b))
