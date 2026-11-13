"""Performance, calibration and interval estimation (Chapter 3 §3.8.2).

Every reported quantity carries a bootstrap interval; the study's primary
construct is a *change*, so `paired_delta_ci` resamples the two evaluations
on a common index where one exists.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def recall_at_budget(y_true, y_score, budget: float) -> float:
    """Recall when the top `budget` fraction of predicted risk is flagged.

    This is the operational quantity: a fixed alert budget expresses finite
    intervention capacity (§3.8.2) and keeps the decision constant across
    contexts of differing prevalence (§3.9.1).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_score)
    if n == 0 or y_true.sum() == 0:
        return np.nan
    k = max(1, int(np.ceil(budget * n)))
    idx = np.argsort(-y_score, kind="mergesort")[:k]
    return float(y_true[idx].sum() / y_true.sum())


def budget_threshold(y_score, budget: float) -> float:
    """The score cut-point that flags the top `budget` fraction."""
    y_score = np.asarray(y_score, dtype=float)
    if len(y_score) == 0:
        return np.nan
    return float(np.quantile(y_score, 1.0 - budget))


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """ECE over equal-frequency bins (§3.8.2)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_prob) < n_bins:
        return np.nan
    edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(y_prob, edges[1:-1], right=True)
    ece, n = 0.0, len(y_prob)
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


def brier_decomposition(y_true, y_prob, n_bins: int = 10) -> tuple[float, float, float]:
    """Return (brier, calibration component, refinement component).

    Murphy's two-component decomposition: Brier = calibration + refinement,
    where refinement = uncertainty - resolution.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    brier = float(np.mean((y_prob - y_true) ** 2))
    if len(y_prob) < n_bins:
        return brier, np.nan, np.nan
    edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(y_prob, edges[1:-1], right=True)
    base = y_true.mean()
    n = len(y_prob)
    cal, res = 0.0, 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        w = m.sum() / n
        obs, pred = y_true[m].mean(), y_prob[m].mean()
        cal += w * (pred - obs) ** 2
        res += w * (obs - base) ** 2
    unc = base * (1 - base)
    return brier, float(cal), float(unc - res)


def core_metrics(y_true, y_prob, budgets=(0.10, 0.20), n_bins: int = 10) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    out = {"n": int(len(y_true)), "prevalence": float(np.mean(y_true)) if len(y_true) else np.nan}
    if len(np.unique(y_true)) < 2:
        for k in ["roc_auc", "aucpr", "brier", "brier_calibration", "brier_refinement", "ece"]:
            out[k] = np.nan
        for b in budgets:
            out[f"recall_at_{int(b*100)}"] = np.nan
        return out
    out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    out["aucpr"] = float(average_precision_score(y_true, y_prob))
    br, cal, ref = brier_decomposition(y_true, y_prob, n_bins)
    out["brier"], out["brier_calibration"], out["brier_refinement"] = br, cal, ref
    out["ece"] = expected_calibration_error(y_true, y_prob, n_bins)
    for b in budgets:
        out[f"recall_at_{int(b*100)}"] = recall_at_budget(y_true, y_prob, b)
    return out


def bootstrap_ci(y_true, y_prob, fn, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap interval for any scalar metric fn(y_true, y_prob)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    if n == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    point = fn(y_true, y_prob)
    stats = np.empty(n_boot, dtype=float)
    stats[:] = np.nan
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            stats[i] = fn(yt, y_prob[idx])
        except ValueError:
            continue
    good = stats[~np.isnan(stats)]
    if len(good) < 20:
        return float(point), np.nan, np.nan
    lo, hi = np.quantile(good, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def paired_delta_ci(y_a, p_a, y_b, p_b, fn, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Interval for fn(A) - fn(B) where A and B are different evaluation sets.

    The two sets are resampled independently, which is correct here because
    internal and transferred evaluations are computed on disjoint registrations.
    """
    y_a, p_a = np.asarray(y_a), np.asarray(p_a, dtype=float)
    y_b, p_b = np.asarray(y_b), np.asarray(p_b, dtype=float)
    rng = np.random.default_rng(seed)
    point = fn(y_a, p_a) - fn(y_b, p_b)
    deltas = []
    for _ in range(n_boot):
        ia = rng.integers(0, len(y_a), len(y_a))
        ib = rng.integers(0, len(y_b), len(y_b))
        if len(np.unique(y_a[ia])) < 2 or len(np.unique(y_b[ib])) < 2:
            continue
        try:
            deltas.append(fn(y_a[ia], p_a[ia]) - fn(y_b[ib], p_b[ib]))
        except ValueError:
            continue
    if len(deltas) < 20:
        return float(point), np.nan, np.nan
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def auc_fn(y, p):
    return roc_auc_score(y, p)


def aucpr_fn(y, p):
    return average_precision_score(y, p)
