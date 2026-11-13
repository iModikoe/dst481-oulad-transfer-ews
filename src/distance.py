"""Source-target distance measures for SRQ2 (§3.8.3).

Computed for every pair *before* results are inspected, following the
distance-based logic of Ding, et al. (2019), who use proxy A-distance between
source and target feature distributions together with the target-to-source
sample-size ratio to explain transfer performance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def _bin_edges(x: np.ndarray, n_bins: int = 10) -> np.ndarray:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.array([-np.inf, np.inf])
    qs = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(qs)
    if len(edges) < 2:
        return np.array([-np.inf, np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _hist(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.full(len(edges) - 1, 1.0 / max(1, len(edges) - 1))
    counts, _ = np.histogram(x, bins=edges)
    p = counts.astype(float)
    total = p.sum()
    if total == 0:
        return np.full(len(p), 1.0 / len(p))
    return p / total


def psi(source: np.ndarray, target: np.ndarray, n_bins: int = 10) -> float:
    """Population stability index. Bins are defined on the source."""
    edges = _bin_edges(np.asarray(source, dtype=float), n_bins)
    p = _hist(np.asarray(source, dtype=float), edges) + EPS
    q = _hist(np.asarray(target, dtype=float), edges) + EPS
    return float(np.sum((q - p) * np.log(q / p)))


def jensen_shannon(source: np.ndarray, target: np.ndarray, n_bins: int = 10) -> float:
    """Jensen-Shannon divergence (base 2, so bounded in [0, 1])."""
    edges = _bin_edges(np.asarray(source, dtype=float), n_bins)
    p = _hist(np.asarray(source, dtype=float), edges) + EPS
    q = _hist(np.asarray(target, dtype=float), edges) + EPS
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        return float(np.sum(a * np.log2(a / b)))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def pair_distances(X_src: pd.DataFrame, X_tgt: pd.DataFrame,
                   y_src, y_tgt, top_features=None, n_bins: int = 10) -> dict:
    """All five distance measures of §3.8.3 for one source-target pair."""
    numeric = [c for c in X_src.columns
               if pd.api.types.is_numeric_dtype(X_src[c]) and c in X_tgt.columns]
    psis, jsds = [], []
    top = [c for c in (top_features or []) if c in numeric] or numeric[:10]
    for c in numeric:
        s = pd.to_numeric(X_src[c], errors="coerce").to_numpy(dtype=float)
        t = pd.to_numeric(X_tgt[c], errors="coerce").to_numpy(dtype=float)
        if np.all(~np.isfinite(s)) or np.all(~np.isfinite(t)):
            continue
        psis.append(psi(s, t, n_bins))
    for c in top:
        s = pd.to_numeric(X_src[c], errors="coerce").to_numpy(dtype=float)
        t = pd.to_numeric(X_tgt[c], errors="coerce").to_numpy(dtype=float)
        if np.all(~np.isfinite(s)) or np.all(~np.isfinite(t)):
            continue
        jsds.append(jensen_shannon(s, t, n_bins))
    prev_s = float(np.mean(y_src)) if len(y_src) else np.nan
    prev_t = float(np.mean(y_tgt)) if len(y_tgt) else np.nan
    return {
        "mean_psi": float(np.mean(psis)) if psis else np.nan,
        "max_psi": float(np.max(psis)) if psis else np.nan,
        "jsd_top10": float(np.mean(jsds)) if jsds else np.nan,
        "abs_prevalence_diff": abs(prev_s - prev_t) if np.isfinite(prev_s) and np.isfinite(prev_t) else np.nan,
        "source_target_size_ratio": (len(y_src) / len(y_tgt)) if len(y_tgt) else np.nan,
        "source_prevalence": prev_s,
        "target_prevalence": prev_t,
    }
