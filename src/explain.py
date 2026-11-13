"""Explanation stability across contexts (SRQ4, §3.10).

The primary analysis holds the fitted model constant: one Tier 3 source model
is scored on each held-out target in turn, so divergence in the resulting
ranking reflects context alone. The secondary analysis compares the different
leave-one-module-out models with one another, where divergence reflects model
and context variation jointly — these are reported separately and never pooled,
because conflating them would mix model instability with context instability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import kendall_tau_b, jaccard_topk


def _feature_names(pipeline, fallback_n: int) -> list[str]:
    try:
        return list(pipeline.named_steps["prep"].get_feature_names_out())
    except Exception:  # pragma: no cover
        return [f"f{i}" for i in range(fallback_n)]


def global_shap_importance(pipeline, X: pd.DataFrame, max_rows: int = 4000,
                           seed: int = 0) -> pd.Series:
    """Mean absolute SHAP value per transformed feature.

    Uses the tree explainer where the final estimator is tree-based; falls back
    to absolute standardised coefficients for logistic regression, which §3.10
    reports alongside SHAP.
    """
    import shap

    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    Xt = prep.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    names = _feature_names(pipeline, Xt.shape[1])

    if Xt.shape[0] > max_rows:
        rng = np.random.default_rng(seed)
        Xt = Xt[rng.choice(Xt.shape[0], max_rows, replace=False)]

    if hasattr(clf, "coef_"):
        vals = np.abs(np.ravel(clf.coef_)) * Xt.std(axis=0)
        return pd.Series(vals, index=names).sort_values(ascending=False)

    try:
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xt, check_additivity=False)
        if isinstance(sv, list):
            sv = sv[-1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[..., -1]
        vals = np.abs(sv).mean(axis=0)
    except Exception:  # pragma: no cover - explainer fallback
        vals = getattr(clf, "feature_importances_", np.zeros(Xt.shape[1]))
    return pd.Series(np.ravel(vals)[: len(names)], index=names).sort_values(ascending=False)


def _ranks(s: pd.Series, index) -> np.ndarray:
    """Rank positions (1 = most important) aligned to a shared feature index."""
    return s.reindex(index).fillna(0.0).rank(ascending=False, method="average").to_numpy()


def concordance(importances: dict[str, pd.Series], k: int = 10) -> pd.DataFrame:
    """Pairwise concordance between context-specific importance rankings."""
    keys = list(importances)
    index = sorted(set().union(*[set(s.index) for s in importances.values()]))
    rows = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ra, rb = _ranks(importances[a], index), _ranks(importances[b], index)
            rows.append({
                "context_a": a, "context_b": b,
                "kendall_tau_b": kendall_tau_b(ra, rb),
                "spearman_rho": float(pd.Series(ra).corr(pd.Series(rb), method="spearman")),
                "top10_jaccard": jaccard_topk(ra, rb, k),
            })
    return pd.DataFrame(rows)


def vs_reference(importances: dict[str, pd.Series], reference: str, k: int = 10) -> pd.DataFrame:
    """Each context's ranking against a single reference ranking."""
    index = sorted(set().union(*[set(s.index) for s in importances.values()]))
    ref = _ranks(importances[reference], index)
    rows = []
    for name, s in importances.items():
        if name == reference:
            continue
        r = _ranks(s, index)
        rows.append({
            "target_context": name,
            "kendall_tau_b": kendall_tau_b(ref, r),
            "spearman_rho": float(pd.Series(ref).corr(pd.Series(r), method="spearman")),
            "top10_jaccard": jaccard_topk(ref, r, k),
        })
    return pd.DataFrame(rows)


def feature_rank_stability(importances: dict[str, pd.Series], top_n: int = 15) -> pd.DataFrame:
    """Per-feature rank mean, SD, min and max across contexts (tests H4)."""
    index = sorted(set().union(*[set(s.index) for s in importances.values()]))
    mat = pd.DataFrame({name: _ranks(s, index) for name, s in importances.items()}, index=index)
    out = pd.DataFrame({
        "mean_rank": mat.mean(axis=1),
        "rank_sd": mat.std(axis=1),
        "min_rank": mat.min(axis=1),
        "max_rank": mat.max(axis=1),
    }).sort_values("mean_rank")
    out["block"] = [_block_of(f) for f in out.index]
    return out.head(top_n).reset_index(names="feature")


ENGAGEMENT_TOKENS = ("click", "active", "distinct", "slope", "wk", "events", "days_since", "pre_")
DEMOGRAPHIC_TOKENS = ("gender", "region", "age", "imd", "disability", "edu", "credits", "attempts")


def _block_of(feature: str) -> str:
    f = feature.lower()
    if any(t in f for t in ENGAGEMENT_TOKENS):
        return "early engagement"
    if any(t in f for t in ("score", "assessment", "submitted")):
        return "early assessment"
    if any(t in f for t in DEMOGRAPHIC_TOKENS):
        return "demographic/enrolment"
    return "other"


def block_stability(stability: pd.DataFrame) -> pd.DataFrame:
    """Mean rank SD by feature block — the comparison H4 actually predicts."""
    return (stability.groupby("block", observed=True)["rank_sd"]
            .agg(["mean", "count"]).reset_index()
            .rename(columns={"mean": "mean_rank_sd", "count": "n_features"}))
