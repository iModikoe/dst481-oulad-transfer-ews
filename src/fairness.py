"""Subgroup fairness audit under transfer (§3.9).

Three design commitments are enforced in code rather than left to the analyst:

1. The decision rule is fixed *before* any fairness quantity is computed — a
   fixed alert budget flagging the top 20% of predicted risk in the target
   context. Holding the operational decision constant is what makes disparities
   comparable across contexts of differing prevalence.
2. Groups are defined against an external national boundary (IMD 40%), applied
   unchanged to every target, so the groups mean the same thing in each context.
3. A subgroup is evaluated only where it holds >=100 members and >=30 positives.
   Below that, rates are too unstable to interpret and the cell is reported as
   non-evaluable rather than estimated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import budget_threshold


def deprivation_group(imd_decile: pd.Series, floor: float = 40.0) -> pd.Series:
    """'protected' below the national 40% boundary, 'reference' at or above it,
    'missing' where the band is absent. Missing is labelled, never imputed —
    imputing a deprivation proxy would fabricate the attribute being audited."""
    d = pd.to_numeric(imd_decile, errors="coerce")
    out = pd.Series(np.where(d < floor, "protected", "reference"), index=imd_decile.index,
                    dtype=object)
    out[d.isna()] = "missing"
    return out


def gender_group(gender: pd.Series) -> pd.Series:
    """Male is the reference group: a labelling convention that fixes the sign of
    EOD, not a claim about advantage."""
    g = gender.astype(str).str.upper().str[0]
    return pd.Series(np.where(g == "M", "reference", np.where(g == "F", "protected", "missing")),
                     index=gender.index, dtype=object)


def evaluable(y, group, which, min_n: int, min_pos: int) -> bool:
    m = np.asarray(group) == which
    return bool(m.sum() >= min_n and np.asarray(y)[m].sum() >= min_pos)


def tpr(y, flagged) -> float:
    y = np.asarray(y)
    f = np.asarray(flagged)
    pos = y == 1
    return float(f[pos].mean()) if pos.sum() else np.nan


def fpr(y, flagged) -> float:
    y = np.asarray(y)
    f = np.asarray(flagged)
    neg = y == 0
    return float(f[neg].mean()) if neg.sum() else np.nan


def eod_at_budget(y, scores, group, budget: float = 0.20, threshold: float | None = None):
    """Equal opportunity difference = TPR(reference) - TPR(protected).

    Positive means the protected group is under-detected. `threshold` allows the
    source-fitted (transferred) rule to be applied unchanged at the target,
    which is the realistic deployment condition of §3.9.2.
    """
    y = np.asarray(y)
    s = np.asarray(scores, dtype=float)
    g = np.asarray(group)
    thr = budget_threshold(s, budget) if threshold is None else threshold
    flagged = s >= thr
    ref_m, pro_m = g == "reference", g == "protected"
    t_ref = tpr(y[ref_m], flagged[ref_m]) if ref_m.sum() else np.nan
    t_pro = tpr(y[pro_m], flagged[pro_m]) if pro_m.sum() else np.nan
    return {
        "threshold": float(thr),
        "tpr_reference": t_ref,
        "tpr_protected": t_pro,
        "eod": (t_ref - t_pro) if np.isfinite(t_ref) and np.isfinite(t_pro) else np.nan,
        "fpr_difference": (fpr(y[ref_m], flagged[ref_m]) - fpr(y[pro_m], flagged[pro_m]))
        if ref_m.sum() and pro_m.sum() else np.nan,
        "n_reference": int(ref_m.sum()),
        "n_protected": int(pro_m.sum()),
    }


def eod_ci(y, scores, group, budget=0.20, threshold=None, n_boot=2000, seed=0, alpha=0.05):
    y, s, g = np.asarray(y), np.asarray(scores, dtype=float), np.asarray(group)
    point = eod_at_budget(y, s, g, budget, threshold)["eod"]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        v = eod_at_budget(y[idx], s[idx], g[idx], budget, threshold)["eod"]
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 20:
        return point, np.nan, np.nan
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def group_thresholds_equalising_tpr(y, scores, group, target_tpr: float):
    """Post-processing mitigation: per-group cut-points that equalise TPR."""
    y, s, g = np.asarray(y), np.asarray(scores, dtype=float), np.asarray(group)
    thr = {}
    for grp in ("reference", "protected"):
        m = (g == grp) & (y == 1)
        if m.sum() == 0:
            thr[grp] = np.nan
            continue
        thr[grp] = float(np.quantile(s[m], 1.0 - target_tpr))
    return thr


def apply_group_thresholds(scores, group, thr: dict) -> np.ndarray:
    s, g = np.asarray(scores, dtype=float), np.asarray(group)
    flagged = np.zeros(len(s), dtype=bool)
    for grp, t in thr.items():
        if not np.isfinite(t):
            continue
        flagged |= (g == grp) & (s >= t)
    return flagged


def mitigation_report(y_src, s_src, g_src, y_tgt, s_tgt, g_tgt, budget=0.20):
    """Transferred versus oracle mitigation (§3.9.2).

    The transferred condition fits group thresholds on *source* validation data
    and applies them unchanged at the target — the realistic deployment case.
    The oracle condition fits them on the target and bounds what is achievable.
    The gap between the two is how much of the fairness benefit is itself
    non-transferable, a quantity the reviewed literature does not report.
    """
    base = eod_at_budget(y_tgt, s_tgt, g_tgt, budget)
    overall_tpr_tgt = tpr(y_tgt, np.asarray(s_tgt, dtype=float) >= base["threshold"])

    thr_src = group_thresholds_equalising_tpr(y_src, s_src, g_src, overall_tpr_tgt)
    flag_tr = apply_group_thresholds(s_tgt, g_tgt, thr_src)
    ref, pro = np.asarray(g_tgt) == "reference", np.asarray(g_tgt) == "protected"
    eod_tr = (tpr(np.asarray(y_tgt)[ref], flag_tr[ref])
              - tpr(np.asarray(y_tgt)[pro], flag_tr[pro]))

    thr_or = group_thresholds_equalising_tpr(y_tgt, s_tgt, g_tgt, overall_tpr_tgt)
    flag_or = apply_group_thresholds(s_tgt, g_tgt, thr_or)
    eod_or = (tpr(np.asarray(y_tgt)[ref], flag_or[ref])
              - tpr(np.asarray(y_tgt)[pro], flag_or[pro]))

    red_tr = abs(base["eod"]) - abs(eod_tr) if np.isfinite(base["eod"]) else np.nan
    red_or = abs(base["eod"]) - abs(eod_or) if np.isfinite(base["eod"]) else np.nan
    non_transferable = ((red_or - red_tr) / red_or) if (np.isfinite(red_or) and red_or > 0) else np.nan

    # The share can exceed 1, and that is substantive rather than a defect: it
    # means the source-fitted thresholds made disparity WORSE at the target than
    # no mitigation at all. Reported as an explicit verdict so a bare ratio above
    # 1 is not mistaken for a broken proportion.
    if not np.isfinite(non_transferable):
        verdict = "not evaluable (no oracle improvement available)"
    elif non_transferable > 1:
        verdict = "mitigation harmful under transfer"
    elif non_transferable > 0.5:
        verdict = "majority of achievable benefit is non-transferable"
    elif non_transferable > 0:
        verdict = "most of achievable benefit survives transfer"
    else:
        verdict = "transferred mitigation matches or exceeds the oracle"

    return {
        "mitigation_verdict": verdict,
        "eod_unmitigated": base["eod"],
        "eod_transferred_threshold": float(eod_tr) if np.isfinite(eod_tr) else np.nan,
        "eod_oracle_threshold": float(eod_or) if np.isfinite(eod_or) else np.nan,
        "eod_reduction_transferred": red_tr,
        "eod_reduction_oracle": red_or,
        "non_transferable_share_of_benefit": non_transferable,
        "alert_rate_transferred": float(flag_tr.mean()),
        "alert_rate_oracle": float(flag_or.mean()),
        "recall_transferred": tpr(y_tgt, flag_tr),
        "recall_oracle": tpr(y_tgt, flag_or),
        "recall_unmitigated": overall_tpr_tgt,
    }
