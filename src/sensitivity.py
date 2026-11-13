"""Pre-specified sensitivity analyses (Table 4.18).

Every analysis here was specified in Chapter 3 before results existed. All are
reported, including those that leave conclusions unchanged — a pre-registered
design that reports only the checks that happened to matter has given up the
advantage of having been pre-registered.

Each variant re-runs Tier 3 with a single model family to keep the cost
proportionate; the model used is recorded in the output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics, models, splits


def _tier3_mean_auc(frame, num_cols, cat_cols, cfg, model_key,
                    enforce_disjoint=True, class_weight=None, log=print):
    pairs = splits.tier3_pairs(frame, enforce_disjoint=enforce_disjoint)
    if not pairs:
        return np.nan, 0
    X = frame[num_cols + cat_cols]
    y = frame["y"].to_numpy()
    groups = frame["id_student"].to_numpy()
    aucs = []
    for pair in pairs:
        tr, te = pair.train_idx, pair.test_idx
        if len(np.unique(y[te])) < 2:
            continue
        pipe, _ = models.fit_with_search(model_key, X.iloc[tr], y[tr], groups[tr],
                                         num_cols, cat_cols, cfg, class_weight)
        p = models.predict_proba(pipe, X.iloc[te])
        aucs.append(metrics.auc_fn(y[te], p))
    return (float(np.mean(aucs)) if aucs else np.nan), len(aucs)


def run(full_frame, base_frame, num_cols, cat_cols, cfg, baseline_auc,
        model_key="xgb", log=print) -> pd.DataFrame:
    rows = []

    def record(name, spec, auc, n, note):
        rows.append({
            "analysis": name, "specified_in": spec, "model": models.MODEL_LABELS[model_key],
            "tier3_mean_roc_auc": auc, "n_pairs": n,
            "change_vs_primary": (auc - baseline_auc) if np.isfinite(auc) else np.nan,
            "direction_preserved": bool(np.isfinite(auc) and abs(auc - baseline_auc) < 0.05),
            "note": note,
        })

    # 1. Withdrawn excluded from the at-risk label (§3.6.1)
    log("  sensitivity 1/5: withdrawn cases excluded from the label")
    f1 = base_frame[base_frame["final_result"] != "Withdrawn"].reset_index(drop=True)
    a, n = _tier3_mean_auc(f1, num_cols, cat_cols, cfg, model_key, log=log)
    record("Withdrawn cases excluded from the at-risk label", "§3.6.1", a, n,
           "Any conclusion not surviving this check is reported as conditional.")

    # 2. Early deregistrations retained (§3.6.1) — relaxes the estimand
    log("  sensitivity 2/5: early deregistrations retained")
    a, n = _tier3_mean_auc(full_frame.reset_index(drop=True), num_cols, cat_cols,
                           cfg, model_key, log=log)
    record("Early deregistrations retained (day-28 rule relaxed)", "§3.6.1", a, n,
           "Quantifies the cost of the restricted estimand. A large gain here is "
           "expected and is trivial separation on absent engagement data, not skill.")

    # 3. Person-level overlap rule disabled (§3.8.1)
    log("  sensitivity 3/5: person-level overlap rule disabled")
    a, n = _tier3_mean_auc(base_frame, num_cols, cat_cols, cfg, model_key,
                           enforce_disjoint=False, log=log)
    record("Person-level overlap rule disabled", "§3.8.1", a, n,
           "The difference estimates how much apparent transfer performance in "
           "comparable published work may reflect repeated individuals.")

    # 4. Alternative imbalance treatment (§3.6.3)
    log("  sensitivity 4/5: class weighting")
    a, n = _tier3_mean_auc(base_frame, num_cols, cat_cols, cfg, model_key,
                           class_weight="balanced", log=log)
    record("Alternative imbalance treatment (class weighting)", "§3.6.3", a, n,
           "Imbalance is a controlled sensitivity analysis, not a research question.")

    # 5. Sensitive attributes used predictively (§3.9.2) — single labelled ablation
    log("  sensitivity 5/5: sensitive attributes dropped from the predictors")
    drop = {"gender", "imd_decile", "imd_missing"}
    nc = [c for c in num_cols if c not in drop]
    cc = [c for c in cat_cols if c not in drop]
    a, n = _tier3_mean_auc(base_frame, nc, cc, cfg, model_key, log=log)
    record("Sensitive attributes removed from the predictors", "§3.9.2", a, n,
           "Labelled ablation: the accuracy cost of excluding sensitive attributes.")

    return pd.DataFrame(rows)


def imbalance_selection(frame, num_cols, cat_cols, cfg, model_key="xgb", log=print) -> pd.DataFrame:
    """Table 4.6 — treatment comparison at Tiers 1 and 2, resampling inside folds."""
    rows = []
    X = frame[num_cols + cat_cols]
    y = frame["y"].to_numpy()
    groups = frame["id_student"].to_numpy()
    folds = splits.tier1_folds(frame, int(cfg["models"]["outer_folds"]),
                               int(cfg["models"]["seed"]))
    t2 = splits.tier2_pairs(frame)

    for label, cw in [("No treatment", None), ("Class weighting", "balanced")]:
        oof = np.full(len(frame), np.nan)
        for tr, te in folds:
            pipe, _ = models.fit_with_search(model_key, X.iloc[tr], y[tr], groups[tr],
                                             num_cols, cat_cols, cfg, cw)
            oof[te] = models.predict_proba(pipe, X.iloc[te])
        m = np.isfinite(oof)
        t1_auc = metrics.auc_fn(y[m], oof[m]) if len(np.unique(y[m])) > 1 else np.nan
        t1_pr = metrics.aucpr_fn(y[m], oof[m]) if len(np.unique(y[m])) > 1 else np.nan

        t2_aucs, t2_prs = [], []
        for pair in t2:
            tr, te = pair.train_idx, pair.test_idx
            if len(np.unique(y[te])) < 2:
                continue
            pipe, _ = models.fit_with_search(model_key, X.iloc[tr], y[tr], groups[tr],
                                             num_cols, cat_cols, cfg, cw)
            p = models.predict_proba(pipe, X.iloc[te])
            t2_aucs.append(metrics.auc_fn(y[te], p))
            t2_prs.append(metrics.aucpr_fn(y[te], p))
        rows.append({
            "treatment": label, "model": models.MODEL_LABELS[model_key],
            "tier1_aucpr": t1_pr, "tier1_roc_auc": t1_auc,
            "tier2_aucpr": float(np.mean(t2_prs)) if t2_prs else np.nan,
            "tier2_roc_auc": float(np.mean(t2_aucs)) if t2_aucs else np.nan,
            "n_tier2_pairs": len(t2_aucs),
        })
    df = pd.DataFrame(rows)
    if len(df):
        best = df["tier2_roc_auc"].idxmax()
        df["carried_forward"] = [i == best for i in df.index]
    return df
