"""Leakage checks that go beyond chance-level baselines.

A majority-class baseline at ROC-AUC 0.500 shows only that a classifier ignoring
the features cannot beat chance. It says nothing about whether some *feature*
encodes information from after the day-27 cutoff. These three checks target that
gap from different directions, and none of them is sufficient alone:

1. `assert_no_outcome_columns` - structural. Outcome-derived and future-dated
   columns must never appear in the design matrix. Cheap, exact, always on.
2. `univariate_audit` - a single feature that separates the outcome almost
   perfectly is the signature of leakage. Flags any feature above a threshold.
   Cheap (no refit), always on.
3. `permutation_check` - shuffle the labels and refit. Above-chance performance
   on shuffled labels means the leak is in the pipeline mechanics (preprocessing
   fitted outside the fold, fold assignment correlated with the label), not in
   any one feature. Costs one extra Tier 1 pass, so it sits behind a flag.

What none of these can do: confirm that a feature which *is* available by day 27
carries no information the institution would not have had at day 27. That is a
question about the recording system, not about this code, and §3.6.2 answers it
with the per-feature availability rule rather than with a test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Columns that encode the outcome or are stamped after the prediction horizon.
# `date_unregistration` defines the estimand (§3.6.1) and is therefore read, but
# it describes a future event and must never become a predictor.
FORBIDDEN = {
    "final_result", "y", "y_t4", "date_unregistration", "excluded_early_dereg",
    "score", "date_submitted",
}


def assert_no_outcome_columns(num_cols, cat_cols) -> None:
    """Raise if any outcome-derived or future-dated column reached the features."""
    used = set(num_cols) | set(cat_cols)
    bad = sorted(used & FORBIDDEN)
    if bad:
        raise AssertionError(
            "Outcome-derived or future-dated columns reached the design matrix: "
            f"{bad}. This is label leakage, not a tuning problem. Fix "
            "features.feature_columns before running anything else."
        )
    # `first_score` is legitimate only because assessment_features filters on
    # both the due date and the submission date; flag it if that ever changes.
    return None


def univariate_audit(frame: pd.DataFrame, num_cols, cat_cols,
                     flag_threshold: float = 0.90) -> pd.DataFrame:
    """Per-feature univariate ROC-AUC against the outcome.

    A feature scoring near 1.0 on its own is almost always leakage: genuine early
    engagement signals on educational data sit well below that. The threshold is
    a flag for inspection, not a hard rule - interpret a flagged feature by
    asking what it is made of and when it was recorded.
    """
    y = frame["y"].to_numpy()
    rows = []
    for c in list(num_cols) + list(cat_cols):
        if c not in frame.columns:
            continue
        s = frame[c]
        if c in set(cat_cols) or not pd.api.types.is_numeric_dtype(s):
            # Encode each category by its outcome rate, which is the most
            # generous reading - if even that cannot separate, the column is safe.
            rate = frame.groupby(s.astype(str), observed=True)["y"].transform("mean")
            v = pd.to_numeric(rate, errors="coerce").to_numpy()
        else:
            v = pd.to_numeric(s, errors="coerce").to_numpy()
        m = np.isfinite(v)
        if m.sum() < 50 or len(np.unique(y[m])) < 2:
            rows.append({"feature": c, "univariate_auc": np.nan,
                         "n_used": int(m.sum()), "flagged": False})
            continue
        auc = roc_auc_score(y[m], v[m])
        auc = max(auc, 1.0 - auc)           # direction-free separability
        rows.append({"feature": c, "univariate_auc": float(auc),
                     "n_used": int(m.sum()), "flagged": bool(auc >= flag_threshold)})
    out = pd.DataFrame(rows).sort_values("univariate_auc", ascending=False)
    return out.reset_index(drop=True)


def permutation_check(frame, num_cols, cat_cols, cfg, model_key="logreg",
                      n_repeats: int = 3, log=print) -> pd.DataFrame:
    """Refit on shuffled labels; AUC should collapse to ~0.5.

    Labels are permuted *within* student groups' absence - i.e. globally - after
    which the grouped fold structure is rebuilt, so any above-chance result
    implicates the pipeline rather than the data.
    """
    from . import models, splits

    rows = []
    rng = np.random.default_rng(int(cfg["models"]["seed"]))
    X = frame[num_cols + cat_cols]
    groups = frame["id_student"].to_numpy()

    for rep in range(n_repeats):
        shuffled = frame.copy()
        shuffled["y"] = rng.permutation(frame["y"].to_numpy())
        y = shuffled["y"].to_numpy()
        folds = splits.tier1_folds(shuffled, int(cfg["models"]["outer_folds"]),
                                   int(cfg["models"]["seed"]) + rep)
        oof = np.full(len(frame), np.nan)
        for tr, te in folds:
            pipe, _ = models.fit_with_search(model_key, X.iloc[tr], y[tr], groups[tr],
                                             num_cols, cat_cols, cfg)
            oof[te] = models.predict_proba(pipe, X.iloc[te])
        m = np.isfinite(oof)
        auc = roc_auc_score(y[m], oof[m]) if len(np.unique(y[m])) > 1 else np.nan
        log(f"    permutation {rep + 1}/{n_repeats}: ROC-AUC = {auc:.4f}")
        rows.append({"repeat": rep + 1, "model": models.MODEL_LABELS[model_key],
                     "shuffled_roc_auc": auc,
                     "within_tolerance": bool(np.isfinite(auc) and abs(auc - 0.5) < 0.05)})
    df = pd.DataFrame(rows)
    if not df["within_tolerance"].all():
        log("    WARNING: shuffled-label AUC departs from 0.5. Do not report this "
            "run until the cause is found - this indicates leakage in the "
            "pipeline mechanics, not in any single feature.")
    return df
