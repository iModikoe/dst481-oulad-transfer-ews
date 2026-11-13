"""Tier 4: external robustness check on a related construct (§3.5.2).

This is deliberately the most constrained part of the study. The two datasets do
not share an outcome construct or a time point, so Tier 4 is restricted to
enrolment-block predictors genuinely available in both, under the narrower
non-completion label. It is reported as a robustness check, never as a
replication of the day-28 task, and no conclusion about the primary research
question rests on it.

The harmonisation map below is small because the honest overlap IS small. That
is itself a finding about cross-institutional transfer evidence: §2.11 argues
the precondition for such studies is aligned variables, and here it barely holds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data_load as dl
from . import metrics, models

# Only variables with a defensible common meaning at or before registration.
#
# This map is deliberately short, and the omissions matter more than the entries.
# OULAD's `num_of_prev_attempts` counts prior attempts at the same module; UCI's
# "Previous qualification" is a nominal code for the qualification held before
# entry. They are different constructs and are NOT paired here, even though both
# are nominally "prior education" fields — pairing them would manufacture an
# equivalence the data does not support. UCI's code is nominal rather than
# ordered by level, so it cannot be aligned with OULAD's ordinal
# `highest_education` either.
#
# The consequence is that exactly two variables harmonise. That is a finding
# about the feasibility of cross-institutional transfer evidence, not a defect
# in this implementation, and §3.5.2 and §2.11 should both reflect it.
HARMONISED = {
    "gender": "Gender",
    "age_at_enrolment": "Age at enrollment",
}


def build_uci(cfg) -> pd.DataFrame | None:
    """Return the UCI target frame on the harmonised schema, or None if absent."""
    try:
        uci = dl.load_uci(cfg["paths"]["uci"])
    except FileNotFoundError:
        return None

    target_col = next((c for c in uci.columns if c.strip().lower() == "target"), None)
    if target_col is None:
        raise KeyError(f"No 'Target' column in the UCI file; found {list(uci.columns)[:8]}...")

    pos = set(cfg["labels"]["uci_positive"])
    neg = set(cfg["labels"]["uci_negative"])
    keep = uci[uci[target_col].isin(pos | neg)].copy()
    keep["y"] = keep[target_col].isin(pos).astype(int)

    out = pd.DataFrame(index=keep.index)
    gcol = next((c for c in keep.columns if c.strip().lower() == "gender"), None)
    # UCI codes gender 1=male, 0=female
    out["gender"] = np.where(pd.to_numeric(keep[gcol], errors="coerce") == 1, "M", "F") \
        if gcol else "M"
    acol = next((c for c in keep.columns if "age at enrollment" in c.strip().lower()), None)
    out["age_at_enrolment"] = pd.to_numeric(keep[acol], errors="coerce") if acol else np.nan
    out["y"] = keep["y"].to_numpy()
    return out.reset_index(drop=True)


def oulad_tier4_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """OULAD side on the same harmonised schema, under the Tier 4 label.

    Fail is excluded here (and only here): the secondary dataset has no category
    for a student who fails but remains enrolled, so including Fail would compare
    unlike things.
    """
    df = frame[frame["y_t4"].notna()].copy()
    age_mid = {"0-35": 27.0, "35-55": 45.0, "55<=": 60.0}
    out = pd.DataFrame({
        "gender": df["gender"].astype(str).str.upper().str[0],
        "age_at_enrolment": df["age_band"].map(age_mid),
        "y": df["y_t4"].astype(int).to_numpy(),
    })
    out["_student"] = df["id_student"].to_numpy()
    return out.reset_index(drop=True)


def run(frame, cfg, log=print) -> list[dict]:
    uci = build_uci(cfg)
    if uci is None:
        log("  Tier 4 skipped: UCI file not present (see README section 1).")
        return []

    src = oulad_tier4_frame(frame)
    if src["y"].nunique() < 2 or uci["y"].nunique() < 2:
        log("  Tier 4 skipped: one side has a single outcome class.")
        return []

    num_cols = ["age_at_enrolment"]
    cat_cols = ["gender"]
    budgets = tuple(cfg["evaluation"]["alert_budgets"])
    n_pred = len(num_cols) + len(cat_cols)
    log(f"  Tier 4: {len(src)} source rows -> {len(uci)} target rows on "
        f"{n_pred} harmonised predictors ({', '.join(num_cols + cat_cols)})")
    log("  NOTE: only two variables harmonise across the two datasets. Tier 4 is a "
        "weak, low-powered check on a related construct and must be reported as such; "
        "no primary conclusion may rest on it.")

    rows = []
    for name in models.LEARNERS:
        pipe, _ = models.fit_with_search(
            name, src[num_cols + cat_cols], src["y"].to_numpy(),
            src["_student"].to_numpy(), num_cols, cat_cols, cfg)
        p = models.predict_proba(pipe, uci[num_cols + cat_cols])
        r = metrics.core_metrics(uci["y"].to_numpy(), p, budgets,
                                 int(cfg["evaluation"]["calibration_bins"]))
        r.update({
            "model": models.MODEL_LABELS[name], "model_key": name, "tier": 4,
            "pair": "OULAD enrolment block -> UCI Dropout",
            "differs_by": "dataset",
            "n_harmonised_predictors": len(num_cols) + len(cat_cols),
            "construct": "related, not equivalent — enrolment-time non-completion",
        })
        rows.append(r)
    return rows
