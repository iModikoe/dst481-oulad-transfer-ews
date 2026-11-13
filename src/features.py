"""Assemble the modelling frame: labels, estimand restriction, feature blocks.

Implements Chapter 3 §3.6. Every exclusion is counted and returned so that
Table 4.1 (sample flow) is generated from the same code that applies it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data_load as dl

KEY = dl.KEY
CTX = dl.CTX

DEMOG_NOMINAL = ["gender", "region", "highest_education", "disability"]
DEMOG_ORDINAL = ["imd_band", "age_band"]
NUMERIC_ENROL = ["num_of_prev_attempts", "studied_credits", "days_reg_before_start"]

IMD_ORDER = ["0-10%", "10-20", "20-30%", "30-40%", "40-50%", "50-60%",
             "60-70%", "70-80%", "80-90%", "90-100%"]
AGE_ORDER = ["0-35", "35-55", "55<="]
EDU_ORDER = ["No Formal quals", "Lower Than A Level", "A Level or Equivalent",
             "HE Qualification", "Post Graduate Qualification"]


def _imd_decile(x):
    """Map OULAD imd_band to its lower decile bound as a percentage (0,10,...,90).

    Returns NaN for missing. The 40% national boundary of §3.9.1 is applied to
    this value, never to a sample percentile, so group meaning is context-stable.
    """
    if not isinstance(x, str) or not x.strip():
        return np.nan
    s = x.strip().replace("%", "")
    head = s.split("-")[0]
    try:
        return float(head)
    except ValueError:
        return np.nan


def build_frame(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (frame, sample_flow)."""
    oulad = cfg["paths"]["oulad"]
    cutoff = int(cfg["censoring"]["early_cutoff_day"])
    active_day = int(cfg["censoring"]["active_at_day"])

    info = dl.load_student_info(oulad)
    reg = dl.load_registration(oulad)

    flow = []
    flow.append(("Raw OULAD registrations", len(info)))

    df = info.merge(reg, on=KEY, how="left")

    # --- exclusion 1: no usable final result -------------------------------
    valid = cfg["labels"]["tier123_positive"] + cfg["labels"]["tier123_negative"]
    before = len(df)
    df = df[df["final_result"].isin(valid)].copy()
    flow.append(("After removing registrations with no usable final result", len(df)))

    # --- exclusion 2: the estimand (still registered at day 28) -------------
    # date_unregistration is NaN for students who never unregistered.
    unreg = pd.to_numeric(df["date_unregistration"], errors="coerce")
    still_active = unreg.isna() | (unreg >= active_day)
    df["excluded_early_dereg"] = ~still_active
    n_active = int(still_active.sum())
    flow.append((f"After removing students deregistered before day {active_day}", n_active))
    # Features are built for ALL rows and the flag retained, so the pre-specified
    # sensitivity analysis of §3.6.1 (retaining early deregistrations) can be run
    # without re-reading the interaction data. run_all.py applies the filter.
    df_active = df.copy()

    # --- feature blocks ------------------------------------------------------
    pre, early_sum = dl.vle_aggregates(oulad, cutoff)
    exact = dl.vle_exact_counts(oulad, cutoff)
    if not early_sum.empty:
        early = early_sum.drop(columns=[c for c in ("active_days", "distinct_sites") if c in early_sum.columns])
        early = early.merge(exact, on=KEY, how="outer")
    else:
        early = exact

    assess, due_ctx = dl.assessment_features(oulad, cutoff)

    out = df_active.merge(pre, on=KEY, how="left").merge(early, on=KEY, how="left")
    out = out.merge(assess, on=KEY, how="left") if len(assess) else out
    out = out.merge(due_ctx, on=CTX, how="left")

    # engagement: fill true zeros, build derived features
    for c in ["pre_clicks", "pre_events", "clicks", "events", "active_days",
              "distinct_sites", "distinct_types", "wk1_clicks", "wk2_clicks",
              "wk3_clicks", "wk4_clicks", "submitted_early"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        else:
            out[c] = 0.0

    out["any_pre_activity"] = (out["pre_clicks"] > 0).astype(int)
    out["log_pre_clicks"] = np.log1p(out["pre_clicks"])
    out["log_clicks"] = np.log1p(out["clicks"])
    out["log_events"] = np.log1p(out["events"])
    out["clicks_per_active_day"] = out["clicks"] / out["active_days"].replace(0, np.nan)
    out["clicks_per_active_day"] = out["clicks_per_active_day"].fillna(0.0)

    if "last_active_day" in out.columns:
        la = pd.to_numeric(out["last_active_day"], errors="coerce")
        out["days_since_last_activity"] = (cutoff - la).fillna(cutoff + 1)
    else:
        out["days_since_last_activity"] = cutoff + 1

    # weekly slope over the four weekly totals (least squares, x = 0..3)
    wk = out[["wk1_clicks", "wk2_clicks", "wk3_clicks", "wk4_clicks"]].to_numpy(dtype=float)
    x = np.arange(4.0)
    xc = x - x.mean()
    denom = float((xc ** 2).sum())
    out["weekly_slope"] = ((wk - wk.mean(axis=1, keepdims=True)) * xc).sum(axis=1) / denom

    # assessment block
    out["has_early_assessment"] = (pd.to_numeric(out.get("n_early_assessments"), errors="coerce")
                                   .fillna(0) > 0).astype(int)
    out["submitted_on_time"] = (out["submitted_early"] > 0).astype(int)
    out["first_score"] = pd.to_numeric(out.get("first_score"), errors="coerce")
    out["first_score_missing"] = out["first_score"].isna().astype(int)

    # enrolment block
    out["days_reg_before_start"] = -pd.to_numeric(out["date_registration"], errors="coerce")
    out["imd_decile"] = out["imd_band"].map(_imd_decile)
    out["imd_missing"] = out["imd_decile"].isna().astype(int)
    out["age_ord"] = out["age_band"].map({v: i for i, v in enumerate(AGE_ORDER)})
    out["edu_ord"] = out["highest_education"].map({v: i for i, v in enumerate(EDU_ORDER)})

    # labels
    pos = set(cfg["labels"]["tier123_positive"])
    out["y"] = out["final_result"].isin(pos).astype(int)
    t4_pos, t4_neg = set(cfg["labels"]["tier4_positive"]), set(cfg["labels"]["tier4_negative"])
    out["y_t4"] = np.where(out["final_result"].isin(t4_pos), 1,
                           np.where(out["final_result"].isin(t4_neg), 0, np.nan))

    out["context"] = out["code_module"].astype(str) + "/" + out["code_presentation"].astype(str)
    out["year"] = out["code_presentation"].astype(str).str[:4].astype(int)

    flow.append(("Final analysis sample (Tiers 1-3)", int((~out["excluded_early_dereg"]).sum())))
    raw = flow[0][1]
    sample_flow = pd.DataFrame(
        [{"stage": s, "registrations_remaining": n,
          "removed_at_this_stage": (flow[i - 1][1] - n) if i else 0,
          "pct_of_raw": round(100.0 * n / raw, 2)} for i, (s, n) in enumerate(flow)]
    )
    return out, sample_flow


def feature_columns(frame: pd.DataFrame, block: str = "all") -> list[str]:
    """Feature lists by block. `enrolment` is the Tier 4 schema (§3.5.2)."""
    enrol_num = ["num_of_prev_attempts", "studied_credits", "days_reg_before_start",
                 "imd_decile", "imd_missing", "age_ord", "edu_ord"]
    enrol_cat = ["gender", "region", "disability"]
    engage = ["any_pre_activity", "log_pre_clicks", "log_clicks", "log_events",
              "active_days", "distinct_sites", "distinct_types",
              "wk1_clicks", "wk4_clicks", "weekly_slope",
              "clicks_per_active_day", "days_since_last_activity"]
    assess = ["has_early_assessment", "submitted_on_time", "first_score", "first_score_missing"]
    if block == "enrolment":
        cols = enrol_num + enrol_cat
    else:
        cols = enrol_num + enrol_cat + engage + assess
    return [c for c in cols if c in frame.columns]


def categorical_columns(cols: list[str]) -> list[str]:
    return [c for c in cols if c in ("gender", "region", "disability", "highest_education")]
