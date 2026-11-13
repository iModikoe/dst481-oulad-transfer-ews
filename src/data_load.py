"""Load OULAD and UCI data, enforcing the day-28 rules of Chapter 3 §3.6.

studentVle.csv is large (~10M rows). It is read in chunks and reduced to
per-registration aggregates *while being read*, so peak memory stays modest.
Every filter applied here is a filter on a date field, which is what makes the
"early prediction" claim auditable rather than asserted (§3.6.1).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

KEY = ["code_module", "code_presentation", "id_student"]
CTX = ["code_module", "code_presentation"]


def _req(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected OULAD file not found: {path}\n"
            "Download the dataset from https://analyse.kmi.open.ac.uk/open_dataset "
            "and unzip the seven CSVs into the directory named in config.yaml."
        )
    return path


def load_student_info(oulad_dir: str) -> pd.DataFrame:
    df = pd.read_csv(_req(os.path.join(oulad_dir, "studentInfo.csv")))
    df.columns = [c.strip() for c in df.columns]
    return df


def load_registration(oulad_dir: str) -> pd.DataFrame:
    df = pd.read_csv(_req(os.path.join(oulad_dir, "studentRegistration.csv")))
    df.columns = [c.strip() for c in df.columns]
    return df


def vle_aggregates(oulad_dir: str, cutoff: int, chunksize: int = 2_000_000) -> pd.DataFrame:
    """Per-registration engagement features from studentVle, censored at `cutoff`.

    Produces pre-start features (date < 0) and early features (0 <= date <= cutoff)
    separately, because §3.6.2 encodes them as distinct blocks.
    """
    path = _req(os.path.join(oulad_dir, "studentVle.csv"))
    vle_path = _req(os.path.join(oulad_dir, "vle.csv"))
    vle = pd.read_csv(vle_path, usecols=["id_site", "activity_type"])
    act_map = dict(zip(vle["id_site"], vle["activity_type"]))

    pre_parts, early_parts = [], []
    usecols = ["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"]
    dtypes = {"id_student": "int64", "id_site": "int64", "date": "int32", "sum_click": "int32"}

    for chunk in pd.read_csv(path, usecols=usecols, dtype=dtypes, chunksize=chunksize):
        pre = chunk[chunk["date"] < 0]
        if len(pre):
            pre_parts.append(
                pre.groupby(KEY, observed=True)
                .agg(pre_clicks=("sum_click", "sum"), pre_events=("date", "size"))
                .reset_index()
            )
        early = chunk[(chunk["date"] >= 0) & (chunk["date"] <= cutoff)].copy()
        if len(early):
            early["activity_type"] = early["id_site"].map(act_map)
            early["week"] = (early["date"] // 7).clip(0, 3)
            g = early.groupby(KEY, observed=True)
            base = g.agg(
                clicks=("sum_click", "sum"),
                events=("date", "size"),
                active_days=("date", "nunique"),
                distinct_sites=("id_site", "nunique"),
                distinct_types=("activity_type", "nunique"),
                last_active_day=("date", "max"),
            ).reset_index()
            wk = (
                early.groupby(KEY + ["week"], observed=True)["sum_click"].sum()
                .unstack("week").reindex(columns=[0, 1, 2, 3]).fillna(0)
                .rename(columns={0: "wk1_clicks", 1: "wk2_clicks", 2: "wk3_clicks", 3: "wk4_clicks"})
                .reset_index()
            )
            early_parts.append(base.merge(wk, on=KEY, how="left"))

    def _combine(parts, sum_cols, max_cols=(), nunique_warn=()):
        if not parts:
            return pd.DataFrame(columns=KEY)
        df = pd.concat(parts, ignore_index=True)
        agg = {c: "sum" for c in sum_cols if c in df.columns}
        for c in max_cols:
            if c in df.columns:
                agg[c] = "max"
        return df.groupby(KEY, observed=True).agg(agg).reset_index()

    pre = _combine(pre_parts, ["pre_clicks", "pre_events"])
    early = _combine(
        early_parts,
        ["clicks", "events", "active_days", "distinct_sites", "distinct_types",
         "wk1_clicks", "wk2_clicks", "wk3_clicks", "wk4_clicks"],
        max_cols=["last_active_day"],
    )
    # NOTE: active_days / distinct_sites / distinct_types are summed across chunks.
    # This is exact only when a registration's rows fall in one chunk. We therefore
    # re-derive them below from a second, narrow pass to keep the claim honest.
    return pre, early


def vle_exact_counts(oulad_dir: str, cutoff: int, chunksize: int = 4_000_000) -> pd.DataFrame:
    """Exact distinct-count features, which cannot be summed across chunks."""
    path = os.path.join(oulad_dir, "studentVle.csv")
    usecols = ["code_module", "code_presentation", "id_student", "id_site", "date"]
    dtypes = {"id_student": "int64", "id_site": "int64", "date": "int32"}
    seen = []
    for chunk in pd.read_csv(path, usecols=usecols, dtype=dtypes, chunksize=chunksize):
        c = chunk[(chunk["date"] >= 0) & (chunk["date"] <= cutoff)]
        if len(c):
            seen.append(c[KEY + ["id_site", "date"]].drop_duplicates())
    if not seen:
        return pd.DataFrame(columns=KEY + ["active_days", "distinct_sites"])
    allrows = pd.concat(seen, ignore_index=True).drop_duplicates()
    return (
        allrows.groupby(KEY, observed=True)
        .agg(active_days=("date", "nunique"), distinct_sites=("id_site", "nunique"))
        .reset_index()
    )


def assessment_features(oulad_dir: str, cutoff: int) -> pd.DataFrame:
    """Early assessment features: only assessments *due* by the cutoff, and only
    submissions made by the cutoff (§3.6.2). No score imputation."""
    assessments = pd.read_csv(_req(os.path.join(oulad_dir, "assessments.csv")))
    sa = pd.read_csv(_req(os.path.join(oulad_dir, "studentAssessment.csv")))
    assessments.columns = [c.strip() for c in assessments.columns]
    sa.columns = [c.strip() for c in sa.columns]

    early_assess = assessments[
        pd.to_numeric(assessments["date"], errors="coerce").fillna(10**6) <= cutoff
    ]
    if early_assess.empty:
        return pd.DataFrame(columns=KEY + ["has_early_assessment", "submitted_early", "first_score"])

    merged = sa.merge(
        early_assess[["id_assessment", "code_module", "code_presentation", "date"]],
        on="id_assessment", how="inner", suffixes=("", "_due"),
    )
    merged = merged[pd.to_numeric(merged["date_submitted"], errors="coerce") <= cutoff]

    due_per_ctx = (
        early_assess.groupby(CTX, observed=True)["id_assessment"].nunique()
        .rename("n_early_assessments").reset_index()
    )
    if merged.empty:
        out = pd.DataFrame(columns=KEY + ["submitted_early", "first_score"])
    else:
        merged = merged.sort_values("date_submitted")
        out = (
            merged.groupby(KEY, observed=True)
            .agg(submitted_early=("id_assessment", "nunique"), first_score=("score", "first"))
            .reset_index()
        )
    return out, due_per_ctx


def load_uci(path: str) -> pd.DataFrame:
    """UCI Predict Students' Dropout and Academic Success. Semicolon-delimited."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected UCI file not found: {path}\n"
            "Download from https://archive.ics.uci.edu/dataset/697 and place data.csv there."
        )
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip() for c in df.columns]
    return df
