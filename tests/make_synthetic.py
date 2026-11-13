#!/usr/bin/env python3
"""Generate synthetic data in the exact OULAD schema, for smoke-testing only.

This exists so the pipeline can be verified end to end before it ever touches
the real dataset. The generated data deliberately contains a *context-varying*
feature-outcome relationship, so transfer degradation is non-zero and the
Tier 2/3 machinery is actually exercised. Numbers produced from it are
meaningless and must never appear in the report.

    python tests/make_synthetic.py --out data/synthetic --students 2500
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd

MODULES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
PRESENTATIONS = ["2013J", "2014J"]
IMD = ["0-10%", "10-20", "20-30%", "30-40%", "40-50%", "50-60%",
       "60-70%", "70-80%", "80-90%", "90-100%"]
AGE = ["0-35", "35-55", "55<="]
EDU = ["No Formal quals", "Lower Than A Level", "A Level or Equivalent",
       "HE Qualification", "Post Graduate Qualification"]
REGION = ["East Anglian Region", "Scotland", "North Western Region", "London Region",
          "South Region", "West Midlands Region", "Wales"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--students", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # --- registrations: students may appear in several modules (tests the
    #     person-level disjointness rule) ------------------------------------
    rows = []
    sid = 100000
    students = []
    for _ in range(args.students):
        sid += 1
        students.append({
            "id_student": sid,
            "gender": rng.choice(["M", "F"], p=[0.45, 0.55]),
            "region": rng.choice(REGION),
            "highest_education": rng.choice(EDU, p=[0.05, 0.35, 0.35, 0.2, 0.05]),
            "imd_band": rng.choice(IMD + [""], p=[0.095] * 10 + [0.05]),
            "age_band": rng.choice(AGE, p=[0.7, 0.26, 0.04]),
            "disability": rng.choice(["N", "Y"], p=[0.9, 0.1]),
        })
    sdf = pd.DataFrame(students)

    for _, s in sdf.iterrows():
        n_reg = rng.choice([1, 2, 3], p=[0.72, 0.22, 0.06])
        mods = rng.choice(MODULES, size=min(n_reg, len(MODULES)), replace=False)
        for m in np.atleast_1d(mods):
            pres = rng.choice(PRESENTATIONS)
            rows.append({**s.to_dict(), "code_module": m, "code_presentation": pres,
                         "num_of_prev_attempts": int(rng.choice([0, 1, 2], p=[0.85, 0.12, 0.03])),
                         "studied_credits": int(rng.choice([30, 60, 90, 120], p=[.4, .35, .15, .1]))})
    reg = pd.DataFrame(rows).drop_duplicates(subset=["id_student", "code_module", "code_presentation"])

    # --- latent engagement and a context-varying outcome rule ---------------
    n = len(reg)
    base = rng.gamma(2.0, 60.0, n)
    mod_effect = reg["code_module"].map({m: 0.6 + 0.14 * i for i, m in enumerate(MODULES)}).to_numpy()
    clicks = np.clip(base * mod_effect, 0, None)
    active_days = np.clip(rng.poisson(np.clip(clicks / 25, 0.4, 22)), 0, 28)
    imd_num = reg["imd_band"].map(lambda v: float(v.replace("%", "").split("-")[0]) if v else np.nan)

    # the weight on engagement DIFFERS by module: this is the concept drift
    # that makes cross-module transfer (Tier 3) genuinely harder than Tier 2.
    w_eng = reg["code_module"].map({m: -1.5 - 0.45 * i for i, m in enumerate(MODULES)}).to_numpy()
    year = reg["code_presentation"].str[:4].astype(int).to_numpy()
    logit = (0.55
             + w_eng * (np.log1p(clicks) - np.log1p(clicks).mean()) / np.log1p(clicks).std()
             - 0.30 * (active_days - active_days.mean()) / max(active_days.std(), 1)
             - 0.012 * np.nan_to_num(imd_num.to_numpy(), nan=45.0)
             + 0.22 * (reg["gender"].to_numpy() == "F")
             + 0.18 * (year == 2014))
    p_risk = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p_risk)

    final = np.where(y == 1,
                     rng.choice(["Fail", "Withdrawn"], size=n, p=[0.55, 0.45]),
                     rng.choice(["Pass", "Distinction"], size=n, p=[0.82, 0.18]))
    reg["final_result"] = final
    reg["date_registration"] = -rng.integers(5, 120, n)
    unreg = np.full(n, np.nan)
    early_leavers = (final == "Withdrawn") & (rng.random(n) < 0.35)
    unreg[early_leavers] = rng.integers(-20, 28, early_leavers.sum())
    late = (final == "Withdrawn") & ~early_leavers
    unreg[late] = rng.integers(30, 260, late.sum())
    reg["date_unregistration"] = unreg

    info_cols = ["code_module", "code_presentation", "id_student", "gender", "region",
                 "highest_education", "imd_band", "age_band", "num_of_prev_attempts",
                 "studied_credits", "disability", "final_result"]
    reg[info_cols].to_csv(os.path.join(args.out, "studentInfo.csv"), index=False)
    reg[["code_module", "code_presentation", "id_student",
         "date_registration", "date_unregistration"]].to_csv(
        os.path.join(args.out, "studentRegistration.csv"), index=False)

    # --- vle ----------------------------------------------------------------
    n_sites = 220
    vle = pd.DataFrame({
        "id_site": np.arange(1, n_sites + 1),
        "code_module": rng.choice(MODULES, n_sites),
        "code_presentation": rng.choice(PRESENTATIONS, n_sites),
        "activity_type": rng.choice(["resource", "oucontent", "forumng", "quiz",
                                     "url", "homepage", "subpage"], n_sites),
        "week_from": np.nan, "week_to": np.nan,
    })
    vle.to_csv(os.path.join(args.out, "vle.csv"), index=False)

    # --- studentVle: interaction rows spanning days -25..200 ---------------
    parts = []
    for i, (_, r) in enumerate(reg.iterrows()):
        total = int(clicks[i])
        n_ev = max(0, int(rng.poisson(max(total / 9.0, 0.6))))
        if n_ev == 0:
            continue
        days = np.clip(rng.normal(14, 22, n_ev).astype(int), -25, 200)
        parts.append(pd.DataFrame({
            "code_module": r["code_module"], "code_presentation": r["code_presentation"],
            "id_student": r["id_student"], "id_site": rng.integers(1, n_sites + 1, n_ev),
            "date": days, "sum_click": rng.integers(1, 12, n_ev),
        }))
    pd.concat(parts, ignore_index=True).to_csv(
        os.path.join(args.out, "studentVle.csv"), index=False)

    # --- assessments --------------------------------------------------------
    arows, aid = [], 1
    for m in MODULES:
        for p in PRESENTATIONS:
            for d in [18, 25, 60, 120, 200]:
                arows.append({"code_module": m, "code_presentation": p, "id_assessment": aid,
                              "assessment_type": "TMA", "date": d, "weight": 20})
                aid += 1
    assess = pd.DataFrame(arows)
    assess.to_csv(os.path.join(args.out, "assessments.csv"), index=False)

    sa_rows = []
    early = assess[assess["date"] <= 27]
    for _, r in reg.iterrows():
        cand = early[(early.code_module == r["code_module"])
                     & (early.code_presentation == r["code_presentation"])]
        for _, a in cand.iterrows():
            if rng.random() < 0.72:
                sa_rows.append({
                    "id_assessment": a["id_assessment"], "id_student": r["id_student"],
                    "date_submitted": int(a["date"] - rng.integers(0, 6)),
                    "is_banked": 0, "score": int(np.clip(rng.normal(66, 17), 0, 100)),
                })
    pd.DataFrame(sa_rows).to_csv(os.path.join(args.out, "studentAssessment.csv"), index=False)

    pd.DataFrame([{"code_module": m, "code_presentation": p, "module_presentation_length": 260}
                  for m in MODULES for p in PRESENTATIONS]).to_csv(
        os.path.join(args.out, "courses.csv"), index=False)

    print(f"synthetic OULAD written to {args.out}: {len(reg)} registrations, "
          f"{reg['id_student'].nunique()} distinct students, "
          f"at-risk prevalence {y.mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
