#!/usr/bin/env python3
"""Synthetic file in the UCI Dropout schema, for smoke-testing Tier 4 only."""
import argparse, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/uci_synthetic/data.csv")
ap.add_argument("--n", type=int, default=4424)
ap.add_argument("--seed", type=int, default=11)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
os.makedirs(os.path.dirname(a.out), exist_ok=True)

n = a.n
gender = rng.integers(0, 2, n)                       # 1 = male, 0 = female
age = np.clip(rng.gamma(4.0, 5.5, n) + 17, 17, 70).round().astype(int)
prevq = rng.integers(1, 20, n)
admission = np.clip(rng.normal(127, 14, n), 95, 190)
logit = -0.4 + 0.035 * (age - age.mean()) - 0.02 * (admission - admission.mean()) + 0.15 * gender
p = 1 / (1 + np.exp(-logit))
draw = rng.random(n)
target = np.where(draw < p * 0.55, "Dropout",
                  np.where(draw < p * 0.55 + 0.18, "Enrolled", "Graduate"))

pd.DataFrame({
    "Marital status": rng.integers(1, 6, n),
    "Application mode": rng.integers(1, 18, n),
    "Course": rng.integers(1, 17, n),
    "Previous qualification": prevq,
    "Admission grade": admission.round(1),
    "Gender": gender,
    "Age at enrollment": age,
    "Scholarship holder": rng.integers(0, 2, n),
    "Debtor": rng.integers(0, 2, n),
    "Tuition fees up to date": rng.integers(0, 2, n),
    "Curricular units 1st sem (approved)": rng.integers(0, 8, n),
    "Curricular units 1st sem (grade)": np.clip(rng.normal(11, 4, n), 0, 20).round(2),
    "Target": target,
}).to_csv(a.out, sep=";", index=False)
print(f"synthetic UCI written to {a.out}: {n} rows; "
      f"Dropout {np.mean(target=='Dropout'):.3f}, Graduate {np.mean(target=='Graduate'):.3f}")
