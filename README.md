# Cross-Contextual Transferability of Early-Warning Models — analysis pipeline

Implementation of the design specified in Chapter 3 of the DST481 research report
(I.R.J. Modikoe, Belgium Campus iTversity). One command regenerates every table
and figure in Chapter 4.

The point of this repository is that the dissertation's claims are checkable. A
third party with the two public datasets should be able to reproduce every number
in Chapter 4 from this code alone — which is the reproducibility deficit Balaji,
et al. (2021) identify in the field, addressed rather than merely noted.

---

## 1. Get the data

Neither dataset is redistributed here; both are openly licensed and permanently
archived, and you download them yourself.

**OULAD (primary).** Download from <https://analyse.kmi.open.ac.uk/open_dataset>
and unzip all seven CSVs into `data/oulad/`:

```
data/oulad/
  studentInfo.csv  studentRegistration.csv  studentVle.csv  vle.csv
  assessments.csv  studentAssessment.csv    courses.csv
```

`studentVle.csv` is about 432 MB. The loader reads it in chunks, so you do not
need to hold it in memory, but you do need the disk space.

**UCI Predict Students' Dropout and Academic Success (secondary, Tier 4 only).**
Download from <https://archive.ics.uci.edu/dataset/697> and place the
semicolon-delimited file at `data/uci/data.csv`.

Record the access date and licence terms — Appendix C of the report asks for both.

## 2. Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip freeze > environment.lock.txt      # commit this: §3.11 promises a pinned environment
```

Python 3.10 or later.

## 3. Check it runs before you commit a long run

```bash
python tests/make_synthetic.py --out data/synthetic --students 2500
sed 's|data/oulad|data/synthetic|' config.yaml > config_test.yaml
python run_all.py --config config_test.yaml --quick --tiers 1,2,3
```

This generates data in the exact OULAD schema and runs the whole pipeline on it
in a few minutes. **Numbers from synthetic data are meaningless and must never
appear in the report** — this step only proves the machinery works end to end.

A correct smoke test shows the majority-class baseline at ROC-AUC exactly 0.500,
the stratified baseline near 0.500, and the three learners well above both. If a
baseline scores above chance, something is leaking and you must stop and find it
before running on the real data.

## 4. Run

```bash
python run_all.py --config config.yaml
```

Expect several hours on a laptop: the nested search refits each model family on
every source–target pair, which is the cost of selecting hyperparameters on
source data alone. Run it overnight. To spread it across sessions:

```bash
python run_all.py --tiers 1      # internal benchmark first
python run_all.py --tiers 2,3    # then the transfer tiers
```

Outputs land in `outputs/tables/*.csv` and `outputs/figures/*.png`, named to
match the `Source:` line printed under each table shell in Chapter 4.

## 5. Before you run it for real

The analysis plan is `config.yaml`, and §3.11 commits you to fixing it *before*
Phase III executes. Concretely:

```bash
git add config.yaml && git commit -m "Pre-register analysis plan before Phase III"
git rev-parse HEAD    # record this hash in Appendix C
```

If you change anything in `config.yaml` after inspecting results — the alert
budget, the confirmatory family, the deprivation cut-point — that is a deviation
from the pre-specification and must be recorded as such in the report. A
pre-registration quietly edited afterwards is worse than none, because it claims
a guarantee it no longer provides.

---

## What the code enforces, and where the report says so

| Guarantee | Where | How it is enforced |
|---|---|---|
| Day-28 censoring | §3.6.1 | Filters on the `date` field in `data_load.py`; nothing later than day 27 enters feature construction |
| Estimand restriction | §3.6.1 | `date_unregistration` is NaN or ≥ 28; the count removed is reported in `sample_flow.csv` |
| No preprocessing leakage | §3.6.2 | Every transformer sits inside a `Pipeline`, fitted within the training fold only |
| Person-level disjointness | §3.8.1 | `splits._drop_overlap` removes training rows whose student appears in the target; counts reported in `overlap_removed.csv` |
| Source-only tuning | §3.7 | The inner search receives source indices only; the target is never passed to `fit_with_search` |
| Fixed decision rule | §3.9.1 | Alert budget set in `config.yaml` and applied before any fairness quantity is computed |
| Externally anchored groups | §3.9.1 | IMD 40% national boundary, never a sample percentile |
| Evaluability rule | §3.9.1 | ≥100 members and ≥30 positives, else the cell is reported non-evaluable |
| Pre-specified test family | §3.8.3 | Exactly the five comparisons listed in `config.yaml`; Holm–Bonferroni across those and no others |
| Explanation comparison | §3.10 | Primary analysis holds one fitted model constant; secondary analysis kept in a separate file |

## Outputs and the Chapter 4 tables they fill

| File | Chapter 4 |
|---|---|
| `sample_flow.csv` | Table 4.1 |
| `context_characteristics.csv` | Table 4.2 |
| `overlap_removed.csv` | Table 4.3 |
| `subgroup_evaluability.csv` | Table 4.4 |
| `tier1_internal.csv` | Table 4.5 |
| `degradation_by_model_tier.csv` | Table 4.7 |
| `tier3_per_pair.csv` | Table 4.8 |
| `calibration_by_tier.csv` | Table 4.9 |
| `distance_measures.csv` | Table 4.10 |
| `distance_degradation_corr.csv` | Table 4.11 |
| `fairness_eod.csv` | Tables 4.12, 4.14 |
| `fairness_mitigation.csv` | Table 4.13 |
| `shap_concordance_primary.csv` | Table 4.15 |
| `shap_feature_stability.csv`, `shap_block_stability.csv` | Table 4.16 |
| `shap_concordance_secondary.csv` | Table 4.17 |
| `confirmatory_family.csv` | Table 4.19 |
| `data_dictionary.csv` | Appendix A |
| `figures/degradation_dotplot.png` | Figure 4.2 |
| `figures/distance_vs_degradation.png` | Figure 4.4 |
| `figures/fairness_tradeoff.png` | Figure 4.5 |
| `figures/concordance_matrix.png` | Figure 4.7 |

## Layout

```
config.yaml          the analysis plan — commit before Phase III
run_all.py           single entry point
src/data_load.py     chunked OULAD readers, date-field filtering
src/features.py      labels, estimand, feature blocks, sample flow
src/splits.py        person-disjoint source-target pair construction
src/models.py        model suite, leakage-free pipelines, source-only search
src/metrics.py       discrimination, calibration, alert-budget recall, bootstrap
src/distance.py      PSI, Jensen-Shannon, prevalence difference, size ratio
src/fairness.py      EOD at a fixed budget, transferred vs oracle mitigation
src/explain.py       SHAP rank stability, model held constant
src/stats.py         corrected resampled t-test, Holm-Bonferroni
src/report.py        CSV and figure writing
tests/make_synthetic.py   OULAD-schema data for smoke testing
```

## Known limitations of this implementation

- Tier 4 requires the UCI file and is restricted to the enrolment block; it is a
  related-construct check at enrolment time, not a replication of the day-28
  task (§3.5.2). Run it only after Tiers 1–3 succeed.
- SHAP uses the tree explainer for Random Forest and XGBoost and falls back to
  standardised absolute coefficients for logistic regression, which §3.10 reports
  alongside rather than instead.
- The SRQ2 correlations rest on a small number of source–target pairs. Wide
  intervals are the expected result, not a defect — see the guidance in §5.3.
