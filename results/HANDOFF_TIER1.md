# Tier 1 real-data handoff

Completed successfully on 13 November 2026 at 22:18 Africa/Johannesburg.
This archive contains Tier 1 only, not the completed four-tier study.

- Command: `run_all.py --config config.yaml --tiers 1 --permutation-check`
- Quick mode: false. Search draws: 20. Bootstrap resamples: 2,000.
- Pre-run plan commit: `5b4e7739a9f86c5f98a43e7d4e166710c111bb49`.
- Successful execution elapsed time: 1,802.566 seconds (30 minutes 3 seconds),
  including feature construction, three permutation checks and final output writing.
- An earlier process ended during fold 1 without an error or completion record.
  It was restarted with unchanged code and configuration. The elapsed time above
  describes the successful restart only.
- Real OULAD: 32,593 raw registrations, 5,040 early deregistrations excluded,
  27,553 analysed. The UCI dataset was downloaded but was not used by this stage.
- Baseline ROC-AUC: majority 0.500000; stratified 0.497209.
- Learner ROC-AUC: logistic regression 0.758996; Random Forest 0.771086;
  XGBoost 0.775993. Confidence intervals and other metrics are in tier1_internal.csv.
- Permutation ROC-AUCs: 0.4999, 0.5006, 0.5086; all within the configured tolerance.
- No feature reached the 0.90 univariate audit threshold; maximum about 0.670
  for active_days. These checks do not establish historical predictor availability.

The pipeline emits confirmatory_family.csv even for this single-tier invocation.
Its transfer comparisons have no Tier 3 observations and are not completed
hypothesis tests. Likewise, transfer degradation, person-overlap removal, external
validation, mitigation, SHAP transfer analyses and sensitivity results are pending.

The included config, source hashes, dependency snapshot, execution record and log
document this stage. No synthetic results are included in this archive.
