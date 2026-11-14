# Staged-run corrections, 13 November 2026

The analysis config is byte-identical to the 20-draw plan committed before Tier 1
at 5b4e7739a9f86c5f98a43e7d4e166710c111bb49. No search budget, endpoint,
subgroup rule or confirmatory comparison was changed after inspecting results.

The supplied v2 runner restores Tier 1 AUC and fairness from disk, and restricts
H3 to the model with the highest Tier 1 ROC-AUC instead of averaging families.
Before transfer execution, local corrections also remove the in-memory OOF
requirement from mitigation and preserve Tier 1 calibration and prediction rows
when writing combined outputs. These correct staged execution rather than
change the analysis specification. They are also recorded in the final run log.

tests/check_staged_resume.py exercises staged orchestration using real inputs
and cached Tier 1 tables, replacing model fitting with dummy classifiers and
writing all check outputs to a temporary directory. Its results are not findings.

The execution command is run_all.py --config config.yaml --tiers 2,3,4, without
quick mode or skipped SHAP/sensitivity analyses. Historical Tier 1 provenance is
retained under outputs/provenance; transfer_execution.json records this stage.
The final run_log.json describes the transfer invocation, not the earlier Tier 1.
