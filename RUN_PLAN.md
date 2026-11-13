# Real-data execution plan

Agreed on 13 November 2026, before real-data modelling results were inspected.

- Search: 20 draws per learner; all other config.yaml settings retained, including
  five outer folds, three inner folds and 2,000 bootstrap resamples.
- Stage 1: `run_all.py --config config.yaml --tiers 1 --permutation-check`.
  No `--quick`. Measure elapsed time and inspect the leakage audits and baseline.
- Remaining tiers will follow after assessment of Stage 1. The repository's
  separate-tier invocation needs review before continuing: internal predictions
  and baseline-dependent analyses are held in memory, not checkpointed.
- Data sources, download times and SHA-256 hashes are in data_sources.json.
  Data files remain untracked. The environment is recorded in environment.lock.txt.
- Synthetic results are machinery checks only and must not enter the report.
- Chance baselines, feature audits and permutation checks do not establish
  historical availability of predictors; that requires a recording-system audit.
- Equal tuning budgets do not guarantee equal tuning adequacy or cancellation
  of tuning effects in degradation estimates. Twenty draws is the agreed budget,
  not an empirically established plateau for this dataset.

The commit containing this plan records the local pre-analysis specification;
it is not an independently timestamped public preregistration.
