# Cross-context transfer of student early-warning models

DST481 research by Itumeleng R. J. Modikoe (student 601430).
Analysis code, fixed plan, dependency snapshot, data provenance, and completed
real-data outputs for OULAD, with a secondary UCI robustness check.

## Original history

Both commits cited in Appendix C are preserved without rewriting:

| Record | Commit | Tag |
| --- | --- | --- |
| Plan, 20 search draws | [5b4e7739](https://github.com/iModikoe/dst481-oulad-transfer-ews/commit/5b4e7739a9f86c5f98a43e7d4e166710c111bb49) | `analysis-plan-20-draws` |
| Executed transfer implementation | [1e4a0fca](https://github.com/iModikoe/dst481-oulad-transfer-ews/commit/1e4a0fca34b5fff204632663352cb58ee835152f) | `transfer-implementation` |

The commits originated in separate repositories and have separate root histories.
The plan is also retained on `original-plan`; the implementation is an ancestor
of `main`. A local commit is not independently timestamped public preregistration.

## Completed results

[results/](results/) contains 26 CSV tables and seven figures, plus provenance.
These are real-data outputs, not synthetic smoke-test results. OULAD contained
32,593 raw registrations; 27,553 remained after excluding 5,040 early withdrawals.
There are 13 transfer pairs, and all five confirmatory comparisons have seven
Tier 3 pairs. Both H3 comparisons name XGBoost.

The study ran in two stages, not one invocation:

```powershell
python run_all.py --config config.yaml --tiers 1 --permutation-check
python run_all.py --config config.yaml --tiers 2,3,4
```

Successful execution took 1,802.566 seconds for Tier 1 and 4,735.570 seconds for
the transfer stage. An earlier Tier 1 process was interrupted and restarted
unchanged. Logs and execution records are under [results/provenance/](results/provenance/).
The [top-level run log](results/run_log.json) describes the transfer invocation.
See [report clarifications](docs/REPRODUCIBILITY.md).

## Reproduce

Python 3.10 or later; the recorded environment was Windows with Python 3.14.3.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r environment.lock.txt
python download_data.py
python run_all.py --config config.yaml --tiers 1 --permutation-check
python run_all.py --config config.yaml --tiers 2,3,4
```

The dependency snapshot describes the original environment; other platforms may
require compatible versions from `requirements.txt`. New outputs go to `outputs/`,
leaving archived `results/` unchanged. To resume using the archived benchmark,
copy `results/` to `outputs/` before running the second analysis command.
Never use `--quick` for reported findings.

The historical PowerShell supervisor retains machine-specific paths; use the
Python commands above for a fresh checkout. The original detailed pipeline guide
is retained in [docs/PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md).

## Data

Raw datasets are excluded. `download_data.py` downloads both UCI archives and
records timestamps and SHA-256 hashes. Original download provenance is in
[data_sources.json](data_sources.json).

- [OULAD](https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset), Kuzilek, Hlosta and Zdrahal, CC BY 4.0.
- [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success), Realinho and colleagues, CC BY 4.0.

See [CITATION.md](CITATION.md). No software licence has been added; dataset
licences do not automatically license the study code. The final report PDF is
not included in this publication.
