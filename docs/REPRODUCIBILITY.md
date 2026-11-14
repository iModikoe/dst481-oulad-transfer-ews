# Reproducibility record and Appendix C clarifications

This note records the state of the report as of revision (4).

1. Both cited commits are original Git objects, preserved under tags. They came
   from separate repositories, not successive commits in one original lineage.
2. A clean initial commit does not prove no results informed a plan. The records
   support the narrower statement that the 20-draw plan was committed locally
   before the recorded real-data runs. Synthetic testing preceded those runs.
   GitHub publication occurred later, not as prospective public preregistration.
3. Results came from two invocations. Tier 1 finished on 13 November 2026 at
   20:18:10.843 UTC (execution record); transfer finished at 21:49:52 UTC. The top-level run log describes
   transfer only; Tier 1 provenance remains under results/provenance/.
4. Successful durations are 1,802.566 and 4,735.570 seconds, totalling about
   1 hour 48 minutes 58 seconds, excluding the interrupted first attempt. This
   timing discrepancy was corrected in report revision (4); the PDF now agrees.
5. Implementation corrections restored staged baseline/fairness loading, limited
   H3 to the model with highest internal ROC-AUC, enabled staged mitigation, and
   retained internal calibration and predictions. The config remained unchanged.
6. The historical PowerShell supervisor script contains original Windows paths. Use the README's
   Python commands for a fresh checkout.

Archived primary outputs were copied without changing values. Raw datasets and
the final report PDF are not included in this repository.
