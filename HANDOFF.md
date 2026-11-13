# What to send back once the run finishes

Chapters 4, 5 and 6 of the report cannot be written until the analysis has
actually been run. This note says exactly what is needed to write them, so the
hand-off takes one message rather than several rounds.

## Send the whole `outputs/` directory

```bash
cd outputs && zip -r ../outputs_$(date +%Y%m%d).zip tables figures run_log.json
```

That is about 20 CSVs, 7 PNGs and the run log. Every table shell in Chapter 4
names the CSV that fills it, so nothing has to be matched up by hand.

Do **not** retype numbers into the report yourself before sending. Transcription
is where errors enter, and the point of the pipeline is that the figures in the
report trace to a file that traces to code.

## Also send, in the message body

1. **Which run it is.** The `config.yaml` commit hash, and confirmation that it
   was committed *before* the run started. If you changed anything in the config
   after seeing any result, say what and why — that is a deviation from the
   pre-specification and belongs in the report, not hidden.

2. **Anything that failed or was skipped.** The log prints a line for every
   skipped step — Tier 4 without the UCI file, SHAP that fell back, subgroups
   that failed the evaluability rule. These are not embarrassments; several of
   them are findings, and the fairness evaluability exclusions in particular
   must be named in §4.6 rather than quietly dropped.

3. **Wall-clock time and hardware.** Appendix B asks for both.

4. **Anything that looked wrong to you.** You know this study better than the
   numbers do. If a figure looks implausible, say so before it goes in the
   report.

## Check these three things before you send

These are the checks most likely to catch a real problem:

- **The baselines.** `tier1_internal.csv` must show the majority-class baseline
  at ROC-AUC exactly 0.500. If any baseline is meaningfully above chance,
  something is leaking and the whole run is void. Stop and find it.

- **The day-28 exclusion.** `sample_flow.csv` reports how many registrations the
  estimand restriction removed. If that number is zero, the `date_unregistration`
  field is not being read correctly. If it is very large, check the sign
  convention before accepting it.

- **The person-overlap removal.** `overlap_removed.csv` should show a non-zero
  count for the Tier 3 leave-one-module-out pairs, since students take more than
  one module. Zero there means the disjointness rule is not firing, and the
  Tier 3 figures would be contaminated.

## What happens next

With the outputs in hand, Chapter 4 gets written from the actual numbers,
Chapter 5 from what those numbers mean, and Chapter 6 from both. The scaffolds
already set out the structure and the interpretive frames, including what each
possible pattern of results would mean — so the writing is fast once the
evidence exists.

Results that contradict the hypotheses are reported as found. H1 through H4 were
stated in advance precisely so that they could fail, and a pre-registered study
reporting only confirmations has thrown away the thing that made it credible.
