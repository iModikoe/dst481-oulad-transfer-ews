#!/usr/bin/env python3
"""Single entry point. Regenerates every table and figure in Chapter 4.

    python run_all.py --config config.yaml

Output file names match the `Source:` lines printed under each table shell in
the Chapter 4 scaffold, so the report and the code stay in correspondence.

Run modes:
    --quick     reduced search budget and bootstrap; for smoke-testing only.
                Results from a --quick run must NOT be reported.
    --tiers     restrict to a subset, e.g. --tiers 1,2
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import data_load as dl
from src import (distance, explain, fairness, features, leakage, metrics,
                 models, sensitivity, splits, stats, tier4)
from src.report import (write_table, make_figures, data_dictionary_rows)


# --------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------
def tier1(frame, num_cols, cat_cols, cfg, out, quick=False):
    """Student-grouped stratified CV. Benchmark from which degradation is measured."""
    seed = int(cfg["models"]["seed"])
    budgets = tuple(cfg["evaluation"]["alert_budgets"])
    nboot = 200 if quick else int(cfg["evaluation"]["bootstrap_resamples"])
    folds = splits.tier1_folds(frame, int(cfg["models"]["outer_folds"]), seed)

    X = frame[num_cols + cat_cols]
    y = frame["y"].to_numpy()
    groups = frame["id_student"].to_numpy()

    oof = {m: np.full(len(frame), np.nan) for m in
           ["majority", "stratified"] + models.LEARNERS}
    per_fold = {m: [] for m in models.LEARNERS}

    for fi, (tr, te) in enumerate(folds):
        log(f"  Tier 1 fold {fi + 1}/{len(folds)}  (train {len(tr)}, test {len(te)})")
        for name in ["majority", "stratified"] + models.LEARNERS:
            if name in ("majority", "stratified"):
                pipe = models.make_pipeline(name, num_cols, cat_cols, seed)
                pipe.fit(X.iloc[tr], y[tr])
            else:
                pipe, _ = models.fit_with_search(
                    name, X.iloc[tr], y[tr], groups[tr], num_cols, cat_cols, cfg)
            p = models.predict_proba(pipe, X.iloc[te])
            oof[name][te] = p
            if name in models.LEARNERS and len(np.unique(y[te])) > 1:
                per_fold[name].append(metrics.auc_fn(y[te], p))

    rows = []
    for name in ["majority", "stratified"] + models.LEARNERS:
        p = oof[name]
        m = np.isfinite(p)
        r = metrics.core_metrics(y[m], p[m], budgets, int(cfg["evaluation"]["calibration_bins"]))
        _, lo, hi = metrics.bootstrap_ci(y[m], p[m], metrics.auc_fn, nboot, seed)
        r.update({"model": models.MODEL_LABELS[name], "model_key": name,
                  "roc_auc_lo": lo, "roc_auc_hi": hi, "tier": 1})
        rows.append(r)
    write_table(pd.DataFrame(rows), out, "tier1_internal")
    return oof, per_fold, pd.DataFrame(rows)


def evaluate_pair(pair, frame, num_cols, cat_cols, cfg, quick=False, class_weight=None):
    """Fit on source (search on source only), score the target."""
    seed = int(cfg["models"]["seed"])
    budgets = tuple(cfg["evaluation"]["alert_budgets"])
    X = frame[num_cols + cat_cols]
    y = frame["y"].to_numpy()
    groups = frame["id_student"].to_numpy()
    tr, te = pair.train_idx, pair.test_idx

    results, fitted = {}, {}
    for name in models.LEARNERS:
        pipe, best = models.fit_with_search(
            name, X.iloc[tr], y[tr], groups[tr], num_cols, cat_cols, cfg, class_weight)
        p = models.predict_proba(pipe, X.iloc[te])
        r = metrics.core_metrics(y[te], p, budgets, int(cfg["evaluation"]["calibration_bins"]))
        r.update({"model": models.MODEL_LABELS[name], "model_key": name,
                  "tier": pair.tier, "pair": pair.name,
                  "differs_by": pair.differs_by,
                  "n_removed_overlap": pair.n_removed_overlap,
                  "pct_removed_overlap": round(pair.pct_removed, 2),
                  "best_params": json.dumps({k: str(v) for k, v in (best or {}).items()})})
        results[name] = r
        fitted[name] = (pipe, p)
    return results, fitted


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--quick", action="store_true",
                    help="reduced budget; smoke test only, results not reportable")
    ap.add_argument("--tiers", default="1,2,3,4")
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--permutation-check", action="store_true",
                    help="refit Tier 1 on shuffled labels; AUC must collapse to ~0.5")
    args = ap.parse_args()

    cfg = load_config(args.config)
    want = {int(t) for t in args.tiers.split(",") if t.strip()}
    out = cfg["paths"]["outputs"]
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)

    if args.quick:
        cfg["models"]["search_draws"] = 4
        cfg["evaluation"]["bootstrap_resamples"] = 200
        log("QUICK MODE — smoke test only. Do not report these numbers.")

    seed = int(cfg["models"]["seed"])
    np.random.seed(seed)
    nboot = int(cfg["evaluation"]["bootstrap_resamples"])
    budgets = tuple(cfg["evaluation"]["alert_budgets"])
    run_log = {
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(), "seed": seed,
        "quick_mode": bool(args.quick), "config": args.config,
    }

    # ---- RO1: frame, flow, context characteristics -------------------------
    log("Building the modelling frame (day-28 censoring, estimand restriction)")
    full_frame, sample_flow = features.build_frame(cfg)
    write_table(sample_flow, out, "sample_flow")
    # The primary analysis is restricted to the estimand of §3.6.1. The excluded
    # rows are retained in `full_frame` solely for the pre-specified sensitivity
    # analysis that relaxes the restriction.
    frame = full_frame[~full_frame["excluded_early_dereg"]].reset_index(drop=True)
    log(f"  estimand restriction removed {len(full_frame) - len(frame):,} registrations "
        f"({100 * (len(full_frame) - len(frame)) / max(len(full_frame), 1):.1f}%)")

    num_cols_all = [c for c in features.feature_columns(frame, "all")
                    if c not in features.categorical_columns(features.feature_columns(frame, "all"))]
    cat_cols_all = features.categorical_columns(features.feature_columns(frame, "all"))
    log(f"  {len(frame):,} registrations · {len(num_cols_all)} numeric + {len(cat_cols_all)} categorical features")

    # ---- leakage checks (§3.11) -------------------------------------------
    # Structural and univariate checks run every time: a chance-level baseline
    # only shows that a feature-ignoring classifier cannot win, which says
    # nothing about a feature that encodes post-cutoff information.
    leakage.assert_no_outcome_columns(num_cols_all, cat_cols_all)
    ua = leakage.univariate_audit(frame, num_cols_all, cat_cols_all)
    write_table(ua, out, "leakage_univariate_audit")
    flagged = ua[ua["flagged"]]
    if len(flagged):
        log("  LEAKAGE WARNING - these features separate the outcome almost alone:")
        for r in flagged.itertuples():
            log(f"    {r.feature}: univariate ROC-AUC {r.univariate_auc:.3f}")
        log("  Inspect each before reporting anything from this run.")
    else:
        log(f"  leakage audit: no feature above 0.90 univariate AUC "
            f"(max {ua['univariate_auc'].max():.3f} on "
            f"{ua['feature'].iloc[0] if len(ua) else 'n/a'})")

    if args.permutation_check:
        log("  permutation check - refitting Tier 1 on shuffled labels")
        write_table(
            leakage.permutation_check(frame, num_cols_all, cat_cols_all, cfg, log=log),
            out, "leakage_permutation_check")

    ctx = (frame.groupby(["code_module", "code_presentation"], observed=True)
           .agg(registrations=("y", "size"), at_risk_n=("y", "sum"),
                distinct_students=("id_student", "nunique"))
           .reset_index())
    ctx["at_risk_prevalence_pct"] = (100 * ctx["at_risk_n"] / ctx["registrations"]).round(2)
    write_table(ctx, out, "context_characteristics")
    write_table(pd.DataFrame(data_dictionary_rows(frame, num_cols_all, cat_cols_all)),
                out, "data_dictionary")

    # fairness groups
    frame["grp_gender"] = fairness.gender_group(frame["gender"])
    frame["grp_deprivation"] = fairness.deprivation_group(
        frame["imd_decile"], float(cfg["fairness"]["imd_reference_floor"]))

    ev_rows = []
    for attr, col in [("gender", "grp_gender"), ("deprivation", "grp_deprivation")]:
        for which in ("reference", "protected", "missing"):
            m = frame[col] == which
            ev_rows.append({
                "attribute": attr, "group": which, "n": int(m.sum()),
                "positives": int(frame.loc[m, "y"].sum()),
                "meets_rule": bool(m.sum() >= cfg["fairness"]["min_group_n"]
                                   and frame.loc[m, "y"].sum() >= cfg["fairness"]["min_group_positives"]),
            })
    write_table(pd.DataFrame(ev_rows), out, "subgroup_evaluability")

    # ---- Tier 1 ------------------------------------------------------------
    tier_rows, cal_rows, fair_rows, dist_rows = [], [], [], []
    pred_rows = []
    oof = per_fold = None
    if 1 in want:
        log("Tier 1 — internal benchmark")
        oof, per_fold, t1 = tier1(frame, num_cols_all, cat_cols_all, cfg, out, args.quick)
        for key in models.LEARNERS:
            p = oof[key]
            m = np.isfinite(p)
            pred_rows.append(pd.DataFrame({
                "tier": 1, "model": models.MODEL_LABELS[key], "pair": "internal",
                "y": frame["y"].to_numpy()[m], "p": p[m]}))
        for _, r in t1.iterrows():
            cal_rows.append({"model": r["model"], "tier": 1, "brier": r["brier"],
                             "brier_calibration": r["brier_calibration"],
                             "brier_refinement": r["brier_refinement"], "ece": r["ece"]})
        y = frame["y"].to_numpy()
        for attr, col in [("gender", "grp_gender"), ("deprivation", "grp_deprivation")]:
            for key in models.LEARNERS:
                p = oof[key]
                m = np.isfinite(p)
                if not fairness.evaluable(y[m], frame[col].to_numpy()[m], "protected",
                                          cfg["fairness"]["min_group_n"],
                                          cfg["fairness"]["min_group_positives"]):
                    continue
                e = fairness.eod_at_budget(y[m], p[m], frame[col].to_numpy()[m],
                                           float(cfg["fairness"]["primary_budget"]))
                _, lo, hi = fairness.eod_ci(y[m], p[m], frame[col].to_numpy()[m],
                                            float(cfg["fairness"]["primary_budget"]),
                                            n_boot=min(nboot, 500), seed=seed)
                e.update({"attribute": attr, "model": models.MODEL_LABELS[key],
                          "model_key": key, "tier": 1, "pair": "internal",
                          "eod_lo": lo, "eod_hi": hi})
                fair_rows.append(e)

    # ---- Tiers 2 and 3 -----------------------------------------------------
    pairs = []
    if 2 in want:
        pairs += splits.tier2_pairs(frame)
    if 3 in want:
        pairs += splits.tier3_pairs(frame)

    overlap_rows = [{"tier": p.tier, "pair": p.name,
                     "source_registrations_before_rule": p.n_train_before_rule,
                     "removed_for_target_overlap": p.n_removed_overlap,
                     "pct_removed": round(p.pct_removed, 2),
                     "target_registrations": len(p.test_idx)} for p in pairs]
    if overlap_rows:
        write_table(pd.DataFrame(overlap_rows), out, "overlap_removed")

    per_pair_rows, shap_importances, t3_delta_by_model = [], {}, {m: [] for m in models.LEARNERS}
    X_all = frame[num_cols_all + cat_cols_all]
    y_all = frame["y"].to_numpy()

    t1_auc = {}
    if oof is not None:
        for key in models.LEARNERS:
            p = oof[key]
            m = np.isfinite(p)
            t1_auc[key] = metrics.auc_fn(y_all[m], p[m])

    reference_shap_model = None
    for pi, pair in enumerate(pairs):
        log(f"Tier {pair.tier} — {pair.name}  ({pi + 1}/{len(pairs)})")
        res, fitted = evaluate_pair(pair, frame, num_cols_all, cat_cols_all, cfg, args.quick)
        for key, r in res.items():
            if key in t1_auc and np.isfinite(r.get("roc_auc", np.nan)):
                r["delta_roc_auc_vs_tier1"] = t1_auc[key] - r["roc_auc"]
                if pair.tier == 3:
                    t3_delta_by_model[key].append(r["delta_roc_auc_vs_tier1"])
            per_pair_rows.append(r)
            pred_rows.append(pd.DataFrame({
                "tier": pair.tier, "model": r["model"], "pair": pair.name,
                "y": y_all[pair.test_idx], "p": fitted[key][1]}))
            cal_rows.append({"model": r["model"], "tier": pair.tier, "pair": pair.name,
                             "brier": r["brier"], "brier_calibration": r["brier_calibration"],
                             "brier_refinement": r["brier_refinement"], "ece": r["ece"]})

        # distances for this pair (SRQ2)
        d = distance.pair_distances(
            X_all.iloc[pair.train_idx], X_all.iloc[pair.test_idx],
            y_all[pair.train_idx], y_all[pair.test_idx])
        d.update({"tier": pair.tier, "pair": pair.name, "differs_by": pair.differs_by})
        best_key = max(res, key=lambda k: res[k].get("roc_auc", -1)
                       if np.isfinite(res[k].get("roc_auc", np.nan)) else -1)
        d["delta_roc_auc_best_model"] = res[best_key].get("delta_roc_auc_vs_tier1", np.nan)
        for key in models.LEARNERS:
            d[f"delta_roc_auc_{key}"] = res[key].get("delta_roc_auc_vs_tier1", np.nan)
        dist_rows.append(d)

        # fairness under transfer
        for attr, col in [("gender", "grp_gender"), ("deprivation", "grp_deprivation")]:
            g_t = frame[col].to_numpy()[pair.test_idx]
            y_t = y_all[pair.test_idx]
            if not fairness.evaluable(y_t, g_t, "protected",
                                      cfg["fairness"]["min_group_n"],
                                      cfg["fairness"]["min_group_positives"]):
                continue
            for key in models.LEARNERS:
                p_t = fitted[key][1]
                e = fairness.eod_at_budget(y_t, p_t, g_t,
                                           float(cfg["fairness"]["primary_budget"]))
                _, lo, hi = fairness.eod_ci(y_t, p_t, g_t,
                                            float(cfg["fairness"]["primary_budget"]),
                                            n_boot=min(nboot, 500), seed=seed)
                e.update({"attribute": attr, "model": models.MODEL_LABELS[key],
                          "model_key": key, "tier": pair.tier, "pair": pair.name,
                          "eod_lo": lo, "eod_hi": hi})
                fair_rows.append(e)

        # SHAP: primary analysis holds ONE Tier 3 source model constant
        if pair.tier == 3 and not args.skip_shap:
            if reference_shap_model is None:
                reference_shap_model = fitted["xgb"][0] if "xgb" in fitted else fitted["rf"][0]
            try:
                imp = explain.global_shap_importance(
                    reference_shap_model, X_all.iloc[pair.test_idx], seed=seed)
                shap_importances[pair.extra.get("held_out_module", pair.name)] = imp
            except Exception as exc:  # pragma: no cover
                log(f"    SHAP skipped for {pair.name}: {exc}")

    if per_pair_rows:
        pp = pd.DataFrame(per_pair_rows)
        write_table(pp, out, "tier3_per_pair")
        agg = (pp.groupby(["model", "model_key", "tier"], observed=True)
               .agg(roc_auc=("roc_auc", "mean"), aucpr=("aucpr", "mean"),
                    delta_roc_auc=("delta_roc_auc_vs_tier1", "mean"),
                    recall_at_20=("recall_at_20", "mean"),
                    n_pairs=("pair", "nunique")).reset_index())
        write_table(agg, out, "degradation_by_model_tier")
    if cal_rows:
        write_table(pd.DataFrame(cal_rows), out, "calibration_by_tier")
    if fair_rows:
        write_table(pd.DataFrame(fair_rows), out, "fairness_eod")

    # ---- Tier 4: external related-construct check --------------------------
    if 4 in want:
        log("Tier 4 - external robustness check (related construct, enrolment time)")
        t4_rows = tier4.run(frame, cfg, log=log)
        if t4_rows:
            per_pair_rows.extend(t4_rows)
            write_table(pd.DataFrame(t4_rows), out, "tier4_external")

    # ---- fairness mitigation: transferred vs oracle (Table 4.13) -----------
    if pairs and oof is not None:
        log("Fairness mitigation - transferred vs oracle thresholds")
        mit_rows = []
        t3 = [p for p in pairs if p.tier == 3]
        for pair in t3[:3]:   # three representative targets keeps the cost proportionate
            res, fitted = evaluate_pair(pair, frame, num_cols_all, cat_cols_all, cfg, args.quick)
            best = max(res, key=lambda k: res[k].get("roc_auc", -1)
                       if np.isfinite(res[k].get("roc_auc", np.nan)) else -1)
            p_t = fitted[best][1]
            src_idx = pair.train_idx
            p_s = models.predict_proba(fitted[best][0], X_all.iloc[src_idx])
            for attr, col in [("gender", "grp_gender"), ("deprivation", "grp_deprivation")]:
                g_s = frame[col].to_numpy()[src_idx]
                g_t = frame[col].to_numpy()[pair.test_idx]
                y_t = y_all[pair.test_idx]
                if not fairness.evaluable(y_t, g_t, "protected",
                                          cfg["fairness"]["min_group_n"],
                                          cfg["fairness"]["min_group_positives"]):
                    continue
                m = fairness.mitigation_report(
                    y_all[src_idx], p_s, g_s, y_t, p_t, g_t,
                    float(cfg["fairness"]["primary_budget"]))
                m.update({"attribute": attr, "pair": pair.name,
                          "model": models.MODEL_LABELS[best], "tier": pair.tier})
                mit_rows.append(m)
        if mit_rows:
            write_table(pd.DataFrame(mit_rows), out, "fairness_mitigation")

    # ---- imbalance treatment selection (Table 4.6) -------------------------
    if 2 in want:
        log("Imbalance treatment selection")
        try:
            write_table(sensitivity.imbalance_selection(
                frame, num_cols_all, cat_cols_all, cfg, log=log),
                out, "imbalance_selection")
        except Exception as exc:
            log(f"  imbalance selection skipped: {exc}")

    # ---- pre-specified sensitivity analyses (Table 4.18) -------------------
    if 3 in want and not args.skip_sensitivity:
        log("Pre-specified sensitivity analyses")
        base_auc = np.nanmean([r.get("roc_auc", np.nan) for r in per_pair_rows
                               if r.get("tier") == 3 and r.get("model_key") == "xgb"]) \
            if per_pair_rows else np.nan
        try:
            write_table(sensitivity.run(full_frame, frame, num_cols_all, cat_cols_all,
                                        cfg, base_auc, log=log), out, "sensitivity")
        except Exception as exc:
            log(f"  sensitivity analyses skipped: {exc}")

    # ---- SRQ2: distance vs degradation ------------------------------------
    if dist_rows:
        dd = pd.DataFrame(dist_rows)
        write_table(dd, out, "distance_measures")
        corr = []
        for measure in ["mean_psi", "max_psi", "jsd_top10",
                        "abs_prevalence_diff", "source_target_size_ratio"]:
            r, lo, hi, p = stats.spearman_ci(dd[measure], dd["delta_roc_auc_best_model"],
                                             n_boot=min(nboot, 500), seed=seed)
            corr.append({"distance_measure": measure, "spearman_rho": r,
                         "ci_lo": lo, "ci_hi": hi, "n_pairs": int(dd[measure].notna().sum()),
                         "p_uncorrected_exploratory": p})
        write_table(pd.DataFrame(corr), out, "distance_degradation_corr")

    # ---- SRQ4: explanation stability --------------------------------------
    if shap_importances and len(shap_importances) > 1:
        ref = sorted(shap_importances)[0]
        write_table(explain.vs_reference(shap_importances, ref), out, "shap_concordance_primary")
        write_table(explain.concordance(shap_importances), out, "shap_concordance_secondary")
        fs = explain.feature_rank_stability(shap_importances)
        write_table(fs, out, "shap_feature_stability")
        write_table(explain.block_stability(fs), out, "shap_block_stability")

    # ---- confirmatory family ----------------------------------------------
    conf = []
    n_tr = int(np.mean([len(p.train_idx) for p in pairs])) if pairs else 1
    n_te = int(np.mean([len(p.test_idx) for p in pairs])) if pairs else 1
    for spec in cfg["stats"]["confirmatory_family"]:
        if spec["kind"] == "delta_auc":
            d = t3_delta_by_model.get(spec["model"], [])
            t = stats.corrected_resampled_ttest(d, n_tr, n_te)
            conf.append({"comparison": f"delta ROC-AUC Tier1->Tier3, {models.MODEL_LABELS[spec['model']]}",
                         "hypothesis": spec["hypothesis"], "estimate": t["mean"],
                         "t": t["t"], "p_raw": t["p"], "n_pairs": len(d)})
        else:
            fr = pd.DataFrame(fair_rows)
            if fr.empty:
                continue
            a = spec["attribute"]
            internal = fr[(fr.tier == 1) & (fr.attribute == a)]["eod"]
            transferred = fr[(fr.tier == 3) & (fr.attribute == a)]["eod"]
            diffs = [v - internal.mean() for v in transferred if np.isfinite(v)] \
                if len(internal) else []
            t = stats.corrected_resampled_ttest(diffs, n_tr, n_te)
            conf.append({"comparison": f"delta EOD Tier1->Tier3, {a}",
                         "hypothesis": spec["hypothesis"], "estimate": t["mean"],
                         "t": t["t"], "p_raw": t["p"], "n_pairs": len(diffs)})
    if conf:
        cf = pd.DataFrame(conf)
        adj, rej = stats.holm_bonferroni(cf["p_raw"].to_numpy(), float(cfg["stats"]["alpha"]))
        cf["p_holm_adjusted"], cf["reject_at_alpha"] = adj, rej
        write_table(cf, out, "confirmatory_family")

    # ---- secondary fairness criteria (Table 4.14) --------------------------
    if fair_rows:
        fr = pd.DataFrame(fair_rows)
        cols = [c for c in ["attribute", "model", "tier", "pair", "fpr_difference",
                            "tpr_reference", "tpr_protected", "n_reference", "n_protected",
                            "threshold"] if c in fr.columns]
        write_table(fr[cols], out, "fairness_secondary")

    # ---- predictions, for the calibration and alert-budget figures ---------
    if pred_rows:
        preds = pd.concat(pred_rows, ignore_index=True)
        # keep the file manageable: the figures need distributional shape, not every row
        if len(preds) > 400_000:
            preds = preds.sample(400_000, random_state=seed)
        write_table(preds, out, "predictions")

    # ---- SHAP importance matrix (Figure 4.6) -------------------------------
    if shap_importances and len(shap_importances) > 1:
        mat = pd.DataFrame(shap_importances).fillna(0.0)
        mat = mat.loc[mat.mean(axis=1).sort_values(ascending=False).index[:25]]
        write_table(mat.reset_index(names="feature"), out, "shap_importance_matrix")

    # ---- figures and run log ----------------------------------------------
    try:
        make_figures(out)
    except Exception as exc:  # pragma: no cover
        log(f"figure generation skipped: {exc}")

    run_log["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_log["n_registrations_analysed"] = int(len(frame))
    run_log["n_registrations_before_estimand_filter"] = int(len(full_frame))
    run_log["n_pairs"] = len(pairs)
    with open(os.path.join(out, "run_log.json"), "w") as fh:
        json.dump(run_log, fh, indent=2)
    log(f"Done. Tables in {out}/tables, figures in {out}/figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
