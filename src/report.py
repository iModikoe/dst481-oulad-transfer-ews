"""Output writing: CSVs named for the Chapter 4 table shells, plus figures."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "font.family": "serif",
})


def write_table(df: pd.DataFrame, out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, "tables", f"{name}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"    wrote tables/{name}.csv  ({len(df)} rows)")
    return path


AVAILABILITY = {
    "demographic/enrolment": "Known at registration; day <= 0",
    "pre-start engagement": "date < 0",
    "early engagement": "0 <= date <= 27",
    "early assessment": "assessment and submission dates <= 27",
}

SOURCE_TABLE = {
    "gender": "studentInfo", "region": "studentInfo", "disability": "studentInfo",
    "highest_education": "studentInfo", "imd_decile": "studentInfo",
    "imd_missing": "studentInfo", "age_ord": "studentInfo", "edu_ord": "studentInfo",
    "num_of_prev_attempts": "studentInfo", "studied_credits": "studentInfo",
    "days_reg_before_start": "studentRegistration",
    "first_score": "studentAssessment", "first_score_missing": "studentAssessment",
    "submitted_on_time": "studentAssessment", "has_early_assessment": "assessments",
}


def _block_of(f: str) -> str:
    from .explain import _block_of as b
    blk = b(f)
    if blk == "early engagement" and f.startswith("pre_") or f in ("any_pre_activity", "log_pre_clicks"):
        return "pre-start engagement"
    return "demographic/enrolment" if blk == "demographic/enrolment" else blk


def data_dictionary_rows(frame, num_cols, cat_cols) -> list[dict]:
    """Generated from the same column lists the pipeline fits on, so Appendix A
    cannot drift out of step with the code."""
    rows = []
    for c in list(num_cols) + list(cat_cols):
        blk = _block_of(c)
        rows.append({
            "variable": c,
            "block": blk,
            "source_table": SOURCE_TABLE.get(c, "studentVle / vle"),
            "availability_rule": AVAILABILITY.get(blk, "0 <= date <= 27"),
            "dtype": "categorical" if c in cat_cols else "numeric",
            "transformation": "one-hot; modal imputation" if c in cat_cols
            else "median imputation with missingness indicator",
            "missing_pct": round(float(frame[c].isna().mean() * 100), 2) if c in frame else np.nan,
            "used_in_tiers": "1-4" if blk == "demographic/enrolment" else "1-3",
        })
    return rows


# --------------------------------------------------------------------------
def _read(out_dir: str, name: str):
    p = os.path.join(out_dir, "tables", f"{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def make_figures(out_dir: str) -> None:
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Figure 4.2 — degradation dot plot (the chapter's headline figure)
    pp = _read(out_dir, "tier3_per_pair")
    if pp is not None and "delta_roc_auc_vs_tier1" in pp.columns:
        g = (pp.dropna(subset=["delta_roc_auc_vs_tier1"])
             .groupby(["model", "tier"], observed=True)["delta_roc_auc_vs_tier1"]
             .agg(["mean", "std", "count"]).reset_index())
        if len(g):
            fig, ax = plt.subplots(figsize=(6.2, 0.5 * len(g) + 1.4))
            labels = [f"{r.model} · Tier {int(r.tier)}" for r in g.itertuples()]
            ypos = np.arange(len(g))
            err = (g["std"] / np.sqrt(g["count"].clip(lower=1))) * 1.96
            ax.errorbar(g["mean"], ypos, xerr=err.fillna(0), fmt="o",
                        color="#1b1b1b", ecolor="#8a8a8a", capsize=3, markersize=4)
            ax.axvline(0, color="#c0392b", lw=1, ls="--")
            ax.set_yticks(ypos, labels)
            ax.set_xlabel("Δ ROC-AUC  (internal − transferred; positive = deterioration)")
            ax.set_title("Transfer degradation by model family and tier", loc="left")
            ax.invert_yaxis()
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, "degradation_dotplot.png"))
            plt.close(fig)

    # Figure 4.4 — distance against degradation
    dd = _read(out_dir, "distance_measures")
    if dd is not None and "delta_roc_auc_best_model" in dd.columns:
        measures = [m for m in ["mean_psi", "max_psi", "jsd_top10",
                                "abs_prevalence_diff", "source_target_size_ratio"]
                    if m in dd.columns]
        if measures:
            fig, axes = plt.subplots(1, len(measures), figsize=(2.5 * len(measures), 2.8),
                                     sharey=True)
            axes = np.atleast_1d(axes)
            for ax, m in zip(axes, measures):
                for tier, sub in dd.groupby("tier", observed=True):
                    ax.scatter(sub[m], sub["delta_roc_auc_best_model"], s=22,
                               label=f"Tier {int(tier)}")
                ax.set_xlabel(m.replace("_", " "))
                ax.axhline(0, color="#c0392b", lw=0.8, ls="--")
            axes[0].set_ylabel("Δ ROC-AUC")
            axes[-1].legend(frameon=False, fontsize=7)
            fig.suptitle("Source–target distance against transfer degradation", x=0.01, ha="left")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, "distance_vs_degradation.png"))
            plt.close(fig)

    # Figure 4.5 — fairness: EOD by tier
    fr = _read(out_dir, "fairness_eod")
    if fr is not None and "eod" in fr.columns:
        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        # Tiers are discrete. Plotting them on a continuous axis produced ticks at
        # 1.25, 1.75 and so on, which name validation tiers that do not exist.
        tiers = sorted(int(t) for t in fr["tier"].dropna().unique())
        xpos = {t: i for i, t in enumerate(tiers)}
        for attr, sub in fr.groupby("attribute", observed=True):
            byt = sub.groupby("tier", observed=True)["eod"].mean()
            xs = [xpos[int(t)] for t in byt.index]
            ax.plot(xs, byt.values, marker="o", label=attr)
        ax.axhline(0, color="#c0392b", lw=0.8, ls="--")
        ax.set_xticks(list(xpos.values()), [f"Tier {t}" for t in tiers])
        ax.set_xlim(-0.25, len(tiers) - 0.75)
        ax.set_xlabel("Validation tier (discrete)")
        ax.set_ylabel("Equal opportunity difference")
        ax.set_title("Subgroup disparity under increasing context shift", loc="left")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fairness_tradeoff.png"))
        plt.close(fig)

    # Figure 4.7 — SHAP rank concordance
    sc = _read(out_dir, "shap_concordance_secondary")
    if sc is not None and len(sc):
        ctxs = sorted(set(sc["context_a"]) | set(sc["context_b"]))
        mat = pd.DataFrame(np.eye(len(ctxs)), index=ctxs, columns=ctxs)
        for r in sc.itertuples():
            mat.loc[r.context_a, r.context_b] = r.kendall_tau_b
            mat.loc[r.context_b, r.context_a] = r.kendall_tau_b
        # The diagonal is 1.0 by construction (a ranking against itself) and says
        # nothing about transfer, so it is masked out. The colour range is set to
        # the observed off-diagonal range rather than the full [-1, 1]: on a fixed
        # [-1, 1] scale a set of values all sitting between 0.85 and 1.00 renders
        # as a uniform block and hides exactly the between-context differences
        # SRQ4 asks about. Cells are annotated so the reader has the numbers.
        arr = mat.to_numpy(dtype=float).copy()
        np.fill_diagonal(arr, np.nan)
        finite = arr[np.isfinite(arr)]
        lo = float(np.floor(finite.min() * 20) / 20) if finite.size else 0.0
        hi = float(np.ceil(finite.max() * 20) / 20) if finite.size else 1.0
        if hi - lo < 0.05:
            lo, hi = lo - 0.025, hi + 0.025
        fig, ax = plt.subplots(figsize=(0.78 * len(ctxs) + 2.4, 0.78 * len(ctxs) + 1.9))
        cmap = plt.get_cmap("RdBu").copy()
        cmap.set_bad("#eeeeee")
        im = ax.imshow(np.ma.masked_invalid(arr), vmin=lo, vmax=hi, cmap=cmap)
        for i in range(len(ctxs)):
            for j in range(len(ctxs)):
                if i == j:
                    continue
                v = arr[i, j]
                if not np.isfinite(v):
                    continue
                shade = (v - lo) / (hi - lo) if hi > lo else 0.5
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if shade > 0.75 or shade < 0.25 else "#1b1b1b")
        ax.set_xticks(range(len(ctxs)), ctxs, rotation=45, ha="right")
        ax.set_yticks(range(len(ctxs)), ctxs)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8,
                     label=f"Kendall's tau-b  (scale {lo:.2f}-{hi:.2f}; diagonal masked)")
        ax.set_title("Explanation rank concordance across contexts", loc="left")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "concordance_matrix.png"))
        plt.close(fig)

    # Figure 4.1 — reliability diagrams by model and tier (§3.8.2 requires one per tier)
    pr = _read(out_dir, "predictions")
    if pr is not None and {"y", "p", "tier", "model"} <= set(pr.columns):
        tiers = sorted(pr["tier"].unique())
        mods = sorted(pr["model"].unique())
        fig, axes = plt.subplots(len(tiers), len(mods),
                                 figsize=(2.6 * len(mods), 2.4 * len(tiers)),
                                 squeeze=False, sharex=True, sharey=True)
        for i, t in enumerate(tiers):
            for j, mdl in enumerate(mods):
                ax = axes[i][j]
                sub = pr[(pr.tier == t) & (pr.model == mdl)]
                ax.plot([0, 1], [0, 1], color="#c0392b", lw=0.8, ls="--")
                if len(sub) >= 50:
                    q = np.quantile(sub["p"], np.linspace(0, 1, 11))
                    q[0], q[-1] = -np.inf, np.inf
                    idx = np.digitize(sub["p"], q[1:-1], right=True)
                    xs, ys = [], []
                    for b in range(10):
                        m = idx == b
                        if m.sum() == 0:
                            continue
                        xs.append(sub["p"][m].mean()); ys.append(sub["y"][m].mean())
                    ax.plot(xs, ys, marker="o", ms=3, color="#1b1b1b")
                if i == 0:
                    ax.set_title(mdl, fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"Tier {int(t)}\nobserved", fontsize=8)
                if i == len(tiers) - 1:
                    ax.set_xlabel("predicted", fontsize=8)
        fig.suptitle("Reliability diagrams by model family and tier", x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "reliability_grid.png"))
        plt.close(fig)

        # Figure 4.3 — recall across the alert-budget range
        budgets = np.linspace(0.05, 0.50, 19)
        fig, axes = plt.subplots(1, len(tiers), figsize=(2.9 * len(tiers), 2.9),
                                 squeeze=False, sharey=True)
        for i, t in enumerate(tiers):
            ax = axes[0][i]
            for mdl in mods:
                sub = pr[(pr.tier == t) & (pr.model == mdl)]
                if len(sub) < 50 or sub["y"].nunique() < 2:
                    continue
                yv, pv = sub["y"].to_numpy(), sub["p"].to_numpy()
                rec = []
                for b in budgets:
                    k = max(1, int(np.ceil(b * len(pv))))
                    top = np.argsort(-pv, kind="mergesort")[:k]
                    rec.append(yv[top].sum() / max(yv.sum(), 1))
                ax.plot(budgets, rec, lw=1.2, label=mdl)
            for b in (0.10, 0.20):
                ax.axvline(b, color="#8a8a8a", lw=0.6, ls=":")
            ax.set_title(f"Tier {int(t)}", fontsize=9)
            ax.set_xlabel("alert budget")
        axes[0][0].set_ylabel("recall")
        axes[0][-1].legend(frameon=False, fontsize=7)
        fig.suptitle("Recall across the alert-budget range", x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "alert_budget_curves.png"))
        plt.close(fig)

    # Figure 4.6 — SHAP global importance by context
    fs = _read(out_dir, "shap_importance_matrix")
    if fs is not None and len(fs):
        mat = fs.set_index(fs.columns[0])
        fig, ax = plt.subplots(figsize=(0.55 * mat.shape[1] + 3.2, 0.26 * mat.shape[0] + 1.8))
        im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="viridis")
        ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(mat.shape[0]), mat.index, fontsize=6)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.7, label="mean |SHAP|")
        ax.set_title("Global SHAP importance by context", loc="left")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "shap_heatmap.png"))
        plt.close(fig)

    print(f"    wrote figures to {fig_dir}")
