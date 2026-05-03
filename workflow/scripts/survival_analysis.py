"""Survival analysis: do predicted CX scores carry prognostic information?

Pulls TCGA OS data (days_to_death / days_to_last_follow_up + vital
status) from GDC for the 289 labelled patients in our predictions,
fits Cox regression of predicted CX1/CX3/CX6/CX10/CX12 (continuous and
quartile) on OS, with cohort as a stratum to control for cancer-type
confounding.

This is the test that turns "we can predict CIN signatures from H&E"
into "predicted CIN signatures stratify patient outcomes."

Output:
- results/cv/.../survival_metrics.json
- figures/fig_survival_km_top3.{pdf,png}  Kaplan-Meier by predicted CX1 quartile
- figures/fig_survival_forest.{pdf,png}    Hazard ratios across signatures
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

BEST_DIR = Path("results/cv/uni_7cohort_sweep/h512_d0.3_lr0.0001")
GDC_CASES = "https://api.gdc.cancer.gov/cases"
TOP_SIGS = ["CX1", "CX3", "CX6", "CX10", "CX12"]


def fetch_survival(patient_ids: list[str]) -> pd.DataFrame:
    """Pull vital_status, days_to_death, days_to_last_follow_up from GDC."""
    rows = []
    chunk = 100
    for i in range(0, len(patient_ids), chunk):
        ids = patient_ids[i:i + chunk]
        params = {
            "filters": json.dumps({
                "op": "in",
                "content": {"field": "submitter_id", "value": ids}
            }),
            "expand": "demographic,diagnoses,follow_ups",
            "format": "json",
            "size": chunk,
        }
        r = requests.get(GDC_CASES, params=params, timeout=60)
        r.raise_for_status()
        for h in r.json().get("data", {}).get("hits", []):
            sid = h.get("submitter_id")
            d = h.get("demographic") or {}
            diags = h.get("diagnoses") or []
            dx0 = diags[0] if diags else {}
            vital = d.get("vital_status")
            dod = d.get("days_to_death")
            dlf = dx0.get("days_to_last_follow_up")
            # Sometimes follow-ups are richer; pull max non-null follow-up time.
            fups = h.get("follow_ups") or []
            if fups:
                fu_days = [f.get("days_to_follow_up") for f in fups
                           if f.get("days_to_follow_up") is not None]
                if fu_days:
                    dlf = max([x for x in (dlf, max(fu_days)) if x is not None])
            event = 1 if vital == "Dead" else 0
            duration = dod if event == 1 else dlf
            rows.append({
                "patient": sid,
                "vital_status": vital,
                "duration": duration,
                "event": event,
            })
        time.sleep(0.2)
    return pd.DataFrame(rows)


def main():
    pred = pd.read_csv(BEST_DIR / "predictions.tsv", sep="\t")

    surv_path = BEST_DIR / "survival_data.tsv"
    if surv_path.exists():
        surv = pd.read_csv(surv_path, sep="\t")
    else:
        print(f"fetching survival data for {pred['patient'].nunique()} patients")
        surv = fetch_survival(pred["patient"].unique().tolist())
        surv.to_csv(surv_path, sep="\t", index=False)
        print(f"wrote {surv_path}")

    df = pred.merge(surv, on="patient", how="inner")
    df = df.dropna(subset=["duration"])
    df["duration_yrs"] = df["duration"] / 365.25
    df = df[df["duration_yrs"] >= 0]
    print(f"survival cohort: {len(df)} patients ({int(df['event'].sum())} events)")

    # Per-cohort event count
    print("\nevents by cohort:")
    print(df.groupby("cancer_type").agg(n=("patient", "size"),
                                        events=("event", "sum")).to_string())

    # Cox regression per signature (continuous, stratified by cohort).
    cox_results = {}
    for sig in TOP_SIGS:
        d = df.dropna(subset=[f"{sig}_pred"]).copy()
        d["pred"] = d[f"{sig}_pred"].astype(float)
        # Z-score the predictor so HR is per-SD increase.
        d["pred_z"] = (d["pred"] - d["pred"].mean()) / d["pred"].std()
        try:
            cph = CoxPHFitter()
            cph.fit(d[["duration_yrs", "event", "pred_z", "cancer_type"]],
                    duration_col="duration_yrs", event_col="event",
                    strata=["cancer_type"])
            row = cph.summary.loc["pred_z"]
            cox_results[sig] = {
                "hr": float(row["exp(coef)"]),
                "hr_ci_lo": float(row["exp(coef) lower 95%"]),
                "hr_ci_hi": float(row["exp(coef) upper 95%"]),
                "p": float(row["p"]),
                "n": int(len(d)),
                "events": int(d["event"].sum()),
                "concordance": float(cph.concordance_index_),
            }
            print(f"\nCox HR for {sig} (per SD increase, cohort-stratified):")
            print(f"  HR = {cox_results[sig]['hr']:.3f}, "
                  f"95% CI [{cox_results[sig]['hr_ci_lo']:.3f}, "
                  f"{cox_results[sig]['hr_ci_hi']:.3f}], "
                  f"p = {cox_results[sig]['p']:.4f}, "
                  f"C = {cox_results[sig]['concordance']:.3f}")
        except Exception as e:
            cox_results[sig] = {"error": str(e)[:200]}
            print(f"  Cox failed for {sig}: {e}")

    # Forest plot of HRs.
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sigs_plot = [s for s in TOP_SIGS if "hr" in cox_results.get(s, {})]
    hrs = [cox_results[s]["hr"] for s in sigs_plot]
    los = [cox_results[s]["hr_ci_lo"] for s in sigs_plot]
    his = [cox_results[s]["hr_ci_hi"] for s in sigs_plot]
    ps = [cox_results[s]["p"] for s in sigs_plot]
    y = np.arange(len(sigs_plot))
    err_lo = np.array(hrs) - np.array(los)
    err_hi = np.array(his) - np.array(hrs)
    colors = ["#d62728" if hr > 1.0 else "#1f77b4" for hr in hrs]
    ax.errorbar(hrs, y, xerr=[err_lo, err_hi], fmt="o", ecolor="gray",
                capsize=3, markersize=7, markerfacecolor="white", lw=1)
    for i, (hr, c) in enumerate(zip(hrs, colors)):
        ax.scatter(hr, i, color=c, s=50, zorder=3)
    for i, (s, hr, p) in enumerate(zip(sigs_plot, hrs, ps)):
        ax.text(max(his) * 1.05, i, f"HR={hr:.2f}, p={p:.3f}",
                va="center", fontsize=9)
    ax.axvline(1.0, ls="--", color="gray", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(sigs_plot)
    ax.invert_yaxis()
    ax.set_xlabel("Cox HR per SD of predicted CX score (cohort-stratified)")
    ax.set_title(f"Predicted CX → overall survival, "
                 f"n={cox_results[sigs_plot[0]]['n']}, "
                 f"{cox_results[sigs_plot[0]]['events']} events")
    ax.set_xlim(0.6, max(his) * 1.4)
    fig.tight_layout()
    fig.savefig("figures/fig_survival_forest.pdf", bbox_inches="tight")
    fig.savefig("figures/fig_survival_forest.png", dpi=140,
                bbox_inches="tight")
    print(f"\nwrote figures/fig_survival_forest.{{pdf,png}}")

    # KM by predicted CX1 quartile.
    sig = "CX1"
    d = df.dropna(subset=[f"{sig}_pred"]).copy()
    d["q"] = pd.qcut(d[f"{sig}_pred"], 4, labels=["Q1 (low)", "Q2", "Q3",
                                                    "Q4 (high)"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    kmf = KaplanMeierFitter()
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    for q, color in zip(d["q"].cat.categories, palette):
        sub = d[d["q"] == q]
        if len(sub) == 0:
            continue
        kmf.fit(sub["duration_yrs"], sub["event"], label=f"{q} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, ci_show=False, color=color)
    lr = multivariate_logrank_test(d["duration_yrs"], d["q"], d["event"])
    ax.set_xlabel("Years from diagnosis")
    ax.set_ylabel("Overall survival probability")
    ax.set_title(f"OS by predicted {sig} quartile "
                 f"(log-rank p = {lr.p_value:.3g})")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, max(15, d["duration_yrs"].quantile(0.99)))
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig("figures/fig_survival_km_cx1.pdf", bbox_inches="tight")
    fig.savefig("figures/fig_survival_km_cx1.png", dpi=140,
                bbox_inches="tight")
    print(f"wrote figures/fig_survival_km_cx1.{{pdf,png}}")
    print(f"  log-rank p = {lr.p_value:.4g}, test_statistic = {lr.test_statistic:.2f}")

    cox_results["_meta"] = {
        "n_patients_total": int(len(df)),
        "n_events": int(df["event"].sum()),
        "median_followup_yrs": float(df.loc[df["event"] == 0,
                                            "duration_yrs"].median()),
        "log_rank_p_cx1_quartile": float(lr.p_value),
    }
    out = BEST_DIR / "survival_metrics.json"
    out.write_text(json.dumps(cox_results, indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
