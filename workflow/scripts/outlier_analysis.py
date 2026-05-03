"""Outlier / hard-case patient analysis.

For each patient, compute a per-signature standardised residual
(pred - true) / std_of_residuals_for_that_sig, then summarise per
patient as the L2 norm across active signatures. Surface the worst-fit
patients and check whether they cluster by cohort, label magnitude,
or other signal.

Output:
- results/cv/.../outlier_patients.tsv (ranked)
- figures/fig_outliers_per_cohort.{pdf,png} (cohort vs residual boxplot)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from workflow.scripts._common import CX_COLUMNS

BEST_DIR = Path("results/cv/uni_7cohort_sweep/h512_d0.3_lr0.0001")


def main():
    pred = pd.read_csv(BEST_DIR / "predictions.tsv", sep="\t")
    n = len(pred)

    # Identify active signatures from the patient_bootstrap.json
    boot = json.load(open(BEST_DIR / "patient_bootstrap.json"))
    active = [s["signature"] for s in boot["signatures"]
              if s.get("status") == "active"]
    print(f"using {len(active)} active signatures: {active}")

    residuals = np.zeros((n, len(active)))
    for k, sig in enumerate(active):
        d = pred[f"{sig}_pred"].values - pred[f"{sig}_true"].values
        std = d.std()
        residuals[:, k] = d / std if std > 0 else d
    pred["resid_l2"] = np.linalg.norm(residuals, axis=1)
    pred["resid_max_sig"] = [active[i] for i in np.argmax(np.abs(residuals), axis=1)]
    pred["resid_max_zscore"] = np.max(np.abs(residuals), axis=1)

    # Top-20 outliers
    out = pred[["patient", "cancer_type", "fold", "resid_l2",
                "resid_max_sig", "resid_max_zscore"]].sort_values(
        "resid_l2", ascending=False).head(20)
    out_path = BEST_DIR / "outlier_patients.tsv"
    pred.sort_values("resid_l2", ascending=False).to_csv(
        out_path, sep="\t", index=False)
    print(f"wrote {out_path} ({len(pred)} rows)")

    print("\nTop 20 worst-fit patients:")
    for _, r in out.iterrows():
        print(f"  {r['patient']}  cohort={r['cancer_type']}  "
              f"fold={int(r['fold'])}  L2={r['resid_l2']:.2f}  "
              f"worst_sig={r['resid_max_sig']} (z={r['resid_max_zscore']:.2f})")

    # Per-cohort residual L2 distribution.
    fig, ax = plt.subplots(figsize=(8.5, 4))
    cohort_order = pred.groupby("cancer_type")["resid_l2"].median(
        ).sort_values(ascending=False).index
    sns.boxplot(data=pred, x="cancer_type", y="resid_l2",
                order=cohort_order, ax=ax, color="#1f77b4",
                showfliers=True, fliersize=3)
    sns.stripplot(data=pred, x="cancer_type", y="resid_l2",
                  order=cohort_order, ax=ax, color="black", size=2.5,
                  alpha=0.5, jitter=0.25)
    ax.set_xlabel("Cohort")
    ax.set_ylabel("Residual L2 across active signatures")
    ax.set_title("Per-cohort distribution of patient-level residuals")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig("figures/fig_outliers_per_cohort.pdf", bbox_inches="tight")
    fig.savefig("figures/fig_outliers_per_cohort.png", dpi=140,
                bbox_inches="tight")
    print(f"\nwrote figures/fig_outliers_per_cohort.{{pdf,png}}")

    # Cohort residual summary
    print("\nResidual L2 summary by cohort:")
    summary = pred.groupby("cancer_type").agg(
        n=("resid_l2", "size"),
        median=("resid_l2", "median"),
        mean=("resid_l2", "mean"),
        max=("resid_l2", "max"),
    ).round(2).sort_values("median", ascending=False)
    print(summary.to_string())

    # Which signature is worst for each cohort?
    print("\nWorst-fit signature per cohort (count of patients with that sig as worst):")
    for c in sorted(pred.cancer_type.unique()):
        sub = pred[pred.cancer_type == c]
        if len(sub) == 0:
            continue
        worst_sig_counts = sub["resid_max_sig"].value_counts().head(2)
        print(f"  {c}: " + ", ".join(
            f"{s}({n})" for s, n in worst_sig_counts.items()))


if __name__ == "__main__":
    main()
