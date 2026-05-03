"""Calibration: predicted vs actual scatter for the q < 0.001 signatures.

Spearman ρ tells you about ranking, not about whether the predicted
*magnitude* of CIN signature activity matches reality. For downstream
clinical use (e.g. stratifying patients by predicted CX1 quartile), we
want to know if predictions are well-calibrated, slope ≈ 1, intercept
≈ 0, or systematically biased.

Output:
- figures/fig_calibration_top5.{pdf,png} — 5-panel scatter
- results/cv/.../calibration_metrics.json — per-sig slope/intercept/MAE/R²
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, linregress

BEST_DIR = Path("results/cv/uni_7cohort_sweep/h512_d0.3_lr0.0001")
TOP_SIGS = ["CX1", "CX3", "CX6", "CX10", "CX12"]


def main():
    pred = pd.read_csv(BEST_DIR / "predictions.tsv", sep="\t")
    metrics = {}
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6), sharey=False)

    for ax, sig in zip(axes, TOP_SIGS):
        t = pred[f"{sig}_true"].values
        p = pred[f"{sig}_pred"].values
        rho, _ = spearmanr(p, t)
        r_pearson, _ = pearsonr(p, t)
        slope, intercept, _, _, _ = linregress(t, p)
        mae = float(np.mean(np.abs(p - t)))
        r2 = 1 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2)
        metrics[sig] = {
            "spearman": float(rho),
            "pearson": float(r_pearson),
            "slope_actual_to_pred": float(slope),
            "intercept": float(intercept),
            "mae": mae,
            "r2": float(r2),
            "n": int(len(t)),
            "true_range": [float(t.min()), float(t.max())],
            "pred_range": [float(p.min()), float(p.max())],
        }

        # Color by cohort
        cohorts = pred["cancer_type"].values
        cohort_palette = {c: f"C{i}" for i, c in enumerate(sorted(set(cohorts)))}
        colors = [cohort_palette[c] for c in cohorts]
        ax.scatter(t, p, c=colors, s=14, alpha=0.7, edgecolor="white",
                   linewidths=0.3)

        # Identity line
        lo = min(t.min(), p.min())
        hi = max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], color="gray", lw=0.7, ls="--",
                label="y = x (perfect)")

        # Linear fit line (cohort-mixed)
        x_fit = np.linspace(t.min(), t.max(), 50)
        ax.plot(x_fit, slope * x_fit + intercept, color="red", lw=1.0,
                label=f"fit: slope={slope:.2f}")

        ax.set_xlabel(f"{sig} true")
        ax.set_ylabel(f"{sig} predicted")
        ax.set_title(f"{sig}: ρ={rho:.3f} r={r_pearson:.3f}\nslope={slope:.2f}, "
                     f"int={intercept:+.3f}", fontsize=10)
        ax.legend(loc="best", fontsize=7)

    # Add a small cohort legend below the figure
    cohorts = sorted(set(pred["cancer_type"].values))
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=f"C{i}", markersize=7, label=c)
               for i, c in enumerate(cohorts)]
    fig.legend(handles=handles, loc="lower center", ncol=len(cohorts),
               fontsize=8, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Calibration on q<0.001 signatures (best HP, n=289)", y=1.02)
    fig.tight_layout()
    fig.savefig("figures/fig_calibration_top5.pdf", bbox_inches="tight")
    fig.savefig("figures/fig_calibration_top5.png", dpi=140,
                bbox_inches="tight")
    print("wrote figures/fig_calibration_top5.{pdf,png}")

    out = BEST_DIR / "calibration_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {out}")
    print()
    print(f"{'sig':<6}{'slope':>8}{'intercept':>11}{'pearson':>10}"
          f"{'spearman':>10}{'mae':>8}{'r2':>8}")
    print("-" * 60)
    for s in TOP_SIGS:
        m = metrics[s]
        print(f"{s:<6}{m['slope_actual_to_pred']:>+8.3f}"
              f"{m['intercept']:>+11.3f}{m['pearson']:>+10.3f}"
              f"{m['spearman']:>+10.3f}{m['mae']:>8.4f}{m['r2']:>+8.3f}")


if __name__ == "__main__":
    main()
