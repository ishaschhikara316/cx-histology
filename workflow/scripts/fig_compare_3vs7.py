"""3-cohort vs 7-cohort cohort-scaling comparison figure.

Shows per-signature ρ for both runs side by side, plus a delta panel,
to make the cohort-scaling lift legible at a glance.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_per_sig(summary_path: str) -> tuple[np.ndarray, np.ndarray]:
    s = json.load(open(summary_path))
    if "pan_cancer_spearman_mean" in s:
        m = np.array(s["pan_cancer_spearman_mean"], dtype=float)
        sd = np.array(s["pan_cancer_spearman_std"], dtype=float)
    else:
        # Fall back to recomputing from fold_metrics
        rho = np.array([
            [float("nan") if (x is None or x != x) else float(x)
             for x in f["spearman_per_sig"]]
            for f in s["fold_metrics"]
        ])
        m = np.nanmean(rho, axis=0)
        sd = np.nanstd(rho, axis=0)
    return m, sd


def main():
    old_path = "results/cv/uni_3cohort/cv_summary.json"
    new_path = "results/cv/uni_7cohort_sweep/h512_d0.3_lr0.0001/cv_summary.json"

    old_mean, old_std = load_per_sig(old_path)
    new_mean, new_std = load_per_sig(new_path)

    sigs = [f"CX{i}" for i in range(1, 18)]
    n_sigs = 17
    x = np.arange(n_sigs)
    bw = 0.4

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5),
                                   gridspec_kw={"height_ratios": [3, 1.2]},
                                   sharex=True)

    # Top panel: side-by-side ρ
    bars_old = ax1.bar(x - bw / 2, np.nan_to_num(old_mean, nan=0.0), bw,
                       yerr=np.nan_to_num(old_std, nan=0.0),
                       label="3-cohort (n=87)", color="#9aa6b2", capsize=2)
    bars_new = ax1.bar(x + bw / 2, np.nan_to_num(new_mean, nan=0.0), bw,
                       yerr=np.nan_to_num(new_std, nan=0.0),
                       label="7-cohort (n=289), best HP h512_d0.3_lr1e-4",
                       color="#1f77b4", capsize=2)

    # Mark NaN signatures explicitly
    for i, (a, b) in enumerate(zip(old_mean, new_mean)):
        if np.isnan(a):
            ax1.text(i - bw / 2, 0.02, "n/a", ha="center", fontsize=7, color="gray")
        if np.isnan(b):
            ax1.text(i + bw / 2, 0.02, "n/a", ha="center", fontsize=7, color="gray")
        # Star annotation for H1 hits in 7-cohort
        if not np.isnan(b) and b >= 0.30:
            ax1.text(i + bw / 2, b + new_std[i] + 0.03, "★",
                     ha="center", fontsize=12, color="#2ca02c")

    ax1.axhline(0.30, ls="--", color="black", alpha=0.4, lw=1,
                label="H1 target ρ ≥ 0.30")
    ax1.axhline(0.0, color="black", lw=0.4)
    ax1.set_ylabel("Spearman ρ (5-fold CV, mean ± std)")
    ax1.set_title("Cohort-scaling effect on H&E → CIN signature prediction "
                  "(UNI + gated-attention MIL)")
    ax1.set_ylim(-0.85, 0.85)
    ax1.legend(loc="lower left", fontsize=9)

    # Bottom panel: delta (7-cohort − 3-cohort)
    delta = new_mean - old_mean
    colors_delta = []
    for d in delta:
        if np.isnan(d):
            colors_delta.append("#cccccc")
        elif d > 0.10:
            colors_delta.append("#2ca02c")
        elif d > 0.0:
            colors_delta.append("#7fbf7b")
        elif d > -0.10:
            colors_delta.append("#fdae61")
        else:
            colors_delta.append("#d62728")
    ax2.bar(x, np.nan_to_num(delta, nan=0.0), color=colors_delta)
    ax2.axhline(0.0, color="black", lw=0.5)
    ax2.set_ylabel("Δρ (7c − 3c)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(sigs, rotation=45)
    ax2.set_ylim(-0.6, 0.7)

    # Footer text with summary stats
    n_h1_old = int(np.sum((~np.isnan(old_mean)) & (old_mean >= 0.30)))
    n_h1_new = int(np.sum((~np.isnan(new_mean)) & (new_mean >= 0.30)))
    summary = (f"mean active ρ: 3-cohort {np.nanmean(old_mean):+.3f} → "
               f"7-cohort {np.nanmean(new_mean):+.3f}    |    "
               f"H1 hits (ρ ≥ 0.30): {n_h1_old} → {n_h1_new}    |    "
               f"max ρ: {np.nanmax(old_mean):+.3f} → {np.nanmax(new_mean):+.3f}")
    fig.text(0.5, -0.01, summary, ha="center", fontsize=9, style="italic")

    fig.tight_layout()
    out_pdf = Path("figures/compare_3vs7_cohort_rho.pdf")
    out_png = Path("figures/compare_3vs7_cohort_rho.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_pdf} and {out_png}")
    print(summary)


if __name__ == "__main__":
    main()
