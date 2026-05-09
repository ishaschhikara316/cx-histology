"""Figure: cohort-scaling decomposition (the headline methodological finding).

Two panels:
  A) Mean active rho vs training cohort count, with the same-patient
     re-evaluation overlay (21-cohort model scored on the 7- and 15-cohort
     subsets). The vertical drop on the 7-subset and 15-subset slices is
     the cancer-type-classification component of the small-cohort signal.
  B) Per-signature paired bars on the same 289 patients (the 7-cohort
     training pool): 7-cohort native ρ vs 21-cohort applied ρ. Top-5
     signatures from the 7-cohort run (CX1, CX3, CX12, CX6, CX10).

Inputs (all already on disk):
  - results/cv/uni_7cohort_sweep/h512_d0.3_lr0.0001/patient_bootstrap.json
  - results/cv/uni_15cohort_final/patient_bootstrap.json
  - results/cv/uni_21cohort_final/patient_bootstrap.json
  - results/cv/uni_21cohort_final/cohort_scaling_summary.json

Outputs:
  - figures/fig_cohort_scaling.pdf
  - figures/fig_cohort_scaling.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_bootstrap_rho(path: str) -> dict[str, float]:
    with open(path) as f:
        d = json.load(f)
    return {s["signature"]: s["rho"] for s in d["signatures"]
            if s.get("status") == "active"}


def main() -> None:
    base = Path(".")
    rho7 = load_bootstrap_rho(
        base / "results/cv/uni_7cohort_sweep/h512_d0.3_lr0.0001/"
        "patient_bootstrap.json")
    rho15 = load_bootstrap_rho(
        base / "results/cv/uni_15cohort_final/patient_bootstrap.json")
    rho21 = load_bootstrap_rho(
        base / "results/cv/uni_21cohort_final/patient_bootstrap.json")
    with open(base / "results/cv/uni_21cohort_final/"
              "cohort_scaling_summary.json") as f:
        scaling = json.load(f)

    # Panel A. Mean active rho vs cohort count, with the
    # same-patient subset re-evaluation overlay.
    # Use the same aggregation as Table 1 of the manuscript
    # (cohort_scaling_summary's "rho" field: pooled across signatures
    # at each scale) rather than mean-of-per-signature, so the figure
    # numbers match the table exactly.
    n_cohorts = [7, 15, 21]
    mean_native = [
        scaling["native_7cohort_on_7"]["rho"],
        scaling["native_15cohort_on_15"]["rho"],
        scaling["native_21cohort_on_21"]["rho"],
    ]
    n_patients = [
        scaling["native_7cohort_on_7"]["n"],
        scaling["native_15cohort_on_15"]["n"],
        3855,  # 21-cohort active patients per the resume note
    ]
    rho_21on7 = scaling["21cohort_on_7subset"]["rho"]
    rho_21on15 = scaling["21cohort_on_15subset"]["rho"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                             gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    ax.plot(n_cohorts, mean_native, "o-", color="#1f77b4", linewidth=2,
            markersize=9, label="Native (trained and tested at this scale)")
    # Same-patient subset overlay points: 21-cohort model scored on the
    # 7-subset and the 15-subset patient pools.
    ax.plot([7], [rho_21on7], "s", color="#d62728", markersize=10,
            label="21-cohort model applied to subset patients")
    ax.plot([15], [rho_21on15], "s", color="#d62728", markersize=10)
    # Drop arrows showing the within-patient attenuation.
    ax.annotate("", xy=(7, rho_21on7), xytext=(7, mean_native[0]),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.2))
    ax.annotate("", xy=(15, rho_21on15), xytext=(15, mean_native[1]),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.2))
    for i, (n, r) in enumerate(zip(n_cohorts, mean_native)):
        ax.text(n, r + 0.011, f"{r:+.3f}", ha="center", fontsize=9,
                color="#1f77b4")
    ax.text(7, rho_21on7 - 0.015, f"{rho_21on7:+.3f}", ha="center",
            fontsize=9, color="#d62728")
    ax.text(15, rho_21on15 - 0.015, f"{rho_21on15:+.3f}", ha="center",
            fontsize=9, color="#d62728")
    ax.set_xticks(n_cohorts)
    ax.set_xticklabels([f"{c}\n(n={n:,})" for c, n in
                        zip(n_cohorts, n_patients)])
    ax.set_xlabel("training cohort count")
    ax.set_ylabel("mean active pooled Spearman ρ")
    ax.set_title("A. Cohort-scaling regression of effect size",
                 fontsize=11, loc="left")
    ax.set_ylim(0, max(mean_native) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)

    # Panel B. Per-signature paired bars on the 289-patient 7-cohort pool.
    # Native 7-cohort vs 21-cohort applied.
    sigs = ["CX1", "CX3", "CX12", "CX6", "CX10"]
    native = [rho7[s] for s in sigs]
    applied = [scaling["21cohort_on_7subset"]["per_sig"][s] for s in sigs]
    x = np.arange(len(sigs))
    w = 0.38
    ax = axes[1]
    bars1 = ax.bar(x - w/2, native, w, color="#1f77b4",
                   label="7-cohort native (n=289)")
    bars2 = ax.bar(x + w/2, applied, w, color="#d62728",
                   label="21-cohort applied to same 289")
    for bx, bv in zip(bars1, native):
        ax.text(bx.get_x() + w/2, bv + 0.008, f"{bv:+.2f}",
                ha="center", fontsize=8.5, color="#1f77b4")
    for bx, bv in zip(bars2, applied):
        ax.text(bx.get_x() + w/2, bv + 0.008, f"{bv:+.2f}",
                ha="center", fontsize=8.5, color="#d62728")
    ax.axhline(0.30, ls="--", color="#888", lw=0.9)
    ax.text(len(sigs) - 0.5, 0.305, "H1 threshold ρ = 0.30",
            fontsize=8, color="#666", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(sigs)
    ax.set_ylabel("pooled Spearman ρ")
    ax.set_title("B. Same patients, larger model, smaller effect size",
                 fontsize=11, loc="left")
    ax.set_ylim(0, max(native) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)

    fig.suptitle(
        "Figure 2. Pan-cohort ρ partly reflects cancer-type "
        "classification: the same patients yield smaller effect sizes "
        "under a more diverse training distribution.",
        fontsize=10, y=1.005, x=0.02, ha="left")
    fig.tight_layout()
    out_pdf = base / "figures/fig_cohort_scaling.pdf"
    out_png = base / "figures/fig_cohort_scaling.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"wrote {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
