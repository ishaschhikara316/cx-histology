"""Fig 2 — UNI HP-sweep heatmap, per-signature ρ across the 6 combos.

Replaces the original encoder-comparison fig2 until H-optimus-0 sweep
exists. Signatures × HP combos heatmap with annotated values.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    sweep_dir = Path("results/cv/uni_7cohort_sweep")
    rows = []
    for cv_path in sorted(sweep_dir.glob("*/cv_summary.json")):
        combo = cv_path.parent.name
        s = json.load(open(cv_path))
        m = s["pan_cancer_spearman_mean"]
        for k, v in enumerate(m):
            rows.append({
                "combo": combo,
                "signature": f"CX{k + 1}",
                "rho": float("nan") if v is None or (v != v) else float(v),
            })
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="signature", columns="combo", values="rho")
    sigs_order = [f"CX{i}" for i in range(1, 18)]
    pivot = pivot.reindex(sigs_order)

    # Order columns by mean active ρ across signatures so the best HP is on the right
    col_score = pivot.mean(axis=0, skipna=True).sort_values(ascending=True)
    pivot = pivot[col_score.index]

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    sns.heatmap(pivot, annot=True, fmt=".2f", vmin=-0.3, vmax=0.5,
                cmap="RdBu_r", center=0,
                cbar_kws={"label": "Spearman ρ (mean across 5 folds)"},
                ax=ax, linewidths=0.3, linecolor="white",
                annot_kws={"size": 8})
    # Mark inactive signatures
    for i, sig in enumerate(sigs_order):
        if pivot.loc[sig].isna().all():
            ax.text(-0.5, i + 0.5, "inactive", ha="right", va="center",
                    fontsize=7, color="gray", style="italic")

    best_combo = col_score.index[-1]
    ax.set_title(f"UNI + ABMIL HP sweep (7 TCGA cohorts, n=289)\n"
                 f"Best HP: {best_combo}  •  mean active ρ = {col_score.iloc[-1]:.3f}")
    ax.set_xlabel("hidden / dropout / lr combo")
    ax.set_ylabel("CIN signature")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()

    out_pdf = "figures/fig2_hp_sweep_heatmap.pdf"
    out_png = "figures/fig2_hp_sweep_heatmap.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_pdf} and {out_png}")
    print("column ordering by mean ρ:")
    for c, v in col_score.items():
        print(f"  {c}: {v:+.3f}")


if __name__ == "__main__":
    main()
