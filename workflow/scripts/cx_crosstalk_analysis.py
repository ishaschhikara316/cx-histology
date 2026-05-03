"""CX signature crosstalk + per-cohort distribution analysis.

Two figures:
- fig_cx_crosstalk: Spearman correlation matrix of CX1..CX17 across all
  pan-TCGA labelled patients, signature-on-signature.
- fig_cx_per_cohort: per-cohort mean activity for each signature,
  showing which cancer types carry which CX patterns.

Output goes to figures/ and a summary JSON to results/cx_label_atlas.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


def main():
    L = pd.read_csv("data/labels/cx_activities.tsv", sep="\t").set_index(
        "bcr_patient_barcode")
    sig_cols = [c for c in L.columns if c.startswith("CX")]
    print(f"loaded {len(L)} pan-TCGA patients × {len(sig_cols)} signatures")

    # Per-signature activity stats and active mask.
    stats = []
    for s in sig_cols:
        v = L[s].astype(float)
        stats.append({
            "signature": s,
            "nonzero_n": int((v != 0).sum()),
            "nonzero_frac": float((v != 0).mean()),
            "mean": float(v.mean()),
            "max": float(v.max()),
            "var": float(v.var()),
        })
    stats_df = pd.DataFrame(stats)

    # Crosstalk: signature-signature Spearman across patients.
    # Drop signatures that are essentially constant.
    active_sigs = [s["signature"] for s in stats if s["nonzero_frac"] > 0.05]
    n_active = len(active_sigs)
    rho_mat = np.zeros((n_active, n_active))
    for i, si in enumerate(active_sigs):
        for j, sj in enumerate(active_sigs):
            if i <= j:
                r, _ = spearmanr(L[si].astype(float),
                                 L[sj].astype(float),
                                 nan_policy="omit")
                rho_mat[i, j] = rho_mat[j, i] = r if r == r else 0.0

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(rho_mat, xticklabels=active_sigs, yticklabels=active_sigs,
                cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=True,
                fmt=".2f", ax=ax, cbar_kws={"label": "Spearman ρ"},
                annot_kws={"size": 8}, linewidths=0.3, linecolor="white")
    ax.set_title(f"CX signature crosstalk across pan-TCGA "
                 f"(n={len(L)}, active sigs only)")
    fig.tight_layout()
    fig.savefig("figures/fig_cx_crosstalk.pdf", bbox_inches="tight")
    fig.savefig("figures/fig_cx_crosstalk.png", dpi=140, bbox_inches="tight")
    print("wrote figures/fig_cx_crosstalk.{pdf,png}")

    # Note any high collinearity (|r| > 0.5).
    print("\nNotable signature pairs (|ρ| > 0.30):")
    for i, si in enumerate(active_sigs):
        for j, sj in enumerate(active_sigs[i + 1:], start=i + 1):
            if abs(rho_mat[i, j]) > 0.30:
                print(f"  {si}–{sj}: ρ = {rho_mat[i, j]:+.3f}")

    # Per-cohort label distribution.
    # We need cancer_type per patient. Use any of the embedding manifests
    # plus the GDC per-cohort manifests for full coverage.
    rows = []
    manifests_dir = Path("data/interim/manifests")
    import re
    barcode_re = re.compile(r"(TCGA-[0-9A-Z]{2}-[0-9A-Z]{4})")
    for mp in sorted(manifests_dir.glob("tcga_*.tsv")):
        cohort = mp.stem.replace("tcga_", "")
        m = pd.read_csv(mp, sep="\t")
        col = m["filename"] if "filename" in m.columns else (
            m["file_name"] if "file_name" in m.columns else None)
        if col is None:
            continue
        for fn in col:
            mt = barcode_re.search(str(fn))
            if mt:
                rows.append({"patient": mt.group(1), "cancer_type": cohort})
    pat2coh = pd.DataFrame(rows).drop_duplicates("patient").set_index("patient")
    joined = L.join(pat2coh, how="inner")
    print(f"\njoined {len(joined)} patients with cancer_type")

    # Mean activity per cohort × signature.
    mean_by_cohort = joined.groupby("cancer_type")[active_sigs].mean()
    # Drop cohorts with <5 patients to keep the heatmap legible.
    counts = joined.groupby("cancer_type").size()
    keep = counts[counts >= 5].index
    mean_by_cohort = mean_by_cohort.loc[keep]
    # Order rows by mean total activity (rough proxy for "how CIN-driven").
    row_order = mean_by_cohort.sum(axis=1).sort_values(ascending=False).index
    mean_by_cohort = mean_by_cohort.loc[row_order]

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(mean_by_cohort, cmap="viridis", ax=ax, annot=True, fmt=".2f",
                cbar_kws={"label": "Mean CX activity"},
                annot_kws={"size": 7}, linewidths=0.2, linecolor="white")
    ax.set_title(f"Per-cohort mean CX activity (TCGA, n_cohort ≥ 5; rows by total CIN)")
    ax.set_xlabel("CX signature")
    ax.set_ylabel("TCGA cohort")
    fig.tight_layout()
    fig.savefig("figures/fig_cx_per_cohort.pdf", bbox_inches="tight")
    fig.savefig("figures/fig_cx_per_cohort.png", dpi=140, bbox_inches="tight")
    print("wrote figures/fig_cx_per_cohort.{pdf,png}")

    # Save atlas summary.
    atlas = {
        "n_patients": int(len(L)),
        "active_signatures": active_sigs,
        "per_signature_stats": stats,
        "crosstalk_matrix": {
            "rows": active_sigs,
            "values": rho_mat.tolist(),
        },
        "per_cohort_mean": mean_by_cohort.round(4).to_dict(orient="index"),
        "per_cohort_n": counts.loc[keep].to_dict(),
    }
    out = Path("results/cx_label_atlas.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(atlas, indent=2))
    print(f"\nwrote {out}")

    # Headline observations.
    print("\nTop 3 cohorts by total CIN burden:")
    totals = mean_by_cohort.sum(axis=1).sort_values(ascending=False)
    for c in totals.head(3).index:
        print(f"  {c}: total CIN = {totals[c]:.3f}")
    print("\nSignature with highest cross-cohort variance (most cohort-specific):")
    var_by_sig = mean_by_cohort.var(axis=0).sort_values(ascending=False)
    for s in var_by_sig.head(3).index:
        top_c = mean_by_cohort[s].idxmax()
        print(f"  {s}: var across cohorts = {var_by_sig[s]:.4f}, "
              f"peaks in {top_c} (mean = {mean_by_cohort.loc[top_c, s]:.3f})")


if __name__ == "__main__":
    main()
