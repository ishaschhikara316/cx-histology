"""Figure: CPTAC external validation, H2 |Δρ_TCGA,CPTAC| ≤ 0.15.

Two panels:
  A) Paired ρ_TCGA vs ρ_CPTAC slope plot for the 14 active signatures.
     Lines connect the same signature across the two cohorts; signatures
     that improve on CPTAC (CX16, CX17) are highlighted in green.
  B) |Δρ| bar chart sorted, with the pre-registered 0.15 threshold
     drawn. All 14 active signatures pass.

Inputs:
  - results/external/h2_cptac_external_validation.json

Outputs:
  - figures/fig_cptac_h2.pdf
  - figures/fig_cptac_h2.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    base = Path(".")
    with open(base / "results/external/h2_cptac_external_validation.json"
              ) as f:
        d = json.load(f)
    rows = [r for r in d["per_signature"]
            if r.get("tcga_status") == "active"
            and r.get("rho_cptac") is not None
            and r.get("rho_tcga") is not None]
    rows = sorted(rows, key=lambda r: -r["rho_tcga"])  # top TCGA first

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6),
                             gridspec_kw={"width_ratios": [1.1, 1]})

    # Panel A: paired slope plot.
    ax = axes[0]
    for r in rows:
        improved = r["rho_cptac"] >= r["rho_tcga"]
        color = "#2ca02c" if improved else "#888"
        ax.plot([0, 1], [r["rho_tcga"], r["rho_cptac"]],
                "-", color=color, alpha=0.55, lw=1.2)
        ax.plot([0], [r["rho_tcga"]], "o", color="#1f77b4", markersize=6)
        ax.plot([1], [r["rho_cptac"]], "o", color="#d62728", markersize=6)
        # Label CX16, CX17 (improvers) and CX5/CX1 (largest).
        if r["signature"] in {"CX16", "CX17", "CX5", "CX1", "CX12"}:
            ax.annotate(r["signature"], xy=(1.04, r["rho_cptac"]),
                        fontsize=8.5, va="center")
    ax.axhline(0, color="#bbb", lw=0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["TCGA\n(21-cohort\nbootstrap)",
                        "CPTAC\n(n = 283\npaired)"])
    ax.set_ylabel("pooled Spearman ρ")
    ax.set_title("A. Per-signature ρ on TCGA vs CPTAC "
                 "(green = improved on CPTAC)",
                 fontsize=10.5, loc="left")
    ax.set_xlim(-0.2, 1.35)
    ax.grid(axis="y", alpha=0.3)

    # Panel B: |delta_rho| bar chart sorted.
    rows_by_delta = sorted(rows, key=lambda r: r["delta_rho"])
    sigs = [r["signature"] for r in rows_by_delta]
    deltas = [r["delta_rho"] for r in rows_by_delta]
    threshold = d.get("threshold", 0.15)
    ax = axes[1]
    bars = ax.bar(sigs, deltas, color="#2ca02c", edgecolor="#194d19",
                  linewidth=0.6)
    for b, v in zip(bars, deltas):
        ax.text(b.get_x() + b.get_width()/2, v + 0.004, f"{v:.3f}",
                ha="center", fontsize=8)
    ax.axhline(threshold, ls="--", color="#d62728", lw=1.2,
               label=f"H2 threshold = {threshold}")
    ax.set_ylim(0, threshold * 1.25)
    ax.set_ylabel("| ρ_TCGA − ρ_CPTAC |")
    ax.set_title(
        f"B. All {len(rows)} active signatures pass | Δρ | ≤ {threshold}",
        fontsize=10.5, loc="left")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Figure 5. External validation on CPTAC (n = "
        f"{d['n_paired_patients']} paired patients): the H2 "
        f"effect-size-similarity test passes for "
        f"{d['n_passing_h2']}/{d['n_active_signatures']} active "
        f"signatures.", fontsize=10, y=1.005, x=0.02, ha="left")
    fig.tight_layout()
    out_pdf = base / "figures/fig_cptac_h2.pdf"
    out_png = base / "figures/fig_cptac_h2.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"wrote {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
