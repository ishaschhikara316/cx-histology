"""Statistical robustness analysis for the 7-cohort sweep best HP.

Computes:
- Per-signature one-sided permutation p-values (H0: ρ = 0)
- Benjamini-Hochberg FDR adjustment across the active signatures
- Bootstrap 95% CI on per-signature ρ using fold-level resampling
- Fold-to-fold consistency: range, IQR, sign agreement

Output:
- results/cv/uni_7cohort_sweep/h512_d0.3_lr0.0001/robustness.json
- figures/fig3_robustness_table.png  (publication-ready table figure)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, spearmanr

BEST_DIR = Path("results/cv/uni_7cohort_sweep/h512_d0.3_lr0.0001")
SIGS = [f"CX{i}" for i in range(1, 18)]


def load_fold_metrics():
    s = json.load(open(BEST_DIR / "cv_summary.json"))
    rho = np.array([
        [float("nan") if (x is None or x != x) else float(x)
         for x in f["spearman_per_sig"]]
        for f in s["fold_metrics"]
    ])
    masks = np.array([f["active_mask"] for f in s["fold_metrics"]])
    return s, rho, masks


def bootstrap_ci(fold_rhos: np.ndarray, n_boot: int = 5000,
                 alpha: float = 0.05, rng_seed: int = 20260503):
    rng = np.random.default_rng(rng_seed)
    valid = fold_rhos[~np.isnan(fold_rhos)]
    if len(valid) == 0:
        return float("nan"), float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(valid), size=len(valid))
        means.append(valid[idx].mean())
    means = np.array(means)
    return (float(valid.mean()),
            float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def perm_p(fold_rhos: np.ndarray, n_perm: int = 100_000,
           rng_seed: int = 20260503) -> float:
    """Sign-flip permutation p-value for H0: ρ = 0 (two-sided).

    Each fold ρ is treated as an independent estimate of the underlying
    correlation; under H0 the per-fold sign is exchangeable.
    """
    rng = np.random.default_rng(rng_seed)
    valid = fold_rhos[~np.isnan(fold_rhos)]
    if len(valid) == 0:
        return float("nan")
    obs = abs(valid.mean())
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(valid)))
    null = np.abs((flips * valid).mean(axis=1))
    return float(((null >= obs).sum() + 1) / (n_perm + 1))


def main():
    summary, rho, masks = load_fold_metrics()
    n_folds = rho.shape[0]
    results = {"signatures": []}

    p_per_sig = []
    rows = []
    for k, sig in enumerate(SIGS):
        fold_rhos = rho[:, k]
        is_active_in_all = bool(masks[:, k].sum() == n_folds)
        if not is_active_in_all:
            entry = {
                "signature": sig,
                "active_in_n_folds": int(masks[:, k].sum()),
                "status": "inactive",
            }
            results["signatures"].append(entry)
            p_per_sig.append(float("nan"))
            rows.append({"signature": sig, "rho": float("nan"),
                         "ci95_lo": float("nan"), "ci95_hi": float("nan"),
                         "p_perm": float("nan"), "q_bh": float("nan"),
                         "fold_signs": "n/a", "status": "inactive"})
            continue

        mean_rho, lo, hi = bootstrap_ci(fold_rhos)
        p = perm_p(fold_rhos)
        p_per_sig.append(p)

        positive = int((fold_rhos > 0).sum())
        negative = int((fold_rhos < 0).sum())
        sign_str = f"+{positive}/-{negative}/{n_folds - positive - negative}=0"

        entry = {
            "signature": sig,
            "n_folds": n_folds,
            "active_in_n_folds": int(masks[:, k].sum()),
            "rho_mean": mean_rho,
            "rho_std": float(np.nanstd(fold_rhos)),
            "rho_min": float(np.nanmin(fold_rhos)),
            "rho_max": float(np.nanmax(fold_rhos)),
            "ci95_lo": lo,
            "ci95_hi": hi,
            "p_perm": p,
            "fold_pos": positive,
            "fold_neg": negative,
        }
        results["signatures"].append(entry)
        rows.append({"signature": sig, "rho": mean_rho,
                     "ci95_lo": lo, "ci95_hi": hi,
                     "p_perm": p, "q_bh": float("nan"),
                     "fold_signs": sign_str, "status": "active"})

    # BH-FDR adjust over the active signatures only.
    valid_idx = [i for i, p in enumerate(p_per_sig) if p == p]
    if valid_idx:
        p_arr = np.array([p_per_sig[i] for i in valid_idx])
        q_arr = false_discovery_control(p_arr, method="bh")
        for i_local, i_global in enumerate(valid_idx):
            results["signatures"][i_global]["q_bh"] = float(q_arr[i_local])
            rows[i_global]["q_bh"] = float(q_arr[i_local])

    # Headline summary.
    active_rhos = np.array([r["rho_mean"] for r in results["signatures"]
                            if r.get("status") != "inactive"])
    n_q05 = int(sum(1 for r in results["signatures"]
                    if r.get("q_bh") is not None
                    and r.get("q_bh") == r.get("q_bh")
                    and r["q_bh"] < 0.05
                    and r.get("rho_mean", 0) > 0))
    n_h1 = int(sum(1 for r in results["signatures"]
                   if r.get("rho_mean") is not None
                   and r.get("rho_mean") == r.get("rho_mean")
                   and r["rho_mean"] >= 0.30))
    results["summary"] = {
        "n_active_signatures": len(active_rhos),
        "mean_active_rho": float(np.mean(active_rhos)) if len(active_rhos) else None,
        "n_h1_hits": n_h1,
        "n_significant_q05_positive": n_q05,
        "n_folds": n_folds,
    }

    out_json = BEST_DIR / "robustness.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_json}")
    print(f"summary: {results['summary']}")

    # Table figure.
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.axis("off")
    cols = ["signature", "rho", "ci95_lo", "ci95_hi", "p_perm", "q_bh",
            "fold_signs", "status"]
    cell_text = []
    for _, r in df.iterrows():
        if r["status"] == "inactive":
            cell_text.append([r["signature"], "—", "—", "—", "—", "—", "—",
                              "inactive"])
        else:
            cell_text.append([
                r["signature"],
                f"{r['rho']:+.3f}",
                f"{r['ci95_lo']:+.3f}",
                f"{r['ci95_hi']:+.3f}",
                f"{r['p_perm']:.3g}",
                f"{r['q_bh']:.3g}",
                r["fold_signs"],
                ("★ H1" if not np.isnan(r['rho']) and r['rho'] >= 0.30
                 else ("✓ q<.05" if not np.isnan(r['q_bh']) and r['q_bh'] < 0.05
                       else "")),
            ])
    table = ax.table(cellText=cell_text,
                     colLabels=["sig", "ρ", "95% CI lo", "95% CI hi",
                                "p (perm)", "q (BH)", "fold +/-/0", ""],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.35)
    for k in range(len(cell_text) + 1):
        for j in range(len(cols)):
            cell = table[k, j]
            if k == 0:
                cell.set_facecolor("#cccccc")
                cell.set_text_props(weight="bold")
    # Highlight active rows by status
    for k, row in enumerate(cell_text, start=1):
        if row[-1].startswith("★"):
            for j in range(len(cols)):
                table[k, j].set_facecolor("#d4edda")
        elif row[-1].startswith("✓"):
            for j in range(len(cols)):
                table[k, j].set_facecolor("#fff3cd")
        elif row[-1] == "inactive":
            for j in range(len(cols)):
                table[k, j].set_facecolor("#eeeeee")

    ax.set_title(f"7-cohort robustness — best HP h512_d0.3_lr1e-4\n"
                 f"{results['summary']['n_h1_hits']} H1 hits "
                 f"(ρ ≥ 0.30), "
                 f"{results['summary']['n_significant_q05_positive']} signatures "
                 f"q < 0.05 with positive ρ (n_folds = {n_folds}, "
                 f"BH-adjusted across {len(valid_idx)} active sigs)")
    fig.tight_layout()
    out_pdf = "figures/fig3_robustness_table.pdf"
    out_png = "figures/fig3_robustness_table.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
