"""Patient-level bootstrap on saved predictions for tighter CIs and BH.

The fold-level test (5 folds → minimum p ≈ 0.063) is power-limited.
Patient-level bootstrap uses all 289 (patient, ŷ, y) tuples as the
resampling pool: each replicate samples 289 patients with replacement,
recomputes Spearman ρ per signature. This gives orders of magnitude
more resolvable p-values.

Output: results/cv/.../patient_bootstrap.json with per-signature CI,
empirical p, and BH q.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, spearmanr

from workflow.scripts._common import CX_COLUMNS

BEST_DIR = Path("results/cv/uni_7cohort_sweep/h512_d0.3_lr0.0001")
N_BOOT = 10_000
RNG_SEED = 20260503


def main():
    pred = pd.read_csv(BEST_DIR / "predictions.tsv", sep="\t")
    n = len(pred)
    print(f"loaded {n} patient-level predictions")

    rng = np.random.default_rng(RNG_SEED)
    results = {"n_patients": n, "n_boot": N_BOOT, "signatures": []}

    raw_p = []
    rows_for_table = []

    for sig in CX_COLUMNS:
        t = pred[f"{sig}_true"].values
        p = pred[f"{sig}_pred"].values

        # If true is constant, signature is inactive in this val pool.
        if np.var(t) < 1e-10:
            results["signatures"].append({
                "signature": sig, "status": "inactive",
                "var_true": float(np.var(t)),
            })
            raw_p.append(float("nan"))
            rows_for_table.append({"sig": sig, "status": "inactive"})
            continue

        # Observed point estimate.
        rho_obs, _ = spearmanr(p, t)
        if rho_obs != rho_obs:
            results["signatures"].append({
                "signature": sig, "status": "undefined",
            })
            raw_p.append(float("nan"))
            rows_for_table.append({"sig": sig, "status": "undefined"})
            continue

        # Bootstrap CI: resample with replacement.
        boot_rhos = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, size=n)
            tb = t[idx]
            pb = p[idx]
            if np.var(tb) < 1e-10 or np.var(pb) < 1e-10:
                continue
            r, _ = spearmanr(pb, tb)
            if r == r:
                boot_rhos.append(r)
        boot_rhos = np.array(boot_rhos)
        if len(boot_rhos) < 100:
            continue
        ci_lo, ci_hi = np.quantile(boot_rhos, [0.025, 0.975])

        # Permutation p (two-sided H0: rho = 0).
        # Shuffle predictions vs targets within the val pool.
        n_perm = 5000
        null = np.empty(n_perm)
        perm_rng = np.random.default_rng(RNG_SEED + 1)
        for k in range(n_perm):
            shuffled = perm_rng.permutation(t)
            r, _ = spearmanr(p, shuffled)
            null[k] = r if r == r else 0.0
        p_perm = float(((np.abs(null) >= abs(rho_obs)).sum() + 1) / (n_perm + 1))

        results["signatures"].append({
            "signature": sig,
            "status": "active",
            "rho": float(rho_obs),
            "ci95_lo": float(ci_lo),
            "ci95_hi": float(ci_hi),
            "boot_mean": float(boot_rhos.mean()),
            "boot_std": float(boot_rhos.std()),
            "p_perm": p_perm,
            "n_boot_valid": int(len(boot_rhos)),
        })
        raw_p.append(p_perm)
        rows_for_table.append({"sig": sig, "rho": rho_obs,
                               "ci_lo": ci_lo, "ci_hi": ci_hi,
                               "p": p_perm, "status": "active"})

    # BH-FDR adjustment over active signatures only.
    valid_idx = [i for i, p in enumerate(raw_p) if p == p]
    if valid_idx:
        p_arr = np.array([raw_p[i] for i in valid_idx])
        q_arr = false_discovery_control(p_arr, method="bh")
        for i_local, i_global in enumerate(valid_idx):
            results["signatures"][i_global]["q_bh"] = float(q_arr[i_local])

    # Headline summary.
    active = [s for s in results["signatures"] if s.get("status") == "active"]
    n_q05 = sum(1 for s in active
                if s.get("q_bh", 1) < 0.05 and s.get("rho", 0) > 0)
    n_q01 = sum(1 for s in active
                if s.get("q_bh", 1) < 0.01 and s.get("rho", 0) > 0)
    n_h1 = sum(1 for s in active if s.get("rho", 0) >= 0.30)
    results["summary"] = {
        "n_active_signatures": len(active),
        "mean_rho_active": float(np.mean([s["rho"] for s in active])) if active else None,
        "n_h1_hits_pan_cohort": n_h1,
        "n_q05_positive": n_q05,
        "n_q01_positive": n_q01,
    }

    out = BEST_DIR / "patient_bootstrap.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
    print(f"summary: {results['summary']}")

    # Pretty table.
    print(f"\n{'sig':<6}{'rho':>8}{'95% CI':>22}{'p (perm)':>12}{'q (BH)':>10}{'tag':>10}")
    print("-" * 68)
    for s in results["signatures"]:
        sig = s["signature"]
        if s.get("status") != "active":
            print(f"{sig:<6}{'inactive':>8}")
            continue
        tag = ""
        if s["rho"] >= 0.30:
            tag = "★ H1"
        elif s.get("q_bh", 1) < 0.001:
            tag = "✓ q<.001"
        elif s.get("q_bh", 1) < 0.01:
            tag = "✓ q<.01"
        elif s.get("q_bh", 1) < 0.05:
            tag = "✓ q<.05"
        print(f"{sig:<6}{s['rho']:>+8.3f}"
              f"  [{s['ci95_lo']:+.3f}, {s['ci95_hi']:+.3f}]"
              f"{s['p_perm']:>12.4f}{s.get('q_bh', float('nan')):>10.4f}{tag:>10}")

    # Forest plot.
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ys = []
    labels_y = []
    for s in results["signatures"]:
        sig = s["signature"]
        if s.get("status") != "active":
            continue
        labels_y.append(sig)
        ys.append((s["rho"], s["ci95_lo"], s["ci95_hi"]))
    ys = np.array(ys)
    y_pos = np.arange(len(ys))
    rho_vals = ys[:, 0]
    err_lo = rho_vals - ys[:, 1]
    err_hi = ys[:, 2] - rho_vals
    colors = ["#2ca02c" if v >= 0.30 else "#1f77b4" if v > 0 else "#d62728"
              for v in rho_vals]
    ax.errorbar(rho_vals, y_pos, xerr=[err_lo, err_hi], fmt="o",
                ecolor="gray", capsize=3, markersize=6,
                markerfacecolor="white")
    for i, (r, c) in enumerate(zip(rho_vals, colors)):
        ax.scatter(r, i, color=c, s=40, zorder=3)
    ax.axvline(0.0, color="black", lw=0.5)
    ax.axvline(0.30, ls="--", color="gray", lw=1, label="H1 target ρ ≥ 0.30")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_y)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman ρ (point estimate, 95% bootstrap CI)")
    ax.set_title(f"Patient-level bootstrap (n={n}, n_boot={N_BOOT})\n"
                 f"7-cohort best HP h512_d0.3_lr1e-4")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig("figures/fig_patient_bootstrap_forest.pdf", bbox_inches="tight")
    fig.savefig("figures/fig_patient_bootstrap_forest.png", dpi=140,
                bbox_inches="tight")
    print("\nwrote figures/fig_patient_bootstrap_forest.{pdf,png}")


if __name__ == "__main__":
    main()
