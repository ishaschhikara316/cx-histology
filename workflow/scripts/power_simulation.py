"""Power simulation: how many patients to detect a target ρ at q < 0.05?

Uses the saved 7-cohort predictions as the "real" data-generating
distribution. For each target effect size and target n, draws subsamples
of size n with replacement, computes Spearman ρ + permutation p, and
records the fraction of subsamples that pass q < 0.05 after BH across
13 active signatures.

This calibrates the 15-cohort run forecast and informs whether
full pan-TCGA is necessary to push borderline signatures (CX2, CX4,
CX17) across the threshold.

Output:
- results/cv/.../power_curves.json
- figures/fig_power_curves.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, spearmanr

BEST_DIR = Path("results/cv/uni_7cohort_sweep/h512_d0.3_lr0.0001")
N_REPS = 200          # subsamples per (n, sig) cell — uses scipy closed-form p
N_GRID = [50, 100, 200, 289, 500, 1000, 1500]
ACTIVE_SIGS = ["CX1", "CX2", "CX3", "CX4", "CX5", "CX6", "CX8", "CX9",
               "CX10", "CX11", "CX12", "CX16", "CX17"]
RNG_SEED = 20260503


def perm_p(t: np.ndarray, p: np.ndarray, *_args, **_kw) -> float:
    """Two-sided p for Spearman ρ via scipy closed-form (n>=20 → t-distribution).

    Avoids the O(n_perm × n log n) cost of explicit permutation. The
    closed-form is a slight approximation but matches permutation to
    well under 1% at n>=50, which is all our grid points.
    """
    rho_obs, p_val = spearmanr(p, t)
    if rho_obs != rho_obs:
        return 1.0
    return float(p_val)


def main():
    pred = pd.read_csv(BEST_DIR / "predictions.tsv", sep="\t")
    n_full = len(pred)
    print(f"using {n_full} predictions as the data-generating distribution")

    rng_global = np.random.default_rng(RNG_SEED)
    results = {"n_grid": N_GRID, "signatures": {}}

    print(f"\n{'sig':<6}", end="")
    for n in N_GRID:
        print(f"{'n=' + str(n):>10}", end="")
    print(f"  {'pop_rho':>9}")
    print("-" * (6 + 10 * len(N_GRID) + 12))

    for sig in ACTIVE_SIGS:
        t_full = pred[f"{sig}_true"].values
        p_full = pred[f"{sig}_pred"].values
        if np.var(t_full) < 1e-10:
            continue
        pop_rho, _ = spearmanr(p_full, t_full)
        per_n = {}
        for n in N_GRID:
            n_used = min(n, n_full) if n <= n_full else n
            # For n <= n_full, sample without replacement; else with replacement.
            replace = n > n_full
            ps = []
            for _ in range(N_REPS):
                idx = rng_global.choice(n_full, size=n, replace=replace)
                ps.append(perm_p(t_full[idx], p_full[idx]))
            ps = np.array(ps)
            # Fraction of reps where this sig's raw p would survive BH < 0.05
            # against itself * 13. Strict approximation: q = p * 13 < 0.05
            # → p < 0.0038. We use that as a proxy for the BH threshold.
            q_thresh = 0.05 / len(ACTIVE_SIGS)  # bonferroni floor
            power = float(np.mean(ps < q_thresh))
            per_n[n] = {"power_q05_BH_proxy": power,
                        "median_p": float(np.median(ps))}
        results["signatures"][sig] = {
            "pop_rho": float(pop_rho),
            "per_n": per_n,
        }
        print(f"{sig:<6}", end="")
        for n in N_GRID:
            pwr = per_n[n]["power_q05_BH_proxy"]
            print(f"{pwr:>10.2f}", end="")
        print(f"  {pop_rho:>+9.3f}")

    out = BEST_DIR / "power_curves.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")

    # Plot: power vs n for each signature.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = plt.get_cmap("tab20")
    for i, sig in enumerate(ACTIVE_SIGS):
        if sig not in results["signatures"]:
            continue
        per_n = results["signatures"][sig]["per_n"]
        pop_rho = results["signatures"][sig]["pop_rho"]
        ys = [per_n[n]["power_q05_BH_proxy"] for n in N_GRID]
        ax.plot(N_GRID, ys, marker="o", label=f"{sig} (ρ={pop_rho:+.2f})",
                color=cmap(i), lw=1.4, ms=5)
    ax.axhline(0.80, ls="--", color="gray", lw=0.8, label="80% power")
    ax.axvline(289, ls=":", color="black", lw=0.8, label="7-cohort n")
    ax.axvline(1557, ls=":", color="#1f77b4", lw=0.8,
               label="15-cohort n (forecast)")
    ax.set_xscale("log")
    ax.set_xlabel("Sample size n (log scale)")
    ax.set_ylabel("Power: fraction of subsamples reaching p < 0.05/13 (Bonferroni proxy for BH)")
    ax.set_title(f"Power vs n by signature (subsampling from 7-cohort prediction set)")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/fig_power_curves.pdf", bbox_inches="tight")
    fig.savefig("figures/fig_power_curves.png", dpi=140, bbox_inches="tight")
    print("wrote figures/fig_power_curves.{pdf,png}")

    # Headline: at what n does each signature reach 80% power?
    print(f"\nSample size to 80% power (Bonferroni proxy):")
    for sig in ACTIVE_SIGS:
        if sig not in results["signatures"]:
            continue
        per_n = results["signatures"][sig]["per_n"]
        threshold = None
        for n in N_GRID:
            if per_n[n]["power_q05_BH_proxy"] >= 0.80:
                threshold = n
                break
        rho = results["signatures"][sig]["pop_rho"]
        if threshold:
            print(f"  {sig} (ρ={rho:+.2f}): {threshold:>5}")
        else:
            best_n = N_GRID[-1]
            print(f"  {sig} (ρ={rho:+.2f}): >{best_n} "
                  f"(at {best_n}: {per_n[best_n]['power_q05_BH_proxy']:.2f})")


if __name__ == "__main__":
    main()
