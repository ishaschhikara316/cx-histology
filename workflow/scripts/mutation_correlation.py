"""TP53 mutation correlation with predicted CX1.

CX1 is the tetraploidy-related signature in Drews 2022; tetraploidy is
mechanistically driven by TP53 loss. If our predicted CX1 carries real
biology, it should associate with TP53 mutation status.

Pulls TCGA somatic mutation calls via GDC's `simple_somatic_mutation` /
`ssm` endpoint, joined to our 289 labelled patients. Tests:

1. Are predicted CX1 values higher in TP53-mutant patients than
   TP53-wildtype patients (Mann-Whitney within each cancer type)?
2. Same for actual (label) CX1 as a sanity check (this should be
   strongly positive if TP53→CX1 is real biology in TCGA).
3. Same for predicted CX3 (BFB) — should be weaker but possibly
   present.

Output:
- results/cv/.../mutation_metrics.json
- figures/fig_tp53_vs_cx1.{pdf,png}  Box plot per cohort
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from scipy.stats import mannwhitneyu

BEST_DIR = Path("results/cv/uni_7cohort_sweep/h512_d0.3_lr0.0001")
GDC_SSM = "https://api.gdc.cancer.gov/ssm_occurrences"


def fetch_tp53_mutations(patient_ids: list[str]) -> set[str]:
    """Return the subset of patient_ids with at least one TP53 SSM."""
    mutated = set()
    chunk = 100
    for i in range(0, len(patient_ids), chunk):
        ids = patient_ids[i:i + chunk]
        params = {
            "filters": json.dumps({
                "op": "and",
                "content": [
                    {"op": "in",
                     "content": {"field": "case.submitter_id", "value": ids}},
                    {"op": "=",
                     "content": {"field": "ssm.consequence.transcript.gene.symbol",
                                 "value": "TP53"}},
                ]
            }),
            "fields": "case.submitter_id",
            "format": "json",
            "size": 5000,
        }
        r = requests.get(GDC_SSM, params=params, timeout=120)
        r.raise_for_status()
        data = r.json().get("data", {}).get("hits", [])
        for h in data:
            cases = h.get("case") or []
            if isinstance(cases, dict):
                cases = [cases]
            for c in cases:
                sid = c.get("submitter_id")
                if sid:
                    mutated.add(sid)
        time.sleep(0.3)
    return mutated


def main():
    pred = pd.read_csv(BEST_DIR / "predictions.tsv", sep="\t")
    patient_ids = sorted(pred["patient"].unique())

    cache = BEST_DIR / "tp53_mutated_patients.tsv"
    if cache.exists():
        mutated = set(pd.read_csv(cache, sep="\t")["patient"])
        print(f"reusing {cache}: {len(mutated)} mutated patients")
    else:
        print(f"fetching TP53 status for {len(patient_ids)} patients via GDC ssm_occurrences")
        mutated = fetch_tp53_mutations(patient_ids)
        pd.DataFrame({"patient": sorted(mutated)}).to_csv(
            cache, sep="\t", index=False)
        print(f"wrote {cache}: {len(mutated)} mutated")

    pred["tp53_mut"] = pred["patient"].isin(mutated).astype(int)

    print("\nTP53 mutation status by cohort (in our 289 patients):")
    summary = pred.groupby("cancer_type").agg(
        n=("patient", "size"),
        tp53_mut=("tp53_mut", "sum"),
    ).assign(mut_pct=lambda d: 100 * d["tp53_mut"] / d["n"]).round(1)
    print(summary.to_string())

    # Test predicted CX1 ~ TP53 mutation status, both pan-cohort and per-cohort.
    metrics = {"cohorts": {}, "summary": {}}
    cohorts = sorted(pred["cancer_type"].unique())

    for sig_label, col in [("pred_CX1", "CX1_pred"),
                           ("true_CX1", "CX1_true"),
                           ("pred_CX3", "CX3_pred"),
                           ("true_CX3", "CX3_true")]:
        # Pan-cohort, controlling for cancer type via stratified Mann-Whitney
        # would be complex; instead we report per-cohort and pan-cohort separately.
        per_cohort = []
        for c in cohorts:
            sub = pred[pred.cancer_type == c]
            mut = sub.loc[sub.tp53_mut == 1, col].values
            wt = sub.loc[sub.tp53_mut == 0, col].values
            if len(mut) >= 3 and len(wt) >= 3:
                stat, p = mannwhitneyu(mut, wt, alternative="greater")
                per_cohort.append({
                    "cohort": c,
                    "n_mut": int(len(mut)),
                    "n_wt": int(len(wt)),
                    "median_mut": float(np.median(mut)),
                    "median_wt": float(np.median(wt)),
                    "delta_median": float(np.median(mut) - np.median(wt)),
                    "p_one_sided": float(p),
                })
            else:
                per_cohort.append({
                    "cohort": c,
                    "n_mut": int(len(mut)),
                    "n_wt": int(len(wt)),
                    "skipped": "n<3 in one group",
                })
        metrics["cohorts"][sig_label] = per_cohort

        # Pan-cohort (caveat: confounded by cancer type, but informative).
        mut_all = pred.loc[pred.tp53_mut == 1, col].values
        wt_all = pred.loc[pred.tp53_mut == 0, col].values
        stat, p = mannwhitneyu(mut_all, wt_all, alternative="greater")
        metrics["summary"][sig_label] = {
            "n_mut": int(len(mut_all)),
            "n_wt": int(len(wt_all)),
            "median_mut": float(np.median(mut_all)),
            "median_wt": float(np.median(wt_all)),
            "p_one_sided_pan_cohort": float(p),
        }

    print("\n=== Pan-cohort Mann-Whitney (TP53 mutated > wild-type) ===")
    for k, v in metrics["summary"].items():
        print(f"  {k}: median mut={v['median_mut']:.3f} vs wt={v['median_wt']:.3f}, "
              f"p (one-sided) = {v['p_one_sided_pan_cohort']:.3g}, "
              f"n={v['n_mut']}/{v['n_wt']}")

    print("\n=== Per-cohort: predicted CX1 by TP53 ===")
    print(f"{'cohort':<8}{'n_mut':>6}{'n_wt':>6}{'med_mut':>9}{'med_wt':>9}"
          f"{'Δ':>8}{'p':>10}")
    for r in metrics["cohorts"]["pred_CX1"]:
        if "skipped" in r:
            print(f"  {r['cohort']:<8}{r['n_mut']:>6}{r['n_wt']:>6}  "
                  f"(skipped: {r['skipped']})")
        else:
            print(f"  {r['cohort']:<8}{r['n_mut']:>6}{r['n_wt']:>6}"
                  f"{r['median_mut']:>+9.3f}{r['median_wt']:>+9.3f}"
                  f"{r['delta_median']:>+8.3f}{r['p_one_sided']:>10.3g}")

    out = BEST_DIR / "mutation_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"\nwrote {out}")

    # Box plot: predicted vs actual CX1, split by TP53.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, sig_col, title in zip(axes,
                                  ["CX1_true", "CX1_pred"],
                                  ["Actual CX1 (label)", "Predicted CX1"]):
        plot_df = pred[["cancer_type", "tp53_mut", sig_col]].copy()
        plot_df["TP53"] = plot_df["tp53_mut"].map({0: "WT", 1: "Mut"})
        sns.boxplot(data=plot_df, x="cancer_type", y=sig_col, hue="TP53",
                    ax=ax, palette={"WT": "#9aa6b2", "Mut": "#d62728"},
                    showfliers=False)
        sns.stripplot(data=plot_df, x="cancer_type", y=sig_col, hue="TP53",
                      dodge=True, ax=ax, size=3, alpha=0.6,
                      palette={"WT": "#444444", "Mut": "#7a0011"},
                      legend=False)
        ax.set_title(title)
        ax.set_xlabel("Cohort")
        ax.set_ylabel(sig_col)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.suptitle("CX1 (tetraploidy) vs TP53 mutation status, "
                 "by cancer type (n=289)")
    fig.tight_layout()
    fig.savefig("figures/fig_tp53_vs_cx1.pdf", bbox_inches="tight")
    fig.savefig("figures/fig_tp53_vs_cx1.png", dpi=140, bbox_inches="tight")
    print("wrote figures/fig_tp53_vs_cx1.{pdf,png}")


if __name__ == "__main__":
    main()
