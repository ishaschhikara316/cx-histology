"""Subgroup analysis: residual ρ broken out by sex, age, and stage.

Pulls clinical metadata for the 289 labelled patients from the GDC
cases endpoint, joins with the saved predictions, then tests whether
the model performs differently across patient subgroups. CPU-only,
needs network access only to GDC (no auth required for this metadata).

Output:
- results/cv/.../clinical_metadata.tsv (per patient)
- results/cv/.../subgroup_metrics.json (per subgroup ρ)
- figures/fig_subgroup_breakdown.{pdf,png}
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr

BEST_DIR = Path("results/cv/uni_7cohort_sweep/h512_d0.3_lr0.0001")
GDC_CASES = "https://api.gdc.cancer.gov/cases"
TOP_SIGS = ["CX1", "CX3", "CX6", "CX10", "CX12"]


def fetch_clinical(patient_ids: list[str]) -> pd.DataFrame:
    """Batch-fetch clinical metadata. GDC max page size is around 1000."""
    rows = []
    chunk = 100
    for i in range(0, len(patient_ids), chunk):
        ids = patient_ids[i:i + chunk]
        params = {
            "filters": json.dumps({
                "op": "in",
                "content": {"field": "submitter_id", "value": ids}
            }),
            "expand": "demographic,diagnoses",
            "format": "json",
            "size": chunk,
        }
        r = requests.get(GDC_CASES, params=params, timeout=60)
        r.raise_for_status()
        for h in r.json().get("data", {}).get("hits", []):
            sid = h.get("submitter_id")
            d = h.get("demographic") or {}
            diags = h.get("diagnoses") or []
            stage = None
            age_at_dx_yr = None
            if diags:
                stage = diags[0].get("ajcc_pathologic_stage")
                ad = diags[0].get("age_at_diagnosis")
                if ad is not None:
                    age_at_dx_yr = ad / 365.25
            rows.append({
                "patient": sid,
                "sex": d.get("gender"),
                "race": d.get("race"),
                "age_yrs": age_at_dx_yr or d.get("age_at_index"),
                "stage": stage,
            })
        time.sleep(0.2)  # be polite
    return pd.DataFrame(rows)


def stage_bucket(s):
    if s is None or pd.isna(s) or s == "Not Reported" or s == "":
        return None
    s = str(s).upper()
    if "IV" in s:
        return "IV"
    if "III" in s:
        return "III"
    if "II" in s:
        return "II"
    if "I" in s:
        return "I"
    return None


def main():
    pred = pd.read_csv(BEST_DIR / "predictions.tsv", sep="\t")
    print(f"loaded {len(pred)} predictions")

    clin_path = BEST_DIR / "clinical_metadata.tsv"
    if clin_path.exists():
        clin = pd.read_csv(clin_path, sep="\t")
        print(f"reusing cached clinical metadata at {clin_path}")
    else:
        print("fetching clinical metadata from GDC...")
        clin = fetch_clinical(pred["patient"].unique().tolist())
        clin.to_csv(clin_path, sep="\t", index=False)
        print(f"wrote {clin_path}")

    df = pred.merge(clin, on="patient", how="left")

    # Bucket age into deciles for plotting; keep raw for correlation.
    df["age_bucket"] = pd.cut(df["age_yrs"], bins=[0, 50, 60, 70, 80, 100],
                              labels=["<50", "50-60", "60-70", "70-80", "80+"])
    df["stage_bucket"] = df["stage"].apply(stage_bucket)

    print(f"\nsex breakdown: {df['sex'].value_counts(dropna=False).to_dict()}")
    print(f"age coverage: {df['age_yrs'].notna().sum()}/{len(df)} have age")
    print(f"stage coverage: {df['stage_bucket'].notna().sum()}/{len(df)} have stage")
    print(f"\nstage distribution: "
          f"{df['stage_bucket'].value_counts(dropna=False).to_dict()}")

    # Per-subgroup Spearman ρ for the top 5 signatures.
    metrics = {}

    def subgroup_rho(d_sub, sig):
        if len(d_sub) < 8:
            return None
        t = d_sub[f"{sig}_true"].values
        p = d_sub[f"{sig}_pred"].values
        if np.var(t) < 1e-10:
            return None
        r, _ = spearmanr(p, t)
        return float(r) if r == r else None

    for split_col in ["sex", "age_bucket", "stage_bucket"]:
        metrics[split_col] = {}
        levels = df[split_col].dropna().unique()
        for lvl in sorted(map(str, levels)):
            sub = df[df[split_col].astype(str) == lvl]
            metrics[split_col][lvl] = {
                "n": int(len(sub)),
                "rho": {sig: subgroup_rho(sub, sig) for sig in TOP_SIGS}
            }

    out = BEST_DIR / "subgroup_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"\nwrote {out}")

    # Plot: 3 panels, one per subgroup type, grouped bars across top sigs.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, split_col, title in zip(axes,
                                    ["sex", "age_bucket", "stage_bucket"],
                                    ["Sex", "Age decile", "AJCC stage"]):
        m = metrics[split_col]
        levels = list(m.keys())
        x = np.arange(len(TOP_SIGS))
        bw = 0.8 / max(len(levels), 1)
        for i, lvl in enumerate(levels):
            ys = [m[lvl]["rho"].get(s) or np.nan for s in TOP_SIGS]
            ax.bar(x + (i - len(levels) / 2 + 0.5) * bw, ys, bw,
                   label=f"{lvl} (n={m[lvl]['n']})")
        ax.axhline(0.30, ls="--", color="gray", lw=0.7)
        ax.axhline(0, color="black", lw=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(TOP_SIGS)
        ax.set_ylabel("Spearman ρ within subgroup")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="best")
        ax.set_ylim(-0.4, 0.7)

    fig.suptitle("Top-5 signature ρ across patient subgroups (best HP, n=289)")
    fig.tight_layout()
    fig.savefig("figures/fig_subgroup_breakdown.pdf", bbox_inches="tight")
    fig.savefig("figures/fig_subgroup_breakdown.png", dpi=140,
                bbox_inches="tight")
    print("wrote figures/fig_subgroup_breakdown.{pdf,png}")

    # Headline observation per subgroup type
    print("\nHeadline:")
    for split_col in ["sex", "age_bucket", "stage_bucket"]:
        m = metrics[split_col]
        if not m:
            continue
        print(f"  {split_col}:")
        for sig in TOP_SIGS:
            rhos = [(lvl, m[lvl]["rho"].get(sig)) for lvl in m
                    if m[lvl]["rho"].get(sig) is not None]
            if not rhos:
                continue
            rhos.sort(key=lambda kv: kv[1], reverse=True)
            best, worst = rhos[0], rhos[-1]
            print(f"    {sig}: best in {best[0]} (ρ={best[1]:+.3f}), "
                  f"worst in {worst[0]} (ρ={worst[1]:+.3f})")


if __name__ == "__main__":
    main()
