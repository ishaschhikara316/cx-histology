"""Comprehensive 21-cohort post-retrain analysis.

Mirrors analyze_15cohort.py + adds:
  - survival_analysis (Cox + KM stratified by predicted-CX1/CX3/CX12 quartile)
  - 15-cohort and 7-cohort subset re-evaluation (the cohort-scaling story)

Triggered automatically by phase3_fold_parallel_watcher.sh after the
21-cohort fold-parallel retrain produces cv_summary.json.

Outputs under results/cv/uni_21cohort_final/:
  predictions.tsv, rho_per_cohort.tsv, patient_bootstrap.json,
  robustness.json, calibration_metrics.json, subgroup_metrics.json,
  outlier_patients.tsv, mutation_metrics.json, survival_metrics.json,
  7cohort_subset_comparison.json, 15cohort_subset_comparison.json
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from workflow.scripts._common import CX_COLUMNS

D21 = Path("results/cv/uni_21cohort_final")
M21 = Path("data/interim/manifests/train_uni_21cohort.tsv")

# ----- per-cohort breakdown -----
import workflow.scripts.per_cohort_breakdown as pcb
pcb.BEST_DIR = D21
pcb.MANIFEST = M21
print("=== per_cohort_breakdown @ 21-cohort ===")
pcb.main()

# ----- patient-level bootstrap -----
import workflow.scripts.patient_bootstrap as pb
pb.BEST_DIR = D21
print("\n=== patient_bootstrap @ 21-cohort ===")
pb.main()

# ----- robustness summary -----
import workflow.scripts.robustness_analysis as ra
ra.BEST_DIR = D21
print("\n=== robustness_analysis @ 21-cohort ===")
ra.main()

# ----- calibration -----
import workflow.scripts.calibration_analysis as cal
cal.BEST_DIR = D21
print("\n=== calibration_analysis @ 21-cohort ===")
try: cal.main()
except Exception as e: print(f"  calibration failed: {e}")

# ----- subgroup -----
import workflow.scripts.subgroup_analysis as sg
sg.BEST_DIR = D21
print("\n=== subgroup_analysis @ 21-cohort ===")
try: sg.main()
except Exception as e: print(f"  subgroup failed: {e}")

# ----- outlier -----
import workflow.scripts.outlier_analysis as oa
oa.BEST_DIR = D21
print("\n=== outlier_analysis @ 21-cohort ===")
try: oa.main()
except Exception as e: print(f"  outlier failed: {e}")

# ----- TP53 mutation correlation -----
import workflow.scripts.mutation_correlation as mc
mc.BEST_DIR = D21
print("\n=== mutation_correlation @ 21-cohort ===")
try: mc.main()
except Exception as e: print(f"  mutation_corr failed: {e}")

# ----- survival -----
import workflow.scripts.survival_analysis as sa
sa.BEST_DIR = D21
print("\n=== survival_analysis @ 21-cohort ===")
try: sa.main()
except Exception as e: print(f"  survival failed: {e}")

# ----- 15-cohort and 7-cohort subset comparisons -----
print("\n=== cohort scaling story: 7- vs 15- vs 21-cohort ===")
SUBSET_7 = {"DLBC", "CHOL", "UVM", "READ", "MESO", "UCS", "KICH"}
SUBSET_15 = SUBSET_7 | {"ESCA", "CESC", "PAAD", "THYM", "COAD", "PCPG", "KIRP", "STAD"}

pred = pd.read_csv(D21 / "predictions.tsv", sep="\t")
def subset_rho(pred_df, cohorts):
    bare = {c.replace("TCGA-", "") for c in cohorts}
    mask = pred_df["cancer_type"].isin(cohorts) | pred_df["cancer_type"].isin(bare)
    sub = pred_df[mask]
    rows = {}
    for sig in CX_COLUMNS:
        t = sub[f"{sig}_true"].values
        p = sub[f"{sig}_pred"].values
        if np.var(t) < 1e-10: rows[sig] = None; continue
        r, _ = spearmanr(p, t)
        rows[sig] = None if r != r else float(r)
    finite = [v for v in rows.values() if v is not None]
    return rows, (float(np.mean(finite)) if finite else None), len(sub)

rows_7, mean_7, n_7   = subset_rho(pred, SUBSET_7)
rows_15, mean_15, n_15 = subset_rho(pred, SUBSET_15)

# Native baselines from cv_summary.json files
def native_mean(p):
    s = json.load(open(p))
    m = np.array(s["pan_cancer_spearman_mean"], dtype=float)
    return float(np.nanmean(m))

native_7  = native_mean("results/cv/uni_7cohort_sweep/h512_d0.3_lr0.0001/cv_summary.json")
native_15 = native_mean("results/cv/uni_15cohort_final/cv_summary.json")
native_21 = native_mean(D21 / "cv_summary.json")

print(f"\nThe cohort-scaling story:")
print(f"  7-cohort native model on 7-cohort subset:    rho = {native_7:+.3f}  (n=289)")
print(f"  15-cohort native model on 15-cohort patients: rho = {native_15:+.3f} (n=1454)")
print(f"  21-cohort native model on 21-cohort patients: rho = {native_21:+.3f}")
print(f"  21-cohort model on 15-cohort subset:          rho = {mean_15:+.3f} (n={n_15})")
print(f"  21-cohort model on 7-cohort subset:           rho = {mean_7:+.3f} (n={n_7})")

with open(D21 / "cohort_scaling_summary.json", "w") as f:
    json.dump({
        "native_7cohort_on_7":   {"rho": native_7,  "n": 289},
        "native_15cohort_on_15": {"rho": native_15, "n": 1454},
        "native_21cohort_on_21": {"rho": native_21},
        "21cohort_on_15subset":  {"rho": mean_15, "n": n_15, "per_sig": rows_15},
        "21cohort_on_7subset":   {"rho": mean_7,  "n": n_7,  "per_sig": rows_7},
    }, f, indent=2)
print(f"\nwrote {D21/'cohort_scaling_summary.json'}")
