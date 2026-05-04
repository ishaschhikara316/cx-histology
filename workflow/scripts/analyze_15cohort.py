"""Run per-cohort breakdown + patient bootstrap + robustness + 7-cohort
subset comparison on the 15-cohort fold checkpoints.

Re-uses the existing committed analysis scripts via module-level
monkey-patch of BEST_DIR / MANIFEST so the same code paths are
exercised. Adds the 7-cohort-subset rho computation, which is the
key apples-to-apples comparison: did the *same 7 cohorts* get worse
under the 15-cohort model, or only the new ones?

Outputs under results/cv/uni_15cohort_final/:
  - predictions.tsv               (per-patient pred + true)
  - rho_per_cohort.tsv            (per-cohort × per-signature rho)
  - patient_bootstrap.json        (n_boot=10k CIs + BH q)
  - robustness.json               (CI-style summary table)
  - 7cohort_subset_comparison.json (7-vs-15 cohort comparison on the
                                    DLBC/CHOL/UVM/READ/MESO/UCS/KICH subset)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from workflow.scripts._common import CX_COLUMNS

D15 = Path("results/cv/uni_15cohort_final")
M15 = Path("data/interim/manifests/train_uni_15cohort.tsv")

# ----- per-cohort breakdown -----------------------------------------------
import workflow.scripts.per_cohort_breakdown as pcb
pcb.BEST_DIR = D15
pcb.MANIFEST = M15
print("=== per_cohort_breakdown @ 15-cohort ===")
pcb.main()

# ----- patient-level bootstrap (n=10k) -----------------------------------
import workflow.scripts.patient_bootstrap as pb
pb.BEST_DIR = D15
print("\n=== patient_bootstrap @ 15-cohort ===")
pb.main()

# ----- robustness summary -------------------------------------------------
import workflow.scripts.robustness_analysis as ra
ra.BEST_DIR = D15
print("\n=== robustness_analysis @ 15-cohort ===")
ra.main()

# ----- 7-cohort subset comparison ----------------------------------------
print("\n=== 7-cohort subset comparison ===")
SUBSET_7 = {"TCGA-DLBC", "TCGA-CHOL", "TCGA-UVM", "TCGA-READ",
            "TCGA-MESO", "TCGA-UCS", "TCGA-KICH"}
# fall back to bare names in case manifest stores without TCGA- prefix
SUBSET_7_BARE = {s.replace("TCGA-", "") for s in SUBSET_7}

pred = pd.read_csv(D15 / "predictions.tsv", sep="\t")
mask = pred["cancer_type"].isin(SUBSET_7) | pred["cancer_type"].isin(SUBSET_7_BARE)
pred_subset = pred[mask].copy()
print(f"  subset rows: {len(pred_subset)} of {len(pred)} (across "
      f"{pred_subset.cancer_type.nunique()} cohorts)")

# Pan-subset rho per signature
rows = []
for sig in CX_COLUMNS:
    t = pred_subset[f"{sig}_true"].values
    p = pred_subset[f"{sig}_pred"].values
    if np.var(t) < 1e-10:
        rows.append({"sig": sig, "rho_15on7subset": None,
                     "status": "inactive"})
        continue
    rho, _ = spearmanr(p, t)
    rows.append({"sig": sig, "rho_15on7subset": None if rho != rho else float(rho),
                 "status": "ok"})
df_subset = pd.DataFrame(rows)

# Compare to original 7-cohort numbers (from the committed RESULTS.md /
# robustness.json on the 7-cohort run)
seven = json.load(open("results/cv/uni_7cohort_sweep/h512_d0.3_lr0.0001/cv_summary.json"))
seven_mean = seven["pan_cancer_spearman_mean"]
df_subset["rho_7cohort_native"] = [seven_mean[i] for i in range(len(CX_COLUMNS))]
df_subset["delta"] = [
    None if r["rho_15on7subset"] is None or pd.isna(df_subset["rho_7cohort_native"][i])
    else r["rho_15on7subset"] - df_subset["rho_7cohort_native"][i]
    for i, r in df_subset.iterrows()
]
print("\nsig    rho_7cohort_native   rho_15on7subset   delta")
print("-" * 60)
for _, r in df_subset.iterrows():
    n = r["rho_7cohort_native"]
    f = r["rho_15on7subset"]
    d = r["delta"]
    n_s = f"{n:+.3f}" if n is not None and not pd.isna(n) else "  n/a "
    f_s = f"{f:+.3f}" if f is not None else "  n/a "
    d_s = f"{d:+.3f}" if d is not None else "  n/a "
    print(f"{r['sig']:>5}      {n_s}            {f_s}        {d_s}")

# Save
df_subset.to_csv(D15 / "7cohort_subset_comparison.tsv", sep="\t", index=False)
out = {
    "subset_cohorts": sorted(SUBSET_7_BARE),
    "n_subset_patients": int(len(pred_subset)),
    "rows": df_subset.to_dict(orient="records"),
}
with open(D15 / "7cohort_subset_comparison.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nwrote {D15/'7cohort_subset_comparison.json'}")

# Headline interpretation
finite_subset = df_subset.dropna(subset=["rho_15on7subset"])["rho_15on7subset"]
finite_native = pd.Series([seven_mean[i] for i in range(len(CX_COLUMNS))]).dropna()
print(f"\n  mean rho on 7-cohort subset under 15-cohort model: {finite_subset.mean():+.3f}")
print(f"  mean rho on 7-cohort native model (from cv_summary):  {finite_native.mean():+.3f}")
print(f"  net change on the same 7 cohorts:                     {finite_subset.mean() - finite_native.mean():+.3f}")
