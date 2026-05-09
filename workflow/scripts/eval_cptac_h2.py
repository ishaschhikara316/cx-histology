"""H2 external-validation test: |Δρ_TCGA,CPTAC| per signature.

Pairs predictions from predict_cptac_inference.py with the labels
produced by score_cx_cptac_v2.R, computes Spearman ρ on CPTAC per
signature, and contrasts against the TCGA 21-cohort native ρ from
patient_bootstrap.json.

H2 acceptance criterion (pre-registered): |Δρ_TCGA - ρ_CPTAC| ≤ 0.15
for any 21-cohort active signature.

Inputs:
    --tcga-bootstrap  results/cv/uni_21cohort_final/patient_bootstrap.json
    --cptac-pred      results/external/cptac_predictions.tsv
    --cptac-labels    data/labels/cx_activities_cptac.tsv

Output:
    results/external/h2_cptac_external_validation.{json,tsv}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcga-bootstrap",
                    default="results/cv/uni_21cohort_final/patient_bootstrap.json")
    ap.add_argument("--cptac-pred",
                    default="results/external/cptac_predictions.tsv")
    ap.add_argument("--cptac-labels",
                    default="data/labels/cx_activities_cptac.tsv")
    ap.add_argument("--out-json",
                    default="results/external/h2_cptac_external_validation.json")
    ap.add_argument("--out-tsv",
                    default="results/external/h2_cptac_external_validation.tsv")
    ap.add_argument("--threshold", type=float, default=0.15)
    args = ap.parse_args()

    with open(args.tcga_bootstrap) as f:
        boot = json.load(f)
    tcga = {s["signature"]: s for s in boot["signatures"]}

    pred = pd.read_csv(args.cptac_pred, sep="\t")
    labels = pd.read_csv(args.cptac_labels, sep="\t")

    label_id_col = "bcr_patient_barcode" if "bcr_patient_barcode" in \
        labels.columns else labels.columns[0]
    labels = labels.rename(columns={label_id_col: "patient"})

    # If a patient has multiple slides in pred, average their predictions
    # before pairing with the single CN-derived label.
    pred_cols = [c for c in pred.columns if c.endswith("_pred")]
    pred_by_pat = (pred.groupby("patient")[pred_cols].mean().reset_index())

    merged = pred_by_pat.merge(labels, on="patient", how="inner")
    n_paired = len(merged)
    print(f"paired patients: {n_paired} "
          f"(pred {pred_by_pat['patient'].nunique()}, "
          f"labels {labels['patient'].nunique()})")
    if n_paired == 0:
        raise SystemExit("no patients matched between predictions and labels")

    # CINSignatureQuantification outputs columns named "CX1".."CX17".
    sig_cols = [c for c in labels.columns if c.startswith("CX") and
                "_pred" not in c]
    rows = []
    for sig in [f"CX{i}" for i in range(1, 18)]:
        if sig not in sig_cols or f"{sig}_pred" not in merged.columns:
            continue
        x = merged[f"{sig}_pred"].to_numpy(dtype=float)
        y = merged[sig].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 30 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
            rows.append({"signature": sig, "n": int(ok.sum()),
                         "rho_cptac": np.nan, "rho_tcga": np.nan,
                         "delta_rho": np.nan, "passes_h2": False,
                         "tcga_status": tcga.get(sig, {}).get("status",
                                                              "unknown"),
                         "note": "insufficient variance or n<30"})
            continue
        rho, _ = spearmanr(x[ok], y[ok])
        rho_tcga = tcga.get(sig, {}).get("rho", np.nan)
        delta = abs(rho - rho_tcga) if np.isfinite(rho_tcga) else np.nan
        passes = bool(np.isfinite(delta) and delta <= args.threshold)
        rows.append({"signature": sig, "n": int(ok.sum()),
                     "rho_cptac": float(rho), "rho_tcga": float(rho_tcga),
                     "delta_rho": float(delta) if np.isfinite(delta)
                                  else np.nan,
                     "passes_h2": passes,
                     "tcga_status": tcga.get(sig, {}).get("status",
                                                          "unknown"),
                     "note": ""})

    df = pd.DataFrame(rows)
    Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_tsv, sep="\t", index=False)

    active = df[df.tcga_status == "active"]
    n_pass = int(active.passes_h2.sum())
    n_active = len(active)
    summary = {
        "n_paired_patients": int(n_paired),
        "threshold": float(args.threshold),
        "n_active_signatures": n_active,
        "n_passing_h2": n_pass,
        "h2_overall_pass": bool(n_pass > 0),
        "per_signature": rows,
        "method": "Spearman rho on per-patient mean prediction vs "
                  "CINSignatureQuantification label, threshold "
                  "|delta_rho| <= 0.15",
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2, default=lambda x:
                  None if (isinstance(x, float) and not np.isfinite(x))
                  else float(x))

    print()
    print(df.to_string(index=False))
    print()
    print(f"H2 active-signature pass rate: {n_pass}/{n_active}")
    print(f"H2 overall: {'PASS' if n_pass > 0 else 'FAIL'} "
          f"(n_paired={n_paired}, threshold=0.15)")
    return 0


if __name__ == "__main__":
    main()
