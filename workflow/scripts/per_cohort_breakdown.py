"""Per-cohort breakdown of best-HP predictions.

Reloads the 5 saved fold checkpoints, runs inference on each fold's
validation set, recovers (patient, cancer_type, ŷ, y) tuples, then
computes per-cohort Spearman ρ for each signature. Also saves the raw
predictions to results/cv/.../predictions.tsv so other downstream
analyses (patient-level bootstrap, attention recovery) can reuse them.

CPU-only: ABMIL is small (a few hundred KB of params), and we just read
embeddings sequentially from .h5.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import spearmanr

from workflow.scripts._common import CX_COLUMNS
from workflow.scripts.abmil import GatedABMIL
from workflow.scripts.loss import invert_label_transform

BEST_DIR = Path("results/cv/uni_7cohort_sweep/h512_d0.3_lr0.0001")
MANIFEST = Path("data/interim/manifests/train_uni_7cohort.tsv")
LABELS = Path("data/labels/cx_activities.tsv")


def load_fold_predictions(fold_idx: int, val_ids: list[str],
                          manifest: pd.DataFrame, labels: pd.DataFrame,
                          device: str = "cpu", max_tiles: int = 1000):
    ckpt = torch.load(BEST_DIR / f"fold{fold_idx}.pt", map_location=device,
                      weights_only=False)
    cfg = ckpt["config"]
    label_mu = torch.from_numpy(np.asarray(ckpt["label_mu"]))
    label_sd = torch.from_numpy(np.asarray(ckpt["label_sd"]))

    model = GatedABMIL(in_dim=cfg["embed_dim"], hidden=cfg["hidden"],
                       n_out=cfg["n_cx"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    rows = []
    rng = np.random.default_rng(cfg["seed"] + 1000 + fold_idx)
    with torch.no_grad():
        for pid in val_ids:
            if pid not in manifest.index:
                continue
            emb_path = manifest.loc[pid, "emb_path"]
            cancer = manifest.loc[pid, "cancer_type"]
            with h5py.File(emb_path, "r") as h:
                emb = h["embeddings"][:]
            if emb.shape[0] == 0:
                continue
            if emb.shape[0] > max_tiles:
                idx = rng.choice(emb.shape[0], max_tiles, replace=False)
                emb = emb[idx]
            x = torch.from_numpy(emb).unsqueeze(0).to(device)
            mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)
            pred_z, _ = model(x, mask=mask)
            pred = invert_label_transform(pred_z.cpu(), label_mu, label_sd)
            pred_np = pred.squeeze(0).numpy()
            target = labels.loc[pid].values.astype(float)
            row = {"patient": pid, "cancer_type": cancer, "fold": fold_idx}
            for k, c in enumerate(CX_COLUMNS):
                row[f"{c}_pred"] = float(pred_np[k])
                row[f"{c}_true"] = float(target[k])
            rows.append(row)
    return rows


def main():
    summary = json.load(open(BEST_DIR / "cv_summary.json"))
    manifest = pd.read_csv(MANIFEST, sep="\t").set_index("patient")
    labels = pd.read_csv(LABELS, sep="\t").set_index("bcr_patient_barcode")[CX_COLUMNS]
    shared = manifest.index.intersection(labels.index)
    manifest = manifest.loc[shared]
    labels = labels.loc[shared]

    all_rows = []
    for fm in summary["fold_metrics"]:
        fold = fm["fold"]
        val_ids = fm["val_ids"]
        print(f"  fold {fold}: {len(val_ids)} val patients")
        all_rows.extend(load_fold_predictions(fold, val_ids, manifest, labels))

    df = pd.DataFrame(all_rows)
    out_pred = BEST_DIR / "predictions.tsv"
    df.to_csv(out_pred, sep="\t", index=False)
    print(f"wrote {out_pred} ({len(df)} patient-level predictions)")

    # Per-cohort × per-signature ρ.
    cohorts = sorted(df.cancer_type.unique())
    rho_per = pd.DataFrame(index=cohorts, columns=CX_COLUMNS, dtype=float)
    n_per = pd.DataFrame(index=cohorts, columns=["n"], dtype=int)
    for c in cohorts:
        sub = df[df.cancer_type == c]
        n_per.loc[c, "n"] = len(sub)
        for sig in CX_COLUMNS:
            t = sub[f"{sig}_true"].values
            p = sub[f"{sig}_pred"].values
            if len(np.unique(t)) < 2 or len(np.unique(p)) < 2:
                rho_per.loc[c, sig] = float("nan")
            else:
                r, _ = spearmanr(p, t)
                rho_per.loc[c, sig] = r if r == r else float("nan")

    # Order rows by sample count (descending).
    rho_per = rho_per.loc[n_per.sort_values("n", ascending=False).index]
    n_per = n_per.loc[rho_per.index]

    # Save table.
    out_tsv = BEST_DIR / "rho_per_cohort.tsv"
    out = rho_per.copy()
    out.insert(0, "n", n_per["n"])
    out.to_csv(out_tsv, sep="\t")
    print(f"wrote {out_tsv}")

    # Heatmap.
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.heatmap(rho_per.astype(float), cmap="RdBu_r", vmin=-0.6, vmax=0.6,
                center=0, annot=True, fmt=".2f",
                annot_kws={"size": 8}, linewidths=0.3, linecolor="white",
                cbar_kws={"label": "Spearman ρ"}, ax=ax)
    yticks = [f"{c} (n={int(n_per.loc[c, 'n'])})" for c in rho_per.index]
    ax.set_yticklabels(yticks, rotation=0)
    ax.set_xlabel("CIN signature")
    ax.set_ylabel("Cohort")
    ax.set_title("Per-cohort per-signature Spearman ρ "
                 "(best HP h512_d0.3_lr1e-4, 7-cohort sweep)")
    fig.tight_layout()
    fig.savefig("figures/fig_per_cohort_rho.pdf", bbox_inches="tight")
    fig.savefig("figures/fig_per_cohort_rho.png", dpi=140, bbox_inches="tight")
    print("wrote figures/fig_per_cohort_rho.{pdf,png}")

    # Headline summary.
    print("\nPer-cohort summary:")
    for c in rho_per.index:
        row = rho_per.loc[c].astype(float)
        active = row.dropna()
        n_h1 = (active >= 0.30).sum()
        if len(active):
            print(f"  {c} (n={int(n_per.loc[c, 'n']):3d}): "
                  f"mean ρ {active.mean():+.3f}, max {active.max():+.3f}, "
                  f"H1 hits {int(n_h1)}")


if __name__ == "__main__":
    main()
