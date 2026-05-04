"""Train a single CV fold in isolation (drop-in for one fold of train_cv).

Pairs with workflow/scripts/train_cv_fold_parallel.sh which fans out 5
folds across 5 GPUs concurrently. Each invocation writes:
  out_dir/fold{k}.pt
  out_dir/fold{k}.metrics.json   (the dict run_fold normally returns)

The aggregator (in the bash launcher) collects all 5 metrics.json files
into a unified cv_summary.json identical in shape to what run_cv writes.
"""
from __future__ import annotations
import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
import yaml

from workflow.scripts._common import CX_COLUMNS, stratified_kfold_indices
from workflow.scripts.train_cv import TrainConfig, run_fold


@click.command()
@click.option("--manifest", required=True, type=click.Path(exists=True))
@click.option("--labels", required=True, type=click.Path(exists=True))
@click.option("--out-dir", required=True, type=click.Path())
@click.option("--encoder", required=True)
@click.option("--models-config", default="config/models.yaml")
@click.option("--params-config", default="config/params.yaml")
@click.option("--hidden", type=int, required=True)
@click.option("--dropout", type=float, required=True)
@click.option("--lr", type=float, required=True)
@click.option("--fold-idx", type=int, required=True)
@click.option("--device", default=None)
def main(manifest, labels, out_dir, encoder, models_config, params_config,
         hidden, dropout, lr, fold_idx, device):
    models_cfg = yaml.safe_load(open(models_config))
    params = yaml.safe_load(open(params_config))
    m = pd.read_csv(manifest, sep="\t").set_index("patient")
    lab_raw = pd.read_csv(labels, sep="\t")
    idx_col = "bcr_patient_barcode" if "bcr_patient_barcode" in lab_raw.columns else "patient"
    lab = lab_raw.set_index(idx_col)[CX_COLUMNS]
    cfg = TrainConfig(
        embed_dim=models_cfg["encoders"][encoder]["embed_dim"],
        hidden=hidden, dropout=dropout, lr=lr,
        batch_size=models_cfg["abmil"]["batch_size_slides"],
        n_epochs=models_cfg["abmil"]["n_epochs"],
        patience=models_cfg["abmil"]["early_stop_patience"],
        n_cx=models_cfg["abmil"]["n_cx"],
        device=device or ("cuda:0" if torch.cuda.is_available() else "cpu"),
        seed=params["reproducibility"]["random_seed"],
        weight_decay=models_cfg["abmil"]["weight_decay"],
        max_tiles_per_slide=params["tile"]["tiles_per_slide"])

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    shared = m.index.intersection(lab.index)
    m = m.loc[shared]; lab = lab.loc[shared]

    folds = stratified_kfold_indices(m, k=params["cv"]["n_folds"],
                                     stratify_col=params["cv"]["stratify_by"],
                                     seed=cfg.seed)
    if fold_idx < 0 or fold_idx >= len(folds):
        raise SystemExit(f"fold_idx {fold_idx} out of range (0..{len(folds)-1})")
    tr_ids, va_ids = folds[fold_idx]

    metrics = run_fold(fold_idx=fold_idx, train_ids=tr_ids, val_ids=va_ids,
                       labels=lab, manifest=m, cfg=cfg,
                       ckpt_path=out_dir / f"fold{fold_idx}.pt")
    with open(out_dir / f"fold{fold_idx}.metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[fold {fold_idx}] wrote fold{fold_idx}.metrics.json")


if __name__ == "__main__":
    main()
