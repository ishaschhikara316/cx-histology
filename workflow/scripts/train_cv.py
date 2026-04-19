"""5-fold cancer-type-stratified CV for ABMIL CX regression."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import click
import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from torch.utils.data import Dataset, DataLoader

from workflow.scripts._common import seed_everything, stratified_kfold_indices, CX_COLUMNS
from workflow.scripts.abmil import GatedABMIL
from workflow.scripts.loss import (MultiTaskHuber, fit_label_transform,
                                   apply_label_transform, invert_label_transform)


@dataclass
class TrainConfig:
    embed_dim: int
    hidden: int
    dropout: float
    lr: float
    batch_size: int
    n_epochs: int
    patience: int
    n_cx: int
    device: str
    seed: int
    weight_decay: float = 1.0e-5
    max_tiles_per_slide: int = 1000


class SlideBagDataset(Dataset):
    def __init__(self, patient_ids, labels_z, manifest, *, max_tiles: int,
                 rng_seed: int):
        self.ids = list(patient_ids)
        self.labels = labels_z.loc[self.ids].values.astype(np.float32)
        self.paths = manifest.loc[self.ids, "emb_path"].values
        self.max_tiles = max_tiles
        self.rng = np.random.default_rng(rng_seed)

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        with h5py.File(self.paths[i], "r") as h:
            emb = h["embeddings"][:]
        if emb.shape[0] > self.max_tiles:
            idx = self.rng.choice(emb.shape[0], self.max_tiles, replace=False)
            emb = emb[idx]
        return torch.from_numpy(emb), torch.from_numpy(self.labels[i])


def _collate_pad(batch):
    xs, ys = zip(*batch)
    N = max(x.shape[0] for x in xs)
    D = xs[0].shape[1]
    B = len(xs)
    out = torch.zeros(B, N, D)
    mask = torch.zeros(B, N, dtype=torch.bool)
    for i, x in enumerate(xs):
        out[i, :x.shape[0]] = x
        mask[i, :x.shape[0]] = True
    return out, mask, torch.stack(ys)


def run_fold(*, fold_idx: int, train_ids, val_ids,
             labels: pd.DataFrame, manifest: pd.DataFrame,
             cfg: TrainConfig, ckpt_path: Path):
    seed_everything(cfg.seed + fold_idx)
    y_train_raw = torch.from_numpy(labels.loc[train_ids].values.astype(np.float32))
    mu, sd = fit_label_transform(y_train_raw)
    labels_z = pd.DataFrame(
        apply_label_transform(
            torch.from_numpy(labels.values.astype(np.float32)), mu, sd).numpy(),
        index=labels.index, columns=labels.columns)

    tr_ds = SlideBagDataset(train_ids, labels_z, manifest,
                            max_tiles=cfg.max_tiles_per_slide,
                            rng_seed=cfg.seed + fold_idx)
    va_ds = SlideBagDataset(val_ids, labels_z, manifest,
                            max_tiles=cfg.max_tiles_per_slide,
                            rng_seed=cfg.seed + 1000 + fold_idx)
    tr_dl = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                       collate_fn=_collate_pad, num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False,
                       collate_fn=_collate_pad, num_workers=0)

    model = GatedABMIL(in_dim=cfg.embed_dim, hidden=cfg.hidden,
                       n_out=cfg.n_cx, dropout=cfg.dropout).to(cfg.device)
    loss_fn = MultiTaskHuber(n_out=cfg.n_cx).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

    best_val, best_state, patience = float("inf"), None, cfg.patience
    for ep in range(cfg.n_epochs):
        model.train()
        for x, mask, y in tr_dl:
            x, mask, y = x.to(cfg.device), mask.to(cfg.device), y.to(cfg.device)
            pred, _ = model(x, mask=mask)
            loss = loss_fn(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
        # Validation
        model.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for x, mask, y in va_dl:
                x, mask = x.to(cfg.device), mask.to(cfg.device)
                p, _ = model(x, mask=mask)
                preds.append(p.cpu()); tgts.append(y)
        P = torch.cat(preds).numpy(); T = torch.cat(tgts).numpy()
        val_loss = float(np.mean((P - T) ** 2))
        if val_loss < best_val - 1e-4:
            best_val, patience = val_loss, cfg.patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(),
                "config": asdict(cfg),
                "label_mu": mu.numpy(), "label_sd": sd.numpy()}, ckpt_path)

    # Final metrics on val, back-transformed.
    P_raw = invert_label_transform(torch.from_numpy(P), mu, sd).numpy()
    T_raw = labels.loc[val_ids].values
    rho = np.array([spearmanr(P_raw[:, k], T_raw[:, k]).correlation
                    for k in range(cfg.n_cx)])
    mae = np.mean(np.abs(P_raw - T_raw), axis=0)
    return {"fold": fold_idx, "val_ids": val_ids,
            "spearman_per_sig": rho.tolist(),
            "mae_per_sig": mae.tolist(),
            "val_loss": best_val}


def run_cv(*, labels: pd.DataFrame, manifest: pd.DataFrame,
           cfg: TrainConfig, n_folds: int, stratify_col: str,
           out_dir: Path):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    shared = manifest.index.intersection(labels.index)
    manifest = manifest.loc[shared]
    labels = labels.loc[shared]
    folds = stratified_kfold_indices(manifest, k=n_folds,
                                     stratify_col=stratify_col, seed=cfg.seed)
    all_metrics = []
    for i, (tr, va) in enumerate(folds):
        ckpt = out_dir / f"fold{i}.pt"
        all_metrics.append(run_fold(fold_idx=i, train_ids=tr, val_ids=va,
                                    labels=labels, manifest=manifest, cfg=cfg,
                                    ckpt_path=ckpt))
    rho = np.stack([np.asarray(m["spearman_per_sig"]) for m in all_metrics], axis=0)
    results = {"fold_metrics": all_metrics,
               "pan_cancer_spearman_mean": rho.mean(axis=0),
               "pan_cancer_spearman_std":  rho.std(axis=0)}
    with open(out_dir / "cv_summary.json", "w") as f:
        json.dump({"fold_metrics": all_metrics,
                   "pan_cancer_spearman_mean": rho.mean(axis=0).tolist(),
                   "pan_cancer_spearman_std":  rho.std(axis=0).tolist()},
                  f, indent=2)
    return results


@click.command()
@click.option("--manifest", required=True, type=click.Path(exists=True),
              help="TSV: patient, cancer_type, emb_path")
@click.option("--labels", required=True, type=click.Path(exists=True))
@click.option("--out-dir", required=True, type=click.Path())
@click.option("--encoder", required=True, help="uni or h_optimus_0")
@click.option("--models-config", default="config/models.yaml")
@click.option("--params-config", default="config/params.yaml")
@click.option("--hidden", type=int, required=True)
@click.option("--dropout", type=float, required=True)
@click.option("--lr", type=float, required=True)
def main(manifest, labels, out_dir, encoder, models_config, params_config,
         hidden, dropout, lr) -> None:
    models_cfg = yaml.safe_load(open(models_config))
    params = yaml.safe_load(open(params_config))
    m = pd.read_csv(manifest, sep="\t").set_index("patient")
    lab = pd.read_csv(labels, sep="\t").set_index("patient")[CX_COLUMNS]
    cfg = TrainConfig(
        embed_dim=models_cfg["encoders"][encoder]["embed_dim"],
        hidden=hidden, dropout=dropout, lr=lr,
        batch_size=models_cfg["abmil"]["batch_size_slides"],
        n_epochs=models_cfg["abmil"]["n_epochs"],
        patience=models_cfg["abmil"]["early_stop_patience"],
        n_cx=models_cfg["abmil"]["n_cx"],
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        seed=params["reproducibility"]["random_seed"],
        weight_decay=models_cfg["abmil"]["weight_decay"],
        max_tiles_per_slide=params["tile"]["tiles_per_slide"])
    run_cv(labels=lab, manifest=m, cfg=cfg,
           n_folds=params["cv"]["n_folds"],
           stratify_col=params["cv"]["stratify_by"],
           out_dir=Path(out_dir))


if __name__ == "__main__":
    main()
