"""Run a trained ABMIL checkpoint over external embeddings and save
per-patient CX predictions back-transformed to the raw scale."""
from __future__ import annotations

from pathlib import Path

import click
import h5py
import numpy as np
import pandas as pd
import torch

from workflow.scripts.abmil import GatedABMIL
from workflow.scripts.loss import invert_label_transform
from workflow.scripts._common import CX_COLUMNS


def _load_ckpt(path: str, device: str):
    # weights_only=False because our checkpoint includes numpy label stats.
    # Only load checkpoints we produced ourselves.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = GatedABMIL(in_dim=cfg["embed_dim"], hidden=cfg["hidden"],
                       n_out=cfg["n_cx"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mu = torch.from_numpy(ckpt["label_mu"])
    sd = torch.from_numpy(ckpt["label_sd"])
    return model, mu, sd


def predict(*, manifest: str, ckpt_path: str, out_tsv: str, device: str):
    model, mu, sd = _load_ckpt(ckpt_path, device=device)
    man = pd.read_csv(manifest, sep="\t")
    rows = []
    with torch.no_grad():
        for _, r in man.iterrows():
            with h5py.File(r["emb_path"], "r") as h:
                emb = torch.from_numpy(h["embeddings"][:]).unsqueeze(0).to(device)
            pred_z, attn = model(emb)
            pred = invert_label_transform(pred_z.cpu(), mu, sd).numpy()[0]
            row = {"patient": r["patient"]}
            row.update({c: float(pred[k]) for k, c in enumerate(CX_COLUMNS)})
            rows.append(row)
    df = pd.DataFrame(rows)
    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep="\t", index=False)


@click.command()
@click.option("--manifest", required=True, type=click.Path(exists=True))
@click.option("--ckpt", required=True, type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
@click.option("--device", default="cuda:0")
def main(manifest, ckpt, out, device):
    predict(manifest=manifest, ckpt_path=ckpt, out_tsv=out, device=device)


if __name__ == "__main__":
    main()
