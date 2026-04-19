"""Select top-K and bottom-K attention tiles per CX signature across the
held-out test set and save as a visual atlas PNG grid."""
from __future__ import annotations

from pathlib import Path

import click
import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image

from workflow.scripts.abmil import GatedABMIL


def _load_model(ckpt_path: str, device: str) -> GatedABMIL:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    m = GatedABMIL(cfg["embed_dim"], cfg["hidden"], cfg["n_cx"],
                   dropout=cfg["dropout"]).to(device).eval()
    m.load_state_dict(ck["state_dict"])
    return m


@click.command()
@click.option("--manifest", required=True, type=click.Path(exists=True),
              help="TSV: patient, tile_path, emb_path")
@click.option("--ckpt", required=True, type=click.Path(exists=True))
@click.option("--out-dir", required=True, type=click.Path())
@click.option("--top-k", default=5, type=int)
@click.option("--n-signatures", default=17, type=int)
@click.option("--device", default="cuda:0")
def main(manifest, ckpt, out_dir, top_k, n_signatures, device):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    model = _load_model(ckpt, device)
    man = pd.read_csv(manifest, sep="\t")
    pooled_tiles = {k: [] for k in range(n_signatures)}
    for _, r in man.iterrows():
        with h5py.File(r["emb_path"], "r") as h:
            emb = torch.from_numpy(h["embeddings"][:]).unsqueeze(0).to(device)
        with torch.no_grad():
            _, attn = model(emb)
        a = attn[0].cpu().numpy()
        with h5py.File(r["tile_path"], "r") as h:
            tiles = h["tiles"][:]
        order = np.argsort(-a)
        for k in range(n_signatures):
            for j in order[:top_k]:
                pooled_tiles[k].append((float(a[j]), tiles[j]))
    for k in range(n_signatures):
        pooled_tiles[k].sort(reverse=True, key=lambda x: x[0])
        top = pooled_tiles[k][:top_k]
        if not top: continue
        grid = np.concatenate([t for _, t in top], axis=1)
        Image.fromarray(grid).save(out / f"cx{k+1}_top.png")


if __name__ == "__main__":
    main()
