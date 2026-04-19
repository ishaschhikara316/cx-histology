"""Per-slide attention heat-maps and top/bottom-tile ranking."""
from __future__ import annotations

from pathlib import Path

import click
import h5py
import numpy as np
import torch

from workflow.scripts.abmil import GatedABMIL


def rank_tiles(attn: np.ndarray, k: int, *, mode: str = "top") -> np.ndarray:
    order = np.argsort(-attn) if mode == "top" else np.argsort(attn)
    return order[:k]


def compute_heatmap(*, model: GatedABMIL, embeddings: np.ndarray,
                    coords: np.ndarray, patch_px: int,
                    slide_wh: tuple[int, int], thumb_px: int,
                    sig_idx: int) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(embeddings).unsqueeze(0).float()
    with torch.no_grad():
        _, attn = model(x)
    a = attn[0].numpy()
    # Gradient w.r.t. the chosen signature not available from the pooled
    # output directly with this head; we use the vanilla attention as a
    # per-signature-agnostic proxy. Per-signature maps come from a
    # per-signature attention refinement step (stretch; see docs/methods).
    W, H = slide_wh
    scale = thumb_px / max(W, H)
    heat = np.zeros((thumb_px, thumb_px), dtype=np.float32)
    for (x0, y0), w in zip(coords, a):
        u0, v0 = int(x0 * scale), int(y0 * scale)
        u1 = min(thumb_px, u0 + max(1, int(patch_px * scale)))
        v1 = min(thumb_px, v0 + max(1, int(patch_px * scale)))
        heat[v0:v1, u0:u1] = np.maximum(heat[v0:v1, u0:u1], w)
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat


@click.command()
@click.option("--emb", required=True, type=click.Path(exists=True))
@click.option("--ckpt", required=True, type=click.Path(exists=True))
@click.option("--slide-wh", required=True, help="W,H")
@click.option("--out", required=True, type=click.Path())
@click.option("--sig-idx", default=0, type=int)
def main(emb, ckpt, slide_wh, out, sig_idx):
    W, H = map(int, slide_wh.split(","))
    with h5py.File(emb, "r") as h:
        e, c = h["embeddings"][:], h["coords"][:]
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = GatedABMIL(cfg["embed_dim"], cfg["hidden"], cfg["n_cx"],
                       dropout=cfg["dropout"])
    model.load_state_dict(ck["state_dict"])
    heat = compute_heatmap(model=model, embeddings=e, coords=c,
                           patch_px=256, slide_wh=(W, H),
                           thumb_px=512, sig_idx=sig_idx)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.save(out, heat)


if __name__ == "__main__":
    main()
