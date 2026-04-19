"""Run a frozen foundation model over a slide's tiles and cache the
embeddings to HDF5.

Usage:
    python embed_tiles.py --tiles tiles.h5 --out emb.h5 --encoder uni
"""
from __future__ import annotations

from pathlib import Path

import click
import h5py
import numpy as np
import torch
import yaml
from torch.nn import functional as F


def preprocess_batch(tiles_u8: np.ndarray, *, input_px: int,
                     mean: tuple[float, float, float],
                     std: tuple[float, float, float]) -> torch.Tensor:
    x = torch.from_numpy(tiles_u8).float().div_(255.0).permute(0, 3, 1, 2)
    if x.shape[-1] != input_px:
        x = F.interpolate(x, size=input_px, mode="bilinear",
                          align_corners=False, antialias=True)
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    return (x - m) / s


def load_encoder(name: str, cfg: dict, device: str):
    """Load a pathology FM from HuggingFace; freeze and eval-mode."""
    import timm
    from huggingface_hub import snapshot_download, login

    tok_path = Path.home() / ".hf_token"
    if tok_path.exists():
        login(tok_path.read_text().strip())

    local = snapshot_download(cfg["hf_repo"], revision=cfg["hf_revision"])
    model = timm.create_model(cfg["arch"], pretrained=False,
                              num_classes=0, dynamic_img_size=True)
    ckpt_files = list(Path(local).glob("*.bin")) + list(Path(local).glob("*.safetensors"))
    if not ckpt_files:
        raise FileNotFoundError(f"No weights in {local}")
    if ckpt_files[0].suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(ckpt_files[0])
    else:
        state = torch.load(ckpt_files[0], map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval().requires_grad_(False).to(device)
    return model


def embed_slide(tile_path, out_path, *, encoder, input_px: int,
                mean: tuple[float, float, float],
                std: tuple[float, float, float],
                batch_size: int, device: str,
                dtype: torch.dtype = torch.float16) -> None:
    encoder = encoder.to(device).eval()
    with h5py.File(tile_path, "r") as h:
        tiles = h["tiles"][:]
        coords = h["coords"][:]
    N = tiles.shape[0]
    feats = []
    for i in range(0, N, batch_size):
        batch = tiles[i : i + batch_size]
        x = preprocess_batch(batch, input_px=input_px, mean=mean, std=std)
        x = x.to(device, dtype=dtype, non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type=device.split(":")[0],
                                             dtype=dtype, enabled=(dtype != torch.float32)):
            z = encoder(x)
        feats.append(z.float().cpu().numpy())
    embs = np.concatenate(feats, axis=0) if feats else np.zeros((0, 0), np.float32)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h:
        h.create_dataset("embeddings", data=embs, compression="gzip",
                         compression_opts=4)
        h.create_dataset("coords", data=coords)


@click.command()
@click.option("--tiles", required=True, type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
@click.option("--encoder", required=True, type=str)
@click.option("--models-config", default="config/models.yaml",
              type=click.Path(exists=True))
@click.option("--batch-size", default=32, type=int)
@click.option("--device", default="cuda:0")
def main(tiles: str, out: str, encoder: str, models_config: str,
         batch_size: int, device: str) -> None:
    cfg = yaml.safe_load(open(models_config))["encoders"][encoder]
    enc = load_encoder(encoder, cfg, device=device)
    embed_slide(tiles, out, encoder=enc, input_px=cfg["input_px"],
                mean=tuple(cfg["mean"]), std=tuple(cfg["std"]),
                batch_size=batch_size, device=device,
                dtype=torch.float16 if device.startswith("cuda") else torch.float32)


if __name__ == "__main__":
    main()
