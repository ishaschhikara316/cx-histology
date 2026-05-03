"""Stream-mode CPTAC ingest from IDC: download → tile → embed → delete.

Mirrors the TCGA stream_ingest.py interface but pulls DICOM-WSI series
from the public S3 bucket s3://idc-open-data/ (HTTPS-mirrored).

For each IDC series, we resolve the URL list, pick the largest .dcm
(highest pyramid level / native resolution), download via HTTP, tile +
embed via the existing tile_wsi pipeline, and delete the raw DICOM.

openslide handles DICOM-WSI directly, exposing it as a single-level
slide; the tile_wsi.py downsample-finding code correctly returns level
0 for native-resolution DICOM, so no tile_wsi changes are needed.

Usage:
    python stream_ingest_cptac.py \
        --manifest data/interim/manifests/cptac_idc_LUAD.tsv \
        --raw-dir data/raw/cptac/LUAD \
        --tiles-dir data/interim/tiles/cptac/LUAD \
        --embed-dir data/interim/embeddings/cptac/uni/LUAD \
        --embed-encoder uni
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import click
import h5py
import pandas as pd
import yaml
from PIL import Image

import numpy as np
import torch

from workflow.scripts.tile_wsi import write_tiles_hdf5
from workflow.scripts.embed_tiles import embed_slide, load_encoder


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _download_largest(urls: list[str], out_dir: Path,
                      timeout: int = 600) -> Path:
    """Download just the largest DICOM file in the series.

    The IDC URL list per series is the multi-resolution pyramid;
    the largest file is the highest-resolution level we want for
    tiling. Downloading only that file (rather than all 4-6 levels)
    saves roughly 4-6× on bandwidth and disk per series, and we
    don't need the lower levels because tile_wsi rebuilds the
    Otsu mask from a get_thumbnail() of the high-res image.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # The S3 URLs don't expose Content-Length without HEAD requests,
    # but the file basenames sort approximately by size for IDC.
    # Strategy: HEAD each URL, pick the largest.
    best_url, best_size = None, 0
    for u in urls:
        http = u.replace("s3://idc-open-data/",
                         "https://s3.amazonaws.com/idc-open-data/") \
                if u.startswith("s3://") else u
        try:
            head = subprocess.run(["curl", "-sIL", "--max-time", "30", http],
                                  capture_output=True, text=True, check=False)
            size = 0
            for line in head.stdout.splitlines():
                if line.lower().startswith("content-length:"):
                    size = int(line.split(":")[1].strip())
            if size > best_size:
                best_size = size
                best_url = http
        except Exception:
            continue
    if not best_url:
        raise RuntimeError("Could not HEAD any URL in series")
    local = out_dir / Path(best_url).name
    rc = subprocess.call(["curl", "-sSLfo", str(local), "--max-time",
                          str(timeout), best_url])
    if rc != 0:
        raise RuntimeError(f"curl failed for {best_url} (rc={rc})")
    return local


@click.command()
@click.option("--manifest", required=True, type=click.Path(exists=True),
              help="IDC manifest TSV (patient_id, series_uid, urls)")
@click.option("--raw-dir", required=True, type=click.Path())
@click.option("--tiles-dir", required=True, type=click.Path())
@click.option("--embed-dir", required=True, type=click.Path())
@click.option("--embed-encoder", required=True)
@click.option("--embed-device", default="cuda:0")
@click.option("--stain-reference", default="resources/stain_reference.png",
              type=click.Path())
@click.option("--models-config", default="config/models.yaml")
@click.option("--patch-px", default=256, type=int)
@click.option("--mpp-target", default=0.5, type=float)
@click.option("--tissue-fraction-min", default=0.5, type=float)
@click.option("--tiles-per-slide", default=1000, type=int)
@click.option("--laplacian-var-min", default=100.0, type=float)
@click.option("--embed-batch-size", default=64, type=int)
@click.option("--seed", default=20260419, type=int)
def main(manifest, raw_dir, tiles_dir, embed_dir, embed_encoder, embed_device,
         stain_reference, models_config, patch_px, mpp_target,
         tissue_fraction_min, tiles_per_slide, laplacian_var_min,
         embed_batch_size, seed):
    df = pd.read_csv(manifest, sep="\t")
    raw_dir = Path(raw_dir)
    tiles_dir = Path(tiles_dir)
    embed_dir = Path(embed_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    embed_dir.mkdir(parents=True, exist_ok=True)

    models_cfg = yaml.safe_load(open(models_config))
    encoder = load_encoder(embed_encoder, models_cfg["encoders"][embed_encoder],
                           device=embed_device)

    n_done, n_skip, n_fail = 0, 0, 0
    for i, row in df.iterrows():
        pid = row["patient_id"]
        suid = row["series_uid"]
        stem = f"{pid}__{suid}"
        out_emb = embed_dir / f"{stem}.h5"
        tile_h5 = tiles_dir / f"{stem}.h5"

        if out_emb.exists() and out_emb.stat().st_size > 0:
            n_skip += 1
            continue

        urls = [u for u in str(row["urls"]).split(";") if u]
        if not urls:
            _log(f"[{i+1}/{len(df)}] {stem}: no URLs")
            n_fail += 1
            continue

        series_raw = raw_dir / suid
        try:
            t0 = time.time()
            _log(f"[{i+1}/{len(df)}] {stem}: download")
            dcm = _download_largest(urls, series_raw)
            _log(f"  download {dcm.stat().st_size/1e6:.0f} MB in {time.time()-t0:.1f}s")

            if not (tile_h5.exists() and tile_h5.stat().st_size > 0):
                t0 = time.time()
                write_tiles_hdf5(
                    str(dcm), str(tile_h5),
                    patch_px=patch_px, mpp_target=mpp_target,
                    tissue_fraction_min=tissue_fraction_min,
                    tiles_per_slide=tiles_per_slide,
                    laplacian_var_min=laplacian_var_min,
                    stain_reference=stain_reference, seed=seed,
                )
                _log(f"  tile in {time.time()-t0:.1f}s")

            with h5py.File(tile_h5, "r") as h:
                n_tiles = h["tiles"].shape[0]
            if n_tiles == 0:
                _log(f"  {stem}: 0 tiles, marking embed empty")
                with h5py.File(out_emb, "w") as h:
                    h.create_dataset("embeddings",
                                     data=__import__("numpy").zeros((0, 1), dtype="float32"))
                n_done += 1
                shutil.rmtree(series_raw, ignore_errors=True)
                tile_h5.unlink(missing_ok=True)
                continue

            t0 = time.time()
            enc_cfg = models_cfg["encoders"][embed_encoder]
            embed_slide(
                tile_h5, out_emb, encoder=encoder,
                input_px=enc_cfg["input_px"],
                mean=tuple(enc_cfg["mean"]),
                std=tuple(enc_cfg["std"]),
                batch_size=embed_batch_size,
                device=embed_device,
                dtype=torch.float16 if embed_device.startswith("cuda")
                      else torch.float32,
            )
            _log(f"  embed {n_tiles} tiles in {time.time()-t0:.1f}s")
            n_done += 1

        except Exception as e:
            _log(f"  {stem}: FAIL {type(e).__name__}: {str(e)[:200]}")
            n_fail += 1
        finally:
            shutil.rmtree(series_raw, ignore_errors=True)
            tile_h5.unlink(missing_ok=True)

    _log(f"\n[summary] done={n_done} skipped={n_skip} failed={n_fail} "
         f"of {len(df)} series")


if __name__ == "__main__":
    main()
