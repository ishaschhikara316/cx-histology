"""Download, tile, and delete TCGA WSIs one at a time to bound disk usage.

For each manifest row:
  1. gdc-client download  →  data/raw/tcga/<cancer>/<file_id>/<slide>.svs
  2. tile_wsi.write_tiles_hdf5  →  data/interim/tiles/tcga/<cancer>/<stem>.h5
  3. shutil.rmtree on the <file_id>/ dir to free disk.

Peak disk footprint is bounded by one slide (~500 MB) plus accumulated
.h5 outputs (~50–100 MB per slide). Resumes by skipping slides whose
.h5 already exists and is non-empty.

Bootstraps `resources/stain_reference.png` from the first slide's first
tile if the reference does not yet exist. The first slide is tiled
un-normalised in that case; re-tile it later if stain consistency
matters for that specific slide.

Usage:
    python stream_ingest.py \
        --manifest data/interim/manifests/tcga_LIHC.tsv \
        --raw-dir data/raw/tcga/LIHC \
        --tiles-dir data/interim/tiles/tcga/LIHC \
        --stain-reference resources/stain_reference.png \
        --patch-px 256 --mpp-target 0.5 \
        --tissue-fraction-min 0.5 --tiles-per-slide 1000 \
        --laplacian-var-min 100 --seed 20260419
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import click
import h5py
import pandas as pd
from PIL import Image

from workflow.scripts.tile_wsi import write_tiles_hdf5


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _download_one(file_id: str, raw_dir: Path, *, token: Path | None) -> Path:
    cmd = ["gdc-client", "download", file_id, "-d", str(raw_dir), "-n", "1"]
    if token is not None:
        cmd += ["-t", str(token)]
    subprocess.run(cmd, check=True)
    candidates = list((raw_dir / file_id).glob("*.svs"))
    if not candidates:
        raise FileNotFoundError(f"no .svs under {raw_dir}/{file_id}")
    return candidates[0]


def _save_reference_from_h5(h5_path: Path, ref_path: Path) -> None:
    with h5py.File(h5_path, "r") as h:
        if h["tiles"].shape[0] == 0:
            return
        tile = h["tiles"][0]
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tile).save(ref_path)
    _log(f"stain_reference bootstrapped: {ref_path}")


@click.command()
@click.option("--manifest", required=True, type=click.Path(exists=True))
@click.option("--raw-dir", required=True, type=click.Path())
@click.option("--tiles-dir", required=True, type=click.Path())
@click.option("--stain-reference", required=True, type=click.Path())
@click.option("--token", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="Optional GDC token for controlled-access files.")
@click.option("--patch-px", default=256, type=int)
@click.option("--mpp-target", default=0.5, type=float)
@click.option("--tissue-fraction-min", default=0.5, type=float)
@click.option("--tiles-per-slide", default=1000, type=int)
@click.option("--laplacian-var-min", default=100.0, type=float)
@click.option("--seed", default=20260419, type=int)
@click.option("--marker", default=None, type=click.Path(),
              help="Optional completion marker file to touch on success.")
def main(manifest, raw_dir, tiles_dir, stain_reference, token,
         patch_px, mpp_target, tissue_fraction_min, tiles_per_slide,
         laplacian_var_min, seed, marker):
    raw = Path(raw_dir);   raw.mkdir(parents=True, exist_ok=True)
    tiles = Path(tiles_dir); tiles.mkdir(parents=True, exist_ok=True)
    ref = Path(stain_reference)
    df = pd.read_csv(manifest, sep="\t")

    n_done, n_skip, n_fail = 0, 0, 0
    t0 = time.time()

    for i, row in df.iterrows():
        stem = Path(row["file_name"]).stem
        out_h5 = tiles / f"{stem}.h5"
        if out_h5.exists() and out_h5.stat().st_size > 0:
            n_skip += 1
            continue

        _log(f"[{i + 1}/{len(df)}] {stem}  ({int(row['file_size']) / 1e6:.0f} MB)")
        try:
            svs = _download_one(row["file_id"], raw,
                                token=Path(token) if token else None)
        except Exception as e:
            _log(f"  download failed: {e}")
            n_fail += 1
            continue

        try:
            use_ref = ref if ref.exists() else None
            write_tiles_hdf5(
                svs, out_h5,
                patch_px=patch_px, mpp_target=mpp_target,
                tissue_fraction_min=tissue_fraction_min,
                tiles_per_slide=tiles_per_slide,
                laplacian_var_min=laplacian_var_min,
                stain_reference=use_ref, seed=seed,
            )
            if not ref.exists() and out_h5.exists():
                _save_reference_from_h5(out_h5, ref)
            n_done += 1
        except Exception as e:
            _log(f"  tile failed: {e}")
            n_fail += 1
        finally:
            shutil.rmtree(raw / row["file_id"], ignore_errors=True)

    dt = time.time() - t0
    _log(f"done: {n_done} tiled, {n_skip} skipped, {n_fail} failed in {dt:.0f}s")
    if marker:
        Path(marker).parent.mkdir(parents=True, exist_ok=True)
        Path(marker).touch()


if __name__ == "__main__":
    main()
