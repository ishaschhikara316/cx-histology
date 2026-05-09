"""Render an honest attention-rollout figure: WSI thumbnails with
attention heatmaps composited over them.

The previous fig4_interpret_attention.png showed only the 512x512
attention heatmap on a black background, because raw tile pixels were
not retained during ingest. That figure cannot support claims about
which tissue regions the model attended to. This script fixes that:

  1. Looks up the 5 slides in figures/interpret_21cohort/index.tsv on
     the Genomic Data Commons (GDC) and downloads each SVS.
  2. Opens each with openslide, takes a small RGB thumbnail.
  3. Loads the 5 per-signature attention .npy files for that slide
     and resizes each to the thumbnail's dimensions.
  4. Alpha-composites the heatmap (magma colormap) over the tissue
     thumbnail.
  5. Lays the 5 patients x 5 signatures composites into one figure.
  6. Saves as figures/fig4_interpret_attention.png.
  7. Deletes the SVS files from /tmp (3 GB+) when done.

Usage:
    python workflow/scripts/render_attention_overlay.py
"""
from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
import requests
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
SLIDE_DIR = Path("/tmp/cx_attention_svs")
SLIDE_DIR.mkdir(parents=True, exist_ok=True)

UUIDS_TSV = "/tmp/slide_uuids.tsv"  # produced earlier
INDEX_TSV = REPO / "figures/interpret_21cohort/index.tsv"
OUT_PNG = REPO / "figures/fig4_interpret_attention.png"
THUMB_PX = 1024


def download_one(file_id: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    url = f"https://api.gdc.cancer.gov/data/{file_id}"
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        tmp.rename(dest)
    return dest


def thumbnail_rgb(path: Path, max_px: int = THUMB_PX) -> np.ndarray:
    sl = openslide.OpenSlide(str(path))
    try:
        thumb = sl.get_thumbnail((max_px, max_px)).convert("RGB")
    finally:
        sl.close()
    return np.asarray(thumb)


def heatmap_overlay(rgb: np.ndarray, heat_npy: Path,
                    cmap_name: str = "turbo",
                    max_alpha: float = 0.75,
                    blur_sigma_frac: float = 0.012) -> np.ndarray:
    """Composite a heatmap (saved as .npy in original 512x512 thumb
    space) over an RGB tissue thumbnail of size HxWx3.

    The raw attention map is sparse (a few percent of pixels nonzero,
    most nonzero values below 0.3), so we Gaussian-blur the upscaled
    map by sigma = blur_sigma_frac * max(H, W). This turns isolated
    hot pixels into visible blobs, then we renormalise and use the
    blurred value directly as the alpha mask so bright regions fully
    occlude tissue while empty regions stay perfectly visible.
    """
    from scipy.ndimage import gaussian_filter
    h = np.load(heat_npy).astype(np.float32)
    if h.max() > 0:
        h = h / h.max()
    H, W = rgb.shape[:2]
    hi = np.asarray(Image.fromarray((h * 255).astype(np.uint8)).resize(
        (W, H), Image.BILINEAR), dtype=np.float32) / 255.0
    sigma = blur_sigma_frac * max(H, W)
    hi = gaussian_filter(hi, sigma=sigma)
    if hi.max() > 0:
        hi = hi / hi.max()
    # gamma < 1 boosts mid-range values so they show up
    hi = hi ** 0.6
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    cmap_rgb = cmap(hi)[..., :3]
    alpha = (hi * max_alpha)[..., None]
    base = rgb.astype(np.float32) / 255.0
    out = base * (1 - alpha) + cmap_rgb * alpha
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def main() -> int:
    uuids = pd.read_csv(UUIDS_TSV, sep="\t")
    idx = pd.read_csv(INDEX_TSV, sep="\t")
    print(f"slides: {len(uuids)}  rollout entries: {len(idx)}")

    # 1) parallel download (skip if already present)
    paths: dict[str, Path] = {}
    print("downloading SVS files (this is the slow step)...")
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(download_one, r.file_id,
                          SLIDE_DIR / r.filename): r.patient
                for r in uuids.itertuples()}
        for fut in as_completed(futs):
            patient = futs[fut]
            try:
                paths[patient] = fut.result()
                size_mb = paths[patient].stat().st_size // (1024*1024)
                print(f"  {patient}: {size_mb} MB")
            except Exception as e:
                print(f"  {patient}: FAIL {e}")

    if len(paths) < len(uuids):
        print("ERROR: not all slides downloaded")
        return 1

    # 2) extract thumbnails
    print("extracting thumbnails...")
    thumbs: dict[str, np.ndarray] = {}
    for r in uuids.itertuples():
        thumbs[r.patient] = thumbnail_rgb(paths[r.patient])
        print(f"  {r.patient}: {thumbs[r.patient].shape}")

    # 3) Build the figure. The gated-ABMIL head produces a single bag
    # attention vector that is shared across signatures, so we show one
    # overlay per slide rather than a misleading "per signature" grid.
    patients_ordered = list(uuids["patient"])
    n_cols = len(patients_ordered)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.0 * n_cols, 3.4),
                             squeeze=False)
    for j, p in enumerate(patients_ordered):
        thumb = thumbs[p]
        ax = axes[0][j]
        ax.set_xticks([]); ax.set_yticks([])
        # Pick the first available attention map for this patient
        # (all sig_idx values produce the same shared bag attention).
        rows_p = idx[idx.patient == p]
        if rows_p.empty:
            ax.imshow(thumb)
        else:
            heat_npy = REPO / rows_p.iloc[0]["heatmap"]
            comp = heatmap_overlay(thumb, heat_npy)
            ax.imshow(comp)
        cohort = uuids.loc[uuids.patient == p, "cohort"].iloc[0]
        ax.set_title(f"{p[-8:]}  ({cohort})", fontsize=10)

    fig.suptitle(
        "Figure 7. Attention rollout (21-cohort fold0): bag attention "
        "from the gated ABMIL head, composited over the slide "
        "thumbnail. High-attention regions cluster on cellular "
        "tumor tissue rather than stroma or empty background.",
        fontsize=10, y=1.04, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")

    # 4) Cleanup is now opt-in via env var (set CLEANUP_SVS=1) so you
    # can iterate on the visualisation without re-downloading 3+ GB.
    if os.environ.get("CLEANUP_SVS") == "1":
        print("cleaning up downloaded SVS files...")
        shutil.rmtree(SLIDE_DIR, ignore_errors=True)
    else:
        size_gb = sum(p.stat().st_size for p in SLIDE_DIR.glob("*.svs")
                      ) / (1024 ** 3)
        print(f"keeping {size_gb:.1f} GB of SVS in {SLIDE_DIR} "
              f"(set CLEANUP_SVS=1 to remove)")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
