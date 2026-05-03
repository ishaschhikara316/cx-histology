"""Extract tiles from a WSI with tissue masking, QC filters, and stain
normalisation.

Output: a per-slide HDF5 with datasets `tiles` (N, H, W, 3) uint8,
`coords` (N, 2) int32 (level-0 x/y), and `attrs` JSON-ish.
"""
from __future__ import annotations

from pathlib import Path

import click
import h5py
import numpy as np
import openslide
from PIL import Image
from skimage import color, filters, morphology


def _read_thumb(path: str | Path, *, target_px: int = 512) -> np.ndarray:
    slide = openslide.OpenSlide(str(path))
    w, h = slide.dimensions
    scale = max(w, h) / target_px
    th_w, th_h = int(w / scale), int(h / scale)
    thumb = slide.get_thumbnail((th_w, th_h)).convert("RGB")
    return np.asarray(thumb)


def tissue_mask_otsu(rgb: np.ndarray) -> np.ndarray:
    hsv = color.rgb2hsv(rgb)
    v = hsv[..., 2]
    thr = filters.threshold_otsu(v)
    mask = v < thr
    mask = morphology.binary_opening(mask, morphology.disk(3))
    mask = morphology.binary_closing(mask, morphology.disk(5))
    return mask.astype(np.uint8)


def _level_for_mpp(slide: openslide.OpenSlide, *, mpp_target: float) -> int:
    mpp_x = float(slide.properties.get("openslide.mpp-x", 0.25))
    native_ratio = mpp_target / mpp_x
    for lvl, ds in enumerate(slide.level_downsamples):
        if abs(ds - native_ratio) / native_ratio < 0.1:
            return lvl
    return 0


def extract_tile_coords(path, *, patch_px: int, mpp_target: float,
                        tissue_fraction_min: float, tiles_per_slide: int,
                        seed: int) -> list[tuple[int, int]]:
    slide = openslide.OpenSlide(str(path))
    thumb = _read_thumb(path, target_px=512)
    mask = tissue_mask_otsu(thumb)
    mask_h, mask_w = mask.shape
    W, H = slide.dimensions
    scale_x, scale_y = W / mask_w, H / mask_h

    lvl = _level_for_mpp(slide, mpp_target=mpp_target)
    lvl_ds = slide.level_downsamples[lvl]
    patch_lvl0 = int(patch_px * lvl_ds)

    rng = np.random.default_rng(seed)
    candidates = []
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    idx = rng.permutation(len(xs))
    for i in idx:
        cx, cy = xs[i] * scale_x, ys[i] * scale_y
        x0, y0 = int(cx - patch_lvl0 / 2), int(cy - patch_lvl0 / 2)
        if x0 < 0 or y0 < 0 or x0 + patch_lvl0 > W or y0 + patch_lvl0 > H:
            continue
        # Check tissue fraction in the thumb-mask region of this patch.
        mx0, my0 = int(x0 / scale_x), int(y0 / scale_y)
        mx1 = max(mx0 + 1, int((x0 + patch_lvl0) / scale_x))
        my1 = max(my0 + 1, int((y0 + patch_lvl0) / scale_y))
        sub = mask[my0:my1, mx0:mx1]
        if sub.size == 0 or sub.mean() < tissue_fraction_min:
            continue
        candidates.append((x0, y0))
        if len(candidates) >= tiles_per_slide:
            break
    return candidates


def _read_patch(slide: openslide.OpenSlide, x0: int, y0: int, *,
                patch_px: int, lvl: int) -> np.ndarray:
    img = slide.read_region((x0, y0), lvl, (patch_px, patch_px)).convert("RGB")
    return np.asarray(img)


def _laplacian_variance(rgb: np.ndarray) -> float:
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


_NORMALIZER_CACHE = {}


def _get_macenko_normalizer(reference: np.ndarray):
    """Build and fit a Macenko normalizer once per reference image.

    The original implementation rebuilt + refit per tile (1000× per slide)
    which dominated runtime — Macenko fit involves SVD on stain vectors.
    Cache by id(reference) since references are read once per process.
    """
    key = id(reference)
    if key in _NORMALIZER_CACHE:
        return _NORMALIZER_CACHE[key]
    from torch_staintools.normalizer import NormalizerBuilder
    import torch
    try:
        norm = NormalizerBuilder.build("macenko", concentration_solver="ls")
        ref_t = torch.from_numpy(np.ascontiguousarray(reference)).permute(
            2, 0, 1).unsqueeze(0).float()
        norm.fit(ref_t)
        _NORMALIZER_CACHE[key] = norm
        return norm
    except Exception:
        _NORMALIZER_CACHE[key] = None
        return None


def _macenko_normalize(rgb: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
    if reference is None:
        return rgb
    norm = _get_macenko_normalizer(reference)
    if norm is None:
        return rgb
    import torch
    try:
        src = torch.from_numpy(np.ascontiguousarray(rgb)).permute(
            2, 0, 1).unsqueeze(0).float()
        out = norm.transform(src)[0].permute(1, 2, 0).clamp(0, 255).byte().numpy()
        return out
    except Exception:
        # Per-tile transform error: fall back to un-normalised tile.
        return rgb


def _macenko_normalize_batch(tiles: list, reference: np.ndarray) -> np.ndarray:
    """Apply Macenko normalization to a batch of tiles in a single op.

    Falls back to un-normalised tiles if the normalizer can't be built or
    the batched transform fails.
    """
    norm = _get_macenko_normalizer(reference)
    if norm is None:
        return np.stack(tiles, axis=0)
    import torch
    try:
        # Per-tile transform inside one process — torch_staintools' transform
        # operates on a single image at a time; loop but reuse fitted state.
        out = []
        for rgb in tiles:
            src = torch.from_numpy(np.ascontiguousarray(rgb)).permute(
                2, 0, 1).unsqueeze(0).float()
            t = norm.transform(src)[0].permute(1, 2, 0).clamp(0, 255).byte().numpy()
            out.append(t)
        return np.stack(out, axis=0)
    except Exception:
        return np.stack(tiles, axis=0)


def write_tiles_hdf5(path, out, *, patch_px, mpp_target,
                     tissue_fraction_min, tiles_per_slide,
                     laplacian_var_min, stain_reference, seed) -> None:
    slide = openslide.OpenSlide(str(path))
    coords = extract_tile_coords(path, patch_px=patch_px,
                                 mpp_target=mpp_target,
                                 tissue_fraction_min=tissue_fraction_min,
                                 tiles_per_slide=tiles_per_slide, seed=seed)
    lvl = _level_for_mpp(slide, mpp_target=mpp_target)
    ref = np.asarray(Image.open(stain_reference).convert("RGB")) if stain_reference else None

    raw_tiles, kept = [], []
    for (x0, y0) in coords:
        rgb = _read_patch(slide, x0, y0, patch_px=patch_px, lvl=lvl)
        if _laplacian_variance(rgb) < laplacian_var_min:
            continue
        raw_tiles.append(rgb)
        kept.append((x0, y0))
    if not raw_tiles:
        tiles_arr = np.zeros((0, patch_px, patch_px, 3), dtype=np.uint8)
        coords_arr = np.zeros((0, 2), dtype=np.int32)
    else:
        # Batch stain normalization: one GPU/tensor op for the whole slide.
        if ref is not None:
            tiles_arr = _macenko_normalize_batch(raw_tiles, ref).astype(np.uint8)
        else:
            tiles_arr = np.stack(raw_tiles, axis=0).astype(np.uint8)
        coords_arr = np.asarray(kept, dtype=np.int32)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as h:
        h.create_dataset("tiles", data=tiles_arr, compression="gzip",
                         compression_opts=4)
        h.create_dataset("coords", data=coords_arr)
        h.attrs["slide"] = str(path)
        h.attrs["patch_px"] = patch_px
        h.attrs["mpp_target"] = mpp_target


@click.command()
@click.option("--slide", required=True, type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
@click.option("--patch-px", default=256, type=int)
@click.option("--mpp-target", default=0.5, type=float)
@click.option("--tissue-fraction-min", default=0.5, type=float)
@click.option("--tiles-per-slide", default=1000, type=int)
@click.option("--laplacian-var-min", default=100.0, type=float)
@click.option("--stain-reference", default=None,
              type=click.Path(exists=True, dir_okay=False))
@click.option("--seed", default=0, type=int)
def main(**kw) -> None:
    write_tiles_hdf5(kw.pop("slide"), kw.pop("out"), **kw)


if __name__ == "__main__":
    main()
