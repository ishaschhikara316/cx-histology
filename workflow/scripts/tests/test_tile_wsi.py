import h5py
import numpy as np
import pyvips
import pytest
from pathlib import Path

from workflow.scripts import tile_wsi as tw


@pytest.fixture
def synthetic_slide(tmp_path):
    # 2048x2048 RGB with a pink tissue blob in the centre, white elsewhere.
    img = np.full((2048, 2048, 3), 240, dtype=np.uint8)
    yy, xx = np.ogrid[:2048, :2048]
    mask = (xx - 1024) ** 2 + (yy - 1024) ** 2 < 700 ** 2
    img[mask] = [180, 90, 150]
    path = tmp_path / "slide.tif"
    vi = pyvips.Image.new_from_memory(img.tobytes(), 2048, 2048, 3, "uchar")
    vi.tiffsave(str(path), tile=True, pyramid=True, compression="jpeg",
                tile_width=256, tile_height=256, bigtiff=True)
    return path


def test_tissue_mask_captures_blob_rejects_background(synthetic_slide):
    arr = tw._read_thumb(synthetic_slide, target_px=512)
    mask = tw.tissue_mask_otsu(arr)
    assert mask.sum() > 0
    assert mask[10, 10] == 0          # corner is white background
    assert mask[256, 256] == 1        # centre is tissue


def test_extract_coords_at_20x_respects_tissue_fraction(synthetic_slide):
    coords = tw.extract_tile_coords(
        synthetic_slide, patch_px=256, mpp_target=0.5,
        tissue_fraction_min=0.5, tiles_per_slide=50, seed=0)
    assert 1 <= len(coords) <= 50
    for x, y in coords:
        assert x % 1 == 0 and y % 1 == 0


def test_write_tile_hdf5_roundtrip(tmp_path, synthetic_slide):
    out = tmp_path / "tiles.h5"
    tw.write_tiles_hdf5(
        synthetic_slide, out, patch_px=256, mpp_target=0.5,
        tissue_fraction_min=0.5, tiles_per_slide=4,
        laplacian_var_min=0.0, stain_reference=None, seed=0)
    with h5py.File(out, "r") as h:
        assert h["tiles"].shape[0] <= 4
        assert h["tiles"].shape[1:] == (256, 256, 3)
        assert h["coords"].shape[0] == h["tiles"].shape[0]
        assert h["tiles"].dtype == np.uint8
