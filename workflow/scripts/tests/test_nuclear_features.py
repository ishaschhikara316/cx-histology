import numpy as np

from workflow.scripts import nuclear_features as nf


def _synthetic_tile(n_nuclei: int, size: int = 256):
    img = np.full((size, size, 3), 220, dtype=np.uint8)
    rng = np.random.default_rng(0)
    for _ in range(n_nuclei):
        cx, cy = rng.integers(20, size - 20, size=2)
        yy, xx = np.ogrid[:size, :size]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < 64   # r=8 circle
        img[mask] = [60, 30, 80]       # dark purple = nuclei
    return img


def test_features_capture_nucleus_density():
    dense = _synthetic_tile(60)
    sparse = _synthetic_tile(10)
    d = nf.compute_tile_features(dense)
    s = nf.compute_tile_features(sparse)
    assert d["nucleus_count"] > s["nucleus_count"]
    assert d["nucleus_density"] > s["nucleus_density"]


def test_enrichment_wilcoxon_detects_difference():
    # High-attention tiles have 60 nuclei, low have 10.
    hi = [nf.compute_tile_features(_synthetic_tile(60)) for _ in range(5)]
    lo = [nf.compute_tile_features(_synthetic_tile(10)) for _ in range(5)]
    res = nf.wilcoxon_enrichment(hi, lo, feature="nucleus_density")
    assert res.p < 0.05
    assert res.direction == "hi>lo"
