"""Per-tile nuclear morphometrics and Wilcoxon enrichment of
high-attention vs low-attention tiles."""
from __future__ import annotations

from dataclasses import dataclass

import click
import h5py
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import ranksums
from skimage import color, filters, measure, morphology


FEATURE_NAMES = [
    "nucleus_count", "nucleus_density", "mean_area", "mean_eccentricity",
    "intensity_var", "stromal_fraction",
]


def _segment_nuclei(rgb: np.ndarray) -> np.ndarray:
    he = color.rgb2hed(rgb)
    h = he[..., 0]
    thr = filters.threshold_otsu(h)
    mask = h > thr
    mask = morphology.remove_small_objects(mask, min_size=20)
    labels, _ = ndi.label(mask)
    return labels


def compute_tile_features(rgb: np.ndarray) -> dict[str, float]:
    labels = _segment_nuclei(rgb)
    H, W, _ = rgb.shape
    props = measure.regionprops(labels, intensity_image=color.rgb2gray(rgb))
    if not props:
        return {k: 0.0 for k in FEATURE_NAMES}
    areas = np.array([p.area for p in props])
    ecc   = np.array([p.eccentricity for p in props])
    ints  = np.array([p.mean_intensity for p in props])
    tissue_mask = color.rgb2hsv(rgb)[..., 2] < 0.8
    return {
        "nucleus_count":     float(len(props)),
        "nucleus_density":   float(len(props) / (H * W)),
        "mean_area":         float(areas.mean()),
        "mean_eccentricity": float(ecc.mean()),
        "intensity_var":     float(ints.var()),
        "stromal_fraction":  float(tissue_mask.mean()),
    }


@dataclass
class EnrichmentResult:
    feature: str
    stat: float
    p: float
    direction: str      # "hi>lo", "lo>hi", or "ns"


def wilcoxon_enrichment(hi: list[dict], lo: list[dict], *,
                        feature: str) -> EnrichmentResult:
    a = np.array([d[feature] for d in hi])
    b = np.array([d[feature] for d in lo])
    s, p = ranksums(a, b)
    dirn = "hi>lo" if a.mean() > b.mean() else "lo>hi"
    if p > 0.05:
        dirn = "ns"
    return EnrichmentResult(feature=feature, stat=float(s), p=float(p),
                            direction=dirn)


@click.command()
@click.option("--tiles", required=True, type=click.Path(exists=True))
@click.option("--attention", required=True, type=click.Path(exists=True),
              help="NPY of per-tile attention weights aligned to tiles.h5")
@click.option("--out", required=True, type=click.Path())
@click.option("--top-k", default=50, type=int)
def main(tiles, attention, out, top_k):
    with h5py.File(tiles, "r") as h:
        T = h["tiles"][:]
    attn = np.load(attention)
    hi_idx = np.argsort(-attn)[:top_k]
    lo_idx = np.argsort(attn)[:top_k]
    hi_feats = [compute_tile_features(T[i]) for i in hi_idx]
    lo_feats = [compute_tile_features(T[i]) for i in lo_idx]
    rows = [vars(wilcoxon_enrichment(hi_feats, lo_feats, feature=f))
            for f in FEATURE_NAMES]
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)


if __name__ == "__main__":
    main()
