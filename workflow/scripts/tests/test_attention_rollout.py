import numpy as np
from pathlib import Path

from workflow.scripts.abmil import GatedABMIL
from workflow.scripts import attention_rollout as ar


def test_attention_heatmap_matches_slide_shape(tmp_path):
    model = GatedABMIL(8, 16, 17)
    coords = np.array([[0, 0], [256, 0], [0, 256], [256, 256]], dtype=np.int32)
    emb = np.random.randn(4, 8).astype(np.float32)

    heat = ar.compute_heatmap(model=model, embeddings=emb, coords=coords,
                              patch_px=256, slide_wh=(1024, 1024),
                              thumb_px=256, sig_idx=0)
    assert heat.shape == (256, 256)
    assert heat.min() >= 0.0
    assert heat.max() <= 1.0


def test_rank_tiles_by_attention_returns_top_indices():
    attn = np.array([0.1, 0.5, 0.2, 0.15, 0.05])
    idx = ar.rank_tiles(attn, k=2, mode="top")
    assert idx.tolist() == [1, 2]
    idx = ar.rank_tiles(attn, k=2, mode="bottom")
    assert idx.tolist() == [4, 0]
