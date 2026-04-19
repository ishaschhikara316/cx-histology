import h5py
import numpy as np
import torch
import pytest
from pathlib import Path

from workflow.scripts import embed_tiles as em


class _MockEncoder(torch.nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.dim = dim
        self.proj = torch.nn.Linear(3 * 4 * 4, dim, bias=False)

    def forward(self, x):  # x: (B, 3, 224, 224)
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, 4)
        return self.proj(pooled.flatten(1))


@pytest.fixture
def tile_h5(tmp_path):
    p = tmp_path / "tiles.h5"
    with h5py.File(p, "w") as h:
        rng = np.random.default_rng(0)
        h.create_dataset("tiles",
                         data=rng.integers(0, 255, (8, 256, 256, 3),
                                           dtype=np.uint8))
        h.create_dataset("coords", data=np.zeros((8, 2), dtype=np.int32))
    return p


def test_preprocess_batch_shape_and_dtype():
    batch = np.random.randint(0, 255, (2, 256, 256, 3), dtype=np.uint8)
    t = em.preprocess_batch(batch, input_px=224,
                            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    assert t.shape == (2, 3, 224, 224)
    assert t.dtype == torch.float32


def test_embed_slide_writes_correct_shape(tile_h5, tmp_path):
    out = tmp_path / "emb.h5"
    encoder = _MockEncoder(dim=16)
    em.embed_slide(tile_h5, out, encoder=encoder,
                   input_px=224, mean=(0.5,)*3, std=(0.5,)*3,
                   batch_size=4, device="cpu", dtype=torch.float32)
    with h5py.File(out) as h:
        assert h["embeddings"].shape == (8, 16)
        assert h["coords"].shape == (8, 2)
