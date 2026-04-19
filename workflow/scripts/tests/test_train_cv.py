import h5py
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from workflow.scripts import train_cv as tc


def _make_fake_dataset(tmp_path: Path, n_patients: int = 60,
                       n_tiles: int = 50, n_cancer: int = 3,
                       n_cx: int = 17, dim: int = 16):
    """Write per-patient embedding HDF5s + labels TSV + manifest TSV."""
    rng = np.random.default_rng(0)
    labels_rows = []
    manifest_rows = []
    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    cancers = [f"C{i}" for i in range(n_cancer)]
    for i in range(n_patients):
        pid = f"P{i:03d}"
        cancer = cancers[i % n_cancer]
        emb_path = emb_dir / f"{pid}.h5"
        with h5py.File(emb_path, "w") as h:
            h.create_dataset("embeddings",
                             data=rng.normal(size=(n_tiles, dim)).astype(np.float32))
            h.create_dataset("coords",
                             data=np.zeros((n_tiles, 2), dtype=np.int32))
        labels_rows.append({"patient": pid,
                            **{f"CX{k+1}": rng.uniform(0, 1) for k in range(n_cx)}})
        manifest_rows.append({"patient": pid, "cancer_type": cancer,
                              "emb_path": str(emb_path)})
    labels = pd.DataFrame(labels_rows).set_index("patient")
    manifest = pd.DataFrame(manifest_rows).set_index("patient")
    return labels, manifest


def test_train_one_fold_returns_metrics(tmp_path):
    labels, manifest = _make_fake_dataset(tmp_path)
    folds = [(manifest.index[:48].tolist(), manifest.index[48:].tolist())]
    cfg = tc.TrainConfig(embed_dim=16, hidden=32, dropout=0.2, lr=1e-3,
                         batch_size=4, n_epochs=2, patience=2, n_cx=17,
                         device="cpu", seed=0)
    out = tc.run_fold(fold_idx=0, train_ids=folds[0][0], val_ids=folds[0][1],
                      labels=labels, manifest=manifest, cfg=cfg,
                      ckpt_path=tmp_path / "ckpt.pt")
    assert "spearman_per_sig" in out
    assert len(out["spearman_per_sig"]) == 17
    assert (tmp_path / "ckpt.pt").exists()


def test_run_cv_aggregates_metrics(tmp_path):
    labels, manifest = _make_fake_dataset(tmp_path, n_patients=30,
                                          n_tiles=20, n_cancer=3)
    cfg = tc.TrainConfig(embed_dim=16, hidden=32, dropout=0.2, lr=1e-3,
                         batch_size=4, n_epochs=1, patience=2, n_cx=17,
                         device="cpu", seed=0)
    results = tc.run_cv(labels=labels, manifest=manifest, cfg=cfg,
                        n_folds=3, stratify_col="cancer_type",
                        out_dir=tmp_path / "cv")
    assert len(results["fold_metrics"]) == 3
    assert results["pan_cancer_spearman_mean"].shape == (17,)
