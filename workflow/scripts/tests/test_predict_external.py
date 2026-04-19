import h5py
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from dataclasses import asdict

from workflow.scripts.abmil import GatedABMIL
from workflow.scripts.train_cv import TrainConfig
from workflow.scripts import predict_external as pe


def test_predict_external_writes_one_row_per_patient(tmp_path):
    # 1) make a tiny checkpoint with mu, sd
    cfg = TrainConfig(embed_dim=8, hidden=16, dropout=0.2, lr=1e-3,
                      batch_size=2, n_epochs=1, patience=1, n_cx=17,
                      device="cpu", seed=0)
    model = GatedABMIL(in_dim=8, hidden=16, n_out=17, dropout=0.2)
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"state_dict": model.state_dict(),
                "config": asdict(cfg),
                "label_mu": np.zeros(17, dtype=np.float32),
                "label_sd": np.ones(17, dtype=np.float32)}, ckpt)

    # 2) make two embedding files
    emb_dir = tmp_path / "emb"; emb_dir.mkdir()
    for pid in ["P001", "P002"]:
        with h5py.File(emb_dir / f"{pid}.h5", "w") as h:
            h.create_dataset("embeddings",
                             data=np.random.randn(30, 8).astype(np.float32))
            h.create_dataset("coords", data=np.zeros((30, 2), dtype=np.int32))

    # 3) manifest
    man = pd.DataFrame({"patient": ["P001", "P002"],
                        "emb_path": [str(emb_dir / "P001.h5"),
                                     str(emb_dir / "P002.h5")]})
    man_tsv = tmp_path / "man.tsv"
    man.to_csv(man_tsv, sep="\t", index=False)

    out_tsv = tmp_path / "preds.tsv"
    pe.predict(manifest=str(man_tsv), ckpt_path=str(ckpt),
               out_tsv=str(out_tsv), device="cpu")
    df = pd.read_csv(out_tsv, sep="\t")
    assert list(df["patient"]) == ["P001", "P002"]
    assert [c for c in df.columns if c.startswith("CX")] == [f"CX{k}" for k in range(1, 18)]
