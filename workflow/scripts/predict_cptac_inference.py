"""Run inference on CPTAC embeddings using a saved fold checkpoint.

Loads the ABMIL head checkpoint, then for every patient under
embed_root/<COHORT>/*.h5 produces (patient, cohort, CX1_pred..CX17_pred).
Output is a TSV that downstream validation scripts can use.
"""
from __future__ import annotations

import re
from pathlib import Path

import click
import h5py
import numpy as np
import pandas as pd
import torch

from workflow.scripts.abmil import GatedABMIL
from workflow.scripts.loss import invert_label_transform
from workflow.scripts._common import CX_COLUMNS

CPTAC_PID_RE = re.compile(r"([A-Z0-9-]+)__")


def patient_from_filename(stem: str) -> str:
    m = CPTAC_PID_RE.match(stem)
    return m.group(1) if m else stem.split("__")[0]


@click.command()
@click.option("--ckpt", required=True, type=click.Path(exists=True))
@click.option("--embed-root", required=True, type=click.Path(exists=True),
              help="data/interim/embeddings/cptac/uni")
@click.option("--out", required=True, type=click.Path())
@click.option("--device", default=None,
              help="cuda:0 / cpu (default: cuda if available)")
@click.option("--max-tiles", default=1000, type=int)
def main(ckpt, embed_root, out, device, max_tiles):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = state["config"]
    label_mu = torch.from_numpy(np.asarray(state["label_mu"]))
    label_sd = torch.from_numpy(np.asarray(state["label_sd"]))

    model = GatedABMIL(in_dim=cfg["embed_dim"], hidden=cfg["hidden"],
                       n_out=cfg["n_cx"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    rows = []
    rng = np.random.default_rng(cfg["seed"])
    embed_root = Path(embed_root)
    for cohort_dir in sorted(p for p in embed_root.iterdir() if p.is_dir()):
        cohort = cohort_dir.name
        for emb in sorted(cohort_dir.glob("*.h5")):
            stem = emb.stem
            pid = patient_from_filename(stem)
            try:
                with h5py.File(emb, "r") as h:
                    feats = h["embeddings"][:]
            except Exception as e:
                print(f"[skip] {emb}: {e}")
                continue
            if feats.shape[0] == 0:
                continue
            if feats.shape[0] > max_tiles:
                idx = rng.choice(feats.shape[0], max_tiles, replace=False)
                feats = feats[idx]
            x = torch.from_numpy(feats).unsqueeze(0).to(device).float()
            mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)
            with torch.no_grad():
                pred_z, _ = model(x, mask=mask)
            pred = invert_label_transform(pred_z.cpu(), label_mu, label_sd
                                          ).squeeze(0).numpy()
            row = {"patient": pid, "cohort": cohort, "series_stem": stem,
                   "n_tiles": int(feats.shape[0])}
            for k, c in enumerate(CX_COLUMNS):
                row[f"{c}_pred"] = float(pred[k])
            rows.append(row)

    df = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(f"wrote {out}: {len(df)} CPTAC patients")
    if len(df):
        print(df.cohort.value_counts().to_string())


if __name__ == "__main__":
    main()
