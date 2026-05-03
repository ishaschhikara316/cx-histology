"""Build the training manifest (patient, cancer_type, emb_path).

Reads embedded HDF5s from INTERIM/embeddings/tcga/<encoder>/<cancer>/*.h5
and the TCGA CDR clinical table for cancer_type.
"""
from __future__ import annotations

import glob
from pathlib import Path

import click
import h5py
import pandas as pd

from workflow.scripts._common import patient_id_from_tcga


def _tile_count(emb_path: str) -> int:
    try:
        with h5py.File(emb_path, "r") as h:
            if "embeddings" not in h:
                return 0
            return int(h["embeddings"].shape[0])
    except Exception:
        return 0


@click.command()
@click.option("--embed-root", required=True, type=click.Path(exists=True))
@click.option("--encoder", required=True)
@click.option("--out", required=True, type=click.Path())
@click.option("--min-tiles", type=int, default=20,
              help="Skip slides with fewer than this many tiles.")
def main(embed_root, encoder, out, min_tiles):
    rows = []
    skipped = 0
    root = Path(embed_root) / encoder
    for cancer_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cancer = cancer_dir.name
        for emb in glob.glob(str(cancer_dir / "*.h5")):
            stem = Path(emb).stem
            try:
                pid = patient_id_from_tcga(stem)
            except ValueError:
                continue
            n_tiles = _tile_count(emb)
            if n_tiles < min_tiles:
                skipped += 1
                continue
            rows.append({"patient": pid, "cancer_type": cancer,
                         "emb_path": emb, "n_tiles": n_tiles})
    df = pd.DataFrame(rows)
    # When a patient has multiple slides, keep the one with most tiles.
    df = df.sort_values("n_tiles", ascending=False).drop_duplicates("patient")
    df = df[["patient", "cancer_type", "emb_path", "n_tiles"]]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(f"manifest: {len(df)} patients, skipped {skipped} slides with <{min_tiles} tiles")


if __name__ == "__main__":
    main()
