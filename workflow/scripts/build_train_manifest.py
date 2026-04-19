"""Build the training manifest (patient, cancer_type, emb_path).

Reads embedded HDF5s from INTERIM/embeddings/tcga/<encoder>/<cancer>/*.h5
and the TCGA CDR clinical table for cancer_type.
"""
from __future__ import annotations

import glob
from pathlib import Path

import click
import pandas as pd

from workflow.scripts._common import patient_id_from_tcga


@click.command()
@click.option("--embed-root", required=True, type=click.Path(exists=True))
@click.option("--encoder", required=True)
@click.option("--out", required=True, type=click.Path())
def main(embed_root, encoder, out):
    rows = []
    root = Path(embed_root) / encoder
    for cancer_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cancer = cancer_dir.name
        for emb in glob.glob(str(cancer_dir / "*.h5")):
            stem = Path(emb).stem
            try:
                pid = patient_id_from_tcga(stem)
            except ValueError:
                continue
            rows.append({"patient": pid, "cancer_type": cancer,
                         "emb_path": emb})
    df = pd.DataFrame(rows).drop_duplicates("patient")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)


if __name__ == "__main__":
    main()
