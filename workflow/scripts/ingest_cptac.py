"""CPTAC WSI ingest via TCIA REST API.

TCIA exposes cohort-level collections for CPTAC-BRCA, CPTAC-LUAD, etc.
We query /getSeries, then /getImage to stream each .svs.

Usage:
    python ingest_cptac.py --cohort CPTAC-LUAD --out-dir data/raw/cptac/LUAD
"""
from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
import requests

TCIA_BASE = "https://services.cancerimagingarchive.net/services/v4/TCIA/query"


def list_series(collection: str, modality: str = "SM") -> list[dict]:
    r = requests.get(f"{TCIA_BASE}/getSeries",
                     params={"Collection": collection, "Modality": modality,
                             "format": "json"}, timeout=60)
    r.raise_for_status()
    return r.json()


def download_series(series_uid: str, out_dir: Path) -> Path:
    r = requests.get(f"{TCIA_BASE}/getImage",
                     params={"SeriesInstanceUID": series_uid},
                     stream=True, timeout=600)
    r.raise_for_status()
    out = out_dir / f"{series_uid}.zip"
    with open(out, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    return out


@click.command()
@click.option("--cohort", required=True, help="e.g. CPTAC-LUAD")
@click.option("--out-dir", required=True, type=click.Path())
def main(cohort: str, out_dir: str) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    series = list_series(cohort)
    rows = []
    for s in series:
        zpath = download_series(s["SeriesInstanceUID"], out)
        rows.append({"series_uid": s["SeriesInstanceUID"],
                     "patient_id": s.get("PatientID"),
                     "zip_path": str(zpath)})
    pd.DataFrame(rows).to_csv(out / "_manifest.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
