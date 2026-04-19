"""Build a GDC manifest for TCGA diagnostic H&E slides.

Usage:
    python gdc_manifest.py --project TCGA-LUAD --out data/manifests/tcga_luad.tsv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
import pandas as pd
import requests

from workflow.scripts._common import patient_id_from_tcga

GDC_FILES = "https://api.gdc.cancer.gov/files"


def build_filters(*, project: str, data_type: str,
                  experimental_strategy: str, data_format: str) -> dict[str, Any]:
    return {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id",
                                     "value": [project]}},
            {"op": "in", "content": {"field": "files.data_type",
                                     "value": [data_type]}},
            {"op": "in", "content": {"field": "files.experimental_strategy",
                                     "value": [experimental_strategy]}},
            {"op": "in", "content": {"field": "files.data_format",
                                     "value": [data_format]}},
        ],
    }


def hits_to_manifest(hits: list[dict], *, project: str) -> list[dict]:
    rows = []
    for h in hits:
        rows.append({
            "file_id":   h["file_id"],
            "file_name": h["file_name"],
            "file_size": int(h["file_size"]),
            "md5":       h["md5sum"],
            "project":   project,
            "patient":   patient_id_from_tcga(h["file_name"]),
        })
    return rows


def query(project: str, *, page_size: int = 1000) -> list[dict]:
    filters = build_filters(project=project, data_type="Slide Image",
                            experimental_strategy="Diagnostic Slide",
                            data_format="SVS")
    params = {
        "filters": json.dumps(filters),
        "fields":  "file_id,file_name,file_size,md5sum",
        "format":  "JSON",
        "size":    str(page_size),
    }
    resp = requests.get(GDC_FILES, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()["data"]["hits"]


@click.command()
@click.option("--project", required=True, help="e.g. TCGA-LUAD")
@click.option("--out", required=True, type=click.Path(), help="manifest TSV")
def main(project: str, out: str) -> None:
    hits = query(project)
    rows = hits_to_manifest(hits, project=project)
    df = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(f"wrote {len(df)} rows to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
