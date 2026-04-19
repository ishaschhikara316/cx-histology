"""Download WSI files referenced in a manifest via gdc-client.

gdc-client must be on PATH (installed separately; not in the conda env
because its distribution on arm/linux is binary).

Usage:
    python download_wsi.py --manifest data/manifests/tcga_luad.tsv \
        --out-dir data/raw/tcga/luad --token ~/.gdc_token
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click
import pandas as pd


def gdc_manifest_format(df: pd.DataFrame) -> str:
    """Emit the GDC-client expected manifest TSV format."""
    cols = ["id", "filename", "md5", "size", "state"]
    m = pd.DataFrame({
        "id":       df["file_id"],
        "filename": df["file_name"],
        "md5":      df["md5"],
        "size":     df["file_size"],
        "state":    "live",
    })[cols]
    return m.to_csv(sep="\t", index=False)


@click.command()
@click.option("--manifest", required=True, type=click.Path(exists=True))
@click.option("--out-dir", required=True, type=click.Path())
@click.option("--token", required=True, type=click.Path(exists=True))
@click.option("--processes", default=4, type=int)
def main(manifest: str, out_dir: str, token: str, processes: int) -> None:
    df = pd.read_csv(manifest, sep="\t")
    gdc_txt = gdc_manifest_format(df)
    gdc_mf = Path(out_dir) / "_gdc_manifest.txt"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    gdc_mf.write_text(gdc_txt)

    cmd = ["gdc-client", "download",
           "-m", str(gdc_mf),
           "-d", out_dir,
           "-t", token,
           "-n", str(processes)]
    print(" ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
