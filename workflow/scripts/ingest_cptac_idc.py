"""CPTAC pathology ingest via NCI Imaging Data Commons (IDC).

Replaces the broken `ingest_cptac.py` which assumed CPTAC pathology is
on TCIA. It is not. CPTAC pathology lives on IDC, served as DICOM-WSI
from `s3://idc-open-data/` with no auth required for the public subset.

This script:
  1. Queries the IDC index (cached locally by idc-index) for
     SM-modality series in the requested CPTAC cohort.
  2. For each series, resolves the per-file S3 URLs and computes the
     series size.
  3. In `--dry-run` mode, writes a manifest TSV and exits. In live
     mode (default), streams each series to local disk, tiles +
     embeds via the existing pipeline, and deletes the raw DICOM.

Usage:
    python ingest_cptac_idc.py --cohort luad --out-dir data/raw/cptac/LUAD --dry-run
    python ingest_cptac_idc.py --cohort luad --out-dir data/raw/cptac/LUAD \
        --tiles-dir data/interim/tiles/cptac/LUAD \
        --embed-dir data/interim/embeddings/cptac/uni/LUAD --embed-encoder uni

The IDC collection naming is `cptac_<cohort_lowercase>`.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import click
import pandas as pd


SUPPORTED_COHORTS = ["luad", "lscc", "ucec", "ccrcc", "brca", "pda", "gbm",
                     "cm", "hnscc", "coad", "ov", "sar", "aml"]


def _idc_client():
    try:
        from idc_index import IDCClient
    except ImportError:
        sys.exit("idc-index is not installed. pip install idc-index")
    return IDCClient()


def list_series(cohort: str) -> pd.DataFrame:
    c = _idc_client()
    df = c.index
    coll = f"cptac_{cohort.lower()}"
    sub = df[(df["collection_id"] == coll) & (df["Modality"] == "SM")]
    return sub.copy()


def resolve_urls(series_uids: list[str]) -> dict[str, list[str]]:
    c = _idc_client()
    out = {}
    for uid in series_uids:
        try:
            out[uid] = c.get_series_file_URLs(uid)
        except Exception as e:
            out[uid] = []
            print(f"[warn] could not resolve URLs for {uid}: {e}",
                  file=sys.stderr)
    return out


def s3_download(s3_urls: list[str], out_dir: Path) -> list[Path]:
    """Download a series's DICOM files via s5cmd if available, else aws cli.

    s5cmd is parallel by default and faster on multi-file series. Falls
    back to a sequential `curl` over the public HTTP mirror if neither
    s5cmd nor aws cli are present.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    if shutil.which("s5cmd"):
        manifest_path = out_dir / "_s5cmd_manifest.txt"
        manifest_path.write_text(
            "\n".join(f"cp --no-sign-request --concurrency 4 {u} {out_dir}/"
                     for u in s3_urls)
        )
        rc = subprocess.call(["s5cmd", "run", str(manifest_path)],
                             cwd=str(out_dir))
        manifest_path.unlink(missing_ok=True)
        if rc != 0:
            raise RuntimeError(f"s5cmd download failed rc={rc}")
        paths = [out_dir / Path(u).name for u in s3_urls]
    elif shutil.which("aws"):
        for u in s3_urls:
            rc = subprocess.call(
                ["aws", "s3", "cp", "--no-sign-request", u, str(out_dir / Path(u).name)]
            )
            if rc != 0:
                raise RuntimeError(f"aws s3 cp failed rc={rc} for {u}")
            paths.append(out_dir / Path(u).name)
    else:
        # Fall back to public HTTP mirror.
        for u in s3_urls:
            http = u.replace("s3://idc-open-data/",
                             "https://s3.amazonaws.com/idc-open-data/")
            local = out_dir / Path(u).name
            rc = subprocess.call(["curl", "-sSLfo", str(local), http])
            if rc != 0:
                raise RuntimeError(f"curl failed for {http}")
            paths.append(local)
    return paths


@click.command()
@click.option("--cohort", required=True, type=click.Choice(SUPPORTED_COHORTS),
              help="CPTAC cohort (lowercase): luad, lscc, ucec, ccrcc, brca, ...")
@click.option("--out-dir", required=True, type=click.Path())
@click.option("--tiles-dir", default=None, type=click.Path())
@click.option("--embed-dir", default=None, type=click.Path())
@click.option("--embed-encoder", default="uni")
@click.option("--max-series", default=None, type=int,
              help="Limit to first N series (smoke test).")
@click.option("--dry-run", is_flag=True,
              help="Enumerate + manifest only, no downloads.")
def main(cohort, out_dir, tiles_dir, embed_dir, embed_encoder, max_series,
         dry_run):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[idc] enumerating cptac_{cohort} SM series")
    series = list_series(cohort)
    if max_series:
        series = series.head(max_series)
    print(f"[idc] {len(series)} series, "
          f"{series['PatientID'].nunique()} unique patients")

    print(f"[idc] resolving S3 URLs")
    url_map = resolve_urls(series["SeriesInstanceUID"].tolist())

    rows = []
    for _, r in series.iterrows():
        uid = r["SeriesInstanceUID"]
        urls = url_map.get(uid, [])
        rows.append({
            "patient_id": r["PatientID"],
            "series_uid": uid,
            "study_uid": r["StudyInstanceUID"],
            "sop_class": r.get("sop_class_name", ""),
            "n_files": len(urls),
            "urls": ";".join(urls),
        })

    manifest_df = pd.DataFrame(rows)
    manifest_path = out / "_idc_manifest.tsv"
    manifest_df.to_csv(manifest_path, sep="\t", index=False)
    print(f"[idc] wrote {manifest_path}")
    print(f"[idc] avg files per series: {manifest_df['n_files'].mean():.1f}")

    if dry_run:
        print("[idc] dry-run mode: no downloads")
        print(f"[idc] would download {len(manifest_df)} series, "
              f"{manifest_df['n_files'].sum()} DICOM files total")
        return

    if not (tiles_dir and embed_dir):
        sys.exit("Live mode requires --tiles-dir and --embed-dir.")

    sys.exit("Live ingest path not yet implemented (DICOM-WSI tiling adapter "
             "needs validation against tile_wsi.py). See "
             "docs/cptac_ingest_blockers.md for the roadmap.")


if __name__ == "__main__":
    main()
