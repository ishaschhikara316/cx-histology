"""Download CPTAC Allele-specific Copy Number Segment files from GDC and
stage them as per-cohort seg TSVs that score_cx_cptac_v2.R can consume.

The original R script comment named PDC as the source, but PDC hosts
proteomics only and does not distribute copy-number segment data. The
canonical open-access source for harmonised CPTAC copy-number is the
Genomic Data Commons (GDC). GDC's ASCAT2 pipeline produces
"Allele-specific Copy Number Segment" files for 1,744 CPTAC-3 cases
across the cohorts we use. ABSOLUTE itself was a TCGA-specific tool;
CINSignatureQuantification only needs (sample, chromosome, start, end,
segVal), so any integer-rounded total-copy-number seg file works.

Output layout matches what score_cx_cptac_v2.R reads:
    data/labels/cptac_pdc/{COHORT}_absolute_seg.tsv

Schema of each output file:
    sample  chromosome  start  end  segVal

where `sample` is the CPTAC submitter_id (e.g. C3N-00388), `chromosome`
is "1", "2", ..., "X", "Y", and `segVal` is the integer total copy
number from ASCAT2 (Major_Copy_Number + Minor_Copy_Number).

Usage:
    python workflow/scripts/fetch_cptac_gdc_seg.py \\
        --out-dir data/labels/cptac_pdc \\
        [--cohorts CCRCC,LUAD,...]   # default: all 13
        [--workers 8]                # parallel downloads
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

GDC_FILES = "https://api.gdc.cancer.gov/files"
GDC_DATA = "https://api.gdc.cancer.gov/data"

# GDC tags CPTAC cases by primary_site + disease_type. We map those
# pairs to the cohort codes used in the cx-histology manifests and the
# R script's pdc_cohorts list. Lung is the only ambiguous primary_site:
# adenocarcinomas -> LUAD, squamous-cell neoplasms -> LSCC.
ALL_COHORTS = ["CCRCC", "LUAD", "LSCC", "UCEC", "GBM", "PDA", "AML",
               "HNSCC", "BRCA", "COAD", "CM", "SAR", "OV"]

PRIMARY_SITE_TO_COHORT = {
    "kidney": "CCRCC",
    "uterus, nos": "UCEC",
    "brain": "GBM",
    "pancreas": "PDA",
    "hematopoietic and reticuloendothelial systems": "AML",
    "breast": "BRCA",
    "colon": "COAD",
    "ovary": "OV",
}

# For head-and-neck, several primary_site labels all map to HNSCC.
HNSCC_SITES = {
    "larynx", "other and unspecified parts of tongue", "floor of mouth",
    "other and ill-defined sites in lip, oral cavity and pharynx", "lip",
    "oropharynx", "gum", "other and unspecified parts of mouth",
    "base of tongue", "tonsil", "hypopharynx", "nasopharynx",
    "palate", "tonsillar fossa",
}

# Sarcoma and melanoma primary_sites we accept; CPTAC SAR / CM cohorts.
SAR_DISEASE_TYPES = {"lipomatous neoplasms",
                     "complex mixed and stromal neoplasms",
                     "soft tissue tumors and sarcomas, nos",
                     "fibromatous neoplasms",
                     "synovial-like neoplasms"}
CM_DISEASE_TYPES = {"nevi and melanomas"}


def cohort_for(primary_site: str, disease_type: str) -> str | None:
    ps = (primary_site or "").lower()
    dt = (disease_type or "").lower()
    if ps == "bronchus and lung":
        if "squamous" in dt:
            return "LSCC"
        if "adenocarc" in dt or "adenomas" in dt:
            return "LUAD"
        return None
    if dt in CM_DISEASE_TYPES:
        return "CM"
    if dt in SAR_DISEASE_TYPES:
        return "SAR"
    if ps in HNSCC_SITES:
        return "HNSCC"
    return PRIMARY_SITE_TO_COHORT.get(ps)


def list_seg_files(session: requests.Session) -> list[dict]:
    """Page through GDC and return ASCAT seg file metadata + case info."""
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {
                "field": "cases.project.program.name", "value": ["CPTAC"]}},
            {"op": "in", "content": {
                "field": "data_type",
                "value": ["Allele-specific Copy Number Segment"]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
        ],
    }
    fields = ",".join([
        "file_id", "file_name", "data_type",
        "cases.submitter_id",
        "cases.primary_site",
        "cases.disease_type",
        "cases.project.project_id",
    ])
    out: list[dict] = []
    page_size = 500
    frm = 0
    while True:
        params = {
            "filters": json.dumps(filters),
            "fields": fields,
            "size": page_size,
            "from": frm,
            "format": "json",
        }
        r = session.get(GDC_FILES, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()["data"]
        hits = data.get("hits", [])
        out.extend(hits)
        total = data.get("pagination", {}).get("total", len(out))
        frm += page_size
        if frm >= total:
            break
    return out


def download_seg(session: requests.Session, file_id: str,
                 retries: int = 3) -> pd.DataFrame:
    """Pull one seg file by file_id and return as DataFrame."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = session.get(f"{GDC_DATA}/{file_id}", timeout=120)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text), sep="\t")
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"download failed for {file_id}: {last_err}")


def normalise_seg(df: pd.DataFrame, sample: str) -> pd.DataFrame:
    # GDC ASCAT2 columns: GDC_Aliquot, Chromosome, Start, End,
    # Copy_Number, Major_Copy_Number, Minor_Copy_Number.
    cols = {c.lower(): c for c in df.columns}
    chrom = df[cols["chromosome"]].astype(str).str.replace("^chr", "",
                                                            regex=True)
    start = df[cols["start"]].astype(int)
    end = df[cols["end"]].astype(int)
    if "copy_number" in cols:
        seg = df[cols["copy_number"]].astype(float).round().astype(int)
    else:
        major = df[cols["major_copy_number"]].astype(float)
        minor = df[cols["minor_copy_number"]].astype(float)
        seg = (major + minor).round().astype(int)
    return pd.DataFrame({
        "sample": sample, "chromosome": chrom,
        "start": start, "end": end, "segVal": seg,
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/labels/cptac_pdc")
    ap.add_argument("--cohorts", default=",".join(ALL_COHORTS),
                    help="comma-separated cohort codes to keep")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="optional cap on files per cohort, for smoke tests")
    args = ap.parse_args()

    keep = [c.strip() for c in args.cohorts.split(",") if c.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sess = requests.Session()
    sess.headers["User-Agent"] = "cx-histology/1.0 (CPTAC-CN fetch)"

    print("listing CPTAC ASCAT seg files on GDC...", flush=True)
    files = list_seg_files(sess)
    print(f"  found {len(files)} files", flush=True)

    by_cohort: dict[str, list[tuple[str, str]]] = {}
    skipped = 0
    for h in files:
        cases = h.get("cases") or []
        if not cases:
            skipped += 1; continue
        c = cases[0]
        cohort = cohort_for(c.get("primary_site"), c.get("disease_type"))
        if cohort is None or cohort not in keep:
            skipped += 1; continue
        sample = c.get("submitter_id")
        if not sample:
            skipped += 1; continue
        by_cohort.setdefault(cohort, []).append((h["file_id"], sample))

    print("plan:", flush=True)
    for cohort in sorted(by_cohort):
        n = len(by_cohort[cohort])
        if args.limit:
            n = min(n, args.limit)
        print(f"  {cohort}: {n} files", flush=True)
    print(f"  skipped (out-of-scope or missing case info): {skipped}",
          flush=True)

    total_ok = 0
    total_fail = 0
    for cohort, items in sorted(by_cohort.items()):
        if args.limit:
            items = items[:args.limit]
        print(f"\n[{cohort}] downloading {len(items)} seg files...",
              flush=True)
        rows: list[pd.DataFrame] = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            fut_to_sample = {
                ex.submit(download_seg, sess, fid): (fid, sample)
                for fid, sample in items
            }
            for i, fut in enumerate(as_completed(fut_to_sample), 1):
                fid, sample = fut_to_sample[fut]
                try:
                    df = fut.result()
                    rows.append(normalise_seg(df, sample))
                    total_ok += 1
                except Exception as e:
                    total_fail += 1
                    print(f"  [skip] {sample} ({fid}): {e}", flush=True)
                if i % 25 == 0:
                    print(f"  {cohort}: {i}/{len(items)} done", flush=True)
        if not rows:
            print(f"  [{cohort}] no usable rows; skipping write",
                  flush=True)
            continue
        out_df = pd.concat(rows, ignore_index=True)
        # Drop chr "Y" segments where copy-number is meaningless for
        # CIN-signature scoring, but keep "X". CINSignatureQuantification
        # tolerates either.
        out_df = out_df[out_df["chromosome"].isin(
            [str(i) for i in range(1, 23)] + ["X"])]
        out_path = out_dir / f"{cohort}_absolute_seg.tsv"
        out_df.to_csv(out_path, sep="\t", index=False)
        print(f"  wrote {out_path}: "
              f"{out_df['sample'].nunique()} patients, "
              f"{len(out_df)} segments", flush=True)

    print(f"\nDONE. ok={total_ok} fail={total_fail}", flush=True)
    return 0 if total_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
