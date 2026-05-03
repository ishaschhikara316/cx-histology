"""Build IDC manifests for all CPTAC pathology cohorts.

Writes one TSV per cohort to data/interim/manifests/cptac_idc_<COHORT>.tsv
with columns: patient_id, series_uid, study_uid, sop_class, n_files, urls.

These manifests are read offline by stream_ingest_cptac.py during the
unattended max-parallel run. Generating them up front avoids redundant
IDC API calls during ingest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from idc_index import IDCClient

ALL_COHORTS = ["luad", "lscc", "ucec", "ccrcc", "brca", "pda", "gbm",
               "cm", "hnscc", "coad", "sar", "ov", "aml"]


def main():
    c = IDCClient()
    df_idx = c.index
    out_dir = Path("data/interim/manifests")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for cohort in ALL_COHORTS:
        sub = df_idx[(df_idx["collection_id"] == f"cptac_{cohort}") &
                     (df_idx["Modality"] == "SM")]
        if len(sub) == 0:
            print(f"[{cohort}] 0 series, skipping")
            continue
        rows = []
        for _, r in sub.iterrows():
            uid = r["SeriesInstanceUID"]
            try:
                urls = c.get_series_file_URLs(uid)
            except Exception as e:
                print(f"  WARN: {uid}: {e}", file=sys.stderr)
                urls = []
            rows.append({
                "patient_id": r["PatientID"],
                "series_uid": uid,
                "study_uid": r["StudyInstanceUID"],
                "sop_class": r.get("sop_class_name", ""),
                "n_files": len(urls),
                "urls": ";".join(urls),
            })
        out = out_dir / f"cptac_idc_{cohort.upper()}.tsv"
        pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
        print(f"[{cohort.upper()}] wrote {out}: "
              f"{len(rows)} series, {sum(r['n_files'] for r in rows)} files")
        summary.append((cohort.upper(), len(rows)))

    print("\nSummary:")
    for c_, n in summary:
        print(f"  {c_}: {n}")
    print(f"  TOTAL: {sum(n for _, n in summary)} series across "
          f"{len(summary)} cohorts")


if __name__ == "__main__":
    main()
