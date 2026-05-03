"""Build IDC manifests for all CPTAC pathology cohorts.

Writes one TSV per cohort to data/interim/manifests/cptac_idc_<COHORT>.tsv
with columns: patient_id, series_uid, study_uid, sop_class, n_files, urls.

These manifests are read offline by stream_ingest_cptac.py during the
unattended max-parallel run. Generating them up front avoids redundant
IDC API calls during ingest.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from idc_index import IDCClient

ALL_COHORTS = ["luad", "lscc", "ucec", "ccrcc", "brca", "pda", "gbm",
               "cm", "hnscc", "coad", "sar", "ov", "aml"]


def _resolve_urls(client, uid):
    try:
        return uid, client.get_series_file_URLs(uid)
    except Exception:
        return uid, []


def main():
    c = IDCClient()
    df_idx = c.index
    out_dir = Path("data/interim/manifests")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for cohort in ALL_COHORTS:
        out = out_dir / f"cptac_idc_{cohort.upper()}.tsv"
        if out.exists() and out.stat().st_size > 100:
            print(f"[{cohort.upper()}] skip: {out} already exists")
            df_existing = pd.read_csv(out, sep="\t")
            summary.append((cohort.upper(), len(df_existing)))
            continue

        sub = df_idx[(df_idx["collection_id"] == f"cptac_{cohort}") &
                     (df_idx["Modality"] == "SM")]
        if len(sub) == 0:
            print(f"[{cohort}] 0 series, skipping")
            continue

        # Parallel URL resolution: 32-thread pool keeps the local IDC
        # SQLite happy and lets us finish a 1000-series cohort in ~30 sec.
        url_map = {}
        with ThreadPoolExecutor(max_workers=32) as pool:
            futures = [pool.submit(_resolve_urls, c, uid)
                       for uid in sub["SeriesInstanceUID"]]
            for f in as_completed(futures):
                uid, urls = f.result()
                url_map[uid] = urls

        rows = []
        for _, r in sub.iterrows():
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
        pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
        print(f"[{cohort.upper()}] wrote {out}: "
              f"{len(rows)} series, {sum(r['n_files'] for r in rows)} files",
              flush=True)
        summary.append((cohort.upper(), len(rows)))

    print("\nSummary:")
    for c_, n in summary:
        print(f"  {c_}: {n}")
    print(f"  TOTAL: {sum(n for _, n in summary)} series across "
          f"{len(summary)} cohorts")


if __name__ == "__main__":
    main()
