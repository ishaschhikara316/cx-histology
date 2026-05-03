# CPTAC pathology ingest: blockers and revised plan

This is a working note from a 2026-05-03 pre-debug pass on the
`ingest_cptac.py` path. The TLDR is that the existing script does not
work and the TCIA route assumed in the design spec is the wrong route.

## What the existing script tries to do

`workflow/scripts/ingest_cptac.py` queries the TCIA REST API v4
(`services.cancerimagingarchive.net/services/v4/TCIA/query`) for series
in a collection (e.g. `CPTAC-BRCA`) with `Modality=SM` (Slide
Microscopy), then downloads each series with `getImage`. The output is
a `_manifest.tsv` of series UIDs and zip paths.

## Why it fails

1. **TCIA v4 endpoint is unreachable** in this checkpoint environment.
   `getSeries` on the v4 path times out at 30 s. The current TCIA
   service responds at `/nbia-api/services/v1/...` and `v2/...` instead.
2. **TCIA does not host CPTAC pathology slides.** I queried
   `getCollectionValues` and confirmed only nine CPTAC collections
   exist on TCIA: AML, CCRCC, CM, LSCC, LUAD, PDA, SAR, STAD, UCEC.
   For every one of them, `getModalityValues` returns CT, MR, PT,
   RTSTRUCT, US — radiology only. `getSeries?Modality=SM` returns
   zero series for all nine collections.
3. **The cohort list in `config/cohorts.yaml` is wrong.** It assumes
   `CPTAC-BRCA`, `CPTAC-LUAD`, `CPTAC-OV`, `CPTAC-COAD`, `CPTAC-GBM`.
   None of those four except CPTAC-LUAD even exist as TCIA
   collections, and none have SM modality.

## Where CPTAC pathology actually lives

Two viable sources:

### Option A: NCI Imaging Data Commons (IDC)

IDC hosts CPTAC pathology under collections like `cptac_lscc`,
`cptac_luad`, `cptac_ccrcc`, `cptac_pda`, `cptac_ucec`, etc. Slides
are served via Google Cloud Storage, BigQuery-indexed metadata, and
DICOM-WSI format. A small `idc-index` Python package gives a
pandas-friendly query layer; alternatively the `s5cmd` or `gsutil`
tools can pull slides directly from `gs://public-datasets-idc/...`.

Pros: stable, standardized, already DICOM-WSI converted.
Cons: DICOM-WSI not SVS, so the existing `tile_wsi.py` may need a
small adapter (openslide does support DICOM via `OpenSlide.open` on
recent libopenslide versions).

### Option B: PDC (Proteomic Data Commons) → Cancer Data Service

PDC has a GraphQL endpoint at `pdc.cancer.gov/graphql` and links study
manifests to image files hosted at CDS / dbGaP-controlled CDS buckets.
Some image files require dbGaP authorisation (controlled access).
For our purposes, the open-access pathology subset of CPTAC is what we
need; that subset is on IDC.

## Recommended path forward

Replace `ingest_cptac.py` with an IDC-based downloader. The minimum
viable script needs to:

1. Use the `idc-index` Python package (pip-installable) to enumerate
   `cptac_<cohort>` collection slides with `series_size_MB` filtered to
   diagnostic resolution.
2. For each series, resolve the GCS URL list and pull the DICOM-WSI
   files with `s5cmd` or HTTP. Keep stream-mode semantics: download
   one slide, tile-and-embed, delete.
3. Recompute CX1 to CX17 activities locally from CPTAC ABSOLUTE copy
   number calls using the R `CINSignatureQuantification` package (the
   design spec had this right; only the slide-pull step is broken).
4. Output `data/interim/embeddings/cptac/uni/<cohort>/.done` to mirror
   the TCGA layout so the same training pipeline works.

For target cohorts: pick CPTAC pathology collections that overlap
TCGA at the disease level so we can sanity-check H2 (|Δρ| ≤ 0.15).
Best matches from IDC's CPTAC offerings:

- LUAD ↔ TCGA-LUAD
- LSCC ↔ TCGA-LUSC
- CCRCC ↔ TCGA-KIRC
- UCEC ↔ TCGA-UCEC
- PDA ↔ TCGA-PAAD
- STAD ↔ TCGA-STAD

That gives six pairs without needing the original BRCA/OV/GBM/COAD
list. The first three (LUAD, LSCC, CCRCC) have well-known H&E-AI
literature for benchmarking.

## Effort estimate

- Install `idc-index`, write the IDC enumeration: 1 hr
- Adapt `stream_ingest.py` to handle DICOM-WSI (openslide already
  speaks DICOM in recent libopenslide builds, so this may be free): 0–2 hr
- CX recomputation in R + bioconductor: 1–2 hr
- Smoke-test on CPTAC-LUAD (the smallest practical target): 1–2 hr

Total: roughly half a day of focused work, after which a four-cohort
external validation run is feasible. The current TCGA pan-cancer
ingest in flight tonight does not depend on any of this.
