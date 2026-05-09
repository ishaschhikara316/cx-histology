#!/usr/bin/env Rscript
# Recompute Drews 2022 CX1-CX17 activities for CPTAC patients via the
# CINSignatureQuantification Bioconductor package on PDC ABSOLUTE
# copy-number calls.
#
# This unblocks the H2 external-validation test |Delta rho(TCGA - CPTAC)|.
#
# Inputs:
#   - GDC API: per-patient ASCAT2 Allele-specific Copy Number Segment
#     files for CPTAC-3 (1,583 cases across 12 cohorts; OV is absent
#     from GDC). Staged by workflow/scripts/fetch_cptac_gdc_seg.py.
#   - Genome build: hg38 / GRCh38 (GDC harmonised; package supports it).
#
# Output:
#   data/labels/cx_activities_cptac.tsv (patient_id x CX1..CX17)
#
# This is the CPTAC analog of cx_activities.tsv that lives for TCGA at
# data/labels/cx_activities.tsv. Once it exists, predict_cptac_inference.py
# can compute |Delta rho| as the H2 test.
#
# Usage:
#   Rscript workflow/scripts/score_cx_cptac_v2.R
#
# Requirements:
#   - R with BiocManager
#   - CINSignatureQuantification from Bioconductor
#   - PDC web API access (no auth needed for public ABSOLUTE data)

suppressPackageStartupMessages({
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cran.r-project.org")
  }
  for (pkg in c("CINSignatureQuantification", "GenomicRanges", "data.table")) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      BiocManager::install(pkg, update = FALSE, ask = FALSE)
    }
  }
  library(CINSignatureQuantification)
  library(GenomicRanges)
  library(data.table)
})

# CPTAC ABSOLUTE seg files via PDC. The endpoint used here is the public
# proteomic data commons pdc.cancer.gov. Each cohort has its own
# manifest of cases with copy-number calls; we iterate cohort by cohort.
#
# Note: as of writing, the easiest way to get ABSOLUTE seg for CPTAC is
# via PDC's `studies` -> `clinical` + `cnv` endpoints. The reference
# implementation that we adapt is:
#   https://github.com/markowetzlab/CINSignatureQuantification
# which provides quantifyCNSignatures() given a GRanges or seg-file
# data.frame with columns: sample, chromosome, start, end, segVal.

pdc_cohorts <- c("LUAD", "LSCC", "UCEC", "CCRCC", "BRCA", "PDA", "GBM",
                 "CM", "HNSCC", "COAD", "SAR", "OV", "AML")

read_pdc_cnv <- function(cohort) {
  # Expected file: data/labels/cptac_pdc/{cohort}_absolute_seg.tsv
  # If you have already curated these, place them there. Otherwise
  # this function returns NULL and we skip the cohort.
  f <- file.path("data/labels/cptac_pdc", paste0(cohort, "_absolute_seg.tsv"))
  if (!file.exists(f)) {
    message(sprintf("  [%s] no seg file at %s -- skip", cohort, f))
    return(NULL)
  }
  dt <- fread(f)
  # Normalize column names
  setnames(dt, tolower(names(dt)))
  needed <- c("sample", "chromosome", "start", "end", "segval")
  if (!all(needed %in% names(dt))) {
    stop(sprintf("seg file %s missing required columns: %s", f,
                 paste(setdiff(needed, names(dt)), collapse = ", ")))
  }
  dt[, .(sample = sample, chromosome = as.character(chromosome),
         start = as.integer(start), end = as.integer(end),
         segVal = as.numeric(segval))]
}

dir.create("data/labels", showWarnings = FALSE, recursive = TRUE)
dir.create("data/labels/cptac_pdc", showWarnings = FALSE, recursive = TRUE)

all_results <- list()
for (cohort in pdc_cohorts) {
  message(sprintf("[%s] reading copy-number...", cohort))
  cnv <- read_pdc_cnv(cohort)
  if (is.null(cnv) || nrow(cnv) == 0) next

  n_samples <- length(unique(cnv$sample))
  message(sprintf("[%s] running CINSignatureQuantification on n=%d samples...",
                  cohort, n_samples))

  # Some CPTAC cohorts have very few patients with copy-number on GDC
  # (e.g. COAD n=1, BRCA n=3). quantifyCNSignatures fits its own
  # mixture parameters and can fail or produce degenerate output at
  # tiny n. Wrap in tryCatch so one bad cohort does not halt the run.
  # quantifyCNSignatures wraps four steps but does not expose the DCIN
  # (drews-CIN-segment) filter. GDC ASCAT2 outputs include diploid
  # (CN=2) regions as segments, and after smooth-diploid merging many
  # near-diploid CPTAC tumors fall below the default DCIN=20 threshold
  # and get dropped. We unroll the pipeline so we can pass DCIN=10,
  # which keeps quieter tumors in the analysis without losing the
  # high-CIN signal we care about for the H2 test.
  qcs <- tryCatch({
    cnq <- createCNQuant(data = as.data.frame(cnv),
                         experimentName = paste0("CPTAC-", cohort),
                         build = "hg38")
    cnq <- calculateFeatures(cnq, method = "drews", cores = 4, DCIN = 10)
    cnq <- calculateSampleByComponentMatrix(cnq, method = "drews")
    cnq <- calculateActivity(cnq, method = "drews", cancer.subset = NULL)
    cnq
  }, error = function(e) {
    message(sprintf("  [%s] CIN pipeline failed: %s",
                    cohort, conditionMessage(e)))
    NULL
  })
  if (is.null(qcs)) next

  acts <- as.data.frame(qcs@activities$thresholdAct2)
  if (nrow(acts) == 0) {
    message(sprintf("  [%s] no activities returned; skip", cohort))
    next
  }
  acts$cohort <- cohort
  acts$bcr_patient_barcode <- rownames(acts)
  all_results[[cohort]] <- acts
  message(sprintf("[%s] done: %d patients x %d signatures",
                  cohort, nrow(acts), ncol(qcs@activities$thresholdAct2)))
}

if (length(all_results) == 0) {
  stop("No CPTAC seg files found. Place ABSOLUTE seg per cohort under ",
       "data/labels/cptac_pdc/{cohort}_absolute_seg.tsv before running.")
}

merged <- rbindlist(lapply(all_results, as.data.table), fill = TRUE)
out <- "data/labels/cx_activities_cptac.tsv"
fwrite(merged, out, sep = "\t")
cat(sprintf("\nwrote %s with %d patients across %d cohorts\n",
            out, nrow(merged), length(all_results)))
