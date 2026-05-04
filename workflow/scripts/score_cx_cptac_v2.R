#!/usr/bin/env Rscript
# Recompute Drews 2022 CX1-CX17 activities for CPTAC patients via the
# CINSignatureQuantification Bioconductor package on PDC ABSOLUTE
# copy-number calls.
#
# This unblocks the H2 external-validation test |Delta rho(TCGA - CPTAC)|.
#
# Inputs:
#   - PDC API: ABSOLUTE seg files for CPTAC patients (~7300 series across
#     13 cohorts). Pull CN profile per patient.
#   - Genome build: hg19 / GRCh37 (matches the original Drews 2022 work).
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

  message(sprintf("[%s] running CINSignatureQuantification on n=%d samples...",
                  cohort, length(unique(cnv$sample))))
  qcs <- quantifyCNSignatures(as.data.frame(cnv),
                              experimentName = paste0("CPTAC-", cohort),
                              method = "drews",
                              cores = 4)
  acts <- as.data.frame(qcs@activities$thresholdAct1)
  acts$cohort <- cohort
  acts$bcr_patient_barcode <- rownames(acts)
  all_results[[cohort]] <- acts
  message(sprintf("[%s] done: %d patients x %d signatures",
                  cohort, nrow(acts), ncol(qcs@activities$thresholdAct1)))
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
