#!/usr/bin/env Rscript
# Compute CX1-CX17 activities for CPTAC patients.
#
# Input:  a TSV of ABSOLUTE segments with columns
#         sample, chromosome, start, end, segVal (absolute CN).
# Output: per-patient CX activity TSV, one row per sample.
suppressPackageStartupMessages({
  library(optparse); library(data.table)
})

opts <- parse_args(OptionParser(option_list = list(
  make_option("--segments", type="character"),
  make_option("--out",      type="character")
)))

if (!requireNamespace("CINSignatureQuantification", quietly = TRUE)) {
  remotes::install_github("markowetzlab/CINSignatureQuantification")
}
suppressPackageStartupMessages(library(CINSignatureQuantification))

seg <- fread(opts$segments)
setnames(seg, old = c("sample", "chromosome", "start", "end", "segVal"),
         new = c("sample", "chromosome", "start", "end", "segVal"))

out <- quantifyCNSignatures(seg, experimentName = "cptac",
                            cancer_type = "pan-cancer",
                            build = "hg38",
                            method = "drews")
act <- as.data.table(out@activities, keep.rownames = "patient")
setnames(act, "patient", "bcr_patient_barcode")
fwrite(act, opts$out, sep = "\t")
