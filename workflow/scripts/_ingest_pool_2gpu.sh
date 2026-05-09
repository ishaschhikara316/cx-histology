#!/bin/bash
# 2-GPU ingest orchestrator for the 6 partial TCGA cohorts.
#
# Runs cohorts sequentially (one at a time), with each cohort using
# both GPUs via stream_ingest_parallel.py n-workers=2. Smallest-
# remaining first so we validate the pipeline early and free disk
# slots progressively.
#
# Usage: bash workflow/scripts/_ingest_pool_2gpu.sh

ROOT="/root/ISHA/cx-histology"
cd "$ROOT"

# Activate conda env (cx-histology) and force its libstdc++ ahead of system.
# set -u must come AFTER `conda activate` because conda's MKL activation
# scripts touch unset variables.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate cx-histology
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
set -u

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || echo)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p logs
LOG="logs/ingest_pool_2gpu.log"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

# Both GPUs visible to every cohort; stream_ingest_parallel will
# round-robin worker k -> GPU (k % 2).
export CUDA_VISIBLE_DEVICES="0,1"

# Smallest-remaining-first (resume_state.md numbers): HNSC, LUSC, LUAD,
# GBM, LGG, BRCA. The original full manifest is fed in; stream_ingest
# auto-skips slides whose .h5 already exists.
COHORTS=(HNSC LUSC LUAD GBM LGG BRCA)

run_cohort() {
  local cohort=$1
  local marker="data/interim/embeddings/tcga/uni/${cohort}/.done"
  local manifest="data/interim/manifests/tcga_${cohort}.tsv"

  if [ -f "$marker" ]; then
    log "  $cohort: .done present, skipping"
    return 0
  fi
  if [ ! -f "$manifest" ]; then
    log "  $cohort: missing manifest $manifest, skipping"
    return 1
  fi

  log "=== $cohort: starting (manifest=$manifest) ==="
  local before=$(ls "data/interim/embeddings/tcga/uni/${cohort}/"*.h5 2>/dev/null | wc -l)
  log "  $cohort: $before .h5 files before"

  python workflow/scripts/stream_ingest_parallel.py \
    --manifest "$manifest" \
    --raw-dir "data/raw/tcga/${cohort}" \
    --tiles-dir "data/interim/tiles/tcga/${cohort}" \
    --embed-dir "data/interim/embeddings/tcga/uni/${cohort}" \
    --embed-encoder uni \
    --stain-reference resources/stain_reference.png \
    --n-workers 2 \
    --gdc-processes 8 \
    --embed-batch-size 32 \
    --marker "$marker" \
    > "logs/ingest_2gpu_${cohort}.log" 2>&1
  local rc=$?

  local after=$(ls "data/interim/embeddings/tcga/uni/${cohort}/"*.h5 2>/dev/null | wc -l)
  log "  $cohort: $after .h5 files after (rc=$rc, +$((after-before)) new)"

  # Free intermediate dirs to keep disk pressure low
  rm -rf "data/raw/tcga/${cohort}" "data/interim/tiles/tcga/${cohort}" 2>/dev/null

  return $rc
}

log "=== 2-GPU ingest pool start: ${COHORTS[*]} ==="
df -h /root | tail -1 | tee -a "$LOG"

fails=()
for c in "${COHORTS[@]}"; do
  if ! run_cohort "$c"; then
    log "  $c: FAILED — continuing with remaining cohorts"
    fails+=("$c")
  fi
  df -h /root | tail -1 | tee -a "$LOG"
done

if [ ${#fails[@]} -gt 0 ]; then
  log "=== ingest pool DONE with failures: ${fails[*]} ==="
  exit 1
else
  log "=== ingest pool DONE — all 6 cohorts complete ==="
fi
