#!/bin/bash
# Parallel-cohort ingest: two cohorts in flight at once on 2 GPUs each.
# Total: 8 cohorts, ~2200 GB raw, streams 32 concurrent GDC connections.
#
# Group A (GPUs 0,1): ESCA, PAAD, COAD, KIRP  (~1022 GB)
# Group B (GPUs 2,3): CESC, THYM, PCPG, STAD  (~1177 GB)
#
# Each cohort uses n-workers=2 (one per GPU in its group) and gdc-processes=8.

ROOT="/root/Isha/cx-histology"
cd "$ROOT"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_TOKEN=$(cat /root/.hf_token)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

mkdir -p logs

log() { echo "[$(date -Iseconds)] $*" | tee -a logs/parallel_ingest.log; }

run_group() {
  local group=$1; shift
  local gpus=$1; shift
  local cohorts=("$@")
  for c in "${cohorts[@]}"; do
    if [ -f "data/interim/embeddings/tcga/uni/$c/.done" ]; then
      log "[$group] $c already done — skipping"
      continue
    fi
    log "[$group] starting $c on GPUs=$gpus"
    t0=$(date +%s)
    CUDA_VISIBLE_DEVICES="$gpus" \
      python workflow/scripts/stream_ingest_parallel.py \
        --manifest "data/interim/manifests/tcga_${c}.tsv" \
        --raw-dir "data/raw/tcga/${c}" \
        --tiles-dir "data/interim/tiles/tcga/${c}" \
        --embed-dir "data/interim/embeddings/tcga/uni/${c}" \
        --embed-encoder uni \
        --stain-reference resources/stain_reference.png \
        --n-workers 2 \
        --gdc-processes 8 \
        --embed-batch-size 64 \
        --marker "data/interim/embeddings/tcga/uni/${c}/.done" \
        > "logs/parallel_${group}_${c}.log" 2>&1
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    if [ $rc -eq 0 ]; then
      log "[$group] $c DONE in $((dt / 60))m $((dt % 60))s"
    else
      log "[$group] $c FAILED rc=$rc in $((dt / 60))m $((dt % 60))s — continuing"
    fi
  done
  log "[$group] all cohorts processed"
}

log "=== parallel ingest: 2 cohort groups across 4 GPUs ==="

run_group A "0,1" ESCA PAAD COAD KIRP &
PID_A=$!
run_group B "2,3" CESC THYM PCPG STAD &
PID_B=$!

wait $PID_A
log "group A finished"
wait $PID_B
log "group B finished"

log "=== ingest complete ==="
