#!/bin/bash
# 4-slot ingest pool pinned to physical GPUs 1,2,3,4. Sourced by
# resume_max_parallel.sh. Each slot runs ONE cohort at a time, longest
# first; refills from a shared queue when free.

set -u
ROOT="/root/ISHA/cx-histology"
cd "$ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_TOKEN="${HF_TOKEN:-$(cat /root/.hf_token 2>/dev/null || echo)}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export TOKENIZERS_PARALLELISM=false
: "${OMP_NUM_THREADS:=6}"
: "${MKL_NUM_THREADS:=6}"
export OMP_NUM_THREADS MKL_NUM_THREADS

mkdir -p logs
LOG="logs/ingest_pool.log"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

# Physical GPU ids this pool may use
GPUS=(1 2 3 4)

# Queue: kind:cohort:size_hint, sorted descending. Phase 2 cohorts only.
TCGA=(
  "tcga:BRCA:1500"
  "tcga:LGG:844"
  "tcga:GBM:860"
  "tcga:LUAD:541"
  "tcga:LUSC:512"
  "tcga:HNSC:472"
)
CPTAC=(
  "cptac:LUAD:1137"
  "cptac:LSCC:1081"
  "cptac:UCEC:883"
  "cptac:CCRCC:782"
  "cptac:BRCA:653"
  "cptac:PDA:557"
  "cptac:GBM:462"
  "cptac:CM:411"
  "cptac:HNSCC:390"
  "cptac:COAD:372"
  "cptac:SAR:305"
  "cptac:OV:222"
  "cptac:AML:122"
)
QUEUE=$(printf '%s\n' "${TCGA[@]}" "${CPTAC[@]}" | sort -t: -k3 -nr)
log "queue (largest first):"
echo "$QUEUE" | sed 's/^/  /' | tee -a "$LOG"

declare -A SLOT_PID SLOT_TAG SLOT_GPU

launch_tcga() {
  local slot=$1 gpu=$2 cohort=$3
  local marker="data/interim/embeddings/tcga/uni/${cohort}/.done"
  if [ -f "$marker" ]; then
    log "  [slot$slot/gpu$gpu] skip TCGA $cohort (.done present)"
    SLOT_PID[$slot]=""; return
  fi
  if [ ! -f "data/interim/manifests/tcga_${cohort}.tsv" ]; then
    log "  [slot$slot/gpu$gpu] no TCGA manifest tcga_${cohort}.tsv — skip"
    SLOT_PID[$slot]=""; return
  fi
  log "  [slot$slot/gpu$gpu] start TCGA $cohort"
  CUDA_VISIBLE_DEVICES=$gpu \
    python workflow/scripts/stream_ingest_parallel.py \
      --manifest "data/interim/manifests/tcga_${cohort}.tsv" \
      --raw-dir "data/raw/tcga/${cohort}" \
      --tiles-dir "data/interim/tiles/tcga/${cohort}" \
      --embed-dir "data/interim/embeddings/tcga/uni/${cohort}" \
      --embed-encoder uni \
      --stain-reference resources/stain_reference.png \
      --n-workers 1 \
      --gdc-processes 8 \
      --embed-batch-size 64 \
      --marker "$marker" \
      > "logs/p2_TCGA_${cohort}.log" 2>&1 &
  SLOT_PID[$slot]=$!
  SLOT_TAG[$slot]="tcga:$cohort"
  SLOT_GPU[$slot]=$gpu
}

launch_cptac() {
  local slot=$1 gpu=$2 cohort=$3
  local marker="data/interim/embeddings/cptac/uni/${cohort}/.done"
  if [ -f "$marker" ]; then
    log "  [slot$slot/gpu$gpu] skip CPTAC $cohort (.done present)"
    SLOT_PID[$slot]=""; return
  fi
  local manifest="data/interim/manifests/cptac_idc_${cohort}.tsv"
  if [ ! -f "$manifest" ]; then
    log "  [slot$slot/gpu$gpu] CPTAC $cohort: no manifest, skip"
    SLOT_PID[$slot]=""; return
  fi
  mkdir -p "data/interim/embeddings/cptac/uni/${cohort}"
  log "  [slot$slot/gpu$gpu] start CPTAC $cohort"
  ( CUDA_VISIBLE_DEVICES=$gpu \
    python workflow/scripts/stream_ingest_cptac.py \
      --manifest "$manifest" \
      --raw-dir "data/raw/cptac/${cohort}" \
      --tiles-dir "data/interim/tiles/cptac/${cohort}" \
      --embed-dir "data/interim/embeddings/cptac/uni/${cohort}" \
      --embed-encoder uni \
      --embed-device cuda:0 \
      --embed-batch-size 64 \
      > "logs/p2_CPTAC_${cohort}.log" 2>&1
    rc=$?; [ $rc -eq 0 ] && touch "$marker"; exit $rc ) &
  SLOT_PID[$slot]=$!
  SLOT_TAG[$slot]="cptac:$cohort"
  SLOT_GPU[$slot]=$gpu
}

dispatch_next() {
  local slot=$1
  local gpu=${GPUS[$slot]}
  local next=$(echo "$QUEUE" | head -1)
  QUEUE=$(echo "$QUEUE" | tail -n +2)
  if [ -z "$next" ]; then SLOT_PID[$slot]=""; return; fi
  IFS=':' read -r kind cohort _ <<< "$next"
  if [ "$kind" = "tcga" ]; then
    launch_tcga "$slot" "$gpu" "$cohort"
  else
    launch_cptac "$slot" "$gpu" "$cohort"
  fi
  # If the launcher fast-skipped (already .done or no manifest), recurse.
  if [ -z "${SLOT_PID[$slot]:-}" ] && [ -n "$(echo "$QUEUE" | head -1)" ]; then
    dispatch_next "$slot"
  fi
}

for s in 0 1 2 3; do dispatch_next "$s"; done

while true; do
  any=0
  for s in 0 1 2 3; do
    pid=${SLOT_PID[$s]:-}
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      any=1; continue
    fi
    if [ -n "$pid" ]; then
      wait "$pid" 2>/dev/null
      log "  [slot$s/gpu${SLOT_GPU[$s]:-?}] done ${SLOT_TAG[$s]:-?}"
      SLOT_PID[$s]=""
    fi
    if [ -n "$(echo "$QUEUE" | head -1)" ]; then
      dispatch_next "$s"
      any=1
    fi
  done
  [ $any -eq 0 ] && break
  sleep 30
done

log "ingest pool drained"
