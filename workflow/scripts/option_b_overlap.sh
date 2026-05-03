#!/bin/bash
# Option B: overlap CPTAC ingest on GPUs 1,2,3 during the 15-cohort retrain.
#
# Phase 1 finishes (~02:30 May 4). Auto-finalizer then runs the 15-cohort
# retrain on cuda:0 for ~30-45 min, leaving GPUs 1,2,3 idle. This helper
# fills that idle window by ingesting the 3 largest CPTAC cohorts on those
# GPUs in parallel.
#
# When cv_summary.json appears (signal that retrain finished and Phase 2
# is about to fire), the helper SIGTERMs the CPTAC processes. Any .h5
# files already produced are preserved; Phase 2's stream_ingest_cptac
# skip-on-.h5-exists logic picks up where this left off.

ROOT="/root/Isha/cx-histology"
cd "$ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_TOKEN=$(cat /root/.hf_token 2>/dev/null || echo "")
export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24

mkdir -p logs
LOG="logs/option_b.log"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

# -----------------------------------------------------------------------
# Stage 0: wait for all 8 Phase 1 .done markers
# -----------------------------------------------------------------------
log "option B: waiting for 8 Phase 1 .done markers"
for c in ESCA CESC PAAD THYM COAD PCPG KIRP STAD; do
  while [ ! -f "data/interim/embeddings/tcga/uni/$c/.done" ]; do
    sleep 60
  done
  log "  [$c] done"
done
log "Phase 1 complete; starting CPTAC overlap on cuda:1, cuda:2, cuda:3"

# -----------------------------------------------------------------------
# Stage 1: kick off 3 CPTAC cohorts (largest matching-pair set)
# -----------------------------------------------------------------------
launch_one() {
  local slot=$1 cohort=$2
  local manifest="data/interim/manifests/cptac_idc_${cohort}.tsv"
  if [ ! -f "$manifest" ]; then
    log "  [cuda:$slot] manifest missing for $cohort, skipping"
    return 1
  fi
  if [ -f "data/interim/embeddings/cptac/uni/${cohort}/.done" ]; then
    log "  [cuda:$slot] $cohort .done already, skipping"
    return 1
  fi
  log "  [cuda:$slot] start CPTAC $cohort"
  mkdir -p "data/interim/embeddings/cptac/uni/${cohort}"
  CUDA_VISIBLE_DEVICES=$slot \
    python workflow/scripts/stream_ingest_cptac.py \
      --manifest "$manifest" \
      --raw-dir "data/raw/cptac/${cohort}" \
      --tiles-dir "data/interim/tiles/cptac/${cohort}" \
      --embed-dir "data/interim/embeddings/cptac/uni/${cohort}" \
      --embed-encoder uni \
      --embed-device cuda:0 \
      --embed-batch-size 64 \
      > "logs/option_b_${cohort}.log" 2>&1 &
  echo $!
}

declare -A PIDS
PID1=$(launch_one 1 LUAD); [ -n "$PID1" ] && PIDS[1]=$PID1
PID2=$(launch_one 2 LSCC); [ -n "$PID2" ] && PIDS[2]=$PID2
PID3=$(launch_one 3 BRCA); [ -n "$PID3" ] && PIDS[3]=$PID3

if [ ${#PIDS[@]} -eq 0 ]; then
  log "no CPTAC cohorts could be launched (manifests not ready); exiting"
  exit 0
fi

# -----------------------------------------------------------------------
# Stage 2: wait for 15-cohort cv_summary.json (signal Phase 2 about to fire)
# -----------------------------------------------------------------------
GATE="results/cv/uni_15cohort_final/cv_summary.json"
log "waiting for $GATE (signal that retrain is done and Phase 2 is firing)"
while [ ! -f "$GATE" ]; do
  # If all CPTAC processes finished naturally, exit early
  any=0
  for slot in "${!PIDS[@]}"; do
    pid=${PIDS[$slot]}
    if kill -0 "$pid" 2>/dev/null; then
      any=1
    fi
  done
  if [ $any -eq 0 ]; then
    log "all CPTAC processes completed before retrain done, exiting"
    exit 0
  fi
  sleep 60
done

log "retrain done; SIGTERM-ing CPTAC processes so Phase 2 can take over"
for slot in "${!PIDS[@]}"; do
  pid=${PIDS[$slot]}
  log "  killing cuda:$slot pid=$pid"
  pkill -TERM -P "$pid" 2>/dev/null || true
  kill -TERM "$pid" 2>/dev/null || true
done
sleep 10
# Hard-kill any survivors
for slot in "${!PIDS[@]}"; do
  pid=${PIDS[$slot]}
  pkill -KILL -P "$pid" 2>/dev/null || true
  kill -KILL "$pid" 2>/dev/null || true
done

# Sanity: count partial embeddings preserved
for c in LUAD LSCC BRCA; do
  n=$(ls data/interim/embeddings/cptac/uni/$c/*.h5 2>/dev/null | wc -l)
  log "  CPTAC $c: $n .h5 preserved (phase 2 will resume from here)"
done

log "option B helper done; phase 2 takes over"
