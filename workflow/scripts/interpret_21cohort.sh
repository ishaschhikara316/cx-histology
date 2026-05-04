#!/bin/bash
# Run interpretability scripts on 21-cohort fold0 checkpoint:
#   - attention_rollout: attention heatmaps over slides
#   - prototype_tiles:   highest-attention tiles per signature
#   - nuclear_features:  morphological features in attended regions

set -u
ROOT="/root/ISHA/cx-histology"
cd "$ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p logs figures/interpret_21cohort

LOG="logs/interpret_21cohort.log"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

CKPT="results/cv/uni_21cohort_final/fold0.pt"
[ ! -f "$CKPT" ] && { log "no $CKPT; skipping interpretability"; exit 1; }

# Each script has hardcoded paths — run via python -c monkey-patch
run_interp() {
  local name=$1
  log "  starting $name"
  CUDA_VISIBLE_DEVICES=0 python -c "
import sys, os
sys.path.insert(0, '/root/ISHA/cx-histology')
os.chdir('/root/ISHA/cx-histology')
import workflow.scripts.${name} as m
from pathlib import Path
m.BEST_DIR = Path('results/cv/uni_21cohort_final')
m.main()
" > "logs/interpret_${name}.log" 2>&1 &
  echo $!
}

PIDS=()
for s in attention_rollout prototype_tiles nuclear_features; do
  PIDS+=($(run_interp "$s"))
done

fails=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    log "  ok pid=${PIDS[$i]}"
  else
    log "  fail pid=${PIDS[$i]} rc=$?"
    fails=$((fails+1))
  fi
done

log "interpretability done ($fails failures)"
