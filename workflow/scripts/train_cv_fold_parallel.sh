#!/bin/bash
# Fan 5 CV folds out across 5 GPUs concurrently. Drop-in for train_cv.py
# but ~5x wall-clock faster on a 5-GPU box.
#
# Usage:
#   bash train_cv_fold_parallel.sh \
#       --manifest data/interim/manifests/train_uni_21cohort.tsv \
#       --out-dir results/cv/uni_21cohort_final \
#       --hidden 512 --dropout 0.3 --lr 0.0001
# Optional: --gpus "0 1 2 3 4" (default: all visible)
# Optional: --encoder uni (default)
# Optional: --labels data/labels/cx_activities.tsv (default)

set -u
ROOT="/root/ISHA/cx-histology"
cd "$ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

MANIFEST=""
OUT_DIR=""
HIDDEN=""
DROPOUT=""
LR=""
ENCODER="uni"
LABELS="data/labels/cx_activities.tsv"
GPUS="0 1 2 3 4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2;;
    --out-dir)  OUT_DIR="$2";  shift 2;;
    --hidden)   HIDDEN="$2";   shift 2;;
    --dropout)  DROPOUT="$2";  shift 2;;
    --lr)       LR="$2";       shift 2;;
    --encoder)  ENCODER="$2";  shift 2;;
    --labels)   LABELS="$2";   shift 2;;
    --gpus)     GPUS="$2";     shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

for v in MANIFEST OUT_DIR HIDDEN DROPOUT LR; do
  if [ -z "${!v}" ]; then echo "missing required --${v,,}"; exit 2; fi
done

mkdir -p "$OUT_DIR" logs
LOG="logs/$(basename "$OUT_DIR")_fold_parallel.log"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

GPU_ARR=($GPUS)
N_GPUS=${#GPU_ARR[@]}
log "fold-parallel: 5 folds × ${N_GPUS} GPUs (${GPUS}) → out=$OUT_DIR"

# Map fold k → gpu (round-robin)
PIDS=()
for k in 0 1 2 3 4; do
  gpu=${GPU_ARR[$((k % N_GPUS))]}
  log "  fold $k → cuda:$gpu"
  CUDA_VISIBLE_DEVICES=$gpu python workflow/scripts/train_one_fold.py \
    --manifest "$MANIFEST" --labels "$LABELS" \
    --out-dir "$OUT_DIR" --encoder "$ENCODER" \
    --hidden "$HIDDEN" --dropout "$DROPOUT" --lr "$LR" \
    --fold-idx "$k" --device cuda:0 \
    > "logs/$(basename "$OUT_DIR")_fold${k}.log" 2>&1 &
  PIDS+=($!)
done

# Wait
fails=0
for k in 0 1 2 3 4; do
  pid=${PIDS[$k]}
  if wait "$pid"; then
    log "  fold $k OK"
  else
    log "  fold $k FAILED (rc=$?)"
    fails=$((fails+1))
  fi
done
if [ "$fails" -gt 0 ]; then
  log "fold-parallel: $fails fold(s) failed; not aggregating"
  exit 1
fi

# Aggregate fold{0..4}.metrics.json -> cv_summary.json
python - <<PY 2>&1 | tee -a "$LOG"
import json, numpy as np
from pathlib import Path
out = Path("$OUT_DIR")
fm = []
for k in range(5):
    p = out / f"fold{k}.metrics.json"
    fm.append(json.load(open(p)))
fm.sort(key=lambda d: d["fold"])
rho = np.stack([np.asarray(m["spearman_per_sig"], dtype=float) for m in fm], axis=0)
masks = np.stack([np.asarray(m["active_mask"], dtype=int) for m in fm], axis=0)
cv = {
    "fold_metrics": fm,
    "pan_cancer_spearman_mean": np.nanmean(rho, axis=0).tolist(),
    "pan_cancer_spearman_std":  np.nanstd(rho, axis=0).tolist(),
    "active_in_n_folds": masks.sum(axis=0).tolist(),
}
with open(out / "cv_summary.json", "w") as f:
    json.dump(cv, f, indent=2)
mean = np.array(cv["pan_cancer_spearman_mean"], dtype=float)
print(f"  mean active rho = {np.nanmean(mean):+.3f}")
print(f"  H1 hits (>=0.30): {int(np.sum(mean >= 0.30))}")
print(f"wrote {out/'cv_summary.json'}")
PY
log "fold-parallel: complete"
