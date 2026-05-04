#!/bin/bash
# HP re-sweep at n=1454 on cuda:0. The previous "best HP" was tuned at
# n=289; the optimum may have shifted. 6 combos at d=0.3.

set -u
ROOT="/root/ISHA/cx-histology"
cd "$ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
mkdir -p logs results/cv/uni_15cohort_sweep

LOG="logs/sweep_15cohort.log"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

# Run 2 jobs concurrently on cuda:0 — model is small (1 GB / 24 GB), GPU
# was at 2-3% util in the single-process retrain, so concurrency is free.
HIDDENS=(128 256 512)
LRS=(0.0001 0.0005)

run_one() {
  local h=$1 lr=$2
  local out="results/cv/uni_15cohort_sweep/h${h}_d0.3_lr${lr}"
  if [ -f "$out/cv_summary.json" ]; then
    log "  skip h=$h lr=$lr (exists)"
    return 0
  fi
  log "  start h=$h lr=$lr -> $out"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=0 python workflow/scripts/train_cv.py \
    --manifest data/interim/manifests/train_uni_15cohort.tsv \
    --labels data/labels/cx_activities.tsv \
    --out-dir "$out" \
    --encoder uni \
    --hidden $h --dropout 0.3 --lr $lr \
    --device cuda:0 \
    > "logs/sweep_h${h}_lr${lr}.log" 2>&1
  log "  done h=$h lr=$lr rc=$?"
}

# Concurrency-2 slot pool
PIDS=()
launch() {
  run_one "$1" "$2" &
  PIDS+=($!)
  while [ ${#PIDS[@]} -ge 2 ]; do
    new=()
    any_running=0
    for p in "${PIDS[@]}"; do
      if kill -0 "$p" 2>/dev/null; then
        new+=("$p")
        any_running=1
      fi
    done
    PIDS=("${new[@]}")
    [ ${#PIDS[@]} -lt 2 ] && break
    sleep 30
  done
}

log "=== 15-cohort HP sweep on cuda:0 (concurrency-2) ==="
for h in "${HIDDENS[@]}"; do
  for lr in "${LRS[@]}"; do
    # Skip the combo we already ran as the official 15-cohort retrain
    if [ "$h" = "512" ] && [ "$lr" = "0.0001" ]; then
      log "  skip h=512 lr=1e-4 (already ran as final)"
      continue
    fi
    launch "$h" "$lr"
  done
done
wait
log "=== 15-cohort sweep complete ==="

# Build summary
python - <<'PY' 2>&1 | tee -a "$LOG"
import json, glob, numpy as np, os
rows = []
# Include the official h512_d0.3_lr1e-4 from uni_15cohort_final
official = "results/cv/uni_15cohort_final/cv_summary.json"
if os.path.exists(official):
    s = json.load(open(official))
    m = np.array(s["pan_cancer_spearman_mean"], dtype=float)
    rows.append({"hp": "h512_d0.3_lr0.0001", "src": "uni_15cohort_final",
                 "mean_active_rho": float(np.nanmean(m)),
                 "n_h1": int(np.sum(m >= 0.30) if not np.all(np.isnan(m)) else 0),
                 "top_sig": int(np.nanargmax(m)),
                 "top_rho": float(np.nanmax(m))})
for f in sorted(glob.glob("results/cv/uni_15cohort_sweep/*/cv_summary.json")):
    s = json.load(open(f))
    m = np.array(s["pan_cancer_spearman_mean"], dtype=float)
    hp = os.path.basename(os.path.dirname(f))
    rows.append({"hp": hp, "src": "uni_15cohort_sweep",
                 "mean_active_rho": float(np.nanmean(m)),
                 "n_h1": int(np.sum(m >= 0.30) if not np.all(np.isnan(m)) else 0),
                 "top_sig": int(np.nanargmax(m)),
                 "top_rho": float(np.nanmax(m))})
rows.sort(key=lambda r: -r["mean_active_rho"])
print("\nHP sweep at n=1454 (sorted by mean_active_rho):")
print("-" * 78)
print(f"{'HP':<22}  {'src':<20}  {'mean_rho':>8}  {'H1':>3}  top sig")
for r in rows:
    print(f"{r['hp']:<22}  {r['src']:<20}  {r['mean_active_rho']:>+8.3f}  "
          f"{r['n_h1']:>3}  CX{r['top_sig']+1} ({r['top_rho']:+.3f})")
with open("results/cv/uni_15cohort_sweep/sweep_summary.json", "w") as fh:
    json.dump({"rows": rows, "best": rows[0] if rows else None}, fh, indent=2)
print(f"\nwrote results/cv/uni_15cohort_sweep/sweep_summary.json")
PY
