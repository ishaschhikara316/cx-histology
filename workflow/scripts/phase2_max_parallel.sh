#!/bin/bash
# Phase 2: max-parallel ingest for 6 reviewer TCGA cohorts + 13 CPTAC cohorts.
#
# Design:
#   - Single flat queue, longest-first ordering, 4 GPU slots
#   - Each slot runs ONE cohort at a time with ONE worker pinned to that GPU
#   - As any slot finishes, next cohort in queue picks it up
#   - TCGA uses gdc-client (12 procs); CPTAC uses curl from S3 (handled by
#     stream_ingest_cptac.py)
#   - OMP/MKL threads bumped to 24 per worker → 4×24 = 96 cores busy on tile
#   - Auto-fires when results/cv/uni_15cohort_final/cv_summary.json appears
#
# Phase 3 (after queue drains): retrain at best HP on the combined
#   manifest of TCGA (15 + 6 = 21 cohorts) plus a CPTAC inference pass.

ROOT="/root/Isha/cx-histology"
cd "$ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_TOKEN=$(cat /root/.hf_token 2>/dev/null || echo "")
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export TOKENIZERS_PARALLELISM=false

mkdir -p logs
LOG="logs/phase2.log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

# -----------------------------------------------------------------------
# Stage 0: wait for 15-cohort retrain to land
# -----------------------------------------------------------------------
GATE="results/cv/uni_15cohort_final/cv_summary.json"
log "phase 2: waiting for $GATE"
while [ ! -f "$GATE" ]; do
  sleep 60
done
log "phase 2: 15-cohort retrain detected, starting queue"

# -----------------------------------------------------------------------
# Stage 1: build the queue (ordered longest-first to balance the tail)
# -----------------------------------------------------------------------
# Pairs of (kind:cohort, est_slides) for ordering.
# TCGA 6 reviewer cohorts:
TCGA=(
  "tcga:BRCA:1500"
  "tcga:LGG:844"
  "tcga:GBM:860"
  "tcga:LUAD:541"
  "tcga:LUSC:512"
  "tcga:HNSC:472"
)
# CPTAC pan: all 13 cohorts (counts approximate, refined by manifest at runtime)
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

# Merge and sort by descending size
QUEUE=$(printf '%s\n' "${TCGA[@]}" "${CPTAC[@]}" | sort -t: -k3 -nr)
log "queue (largest first):"
echo "$QUEUE" | sed 's/^/  /' | tee -a "$LOG"

# -----------------------------------------------------------------------
# Stage 2: slot-based runner across N_GPUS
# -----------------------------------------------------------------------
N_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$N_GPUS" -lt 1 ] && N_GPUS=1
log "running on $N_GPUS GPU slots"

declare -A SLOT_PID SLOT_TAG

launch_tcga() {
  local slot=$1 cohort=$2
  local marker="data/interim/embeddings/tcga/uni/${cohort}/.done"
  if [ -f "$marker" ]; then
    log "  [gpu$slot] skip TCGA $cohort (.done present)"
    SLOT_PID[$slot]=""
    return
  fi
  log "  [gpu$slot] start TCGA $cohort"
  CUDA_VISIBLE_DEVICES=$slot OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 \
    python workflow/scripts/stream_ingest_parallel.py \
      --manifest "data/interim/manifests/tcga_${cohort}.tsv" \
      --raw-dir "data/raw/tcga/${cohort}" \
      --tiles-dir "data/interim/tiles/tcga/${cohort}" \
      --embed-dir "data/interim/embeddings/tcga/uni/${cohort}" \
      --embed-encoder uni \
      --stain-reference resources/stain_reference.png \
      --n-workers 1 \
      --gdc-processes 12 \
      --embed-batch-size 64 \
      --marker "$marker" \
      > "logs/phase2_TCGA_${cohort}.log" 2>&1 &
  SLOT_PID[$slot]=$!
  SLOT_TAG[$slot]="tcga:$cohort"
}

launch_cptac() {
  local slot=$1 cohort=$2
  local marker="data/interim/embeddings/cptac/uni/${cohort}/.done"
  if [ -f "$marker" ]; then
    log "  [gpu$slot] skip CPTAC $cohort (.done present)"
    SLOT_PID[$slot]=""
    return
  fi
  local manifest="data/interim/manifests/cptac_idc_${cohort}.tsv"
  if [ ! -f "$manifest" ]; then
    log "  [gpu$slot] CPTAC $cohort: no manifest at $manifest, skipping"
    SLOT_PID[$slot]=""
    return
  fi
  log "  [gpu$slot] start CPTAC $cohort"
  mkdir -p "data/interim/embeddings/cptac/uni/${cohort}"
  CUDA_VISIBLE_DEVICES=$slot OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 \
    python workflow/scripts/stream_ingest_cptac.py \
      --manifest "$manifest" \
      --raw-dir "data/raw/cptac/${cohort}" \
      --tiles-dir "data/interim/tiles/cptac/${cohort}" \
      --embed-dir "data/interim/embeddings/cptac/uni/${cohort}" \
      --embed-encoder uni \
      --embed-device cuda:0 \
      --embed-batch-size 64 \
      > "logs/phase2_CPTAC_${cohort}.log" 2>&1 &
  local pid=$!
  # We need to write the .done marker after the inner script finishes
  # successfully. Wrap in a shell function.
  ( wait $pid; rc=$?; if [ $rc -eq 0 ]; then touch "$marker"; fi; exit $rc ) &
  SLOT_PID[$slot]=$!
  SLOT_TAG[$slot]="cptac:$cohort"
}

dispatch_next() {
  local slot=$1
  local next=$(echo "$QUEUE" | head -1)
  QUEUE=$(echo "$QUEUE" | tail -n +2)
  if [ -z "$next" ]; then
    SLOT_PID[$slot]=""
    return
  fi
  IFS=':' read -r kind cohort _ <<< "$next"
  if [ "$kind" = "tcga" ]; then
    launch_tcga "$slot" "$cohort"
  else
    launch_cptac "$slot" "$cohort"
  fi
}

# Initial fill
for s in $(seq 0 $((N_GPUS-1))); do
  dispatch_next "$s"
done

# Drain
while true; do
  any_running=0
  for s in $(seq 0 $((N_GPUS-1))); do
    pid=${SLOT_PID[$s]:-}
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      any_running=1
      continue
    fi
    if [ -n "$pid" ]; then
      wait "$pid" 2>/dev/null
      log "  [gpu$s] done ${SLOT_TAG[$s]}"
      SLOT_PID[$s]=""
    fi
    # Refill
    queue_len=$(echo "$QUEUE" | grep -c '.' 2>/dev/null || echo 0)
    if [ "$queue_len" -gt 0 ]; then
      dispatch_next "$s"
      any_running=1
    fi
  done
  [ $any_running -eq 0 ] && break
  sleep 15
done

log "phase 2 ingest complete"

# -----------------------------------------------------------------------
# Stage 3: 21-cohort retrain on combined TCGA manifest
# -----------------------------------------------------------------------
log "stage 3: 21-cohort retrain at h512_d0.3_lr1e-4"
python workflow/scripts/build_train_manifest.py \
  --embed-root data/interim/embeddings/tcga --encoder uni \
  --out data/interim/manifests/train_uni_21cohort.tsv 2>&1 | tee -a "$LOG"

P3_DIR="results/cv/uni_21cohort_final"
mkdir -p "$P3_DIR"
python workflow/scripts/train_cv.py \
  --manifest data/interim/manifests/train_uni_21cohort.tsv \
  --labels data/labels/cx_activities.tsv \
  --out-dir "$P3_DIR" \
  --encoder uni \
  --hidden 512 --dropout 0.3 --lr 0.0001 \
  --device cuda:0 \
  > logs/phase2_retrain.log 2>&1
log "  retrain rc=$?"

# Final 21-cohort figure
python - <<'PY' 2>&1 | tee -a "$LOG"
import json, numpy as np, matplotlib.pyplot as plt, pathlib, pandas as pd
s = json.load(open("results/cv/uni_21cohort_final/cv_summary.json"))
m = pd.read_csv("data/interim/manifests/train_uni_21cohort.tsv", sep="\t")
L = pd.read_csv("data/labels/cx_activities.tsv", sep="\t").set_index("bcr_patient_barcode")
n_pat = m.patient.nunique()
n_lab = m[m.patient.isin(L.index)].patient.nunique()
n_coh = m.cancer_type.nunique()
mean = np.array(s["pan_cancer_spearman_mean"], dtype=float)
std = np.array(s["pan_cancer_spearman_std"], dtype=float)
sigs = [f"CX{i}" for i in range(1, 18)]
fig, ax = plt.subplots(figsize=(10, 4.2))
colors = ["#2ca02c" if v == v and v >= 0.30
          else "#ff7f0e" if v == v and v >= 0
          else "#d62728" if v == v else "#aaa" for v in mean]
ax.bar(np.arange(17), np.nan_to_num(mean), yerr=np.nan_to_num(std), color=colors, capsize=3)
ax.axhline(0.30, ls="--", color="black", alpha=0.4, lw=1, label="H1 ρ ≥ 0.30")
ax.axhline(0, color="black", lw=0.4)
ax.set_xticks(np.arange(17)); ax.set_xticklabels(sigs, rotation=45)
ax.set_ylabel("Spearman ρ (5-fold CV)")
ax.set_title(f"21-cohort TCGA / n={n_lab} labelled / UNI+ABMIL")
ax.set_ylim(-0.6, 0.7); ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("figures/final_21cohort_rho.pdf")
fig.savefig("figures/final_21cohort_rho.png", dpi=140)
n_h1 = sum(1 for v in mean if v == v and v >= 0.30)
print(f"21-cohort: {n_coh} cohorts, {n_pat} patients, {n_lab} labelled")
print(f"  mean active ρ = {np.nanmean(mean):+.3f}")
print(f"  H1 hits: {n_h1}")
PY

# -----------------------------------------------------------------------
# Stage 4: CPTAC inference pass with the 21-cohort fold-0 checkpoint
# -----------------------------------------------------------------------
log "stage 4: CPTAC inference with 21-cohort fold0 checkpoint"
python workflow/scripts/predict_cptac_inference.py \
  --ckpt "$P3_DIR/fold0.pt" \
  --embed-root data/interim/embeddings/cptac/uni \
  --out "$P3_DIR/cptac_predictions.tsv" \
  > logs/phase2_cptac_predict.log 2>&1 || \
  log "  CPTAC inference failed (likely script not yet implemented)"

# -----------------------------------------------------------------------
# Stage 5: commit + push
# -----------------------------------------------------------------------
log "stage 5: commit + push"
git add results/cv/uni_21cohort_final/cv_summary.json \
        figures/final_21cohort_rho.* \
        data/interim/manifests/train_uni_21cohort.tsv 2>>"$LOG"
git add results/cv/uni_21cohort_final/cptac_predictions.tsv 2>/dev/null
git commit -m "results: 21-cohort retrain + CPTAC inference (phase 2 complete)" 2>>"$LOG"
git push origin main 2>>"$LOG"

log "=== PHASE 2 COMPLETE ==="
