#!/bin/bash
# One-off resume orchestrator for the 5-GPU server.
#
# Old phase2_max_parallel.sh was sequential (retrain THEN ingest) on 4 GPUs.
# This server has 5 GPUs, so:
#   - cuda:0          15-cohort retrain at h512_d0.3_lr1e-4 (the gating run)
#   - cuda:{1,2,3,4}  Phase 2 ingest queue (6 TCGA + 13 CPTAC, longest-first)
# When both finish, runs the 21-cohort retrain + CPTAC inference + push.

set -u
ROOT="/root/ISHA/cx-histology"
cd "$ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || echo)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export TOKENIZERS_PARALLELISM=false

# 30-core box, 4 ingest workers + 1 retrain process. Give each ~6 threads
# during the concurrent phase. After ingest drains, Phase 3 retrain gets
# full CPU.
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

mkdir -p logs results/cv/uni_15cohort_final results/cv/uni_21cohort_final \
         data/raw/tcga data/raw/cptac
LOG="logs/resume_max_parallel.log"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

GATE15="results/cv/uni_15cohort_final/cv_summary.json"

# ----------------------------------------------------------------------
# Child A: 15-cohort retrain on cuda:0
# ----------------------------------------------------------------------
launch_retrain_15() {
  if [ -f "$GATE15" ]; then
    log "[retrain15] $GATE15 already exists — skipping"
    return 0
  fi
  log "[retrain15] launching on cuda:0 (h512_d0.3_lr1e-4)"
  CUDA_VISIBLE_DEVICES=0 python workflow/scripts/train_cv.py \
    --manifest data/interim/manifests/train_uni_15cohort.tsv \
    --labels data/labels/cx_activities.tsv \
    --out-dir results/cv/uni_15cohort_final \
    --encoder uni \
    --hidden 512 --dropout 0.3 --lr 0.0001 \
    --device cuda:0 \
    > logs/retrain_15cohort.log 2>&1 &
  echo $! > logs/.retrain15.pid
}

# ----------------------------------------------------------------------
# Child B: Phase 2 ingest queue on physical GPUs 1..4
# ----------------------------------------------------------------------
launch_ingest_pool() {
  log "[ingest] starting 4-slot queue on physical GPUs 1,2,3,4"
  bash workflow/scripts/_ingest_pool.sh > logs/ingest_pool.log 2>&1 &
  echo $! > logs/.ingest.pid
}

# ----------------------------------------------------------------------
# Wait for both
# ----------------------------------------------------------------------
launch_retrain_15
launch_ingest_pool

R_PID=$(cat logs/.retrain15.pid 2>/dev/null || echo)
I_PID=$(cat logs/.ingest.pid 2>/dev/null || echo)
log "retrain pid=$R_PID  ingest pid=$I_PID"

if [ -n "$R_PID" ]; then
  wait "$R_PID" 2>/dev/null
  log "[retrain15] exit code: $?"
fi
if [ -n "$I_PID" ]; then
  wait "$I_PID" 2>/dev/null
  log "[ingest] exit code: $?"
fi

# ----------------------------------------------------------------------
# Stage: 15-cohort headline figure + push
# ----------------------------------------------------------------------
if [ -f "$GATE15" ]; then
  log "stage 15-fig: building final_15cohort_rho figure"
  python - <<'PY' 2>&1 | tee -a "$LOG"
import json, numpy as np, matplotlib.pyplot as plt, pandas as pd
s = json.load(open("results/cv/uni_15cohort_final/cv_summary.json"))
m = pd.read_csv("data/interim/manifests/train_uni_15cohort.tsv", sep="\t")
L = pd.read_csv("data/labels/cx_activities.tsv", sep="\t").set_index("bcr_patient_barcode")
n_pat = m.patient.nunique()
n_lab = m[m.patient.isin(L.index)].patient.nunique()
n_coh = m.cancer_type.nunique()
mean = np.array(s["pan_cancer_spearman_mean"], dtype=float)
std  = np.array(s["pan_cancer_spearman_std"],  dtype=float)
sigs = [f"CX{i}" for i in range(1, 18)]
fig, ax = plt.subplots(figsize=(10, 4.2))
colors = ["#2ca02c" if not np.isnan(v) and v >= 0.30
          else "#ff7f0e" if not np.isnan(v) and v >= 0.0
          else "#d62728" if not np.isnan(v) else "#aaaaaa" for v in mean]
ax.bar(np.arange(17), np.nan_to_num(mean), yerr=np.nan_to_num(std), color=colors, capsize=3)
for i, v in enumerate(mean):
    if np.isnan(v):
        ax.text(i, 0.02, "n/a", ha="center", va="bottom", fontsize=7, color="gray")
ax.axhline(0.3, ls="--", color="black", alpha=0.4, lw=1, label="H1 rho >= 0.30")
ax.axhline(0.0, color="black", lw=0.5)
ax.set_xticks(np.arange(17)); ax.set_xticklabels(sigs, rotation=45)
ax.set_ylabel("Spearman rho (5-fold CV, mean +/- std)")
ax.set_title(f"15-cohort TCGA / n={n_pat} slides ({n_lab} labelled) - UNI + ABMIL h512_d0.3_lr1e-4")
ax.legend(loc="upper right", fontsize=9); ax.set_ylim(-0.8, 0.8)
fig.tight_layout()
fig.savefig("figures/final_15cohort_rho.pdf"); fig.savefig("figures/final_15cohort_rho.png", dpi=140)
n_h1 = sum(1 for v in mean if not np.isnan(v) and v >= 0.30)
print(f"15-cohort: {n_coh} cohorts, {n_pat} slides, {n_lab} labelled")
print(f"  mean active rho = {np.nanmean(mean):+.3f}  H1 hits = {n_h1}")
print("  top 5: " + ", ".join(f"{sigs[i]}:{mean[i]:+.3f}"
                              for i in np.argsort(-np.nan_to_num(mean, nan=-1))[:5]))
PY

  log "stage 15-push"
  git add results/cv/uni_15cohort_final/cv_summary.json \
          figures/final_15cohort_rho.pdf figures/final_15cohort_rho.png \
          data/interim/manifests/train_uni_15cohort.tsv 2>>"$LOG"
  git commit -m "results: 15-cohort final retrain (h512_d0.3_lr1e-4)" 2>>"$LOG" || true
  git push origin main 2>>"$LOG" || true
else
  log "[retrain15] cv_summary.json missing — skipping figure + push"
fi

# ----------------------------------------------------------------------
# Stage: 21-cohort retrain (only if Phase 2 ingest produced new TCGA cohorts)
# ----------------------------------------------------------------------
GATE21="results/cv/uni_21cohort_final/cv_summary.json"
if [ -f "$GATE21" ]; then
  log "[retrain21] $GATE21 already exists — skipping"
else
  log "stage 21: building 21-cohort manifest"
  python workflow/scripts/build_train_manifest.py \
    --embed-root data/interim/embeddings/tcga --encoder uni \
    --out data/interim/manifests/train_uni_21cohort.tsv 2>&1 | tee -a "$LOG"

  log "[retrain21] launching on cuda:0 (h512_d0.3_lr1e-4)"
  CUDA_VISIBLE_DEVICES=0 python workflow/scripts/train_cv.py \
    --manifest data/interim/manifests/train_uni_21cohort.tsv \
    --labels data/labels/cx_activities.tsv \
    --out-dir results/cv/uni_21cohort_final \
    --encoder uni \
    --hidden 512 --dropout 0.3 --lr 0.0001 \
    --device cuda:0 \
    > logs/retrain_21cohort.log 2>&1
  log "[retrain21] exit code: $?"
fi

# ----------------------------------------------------------------------
# Stage: 21-cohort figure + CPTAC inference + push
# ----------------------------------------------------------------------
if [ -f "$GATE21" ]; then
  log "stage 21-fig"
  python - <<'PY' 2>&1 | tee -a "$LOG"
import json, numpy as np, matplotlib.pyplot as plt, pandas as pd
s = json.load(open("results/cv/uni_21cohort_final/cv_summary.json"))
m = pd.read_csv("data/interim/manifests/train_uni_21cohort.tsv", sep="\t")
L = pd.read_csv("data/labels/cx_activities.tsv", sep="\t").set_index("bcr_patient_barcode")
n_pat = m.patient.nunique()
n_lab = m[m.patient.isin(L.index)].patient.nunique()
n_coh = m.cancer_type.nunique()
mean = np.array(s["pan_cancer_spearman_mean"], dtype=float)
std  = np.array(s["pan_cancer_spearman_std"],  dtype=float)
sigs = [f"CX{i}" for i in range(1, 18)]
fig, ax = plt.subplots(figsize=(10, 4.2))
colors = ["#2ca02c" if not np.isnan(v) and v >= 0.30
          else "#ff7f0e" if not np.isnan(v) and v >= 0.0
          else "#d62728" if not np.isnan(v) else "#aaaaaa" for v in mean]
ax.bar(np.arange(17), np.nan_to_num(mean), yerr=np.nan_to_num(std), color=colors, capsize=3)
ax.axhline(0.3, ls="--", color="black", alpha=0.4, lw=1, label="H1 rho >= 0.30")
ax.axhline(0.0, color="black", lw=0.5)
ax.set_xticks(np.arange(17)); ax.set_xticklabels(sigs, rotation=45)
ax.set_ylabel("Spearman rho (5-fold CV)")
ax.set_title(f"21-cohort TCGA / n={n_lab} labelled / UNI + ABMIL h512_d0.3_lr1e-4")
ax.set_ylim(-0.8, 0.8); ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("figures/final_21cohort_rho.pdf"); fig.savefig("figures/final_21cohort_rho.png", dpi=140)
n_h1 = sum(1 for v in mean if not np.isnan(v) and v >= 0.30)
print(f"21-cohort: {n_coh} cohorts, {n_pat} slides, {n_lab} labelled")
print(f"  mean active rho = {np.nanmean(mean):+.3f}  H1 hits = {n_h1}")
PY

  log "stage cptac-infer"
  CUDA_VISIBLE_DEVICES=0 python workflow/scripts/predict_cptac_inference.py \
    --ckpt results/cv/uni_21cohort_final/fold0.pt \
    --embed-root data/interim/embeddings/cptac/uni \
    --out results/cv/uni_21cohort_final/cptac_predictions.tsv \
    > logs/cptac_inference.log 2>&1 || \
    log "[cptac-infer] non-zero exit (script may need debug)"

  log "stage push"
  git add results/cv/uni_21cohort_final/cv_summary.json \
          figures/final_21cohort_rho.pdf figures/final_21cohort_rho.png \
          data/interim/manifests/train_uni_21cohort.tsv 2>>"$LOG"
  git add results/cv/uni_21cohort_final/cptac_predictions.tsv 2>/dev/null
  git commit -m "results: 21-cohort retrain + CPTAC inference (phase 2 complete)" 2>>"$LOG" || true
  git push origin main 2>>"$LOG" || true
fi

log "=== RESUME COMPLETE ==="
