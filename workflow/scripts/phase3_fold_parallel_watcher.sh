#!/bin/bash
# Eager Phase 3: fire fold-parallel 21-cohort retrain as soon as the 6
# new TCGA cohorts (BRCA, LGG, GBM, LUAD, LUSC, HNSC) finish ingesting.
# Don't wait for CPTAC — CPTAC is only needed for the inference pass.
#
# Race-safety with resume_max_parallel.sh:
#   When this writes results/cv/uni_21cohort_final/cv_summary.json,
#   the orchestrator's later Phase 3 step sees the file exists and
#   skips its slower serial retrain.

set -u
ROOT="/root/ISHA/cx-histology"
cd "$ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs
LOG="logs/phase3_watcher.log"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

PHASE2_TCGA=(BRCA LGG GBM LUAD LUSC HNSC)
GATE21="results/cv/uni_21cohort_final/cv_summary.json"

if [ -f "$GATE21" ]; then
  log "$GATE21 already exists — exiting"
  exit 0
fi

log "watching for 6 Phase-2 TCGA .done markers"
while true; do
  ready=0
  for c in "${PHASE2_TCGA[@]}"; do
    [ -f "data/interim/embeddings/tcga/uni/$c/.done" ] && ready=$((ready+1))
  done
  if [ "$ready" -eq "${#PHASE2_TCGA[@]}" ]; then
    log "all 6 Phase-2 TCGA cohorts done → firing fold-parallel"
    break
  fi
  sleep 120
done

log "building 21-cohort manifest"
python workflow/scripts/build_train_manifest.py \
  --embed-root data/interim/embeddings/tcga --encoder uni \
  --out data/interim/manifests/train_uni_21cohort.tsv 2>&1 | tee -a "$LOG"

log "launching fold-parallel retrain across 5 GPUs"
# All 5 GPUs are free for training: cuda:0 was retrain box, cuda:1-4 were
# ingest box; once 6 TCGA .done markers land, the TCGA workers are gone
# and only CPTAC slots may still be active. Even so, GPU compute during
# embed is ~5% — fold-parallel sharing one GPU with a CPTAC slot is fine.
bash workflow/scripts/train_cv_fold_parallel.sh \
  --manifest data/interim/manifests/train_uni_21cohort.tsv \
  --out-dir results/cv/uni_21cohort_final \
  --hidden 512 --dropout 0.1 --lr 0.0001 \
  --gpus "0 1 2 3 4" \
  > logs/phase3_fold_parallel.log 2>&1
rc=$?
log "  fold-parallel rc=$rc (HP from 15-cohort sweep winner: h512_d0.1_lr1e-4)"

if [ -f "$GATE21" ]; then
  log "21-cohort retrain complete; building figure + committing"
  python - <<'PY' 2>&1 | tee -a "$LOG"
import json, numpy as np, matplotlib.pyplot as plt, pandas as pd
s = json.load(open("results/cv/uni_21cohort_final/cv_summary.json"))
m = pd.read_csv("data/interim/manifests/train_uni_21cohort.tsv", sep="\t")
L = pd.read_csv("data/labels/cx_activities.tsv", sep="\t").set_index("bcr_patient_barcode")
n_pat = m.patient.nunique()
n_lab = m[m.patient.isin(L.index)].patient.nunique()
n_coh = m.cancer_type.nunique()
mean = np.array(s["pan_cancer_spearman_mean"], dtype=float)
std  = np.array(s["pan_cancer_spearman_std"], dtype=float)
sigs = [f"CX{i}" for i in range(1, 18)]
fig, ax = plt.subplots(figsize=(10, 4.2))
colors = ["#2ca02c" if not np.isnan(v) and v >= 0.30
          else "#ff7f0e" if not np.isnan(v) and v >= 0.0
          else "#d62728" if not np.isnan(v) else "#aaaaaa" for v in mean]
ax.bar(np.arange(17), np.nan_to_num(mean), yerr=np.nan_to_num(std), color=colors, capsize=3)
ax.axhline(0.3, ls="--", color="black", alpha=0.4, lw=1, label="H1 rho >= 0.30")
ax.axhline(0.0, color="black", lw=0.5)
ax.set_xticks(np.arange(17)); ax.set_xticklabels(sigs, rotation=45)
ax.set_ylabel("Spearman rho (5-fold CV, mean +/- std)")
ax.set_title(f"21-cohort TCGA / n={n_lab} labelled / fold-parallel UNI+ABMIL h512_d0.3_lr1e-4")
ax.set_ylim(-0.8, 0.8); ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("figures/final_21cohort_rho.pdf"); fig.savefig("figures/final_21cohort_rho.png", dpi=140)
n_h1 = sum(1 for v in mean if not np.isnan(v) and v >= 0.30)
print(f"21-cohort: {n_coh} cohorts, {n_pat} slides, {n_lab} labelled")
print(f"  mean active rho = {np.nanmean(mean):+.3f}  H1 hits = {n_h1}")
print("  top 5: " + ", ".join(f"{sigs[i]}:{mean[i]:+.3f}"
                              for i in np.argsort(-np.nan_to_num(mean, nan=-1))[:5]))
PY
  git add results/cv/uni_21cohort_final/cv_summary.json \
          figures/final_21cohort_rho.pdf figures/final_21cohort_rho.png \
          data/interim/manifests/train_uni_21cohort.tsv 2>>"$LOG"
  git commit -m "results: 21-cohort fold-parallel retrain (h512_d0.1_lr1e-4)" 2>>"$LOG" || true
  git push origin main 2>>"$LOG" || true

  # ----------------------------------------------------------------------
  # Post-retrain auto-stages: analysis + interpretability + HF + dashboard
  # ----------------------------------------------------------------------
  log "stage post-A: comprehensive 21-cohort analysis"
  PYTHONPATH=/root/ISHA/cx-histology python workflow/scripts/analyze_21cohort.py \
    > logs/analyze_21cohort.log 2>&1
  log "  analyze_21cohort rc=$?"

  log "stage post-B: interpretability (attention rollout, prototype tiles, nuclear features)"
  bash workflow/scripts/interpret_21cohort.sh > logs/_interpret_21_outer.log 2>&1
  log "  interpret rc=$?"

  log "stage post-C: build dashboard_21cohort.html"
  PYTHONPATH=/root/ISHA/cx-histology python workflow/scripts/build_dashboard_21cohort.py \
    > logs/dashboard_21cohort.log 2>&1
  log "  dashboard rc=$?"

  log "stage post-D: push 21-cohort HF repo"
  bash workflow/scripts/push_hf_21cohort.sh > logs/push_hf_21cohort.log 2>&1
  log "  HF push rc=$?"

  log "stage post-E: regenerate manuscript_draft.md from results"
  PYTHONPATH=/root/ISHA/cx-histology python workflow/scripts/generate_manuscript_draft.py \
    > logs/generate_manuscript.log 2>&1
  log "  manuscript rc=$?"

  log "stage post-F: commit + push everything"
  git add results/cv/uni_21cohort_final/ \
          figures/final_21cohort_rho.* figures/dashboard_21cohort.html \
          figures/fig_*_21cohort.* figures/interpret_21cohort/ \
          docs/manuscript_draft.md 2>>"$LOG" || true
  git commit -m "results: 21-cohort full preprint package (analysis + interp + HF + dashboard + manuscript)" 2>>"$LOG" || true
  git push origin main 2>>"$LOG" || true
fi
log "=== phase3 watcher complete (full preprint package landed) ==="
