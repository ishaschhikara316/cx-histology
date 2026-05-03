#!/bin/bash
# Wait for all 8 ingest .done markers, then run stage 5 final retrain on
# the combined 15-cohort manifest at the sweep-best hyperparameters.
#
# Best HP from 7-cohort sweep: h=512, dropout=0.3, lr=1e-4.

ROOT="/root/Isha/cx-histology"
cd "$ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_TOKEN=$(cat /root/.hf_token 2>/dev/null || echo "")

mkdir -p logs
LOG="logs/finalize_15cohort.log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

NEW=(ESCA CESC PAAD THYM COAD PCPG KIRP STAD)

log "waiting for all 8 ingest .done markers"
for c in "${NEW[@]}"; do
  marker="data/interim/embeddings/tcga/uni/$c/.done"
  while [ ! -f "$marker" ]; do
    sleep 60
  done
  log "  [$c] done"
done
log "all 8 cohorts complete"

log "building 15-cohort training manifest"
python workflow/scripts/build_train_manifest.py \
  --embed-root data/interim/embeddings/tcga --encoder uni \
  --out data/interim/manifests/train_uni_15cohort.tsv 2>&1 | tee -a "$LOG"

FINAL_DIR="results/cv/uni_15cohort_final"
mkdir -p "$FINAL_DIR"

log "running stage 5 final retrain at best HP (h512_d0.3_lr1e-4) on cuda:0"
python workflow/scripts/train_cv.py \
  --manifest data/interim/manifests/train_uni_15cohort.tsv \
  --labels data/labels/cx_activities.tsv \
  --out-dir "$FINAL_DIR" \
  --encoder uni \
  --hidden 512 --dropout 0.3 --lr 0.0001 \
  --device cuda:0 \
  > logs/finalize_15cohort_train.log 2>&1
rc=$?
log "  train_cv exit code: $rc"

log "generating final figure"
python - <<'PY' 2>&1 | tee -a logs/finalize_15cohort.log
import json, numpy as np, matplotlib.pyplot as plt, pandas as pd, pathlib
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
x = np.arange(17)
ax.bar(x, np.nan_to_num(mean, nan=0.0), yerr=np.nan_to_num(std, nan=0.0),
       color=colors, capsize=3)
for i, v in enumerate(mean):
    if np.isnan(v):
        ax.text(i, 0.02, "n/a", ha="center", va="bottom", fontsize=7, color="gray")
ax.axhline(0.3, ls="--", color="black", alpha=0.4, lw=1, label="H1 target rho >= 0.30")
ax.axhline(0.0, color="black", lw=0.5)
ax.set_xticks(x); ax.set_xticklabels(sigs, rotation=45)
ax.set_ylabel("Spearman rho (5-fold CV, mean +/- std)")
ax.set_title(f"15-cohort TCGA / n={n_pat} slides ({n_lab} labelled) - UNI + ABMIL h512_d0.3_lr1e-4")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(-0.8, 0.8)
fig.tight_layout()
fig.savefig("figures/final_15cohort_rho.pdf")
fig.savefig("figures/final_15cohort_rho.png", dpi=140)

n_h1 = sum(1 for v in mean if not np.isnan(v) and v >= 0.30)
print(f"15-cohort final: {n_coh} cohorts, {n_pat} slides, {n_lab} labelled")
print(f"  mean active rho = {np.nanmean(mean):.3f}")
print(f"  H1 hits (rho >= 0.30): {n_h1}")
print(f"  per-signature top 5: " + ", ".join(
    f"{sigs[i]}:{mean[i]:.3f}" for i in np.argsort(-np.nan_to_num(mean, nan=-1))[:5]))
PY

log "committing results"
git add results/cv/uni_15cohort_final/cv_summary.json \
        figures/final_15cohort_rho.pdf figures/final_15cohort_rho.png \
        data/interim/manifests/train_uni_15cohort.tsv 2>>"$LOG"
git commit -m "results: 15-cohort final retrain at best HP (h512_d0.3_lr1e-4)" 2>>"$LOG"
git push origin main 2>>"$LOG"
log "pushed to origin/main"

log "=== FINALIZE COMPLETE ==="
