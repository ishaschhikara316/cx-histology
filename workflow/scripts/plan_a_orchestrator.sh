#!/bin/bash
# Plan A orchestrator: runs post-batch-1 chain unattended.
#
# Assumes batch 1 (READ,MESO,UCS,KICH) is already running or complete.
# Waits for all 4 .done markers, then:
#   1. Train 7-cohort UNI+ABMIL + HP sweep
#   2. CPTAC-BRCA smoke test (tests TCIA ingest path)
#   3. Second TCGA batch (STAD + COAD)
#   4. Final 10-cohort retrain + figures + dashboard
#
# All output goes to logs/plan_a_<stage>.log. Safe to re-run — each stage
# checks for its own output before doing work.

set -u  # no -e: we want to continue past stage failures

ROOT="/root/Project/Isha/cx-histology"
cd "$ROOT"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate cx-histology
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p logs results figures

log() { echo "[$(date -Iseconds)] $*" | tee -a logs/plan_a.log; }

# ----------------------------------------------------------------------
# Stage 0: wait for batch 1
# ----------------------------------------------------------------------
log "stage 0: waiting for batch 1 .done markers"
for c in READ MESO UCS KICH; do
  marker="data/interim/embeddings/tcga/uni/$c/.done"
  while [ ! -f "$marker" ]; do
    sleep 60
  done
  log "  [$c] done"
done
log "batch 1 complete"

# ----------------------------------------------------------------------
# Stage 1: retrain 7-cohort + HP sweep
# ----------------------------------------------------------------------
log "stage 1: 7-cohort train + HP sweep"

python workflow/scripts/build_train_manifest.py \
  --embed-root data/interim/embeddings/tcga --encoder uni \
  --out data/interim/manifests/train_uni_7cohort.tsv

HP_SWEEP_DIR="results/cv/uni_7cohort_sweep"
mkdir -p "$HP_SWEEP_DIR"

# Coarse sweep: 6 combos (3 hidden × 2 lr), dropout fixed at 0.3 for speed.
for HIDDEN in 128 256 512; do
  for LR in 0.0001 0.0005; do
    out="$HP_SWEEP_DIR/h${HIDDEN}_d0.3_lr${LR}"
    if [ -f "$out/cv_summary.json" ]; then
      log "  skip h=${HIDDEN} lr=${LR} (done)"
      continue
    fi
    log "  train h=${HIDDEN} d=0.3 lr=${LR}"
    mkdir -p "$out"
    python workflow/scripts/train_cv.py \
      --manifest data/interim/manifests/train_uni_7cohort.tsv \
      --labels data/labels/cx_activities.tsv \
      --out-dir "$out" \
      --encoder uni \
      --hidden "$HIDDEN" --dropout 0.3 --lr "$LR" \
      > "logs/plan_a_train_h${HIDDEN}_lr${LR}.log" 2>&1
  done
done

# Pick best HP by mean ρ over well-defined signatures.
python - <<'PY'
import json, pathlib, numpy as np
best = (None, -1e9)
for p in pathlib.Path("results/cv/uni_7cohort_sweep").glob("*/cv_summary.json"):
    s = json.load(open(p))
    m = np.array(s["pan_cancer_spearman_mean"], dtype=float)
    score = np.nanmean(m)
    print(f"  {p.parent.name}: mean ρ = {score:.3f}")
    if score > best[1]:
        best = (p.parent.name, score)
print(f"best: {best[0]} (mean ρ = {best[1]:.3f})")
pathlib.Path("results/cv/uni_7cohort_sweep/_best.txt").write_text(best[0])
PY

BEST=$(cat results/cv/uni_7cohort_sweep/_best.txt)
log "stage 1 done. best HP: $BEST"

# ----------------------------------------------------------------------
# Stage 2: interpretability figures on the 7-cohort model
# ----------------------------------------------------------------------
log "stage 2: 7-cohort figures"

BEST_CKPT="results/cv/uni_7cohort_sweep/$BEST/fold0.pt"

# Per-signature ρ bar chart for 7-cohort model
python - <<'PY'
import json, numpy as np, matplotlib.pyplot as plt, pathlib
best = pathlib.Path("results/cv/uni_7cohort_sweep/_best.txt").read_text().strip()
s = json.load(open(f"results/cv/uni_7cohort_sweep/{best}/cv_summary.json"))
mean = np.array(s["pan_cancer_spearman_mean"], dtype=float)
std  = np.array(s["pan_cancer_spearman_std"],  dtype=float)
sigs = [f"CX{i}" for i in range(1, 18)]
fig, ax = plt.subplots(figsize=(10, 4.2))
colors = ["#2ca02c" if not np.isnan(m) and m >= 0.30
          else "#ff7f0e" if not np.isnan(m) and m >= 0.0
          else "#d62728" if not np.isnan(m) else "#aaaaaa" for m in mean]
x = np.arange(17)
ax.bar(x, np.nan_to_num(mean, nan=0.0), yerr=np.nan_to_num(std, nan=0.0),
       color=colors, capsize=3)
for i, m in enumerate(mean):
    if np.isnan(m):
        ax.text(i, 0.02, "n/a", ha="center", va="bottom", fontsize=7, color="gray")
ax.axhline(0.3, ls="--", color="black", alpha=0.4, lw=1, label="H1 target ρ ≥ 0.30")
ax.axhline(0.0, color="black", lw=0.5)
ax.set_xticks(x); ax.set_xticklabels(sigs, rotation=45)
ax.set_ylabel("Spearman ρ (5-fold CV, mean ± std)")
ax.set_title(f"7-cohort TCGA — UNI + ABMIL [{best}]")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(-0.8, 0.8)
fig.tight_layout()
fig.savefig("figures/prelim_7cohort_rho.pdf")
fig.savefig("figures/prelim_7cohort_rho.png", dpi=140)
PY

log "stage 2 done (figures/prelim_7cohort_rho.{pdf,png})"

# ----------------------------------------------------------------------
# Stage 3: CPTAC-BRCA smoke test
# ----------------------------------------------------------------------
log "stage 3: CPTAC-BRCA ingest (TCIA REST — untested path)"

if [ ! -f data/interim/embeddings/cptac/uni/BRCA/.done ]; then
  # First, do the TCIA manifest + download in a single call via ingest_cptac.py
  # then pipe those slides through stream ingest. The existing TCIA path
  # doesn't use gdc-client, so we run ingest_cptac.py to pull .zip files,
  # then unpack and stream-tile them.
  mkdir -p data/raw/cptac/BRCA data/interim/embeddings/cptac/uni/BRCA
  python workflow/scripts/ingest_cptac.py \
    --cohort CPTAC-BRCA \
    --out-dir data/raw/cptac/BRCA \
    > logs/plan_a_cptac_brca_manifest.log 2>&1 \
    && log "  CPTAC-BRCA manifest + download succeeded" \
    || log "  CPTAC-BRCA ingest FAILED — see logs/plan_a_cptac_brca_manifest.log; skipping stage"
fi

# Note: CPTAC path is known-untested. If it fails, we skip cleanly and
# continue with stages 4-5. Don't waste time debugging TCIA here;
# bubble up a failure marker instead.
if [ -f data/raw/cptac/BRCA/_manifest.tsv ]; then
  log "  CPTAC-BRCA: $(wc -l < data/raw/cptac/BRCA/_manifest.tsv) series listed"
fi

# ----------------------------------------------------------------------
# Stage 4: second TCGA batch — STAD + COAD
# ----------------------------------------------------------------------
log "stage 4: batch 2 — STAD + COAD"

python workflow/scripts/stream_ingest_batch.py \
  --cohorts STAD,COAD \
  --embed-encoder uni \
  --n-workers 4 --gdc-processes 4 \
  > logs/plan_a_batch_2.log 2>&1

for c in STAD COAD; do
  marker="data/interim/embeddings/tcga/uni/$c/.done"
  if [ -f "$marker" ]; then log "  [$c] done"; else log "  [$c] NOT DONE"; fi
done

# ----------------------------------------------------------------------
# Stage 5: final retrain + figures on whatever cohorts finished
# ----------------------------------------------------------------------
log "stage 5: final retrain on all finished cohorts"

python workflow/scripts/build_train_manifest.py \
  --embed-root data/interim/embeddings/tcga --encoder uni \
  --out data/interim/manifests/train_uni_final.tsv

BEST_HP=$(cat results/cv/uni_7cohort_sweep/_best.txt)
HIDDEN=$(echo "$BEST_HP" | grep -oP 'h\K\d+')
DROPOUT=$(echo "$BEST_HP" | grep -oP 'd\K[0-9.]+')
LR=$(echo "$BEST_HP" | grep -oP 'lr\K[0-9.]+')

FINAL_DIR="results/cv/uni_final"
mkdir -p "$FINAL_DIR"
python workflow/scripts/train_cv.py \
  --manifest data/interim/manifests/train_uni_final.tsv \
  --labels data/labels/cx_activities.tsv \
  --out-dir "$FINAL_DIR" \
  --encoder uni \
  --hidden "$HIDDEN" --dropout "$DROPOUT" --lr "$LR" \
  > logs/plan_a_final_train.log 2>&1

# Final figure
python - <<'PY'
import json, numpy as np, matplotlib.pyplot as plt, pathlib, pandas as pd
s = json.load(open("results/cv/uni_final/cv_summary.json"))
m = pd.read_csv("data/interim/manifests/train_uni_final.tsv", sep="\t")
L = pd.read_csv("data/labels/cx_activities.tsv", sep="\t").set_index("bcr_patient_barcode")
n_pat = m.patient.nunique()
n_lab = m[m.patient.isin(L.index)].patient.nunique()
n_coh = m.cancer_type.nunique()

mean = np.array(s["pan_cancer_spearman_mean"], dtype=float)
std  = np.array(s["pan_cancer_spearman_std"],  dtype=float)
sigs = [f"CX{i}" for i in range(1, 18)]
fig, ax = plt.subplots(figsize=(10, 4.2))
colors = ["#2ca02c" if not np.isnan(m_) and m_ >= 0.30
          else "#ff7f0e" if not np.isnan(m_) and m_ >= 0.0
          else "#d62728" if not np.isnan(m_) else "#aaaaaa" for m_ in mean]
x = np.arange(17)
ax.bar(x, np.nan_to_num(mean, nan=0.0), yerr=np.nan_to_num(std, nan=0.0),
       color=colors, capsize=3)
for i, m_ in enumerate(mean):
    if np.isnan(m_):
        ax.text(i, 0.02, "n/a", ha="center", va="bottom", fontsize=7, color="gray")
ax.axhline(0.3, ls="--", color="black", alpha=0.4, lw=1, label="H1 target ρ ≥ 0.30")
ax.axhline(0.0, color="black", lw=0.5)
ax.set_xticks(x); ax.set_xticklabels(sigs, rotation=45)
ax.set_ylabel("Spearman ρ (5-fold CV, mean ± std)")
ax.set_title(f"TCGA {n_coh} cohorts / n={n_pat} slides ({n_lab} labeled) — UNI + ABMIL")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(-0.8, 0.8)
fig.tight_layout()
fig.savefig("figures/final_rho.pdf")
fig.savefig("figures/final_rho.png", dpi=140)
print(f"final: {n_coh} cohorts, {n_pat} slides, {n_lab} labeled; "
      f"max ρ = {np.nanmax(mean):.2f}")
PY

log "stage 5 done. Final figure at figures/final_rho.pdf"
log "PLAN A COMPLETE"
