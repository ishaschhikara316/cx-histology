#!/bin/bash
# Run interpretability on the 21-cohort fold0 checkpoint.
#
# Disk reality on this box: embedding .h5 files contain only `coords` and
# `embeddings` — raw tile pixels were not persisted during ingest to save
# disk. That means:
#   - attention_rollout: runs (only needs embeddings + coords; slide_wh
#     is derived from coords.max() + patch_px).
#   - prototype_tiles:   skipped (script reads h5["tiles"][:] which does
#     not exist; would need ingest re-run with tile-cache flag).
#   - nuclear_features:  skipped (needs raw tile dirs which were deleted).
#
# To re-enable the skipped scripts, re-run ingest with a per-slide tile
# cache and add a `tile_path` column to the train manifest.
#
# Outputs: figures/interpret_21cohort/<patient>__cx<sig>.npy heatmaps and
# a small index TSV the dashboard can pick up.

# Resolve repo root: prefer the GPU box layout, fall back to the
# script's grandparent directory so this works on a developer Mac too.
if [ -d "/root/ISHA/cx-histology" ]; then
  ROOT="/root/ISHA/cx-histology"
else
  ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fi
cd "$ROOT"

# Activate conda only on the GPU box. On Mac/CI, the caller is
# expected to have already activated the right env (or have python
# on PATH with torch + h5py + click installed).
if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate cx-histology
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
set -u

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p logs figures/interpret_21cohort

LOG="logs/interpret_21cohort.log"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG" >&2; }

CKPT="results/cv/uni_21cohort_final/fold0.pt"
MANIFEST="data/interim/manifests/train_uni_21cohort.tsv"
OUT_DIR="figures/interpret_21cohort"
INDEX_TSV="$OUT_DIR/index.tsv"
PATCH_PX=256
N_SLIDES=5
# Signatures to map: CX1 (idx 0), CX3 (2), CX5 (4), CX10 (9), CX12 (11)
# — the top-active heads from the 21-cohort retrain.
SIG_IDXS=(0 2 4 9 11)

[ ! -f "$CKPT" ]     && { log "no $CKPT; skipping interpretability";    exit 1; }
[ ! -f "$MANIFEST" ] && { log "no $MANIFEST; skipping interpretability"; exit 1; }

log "attention_rollout: $N_SLIDES slides x ${#SIG_IDXS[@]} signatures"
log "prototype_tiles  : SKIP (h5['tiles'] not on disk)"
log "nuclear_features : SKIP (raw tile dirs deleted)"

# Pick the first N_SLIDES rows with >=100 tiles. Manifest cols:
# patient, cancer_type, emb_path, n_tiles
mapfile -t ROWS < <(awk -F'\t' 'NR>1 && $4>=100 {print $1"\t"$3}' "$MANIFEST" | head -n "$N_SLIDES")

if [ "${#ROWS[@]}" -eq 0 ]; then
  log "no eligible slides in manifest; aborting"
  exit 1
fi

echo -e "patient\tsig_idx\theatmap" > "$INDEX_TSV"

run_attention() {
  local patient=$1 emb=$2 sig=$3
  local out="$OUT_DIR/${patient}__cx$((sig+1)).npy"
  # Derive slide W,H from coords.max() + patch_px so we don't need to
  # round-trip to the original WSI just for thumbnail dimensions.
  local wh
  wh=$(python -c "
import h5py, numpy as np
with h5py.File('$emb','r') as h:
    c = h['coords'][:]
W = int(c[:,0].max()) + $PATCH_PX
H = int(c[:,1].max()) + $PATCH_PX
print(f'{W},{H}')
") || { log "  fail wh $patient"; return 1; }
  # attention_rollout.py loads with map_location="cpu" and never calls
  # .cuda(), so it runs on either a GPU box or a Mac with the same
  # invocation. CUDA_VISIBLE_DEVICES is harmless when no GPU exists.
  CUDA_VISIBLE_DEVICES=0 python -m workflow.scripts.attention_rollout \
    --emb "$emb" --ckpt "$CKPT" --slide-wh "$wh" \
    --out "$out" --sig-idx "$sig" \
    > "logs/interpret_attn_${patient}_cx$((sig+1)).log" 2>&1 \
    && echo -e "${patient}\tcx$((sig+1))\t${out}" >> "$INDEX_TSV"
}

fails=0
for row in "${ROWS[@]}"; do
  patient="${row%%	*}"
  emb="${row##*	}"
  for sig in "${SIG_IDXS[@]}"; do
    if ! run_attention "$patient" "$emb" "$sig"; then
      fails=$((fails+1))
      log "  fail $patient cx$((sig+1))"
    fi
  done
done

n_done=$(($(wc -l < "$INDEX_TSV") - 1))
log "interpretability done: $n_done heatmaps, $fails failures"
log "index: $INDEX_TSV"

# Compose figures/fig4_interpret_attention.png from the heatmap grid so the
# dashboard's interp slot is populated. Slide background not available
# (raw tiles deleted), so we render colormap-only heatmaps in a grid.
log "composing fig4_interpret_attention.png"
python - <<'PY' || log "  compose failed (non-fatal)"
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

idx = Path("figures/interpret_21cohort/index.tsv")
if not idx.exists():
    raise SystemExit(0)
df = pd.read_csv(idx, sep="\t")
if df.empty:
    raise SystemExit(0)
patients = df["patient"].unique().tolist()
sigs = df["sig_idx"].unique().tolist()
fig, axes = plt.subplots(len(patients), len(sigs),
                         figsize=(2.0 * len(sigs), 2.0 * len(patients)),
                         squeeze=False)
for i, p in enumerate(patients):
    for j, s in enumerate(sigs):
        row = df[(df.patient == p) & (df.sig_idx == s)]
        ax = axes[i][j]
        ax.set_xticks([]); ax.set_yticks([])
        if row.empty:
            ax.set_facecolor("#222"); continue
        h = np.load(row.iloc[0]["heatmap"])
        ax.imshow(h, cmap="magma", vmin=0, vmax=1)
        if i == 0: ax.set_title(s, fontsize=8)
        if j == 0: ax.set_ylabel(p[-8:], fontsize=7)
fig.suptitle("attention rollout (21-cohort fold0) — heatmap only; raw tiles unavailable",
             fontsize=9)
fig.tight_layout()
fig.savefig("figures/fig4_interpret_attention.png", dpi=150, bbox_inches="tight")
PY
log "compose step done"
