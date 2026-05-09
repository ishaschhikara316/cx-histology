#!/bin/bash
# Wait for the full 11-combo sweep to finish, then if anything beats the
# current 15cohort HF leader (rho=+0.138), re-upload the better weights.

set -u
ROOT="/root/ISHA/cx-histology"
cd "$ROOT"
export PYTHONPATH=/root/ISHA/cx-histology

mkdir -p logs
LOG="logs/sweep_finalize_watcher.log"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

EXPECTED=(
  "h128_d0.1_lr0.0001"
  "h128_d0.3_lr0.0001"
  "h128_d0.3_lr0.0005"
  "h128_d0.5_lr0.0001"
  "h256_d0.1_lr0.0001"
  "h256_d0.3_lr0.0001"
  "h256_d0.3_lr0.0005"
  "h256_d0.5_lr0.0001"
  "h512_d0.1_lr0.0001"
  "h512_d0.3_lr0.0005"
  "h512_d0.5_lr0.0001"
)

log "watching for ${#EXPECTED[@]} cv_summary.json files to land"
while true; do
  ready=0
  for c in "${EXPECTED[@]}"; do
    [ -f "results/cv/uni_15cohort_sweep/$c/cv_summary.json" ] && ready=$((ready+1))
  done
  if [ "$ready" -eq "${#EXPECTED[@]}" ]; then
    log "all $ready/${#EXPECTED[@]} sweep combos done"
    break
  fi
  sleep 60
done

# Build sweep summary, identify best
python - <<'PY' | tee -a "$LOG"
import json, glob, numpy as np, os
rows = []
# Include the official h512_d0.3_lr1e-4 from uni_15cohort_final
official = "results/cv/uni_15cohort_final/cv_summary.json"
if os.path.exists(official):
    s = json.load(open(official))
    m = np.array(s["pan_cancer_spearman_mean"], dtype=float)
    rows.append({"hp": "h512_d0.3_lr0.0001", "src": "uni_15cohort_final",
                 "mean_active_rho": float(np.nanmean(m)),
                 "n_h1": int(np.sum(m >= 0.30) if not np.all(np.isnan(m)) else 0)})
for f in sorted(glob.glob("results/cv/uni_15cohort_sweep/*/cv_summary.json")):
    s = json.load(open(f))
    m = np.array(s["pan_cancer_spearman_mean"], dtype=float)
    hp = os.path.basename(os.path.dirname(f))
    rows.append({"hp": hp, "src": "uni_15cohort_sweep",
                 "mean_active_rho": float(np.nanmean(m)),
                 "n_h1": int(np.sum(m >= 0.30) if not np.all(np.isnan(m)) else 0)})
rows.sort(key=lambda r: -r["mean_active_rho"])
print("Final 15-cohort HP sweep leaderboard:")
for r in rows:
    print(f"  {r['hp']:<26} {r['src']:<22} rho={r['mean_active_rho']:+.4f} H1={r['n_h1']}")
with open("results/cv/uni_15cohort_sweep/sweep_summary.json", "w") as fh:
    json.dump({"rows": rows, "best": rows[0]}, fh, indent=2)
print(f"\nbest: {rows[0]['hp']} {rows[0]['src']} rho={rows[0]['mean_active_rho']:+.4f}")
PY

# Determine best and decide whether to update HF
BEST=$(python -c "import json; s=json.load(open('results/cv/uni_15cohort_sweep/sweep_summary.json')); print(s['best']['hp'], s['best']['src'], s['best']['mean_active_rho'])")
read -r BEST_HP BEST_SRC BEST_RHO <<< "$BEST"
log "best overall: $BEST_HP from $BEST_SRC at rho=$BEST_RHO"

# Current HF leader is h512_d0.1_lr0.0001 at +0.138 (uploaded just now).
# Only re-upload if a different combo wins by >+0.001 (avoid noise re-uploads).
CURRENT_HF_RHO=0.138
NEEDS_UPDATE=$(python -c "print('yes' if (float('$BEST_RHO') > float('$CURRENT_HF_RHO') + 1e-3 and '$BEST_HP' != 'h512_d0.1_lr0.0001') else 'no')")
log "needs HF update: $NEEDS_UPDATE  (current HF leader=h512_d0.1_lr0.0001 rho=$CURRENT_HF_RHO; new winner=$BEST_HP rho=$BEST_RHO)"

if [ "$NEEDS_UPDATE" = "yes" ]; then
  if [ "$BEST_SRC" = "uni_15cohort_sweep" ]; then
    NEW_DIR="results/cv/uni_15cohort_sweep/$BEST_HP"
  else
    NEW_DIR="results/cv/$BEST_SRC"
  fi
  log "copying $NEW_DIR/fold*.pt to HF staging"
  cp $NEW_DIR/fold0.pt $NEW_DIR/fold1.pt $NEW_DIR/fold2.pt $NEW_DIR/fold3.pt $NEW_DIR/fold4.pt /tmp/hf_repos/cx-uni-abmil-15cohort/
  python <<PY
from huggingface_hub import HfApi
import json
token = open('/root/.hf_token_isha').read().strip()
api = HfApi(token=token)
# Update config.json with new HP info
cfg_path = '/tmp/hf_repos/cx-uni-abmil-15cohort/config.json'
cfg = json.load(open(cfg_path))
hp = '$BEST_HP'  # e.g. "h128_d0.1_lr0.0001"
parts = hp.split('_')
cfg['hidden'] = int(parts[0][1:])
cfg['dropout'] = float(parts[1][1:])
cfg['lr'] = float(parts[2][2:])
cfg['cv_summary']['mean_active_rho_fold_level'] = float($BEST_RHO)
cfg['cv_summary']['comment'] = f"Final HP-sweep winner: {hp} at rho={float('$BEST_RHO'):+.4f}"
json.dump(cfg, open(cfg_path, 'w'), indent=2)
api.upload_folder(repo_id='IshaChhikara/cx-uni-abmil-15cohort',
                  folder_path='/tmp/hf_repos/cx-uni-abmil-15cohort',
                  commit_message=f'update: post-sweep best HP ({hp}) rho={float("$BEST_RHO"):+.4f}')
print("HF re-uploaded with $BEST_HP weights")
PY
fi

# Commit final sweep summary to GitHub
log "committing sweep summary to git"
git add results/cv/uni_15cohort_sweep/sweep_summary.json \
        results/cv/uni_15cohort_sweep/*/cv_summary.json 2>>"$LOG"
git -c user.email='isha.s.chhikara316@gmail.com' -c user.name='Isha Chhikara' commit -m "results: 15-cohort HP sweep complete; best=$BEST_HP rho=$BEST_RHO" 2>>"$LOG" || true
git push origin main 2>>"$LOG" || true
log "=== sweep finalize complete ==="
