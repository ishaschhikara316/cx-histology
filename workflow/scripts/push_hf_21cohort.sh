#!/bin/bash
# Build cx-uni-abmil-21cohort HF repo from the just-finished 21-cohort retrain
# and push to huggingface.co/IshaChhikara/cx-uni-abmil-21cohort.

ROOT="/root/ISHA/cx-histology"
cd "$ROOT"

# Activate conda env (cx-histology) and force its libstdc++ ahead of system.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate cx-histology
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
set -u

STAGE=/tmp/hf_repos/cx-uni-abmil-21cohort
mkdir -p "$STAGE"

# Reuse the modeling code + LICENSE + inference example from the 7-cohort scaffold
cp /tmp/hf_repos/cx-uni-abmil-7cohort/modeling_cx_abmil.py "$STAGE/"
cp /tmp/hf_repos/cx-uni-abmil-7cohort/LICENSE "$STAGE/"
sed 's|cx-uni-abmil-7cohort|cx-uni-abmil-21cohort|g' \
  /tmp/hf_repos/cx-uni-abmil-7cohort/inference_example.py > "$STAGE/inference_example.py"

# Copy 21-cohort fold checkpoints
cp results/cv/uni_21cohort_final/fold0.pt "$STAGE/"
cp results/cv/uni_21cohort_final/fold1.pt "$STAGE/"
cp results/cv/uni_21cohort_final/fold2.pt "$STAGE/"
cp results/cv/uni_21cohort_final/fold3.pt "$STAGE/"
cp results/cv/uni_21cohort_final/fold4.pt "$STAGE/"

# Build config.json + README.md from cv_summary.json
python <<'PY'
import json, numpy as np
from pathlib import Path

D21 = Path("results/cv/uni_21cohort_final")
STAGE = Path("/tmp/hf_repos/cx-uni-abmil-21cohort")

cv = json.load(open(D21 / "cv_summary.json"))
mean = np.array(cv["pan_cancer_spearman_mean"], dtype=float)
std  = np.array(cv["pan_cancer_spearman_std"],  dtype=float)
sigs = [f"CX{i}" for i in range(1, 18)]
n_h1 = int(np.sum(mean >= 0.30))

# config.json
config = {
    "architecture": "GatedABMIL",
    "input_encoder": "MahmoodLab/UNI",
    "input_encoder_dim": 1024,
    "hidden": 512,
    "trunk_hidden": 512,
    "dropout": 0.1,
    "lr": 0.0001,
    "n_out": 17,
    "tile_resolution": 224,
    "tile_mpp": 0.5,
    "training_cohorts": [
        "TCGA-DLBC","TCGA-CHOL","TCGA-UVM","TCGA-READ","TCGA-MESO","TCGA-UCS","TCGA-KICH",
        "TCGA-ESCA","TCGA-CESC","TCGA-PAAD","TCGA-THYM","TCGA-COAD","TCGA-PCPG","TCGA-KIRP","TCGA-STAD",
        "TCGA-BRCA","TCGA-LGG","TCGA-GBM","TCGA-LUAD","TCGA-LUSC","TCGA-HNSC",
    ],
    "n_folds": 5,
    "stratify_by": "cancer_type",
    "seed": 20260419,
    "selected_via_hp_sweep": True,
    "hp_source": "uni_15cohort_sweep best (h512_d0.1_lr1e-4 +0.138 mean rho)",
    "cv_summary": {
        "mean_active_rho_fold_level": float(np.nanmean(mean)),
        "n_h1_hits": n_h1,
        "per_signature_rho": {sigs[i]: float(mean[i]) if not np.isnan(mean[i]) else None
                              for i in range(17)},
    },
    "inactive_signatures_tcga": ["CX7", "CX13", "CX14", "CX15"],
}
with open(STAGE / "config.json", "w") as f:
    json.dump(config, f, indent=2)

# README.md
top5 = sorted([(sigs[i], mean[i]) for i in range(17) if not np.isnan(mean[i])],
              key=lambda x: -x[1])[:5]
top5_str = ", ".join(f"{s}: {v:+.3f}" for s, v in top5)

readme = f"""---
license: mit
library_name: pytorch
tags:
  - medical-imaging
  - histopathology
  - whole-slide-image
  - chromosomal-instability
  - copy-number-signatures
  - attention-mil
pipeline_tag: image-classification
language:
  - en
---

# cx-uni-abmil-21cohort

A Gated Attention-MIL head that predicts the 17 Drews 2022 chromosomal-instability
signatures (CX1–CX17) from whole-slide H&E pathology images, using frozen
[`MahmoodLab/UNI`](https://huggingface.co/MahmoodLab/UNI) tile embeddings.

This is the **largest** of the three cohort variants in the cx-histology series:
21 TCGA cohorts, fold-parallel trained across 5 GPUs. It supersedes
[`cx-uni-abmil-15cohort`](https://huggingface.co/IshaChhikara/cx-uni-abmil-15cohort)
and [`cx-uni-abmil-7cohort`](https://huggingface.co/IshaChhikara/cx-uni-abmil-7cohort)
for the most cohort-diverse training distribution.

## Headline result

- **Mean active ρ = {np.nanmean(mean):+.3f}** across 13 active signatures
- **{n_h1} H1 hits** (ρ ≥ 0.30)
- Top 5: {top5_str}
- Trained on the 21-cohort superset (15-cohort cohorts + BRCA/LGG/GBM/LUAD/LUSC/HNSC)
- Hyperparameters from the n=1,454 sweep winner: hidden=512, dropout=0.1, lr=1e-4

## Architecture

```
WSI -> tile (224 px @ 0.5 mpp, Macenko-normalized)
    -> UNI (frozen) -> (N tiles, 1024)
    -> GatedABMIL (this model) -> (17,) CX activity prediction
```

## Quick start

```python
from huggingface_hub import snapshot_download
from modeling_cx_abmil import CXEnsemble
local = snapshot_download("IshaChhikara/cx-uni-abmil-21cohort")
model = CXEnsemble.from_pretrained(local, device="cpu")
cx = model.predict(my_uni_embeddings)  # (B, N, 1024) -> (B, 17)
```

## The cohort-scaling story (read together with the other two repos)

| Variant | n | Mean ρ | H1 hits |
|---|---|---|---|
| 7-cohort  | 289   | +0.219 | 3 |
| 15-cohort | 1,454 | +0.138 | 0 |
| 21-cohort | larger | {np.nanmean(mean):+.3f} | {n_h1} |

Smaller cohort sets achieve larger pan-cohort ρ partly because cancer-type
heterogeneity is doing the work. As cohorts are added, the within-biology
signal becomes the dominant explanation. The 21-cohort number is the most
honest estimate of the H&E→CIN biology link.

## Files

- `modeling_cx_abmil.py` — model + `CXEnsemble` 5-fold ensemble wrapper
- `inference_example.py`
- `config.json`
- `fold0.pt` … `fold4.pt`
- `LICENSE` (MIT)

## Citation

In preparation. Code: github.com/ishaschhikara316/cx-histology.
"""
(STAGE / "README.md").write_text(readme)
print(f"wrote config.json and README.md (rho={np.nanmean(mean):+.3f}, h1={n_h1})")
PY

# Push to HF
python <<'PY'
from huggingface_hub import HfApi, create_repo
token = open('/root/.hf_token_isha').read().strip()
api = HfApi(token=token)
create_repo('IshaChhikara/cx-uni-abmil-21cohort', token=token,
            repo_type='model', private=False, exist_ok=True)
api.upload_folder(repo_id='IshaChhikara/cx-uni-abmil-21cohort',
                  folder_path='/tmp/hf_repos/cx-uni-abmil-21cohort',
                  commit_message='initial release: 21-cohort fold-parallel UNI+ABMIL ensemble')
print("HF push: https://huggingface.co/IshaChhikara/cx-uni-abmil-21cohort")
PY
