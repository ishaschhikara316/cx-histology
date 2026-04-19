# cx-histology — methods

## Data
TCGA pan-cancer H&E (~30k diagnostic WSIs, 33 types; GDC data_type=SlideImage,
experimental_strategy=DiagnosticSlide, data_format=SVS). CPTAC (~2k WSIs,
BRCA/LUAD/OV/COAD/GBM; TCIA). Labels: CX1–CX17 activities from Drews 2022
reused from `cin-ici-mediation/results/tcga/cx_activities.tsv` (TCGA) or
recomputed via `CINSignatureQuantification` on PDC ABSOLUTE calls (CPTAC).

## Tile extraction
20× (mpp=0.5 µm/px), 256×256, Otsu on HSV-V + morphological open/close,
tissue fraction ≥ 0.5, Laplacian variance ≥ 100, Macenko stain
normalisation to a fixed reference in `resources/stain_reference.png`,
1000 tiles per slide stratified random. Implemented in
`workflow/scripts/tile_wsi.py`.

## Foundation models
- **UNI** (`MahmoodLab/UNI`, ViT-L/16, d=1024) — baseline.
- **H-optimus-0** (`bioptimus/H-optimus-0`, ViT-g/14, d=1536) — comparator.

Both frozen, fp16 inference via `workflow/scripts/embed_tiles.py`. No
fine-tuning (VRAM and time constrained).

## ABMIL head
Gated attention-MIL (Ilse et al., ICML 2018) with multi-task regression
to 17 CX outputs. Only trainable component. Hyperparameters swept:
hidden ∈ {128, 256, 512}, dropout ∈ {0.2, 0.3, 0.5}, AdamW lr ∈
{1e-4, 5e-4}, weight-decay 1e-5, 40 epochs with patience 6. Implemented
in `workflow/scripts/abmil.py`.

## Loss and label transform
Multi-task Huber (δ=1.0) with optional per-signature scaling. Labels
transformed per fold by log1p + z-score fit on the training split;
predictions inverted before metric reporting. Implemented in
`workflow/scripts/loss.py`.

## Training
Cancer-type-stratified 5-fold CV on TCGA; pan-cancer head, not
per-cancer. 20 % patient hold-out lock-box preserved across all folds.
Implemented in `workflow/scripts/train_cv.py`.

## External validation
Full CPTAC inference with the fold-0 checkpoint at the sweep-best HP.
Primary metric: Spearman ρ per signature. H2 target: |Δρ(TCGA − CPTAC)|
≤ 0.15 on primary signatures (CX1, CX4). Implemented in
`workflow/scripts/predict_external.py`.

## Interpretability
1. Per-tile attention heat-maps (`attention_rollout.py`).
2. Wilcoxon rank-sum enrichment of nuclear morphometrics (nucleus count,
   density, area, eccentricity, intensity variance, stromal fraction)
   between high- and low-attention tiles (`nuclear_features.py`).
3. Per-signature top/bottom-K prototype tile atlases
   (`prototype_tiles.py`).

Excluded from v1: formal pathologist review, spatial proteomics,
gradient-based saliency.

## Statistical reporting
Per signature: Spearman ρ (primary), MAE, Pearson r, R². BH-adjusted
p-values where applicable. Bootstrap 95% CIs (1000 replicates) in
figures that report ρ.

## Compute
Target rental: 4× RTX 5060 Ti on Vast.ai (64 GB aggregate VRAM, 96 CPU
cores, 1 TB RAM, 2.6 TB NVMe). Budget $50–60 ≈ 85 wall-clock hours.

## Reproducibility
- `seed = 20260419` across numpy, torch, and `StratifiedKFold`.
- `torch.use_deterministic_algorithms(True)`, `cudnn.benchmark = False`
  set by `_common.seed_everything`.
- All hyperparameters in `config/models.yaml` and `config/params.yaml`.
- Software pinned in `environment.yml` (conda + pip combined).
- Snakemake 8 pipeline from clean checkout to final figures.
