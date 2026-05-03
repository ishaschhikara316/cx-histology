# cx-histology — methods

## Data

TCGA pan-cancer H&E (open-access diagnostic WSIs from GDC:
`data_type=Slide Image`, `experimental_strategy=Diagnostic Slide`,
`data_format=SVS`). CPTAC (BRCA/LUAD/OV/COAD/GBM via TCIA, planned).
Labels: CX1–CX17 activities from Drews et al. 2022 (Nature),
reused for TCGA from `cin-ici-mediation/results/tcga/cx_activities.tsv`
(7,309 patients), recomputed via `CINSignatureQuantification` on PDC
ABSOLUTE calls for CPTAC.

### Active signature scoping

Empirically, four CX signatures are dead in TCGA:

| Signature | Nonzero patients | Zero % |
|-----------|------------------|--------|
| CX7       | 0 / 7,309        | 100.0% |
| CX15      | 0 / 7,309        | 100.0% |
| CX14      | 6 / 7,309        | 99.9%  |
| CX13      | 509 / 7,309      | 93.0%  |

Per-fold an active-signature mask is computed from training and validation
label variance (threshold 1e-5). Signatures with constant labels in a fold
are excluded from the loss and from the Spearman ρ metric. Reported numbers
are over the active subset (typically 13 of 17).

## Tile extraction

20× (`mpp = 0.5 µm/px`), 256 × 256 patches. Tissue mask via Otsu on the
HSV-V channel followed by morphological open/close. QC filters:
tissue fraction ≥ 0.5, Laplacian variance ≥ 100. Macenko stain
normalisation (Macenko et al., ISBI 2009) to a fixed reference in
`resources/stain_reference.png`. 1,000 tiles per slide, stratified random.
Implementation: `workflow/scripts/tile_wsi.py`. The Macenko normaliser is
built and fit **once per process** and cached; transform is per-tile.
This delivers a roughly 700× speedup vs the naive per-tile fit and is
why end-to-end ingest is feasible on a single 4-GPU box.

After embedding, slides with fewer than 20 retained tiles are dropped
from the training manifest (`build_train_manifest.py --min-tiles 20`).

## Foundation models

- **UNI** (`MahmoodLab/UNI`, ViT-L/16, d=1024) — Chen et al., Nature
  Medicine 2024. Primary encoder.
- **H-optimus-0** (`bioptimus/H-optimus-0`, ViT-g/14, d=1536) —
  comparator (planned).

Both frozen, fp16 inference via `workflow/scripts/embed_tiles.py`. No
fine-tuning. UNI access requires an HF token at `~/.hf_token`.

## ABMIL head

Gated attention multiple instance learning (Ilse et al., ICML 2018) with
a multi-task regression head over 17 CX outputs. Only trainable
component. Hyperparameter sweep used in the 7-cohort run:

| Param | Values |
|-------|--------|
| hidden | {128, 256, 512} |
| dropout | {0.3} (broader sweep planned) |
| AdamW lr | {1e-4, 5e-4} |
| weight decay | 1e-5 |
| epochs | 40 |
| early-stop patience | 6 |

Implementation: `workflow/scripts/abmil.py`. Best HP from the 7-cohort
sweep was **hidden = 512, dropout = 0.3, lr = 1e-4** (mean active ρ +0.219).

## Loss and label transform

Multi-task Huber (δ = 1.0) with a learnable per-signature scale and a
per-fold active-signature mask. Inactive signatures contribute zero to
the loss. Labels transformed per fold by `log1p` + z-score fit on the
training split; predictions inverted before metric reporting.
Implementation: `workflow/scripts/loss.py`.

## Training

Cancer-type-stratified 5-fold CV on TCGA. Single pan-cancer head, not
one head per cancer. Implementation: `workflow/scripts/train_cv.py`.
Parallel HP sweep across 4 GPUs via slot-based bash queue in
`plan_a_orchestrator.sh`.

## External validation (planned)

Full CPTAC inference with the best fold-0 checkpoint. Primary metric:
Spearman ρ per signature. H2 target: |Δρ(TCGA − CPTAC)| ≤ 0.15 on
primary signatures (CX1, CX3, CX12). Implementation:
`workflow/scripts/predict_external.py`.

## Interpretability (planned)

1. Per-tile attention heat-maps (`attention_rollout.py`).
2. Wilcoxon rank-sum enrichment of nuclear morphometrics (nucleus count,
   density, area, eccentricity, intensity variance, stromal fraction)
   between high- and low-attention tiles (`nuclear_features.py`).
3. Per-signature top/bottom-K prototype tile atlases
   (`prototype_tiles.py`).

Excluded from v1: formal pathologist review, spatial proteomics,
gradient-based saliency.

## Statistical reporting

Per signature, the primary metric is Spearman ρ (mean across 5 folds).
Secondary: MAE, Pearson r, R². For the best HP we additionally report
sign-flip permutation p-values per signature, BH-adjusted across active
signatures, and bootstrap 95% CIs from per-fold ρ resampling
(n_boot = 5000). With 5 folds the sign-flip permutation has a floor
p ≈ 2/32 ≈ 0.063, so q < 0.05 is unreachable in this design even with
all-positive folds. Bootstrap CIs that exclude zero are the more
informative statistic at this fold count. Implementation:
`workflow/scripts/robustness_analysis.py`.

## Ingest pipeline

Stream mode: per-slide download (gdc-client) → tile + filter →
fp16 embed via UNI → write `.h5` of embeddings → delete raw `.svs`
and tile cache. Peak disk per-slide ≈ size of one slide; a whole
cohort's final footprint is ~300 MB of embeddings. Implementation:
`workflow/scripts/stream_ingest.py` and `stream_ingest_parallel.py`.
For pan-cohort runs, `parallel_ingest_8cohort.sh` runs two cohorts
concurrently across two GPU pairs.

`stream_ingest_parallel.py` shards the manifest round-robin across
workers and pins each worker to one physical GPU via a
`CUDA_VISIBLE_DEVICES`-aware indexing scheme that respects any
restriction the parent shell already imposed.

## Compute

This snapshot was produced on **4 × RTX 4090** (24 GB each, 96 GB
aggregate), AMD EPYC 7B13 (122 cores), 396 GB RAM, 126 GB SSD.
Sustained download from GDC: 70–85 MB/s per connection, ~150–280 MB/s
aggregate with 4 workers. Pan-TCGA ingest is bandwidth-feasible in 3–5
days on this hardware in stream mode without an NVMe rental.

## Reproducibility

- `seed = 20260419` across numpy, torch, and `StratifiedKFold`.
- `torch.use_deterministic_algorithms(True)`, `cudnn.benchmark = False`
  set by `_common.seed_everything`.
- All hyperparameters in `config/models.yaml` and `config/params.yaml`.
- Software pinned in `environment.yml` (conda + pip combined).
- Snakemake 8 pipeline from clean checkout to final figures.

## References

- Drews et al., 2022. *A pan-cancer compendium of chromosomal instability.* Nature.
- Ilse et al., 2018. *Attention-based Deep Multiple Instance Learning.* ICML.
- Chen et al., 2024. *Towards a general-purpose foundation model for computational pathology.* Nature Medicine.
- Macenko et al., 2009. *A method for normalizing histology slides for quantitative analysis.* ISBI.
- Coudray et al., 2018. *Classification and mutation prediction from non-small cell lung cancer histopathology images using deep learning.* Nature Medicine.
- Kather et al., 2019. *Deep learning can predict microsatellite instability directly from histology in gastrointestinal cancer.* Nature Medicine.
- Fu et al., 2020. *Pan-cancer computational histopathology reveals mutations, tumor composition and prognosis.* Nature Cancer.
