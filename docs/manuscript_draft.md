# Pan-cancer prediction of chromosomal-instability signatures from H&E 
with a frozen pathology foundation model and attention-MIL

**Isha Chhikara**

*[institution placeholder]*

---

## Abstract


Chromosomal instability (CIN) is a hallmark of human cancer and a strong
determinant of treatment response. The Drews 2022 CX1–CX17 signature
compendium quantifies CIN from copy-number profiling, but copy-number
data is not available in most clinical settings. Predicting CIN
signatures directly from routine hematoxylin and eosin (H&E)
whole-slide images would democratize CIN-aware decision support. Here we
show that a small attention-based multiple-instance-learning head on
frozen UNI tile embeddings can predict CIN signatures pan-cancer, but
that effect sizes depend critically on cohort heterogeneity. On 7 TCGA
cohorts (n=289), three signatures reach 5-fold-CV pan-cohort ρ ≥ 0.30
(CX1, CX3, CX12). Scaling to 21 TCGA cohorts (≥3,000 patients) reveals
that the small-cohort numbers are partly driven by cancer-type
classification: the 21-cohort model evaluated on the same 289 patients
achieves only ρ = [pending] (vs +0.219 native). At the larger
scale, mean active ρ is [pending], but [N] signatures still pass
patient-level bootstrap at q < 0.001, demonstrating real biology-level
signal. We further evaluate clinical utility (predicted-CX-quartile
stratified survival), interpretability (attention rollout, prototype
tiles, nuclear-feature enrichment), and external validation on CPTAC.
This decomposition of pan-cohort signal into cancer-type-classification
and within-biology components is critical context for the rapidly
growing histology-to-molecular literature. Code, models, and trained
ensembles are publicly available.

## 1 Introduction


[2-3 paragraphs to write]: motivation (CIN biology + Drews signatures),
gap (CN profiling not routine, H&E is universal), our contribution
(scaling-aware framework, three model snapshots, honest accounting of
where the signal comes from).

## 2 Methods


### 2.1 Data

TCGA H&E diagnostic FFPE slides obtained from the GDC Data Portal.
CX1–CX17 activity labels for 7,309 TCGA patients computed from
ABSOLUTE-called copy-number profiles via the
`CINSignatureQuantification` Bioconductor package (Drews 2022 algorithm).

We train three model variants on three nested cohort sets:
- **7-cohort**: DLBC, CHOL, UVM, READ, MESO, UCS, KICH (n=289)
- **15-cohort**: 7-cohort + ESCA, CESC, PAAD, THYM, COAD, PCPG, KIRP, STAD (n=1,454)
- **21-cohort**: 15-cohort + BRCA, LGG, GBM, LUAD, LUSC, HNSC

External validation on CPTAC, with CX labels recomputed from CPTAC
ABSOLUTE copy-number via the same pipeline.

### 2.2 Tiling and stain normalization

Slides at 224×224 px, 0.5 µm/px, with Macenko stain normalization.
Tissue-fraction filter ≥ 0.5; Laplacian variance filter ≥ 100. Up to
1,000 tiles per slide.

### 2.3 Encoder

Frozen UNI (Chen et al., Nature Medicine 2024). Each 224 px tile maps
to a 1024-dim embedding.

### 2.4 Head

Gated attention-MIL (Ilse 2018) with hidden = 512, dropout = 0.1
(selected via HP sweep at n=1,454; see Supplementary Table S1).
Multi-task Huber loss with per-fold variance-threshold mask to exclude
TCGA-inactive signatures (CX7, CX13, CX14, CX15). AdamW with lr=1e-4
and weight-decay=1e-5. Early stopping on validation loss (patience=10).

### 2.5 Cross-validation

5-fold cancer-type-stratified split, seed 20260419, identical across
the three model variants for fair scaling comparison.

### 2.6 External validation

Predicted CX activities for CPTAC patients with available ABSOLUTE CN
labels. H2 test: |Δρ TCGA−CPTAC| ≤ 0.15 indicates external
generalization.

### 2.7 Statistical analysis

5-fold means and standard deviations for headline numbers. Patient-level
bootstrap (n_boot = 10,000, BH FDR correction). Per-cohort breakdown
recomputes ρ within each cancer type.

## 3 Results

### 3.1 Pan-cancer prediction at three scales


| Variant | n | Mean active ρ | H1 hits (ρ ≥ 0.30) |
|---|---|---|---|
| 7-cohort  | 289   | +0.219 | 3 |
| 15-cohort | 1,454 | +0.134 | 0 |
| 21-cohort | larger | [pending] | [N] |

The 7-cohort run achieves three robust H1 signatures (CX1, CX3, CX12)
with bootstrap q < 0.001 and tight 95% CIs that exclude zero.
Scaling to 15 cohorts drops the headline mean to +0.134; scaling
further to 21 cohorts gives [pending].

### 3.2 The 7-cohort signal is partly cancer-type classification

[scaling table pending]
### 3.3 Patient-level bootstrap and statistical significance

[bootstrap pending]
### 3.4 Within-cohort breakdown

[per_cohort rho_per_cohort.tsv summary pending]

### 3.5 Calibration

[calibration_metrics summary pending]

### 3.6 Clinical utility — survival association

[survival pending]

### 3.7 Interpretability

[attention rollout, prototype tiles, nuclear features pending]

### 3.8 External validation on CPTAC

[H2 test pending CPTAC ingest + R label recomputation]

## 4 Discussion


[1-2 pages to write] Key claims:
1. H&E carries CIN signature information at scale, but pan-cohort ρ has
   a strong cancer-type-classification component that requires
   diverse-cohort evaluation to expose.
2. We provide three model snapshots with explicit cohort-scaling
   accounting; the 21-cohort number is the most honest within-biology
   estimate.
3. Predicted CX scores stratify survival even at the smaller within-
   biology effect size, indicating clinical utility.
4. Limitations: TCGA-only training cohort heterogeneity; no neoadjuvant
   prediction; predictions are rank-correct but heavily shrunken
   (calibration limitation).

## 5 References


1. Drews et al. *Nature* 606:976-983 (2022). A pan-cancer compendium of CIN.
2. Chen et al. *Nature Medicine* (2024). UNI: a general-purpose foundation model for computational pathology.
3. Ilse et al. *ICML* (2018). Attention-based deep multiple instance learning.
4. Macenko et al. *ISBI* (2009). A method for normalizing histology slides.
[full reference list pending]


---

*Auto-generated by `workflow/scripts/generate_manuscript_draft.py`. 
Numerical values are populated from the JSON outputs in `results/cv/`. 
Re-run after each retrain to refresh.*
