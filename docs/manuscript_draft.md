# Pan-cancer prediction of chromosomal-instability signatures from H&E 
with a frozen pathology foundation model and attention-MIL

**Isha Chhikara**

*[institution placeholder]*

---

## Abstract


Chromosomal instability (CIN) signatures (CX1 through CX17) capture
mechanistically distinct copy-number processes and have been shown to
predict survival and treatment response, but the ABSOLUTE-quality
copy-number inputs they require are unavailable in routine clinical
practice. We trained a gated attention multiple-instance-learning head
on frozen UNI tile embeddings to predict CX activities directly from
hematoxylin and eosin (H&E) whole-slide images, across three nested
sets of TCGA cohorts (7, 15, and 21 cohorts). On the 7-cohort training
set (n = 289 patients with rare and CIN-homogeneous tumors), three
signatures reach pooled 5-fold-CV ρ ≥ 0.30 (CX1, CX3, CX12). On the
21-cohort superset (n = 3,855 patients), 9 of 14 active signatures
pass patient-level bootstrap at q < 0.001 and predicted CX1 quartiles
stratify overall survival at a cohort-stratified log-rank
p = 5.4 × 10⁻⁵. Re-evaluating the 21-cohort model on the same 289
patients used to train the 7-cohort model gives ρ = +0.116, lower
than the 7-cohort native +0.219, so a portion of the small-cohort
effect derives from cancer-type classification while a substantial
within-biology component survives at scale. External validation on CPTAC (n = 283 patients with paired
predictions and CINSignatureQuantification labels) meets the
pre-registered acceptance criterion |ρ_TCGA - ρ_CPTAC| ≤ 0.15 for
all 14 active signatures. We release all three trained models, the
cross-validated predictions, and the analysis pipeline publicly.
This decomposition of pan-cohort signal into cancer-type-classification
and within-biology components is critical context for the rapidly
growing histology-to-molecular literature. Code, models, and trained
ensembles are publicly available.

## 1 Introduction

Chromosomal instability (CIN), the persistent mis-segregation of whole
chromosomes or chromosomal segments during cell division, is observed
in the great majority of solid tumors and contributes to intra-tumor
heterogeneity, metastatic competence, and therapeutic resistance.1
The phenotype had long been treated as a single property of the
tumor, unstable or stable, rather than as a set of distinct
mutational processes. This changed when Drews and colleagues
decomposed pan-cancer CIN into 17 copy-number signatures (CX1
through CX17) by mixture modeling of features extracted from
ABSOLUTE-called whole-genome copy-number profiles.1 The CX
signatures recapitulate known biology, including replication stress
(CX1), impaired homologous recombination (CX3), and chromothripsis
(CX12), and several have been shown to predict survival or
treatment response.

The CX framework has not, however, been broadly deployed in
clinical or translational settings, mainly because its required
inputs are unavailable in routine practice. ABSOLUTE-quality
copy-number profiles depend on whole-exome or whole-genome
sequencing with carefully calibrated purity and ploidy estimation,
capabilities that are not part of standard diagnostic workups for
most patients. Hematoxylin and eosin (H&E) stained whole-slide
images, by contrast, are produced for nearly every solid tumor
diagnosis, and recent work has shown that diverse genomic and
transcriptomic readouts can be inferred from H&E.2,6,7 We asked
whether the CX activities themselves can be predicted directly
from H&E across a broad pan-cancer population, and whether the
effect sizes obtained at small-cohort scale generalize to large,
heterogeneous training sets.

We trained a gated attention multiple-instance-learning head3 on
frozen UNI tile embeddings2 to predict CX1 through CX17 activities
from H&E across three nested sets of TCGA cohorts (7, 15, and 21
cohorts). On the 21-cohort superset of 3,855 patients with paired
CX-activity labels, 9 of 14 TCGA-active signatures pass a
patient-level bootstrap at q < 0.001, and predicted CX1 quartiles
stratify overall survival at a cohort-stratified log-rank
p = 5.4 × 10⁻⁵. The 7-cohort model, trained on rare-tumor
populations (DLBC, CHOL, UVM, READ, MESO, UCS, KICH), reaches the
higher H1 threshold of pooled ρ ≥ 0.30 for CX1, CX3, and CX12.
The nested cohort design lets us quantify how much of the
pan-cohort correlation reflects cancer-type classification and how
much reflects within-biology signal. All three trained models, the
cross-validated predictions, and the analysis pipeline are
publicly released.

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
| 21-cohort | 3,855 | +0.106 | 0 |

The 7-cohort run achieves three H1 signatures (CX1, CX3, CX12) with
pooled ρ ≥ 0.30, bootstrap q < 0.001, and tight 95 % CIs that exclude
zero. Scaling to 15 cohorts drops the mean active ρ to +0.134; scaling
further to 21 cohorts gives +0.106. The five strongest signatures at
21-cohort scale are CX5 (ρ = +0.147), CX1 (+0.137), CX12 (+0.130),
CX3 (+0.114), and CX10 (+0.101). All five 95 % bootstrap CIs exclude
zero on 3,855 patients. The cohort-scaling regression is monotonic
across the three scales (Figure 2A).

### 3.2 The 7-cohort signal is partly cancer-type classification


We re-evaluated the 21-cohort model on the same 289 patients used for
7-cohort training:

| Setting | Mean ρ | n |
|---|---|---|
| 7-cohort native model on its own patients | +0.219 | 289 |
| 21-cohort model on the 7-cohort subset | +0.116 | 289 |
| 21-cohort model on the 15-cohort subset | +0.073 | 1454 |
| 21-cohort native | +0.106 | (full) |

The same 289 patients receive lower predictions under the 21-cohort
model. This is consistent with the multi-task learning objective
trading per-cohort fit for cross-cohort coverage. Mechanistically, the
larger cohorts (STAD, CESC, COAD, BRCA) pull the shared feature
representation toward their CIN-burden distributions, blunting the
cancer-type-classification shortcut that the small-cohort model
exploited. The per-signature decomposition is shown in Figure 2B:
CX1, CX3, and CX12, the three 7-cohort H1 signatures, attenuate
substantially when the same 289 patients are scored by the
21-cohort model.

### 3.3 Patient-level bootstrap and statistical significance

We resampled the 3,855 21-cohort patients with replacement
(n_boot = 10,000) and recomputed pooled Spearman ρ within each draw.
Of the 14 TCGA-active signatures (CX7, CX13, CX14, and CX15 are
pan-TCGA inactive and are masked from the loss; see Methods), 9 pass
at q_BH < 0.001 and 12 pass at q_BH < 0.05. Permutation p-values for
the top-ranked signatures bottom out at the resolution of the test
(p_perm ≈ 2 × 10⁻⁴). The five strongest signatures by pooled ρ are
CX5 (0.147), CX1 (0.137), CX12 (0.130), CX3 (0.114), and CX10
(0.101), with 95 % bootstrap confidence intervals that exclude zero
in every case.

### 3.4 Within-cohort breakdown

Recomputing ρ within each cancer type isolates the intra-cohort
component of the prediction, since the cancer-type label is constant
inside a cohort and any non-zero ρ must come from intra-cohort
tile-feature signal. The strongest intra-cohort correlations cluster
in the rare and CIN-homogeneous cohorts (UCS: CX12 ρ = 0.236,
CX5 = 0.229, CX1 = 0.204; UVM: CX1 ρ = 0.480 at n = 24; THYM:
CX17 ρ = 0.418 at n = 37). Larger heterogeneous cohorts (BRCA, LUAD,
HNSC) yield small or near-zero per-signature ρ at the 21-cohort
scale. The full per-cohort by per-signature matrix is in
`results/cv/uni_21cohort_final/rho_per_cohort.tsv` and visualised
in `figures/fig_per_cohort_rho.pdf`.

### 3.5 Calibration

Predicted CX activities preserve rank order, while their absolute
scale is shrunken. For CX1, the regression of true on predicted
activity has slope 0.024 and intercept 0.557, and the predicted
range is [0.43, 0.72] against a true range of [0, 0.92], a
4.4-fold contraction. CX3, CX10, and CX12 show the same pattern at
smaller absolute scales. R² is near zero or slightly negative for
several signatures, while Spearman ρ remains in the 0.10 to 0.15
range. Full calibration metrics (Spearman, Pearson, MAE, R²,
predicted range) are in `calibration_metrics.json`.

### 3.6 Clinical utility: survival association

We tested whether predicted CX1 quartiles stratify overall survival
across the 21-cohort population. Patients were split into quartiles
of predicted CX1 from held-out fold predictions, and a
cohort-stratified log-rank test was applied to the resulting
Kaplan-Meier curves. The test rejects the null of identical survival
distributions at p = 5.4 × 10⁻⁵ (n = 3,831 patients with
overall-survival annotation, 1,379 events, median follow-up
1.86 years). Univariate Cox proportional-hazards models on
continuous predicted CX1 give HR = 0.98 (95 % CI 0.93 to 1.04,
p = 0.51), reflecting the dominance of the cohort effect in the
unstratified hazard form. CX10 and CX12 show borderline univariate
trends (p = 0.08 and 0.07 respectively). Survival curves and
per-signature hazard ratios are in `figures/fig_survival_km_cx1.pdf`
and `survival_metrics.json`.

### 3.7 Interpretability

Attention rollout was performed for five representative slides at
five top signatures (CX1, CX3, CX5, CX10, CX12; fold-0 checkpoint).
High-attention regions cluster on epithelial tile groups rather
than on stroma or background. Prototype-tile visualisation and
nuclear-feature enrichment were not run at the 21-cohort scale
because the raw tile pixels were not retained during ingest; both
analyses can be revisited in a follow-up run with a per-slide tile
cache enabled. Heatmaps are in `figures/interpret_21cohort/`.

### 3.8 External validation on CPTAC

CPTAC slide ingest into UNI embeddings produced 1,729 series across
11 cohorts (508 unique patients). Allele-specific copy-number
segments for CPTAC-3 were obtained from the Genomic Data Commons
(1,583 cases; OV is absent from GDC) and CX activities were
recomputed with the CINSignatureQuantification package (v1.2.0,
build hg38, DCIN = 10), yielding signature scores for 1,200 patients
across 12 cohorts. After patient-level matching of predictions to
recomputed labels, 283 patients were available for the H2 test. The
pre-registered acceptance criterion |ρ_TCGA - ρ_CPTAC| ≤ 0.15 was
met for all 14 active signatures, with the largest gap at CX5
(Δρ = 0.125) and the smallest at CX6 (Δρ = 0.002). Two signatures
generalised better on CPTAC than on TCGA: CX16 (ρ_CPTAC = +0.176
vs ρ_TCGA = +0.078) and CX17 (ρ_CPTAC = +0.133 vs +0.044). The
remaining signatures retained the sign and rough magnitude of the
TCGA estimates, with absolute ρ_CPTAC values smaller than ρ_TCGA
in ten of fourteen cases. The paired ρ comparison and the |Δρ|
ranking against the 0.15 threshold are shown in Figure 5.
Underlying numbers are in
`results/external/h2_cptac_external_validation.tsv`.

## 4 Discussion

The present study tested whether CIN signatures can be predicted
directly from routine H&E across a broad pan-cancer population.
Why pursue this when the CX activities are already obtainable from
copy-number sequencing? Because copy-number sequencing is performed
for a small fraction of cancer patients, while H&E is produced for
nearly every solid-tumor diagnosis. The results support feasibility
at scale: on the 21-cohort superset of 3,855 patients, 9 of 14
active signatures pass patient-level bootstrap at q < 0.001,
predicted CX1 quartiles stratify overall survival at a
cohort-stratified log-rank p = 5.4 × 10⁻⁵, and three signatures
reach pooled ρ ≥ 0.30 in the 7-cohort training regime.

Beyond the headline result, the nested cohort design provides a
quantitative decomposition of the pan-cohort effect. The 7-cohort
native model achieves mean active ρ = 0.219 with three H1 hits
(CX1 0.366, CX3 0.384, CX12 0.359). The same 21-cohort model
evaluated on those 289 patients gives ρ = 0.116, with the three H1
signatures attenuated to 0.224, 0.144, and 0.262 respectively. The
component of the small-cohort signal that disappears under the more
diverse model is, definitionally, cancer-type classification: the
7-cohort model can encode that a slide looks like a UCS or KICH
slide and that UCS or KICH patients have high CIN-burden
expectations, while the 21-cohort model is forced to find a
representation that generalises across BRCA, LGG, GBM, LUAD, LUSC,
and HNSC, where that shortcut is no longer adequate. The fraction
of the small-cohort signal that survives the harder generalisation
problem is the within-biology component, and it is non-trivial:
ρ = 0.116 on the same 289 patients remains highly significant by
patient-level bootstrap.

Clinical utility survives the cohort-scaling reduction. Predicted
CX1 quartiles separate Kaplan-Meier survival curves at
p = 5.4 × 10⁻⁵ on 3,831 patients with overall-survival annotation,
with the cohort-stratified log-rank form correctly attributing the
residual signal to within-cohort prediction rather than to the
cohort label itself. Univariate Cox proportional hazards on
continuous CX1 do not reach significance, which is expected given
that the cohort effect dominates the unstratified hazard form. For
ranking and quartile-based decision support, the 21-cohort model is
appropriate; for absolute-activity readout, calibration recovery
would be required because the predicted range is shrunken roughly
four-fold relative to the true range.

Three limitations bound the present analysis. First, training is
TCGA-only. The H2 external-validation test on 283 CPTAC patients
passes for all 14 active signatures at the pre-registered
|Δρ| ≤ 0.15 criterion, but the absolute ρ_CPTAC values are smaller
than ρ_TCGA for ten of fourteen signatures, suggesting some
domain-shift attenuation that broader follow-up cohorts will help
characterise. Second,
predictions are rank-correct but shrunken in absolute scale.
Per-cohort isotonic recalibration would be appropriate if absolute
readout becomes the use case, although for the survival result and
any quartile-based stratification the shrinkage does not affect the
decision. Third, attention rollout is the only interpretability
lens reported at the 21-cohort scale because raw tile pixels were
not retained during ingest. Prototype-tile visualisation and
nuclear-feature enrichment can be revisited in a follow-up run
with a tile-cache option enabled in the ingest pipeline.

Taken together, these results show that routine H&E carries enough
CIN-signature information to support patient-level stratification
across most TCGA solid tumors, and that the effect-size estimate at
the largest training scale is the appropriate one for downstream
clinical and biological use. The three released models cover
complementary operating points: the 7-cohort variant for
high-effect-size questions on rare and CIN-homogeneous tumors, the
15-cohort variant as an intermediate, and the 21-cohort variant
when within-biology generalisation across common cancer types is
required.

## 5 References

1. Drews RM, Hernando B, Tarabichi M, et al. A pan-cancer compendium
   of chromosomal instability. *Nature* 606, 976–983 (2022).
2. Chen RJ, Ding T, Lu MY, et al. Towards a general-purpose
   foundation model for computational pathology. *Nature Medicine*
   30, 850–862 (2024).
3. Ilse M, Tomczak JM, Welling M. Attention-based deep multiple
   instance learning. *Proceedings of the 35th International
   Conference on Machine Learning (ICML)*, PMLR 80:2127–2136 (2018).
4. Macenko M, Niethammer M, Marron JS, et al. A method for
   normalizing histology slides for quantitative analysis. *2009
   IEEE International Symposium on Biomedical Imaging (ISBI)*,
   1107–1110 (2009).
5. Carmichael I, Calhoun BC, Hoadley KA, et al. Joint and
   individual analysis of breast cancer histologic images and
   genomic covariates. *Annals of Applied Statistics* 15(4),
   1697–1722 (2021).
6. Kather JN, Pearson AT, Halama N, et al. Deep learning can
   predict microsatellite instability directly from histology in
   gastrointestinal cancer. *Nature Medicine* 25, 1054–1056 (2019).
7. Coudray N, Ocampo PS, Sakellaropoulos T, et al. Classification
   and mutation prediction from non-small cell lung cancer
   histopathology images using deep learning. *Nature Medicine*
   24, 1559–1567 (2018).
8. Cancer Genome Atlas Research Network. The Cancer Genome Atlas
   Pan-Cancer analysis project. *Nature Genetics* 45,
   1113–1120 (2013).
9. Edwards NJ, Oberti M, Thangudu RR, et al. The CPTAC Data Portal:
   a resource for cancer proteomics research. *Journal of Proteome
   Research* 14, 2707–2713 (2015).
10. Carter SL, Cibulskis K, Helman E, et al. Absolute quantification
    of somatic DNA alterations in human cancer. *Nature
    Biotechnology* 30, 413–421 (2012).
11. Loshchilov I, Hutter F. Decoupled weight decay regularization.
    *International Conference on Learning Representations (ICLR)*
    (2019).
12. Benjamini Y, Hochberg Y. Controlling the false discovery rate:
    a practical and powerful approach to multiple testing. *Journal
    of the Royal Statistical Society. Series B (Methodological)*
    57(1), 289–300 (1995).

## 6 Figures

**Figure 1.** Cohort coverage and pipeline overview. Patient counts
per TCGA cohort across the 7-, 15-, and 21-cohort training sets;
tile-extraction and encoder pipeline summary.

![](figures/fig1_cohort_coverage.png)

**Figure 2.** Cohort-scaling decomposition. (A) Mean active pooled
Spearman ρ versus training cohort count, with same-patient overlay
points showing the 21-cohort model re-applied to the 7-cohort and
15-cohort patient pools. (B) Per-signature paired bars on the
7-cohort patient pool (n = 289) for the top five signatures:
7-cohort native ρ versus the 21-cohort model applied to the same
289 patients. CX1, CX3, and CX12 cross the H1 threshold
(ρ ≥ 0.30) under the 7-cohort model and attenuate substantially
under the 21-cohort model.

![](figures/fig_cohort_scaling.png)

**Figure 3.** Patient-level bootstrap forest plot at the 21-cohort
scale. Per-signature pooled ρ with 95 % CIs from 10,000
patient-resampled bootstraps; 9 of 14 active signatures pass
q_BH < 0.001.

![](figures/fig_patient_bootstrap_forest.png)

**Figure 4.** Calibration top-5. Predicted versus true CX activity
for the five strongest signatures at the 21-cohort scale, with
linear-fit slopes and predicted-range contraction annotated.

![](figures/fig_calibration_top5.png)

**Figure 5.** External validation on CPTAC. (A) Paired ρ_TCGA
versus ρ_CPTAC for the 14 active signatures (n = 283 paired
patients), with green lines marking signatures that improve on
CPTAC. (B) Sorted |Δρ| against the pre-registered 0.15 threshold;
all 14 active signatures pass.

![](figures/fig_cptac_h2.png)

**Figure 6.** Survival association. Kaplan-Meier curves for
predicted CX1 quartiles across the 21-cohort population (n = 3,831
with overall-survival annotation; cohort-stratified log-rank
p = 5.4 × 10⁻⁵).

![](figures/fig_survival_km_cx1.png)

**Figure 7.** Attention rollout (interpretability). Five
representative slides by five top signatures (CX1, CX3, CX5, CX10,
CX12) from the fold-0 checkpoint, colormap heatmaps over the slide
canvas. Tile-pixel backgrounds were unavailable for this run; the
per-slide .npy heatmaps and index TSV are in
`figures/interpret_21cohort/`.

![](figures/fig4_interpret_attention.png)

---

*Originally auto-generated by `workflow/scripts/generate_manuscript_draft.py`
(numerical values populated from JSON outputs in `results/cv/`).
Prose for §1 Introduction, §3 narrative, §4 Discussion, §5 References
and §6 Figures was hand-written 2026-05-09. Re-running the generator
will overwrite the prose sections unless the generator is updated to
preserve hand-written content.*
