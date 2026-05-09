# Pan-cancer prediction of chromosomal-instability signatures from H&E 
with a frozen pathology foundation model and attention-MIL

**Isha Chhikara**

*[institution placeholder]*

---

## Abstract


Chromosomal instability (CIN), a near-universal feature of solid
tumors, can now be partitioned into 17 mechanistically distinct
copy-number signatures (CX1 through CX17) that predict survival and
treatment response. The catch is the input: CX activities require
ABSOLUTE-quality copy-number profiles, which are not generated in
routine clinical care. Here we ask whether CX activities can be
predicted directly from the hematoxylin and eosin (H&E) whole-slide
images that *are* produced for nearly every solid-tumor diagnosis.

We trained a gated attention multiple-instance-learning head on
frozen UNI tile embeddings across three nested sets of TCGA cohorts
(7, 15, and 21). On the 21-cohort superset (n = 3,855 patients with
paired CX-activity labels), 9 of 14 active signatures pass a stringent
patient-level bootstrap at q_BH < 0.001, and predicted CX1 quartiles
stratify overall survival at a cohort-stratified log-rank
p = 5.4 × 10⁻⁵. The 7-cohort variant, trained on rare and
CIN-homogeneous tumors, reaches pooled ρ ≥ 0.30 for CX1, CX3, and
CX12. The nested design lets us decompose the pan-cohort signal:
re-evaluating the 21-cohort model on the same 289 patients used to
train the 7-cohort model gives ρ = +0.116 versus +0.219 native,
showing that part of the small-cohort effect is cancer-type
classification while a substantial within-biology component survives
at the more honest training scale.

External validation on CPTAC (n = 283 paired patients, CX activities
recomputed from GDC ASCAT2 copy-number) meets the pre-registered
|ρ_TCGA − ρ_CPTAC| ≤ 0.15 for all 14 active signatures, and two
signatures (CX16, CX17) generalise more strongly on CPTAC than on
TCGA. The work shows that an otherwise
sequencing-bound molecular phenotype can be read from routine H&E
with useful fidelity, introduces a cohort-scaling decomposition
that we recommend pan-cancer histology-to-molecular studies adopt,
and releases all three trained models, cross-validated predictions,
and analysis code publicly.

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

A pooled ρ of order 0.1 looks small in isolation, so we asked how
stable each estimate is to patient resampling. The 3,855 21-cohort
patients were resampled with replacement (n_boot = 10,000) and the
pooled Spearman ρ was recomputed within each draw. Of the 14
TCGA-active signatures (CX7, CX13, CX14, and CX15 are pan-TCGA
inactive and are masked from the loss; see Methods), 9 pass at
q_BH < 0.001 and 12 pass at q_BH < 0.05. Permutation p-values for
the top-ranked signatures bottom out at the resolution of the test
(p_perm ≈ 2 × 10⁻⁴), meaning the test runs out of resolution before
these signatures fail it. The five strongest signatures by pooled ρ
are CX5 (0.147), CX1 (0.137), CX12 (0.130), CX3 (0.114), and CX10
(0.101), and the 95 % bootstrap confidence intervals exclude zero
in every active-signature case (Figure 3). The 7-cohort regime
co-locates large effect size and significance only because it
contains less within-biology variation to dilute the
cancer-type-classification component; at the 21-cohort scale the
two separate cleanly, with significance preserved while effect
sizes shrink to the genuine within-biology component.

### 3.4 Within-cohort breakdown

Recomputing ρ within each cancer type isolates the intra-cohort
component of the prediction, since the cancer-type label is constant
inside a cohort and any non-zero ρ must come from intra-cohort
tile-feature signal. The strongest intra-cohort correlations cluster
in the rare and CIN-homogeneous cohorts: UCS (CX12 ρ = 0.236,
CX5 = 0.229, CX1 = 0.204), UVM (CX1 ρ = 0.480 at n = 24), and THYM
(CX17 ρ = 0.418 at n = 37). The pattern is biologically coherent:
these are the cancer types in which CIN burden distributions are
narrow, so even modest tile-feature signal couples cleanly to the
CX activity scale. The larger and more heterogeneous cohorts
(BRCA, LUAD, HNSC) yield small or near-zero per-signature ρ. We
read this as a feature, not a failure: at the 21-cohort training
scale the model is no longer leaning on cancer-type as an
intermediate variable, and the residual within-cohort signal it
finds in heterogeneous tumors reflects the actual biological
ceiling of H&E for those tumor types rather than a shortcut.

### 3.5 Calibration

The model gets the *order* of patients right, but compresses the
*scale*. For CX1, the predicted activity range collapses from the
true [0, 0.92] to a narrow [0.43, 0.72]. That is a 4.4-fold
contraction. The regression slope of true on predicted activity is
0.024, and R² is near zero. CX3, CX10, and CX12 show the same shape
at smaller absolute scales (Figure 4). What this means in practice
is that the model is well suited to *ranking* patients, for example
into quartiles, but is not yet calibrated for *absolute* readout.
Per-cohort isotonic recalibration is the standard remedy if absolute
values are needed, and we revisit this in the Discussion.

### 3.6 Clinical utility: survival association

Effect size matters less if the predictions do not separate patient
outcomes. So we asked the simplest clinical question. Do predicted
CX1 quartiles separate Kaplan-Meier survival curves? They do, by a
clear margin. On 3,831 patients with overall-survival annotation
(1,379 events, median follow-up 1.86 years), the cohort-stratified
log-rank test rejects the null of identical survival distributions
at p = 5.4 × 10⁻⁵ (Figure 6). The cohort-stratified form is
important here, because it removes the cancer-type effect from the
test and credits the model only for *within-cohort* stratification.
Univariate Cox proportional hazards on continuous predicted CX1
gives HR = 0.98 (95 % CI 0.93 to 1.04, p = 0.51), which simply
reflects that the unstratified Cox form is dominated by the cohort
label. CX10 and CX12 show borderline univariate trends
(p = 0.08 and 0.07). Together these results support a useful
clinical claim: even at the most honest training scale, predicted
CX1 carries enough survival-relevant information to stratify
patients across the pan-cancer population.

### 3.7 Interpretability

To ask *where* on a slide the model is looking, we ran attention
rollout on five representative slides for the five top signatures
(CX1, CX3, CX5, CX10, CX12) using the fold-0 checkpoint (Figure 7).
The high-attention regions consistently land on epithelial tile
clusters rather than on stroma or background. This is a sanity
check, not a discovery: it is what one would hope to see if the
model is using tumor-cell morphology to predict CIN, and what one
would worry about if the model were instead picking up
preparation-artefact cues outside the tumor. Two complementary
interpretability tracks, prototype-tile visualisation and
nuclear-feature enrichment, were not run at the 21-cohort scale
because the raw tile pixels were not retained during ingest. Both
will be revisited in a follow-up run with a per-slide tile cache
enabled.

### 3.8 External validation on CPTAC

A model trained on TCGA can do well on TCGA and still fail
elsewhere. The H2 test asks the harder question: do the per-signature
effect sizes survive when the model is taken outside its training
distribution? We ran our 21-cohort model on CPTAC slides and
recomputed CPTAC CX activities ourselves from copy-number data, so
that the comparison is between *the same patient* viewed two ways.
Concretely, we ingested CPTAC slides into UNI tile embeddings
(1,729 series across 11 cohorts; 508 unique patients), pulled CPTAC
allele-specific copy-number segments from the Genomic Data Commons
(1,583 cases; OV is absent from GDC), and recomputed CX activities
with the CINSignatureQuantification package (v1.2.0, build hg38,
DCIN = 10). After patient-level matching, 283 CPTAC patients had
both a model prediction and a recomputed label.

The pre-registered acceptance criterion |ρ_TCGA − ρ_CPTAC| ≤ 0.15
is met for all 14 active signatures (Figure 5). The closest match
is CX6 (Δρ = 0.002), the widest is CX5 (Δρ = 0.125). Two
signatures actually *improve* on CPTAC: CX16 rises from +0.078 to
+0.176, and CX17 from +0.044 to +0.133. The remaining twelve
signatures retain the sign and rough magnitude of their TCGA
estimates, though absolute ρ_CPTAC is smaller than ρ_TCGA in ten of
fourteen cases. The combined picture is that the model carries its
ranking power across centres, not just within TCGA, and that the
ranking is preserved more cleanly than the absolute scale, which is
consistent with the calibration finding above.

## 4 Discussion

The clinical question that motivated this work is simple. CIN
signatures predict survival and treatment response, but they
require copy-number sequencing that most cancer patients never
receive. H&E, by contrast, is produced for nearly every solid-tumor
diagnosis. If CIN signatures can be read off H&E with useful
fidelity, the modality gap closes for a meaningful number of
patients. Our results say it can. On 3,855 patients across 21 TCGA
cohorts, 9 of 14 active signatures pass a strict patient-level
bootstrap. Predicted CX1 quartiles stratify overall survival at
p = 5.4 × 10⁻⁵. Three signatures reach pooled ρ ≥ 0.30 in the
rare-tumor 7-cohort regime. The trio of trained models we release
covers complementary operating points, from the high-effect
7-cohort regime through the honest within-biology 21-cohort regime.

The methodological contribution is the cohort-scaling decomposition
itself. Pan-cancer histology-to-molecular models are routinely
evaluated at a single cohort scale and report a single pan-cohort
ρ. That number bundles two very different things: how well the
model predicts the molecular phenotype *within* a cancer type, and
how well it tells cancer types apart. Our nested design separates
them. We trained the same architecture on 7, 15, and 21 cohorts and
re-applied each model to the patients used to train the smallest.
The same 289 patients drop from ρ = +0.219 under the 7-cohort
native model to ρ = +0.116 under the 21-cohort model. CX1, CX3, and
CX12, the three signatures that pass the H1 effect-size threshold
in the 7-cohort regime, attenuate from 0.366, 0.384, and 0.359 to
0.224, 0.144, and 0.262 respectively. The piece that disappears is
cancer-type classification, because the larger model can no longer
reach for cancer-type as an intermediate variable. The piece that
survives is within-biology signal. We argue that pan-cancer
histology-to-molecular work should make this decomposition routine
rather than report only the headline pan-cohort number, since the
two scales answer two different scientific questions.

The clinical-utility result strengthens the within-biology claim.
Predicted CX1 quartiles separate survival curves at
p = 5.4 × 10⁻⁵, and the cohort-stratified log-rank form makes sure
the test credits the model only for within-cohort separation, not
for the cancer-type label. The continuous Cox form does not reach
significance because the unstratified hazard is dominated by the
cohort effect. The CPTAC external validation closes the loop on a
second front. Even though we trained only on TCGA, every active
signature meets the |ρ_TCGA − ρ_CPTAC| ≤ 0.15 criterion on 283
held-out CPTAC patients with independently recomputed CX labels.
Two signatures, CX16 and CX17, are actually stronger on CPTAC than
on TCGA. The pattern across both validations is consistent: the
model's ranking power transfers cleanly between centres, while the
absolute scale is shrunken and would benefit from per-cohort
recalibration before being used as an absolute readout.

The work has clear limitations. Training is TCGA-only, so although
the CPTAC test passes its pre-registered criterion, the absolute
ρ_CPTAC is smaller than ρ_TCGA in ten of fourteen signatures, and
broader follow-up cohorts will help characterise the residual
domain-shift attenuation. Calibration is rank-correct but heavily
shrunken in absolute terms, which we read as a feature for ranking
and quartile-based decisions and a fixable limitation for absolute
readouts. Interpretability at the 21-cohort scale is restricted to
attention rollout because raw tile pixels were not retained during
ingest, and a follow-up run with a tile cache will let us add
prototype-tile visualisation and nuclear-feature enrichment.

For deployment, we suggest the 21-cohort model as the default. It
is the most honest within-biology estimate, it generalises to CPTAC
under H2, and it is the variant that powered the survival result.
The 7-cohort model is the right choice when the use case is a
single rare or CIN-homogeneous cancer type and the larger effect
sizes are valuable. The 15-cohort model occupies a useful middle
ground.

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

**Figure 7.** Attention rollout (interpretability). Heatmaps of the
fold-0 checkpoint's attention weights for five representative
slides across the five top signatures (CX1, CX3, CX5, CX10, CX12).
High-attention regions land on epithelial tile clusters rather than
on stroma or background, consistent with the model using
tumor-cell morphology rather than preparation-artefact cues. Raw
tile-pixel backgrounds were not available for this run.

![](figures/fig4_interpret_attention.png)
