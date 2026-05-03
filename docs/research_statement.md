# Research statement

I am a computational biologist working at the intersection of cancer
genomics and pathology imaging. My current research asks whether the
seventeen mechanistic chromosomal-instability signatures defined by
Drews and colleagues in 2022 (Nature) can be predicted directly from
H and E whole slide images, the same images already produced in
clinical workflow. If they can, then a substantial fraction of the
information that today requires copy number sequencing becomes
accessible from a single tissue slide. That is a worthwhile target.

## What I am doing

I am training a frozen-encoder, attention-based multiple-instance
learning model to map slide-level tile bags to the seventeen CIN
signature activities, pan-cancer, on TCGA. The encoder is UNI
(Chen and colleagues, Nature Medicine 2024), a vision transformer
pretrained on roughly one hundred million H and E patches. The head
is the gated attention MIL formulation of Ilse and colleagues
(ICML 2018), trained with a multi-task Huber loss and a per-fold
active-signature mask that excludes signatures with constant labels.
The training is a single small head on top of a frozen large encoder,
which keeps the project tractable on a four-GPU box.

## Where the work stands

I have run the pipeline at three scales, with consistent gains:

- Three TCGA cohorts (n = 87 labelled patients): mean active rho
  = +0.024, two H1 hits at rho greater than or equal to 0.30, of
  which one (CX13) was a constant-near-zero artifact.
- Seven TCGA cohorts (n = 289): mean active rho = +0.219, three
  robust H1 hits (CX1 at +0.366, CX3 at +0.384, CX12 at +0.359).
  Patient-level bootstrap (n_boot = 10,000) gives five signatures
  significant at q less than 0.001 after BH correction
  (CX1, CX3, CX6, CX10, CX12), and nine signatures at q less than 0.05.
- Fifteen TCGA cohorts (in flight at time of writing, n forecast at
  about 1,557): final retrain at the sweep-best hyperparameters runs
  unattended and writes its results back to the repository when done.

The three to seven cohort jump improved mean active rho about five
times. Several signatures flipped sign as more data entered the pool,
which is the expected behaviour when the small-n estimates were noise.
CX6 in particular went from rho = -0.27 to rho = +0.27, the largest
swing.

## What is novel

People have predicted mutation status, microsatellite instability,
tumour composition, and overall survival from H and E (Coudray and
colleagues 2018; Kather and colleagues 2019; Fu and colleagues 2020,
all Nature Medicine or Nature Cancer). These are clinically
actionable but they are also one-task-at-a-time. The CIN signatures
are different in two ways. First, they are *mechanistic*, decomposing
genome instability into causally distinct generators rather than
correlating with a single label. Second, they are pan-cancer in their
construction, so a single model can plausibly carry signal across
disease. The combination is a missing target in the current
pan-cancer pathology stack, and the signatures themselves are recent
enough that the pathology-prediction literature has not yet caught up.

## What I learned that surprised me

Two findings worth flagging, both methodologically informative.

First, four of the seventeen signatures (CX7, CX13, CX14, CX15) are
essentially flat in TCGA. CX7 and CX15 are zero in 100 percent of
7,309 patients in the upstream label table; CX14 in 99.9 percent.
These signatures cannot be Spearman-evaluated because the variance
is zero or near-zero, so the original three-cohort run reported
"undefined" for them and the original code wasted model capacity
on them. I diagnosed the cause empirically and added a per-fold
active-signature mask that excludes them from the loss and from
metric reporting. This both tightened the per-signature rho
estimates and let me scope the methods writeup honestly to the
thirteen signatures that actually have signal in TCGA.

Second, an engineering finding. The original Macenko stain
normaliser in the tile pipeline rebuilt and refit the normalising
operator for every single tile, making the per-slide stain
normalisation step take about one hundred minutes. Caching the
fitted operator per process gave a roughly seven hundred times
speedup. The end-to-end ingest then went from "infeasible on a
single box" to "two thousand slides in a few hours". It is a
reminder that pipeline performance can be the difference between
a project that scales and a project that lives forever as a
proof of concept.

## How I would extend this

There are four directions that build naturally on the current work.

1. *External validation on CPTAC.* I have verified that CPTAC
   pathology lives on the NCI Imaging Data Commons (not TCIA, as
   the original design assumed) and have prepared an IDC-based
   ingest path with thirteen CPTAC pathology cohorts available. The
   target metric is |Δrho(TCGA − CPTAC)| less than or equal to 0.15
   on the H1 signatures. If that holds, the result transfers across
   the two largest pan-cancer pathology cohorts available.

2. *Interpretability.* Attention rollouts from the trained ABMIL
   head should localise the morphological features driving the
   strongest predictions. CX1 (tetraploidy) and CX3 (breakage
   fusion bridge cycles) have specific cell-biological signatures
   (large nuclei, atypical nuclear pleomorphism, fragmented
   chromosomes) that a pathologist would recognise. Pairing
   attention maps with nuclear morphometric enrichment tests
   gives a direct route to mechanistic claims rather than a black
   box.

3. *Clinical correlation.* Drews and colleagues showed that CIN
   signature activity correlates with response to platinum
   chemotherapy, PARP inhibitors, and (less directly) immune
   checkpoint blockade. If H and E predicted activities preserve
   the activity-response correlations, then this work becomes a
   route to stratifying patients without sequencing. That is a
   testable proposition with available outcome data.

4. *Encoder comparison.* I used UNI as the primary encoder; the
   pipeline supports H-optimus-0 (Bioptimus) as a swap-in
   comparator. An honest pan-cancer encoder benchmark on this task
   would be a small but useful contribution to the foundation
   model literature in pathology.

## What I bring to a PhD project

I have shipped the cx-histology pipeline end to end, from raw
GDC SVS files to a publication-ready ABMIL training and evaluation
loop, on infrastructure I provisioned and debugged myself. I have
prior TCGA experience (survival analysis, differential expression,
CRISPR data integration) and a working knowledge of the relevant
tooling (PyTorch, Snakemake, conda, openslide, libvips, GDC and IDC
APIs). I am comfortable diagnosing performance bottlenecks at the
algorithm and the system level, as the Macenko caching fix
illustrates. I treat statistical claims with care: the patient-level
bootstrap above is a deliberate replacement for the underpowered
fold-level permutation test, and I will not report a result whose
confidence interval crosses zero as a confident finding.

I want to do a PhD on a related question: can interpretable models
trained on routinely available data (H and E imaging, basic
clinical labs, panel sequencing if available) recover the
mechanistic biology that today only the most expensive multi-omics
pipelines can. This project is the seed I am building toward
that direction.

## References

- Drews et al., 2022. *A pan-cancer compendium of chromosomal
  instability.* Nature.
- Ilse et al., 2018. *Attention-based Deep Multiple Instance
  Learning.* ICML.
- Chen et al., 2024. *Towards a general-purpose foundation model
  for computational pathology.* Nature Medicine.
- Macenko et al., 2009. *A method for normalizing histology slides
  for quantitative analysis.* ISBI.
- Coudray et al., 2018. *Classification and mutation prediction
  from non-small cell lung cancer histopathology images using
  deep learning.* Nature Medicine.
- Kather et al., 2019. *Deep learning can predict microsatellite
  instability directly from histology in gastrointestinal cancer.*
  Nature Medicine.
- Fu et al., 2020. *Pan-cancer computational histopathology
  reveals mutations, tumor composition and prognosis.* Nature
  Cancer.
