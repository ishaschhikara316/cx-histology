# The story of cx-histology

This is a working document I wrote to keep the reasoning behind this project
in one place. I am Isha. I want a record of how the idea actually came
together, what I was reading at the time, what I tried first, what broke,
and what I learned. Future me will thank present me for not letting this
sit only in my head.

## Where the idea came from

I have been working with TCGA data for a couple of years now. Most of my
prior work used the molecular layers, copy number, expression, mutations,
and clinical outcome. I was reading Drews and colleagues' 2022 Nature
paper, "A pan-cancer compendium of chromosomal instability", which
decomposes copy number profiles into seventeen mechanistic signatures,
CX1 through CX17. Each one corresponds to a specific way that genome
instability is generated in a tumour, things like whole genome doubling,
breakage fusion bridge cycles, or chromothripsis. The paper does the
work of going from raw segments to interpretable causes.

What stuck with me was a simple question. These signatures explain
something fundamental about the disease, but you only get them if you
have a copy number profile. Most clinical labs around the world do not
sequence every patient. What they do have, almost universally, is an
H and E stained slide. So I started wondering whether the same
mechanistic signal that shows up in copy number is also visible in the
tissue itself. Tetraploidy ought to leave a footprint on nuclear size
and chromatin density. Breakage fusion bridge cycles produce gross
nuclear atypia. If the underlying biology is real, the morphology should
carry information about it.

That is the thesis in one sentence. Predict Drews 2022 CX1 to CX17
activities from H and E whole slide images, pan cancer, and find out
which signatures are recoverable and which are not.

## Why this is a reasonable bet

The bet only makes sense because the foundation model situation in
pathology has changed in the last two years. Coudray and colleagues
showed in 2018 (Nature Medicine, lung cancer) that simple convolutional
networks trained from scratch on H and E can predict mutation status,
which was a watershed result. Kather and colleagues followed with
microsatellite instability prediction in 2019, also Nature Medicine.
But these were per disease, per task, trained tile by tile.

What changed was Chen and colleagues' 2024 Nature Medicine paper
introducing UNI, a vision transformer pretrained at scale on
non identifying H and E patches from Mass General Brigham. UNI gives
you a 1024 dimensional patch embedding that is genuinely useful out of
the box. It means I do not need to fine tune anything on my problem.
I can freeze it, compute embeddings once, and only train a small head.
That is what makes the project doable on a graduate student's compute
budget.

For the head, I used the gated attention multiple instance learning
formulation from Ilse and colleagues, ICML 2018. A whole slide is a
bag of tiles, and the attention head learns which tiles matter for the
target. The attention map also gives me interpretability for free,
which I will need later when I want to know which morphological
patterns drive the prediction.

## How the pipeline came together

I wrote the design first, then the code. The pipeline has six stages.

1. WSI ingestion. Pull diagnostic slides from the GDC, tile to
   256 by 256 patches at 0.5 microns per pixel (twenty times
   magnification), filter by tissue fraction and Laplacian variance.
   Stain normalise with the Macenko method (Macenko and colleagues,
   ISBI 2009) using a fixed reference image so all cohorts live in the
   same colour space.
2. Embed every tile with frozen UNI (also tested H optimus 0 from
   Bioptimus as a comparator).
3. Train a gated attention MIL head with five fold cancer type
   stratified cross validation. Multi task Huber loss across the
   seventeen CX outputs. AdamW, log1p plus z score label transform per
   fold.
4. External validation on CPTAC.
5. Interpretability. Attention maps, prototype tile atlases, and
   morphometric enrichment tests on high attention versus low
   attention tiles.
6. Figures and a small dashboard.

I wrote the science and the structure. For the orchestration scripts,
the Snakemake rules, and a few bash launchers (the kind of thing that
chains stages together and routes logs), I worked with Claude as a
pair. Claude is good at the boring plumbing, the kind of code that has
to be correct but adds no scientific content. Anything that touches the
biology or the model is mine. There is a real division of labour there
and I want to be honest about it.

## First milestone, 19 April 2026

I ran the pipeline end to end for the first time on three small
cohorts: DLBC (lymphoma), CHOL (cholangiocarcinoma), and UVM (uveal
melanoma). Eighty seven labelled patients in total. The choice of
cohorts was deliberate. They are small, so they downloaded fast on a
modest box, and they are biologically diverse, which makes the cross
type stratification meaningful even at small n.

The headline numbers, mean Spearman rho across folds:

- CX1 (tetraploidy related), rho equals 0.36 plus or minus 0.24
- CX3 (breakage fusion bridge), rho equals 0.33 plus or minus 0.13
- Both pass the H1 target of rho greater than or equal to 0.30
- A few signatures (CX7, 9, 13, 14, 15) came back undefined

That last point was important. Five signatures had constant labels
within at least one fold, so Spearman correlation was undefined. I
flagged it as something to investigate but did not yet have a clean
explanation.

CX1 and CX3 passing on eighty seven patients was the first real
evidence that the morphology actually carries the signal. I committed
the figure and the cv summary, took a breath, and started planning the
next run.

## The 3 May 2026 run, scaling and a real diagnosis

This is the run I am writing this document during. I scaled the
training set to seven cohorts by adding READ, MESO, UCS, and KICH.
After filtering slides with too few usable tiles, I had two hundred
eighty nine labelled patients, more than three times the previous
sample.

Three things came out of the run.

First, the headline. Best hyperparameters were hidden equals 512,
dropout 0.3, learning rate 1e-4. Mean rho across active signatures
went from 0.047 (three cohort) to 0.219 (seven cohort), almost a five
times improvement. Three signatures cleanly pass H1 now: CX1 at 0.366,
CX3 at 0.384, and a new one, CX12, at 0.359. The CX12 hit is the
result I am most pleased about because it was indistinguishable from
zero on the small cohort run. Adding cohorts made a real signal show
up that was not there before.

Several signatures changed sign. CX6 went from -0.27 to +0.27, CX4
from -0.22 to +0.17. With only eighty seven patients those negative
correlations were noise. I did not want to publish anything based on
the three cohort run for exactly that reason.

Second, the diagnosis on the undefined signatures. I went back and
profiled the upstream label table directly. Across all 7,309 TCGA
patients in cx_activities.tsv, CX7 and CX15 are zero in 100 percent
of cases. CX14 is zero in 99.9 percent. CX13 is zero in 93 percent.
These signatures simply do not manifest at meaningful frequency in
adult TCGA tumours. There is nothing to predict. The Spearman was
not undefined because of fold luck, it was undefined because the
truth value was a constant. I patched the training script to compute
an active signature mask per fold from the training label variance,
and to exclude those signatures from the loss and the metric. The
model now spends its capacity only on signatures that actually have
signal in the data. That alone tightened the per signature rho
estimates noticeably.

Third, an engineering finding worth writing down. The original
Macenko stain normalisation in tile_wsi.py rebuilt and refit the
normaliser for every single tile. With one thousand tiles per slide,
that meant a thousand SVD decompositions of stain vectors per slide.
On a typical WSI it cost about six seconds per tile in the cold path,
which works out to one hundred minutes of wall time per slide for
stain normalisation alone. I cached the fitted normaliser per
process, which gave roughly a seven hundred times speedup. Each slide
now takes about thirty seconds end to end. It is the kind of
optimisation that changes whether a project is feasible at all on a
single box.

## Inspiration and how I think about the field

A few papers shape how I work on this:

- Drews and colleagues 2022 (Nature) for the CX framework itself.
  Without their compendium I would have no labels.
- Ilse and colleagues 2018 (ICML) for the gated attention MIL
  formulation. Still the right inductive bias for whole slides.
- Chen and colleagues 2024 (Nature Medicine) for UNI. The reason a
  pan cancer model is now achievable from a small head.
- Macenko and colleagues 2009 (ISBI) for stain normalisation. The
  pre deep learning toolchain still matters at the front end.
- Coudray and colleagues 2018 (Nature Medicine) and Kather and
  colleagues 2019 (Nature Medicine) for the precedent that H and E
  carries molecular information at all. They proved the conceptual
  case I am extending.
- Fu and colleagues 2020 (Nature Cancer) for the pan cancer
  histopathology framing. They predicted mutation, tumour composition,
  and prognosis from a unified model. My target list is different but
  the philosophy is the same.

My reading of where this project sits is that it adds a missing
target to the existing pan cancer pathology stack. People have
predicted mutations, MSI, immune infiltration, and survival from
H and E. As far as I have seen, no one has explicitly predicted the
Drews CIN signatures slide by slide. If those signatures are
biologically real (and the Nature paper makes a strong case that they
are), then having them in pathology workflow space is useful.

## What I learned that I did not expect, the second pass

After the seven cohort run was in, I went deeper on the same data and
two more findings came out that I want to record before they slip into
the background.

The first is about statistical power. The five fold cross validation
gives you a clean fold level estimate of rho per signature, but if you
try to use it for a frequentist significance test you hit a power
floor. With only five folds, sign flip permutation has thirty two
unique outcomes, so the smallest possible two sided p value is about
0.063. After Benjamini Hochberg adjustment across thirteen active
signatures, no signature reaches q less than 0.05 even when its mean
rho is +0.38 with all five folds positive. That is a design limitation,
not an absence of signal. I rebuilt the test at the patient level
instead, using the saved fold predictions and bootstrapping ten thousand
times. With patient level resampling, five signatures cross q less than
0.001 (CX1, CX3, CX6, CX10, CX12) and nine cross q less than 0.05.
That is the headline statistical artifact for the project. The forest
plot in figures/fig_patient_bootstrap_forest.pdf is the figure I would
include in a paper.

The second is about what the rho actually measures. When I broke the
predictions out by cancer type, the within cohort rho values are much
smaller than the pan cohort rho. CHOL gets within cohort mean rho of
+0.122 and a maximum of +0.337; UCS gets +0.037; READ gets close to
zero. The pan cohort number of +0.219 is partly carried by the model
predicting cancer type level CIN burden correctly rather than within
type biology. That is still a useful prediction. Cancer types differ
in CIN by an order of magnitude and a clinical workflow that needs
to flag high CIN tumours from H and E benefits from getting the type
right. But it is a subtler claim than "predict CIN signatures from
morphology" full stop, and I want to be honest about it in the methods.

Two subgroup findings also came out. Performance improves strongly
with age. CX1 goes from rho +0.054 in patients under fifty to rho
+0.518 in patients eighty and older; CX3 goes +0.147 to +0.597. Stage
also matters: CX1 and CX6 are easiest to predict at stage IV. The
biological reading is that CIN is more pronounced in older patients
and advanced disease, and the model rides that variance. The
methodological reading is that the headline number on a balanced age
distribution would be different from the headline number on an enriched
late stage distribution. Reporting subgroup numbers honestly is part
of the deal.

A last calibration finding. The model is rank correct but amplitude
shrunk. Linear regression of predicted on actual gives slopes between
0.05 and 0.16 across the q less than 0.001 signatures. Spearman holds
because it is rank based, but the predicted CX1 of a tetraploid tumour
will be lower than the actual CX1, and the absolute predicted values
are not directly usable as activity estimates without post hoc
isotonic calibration. A reader who wants quartile assignment is fine,
a reader who wants an absolute number needs the calibration step.

## What I am running tonight

Right now, eight more cohorts are streaming in: ESCA, CESC, PAAD,
THYM, COAD, PCPG, KIRP, and STAD. About 2,200 GB of raw slides will
flow through the pipeline, get tiled and embedded, then thrown away.
The peak disk footprint stays under twenty gigabytes because of
stream mode. The four GPUs run two cohorts in parallel. By morning
I should have fifteen cohorts in total, roughly half of pan TCGA.

After the ingest, I will run the final retrain at the best
hyperparameters from the seven cohort sweep. That run is the first
honest pan cancer number I will be able to report. If the lift from
three to seven cohorts holds when I go from seven to fifteen, mean
rho should land somewhere around 0.30 and the H1 hit count should
grow.

## What is still pending

- CPTAC external validation. The TCIA download path is currently
  untested in the codebase. I need to debug the ingest_cptac.py path
  and get the bioconductor based CX recomputation running, since
  CPTAC does not come with CX activities precomputed.
- Interpretability figures. The scripts exist (attention_rollout.py,
  prototype_tiles.py, nuclear_features.py) but I have not run them on
  the seven cohort model yet.
- A formal model card and methods write up that match the data state
  of whichever is the latest run.
- Eventually, full pan TCGA. With the network throughput on the
  current box, that is a three to five day unattended job rather than
  a multi week project. The 2.6 TB NVMe rental I had planned is no
  longer needed because stream mode keeps the disk footprint tiny.

## Why I am writing this

This document is partly for me, to keep the chain of reasoning
visible, and partly for whoever ends up reading the project later.
Research moves fast in this field. There will be papers I have not
caught up with, and there will be edge cases in the pipeline that I
have not hit yet. But the core claim that motivates the whole project
is that morphology and genome instability are not independent. If you
believe that, then the rest is engineering and patience. The first
seven cohorts say the claim is at least partly right. The next eight
will say more.
