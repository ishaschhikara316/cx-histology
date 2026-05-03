# Label availability forecast: 8 new cohorts

For the cohorts ingesting tonight (ESCA, CESC, PAAD, THYM, COAD, PCPG,
KIRP, STAD), this is the expected number of patients with both a slide
and a CX activity label, computed by joining each cohort's GDC manifest
patient barcodes with `data/labels/cx_activities.tsv`.

| Cohort | Slides | Unique patients | Labelled | Coverage |
|--------|--------|-----------------|----------|----------|
| ESCA   | 158    | 156             | 134      | 86%      |
| CESC   | 279    | 269             | 220      | 82%      |
| PAAD   | 209    | 183             | 114      | 62%      |
| THYM   | 181    | 121             | 37       | 31%      |
| COAD   | 459    | 451             | 278      | 62%      |
| PCPG   | 196    | 176             | 83       | 47%      |
| KIRP   | 299    | 275             | 81       | 29%      |
| STAD   | 442    | 416             | 321      | 77%      |
| **Total**| **2,223** | **2,047**     | **1,268**|          |

## Expected combined dataset after ingest finishes

- 7 already-embedded cohorts: 289 labelled patients
- 8 new cohorts (above): 1,268 labelled patients
- **Combined 15-cohort total: ~1,557 labelled patients**

That is 5.4× the 7-cohort run (n=289) and 17.9× the 3-cohort baseline
(n=87). With 5-fold CV that is roughly 311 validation patients per fold,
which is enough for the per-fold ρ estimates to stabilise considerably
and for the bootstrap CIs to tighten.

## Caveats and the sparse cohorts

Three cohorts have markedly low label coverage:

- **THYM (31%)**: thymoma is a small cohort upstream and many slides
  here are from patients without copy number profiles in the
  `cin-ici-mediation` table. Adding THYM contributes more
  *cancer-type diversity* than *labelled patient count*.
- **KIRP (29%)**: similar story, papillary kidney is sparsely covered.
- **PCPG (47%)**: pheochromocytoma and paraganglioma also sparsely
  covered.

These cohorts are still worth keeping because the cancer-type
stratification benefits from breadth. The unlabelled slides simply
fall out of training when the manifest is joined to labels.

## What this implies for the 15-cohort retrain (Stage 5 tonight)

Best HP from the 7-cohort sweep was **h=512, dropout=0.3, lr=1e-4**.
On the 15-cohort manifest, the model should see:

- 1,557 labelled patients across 15 cancer types
- 5-fold CV: ~1,246 train and 311 val per fold
- Three signatures (CX1, CX3, CX12) already pass H1 at n=289;
  expectation is they hold or improve, plus 1–3 more signatures
  cross 0.30 as the per-fold variance shrinks.
- CX13 was inactive at n=289 (only 17/337 nonzero in the labelled
  subset); it may become active again at n=1557 if STAD or COAD
  carries the underlying CN pattern. Worth watching the active-mask
  count printed in stage 5 logs.

## Headline scaling story so far

| Run | Cohorts | Labelled n | Mean active ρ | H1 hits | Best ρ |
|-----|---------|------------|---------------|---------|--------|
| Baseline | 3 | 87 | +0.024 | 2 (incl. suspect CX13) | +0.382 |
| 7-cohort sweep | 7 | 289 | +0.219 | 3 robust | +0.384 |
| 15-cohort (forecast) | 15 | ~1,557 | TBD | TBD | TBD |
