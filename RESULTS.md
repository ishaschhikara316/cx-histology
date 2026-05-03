# Results — cx-histology, snapshot 2026-05-03

A concise, paper-style results page for the 7-cohort run. The
interactive version with all figures embedded is at
[`figures/dashboard_7cohort.html`](figures/dashboard_7cohort.html).

## Headline numbers

Best HP from sweep: `hidden=512, dropout=0.3, lr=1e-4`. Trained on
**289 labelled TCGA patients** across 7 cohorts (DLBC, CHOL, UVM, READ,
MESO, UCS, KICH), 5-fold cancer-type-stratified CV, frozen UNI
embeddings, gated ABMIL head, multi-task Huber.

| Metric | 3-cohort baseline | 7-cohort sweep best | Δ |
|---|---|---|---|
| Patients | 87 | 289 | +233% |
| Mean active ρ | +0.024 | **+0.219** | +0.195 (~5×) |
| H1 hits (ρ ≥ 0.30) | 2 (incl. spurious CX13) | **3 robust** | qualitative |
| Active signatures | 14 (1 spurious) | 13 (clean) | 4 dead masked |

## Per-signature ρ at the best HP

Spearman ρ across 5 folds (mean ± std), patient-level bootstrap CI on
the same data (n_boot=10,000), permutation p-value, BH-adjusted q.

| Sig | ρ ± std | 95% CI | p (perm) | q (BH) | Tag |
|-----|---------|--------|----------|--------|-----|
| CX1 | +0.366 ± 0.052 | [+0.246, +0.456] | 0.0002 | **0.0006** | ★ H1, q<0.001 |
| CX3 | +0.384 ± 0.096 | [+0.255, +0.459] | 0.0002 | **0.0006** | ★ H1, q<0.001 |
| CX6 | +0.271 ± 0.046 | [+0.146, +0.372] | 0.0002 | **0.0006** | ✓ q<0.001 |
| CX10 | +0.236 ± 0.112 | [+0.098, +0.322] | 0.0002 | **0.0006** | ✓ q<0.001 |
| CX12 | +0.359 ± 0.048 | [+0.248, +0.452] | 0.0002 | **0.0006** | ★ H1, q<0.001 |
| CX2 | +0.193 ± 0.125 | [+0.032, +0.268] | 0.0106 | 0.0231 | ✓ q<0.05 |
| CX4 | +0.171 ± 0.135 | [+0.034, +0.262] | 0.0132 | 0.0231 | ✓ q<0.05 |
| CX5 | +0.189 ± 0.067 | [+0.028, +0.247] | 0.0170 | 0.0264 | ✓ q<0.05 |
| CX17 | +0.205 ± 0.142 | [+0.040, +0.251] | 0.0126 | 0.0231 | ✓ q<0.05 |
| CX8 | +0.087 ± 0.092 | [-0.006, +0.216] | 0.0724 | 0.0921 | n.s. |
| CX11 | +0.112 ± 0.076 | [-0.064, +0.165] | 0.3871 | 0.4516 | n.s. |
| CX16 | +0.149 ± 0.174 | [-0.003, +0.227] | 0.0522 | 0.0731 | n.s. (borderline) |
| CX9 | +0.125 ± 0.193 | [-0.068, +0.147] | 0.5143 | 0.5539 | n.s. |
| CX13 | n/a | [-0.152, +0.105] | 0.6933 | 0.6933 | n.s. (mostly inactive) |
| CX7, CX14, CX15 | masked | — | — | — | inactive in TCGA |

**5 signatures at q < 0.001. 9 at q < 0.05.** 3 of those 5 also pass H1
(ρ ≥ 0.30). Source: `results/cv/uni_7cohort_sweep/h512_d0.3_lr0.0001/{cv_summary,robustness,patient_bootstrap}.json`.

## Cohort scaling

Adding READ, MESO, UCS, KICH to the original DLBC + CHOL + UVM cohort
set lifted mean active ρ ~5×. Several signatures flipped sign: CX6
went from −0.27 to +0.27, CX4 from −0.22 to +0.17. Both are consistent
with small-n estimates being noise that washed out with more data.
CX12 emerged as a robust H1 hit at the larger n; it was indistinguishable
from zero at n=87.

See [`figures/compare_3vs7_cohort_rho.pdf`](figures/compare_3vs7_cohort_rho.pdf).

## Caveats and what the rho is actually measuring

Two findings that should be reported with the headline number, not buried.

### Pan-cohort vs within-cohort

Per-cohort breakdown (re-running inference on the saved fold checkpoints
and computing ρ within each cohort) gives noticeably smaller numbers:

| Cohort | n | Within-cohort mean ρ | Within-cohort max ρ | H1 hits |
|---|---|---|---|---|
| READ | 80 | +0.003 | +0.189 | 0 |
| MESO | 58 | +0.001 | +0.202 | 0 |
| UCS | 55 | +0.037 | +0.302 | 1 |
| CHOL | 32 | +0.122 | +0.337 | 2 |
| DLBC | 31 | -0.063 | +0.169 | 0 |
| UVM | 24 | -0.061 | +0.196 | 0 |
| KICH | 9 | +0.082 | +0.707 | 2 (small-n) |

The pan-cohort ρ of +0.219 is partly the model getting the cancer-type-level
CIN burden right rather than within-cancer biology. That is still a
useful capability (cancer types differ in mean CIN by an order of magnitude
and predicting type-level burden from H&E is non-trivial) but it is a
narrower claim than "predict CIN signatures from morphology". Both numbers
should be reported.

Source: `results/cv/uni_7cohort_sweep/h512_d0.3_lr0.0001/rho_per_cohort.tsv`.

### Calibration

Linear regression of predicted on actual (per signature):

| Sig | slope | intercept | Pearson r |
|---|---|---|---|
| CX1 | +0.156 | +0.502 | +0.402 |
| CX3 | +0.130 | +0.117 | +0.351 |
| CX6 | +0.058 | +0.008 | +0.221 |
| CX10 | +0.047 | +0.013 | +0.290 |
| CX12 | +0.061 | +0.007 | +0.247 |

All slopes are far below 1.0 — the model gives **rank-correct** predictions
but with strong shrinkage to the mean. For "rank by predicted CX1" use,
this is fine. For "give me an estimate of patient X's CX1 activity" use,
post-hoc isotonic calibration is required.

Source: `results/cv/uni_7cohort_sweep/h512_d0.3_lr0.0001/calibration_metrics.json`.

### Subgroup performance

Performance varies systematically across patient subgroups:

| Subgroup | CX1 | CX3 | CX6 | CX10 | CX12 |
|---|---|---|---|---|---|
| age <50 | +0.054 | +0.350 | +0.391 | -0.179 | +0.163 |
| age 50-60 | +0.297 | +0.147 | +0.015 | +0.149 | +0.291 |
| age 60-70 | +0.353 | +0.391 | +0.166 | +0.157 | +0.548 |
| age 70-80 | +0.395 | +0.529 | +0.265 | +0.245 | +0.438 |
| age 80+ | **+0.518** | **+0.597** | +0.247 | +0.262 | +0.342 |
| stage I | +0.175 | +0.456 | +0.341 | -0.048 | +0.380 |
| stage II | +0.347 | +0.421 | +0.139 | +0.121 | **+0.590** |
| stage III | +0.314 | +0.518 | +0.398 | +0.098 | +0.193 |
| stage IV | +0.367 | +0.493 | **+0.503** | +0.092 | +0.452 |
| female | +0.423 | +0.337 | +0.258 | +0.338 | +0.349 |
| male | +0.258 | +0.378 | +0.265 | -0.016 | +0.242 |

Older patients and later stages are easier to predict, consistent with
CIN being more pronounced in advanced disease. CX10 has a notable
sex gap: ρ=0.338 in females versus −0.016 in males.

Source: `results/cv/uni_7cohort_sweep/h512_d0.3_lr0.0001/subgroup_metrics.json`.

## Signature crosstalk

Pan-TCGA across 7,309 patients, the 13 active CX signatures are not
independent. Notable correlations (|ρ| > 0.30):

- CX4 ↔ CX6: ρ = +0.544 (largest positive)
- CX5 ↔ CX12: ρ = −0.510 (largest negative)
- CX5 ↔ CX17: ρ = +0.474
- CX2 ↔ CX6: ρ = −0.434
- CX5 ↔ CX8: ρ = +0.439
- CX12 ↔ CX17: ρ = −0.408

This means a single H&E slide carries information about a correlated
manifold rather than 13 independent dimensions. Implications for
multi-task training are visible: signatures with strong positive
crosstalk tend to share fold-to-fold ρ trajectories.

See [`figures/fig_cx_crosstalk.pdf`](figures/fig_cx_crosstalk.pdf).

## Outliers

UCS (uterine carcinosarcoma) is the hardest cohort: median residual L2
4.83, almost twice the next-worst. UCS dominates the top-20 hardest
patients. Plausibly because UCS is a rare cancer with both high CIN
burden and morphological heterogeneity (sarcomatoid + carcinomatous
components). Worth a separate analysis once we have more UCS or its
biological cousin (CESC or UCEC) in the training set.

Source: `results/cv/uni_7cohort_sweep/h512_d0.3_lr0.0001/outlier_patients.tsv`.

## What is in flight tonight

Eight more cohorts ingesting (ESCA, CESC, PAAD, THYM, COAD, PCPG, KIRP,
STAD). Forecast: combined 15-cohort dataset will have ~1,557 labelled
patients, 5.4× the 7-cohort run. The auto-finalizer in
`workflow/scripts/finalize_15cohort.sh` runs the final retrain at the
sweep-best HP and pushes results when ingest completes.

Expected finish: ~02:00 tomorrow morning UTC. The 15-cohort
`results/cv/uni_15cohort_final/cv_summary.json` and
`figures/final_15cohort_rho.{pdf,png}` will appear on origin/main
overnight.

## Reproducibility

- Code, configs, and labels are in this repo at the commit hash
  reported in [`CHANGELOG.md`](CHANGELOG.md).
- Random seed `20260419` across NumPy, PyTorch, and StratifiedKFold.
- All hyperparameters in `config/{models,params}.yaml`.
- Fold checkpoints, predictions, and JSON metric files are tracked
  for the best HP under `results/cv/uni_7cohort_sweep/h512_d0.3_lr0.0001/`.
