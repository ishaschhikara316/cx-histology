# Changelog

Notable work in reverse chronological order. Commit hashes link the entry
to a specific change in the repo.

## 2026-05-03 — 7-cohort sweep, NaN-aware training, ingest revival

### Headline science

- **Best HP (`h512_d0.3_lr1e-4`) on 7 TCGA cohorts (n=289):**
  mean active ρ = +0.219 (~5× the 3-cohort baseline's +0.024).
  Three robust H1 hits: CX1 (+0.366), CX3 (+0.384), CX12 (+0.359).
  Patient-level bootstrap (n_boot=10,000) gives **5 signatures
  significant at q < 0.001** (CX1, CX3, CX6, CX10, CX12) and
  **9 signatures at q < 0.05**. See
  [`figures/dashboard_7cohort.html`](figures/dashboard_7cohort.html).

- **8 more cohorts (ESCA, CESC, PAAD, THYM, COAD, PCPG, KIRP, STAD)**
  ingest started 16:00. ESCA + CESC complete; PAAD/THYM in progress at
  time of writing. Combined 15-cohort dataset will be ~1,557 labelled
  patients (5.4× the 7-cohort run). The auto-finalizer in
  `workflow/scripts/finalize_15cohort.sh` will retrain at the
  sweep-best HP and push results when ingest completes.

### Code changes

- **NaN-aware training**
  ([7a7e1d6](https://github.com/ishaschhikara316/cx-histology/commit/7a7e1d6)).
  `train_cv.py` and `loss.py` now compute a per-fold active-signature
  mask from training-label variance (threshold 1e-5). CX7, CX14, CX15
  are zero in 100% / 99.9% of pan-TCGA patients; CX13 is zero in 93%.
  These get auto-excluded from the loss and ρ computation. The mask
  is reported in `cv_summary.json["active_in_n_folds"]`.

- **Macenko 700× speedup**
  ([ae7d9a7](https://github.com/ishaschhikara316/cx-histology/commit/ae7d9a7)).
  `tile_wsi.py` cached the fitted Macenko normaliser per process. The
  original code rebuilt and refit the operator on every tile, costing
  ~6 s per tile cold-path, i.e. ~100 minutes per slide just for stain
  normalisation. With the cache, warm transforms drop to 2.4 ms each.
  Also: `build_train_manifest.py` now skips slides with fewer than 20
  retained tiles and records `n_tiles` per slide for downstream analysis.

- **CUDA_VISIBLE_DEVICES fix in parallel ingest**
  ([d57b7e9](https://github.com/ishaschhikara316/cx-histology/commit/d57b7e9)).
  `stream_ingest_parallel.py` now respects an existing
  CUDA_VISIBLE_DEVICES restriction when the parent shell pins a cohort
  to a GPU subset. Without this, two parallel cohorts on `0,1` and
  `2,3` would both collapse onto physical GPUs 0,1.

- **4-GPU parallel HP sweep**
  ([d57b7e9](https://github.com/ishaschhikara316/cx-histology/commit/d57b7e9)).
  `plan_a_orchestrator.sh` rewritten to slot-schedule the 6 HP combos
  across N visible GPUs with a small bash slot pool. Sweep wall time
  fell from ~6 hours sequential to ~30 minutes on 4 GPUs.

- **8-cohort dual-group ingest launcher**
  ([d57b7e9](https://github.com/ishaschhikara316/cx-histology/commit/d57b7e9)).
  `parallel_ingest_8cohort.sh` runs two cohorts simultaneously on
  GPU pairs (0,1) and (2,3), with each cohort using `n-workers=2,
  gdc-processes=8`.

- **Auto-finalizer for the 15-cohort retrain**
  ([a458f14](https://github.com/ishaschhikara316/cx-histology/commit/a458f14)).
  `finalize_15cohort.sh` polls for all 8 `.done` markers, builds the
  combined manifest, runs the final retrain at the sweep-best HP,
  generates the headline figure, and pushes to origin.

### Analysis and figures

- **3-cohort vs 7-cohort comparison figure**
  ([90a7d42](https://github.com/ishaschhikara316/cx-histology/commit/90a7d42)).
  `figures/compare_3vs7_cohort_rho.{pdf,png}` shows side-by-side
  per-signature ρ with delta panel.

- **HP sweep heatmap**
  ([90a7d42](https://github.com/ishaschhikara316/cx-histology/commit/90a7d42)).
  `figures/fig2_hp_sweep_heatmap.{pdf,png}` orders the 6 combos by
  mean active ρ and shows per-signature performance.

- **Robustness analysis**
  ([90a7d42](https://github.com/ishaschhikara316/cx-histology/commit/90a7d42)).
  `workflow/scripts/robustness_analysis.py` writes BH-adjusted q-values
  and bootstrap CIs to `robustness.json` and a table figure.

- **Per-cohort breakdown**
  ([b748927](https://github.com/ishaschhikara316/cx-histology/commit/b748927)).
  `per_cohort_breakdown.py` re-runs inference on saved fold checkpoints,
  saves `predictions.tsv` and per-cohort ρ. CHOL has 2 H1 within-cohort
  hits at n=32; KICH has max ρ=0.71 but n=9 so noise. Pan-cohort ρ
  partly reflects cancer-type effects.

- **Patient-level bootstrap**
  ([b748927](https://github.com/ishaschhikara316/cx-histology/commit/b748927)).
  `patient_bootstrap.py` resamples at the patient level (much higher
  power than the 5-fold-permutation). Five signatures cross q < 0.001.
  The forest plot in `figures/fig_patient_bootstrap_forest.{pdf,png}`
  is the strongest statistical artifact in the project.

- **CX signature crosstalk and per-cohort distribution**
  ([b748927](https://github.com/ishaschhikara316/cx-histology/commit/b748927)).
  `cx_crosstalk_analysis.py` produces a 13×13 signature correlation
  matrix and a per-cohort mean activity heatmap.
  CX1 (tetraploidy) peaks in THCA at mean 0.842; CX3 (BFB) peaks in
  OV. Strongest signature pair: CX4–CX6 at ρ = +0.54.

- **Interactive dashboard**
  ([90a7d42](https://github.com/ishaschhikara316/cx-histology/commit/90a7d42)).
  `figures/dashboard_7cohort.html` is a self-contained HTML with
  embedded figures, sweep table, robustness rows, and references.

### Documentation

- [`docs/journey.md`](docs/journey.md): first-person project narrative
  ([8e8b76f](https://github.com/ishaschhikara316/cx-histology/commit/8e8b76f)).
- [`docs/methods.md`](docs/methods.md) and [`README.md`](README.md):
  rewritten with concrete 7-cohort numbers
  ([4b90e3e](https://github.com/ishaschhikara316/cx-histology/commit/4b90e3e)).
- [`docs/cptac_ingest_blockers.md`](docs/cptac_ingest_blockers.md):
  TCIA route is wrong; IDC is the right path
  ([4b90e3e](https://github.com/ishaschhikara316/cx-histology/commit/4b90e3e)).
- [`docs/cohort_label_forecast.md`](docs/cohort_label_forecast.md):
  expected labelled-patient counts for the 8 ingesting cohorts
  ([b748927](https://github.com/ishaschhikara316/cx-histology/commit/b748927)).
- [`docs/research_statement.md`](docs/research_statement.md):
  two-page condensed version of the journey, written for PhD
  applications, with no PI targeting
  ([b748927](https://github.com/ishaschhikara316/cx-histology/commit/b748927)).

### Tests

- [`workflow/scripts/tests/test_fixes_2026_05_03.py`](workflow/scripts/tests/test_fixes_2026_05_03.py)
  ([b748927](https://github.com/ishaschhikara316/cx-histology/commit/b748927)).
  9 tests, all pass: Macenko cache instance identity and ≥10×
  speedup, MultiTaskHuber default mask and exclusion logic,
  CUDA_VISIBLE_DEVICES sub-restriction, build_train_manifest min-tiles
  filter.

### CPTAC ingest, IDC dry-run

- [`workflow/scripts/ingest_cptac_idc.py`](workflow/scripts/ingest_cptac_idc.py)
  ([b748927](https://github.com/ishaschhikara316/cx-histology/commit/b748927)).
  Replaces the broken TCIA-based `ingest_cptac.py`. Confirmed 7,377 SM
  series across 13 CPTAC pathology cohorts on IDC. Dry-run for
  CPTAC-LUAD (1,137 series) passes. Live ingest path documented but
  blocked on DICOM-WSI tiling adapter validation; see
  `docs/cptac_ingest_blockers.md`.

## 2026-04-19 — 3-cohort milestone

- First end-to-end run on DLBC + CHOL + UVM (n=87 labelled).
  CX1 ρ=0.36, CX3 ρ=0.33; both pass H1. Five signatures came back
  as undefined Spearman because of constant labels (the inactive
  signature problem that drove the 2026-05-03 mask fix).
  Commits leading up to this: `71d2c9d`, `79d70a8`, `3a765e1`,
  `78679a7`, `86751ba`, `f9fb21c`.
