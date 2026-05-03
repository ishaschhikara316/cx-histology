# cx-histology

Predicting Drews 2022 chromosomal-instability (CX1–CX17) signatures from
H&E whole-slide images, pan-cancer, with interpretable attention-MIL on
frozen pathology foundation models.

**Status:** Active development. 7-cohort UNI+ABMIL model trained, 8 more
cohorts ingesting toward a 15-cohort scaling run.

**Project narrative:** see [`docs/journey.md`](docs/journey.md).
**Design spec:** `../docs/superpowers/specs/2026-04-19-cx-histology-design.md`.

## Headline results (7-cohort, 2026-05-03)

Best HP from the sweep was hidden = 512, dropout = 0.3, lr = 1e-4
(UNI + gated ABMIL). On 289 labelled TCGA patients across 7 cohorts
(DLBC, CHOL, UVM, READ, MESO, UCS, KICH), 5-fold cancer-type stratified CV:

| Signature | Spearman ρ ± std | 95% CI | Note |
|-----------|------------------|--------|------|
| **CX3**  | **+0.384 ± 0.096** | [+0.300, +0.469] | ★ H1 |
| **CX1**  | **+0.366 ± 0.052** | [+0.322, +0.410] | ★ H1 |
| **CX12** | **+0.359 ± 0.048** | [+0.316, +0.396] | ★ H1 |
| CX6      | +0.271 ± 0.046     | [+0.231, +0.311] | CI &gt; 0 |
| CX10     | +0.236 ± 0.112     | [+0.163, +0.344] | CI &gt; 0 |
| CX17     | +0.205 ± 0.142     | [+0.074, +0.330] |          |
| CX2      | +0.193 ± 0.125     | [+0.068, +0.266] |          |
| CX5      | +0.189 ± 0.067     | [+0.131, +0.246] | CI &gt; 0 |
| CX4      | +0.171 ± 0.135     | [+0.066, +0.301] | CI &gt; 0 |
| CX16     | +0.149 ± 0.174     | [-0.017, +0.283] |          |
| CX9      | +0.125 ± 0.193     | [-0.039, +0.290] |          |
| CX11     | +0.112 ± 0.076     | [+0.048, +0.177] | CI &gt; 0 |
| CX8      | +0.087 ± 0.092     | [+0.005, +0.159] | CI &gt; 0 |
| CX7, CX13–15 | n/a            | inactive in TCGA |     |

**Mean active ρ: +0.219** (3 H1 hits, 8 signatures with bootstrap CI &gt; 0).
This is a 5× lift over the 3-cohort baseline (mean ρ +0.024). See
[`figures/compare_3vs7_cohort_rho.pdf`](figures/compare_3vs7_cohort_rho.pdf)
for the side-by-side comparison.

Open the **interactive dashboard** at
[`figures/dashboard_7cohort.html`](figures/dashboard_7cohort.html)
for the full sweep, robustness table, and references.

## Quick start (this checkpoint, no conda)

If you have a working pip Python with CUDA torch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install h5py scipy scikit-learn matplotlib seaborn click tqdm pyyaml \
            timm transformers huggingface_hub einops opencv-python \
            scikit-image torch-staintools openslide-python pyvips
apt-get install -y libvips-tools libopenslide0 openslide-tools
# gdc-client v1.6.1 binary into /usr/local/bin
echo "$YOUR_HF_TOKEN" > ~/.hf_token && chmod 600 ~/.hf_token
```

## Quick start (canonical conda path)

```bash
conda env create -f environment.yml
conda activate cx-histology
snakemake --cores 4 all
```

## Entry points

- `snakemake --cores 8 ingest_all` — download WSIs + tile extraction
- `snakemake --cores 8 embed_all` — foundation-model tile embeddings
- `snakemake --cores 1 train_all` — 5-fold CV ABMIL training
- `snakemake --cores 4 predict_cptac` — external CPTAC validation
- `snakemake --cores 4 interpret_all` — attention + nuclear-feature enrichment
- `snakemake --cores 2 figures_all` — main figures + dashboard

For unattended multi-cohort runs:

```bash
bash workflow/scripts/parallel_ingest_8cohort.sh   # 2-group 4-GPU ingest
bash workflow/scripts/finalize_15cohort.sh         # auto-trains when done
```

## Labels

CX1–CX17 activities for 7,309 TCGA patients are produced upstream by
`cin-ici-mediation` and consumed here via `config/cohorts.yaml:tcga:labels`.
Do not recompute on TCGA. Four signatures (CX7, CX13–15) are essentially
zero pan-TCGA and are masked out per fold by training-label variance;
see `docs/methods.md` for the threshold and rationale.

## Results

Interactive dashboard: `figures/dashboard_7cohort.html` (open locally).

Key figures:
- `figures/compare_3vs7_cohort_rho.{pdf,png}` — cohort-scaling effect
- `figures/fig2_hp_sweep_heatmap.{pdf,png}` — HP combos × signatures heatmap
- `figures/fig3_robustness_table.{pdf,png}` — per-signature stats (best HP)
- `figures/prelim_7cohort_rho.{pdf,png}` — best-HP per-signature bar
- `figures/fig1_cohort_coverage.pdf` — TCGA labelled-patient coverage

Per-signature ρ on held-out TCGA: see
`results/cv/uni_7cohort_sweep/h512_d0.3_lr0.0001/cv_summary.json`
and `robustness.json` next to it.

## Reproducing the pipeline

1. Set up the environment (see Quick start sections above).
2. On some systems libvips/libstdc++ may be shadowed by the host; if
   pytest collection fails, export
   `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` before
   invoking `pytest` or `snakemake`.
3. Place `~/.hf_token` (chmod 600) on disk. TCGA diagnostic slides are
   open-access so a `~/.gdc_token` is not required for the cohorts in
   `config/cohorts.yaml`.
4. Ensure CX activity labels are at `data/labels/cx_activities.tsv`
   (or `../cin-ici-mediation/results/tcga/cx_activities.tsv`).
5. Run stages in order:
   ```
   snakemake --cores 8 ingest_all
   snakemake --cores 8 embed_all
   snakemake --cores 1 train_all
   snakemake --cores 4 predict_cptac
   snakemake --cores 4 interpret_all
   snakemake --cores 2 figures_all
   ```

## Hardware footprint

Stream-mode ingest keeps peak disk under ~20 GB regardless of cohort
size. Final embeddings are ~300 MB per cohort, ~10 GB for full
pan-TCGA. Tested on 4× RTX 4090 (96 GB aggregate VRAM), 122-core
EPYC, 126 GB system disk; sustained 70–85 MB/s per GDC connection.
NVMe rental no longer required.

## Citations

If you use this code or its outputs, please cite the source models and
labels you depend on (full list in `docs/methods.md`):

- Drews et al., *Nature* 2022 — CIN signatures
- Chen et al., *Nature Medicine* 2024 — UNI foundation model
- Ilse et al., *ICML* 2018 — Attention-based MIL
- Macenko et al., *ISBI* 2009 — stain normalisation
