# cx-histology

Predicting Drews 2022 chromosomal-instability (CX1–CX17) signatures from
H&E whole-slide images, pan-cancer, with interpretable attention-MIL on
frozen pathology foundation models.

**Status:** Under active development.
**Design spec:** `../docs/superpowers/specs/2026-04-19-cx-histology-design.md`
**Plan:** `../docs/superpowers/plans/2026-04-19-cx-histology-implementation.md`

## Quick start

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

## Labels

CX1–CX17 activities for 7,309 TCGA patients are produced upstream by
`cin-ici-mediation` and consumed here via `config/cohorts.yaml:tcga:labels`.
Do not recompute on TCGA.
