"""WSI ingest — manifest, download, tile extraction.

Glob pattern note: TCGA slide filenames are not deterministic from the
patient barcode (the UUID suffix is assigned by GDC). We therefore
snakemake-glob the downloaded .svs files after the download rule
completes and fan out tiling per discovered slide.
"""

rule gdc_manifest:
    output:
        tsv = INTERIM / "manifests" / "tcga_{cancer}.tsv"
    log:
        "logs/gdc_manifest_{cancer}.log"
    shell:
        "python workflow/scripts/gdc_manifest.py "
        "--project TCGA-{wildcards.cancer} "
        "--out {output.tsv} "
        "2> {log}"


rule download_wsi:
    input:
        manifest = INTERIM / "manifests" / "tcga_{cancer}.tsv"
    output:
        marker = touch(RAW / "tcga" / "{cancer}" / ".download_complete")
    log:
        "logs/download_wsi_{cancer}.log"
    threads: 4
    shell:
        # No --token: TCGA diagnostic slides are open-access.
        "python workflow/scripts/download_wsi.py "
        "--manifest {input.manifest} "
        "--out-dir {RAW}/tcga/{wildcards.cancer} "
        "--processes {threads} "
        "2> {log}"


def _slide_targets(wildcards):
    """After download, list expected tile HDF5 outputs by scanning raw dir."""
    import glob
    raw = RAW / "tcga" / wildcards.cancer
    svs = glob.glob(str(raw / "**/*.svs"), recursive=True)
    return [INTERIM / "tiles" / "tcga" / wildcards.cancer /
            f"{Path(p).stem}.h5" for p in svs]


rule tile_wsi:
    input:
        slide = RAW / "tcga" / "{cancer}" / "{slide_dir}" / "{slide}.svs",
        ref   = "resources/stain_reference.png"
    output:
        h5 = INTERIM / "tiles" / "tcga" / "{cancer}" / "{slide}.h5"
    log:
        "logs/tile_{cancer}_{slide}.log"
    resources:
        mem_mb = 4000
    shell:
        "python workflow/scripts/tile_wsi.py "
        "--slide {input.slide} "
        "--out {output.h5} "
        "--patch-px {config[tile][patch_size_px]} "
        "--mpp-target {config[tile][mpp_target]} "
        "--tissue-fraction-min {config[tile][tissue_fraction_min]} "
        "--tiles-per-slide {config[tile][tiles_per_slide]} "
        "--laplacian-var-min {config[tile][laplacian_var_min]} "
        "--stain-reference {input.ref} "
        "--seed {config[reproducibility][random_seed]} "
        "2> {log}"


rule ingest_cancer:
    """Per-cancer-type ingest: download + tile."""
    input:
        _slide_targets
    output:
        marker = touch(INTERIM / "tiles" / "tcga" / "{cancer}" / ".ingest_complete")


rule ingest_all:
    input:
        expand(INTERIM / "tiles" / "tcga" / "{cancer}" / ".ingest_complete",
               cancer=CANCER_TYPES),


# -----------------------------------------------------------------------
# CPTAC ingest (Task 15). Same tile primitives as TCGA; different source.
# -----------------------------------------------------------------------

rule ingest_cptac:
    output:
        manifest = RAW / "cptac" / "{cohort}" / "_manifest.tsv"
    log:
        "logs/ingest_cptac_{cohort}.log"
    shell:
        "python workflow/scripts/ingest_cptac.py "
        "--cohort CPTAC-{wildcards.cohort} "
        "--out-dir {RAW}/cptac/{wildcards.cohort} "
        "2> {log}"


def _cptac_tile_targets(wildcards):
    import glob
    raw = RAW / "cptac" / wildcards.cohort
    svs = glob.glob(str(raw / "**/*.svs"), recursive=True)
    return [INTERIM / "tiles" / "cptac" / wildcards.cohort /
            f"{Path(p).stem}.h5" for p in svs]


rule tile_cptac:
    input:
        slide = RAW / "cptac" / "{cohort}" / "{slide_dir}" / "{slide}.svs",
        ref   = "resources/stain_reference.png"
    output:
        h5 = INTERIM / "tiles" / "cptac" / "{cohort}" / "{slide}.h5"
    log:
        "logs/tile_cptac_{cohort}_{slide}.log"
    shell:
        "python workflow/scripts/tile_wsi.py "
        "--slide {input.slide} --out {output.h5} "
        "--patch-px {config[tile][patch_size_px]} "
        "--mpp-target {config[tile][mpp_target]} "
        "--tissue-fraction-min {config[tile][tissue_fraction_min]} "
        "--tiles-per-slide {config[tile][tiles_per_slide]} "
        "--laplacian-var-min {config[tile][laplacian_var_min]} "
        "--stain-reference {input.ref} "
        "--seed {config[reproducibility][random_seed]} "
        "2> {log}"


rule cptac_ingest_cohort:
    input:
        _cptac_tile_targets
    output:
        touch(INTERIM / "tiles" / "cptac" / "{cohort}" / ".ingest_complete")


rule cptac_ingest_all:
    input:
        expand(INTERIM / "tiles" / "cptac" / "{cohort}" / ".ingest_complete",
               cohort=CPTAC_COHORTS)
