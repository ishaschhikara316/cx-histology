"""WSI ingest — manifest + streaming download-tile-delete loop.

TCGA is streamed: one slide at a time is downloaded, tiled, and the raw
.svs is deleted, so peak disk footprint is bounded by a single slide
plus accumulated .h5 outputs. This makes ingest viable on modest disks
(~95 GB) that couldn't hold the whole raw cohort at once.
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


rule stream_ingest_cancer:
    """Download-tile-delete loop for one TCGA cancer type.

    Output is just a completion marker; per-slide .h5 files are
    discovered downstream (embed.smk) by globbing the tiles directory.
    Resumes by skipping slides whose .h5 already exists.
    """
    input:
        manifest = INTERIM / "manifests" / "tcga_{cancer}.tsv"
    output:
        marker = INTERIM / "tiles" / "tcga" / "{cancer}" / ".ingest_complete"
    log:
        "logs/stream_ingest_{cancer}.log"
    params:
        ref = "resources/stain_reference.png"
    shell:
        "python workflow/scripts/stream_ingest.py "
        "--manifest {input.manifest} "
        "--raw-dir {RAW}/tcga/{wildcards.cancer} "
        "--tiles-dir {INTERIM}/tiles/tcga/{wildcards.cancer} "
        "--stain-reference {params.ref} "
        "--patch-px {config[tile][patch_size_px]} "
        "--mpp-target {config[tile][mpp_target]} "
        "--tissue-fraction-min {config[tile][tissue_fraction_min]} "
        "--tiles-per-slide {config[tile][tiles_per_slide]} "
        "--laplacian-var-min {config[tile][laplacian_var_min]} "
        "--seed {config[reproducibility][random_seed]} "
        "--marker {output.marker} "
        "2> {log}"


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
