"""CPTAC external prediction + metrics vs. recomputed CX labels."""

rule embed_cptac:
    input:
        tiles = INTERIM / "tiles" / "cptac" / "{cohort}" / "{slide}.h5"
    output:
        emb = INTERIM / "embeddings" / "cptac" / "{encoder}" / "{cohort}" / "{slide}.h5"
    log:
        "logs/embed_cptac_{encoder}_{cohort}_{slide}.log"
    resources:
        gpu = 1
    shell:
        "python workflow/scripts/embed_tiles.py "
        "--tiles {input.tiles} --out {output.emb} "
        "--encoder {wildcards.encoder} "
        "--models-config config/models.yaml "
        "--batch-size 32 --device cuda:0 "
        "2> {log}"


rule cptac_manifest:
    output:
        tsv = INTERIM / "manifests" / "cptac_{encoder}.tsv"
    params:
        root = lambda _: INTERIM / "embeddings" / "cptac"
    shell:
        "python workflow/scripts/build_train_manifest.py "
        "--embed-root {params.root} --encoder {wildcards.encoder} "
        "--out {output.tsv}"


rule cptac_cx_labels:
    output:
        labels = RESULTS / "cptac" / "cx_activities.tsv"
    log:
        "logs/cptac_cx_labels.log"
    shell:
        "Rscript workflow/scripts/score_cx_cptac.R "
        "--segments {RAW}/cptac/absolute/segments_merged.tsv "
        "--out {output.labels} "
        "2> {log}"


rule predict_cptac_one_ckpt:
    input:
        manifest = INTERIM / "manifests" / "cptac_{encoder}.tsv",
        ckpt     = RESULTS / "cv" / "{encoder}" /
                   "h{hidden}_d{dropout}_lr{lr}" / "fold0.pt"
    output:
        preds = RESULTS / "cptac" / "{encoder}" /
                "h{hidden}_d{dropout}_lr{lr}" / "predictions.tsv"
    log:
        "logs/predict_cptac_{encoder}_h{hidden}_d{dropout}_lr{lr}.log"
    resources:
        gpu = 1
    shell:
        "python workflow/scripts/predict_external.py "
        "--manifest {input.manifest} --ckpt {input.ckpt} "
        "--out {output.preds} --device cuda:0 "
        "2> {log}"


rule predict_cptac:
    """Inference with the best HP combination — selected post-sweep manually
    or by a picker script; here we assume the h256_d0.3_lr0.0001 baseline."""
    input:
        expand(RESULTS / "cptac" / "{encoder}" /
               "h256_d0.3_lr0.0001" / "predictions.tsv", encoder=ENCODERS),
        RESULTS / "cptac" / "cx_activities.tsv"


# -----------------------------------------------------------------------
# Hugo 2016 stretch (Task 26). Runs only if ~27 slides are placed at
# data/raw/hugo_2016/slides/ and the predictor already holds up on CPTAC.
# -----------------------------------------------------------------------

rule tile_hugo:
    input:
        slide = RAW / "hugo_2016" / "slides" / "{slide}.svs",
        ref   = "resources/stain_reference.png"
    output:
        h5 = INTERIM / "tiles" / "hugo_2016" / "{slide}.h5"
    shell:
        "python workflow/scripts/tile_wsi.py "
        "--slide {input.slide} --out {output.h5} "
        "--patch-px 256 --mpp-target 0.5 "
        "--tissue-fraction-min 0.5 --tiles-per-slide 1000 "
        "--laplacian-var-min 100 --stain-reference {input.ref} "
        "--seed {config[reproducibility][random_seed]}"


rule embed_hugo:
    input:
        tiles = INTERIM / "tiles" / "hugo_2016" / "{slide}.h5"
    output:
        emb = INTERIM / "embeddings" / "hugo_2016" / "{encoder}" / "{slide}.h5"
    resources: gpu = 1
    shell:
        "python workflow/scripts/embed_tiles.py "
        "--tiles {input.tiles} --out {output.emb} "
        "--encoder {wildcards.encoder} "
        "--models-config config/models.yaml "
        "--batch-size 32 --device cuda:0"


rule hugo_manifest:
    output:
        tsv = INTERIM / "manifests" / "hugo_{encoder}.tsv"
    shell:
        "python workflow/scripts/build_train_manifest.py "
        "--embed-root {INTERIM}/embeddings/hugo_2016 --encoder {wildcards.encoder} "
        "--out {output.tsv}"


rule predict_hugo:
    input:
        manifest = INTERIM / "manifests" / "hugo_{encoder}.tsv",
        ckpt     = RESULTS / "cv" / "{encoder}" / "h256_d0.3_lr0.0001" / "fold0.pt"
    output:
        preds = RESULTS / "hugo_2016" / "{encoder}" / "predictions.tsv"
    resources: gpu = 1
    shell:
        "python workflow/scripts/predict_external.py "
        "--manifest {input.manifest} --ckpt {input.ckpt} "
        "--out {output.preds} --device cuda:0"
