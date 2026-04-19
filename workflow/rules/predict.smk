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
