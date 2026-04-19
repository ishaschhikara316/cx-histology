"""ABMIL training + hyperparameter sweep."""

# Hyperparameter sweep grid. Use string literals for LR so wildcard values
# appear consistently in file paths (e.g. lr0.0001 not lr1e-4).
HP_HIDDEN  = [128, 256, 512]
HP_DROPOUT = [0.2, 0.3, 0.5]
HP_LR      = ["0.0001", "0.0005"]


rule train_manifest:
    input:
        marker = expand(INTERIM / "embeddings" / "tcga" / "{{encoder}}" / "{cancer}" / ".done",
                        cancer=CANCER_TYPES)
    output:
        tsv = INTERIM / "manifests" / "train_{encoder}.tsv"
    log:
        "logs/train_manifest_{encoder}.log"
    shell:
        "python workflow/scripts/build_train_manifest.py "
        "--embed-root {INTERIM}/embeddings/tcga "
        "--encoder {wildcards.encoder} "
        "--out {output.tsv} "
        "2> {log}"


rule train_cv_one:
    input:
        manifest = INTERIM / "manifests" / "train_{encoder}.tsv",
        labels   = COHORTS["tcga"]["labels"]
    output:
        summary = RESULTS / "cv" / "{encoder}" /
                  "h{hidden}_d{dropout}_lr{lr}" / "cv_summary.json"
    log:
        "logs/train_{encoder}_h{hidden}_d{dropout}_lr{lr}.log"
    resources:
        gpu = 1, mem_mb = 32000
    shell:
        "python workflow/scripts/train_cv.py "
        "--manifest {input.manifest} "
        "--labels {input.labels} "
        "--encoder {wildcards.encoder} "
        "--out-dir {RESULTS}/cv/{wildcards.encoder}/h{wildcards.hidden}_d{wildcards.dropout}_lr{wildcards.lr} "
        "--hidden {wildcards.hidden} "
        "--dropout {wildcards.dropout} "
        "--lr {wildcards.lr} "
        "2> {log}"


rule train_sweep_encoder:
    """All HP combinations for a single encoder."""
    input:
        expand(RESULTS / "cv" / "{{encoder}}" /
               "h{hidden}_d{dropout}_lr{lr}" / "cv_summary.json",
               hidden=HP_HIDDEN, dropout=HP_DROPOUT, lr=HP_LR)
    output:
        touch(RESULTS / "cv" / "{encoder}" / ".sweep_done")


rule train_all:
    input:
        expand(RESULTS / "cv" / "{encoder}" / ".sweep_done",
               encoder=ENCODERS)
