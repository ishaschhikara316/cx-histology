"""Per-slide foundation-model embeddings, parameterised by encoder."""

rule embed_tiles:
    input:
        tiles = INTERIM / "tiles" / "tcga" / "{cancer}" / "{slide}.h5"
    output:
        emb = INTERIM / "embeddings" / "tcga" / "{encoder}" / "{cancer}" / "{slide}.h5"
    log:
        "logs/embed_{encoder}_{cancer}_{slide}.log"
    resources:
        gpu = 1, mem_mb = 16000
    shell:
        "python workflow/scripts/embed_tiles.py "
        "--tiles {input.tiles} "
        "--out {output.emb} "
        "--encoder {wildcards.encoder} "
        "--models-config config/models.yaml "
        "--batch-size 32 "
        "--device cuda:0 "
        "2> {log}"


def _embed_targets(wildcards):
    import glob
    pat = str(INTERIM / "tiles" / "tcga" / wildcards.cancer / "*.h5")
    hits = glob.glob(pat)
    return [INTERIM / "embeddings" / "tcga" / wildcards.encoder /
            wildcards.cancer / Path(p).name for p in hits]


rule embed_cancer:
    input:
        _embed_targets
    output:
        touch(INTERIM / "embeddings" / "tcga" / "{encoder}" / "{cancer}" / ".done")


rule embed_all:
    input:
        expand(INTERIM / "embeddings" / "tcga" / "{encoder}" / "{cancer}" / ".done",
               encoder=ENCODERS, cancer=CANCER_TYPES)
