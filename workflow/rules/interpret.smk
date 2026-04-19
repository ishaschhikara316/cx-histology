"""Interpretability: attention heatmaps, nuclear enrichment, prototype tiles."""

rule attention_heatmap:
    input:
        emb  = INTERIM / "embeddings" / "tcga" / "{encoder}" / "{cancer}" / "{slide}.h5",
        ckpt = RESULTS / "cv" / "{encoder}" / "h256_d0.3_lr0.0001" / "fold0.pt"
    output:
        npy = RESULTS / "interpret" / "{encoder}" / "{cancer}" / "{slide}_heat.npy"
    log:
        "logs/heat_{encoder}_{cancer}_{slide}.log"
    shell:
        "python workflow/scripts/attention_rollout.py "
        "--emb {input.emb} --ckpt {input.ckpt} "
        "--slide-wh 40000,40000 "   # overridden if per-slide metadata available
        "--out {output.npy} "
        "2> {log}"


rule nuclear_enrichment:
    input:
        tiles = INTERIM / "tiles" / "tcga" / "{cancer}" / "{slide}.h5",
        heat  = RESULTS / "interpret" / "{encoder}" / "{cancer}" / "{slide}_heat.npy"
    output:
        tsv = RESULTS / "interpret" / "{encoder}" / "{cancer}" / "{slide}_enrich.tsv"
    log:
        "logs/enrich_{encoder}_{cancer}_{slide}.log"
    shell:
        "python workflow/scripts/nuclear_features.py "
        "--tiles {input.tiles} --attention {input.heat} "
        "--out {output.tsv} --top-k 50 "
        "2> {log}"


rule prototype_tiles:
    input:
        ckpt     = RESULTS / "cv" / "{encoder}" / "h256_d0.3_lr0.0001" / "fold0.pt",
        manifest = INTERIM / "manifests" / "interpret_{encoder}.tsv"
    output:
        touch(RESULTS / "interpret" / "{encoder}" / ".prototypes_done")
    shell:
        "python workflow/scripts/prototype_tiles.py "
        "--manifest {input.manifest} --ckpt {input.ckpt} "
        "--out-dir {RESULTS}/interpret/{wildcards.encoder}/prototypes "
        "--top-k 5 --device cuda:0"


rule interpret_all:
    input:
        expand(RESULTS / "interpret" / "{encoder}" / ".prototypes_done",
               encoder=ENCODERS)
