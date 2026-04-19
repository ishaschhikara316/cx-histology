"""Figure + dashboard assembly."""

rule fig1:
    input:
        labels   = COHORTS["tcga"]["labels"],
        manifest = INTERIM / "manifests" / "train_uni.tsv"
    output:
        pdf = FIGURES / "fig1_pipeline.pdf"
    shell:
        "python workflow/scripts/fig1_pipeline.py "
        "--labels {input.labels} --manifest {input.manifest} --out {output.pdf}"


rule fig2:
    input:
        uni  = RESULTS / "cv" / "uni"         / "h256_d0.3_lr0.0001" / "cv_summary.json",
        hopt = RESULTS / "cv" / "h_optimus_0" / "h256_d0.3_lr0.0001" / "cv_summary.json"
    output:
        pdf = FIGURES / "fig2_benchmark.pdf"
    shell:
        "python workflow/scripts/fig2_benchmark.py "
        "--summary-uni {input.uni} --summary-hopt {input.hopt} --out {output.pdf}"


rule fig3:
    input:
        tcga_summary = RESULTS / "cv" / "uni" / "h256_d0.3_lr0.0001" / "cv_summary.json",
        cptac_preds  = RESULTS / "cptac" / "uni" / "h256_d0.3_lr0.0001" / "predictions.tsv",
        cptac_labels = RESULTS / "cptac" / "cx_activities.tsv"
    output:
        pdf = FIGURES / "fig3_external.pdf"
    shell:
        "python workflow/scripts/fig3_external.py "
        "--tcga-summary {input.tcga_summary} "
        "--cptac-preds {input.cptac_preds} "
        "--cptac-labels {input.cptac_labels} "
        "--out {output.pdf}"


rule fig4:
    input:
        enrichment = RESULTS / "interpret" / "uni" / "enrichment_all.tsv",
        heatmap    = RESULTS / "interpret" / "uni" / "representative_heat.npy",
        thumb      = RESULTS / "interpret" / "uni" / "representative_thumb.png",
        prototypes = RESULTS / "interpret" / "uni" / "prototypes" / "cx1_top.png"
    output:
        pdf = FIGURES / "fig4_interpret.pdf"
    shell:
        "python workflow/scripts/fig4_interpret.py "
        "--enrichment {input.enrichment} --heatmap {input.heatmap} "
        "--thumb {input.thumb} --prototypes {input.prototypes} "
        "--out {output.pdf}"


rule dashboard:
    input:
        bench      = RESULTS / "cv" / "uni" / "h256_d0.3_lr0.0001" / "cv_summary.json",
        cptac_rho  = RESULTS / "cptac" / "uni" / "h256_d0.3_lr0.0001" / "predictions.tsv",
        prototypes = RESULTS / "interpret" / "uni" / "prototypes"
    output:
        html = FIGURES / "dashboard.html"
    shell:
        "python workflow/scripts/build_dashboard.py "
        "--bench-json {input.bench} --cptac-rho-tsv {input.cptac_rho} "
        "--prototypes-dir {input.prototypes} --out {output.html}"


rule figures_all:
    input:
        FIGURES / "fig1_pipeline.pdf",
        FIGURES / "fig2_benchmark.pdf",
        FIGURES / "fig3_external.pdf",
        FIGURES / "fig4_interpret.pdf",
        FIGURES / "dashboard.html",
