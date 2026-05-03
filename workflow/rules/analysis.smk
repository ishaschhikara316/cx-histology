"""Analysis rules for the 7-cohort sweep best-HP results.

Each rule produces a small JSON/TSV plus one or two figures. They run
on CPU in seconds-to-minutes. Wire `analysis_all` into the top-level
target list to regenerate everything from a clean checkout.

Inputs assume the best-HP cv_summary.json and fold0..4 checkpoints
already exist (see train_cv.py + plan_a_orchestrator.sh).
"""

BEST = "uni_7cohort_sweep/h512_d0.3_lr0.0001"
BEST_DIR = RESULTS / "cv" / BEST
LABELS = "data/labels/cx_activities.tsv"
SUMMARY = BEST_DIR / "cv_summary.json"
PREDICTIONS = BEST_DIR / "predictions.tsv"


# Cohort comparison: 3-cohort vs 7-cohort headline number.
rule fig_compare_3vs7:
    input:
        old = RESULTS / "cv" / "uni_3cohort" / "cv_summary.json",
        new = SUMMARY,
    output:
        pdf = FIGURES / "compare_3vs7_cohort_rho.pdf",
        png = FIGURES / "compare_3vs7_cohort_rho.png",
    shell:
        "python workflow/scripts/fig_compare_3vs7.py"


# HP sweep heatmap across all 6 combos.
rule fig_hp_sweep_heatmap:
    output:
        pdf = FIGURES / "fig2_hp_sweep_heatmap.pdf",
        png = FIGURES / "fig2_hp_sweep_heatmap.png",
    shell:
        "python workflow/scripts/fig2_hp_sweep_heatmap.py"


# Robustness: BH-FDR + bootstrap CI per signature (fold-level).
rule robustness_analysis:
    input:
        summary = SUMMARY,
    output:
        json = BEST_DIR / "robustness.json",
        pdf = FIGURES / "fig3_robustness_table.pdf",
        png = FIGURES / "fig3_robustness_table.png",
    shell:
        "python workflow/scripts/robustness_analysis.py"


# Per-cohort breakdown: re-run inference and compute per-cohort rho.
rule per_cohort_breakdown:
    input:
        summary = SUMMARY,
    output:
        predictions = PREDICTIONS,
        rho_per = BEST_DIR / "rho_per_cohort.tsv",
        pdf = FIGURES / "fig_per_cohort_rho.pdf",
        png = FIGURES / "fig_per_cohort_rho.png",
    shell:
        "python workflow/scripts/per_cohort_breakdown.py"


# Patient-level bootstrap on the saved predictions.
rule patient_bootstrap:
    input:
        predictions = PREDICTIONS,
    output:
        json = BEST_DIR / "patient_bootstrap.json",
        pdf = FIGURES / "fig_patient_bootstrap_forest.pdf",
        png = FIGURES / "fig_patient_bootstrap_forest.png",
    shell:
        "python workflow/scripts/patient_bootstrap.py"


# Calibration analysis on top-5 q<0.001 signatures.
rule calibration_analysis:
    input:
        predictions = PREDICTIONS,
    output:
        json = BEST_DIR / "calibration_metrics.json",
        pdf = FIGURES / "fig_calibration_top5.pdf",
        png = FIGURES / "fig_calibration_top5.png",
    shell:
        "python workflow/scripts/calibration_analysis.py"


# Outlier patient analysis.
rule outlier_analysis:
    input:
        predictions = PREDICTIONS,
        bootstrap = BEST_DIR / "patient_bootstrap.json",
    output:
        tsv = BEST_DIR / "outlier_patients.tsv",
        pdf = FIGURES / "fig_outliers_per_cohort.pdf",
        png = FIGURES / "fig_outliers_per_cohort.png",
    shell:
        "python workflow/scripts/outlier_analysis.py"


# Subgroup analysis (sex / age / stage), needs network for GDC clinical.
rule subgroup_analysis:
    input:
        predictions = PREDICTIONS,
    output:
        json = BEST_DIR / "subgroup_metrics.json",
        clin = BEST_DIR / "clinical_metadata.tsv",
        pdf = FIGURES / "fig_subgroup_breakdown.pdf",
        png = FIGURES / "fig_subgroup_breakdown.png",
    shell:
        "python workflow/scripts/subgroup_analysis.py"


# CX label crosstalk and per-cohort distribution (pan-TCGA).
rule cx_crosstalk:
    input:
        labels = LABELS,
    output:
        json = RESULTS / "cx_label_atlas.json",
        crosstalk_pdf = FIGURES / "fig_cx_crosstalk.pdf",
        crosstalk_png = FIGURES / "fig_cx_crosstalk.png",
        per_cohort_pdf = FIGURES / "fig_cx_per_cohort.pdf",
        per_cohort_png = FIGURES / "fig_cx_per_cohort.png",
    shell:
        "python workflow/scripts/cx_crosstalk_analysis.py"


# Survival analysis: Cox + KM per signature, network for GDC OS data.
rule survival_analysis:
    input:
        predictions = PREDICTIONS,
    output:
        json = BEST_DIR / "survival_metrics.json",
        forest_pdf = FIGURES / "fig_survival_forest.pdf",
        forest_png = FIGURES / "fig_survival_forest.png",
        km_pdf = FIGURES / "fig_survival_km_cx1.pdf",
        km_png = FIGURES / "fig_survival_km_cx1.png",
    shell:
        "python workflow/scripts/survival_analysis.py"


# TP53 mutation correlation, network for GDC ssm_occurrences.
rule mutation_correlation:
    input:
        predictions = PREDICTIONS,
    output:
        json = BEST_DIR / "mutation_metrics.json",
        pdf = FIGURES / "fig_tp53_vs_cx1.pdf",
        png = FIGURES / "fig_tp53_vs_cx1.png",
    shell:
        "python workflow/scripts/mutation_correlation.py"


# Power simulation: subsample-based, ~10 minutes CPU.
rule power_simulation:
    input:
        predictions = PREDICTIONS,
    output:
        json = BEST_DIR / "power_curves.json",
        pdf = FIGURES / "fig_power_curves.pdf",
        png = FIGURES / "fig_power_curves.png",
    shell:
        "python workflow/scripts/power_simulation.py"


# Self-contained 7-cohort dashboard.
rule dashboard_7cohort:
    input:
        rules.robustness_analysis.output.json,
        rules.fig_compare_3vs7.output.png,
        rules.fig_hp_sweep_heatmap.output.png,
    output:
        html = FIGURES / "dashboard_7cohort.html",
    shell:
        "python workflow/scripts/build_dashboard_7cohort.py"


rule analysis_all:
    input:
        rules.fig_compare_3vs7.output.pdf,
        rules.fig_hp_sweep_heatmap.output.pdf,
        rules.robustness_analysis.output.json,
        rules.per_cohort_breakdown.output.predictions,
        rules.patient_bootstrap.output.json,
        rules.calibration_analysis.output.json,
        rules.outlier_analysis.output.tsv,
        rules.subgroup_analysis.output.json,
        rules.cx_crosstalk.output.json,
        rules.survival_analysis.output.json,
        rules.mutation_correlation.output.json,
        rules.power_simulation.output.json,
        rules.dashboard_7cohort.output.html,
