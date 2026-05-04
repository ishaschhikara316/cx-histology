"""Build dashboard_21cohort.html — replaces dashboard_7cohort.html as the
canonical interactive results page. Embeds:
  - Headline numbers + the 7->15->21 cohort scaling story
  - HP sweep leaderboard (from uni_15cohort_sweep)
  - Per-cohort breakdown + within-cohort rho
  - Patient-level bootstrap forest plot
  - Calibration scatter
  - Subgroup performance (age/stage/sex)
  - Outlier patients
  - Survival KM (predicted CX1 quartile vs OS)
  - Interpretability previews (attention + prototype tiles)
  - CPTAC external validation status (when available)

Reads JSON outputs from results/cv/uni_21cohort_final/.
Writes figures/dashboard_21cohort.html (self-contained).
"""
from __future__ import annotations
import base64
import json
from pathlib import Path

ROOT = Path("/root/ISHA/cx-histology")
D21 = ROOT / "results/cv/uni_21cohort_final"
SWEEP15 = ROOT / "results/cv/uni_15cohort_sweep"
FIG = ROOT / "figures"


def _b64(p: Path) -> str:
    if not p.exists(): return ""
    return base64.b64encode(p.read_bytes()).decode("ascii")


def _img(p: Path, alt: str = "", width: int = 720) -> str:
    if not p.exists():
        return f"<p><em>(figure not yet generated: {p.name})</em></p>"
    mime = "image/png" if p.suffix == ".png" else "image/jpeg"
    return (f'<img alt="{alt}" width="{width}" '
            f'src="data:{mime};base64,{_b64(p)}" />')


def _safe_json(p: Path) -> dict:
    try: return json.loads(p.read_text())
    except Exception: return {}


def main():
    cv_summary = _safe_json(D21 / "cv_summary.json")
    bootstrap  = _safe_json(D21 / "patient_bootstrap.json")
    cohort_scaling = _safe_json(D21 / "cohort_scaling_summary.json")
    sweep15 = _safe_json(SWEEP15 / "sweep_summary.json")
    survival = _safe_json(D21 / "survival_metrics.json")
    calibration = _safe_json(D21 / "calibration_metrics.json")
    subgroup = _safe_json(D21 / "subgroup_metrics.json")

    import numpy as np
    mean = np.array(cv_summary.get("pan_cancer_spearman_mean", []), dtype=float) \
        if cv_summary.get("pan_cancer_spearman_mean") else np.array([])

    sigs = [f"CX{i}" for i in range(1, 18)]

    html = []
    html.append("""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>cx-histology · 21-cohort results</title>
<style>
  body { font: 14px/1.45 -apple-system, BlinkMacSystemFont, sans-serif;
         max-width: 980px; margin: 24px auto; padding: 0 18px; color: #222; }
  h1 { font-size: 28px; }
  h2 { font-size: 22px; border-bottom: 1px solid #ddd; padding-bottom: 6px;
       margin-top: 36px; }
  h3 { font-size: 18px; margin-top: 24px; }
  table { border-collapse: collapse; margin: 12px 0; }
  th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; }
  th { background: #fafafa; }
  .star { color: #2ca02c; font-weight: bold; }
  .pos  { color: #ff7f0e; }
  .neg  { color: #d62728; }
  .meta { color: #666; font-size: 12px; }
  pre { background: #f8f8f8; padding: 10px; border-radius: 4px; overflow-x: auto; }
  .caveat { background: #fff8e1; border-left: 4px solid #ffb300; padding: 10px; }
</style>
</head><body>
""")
    html.append(f"<h1>cx-histology — 21-cohort results</h1>")
    html.append('<p class="meta">Pan-cancer prediction of Drews 2022 chromosomal-instability '
                'signatures (CX1–CX17) from H&E whole-slide images, frozen UNI tile embeddings, '
                'gated attention-MIL head, 5-fold cancer-type-stratified CV.</p>')

    if mean.size > 0:
        n_h1 = int(np.sum(mean >= 0.30))
        finite = mean[~np.isnan(mean)]
        html.append("<h2>Headline result</h2>")
        html.append(f"<p>Mean active ρ = <b>{np.nanmean(mean):+.3f}</b> across "
                    f"{len(finite)} active signatures. <b>{n_h1} H1 hits</b> (ρ ≥ 0.30).</p>")
        html.append("<table><thead><tr><th>Signature</th><th>ρ</th><th>std</th></tr></thead><tbody>")
        std = np.array(cv_summary.get("pan_cancer_spearman_std", []), dtype=float)
        for i, sig in enumerate(sigs):
            v = mean[i] if i < len(mean) else float("nan")
            sd = std[i] if i < len(std) else float("nan")
            cls = "star" if not np.isnan(v) and v >= 0.30 else ("pos" if not np.isnan(v) and v >= 0 else "neg")
            html.append(f'<tr><td>{sig}</td><td class="{cls}">{v:+.3f}</td><td>{sd:.3f}</td></tr>')
        html.append("</tbody></table>")
        html.append(_img(FIG / "final_21cohort_rho.png", "21-cohort per-signature rho", 720))

    if cohort_scaling:
        html.append("<h2>The cohort-scaling story</h2>")
        html.append('<p>Three model snapshots on the same prediction task — 7 cohorts (n=289), '
                    '15 cohorts (n=1,454), 21 cohorts. Smaller cohort sets achieve larger '
                    'pan-cohort ρ partly because cancer-type heterogeneity is doing the work.</p>')
        cs = cohort_scaling
        html.append("<table><tr><th>Setting</th><th>Mean ρ</th><th>n</th></tr>")
        rows = [
            ("7-cohort native (own n=289 patients)",  cs.get("native_7cohort_on_7", {})),
            ("15-cohort native (own n=1,454 patients)", cs.get("native_15cohort_on_15", {})),
            ("21-cohort native",                      cs.get("native_21cohort_on_21", {})),
            ("21-cohort model on 15-cohort subset",   cs.get("21cohort_on_15subset", {})),
            ("21-cohort model on 7-cohort subset",    cs.get("21cohort_on_7subset", {})),
        ]
        for label, d in rows:
            r = d.get("rho")
            n = d.get("n", "—")
            r_s = f"{r:+.3f}" if r is not None else "—"
            html.append(f'<tr><td>{label}</td><td>{r_s}</td><td>{n}</td></tr>')
        html.append("</table>")
        html.append('<div class="caveat">Pan-cohort ρ has a strong cancer-type-classification '
                    'component. As more cohorts are added, the within-biology signal becomes '
                    'the dominant explanation, and effect sizes shrink toward their unbiased '
                    'estimate. The 21-cohort number is the most honest estimate of the H&E→CIN '
                    'biology link.</div>')

    if sweep15.get("rows"):
        html.append("<h2>HP sweep at n=1,454 (informed the 21-cohort HP choice)</h2>")
        html.append("<table><tr><th>Combo</th><th>Mean ρ</th><th>H1 hits</th></tr>")
        for r in sweep15["rows"][:10]:
            html.append(f'<tr><td>{r["hp"]}</td><td>{r["mean_active_rho"]:+.4f}</td><td>{r["n_h1"]}</td></tr>')
        html.append("</table>")
        html.append("<p>Best HP: <code>" + sweep15["rows"][0]["hp"] + "</code> "
                    f"at ρ = {sweep15['rows'][0]['mean_active_rho']:+.4f}. Used for 21-cohort.</p>")

    if bootstrap.get("signatures"):
        html.append("<h2>Patient-level bootstrap (n_boot=10,000)</h2>")
        html.append(_img(FIG / "fig_patient_bootstrap_forest.png", "21-cohort bootstrap forest", 720))

    html.append("<h2>Per-cohort breakdown</h2>")
    html.append(_img(FIG / "fig_per_cohort_rho.png", "21-cohort per-cohort rho", 720))

    html.append("<h2>Calibration</h2>")
    html.append(_img(FIG / "fig_calibration_top5.png", "21-cohort calibration", 720))

    html.append("<h2>Subgroup performance</h2>")
    html.append(_img(FIG / "fig_subgroup_breakdown.png", "21-cohort subgroup", 720))

    html.append("<h2>Survival association (clinical utility)</h2>")
    if survival:
        html.append("<p>Cox + KM stratified by predicted-CX1 quartile, cohort-stratified.</p>")
        html.append(_img(FIG / "fig_survival_km_top3.png", "21-cohort survival", 720))
        html.append(_img(FIG / "fig_survival_forest.png", "21-cohort survival forest", 720))
    else:
        html.append("<p><em>(survival_analysis pending or failed)</em></p>")

    html.append("<h2>Interpretability</h2>")
    html.append("<h3>Attention rollout</h3>")
    html.append(_img(FIG / "fig4_interpret_attention.png", "attention rollout", 720))
    html.append("<h3>Prototype tiles per signature</h3>")
    html.append(_img(FIG / "fig4_interpret_prototypes.png", "prototype tiles", 720))
    html.append("<h3>Nuclear features in attended regions</h3>")
    html.append(_img(FIG / "fig4_interpret_nuclear.png", "nuclear features", 720))

    html.append("<h2>External validation (CPTAC)</h2>")
    h2_path = D21 / "h2_test.json"
    if h2_path.exists():
        h2 = json.loads(h2_path.read_text())
        passed = h2.get("h2_passed", False)
        html.append(f"<p>H2 test |Δρ TCGA−CPTAC| ≤ 0.15: <b>{'PASSED' if passed else 'FAILED'}</b></p>")
        html.append(f"<p>TCGA mean ρ: {h2.get('tcga_mean_rho','?'):+.3f}, "
                    f"CPTAC mean ρ: {h2.get('cptac_mean_rho','?'):+.3f}, "
                    f"|Δρ|: {h2.get('abs_delta','?'):.3f}</p>")
    else:
        html.append("<p><em>CPTAC ingest + R label recomputation in progress. "
                    "H2 test runs once labels exist.</em></p>")

    html.append("<h2>Reproducibility</h2>")
    html.append("<ul>")
    html.append('<li>Code: <a href="https://github.com/ishaschhikara316/cx-histology">github.com/ishaschhikara316/cx-histology</a></li>')
    html.append('<li>Models: <a href="https://huggingface.co/IshaChhikara/cx-uni-abmil-7cohort">cx-uni-abmil-7cohort</a> · <a href="https://huggingface.co/IshaChhikara/cx-uni-abmil-15cohort">cx-uni-abmil-15cohort</a> · <a href="https://huggingface.co/IshaChhikara/cx-uni-abmil-21cohort">cx-uni-abmil-21cohort</a></li>')
    html.append('<li>Random seed: 20260419 (NumPy + PyTorch + StratifiedKFold)</li>')
    html.append('<li>Hyperparameters: hidden=512, dropout=0.1, lr=1e-4, AdamW, Multi-Task Huber loss</li>')
    html.append("</ul>")

    html.append("</body></html>")

    out = FIG / "dashboard_21cohort.html"
    out.write_text("\n".join(html))
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
