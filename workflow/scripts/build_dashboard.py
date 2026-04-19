"""Build figures/dashboard.html — a single-file interactive page that
surfaces benchmark tables, CPTAC generalisation, attention rollouts, and
prototype-tile atlases."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import click
import pandas as pd

TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>cx-histology dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<style>
body{font-family:-apple-system,Segoe UI,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;}
h1{border-bottom:1px solid #ccc;padding-bottom:.3em}
section{margin:2em 0} img{max-width:100%;border:1px solid #eee}
</style></head><body>
<h1>cx-histology &mdash; pan-cancer CX from H&amp;E</h1>
{sections}
</body></html>
"""


def _img_b64(path: Path) -> str:
    b = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{b}"


@click.command()
@click.option("--bench-json", required=True, type=click.Path(exists=True))
@click.option("--cptac-rho-tsv", required=True, type=click.Path(exists=True))
@click.option("--prototypes-dir", required=True, type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
def main(bench_json, cptac_rho_tsv, prototypes_dir, out):
    bench = json.load(open(bench_json))
    cptac = pd.read_csv(cptac_rho_tsv, sep="\t")
    sections = []

    sections.append("<section><h2>Per-signature Spearman &rho; (CV)</h2><div id=\"bench\"></div>"
                    f"<script>Plotly.newPlot('bench', [{{type:'bar',"
                    f"x:{json.dumps([f'CX{i}' for i in range(1,18)])},"
                    f"y:{json.dumps(bench['pan_cancer_spearman_mean'])}}}]);</script></section>")

    sections.append("<section><h2>CPTAC external validation</h2>"
                    f"<pre>{cptac.to_html(index=False)}</pre></section>")

    proto = "".join(
        f"<div><h4>{p.stem}</h4><img src='{_img_b64(p)}'></div>"
        for p in sorted(Path(prototypes_dir).glob("*.png")))
    sections.append(f"<section><h2>Prototype tiles</h2>{proto}</section>")

    Path(out).write_text(TEMPLATE.format(sections="\n".join(sections)))


if __name__ == "__main__":
    main()
