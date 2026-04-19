"""Fig 2 — per-signature ρ and MAE for both encoders (post-sweep best HP)."""
import json
from pathlib import Path
import click, matplotlib.pyplot as plt, numpy as np, pandas as pd, seaborn as sns

@click.command()
@click.option("--summary-uni", required=True, type=click.Path(exists=True))
@click.option("--summary-hopt", required=True, type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
def main(summary_uni, summary_hopt, out):
    d = []
    for enc, p in [("UNI", summary_uni), ("H-optimus-0", summary_hopt)]:
        s = json.load(open(p))
        for k, rho in enumerate(s["pan_cancer_spearman_mean"]):
            d.append({"encoder": enc, "signature": f"CX{k+1}", "rho": rho})
    df = pd.DataFrame(d).pivot(index="signature", columns="encoder",
                               values="rho").reindex([f"CX{i}" for i in range(1, 18)])
    fig, ax = plt.subplots(figsize=(6, 7))
    sns.heatmap(df, annot=True, fmt=".2f", vmin=-0.1, vmax=0.6,
                cmap="RdBu_r", center=0, ax=ax)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)

if __name__ == "__main__":
    main()
