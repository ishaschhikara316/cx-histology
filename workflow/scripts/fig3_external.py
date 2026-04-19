"""Fig 3 — external generalisation: TCGA (CV) ρ vs CPTAC (locked) ρ per signature."""
import json
from pathlib import Path
import click, matplotlib.pyplot as plt, numpy as np, pandas as pd
from scipy.stats import spearmanr

@click.command()
@click.option("--tcga-summary", required=True, type=click.Path(exists=True))
@click.option("--cptac-preds", required=True, type=click.Path(exists=True))
@click.option("--cptac-labels", required=True, type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
def main(tcga_summary, cptac_preds, cptac_labels, out):
    tcga_rho = np.asarray(json.load(open(tcga_summary))["pan_cancer_spearman_mean"])
    preds = pd.read_csv(cptac_preds, sep="\t")
    lab = pd.read_csv(cptac_labels, sep="\t").rename(
        columns={"bcr_patient_barcode": "patient"})
    df = preds.merge(lab, on="patient", suffixes=("_pred", "_true"))
    cptac_rho = np.array([
        spearmanr(df[f"CX{i}_pred"], df[f"CX{i}_true"]).correlation
        for i in range(1, 18)])
    x = np.arange(17)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - 0.2, tcga_rho, width=0.4, label="TCGA (CV)")
    ax.bar(x + 0.2, cptac_rho, width=0.4, label="CPTAC (external)")
    ax.set_xticks(x); ax.set_xticklabels([f"CX{i}" for i in range(1, 18)], rotation=45)
    ax.axhline(0.3, ls="--", color="grey", alpha=0.5, label="H1 target")
    ax.set_ylabel("Spearman ρ"); ax.legend(); fig.tight_layout()
    fig.savefig(out); plt.close(fig)

if __name__ == "__main__":
    main()
