"""Fig 1 — pipeline schematic and CX signature coverage across TCGA."""
from pathlib import Path
import click, matplotlib.pyplot as plt, pandas as pd

@click.command()
@click.option("--labels", required=True, type=click.Path(exists=True))
@click.option("--manifest", required=True, type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
def main(labels, manifest, out):
    lab = pd.read_csv(labels, sep="\t")
    man = pd.read_csv(manifest, sep="\t")
    joined = man.merge(lab, left_on="patient",
                       right_on="bcr_patient_barcode", how="inner")
    counts = joined.groupby("cancer_type").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set_ylabel("Patients with slides + CX labels")
    ax.set_title("TCGA coverage across cancer types")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)

if __name__ == "__main__":
    main()
