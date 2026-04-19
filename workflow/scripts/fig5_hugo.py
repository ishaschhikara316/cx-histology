"""Fig 5 — predicted CX1 vs anti-PD1 RECIST response in Hugo 2016."""
from pathlib import Path
import click, matplotlib.pyplot as plt, pandas as pd, seaborn as sns
from scipy.stats import mannwhitneyu

@click.command()
@click.option("--preds", required=True, type=click.Path(exists=True))
@click.option("--clinical", required=True, type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
def main(preds, clinical, out):
    p = pd.read_csv(preds, sep="\t")
    c = pd.read_csv(clinical, sep="\t")[["patient", "response"]]
    df = p.merge(c, on="patient")
    responders = df[df.response == "R"]["CX1"]
    non        = df[df.response == "NR"]["CX1"]
    u, pval = mannwhitneyu(responders, non, alternative="two-sided")
    fig, ax = plt.subplots(figsize=(4, 5))
    sns.boxplot(data=df, x="response", y="CX1", ax=ax)
    sns.stripplot(data=df, x="response", y="CX1", ax=ax, color="black", size=4)
    ax.set_title(f"Hugo 2016: predicted CX1 by response (p={pval:.3g})")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)

if __name__ == "__main__":
    main()
