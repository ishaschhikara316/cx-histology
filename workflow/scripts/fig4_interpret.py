"""Fig 4 — interpretability: attention overlay + enrichment forest + prototype strip."""
from pathlib import Path
import click, matplotlib.pyplot as plt, numpy as np, pandas as pd
from PIL import Image

@click.command()
@click.option("--enrichment", required=True, type=click.Path(exists=True),
              help="concatenated per-slide enrichment TSVs")
@click.option("--heatmap", required=True, type=click.Path(exists=True),
              help="a single representative heatmap NPY")
@click.option("--thumb", required=True, type=click.Path(exists=True),
              help="thumbnail PNG matching the heatmap")
@click.option("--prototypes", required=True, type=click.Path(exists=True),
              help="cx1_top.png (representative)")
@click.option("--out", required=True, type=click.Path())
def main(enrichment, heatmap, thumb, prototypes, out):
    enr = pd.read_csv(enrichment, sep="\t")
    heat = np.load(heatmap)
    thumb_img = np.asarray(Image.open(thumb).convert("RGB"))
    proto = np.asarray(Image.open(prototypes).convert("RGB"))

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(thumb_img); ax1.set_title("H&E")
    ax1.axis("off")
    ax2 = fig.add_subplot(gs[0, 1]); ax2.imshow(thumb_img)
    ax2.imshow(heat, cmap="hot", alpha=0.5); ax2.set_title("CX1 attention")
    ax2.axis("off")
    ax3 = fig.add_subplot(gs[0, 2]); ax3.imshow(proto); ax3.set_title("CX1 top-5 tiles")
    ax3.axis("off")
    ax4 = fig.add_subplot(gs[1, :])
    enr_plot = enr.groupby("feature")["stat"].mean().sort_values()
    enr_plot.plot(kind="barh", ax=ax4)
    ax4.set_xlabel("Wilcoxon statistic (high-attention vs low-attention tiles)")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)

if __name__ == "__main__":
    main()
