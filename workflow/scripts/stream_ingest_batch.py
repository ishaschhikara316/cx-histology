"""Run stream_ingest_parallel sequentially across a list of TCGA cohorts.

One cohort at a time keeps the total connection count to GDC bounded
(n_workers × gdc_processes, default 16). Running multiple launchers
concurrently would risk hitting GDC's per-IP rate limit.

Per-cohort marker at `data/interim/embeddings/tcga/<encoder>/<cohort>/.done`
is written by the inner launcher on success, so already-completed cohorts
are skipped on rerun.

Usage:
    python stream_ingest_batch.py \
        --cohorts READ,MESO,UCS,KICH \
        --embed-encoder uni \
        --n-workers 4 --gdc-processes 4
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import click
import pandas as pd


@click.command()
@click.option("--cohorts", required=True,
              help="Comma-separated list of TCGA cohort codes (e.g. READ,MESO).")
@click.option("--embed-encoder", default="uni")
@click.option("--stain-reference", default="resources/stain_reference.png")
@click.option("--n-workers", default=4, type=int)
@click.option("--gdc-processes", default=4, type=int)
@click.option("--manifests-dir", default="data/interim/manifests")
@click.option("--raw-root", default="data/raw/tcga")
@click.option("--tiles-root", default="data/interim/tiles/tcga")
@click.option("--embed-root", default="data/interim/embeddings/tcga")
def main(cohorts, embed_encoder, stain_reference, n_workers, gdc_processes,
         manifests_dir, raw_root, tiles_root, embed_root):
    cs = [c.strip().upper() for c in cohorts.split(",") if c.strip()]
    rows = []
    for c in cs:
        manifest = Path(manifests_dir) / f"tcga_{c}.tsv"
        if not manifest.exists():
            print(f"[{c}] no manifest at {manifest} — skipping", file=sys.stderr)
            continue
        n = len(pd.read_csv(manifest, sep="\t"))
        gb = pd.read_csv(manifest, sep="\t").file_size.sum() / 1e9
        rows.append((c, manifest, n, gb))

    print(f"batch plan: {len(rows)} cohorts, "
          f"{sum(r[2] for r in rows)} slides, {sum(r[3] for r in rows):.0f} GB",
          file=sys.stderr)

    for c, manifest, n, gb in rows:
        done_marker = Path(embed_root) / embed_encoder / c / ".done"
        if done_marker.exists():
            print(f"[{c}] already done — skipping", file=sys.stderr)
            continue

        print(f"\n=== [{c}] {n} slides, {gb:.0f} GB ===", file=sys.stderr)
        t0 = time.time()
        cmd = [
            sys.executable, "workflow/scripts/stream_ingest_parallel.py",
            "--manifest", str(manifest),
            "--raw-dir", f"{raw_root}/{c}",
            "--tiles-dir", f"{tiles_root}/{c}",
            "--embed-dir", f"{embed_root}/{embed_encoder}/{c}",
            "--embed-encoder", embed_encoder,
            "--stain-reference", stain_reference,
            "--n-workers", str(n_workers),
            "--gdc-processes", str(gdc_processes),
            "--marker", str(done_marker),
        ]
        rc = subprocess.call(cmd)
        dt = time.time() - t0
        status = "ok" if rc == 0 else f"FAIL(rc={rc})"
        print(f"[{c}] {status} in {dt / 60:.1f} min", file=sys.stderr)
        if rc != 0:
            # Continue with the next cohort rather than aborting the whole batch.
            print(f"[{c}] partial results retained; continuing.", file=sys.stderr)


if __name__ == "__main__":
    main()
