"""Shard a TCGA manifest across N GPU workers and run stream_ingest in parallel.

Each worker:
  * Gets a round-robin shard of the manifest (shard_k = rows where i % N == k).
  * Is pinned to GPU k via CUDA_VISIBLE_DEVICES.
  * Uses its own raw-dir subfolder to avoid collisions during gdc-client download.
  * Shares the tiles-dir and embed-dir with peers (per-slide stems are unique).

Shared stain_reference has a benign race: if worker 0 hasn't bootstrapped
the reference by the time worker k starts its first slide, worker k tiles
without stain normalisation. At most (N-1) slides per cancer are un-normalised
in the worst case. Acceptable for production runs.

Usage:
    python stream_ingest_parallel.py \
        --manifest data/interim/manifests/tcga_DLBC.tsv \
        --raw-dir data/raw/tcga/DLBC \
        --tiles-dir data/interim/tiles/tcga/DLBC \
        --embed-dir data/interim/embeddings/tcga/uni/DLBC \
        --embed-encoder uni \
        --stain-reference resources/stain_reference.png \
        --n-workers 4 --marker data/interim/embeddings/tcga/uni/DLBC/.done
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import click
import pandas as pd


@click.command()
@click.option("--manifest", required=True, type=click.Path(exists=True))
@click.option("--raw-dir", required=True, type=click.Path())
@click.option("--tiles-dir", required=True, type=click.Path())
@click.option("--embed-dir", required=True, type=click.Path())
@click.option("--embed-encoder", required=True)
@click.option("--stain-reference", required=True, type=click.Path())
@click.option("--n-workers", default=4, type=int)
@click.option("--models-config", default="config/models.yaml")
@click.option("--patch-px", default=256, type=int)
@click.option("--mpp-target", default=0.5, type=float)
@click.option("--tissue-fraction-min", default=0.5, type=float)
@click.option("--tiles-per-slide", default=1000, type=int)
@click.option("--laplacian-var-min", default=100.0, type=float)
@click.option("--embed-batch-size", default=32, type=int)
@click.option("--seed", default=20260419, type=int)
@click.option("--marker", default=None, type=click.Path())
def main(manifest, raw_dir, tiles_dir, embed_dir, embed_encoder,
         stain_reference, n_workers, models_config,
         patch_px, mpp_target, tissue_fraction_min, tiles_per_slide,
         laplacian_var_min, embed_batch_size, seed, marker):
    df = pd.read_csv(manifest, sep="\t")

    shards_dir = Path(manifest).parent / "_shards" / Path(manifest).stem
    shards_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = []
    for k in range(n_workers):
        shard = df.iloc[k::n_workers]
        sp = shards_dir / f"shard_{k}.tsv"
        shard.to_csv(sp, sep="\t", index=False)
        shard_paths.append(sp)

    raw = Path(raw_dir); raw.mkdir(parents=True, exist_ok=True)
    log_dir = Path("logs"); log_dir.mkdir(exist_ok=True)

    procs = []
    for k, sp in enumerate(shard_paths):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(k)
        env["PYTHONPATH"] = str(Path(".").resolve()) + ":" + env.get("PYTHONPATH", "")
        worker_raw = raw / f"worker{k}"
        cmd = [
            sys.executable, "workflow/scripts/stream_ingest.py",
            "--manifest", str(sp),
            "--raw-dir", str(worker_raw),
            "--tiles-dir", tiles_dir,
            "--embed-dir", embed_dir,
            "--embed-encoder", embed_encoder,
            "--embed-device", "cuda:0",  # each worker sees exactly one GPU
            "--stain-reference", stain_reference,
            "--models-config", models_config,
            "--embed-batch-size", str(embed_batch_size),
            "--patch-px", str(patch_px),
            "--mpp-target", str(mpp_target),
            "--tissue-fraction-min", str(tissue_fraction_min),
            "--tiles-per-slide", str(tiles_per_slide),
            "--laplacian-var-min", str(laplacian_var_min),
            "--seed", str(seed),
        ]
        log_path = log_dir / f"stream_worker_{Path(manifest).stem}_{k}.log"
        print(f"[worker {k}] GPU {k}, {len(pd.read_csv(sp, sep=chr(9)))} slides, "
              f"log={log_path}", file=sys.stderr, flush=True)
        fh = open(log_path, "w")
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        procs.append((k, p, fh))

    # Wait for all workers. If any fails, keep waiting for the rest but record.
    failures = []
    for k, p, fh in procs:
        rc = p.wait()
        fh.close()
        status = "ok" if rc == 0 else f"FAIL(rc={rc})"
        print(f"[worker {k}] {status}", file=sys.stderr, flush=True)
        if rc != 0:
            failures.append(k)

    if failures:
        print(f"failed workers: {failures}", file=sys.stderr)
        sys.exit(1)

    if marker:
        Path(marker).parent.mkdir(parents=True, exist_ok=True)
        Path(marker).touch()


if __name__ == "__main__":
    main()
