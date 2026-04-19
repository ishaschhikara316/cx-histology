"""Tests for stream_ingest.py using a mocked gdc-client.

The mock places a pre-made synthetic .svs file at the expected path
whenever `gdc-client download` is invoked. This lets us exercise the
download → tile → cleanup loop without real network.
"""
from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips
import pytest
from click.testing import CliRunner

from workflow.scripts import stream_ingest as si


def _make_synth_slide(path: Path) -> None:
    img = np.full((1024, 1024, 3), 240, dtype=np.uint8)
    yy, xx = np.ogrid[:1024, :1024]
    m = (xx - 512) ** 2 + (yy - 512) ** 2 < 300 ** 2
    img[m] = [180, 90, 150]
    vi = pyvips.Image.new_from_memory(img.tobytes(), 1024, 1024, 3, "uchar")
    vi.tiffsave(str(path), tile=True, pyramid=True, compression="jpeg",
                tile_width=256, tile_height=256, bigtiff=True)


def _write_fake_gdc_client(bin_dir: Path, slides: dict[str, Path]) -> None:
    """Write a fake `gdc-client` shim that copies pre-made slides into place
    to simulate a download.

    `slides` maps file_id → path to a pre-made .svs file.
    """
    mapping = "\n".join(f'"{fid}") cp "{src}" "$outdir/{fid}/{src.name}" ;;'
                        for fid, src in slides.items())
    script = f"""#!/bin/bash
# Fake gdc-client that copies pre-staged slides.
# Usage: gdc-client download <file_id> -d <outdir> -n <n>
if [ "$1" != "download" ]; then exit 0; fi
shift
fid=$1; shift
outdir="."
while [ $# -gt 0 ]; do
  case "$1" in
    -d) outdir="$2"; shift 2 ;;
    -n) shift 2 ;;
    -t) shift 2 ;;
    *)  shift ;;
  esac
done
mkdir -p "$outdir/$fid"
case "$fid" in
{mapping}
  *) echo "unknown $fid" >&2; exit 1 ;;
esac
"""
    bin_dir.mkdir(parents=True, exist_ok=True)
    p = bin_dir / "gdc-client"
    p.write_text(script)
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def staged(tmp_path, monkeypatch):
    # Pre-make two slides the fake client will "download".
    stage = tmp_path / "stage"; stage.mkdir()
    s1, s2 = stage / "slide1.svs", stage / "slide2.svs"
    _make_synth_slide(s1); _make_synth_slide(s2)

    bin_dir = tmp_path / "bin"
    _write_fake_gdc_client(bin_dir, {"fid1": s1, "fid2": s2})
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")

    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame([
        {"file_id": "fid1", "file_name": "slide1.svs", "file_size": 1_000_000,
         "md5": "x", "project": "TCGA-TST", "patient": "TCGA-01-0001"},
        {"file_id": "fid2", "file_name": "slide2.svs", "file_size": 1_000_000,
         "md5": "y", "project": "TCGA-TST", "patient": "TCGA-01-0002"},
    ]).to_csv(manifest, sep="\t", index=False)
    return tmp_path, manifest


def test_stream_ingest_produces_h5_and_cleans_raw(staged):
    tmp_path, manifest = staged
    raw = tmp_path / "raw"
    tiles = tmp_path / "tiles"
    ref = tmp_path / "stain_reference.png"

    result = CliRunner().invoke(si.main, [
        "--manifest", str(manifest),
        "--raw-dir", str(raw),
        "--tiles-dir", str(tiles),
        "--stain-reference", str(ref),
        "--patch-px", "256", "--mpp-target", "0.5",
        "--tissue-fraction-min", "0.3", "--tiles-per-slide", "4",
        "--laplacian-var-min", "0.0", "--seed", "0",
    ])
    assert result.exit_code == 0, result.output + str(result.exception)
    # Both .h5 outputs exist, raw/ is empty (cleaned up), ref bootstrapped.
    assert (tiles / "slide1.h5").exists()
    assert (tiles / "slide2.h5").exists()
    assert ref.exists()
    assert not any(raw.glob("fid*"))


def test_stream_ingest_is_idempotent(staged):
    tmp_path, manifest = staged
    raw = tmp_path / "raw"
    tiles = tmp_path / "tiles"
    ref = tmp_path / "stain_reference.png"
    args = [
        "--manifest", str(manifest),
        "--raw-dir", str(raw),
        "--tiles-dir", str(tiles),
        "--stain-reference", str(ref),
        "--patch-px", "256", "--mpp-target", "0.5",
        "--tissue-fraction-min", "0.3", "--tiles-per-slide", "4",
        "--laplacian-var-min", "0.0", "--seed", "0",
    ]
    CliRunner().invoke(si.main, args)
    mtime_first = (tiles / "slide1.h5").stat().st_mtime
    # Second run should skip existing outputs.
    result = CliRunner().invoke(si.main, args)
    assert result.exit_code == 0
    assert (tiles / "slide1.h5").stat().st_mtime == mtime_first
