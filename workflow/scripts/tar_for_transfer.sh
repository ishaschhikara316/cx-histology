#!/bin/bash
# Wait for KICH .done, then tar everything experiment-related for
# off-server transfer. Excludes raw/tiles (stream-deleted), conda env,
# .snakemake cache, and Python bytecode.

set -e

ROOT="/root/Project/Isha/cx-histology"
cd "$ROOT"

echo "[$(date -Iseconds)] waiting for KICH .done"
while [ ! -f data/interim/embeddings/tcga/uni/KICH/.done ]; do
  sleep 60
done
echo "[$(date -Iseconds)] KICH done"

# Let any in-flight cleanup from stream_ingest finish
sleep 10

OUT=/root/cx_histology_checkpoint_$(date -u +%Y%m%d_%H%M).tar.gz

echo "[$(date -Iseconds)] tarring to $OUT"

# Content list:
#   - Whole repo (code + .git + configs + results + figures + docs + logs)
#   - data/labels/ (the CX labels TSV)
#   - data/interim/manifests/  (so we can resume without re-querying GDC)
#   - data/interim/embeddings/ (the expensive computed output — ~500 MB)
#   - /root/.claude/projects/-root-Project-Isha/memory/  (chat memory)
# Exclusions: raw/, tiles/ (empty per stream mode), __pycache__, .pytest_cache,
#             .snakemake, data/interim/manifests/_shards (transient)

tar czf "$OUT" \
  --exclude='cx-histology/data/raw' \
  --exclude='cx-histology/data/interim/tiles' \
  --exclude='cx-histology/data/interim/manifests/_shards' \
  --exclude='cx-histology/.snakemake' \
  --exclude='**/__pycache__' \
  --exclude='**/.pytest_cache' \
  --exclude='**/.ipynb_checkpoints' \
  -C /root/Project/Isha \
  cx-histology \
  -C /root/.claude/projects \
  -root-Project-Isha/memory

echo "[$(date -Iseconds)] tar done"
ls -lh "$OUT"
echo
echo "TAR READY AT: $OUT"
echo
echo "From your laptop, pull with:"
echo "  scp -P 59982 root@77.104.167.149:$OUT ~/"
