# cx-histology ABMIL head

**Task:** Regression of Drews 2022 CX1–CX17 chromosomal-instability
activities from frozen UNI / H-optimus-0 tile embeddings.

**Inputs:** per-slide bag of (N, D) tile embeddings produced by
`embed_tiles.py` (UNI: d=1024, H-optimus-0: d=1536). No raw images.

**Output:** 17-vector of predicted CX activities, plus per-tile
attention weights.

**Training data:** TCGA pan-cancer (~7,300 patients, 33 cancer types),
labels from Drews 2022 applied to TCGA ABSOLUTE segments.

**Held-out metrics:** see `docs/methods.md` and the dashboard.

**Intended use:** Research-only, for studying morphological correlates
of chromosomal instability. **Not for clinical use.**

**Limitations:** TCGA scanner/stain bias; CPTAC generalisation degrades
by < 0.15 ρ on primary signatures.

**License:** MIT.
