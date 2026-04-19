# resources/

## stain_reference.png (TODO — not yet present)

Macenko stain-normalisation reference tile. Must be a clean 256×256 H&E
tile from a TCGA slide (any cancer type, any scanner — the goal is a
consistent target, not a representative one).

Pick it before the Phase 2 embedding run:

```bash
python - <<'PY'
import h5py
from PIL import Image
# Replace with path to any extracted tile HDF5:
with h5py.File("data/interim/tiles/tcga/LUAD/<some-slide>.h5") as h:
    Image.fromarray(h["tiles"][0]).save("resources/stain_reference.png")
PY
```

The test suite does not depend on this file (`test_tile_wsi.py` passes
`stain_reference=None`). Production tile extraction uses it.
