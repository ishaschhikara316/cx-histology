import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from workflow.scripts import _common


def test_tcga_patient_barcode_from_slide_id():
    # GDC slide filenames contain a full TCGA barcode.
    slide = "TCGA-02-0001-01Z-00-DX1.83fce43e-0d9b.svs"
    assert _common.patient_id_from_tcga(slide) == "TCGA-02-0001"


def test_load_cx_labels_returns_patient_indexed_frame(tmp_path):
    tsv = tmp_path / "cx.tsv"
    tsv.write_text(
        "patient\tCX1\tCX2\tCX3\n"
        "TCGA-02-0001\t0.1\t0.2\t0.3\n"
        "TCGA-02-0002\t0.4\t0.5\t0.6\n"
    )
    df = _common.load_cx_labels(tsv, patient_col="patient")
    assert df.index.name == "patient"
    assert list(df.columns) == ["CX1", "CX2", "CX3"]
    assert df.loc["TCGA-02-0001", "CX2"] == pytest.approx(0.2)


def test_stratified_kfold_preserves_cancer_distribution():
    rng = np.random.default_rng(0)
    patients = [f"P{i:04d}" for i in range(200)]
    cancer = rng.choice(["BRCA", "LUAD", "COAD", "SKCM"], size=200,
                        p=[0.4, 0.3, 0.2, 0.1])
    df = pd.DataFrame({"patient": patients, "cancer_type": cancer}).set_index("patient")
    folds = _common.stratified_kfold_indices(df, k=5, stratify_col="cancer_type",
                                             seed=42)
    assert len(folds) == 5
    assert sum(len(va) for _, va in folds) == len(df)
    for tr, va in folds:
        assert set(tr).isdisjoint(set(va))
        # Cancer-type distribution in val fold matches overall within 5 pp.
        overall = df["cancer_type"].value_counts(normalize=True)
        local = df.loc[va, "cancer_type"].value_counts(normalize=True)
        for c in overall.index:
            assert abs(overall[c] - local.get(c, 0)) < 0.05


def test_seed_everything_is_reproducible():
    import torch
    _common.seed_everything(123)
    a = torch.rand(4)
    _common.seed_everything(123)
    b = torch.rand(4)
    assert torch.equal(a, b)
