"""Shared helpers across the cx-histology pipeline.

Kept deliberately small. Anything cohort- or model-specific belongs in
its own script.
"""
from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# TCGA barcode: TCGA-XX-XXXX (participant level).
_TCGA_BARCODE = re.compile(r"(TCGA-[0-9A-Z]{2}-[0-9A-Z]{4})")

CX_COLUMNS = [f"CX{i}" for i in range(1, 18)]


def patient_id_from_tcga(slide_name: str) -> str:
    """Extract the TCGA participant barcode from a slide filename."""
    m = _TCGA_BARCODE.search(slide_name)
    if not m:
        raise ValueError(f"No TCGA barcode in {slide_name!r}")
    return m.group(1)


def load_cx_labels(path: str | Path, *, patient_col: str = "bcr_patient_barcode",
                   signatures: Iterable[str] = CX_COLUMNS) -> pd.DataFrame:
    """Load CX1–CX17 activities, indexed by patient barcode."""
    df = pd.read_csv(path, sep="\t")
    if patient_col not in df.columns:
        raise KeyError(f"{patient_col} not in {path}: {list(df.columns)[:5]}…")
    sig_cols = [c for c in signatures if c in df.columns]
    if not sig_cols:
        raise ValueError(f"No CX columns found in {path}")
    df = df.set_index(patient_col)[sig_cols]
    return df


def stratified_kfold_indices(df: pd.DataFrame, *, k: int,
                             stratify_col: str, seed: int):
    """Return k (train_idx, val_idx) tuples stratified by `stratify_col`."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    y = df[stratify_col].values
    folds = []
    idx = np.array(df.index)
    for tr, va in skf.split(idx, y):
        folds.append((idx[tr].tolist(), idx[va].tolist()))
    return folds


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
