"""Project-wide paths and constants.

Paths can be overridden with environment variables so the code runs the same way
locally, in Colab and in CI:

* ``ROSSMANN_ROOT`` - repository root (default: two levels above this file)
* ``ROSSMANN_RAW``  - folder that holds ``train.csv``, ``store.csv``, ``test.csv``
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("ROSSMANN_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
RAW_DIR = Path(os.environ.get("ROSSMANN_RAW", DATA_DIR / "raw"))
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
MET_DIR = OUT_DIR / "metrics"
PRED_DIR = OUT_DIR / "predictions"

# --- evaluation design -------------------------------------------------------
# The last 6 weeks of the training data are the hold-out (same window as the
# Kaggle test horizon: 48 days, here 42 = 6 full weeks).  The 6 weeks before the
# hold-out are an inner validation window used only for early stopping and for
# choosing the RMSPE correction factor - the hold-out is never used for tuning.
HOLDOUT_DAYS = 42
LAST_TRAIN_DATE = pd.Timestamp("2015-07-31")
HOLDOUT_START = LAST_TRAIN_DATE - pd.Timedelta(days=HOLDOUT_DAYS - 1)  # 2015-06-20
VALID_START = HOLDOUT_START - pd.Timedelta(days=HOLDOUT_DAYS)  # 2015-05-09

SEED = 42


def ensure_dirs() -> None:
    for p in (DATA_DIR, OUT_DIR, FIG_DIR, MET_DIR, PRED_DIR):
        p.mkdir(parents=True, exist_ok=True)
