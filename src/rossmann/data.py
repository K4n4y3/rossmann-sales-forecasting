"""Loading and cleaning (same rules as notebook 01, so the pipeline can run without it).

``build_clean`` reproduces ``data/clean.parquet`` from the raw Kaggle files:
explicit dtypes, merge with ``store.csv``, minimal documented imputation and
data-quality flags.  Rows are never dropped here - modelling code decides.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

TRAIN_DTYPES = {
    "Store": "int16", "DayOfWeek": "int8", "Sales": "int32", "Customers": "int16",
    "Open": "int8", "Promo": "int8", "StateHoliday": "str", "SchoolHoliday": "int8",
}
TEST_DTYPES = {
    "Id": "int32", "Store": "int16", "DayOfWeek": "int8", "Open": "float32",
    "Promo": "int8", "StateHoliday": "str", "SchoolHoliday": "int8",
}


def _find(raw_dir: Path, name: str) -> Path:
    """Return raw_dir/name, unpacking a matching zip (e.g. train.csv.zip) if needed."""
    raw_dir = Path(raw_dir)
    p = raw_dir / name
    if p.exists():
        return p
    for z in raw_dir.glob(f"{Path(name).stem}*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(raw_dir)
        if p.exists():
            return p
    raise FileNotFoundError(f"{name} not found in {raw_dir} - see data/README.md")


def _clean_state_holiday(s: pd.Series) -> pd.Series:
    return s.astype(str).replace({"0.0": "0", "nan": "0"}).astype("category")


def load_train(raw_dir: Path = config.RAW_DIR, chunksize: int = 200_000) -> pd.DataFrame:
    parts = list(pd.read_csv(_find(raw_dir, "train.csv"), dtype=TRAIN_DTYPES,
                             parse_dates=["Date"], chunksize=chunksize))
    df = pd.concat(parts, ignore_index=True)
    df["StateHoliday"] = _clean_state_holiday(df["StateHoliday"])
    return df


def load_test(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(_find(raw_dir, "test.csv"), dtype=TEST_DTYPES, parse_dates=["Date"])
    df["StateHoliday"] = _clean_state_holiday(df["StateHoliday"])
    # 11 rows of store 622 have Open = NaN; they are ordinary weekdays -> assume open
    df["Open"] = df["Open"].fillna(1).astype("int8")
    return df


def load_store(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    store = pd.read_csv(_find(raw_dir, "store.csv"))
    for c in ["CompetitionOpenSinceMonth", "Promo2SinceWeek"]:
        store[c] = store[c].astype("Int8")
    for c in ["CompetitionOpenSinceYear", "Promo2SinceYear"]:
        store[c] = store[c].astype("Int16")
    store["Store"] = store["Store"].astype("int16")
    store["CompetitionDistance"] = store["CompetitionDistance"].astype("float32")
    store["Promo2"] = store["Promo2"].astype("int8")
    store["PromoInterval"] = store["PromoInterval"].fillna("None")
    for c in ["StoreType", "Assortment", "PromoInterval"]:
        store[c] = store[c].astype("category")
    return store


def build_clean(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """Train + store merged, with the data-quality flags of notebook 01."""
    train, store = load_train(raw_dir), load_store(raw_dir)
    df = train.merge(store, on="Store", how="left", validate="m:1")

    df["CompetitionDistance_missing"] = df["CompetitionDistance"].isna()
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
        store["CompetitionDistance"].median()).astype("float32")

    open_pos = (df["Open"] == 1) & (df["Sales"] > 0)
    df["open_zero_sales"] = (df["Open"] == 1) & (df["Sales"] == 0)

    # extreme outliers: per-store 3 x IQR on open days (flagged, not removed)
    q = df.loc[open_pos].groupby("Store")["Sales"].quantile([0.25, 0.75]).unstack()
    iqr = q[0.75] - q[0.25]
    hi, lo = q[0.75] + 3 * iqr, q[0.25] - 3 * iqr
    s = df.loc[open_pos, "Sales"]
    st = df.loc[open_pos, "Store"]
    df["sales_outlier"] = False
    df.loc[open_pos, "sales_outlier"] = (s > st.map(hi)) | (s < st.map(lo))

    n_days = (df["Date"].max() - df["Date"].min()).days + 1
    counts = df.groupby("Store")["Date"].count()
    df["has_gap"] = df["Store"].isin(counts.index[counts < n_days])
    df["IsStateHoliday"] = (df["StateHoliday"] != "0").astype("int8")
    return df.sort_values(["Store", "Date"]).reset_index(drop=True)


def get_clean(raw_dir: Path = config.RAW_DIR, cache: Path | None = None,
              rebuild: bool = False) -> pd.DataFrame:
    """Load ``data/clean.parquet`` (Dev A hand-off) or rebuild it from the raw CSVs."""
    cache = Path(cache) if cache else config.DATA_DIR / "clean.parquet"
    if cache.exists() and not rebuild:
        df = pd.read_parquet(cache)
        df["StateHoliday"] = _clean_state_holiday(df["StateHoliday"])
        return df
    df = build_clean(raw_dir)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def summary(df: pd.DataFrame) -> dict:
    """Headline data facts used in the report."""
    open_pos = (df["Open"] == 1) & (df["Sales"] > 0)
    return {
        "rows": len(df),
        "stores": int(df["Store"].nunique()),
        "date_min": str(df["Date"].min().date()),
        "date_max": str(df["Date"].max().date()),
        "closed_share": float((df["Open"] == 0).mean()),
        "open_zero_sales": int(df["open_zero_sales"].sum()),
        "extreme_outliers": int(df["sales_outlier"].sum()),
        "stores_with_gap": int(df.loc[df["has_gap"], "Store"].nunique()),
        "mean_sales_open": float(df.loc[open_pos, "Sales"].mean()),
        "median_sales_open": float(df.loc[open_pos, "Sales"].median()),
        "total_sales": float(df["Sales"].sum()),
    }


def np_seed(seed: int = config.SEED) -> np.random.Generator:
    return np.random.default_rng(seed)
