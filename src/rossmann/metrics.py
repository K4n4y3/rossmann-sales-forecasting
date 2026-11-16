"""Forecast accuracy metrics.

Both metrics follow the Kaggle convention: days with zero actual sales
(closed stores) are ignored, because the percentage error is undefined there
and closed days are known in advance (forecast = 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _mask(y, yhat):
    y = np.asarray(y, dtype="float64")
    yhat = np.asarray(yhat, dtype="float64")
    m = y > 0
    return y[m], yhat[m]


def rmspe(y, yhat) -> float:
    """Root Mean Square Percentage Error (Kaggle competition metric)."""
    y, yhat = _mask(y, yhat)
    return float(np.sqrt(np.mean(((y - yhat) / y) ** 2)))


def smape(y, yhat) -> float:
    """Symmetric MAPE in percent, range 0-200."""
    y, yhat = _mask(y, yhat)
    denom = np.abs(y) + np.abs(yhat)
    return float(100 * np.mean(np.where(denom == 0, 0, 2 * np.abs(y - yhat) / denom)))


def mape(y, yhat) -> float:
    y, yhat = _mask(y, yhat)
    return float(100 * np.mean(np.abs(y - yhat) / y))


def mae(y, yhat) -> float:
    y, yhat = _mask(y, yhat)
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat) -> float:
    y, yhat = _mask(y, yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mase(y, yhat, scale) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler 2006) for many series.

    ``scale`` is, for every row, the in-sample MAE of the seasonal-naive (lag-7)
    forecast of that row's store on its training data.  MASE < 1 means the
    forecast beats an in-sample weekly seasonal-naive forecast.
    """
    y = np.asarray(y, dtype="float64")
    m = y > 0
    e = np.abs(y[m] - np.asarray(yhat, dtype="float64")[m])
    return float(np.mean(e / np.asarray(scale, dtype="float64")[m]))


def bias_pct(y, yhat) -> float:
    """Total forecast bias in percent of actual sales (+ = over-forecast)."""
    y, yhat = _mask(y, yhat)
    return float(100 * (yhat.sum() - y.sum()) / y.sum())


def score(y, yhat, scale=None) -> dict:
    out = {"RMSPE": rmspe(y, yhat), "sMAPE": smape(y, yhat), "MAPE": mape(y, yhat),
           "MAE": mae(y, yhat), "RMSE": rmse(y, yhat), "Bias_%": bias_pct(y, yhat)}
    if scale is not None:
        out["MASE"] = mase(y, yhat, scale)
    return out


def score_by(df: pd.DataFrame, pred_col: str, by: str | list[str], y_col: str = "Sales") -> pd.DataFrame:
    """RMSPE and sMAPE per group (e.g. per store or per hold-out week)."""
    rows = []
    for key, g in df.groupby(by, observed=True):
        rows.append({**(dict(zip(by, key)) if isinstance(by, list) else {by: key}),
                     "RMSPE": rmspe(g[y_col], g[pred_col]), "sMAPE": smape(g[y_col], g[pred_col]),
                     "n": int((g[y_col] > 0).sum())})
    return pd.DataFrame(rows)


def best_correction(y, yhat, grid=np.arange(0.95, 1.02, 0.0025)) -> tuple[float, float]:
    """Multiplicative factor that minimises RMSPE (RMSPE punishes over-forecasts
    more than under-forecasts, so the optimum is usually slightly below 1)."""
    scores = [(rmspe(y, yhat * f), f) for f in grid]
    best = min(scores)
    return float(best[1]), float(best[0])
