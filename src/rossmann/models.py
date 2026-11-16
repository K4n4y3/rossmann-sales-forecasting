"""Forecasting models: baselines, per-store models and the global LightGBM.

All models predict daily sales for open store-days; closed days are forecast
as 0 (opening hours are known in advance, as in the Kaggle test file).
Model-based models are trained on ``log1p(Sales)`` of open days with
``Sales > 0``; minimising squared error in log space is close to minimising a
relative (percentage) error, which matches RMSPE / sMAPE.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge

from . import config, features as F, metrics

GLOBAL_PARAMS = dict(
    objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
    max_bin=255, cat_smooth=10, max_cat_to_onehot=4, verbose=-1, seed=config.SEED,
)
LOCAL_PARAMS = dict(
    objective="regression", learning_rate=0.05, num_leaves=15, min_data_in_leaf=15,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbose=-1, seed=config.SEED, num_threads=1,
)


@dataclass
class Split:
    """Train / inner-validation / hold-out windows for one backtest origin."""
    holdout_start: pd.Timestamp = config.HOLDOUT_START
    holdout_days: int = config.HOLDOUT_DAYS

    @property
    def holdout_end(self) -> pd.Timestamp:
        return self.holdout_start + pd.Timedelta(days=self.holdout_days - 1)

    @property
    def valid_start(self) -> pd.Timestamp:
        return self.holdout_start - pd.Timedelta(days=self.holdout_days)

    def masks(self, p: pd.DataFrame) -> dict[str, pd.Series]:
        d = p["Date"]
        known = p["is_test"] == 0
        return {
            "fit": known & (d < self.valid_start),                        # inner training
            "valid": known & (d >= self.valid_start) & (d < self.holdout_start),
            "train": known & (d < self.holdout_start),                    # full training
            "holdout": known & (d >= self.holdout_start) & (d <= self.holdout_end),
        }


@dataclass
class Result:
    name: str
    pred: pd.Series                         # aligned with the hold-out rows
    train_seconds: float
    n_models: int
    info: dict = field(default_factory=dict)


def trainable(p: pd.DataFrame, mask: pd.Series) -> pd.Series:
    return mask & (p["Open"] == 1) & (p["Sales"] > 0)


def _closed_to_zero(p_rows: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    return np.where(p_rows["Open"].to_numpy() == 1, pred, 0.0)


# -----------------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------------
def seasonal_naive(p: pd.DataFrame, split: Split) -> Result:
    """Last observed open-day sales of the same store and weekday (lag-7 logic)."""
    t0 = time.perf_counter()
    m = split.masks(p)
    hist = p.loc[trainable(p, m["train"]), ["Store", "DayOfWeek", "Date", "Sales"]]
    last = hist.sort_values("Date").groupby(["Store", "DayOfWeek"]).tail(1)
    ho = p.loc[m["holdout"], ["Store", "DayOfWeek", "Open"]]
    pred = ho.merge(last[["Store", "DayOfWeek", "Sales"]], on=["Store", "DayOfWeek"], how="left")["Sales"]
    # store never open on that weekday before (rare, e.g. Sundays): store median
    med = hist.groupby("Store")["Sales"].median()
    pred = pred.fillna(ho["Store"].map(med).reset_index(drop=True)).to_numpy()
    return Result("Seasonal naive (last same weekday)", pd.Series(_closed_to_zero(ho, pred), index=ho.index),
                  time.perf_counter() - t0, p["Store"].nunique())


def profile_baseline(p: pd.DataFrame, split: Split, months: int = 12) -> Result:
    """Geometric mean of the last `months` of sales per store x weekday x promo."""
    t0 = time.perf_counter()
    m = split.masks(p)
    start = split.holdout_start - pd.DateOffset(months=months)
    hist = p.loc[trainable(p, m["train"]) & (p["Date"] >= start), ["Store", "DayOfWeek", "Promo", "Sales"]].copy()
    hist["ls"] = np.log(hist["Sales"])
    prof = hist.groupby(["Store", "DayOfWeek", "Promo"])["ls"].mean().rename("p1").reset_index()
    prof2 = hist.groupby(["Store", "DayOfWeek"])["ls"].mean().rename("p2").reset_index()
    prof3 = hist.groupby("Store")["ls"].mean().rename("p3").reset_index()
    ho = p.loc[m["holdout"], ["Store", "DayOfWeek", "Promo", "Open"]]
    x = ho.merge(prof, on=["Store", "DayOfWeek", "Promo"], how="left") \
          .merge(prof2, on=["Store", "DayOfWeek"], how="left").merge(prof3, on="Store", how="left")
    pred = np.exp(x["p1"].fillna(x["p2"]).fillna(x["p3"])).to_numpy()
    return Result("Store x weekday x promo profile", pd.Series(_closed_to_zero(ho, pred), index=ho.index),
                  time.perf_counter() - t0, p["Store"].nunique())


# -----------------------------------------------------------------------------
# Global LightGBM (one model for all stores)
# -----------------------------------------------------------------------------
def _lgb_data(p, mask, feats):
    rows = trainable(p, mask)
    X = p.loc[rows, feats]
    y = np.log1p(p.loc[rows, "Sales"].to_numpy())
    return X, y


def fit_global(p_fit: pd.DataFrame, p_train: pd.DataFrame, split: Split,
               feats: list[str] | None = None, params: dict | None = None,
               max_rounds: int = 4000, early_stopping: int = 200,
               name: str = "Global LightGBM", log_every: int = 0,
               correct: bool = False) -> tuple[Result, lgb.Booster]:
    """Two-stage fit.

    1. Fit on the inner training window, early-stop on the inner validation
       window -> number of trees and RMSPE correction factor.
    2. Refit on the full training window (store statistics recomputed) with that
       number of trees and predict the hold-out.

    ``p_fit`` has store statistics computed before ``split.valid_start``;
    ``p_train`` before ``split.holdout_start``.
    """
    feats = feats or F.GLOBAL_FEATURES
    params = {**GLOBAL_PARAMS, **(params or {})}
    cats = [c for c in F.CATEGORICAL if c in feats]
    t0 = time.perf_counter()

    m = split.masks(p_fit)
    Xf, yf = _lgb_data(p_fit, m["fit"], feats)
    Xv, yv = _lgb_data(p_fit, m["valid"], feats)
    dtr = lgb.Dataset(Xf, yf, categorical_feature=cats, free_raw_data=False)
    dva = lgb.Dataset(Xv, yv, categorical_feature=cats, reference=dtr)
    cbs = [lgb.early_stopping(early_stopping, verbose=False)]
    if log_every:
        cbs.append(lgb.log_evaluation(log_every))
    stage1 = lgb.train(params, dtr, max_rounds, valid_sets=[dva], callbacks=cbs)
    best_iter = stage1.best_iteration or max_rounds
    pv = np.expm1(stage1.predict(Xv, num_iteration=best_iter))
    factor, valid_rmspe_corr = metrics.best_correction(np.expm1(yv), pv)
    valid_rmspe = metrics.rmspe(np.expm1(yv), pv)

    m2 = split.masks(p_train)
    Xt, yt = _lgb_data(p_train, m2["train"], feats)
    n_final = int(round(best_iter * 1.05))  # ~10% more data -> slightly more trees
    booster = lgb.train(params, lgb.Dataset(Xt, yt, categorical_feature=cats), n_final)
    ho = p_train.loc[m2["holdout"]]
    raw = np.expm1(booster.predict(ho[feats]))
    pred = _closed_to_zero(ho, raw * (factor if correct else 1.0))
    info = {"best_iter": best_iter, "n_trees": n_final, "correction": factor,
            "valid_RMSPE": valid_rmspe, "valid_RMSPE_corrected": valid_rmspe_corr,
            "pred_corrected": pd.Series(_closed_to_zero(ho, raw * factor), index=ho.index), "features": feats}
    return Result(name, pd.Series(pred, index=ho.index), time.perf_counter() - t0, 1, info), booster


# -----------------------------------------------------------------------------
# Per-store models (one model per store)
# -----------------------------------------------------------------------------
CHECKPOINTS = [25, 50, 75, 100, 150, 200, 300, 400, 600]


def _fit_one_store_lgb(Xf, yf, Xv, Xt, yt, Xh, n_rounds=None):
    """Stage 1 (n_rounds None): fit on inner window, return validation predictions at
    all checkpoints.  Stage 2: refit on the full window, predict the hold-out."""
    if n_rounds is None:
        b = lgb.train(LOCAL_PARAMS, lgb.Dataset(Xf, yf), max(CHECKPOINTS))
        return np.vstack([b.predict(Xv, num_iteration=k) for k in CHECKPOINTS])
    b = lgb.train(LOCAL_PARAMS, lgb.Dataset(Xt, yt), n_rounds)
    return b.predict(Xh)


def per_store_lgb(p_train: pd.DataFrame, split: Split, n_jobs: int = -1,
                  feats: list[str] | None = None, correct: bool = False) -> Result:
    feats = feats or F.LOCAL_FEATURES
    t0 = time.perf_counter()
    m = split.masks(p_train)
    fit_r, val_r = trainable(p_train, m["fit"]), trainable(p_train, m["valid"])
    tr_r, ho_r = trainable(p_train, m["train"]), m["holdout"]
    X = p_train[feats].to_numpy(dtype="float32")
    y = np.log1p(p_train["Sales"].fillna(0).to_numpy())
    groups = p_train.groupby("Store").indices
    stores = sorted(groups)

    def parts(s):
        i = groups[s]
        f, v, t, h = (i[fit_r.to_numpy()[i]], i[val_r.to_numpy()[i]],
                      i[tr_r.to_numpy()[i]], i[ho_r.to_numpy()[i]])
        return f, v, t, h

    idx = {s: parts(s) for s in stores}
    # stage 1: choose the number of trees on the inner validation window (pooled over stores)
    val_preds = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one_store_lgb)(X[f], y[f], X[v], None, None, None)
        for s in stores for f, v, t, h in [idx[s]] if len(f) > 30 and len(v) > 0)
    vs = [s for s in stores if len(idx[s][0]) > 30 and len(idx[s][1]) > 0]
    yv = np.concatenate([y[idx[s][1]] for s in vs])
    pv = np.hstack(val_preds)
    scores = [metrics.rmspe(np.expm1(yv), np.expm1(pv[k])) for k in range(len(CHECKPOINTS))]
    n_rounds = CHECKPOINTS[int(np.argmin(scores))]
    factor, _ = metrics.best_correction(np.expm1(yv), np.expm1(pv[int(np.argmin(scores))]))

    # stage 2: refit every store on the full training window
    out = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one_store_lgb)(None, None, None, X[t], y[t], X[h], n_rounds)
        for s in stores for f, v, t, h in [idx[s]])
    pred = pd.Series(np.nan, index=p_train.index)
    for s, ph in zip(stores, out):
        pred.iloc[idx[s][3]] = np.expm1(ph)
    ho = p_train.loc[ho_r]
    pred = pred.loc[ho.index].to_numpy()
    info = {"n_rounds": n_rounds, "correction": factor, "valid_RMSPE": float(min(scores)),
            "pred_corrected": pd.Series(_closed_to_zero(ho, pred * factor), index=ho.index),
            "checkpoint_scores": dict(zip(CHECKPOINTS, scores))}
    pred = pred * (factor if correct else 1.0)
    return Result("Per-store LightGBM", pd.Series(_closed_to_zero(ho, pred), index=ho.index),
                  time.perf_counter() - t0, len(stores), info)


# --- per-store ridge regression (a classical "one regression per store") ------
RIDGE_ONEHOT = {"DayOfWeek": range(1, 8), "Month": range(1, 13), "promo_day": range(0, 6),
                "StateHoliday_code": range(0, 4)}
RIDGE_BINNED = ["days_to_holiday", "days_since_holiday", "days_to_christmas", "days_since_christmas",
                "days_to_easter", "days_since_easter", "days_since_closed", "days_until_closed",
                "days_to_school_hol", "days_since_school_hol"]
RIDGE_NUMERIC = ["Promo", "PromoNextWeek", "PromoLastWeek", "SchoolHoliday", "is_promo2_month",
                 "promo2_active", "comp_active", "holidays_this_week", "holidays_next_week"]


def ridge_design(p: pd.DataFrame) -> np.ndarray:
    cols = []
    for c, levels in RIDGE_ONEHOT.items():
        v = p[c].to_numpy()
        cols += [(v == lv).astype("float32") for lv in list(levels)[1:]]
    for c in RIDGE_BINNED:  # distance 1..7 one-hot, anything further = reference
        v = p[c].to_numpy()
        cols += [(v == k).astype("float32") for k in range(0, 8)]
    cols += [p[c].to_numpy(dtype="float32") for c in RIDGE_NUMERIC]
    cols.append(((p["t"].to_numpy() - 500) / 365).astype("float32"))  # trend in years
    cols.append((p["Day"].to_numpy() <= 3).astype("float32"))           # start of month
    cols.append((p["Day"].to_numpy() >= 28).astype("float32"))          # end of month
    cols.append((p["prev_closed_len"].to_numpy() >= 3).astype("float32") *
                (p["days_since_closed"].to_numpy() <= 7))               # reopening after a long closure
    cols.append((p["next_closed_len"].to_numpy() >= 3).astype("float32") *
                (p["days_until_closed"].to_numpy() <= 7))               # before a long closure
    return np.column_stack(cols)


def per_store_ridge(p_train: pd.DataFrame, split: Split, alpha: float = 3.0) -> Result:
    t0 = time.perf_counter()
    m = split.masks(p_train)
    tr_r, ho_r = trainable(p_train, m["train"]).to_numpy(), m["holdout"].to_numpy()
    X = ridge_design(p_train)
    y = np.log1p(p_train["Sales"].fillna(0).to_numpy())
    pred = np.full(len(p_train), np.nan)
    for s, i in p_train.groupby("Store").indices.items():
        t, h = i[tr_r[i]], i[ho_r[i]]
        if len(h) == 0:
            continue
        mdl = Ridge(alpha=alpha).fit(X[t], y[t])
        pred[h] = np.expm1(mdl.predict(X[h]))
    ho = p_train.loc[ho_r]
    return Result("Per-store ridge regression", pd.Series(_closed_to_zero(ho, pred[ho_r]), index=ho.index),
                  time.perf_counter() - t0, p_train["Store"].nunique())


# --- per-store regression with ARMA errors (statsmodels SARIMAX) --------------
def sarimax_exog(p: pd.DataFrame) -> pd.DataFrame:
    """Compact exogenous design (~30 columns) for a regression with ARMA errors."""
    X = pd.DataFrame(index=p.index)
    for k in range(2, 8):
        X[f"dow{k}"] = (p["DayOfWeek"] == k).astype(float)
    for k in range(2, 13):
        X[f"m{k}"] = (p["Month"] == k).astype(float)
    X["promo"] = p["Promo"].astype(float)
    X["promo_next_week"] = p["PromoNextWeek"].astype(float)
    X["school"] = p["SchoolHoliday"].astype(float)
    X["open_holiday"] = (p["StateHoliday_code"] > 0).astype(float)
    X["pre_hol_1"] = (p["days_to_holiday"] == 1).astype(float)
    X["pre_hol_2"] = (p["days_to_holiday"] == 2).astype(float)
    X["post_hol_1"] = (p["days_since_holiday"] == 1).astype(float)
    X["xmas_week"] = p["days_to_christmas"].between(1, 7).astype(float)
    X["after_xmas"] = p["days_since_christmas"].between(2, 6).astype(float)
    X["reopen"] = ((p["prev_closed_len"] >= 3) & (p["days_since_closed"] <= 7)).astype(float)
    X["pre_close"] = ((p["next_closed_len"] >= 3) & (p["days_until_closed"] <= 7)).astype(float)
    X["month_start"] = (p["Day"] <= 3).astype(float)
    X["month_end"] = (p["Day"] >= 28).astype(float)
    return X


def _fit_one_sarimax(dates, y, X, holdout_start, order=(1, 0, 1)):
    import warnings
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    full = pd.date_range(dates.min(), dates.max(), freq="D")
    ys = pd.Series(y, index=dates).reindex(full)
    Xs = pd.DataFrame(X, index=dates).reindex(full).fillna(0)
    Xs = Xs.loc[:, Xs.loc[full < holdout_start].std() > 0]  # drop constant columns in training
    Xs["const"] = 1.0  # intercept as a regressor (more stable than trend="c" with AR errors)
    tr = full < holdout_start
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdl = SARIMAX(ys[tr], exog=Xs[tr], order=order, trend="n",
                      enforce_stationarity=True, enforce_invertibility=True)
        res = mdl.fit(disp=False, maxiter=200)
        fc = res.get_forecast(steps=int((~tr).sum()), exog=Xs[~tr]).predicted_mean
    fc.index = full[~tr]
    return fc.reindex(dates[dates >= holdout_start]).to_numpy()


def per_store_sarimax(p_train: pd.DataFrame, split: Split, stores: list[int], n_jobs: int = -1) -> Result:
    """Regression with ARMA(1,1) errors per store, closed days treated as missing
    observations (the Kalman filter skips them).  Run on a sample of stores."""
    t0 = time.perf_counter()
    m = split.masks(p_train)
    sub = p_train[p_train["Store"].isin(stores) & (m["train"] | m["holdout"])]
    X = sarimax_exog(sub)
    y = np.where((sub["Open"] == 1) & (sub["Sales"] > 0), np.log(sub["Sales"].clip(lower=1)), np.nan)
    groups = sub.groupby("Store").indices
    out = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one_sarimax)(pd.DatetimeIndex(sub["Date"].to_numpy()[i]), y[i], X.to_numpy()[i],
                                  split.holdout_start)
        for s, i in groups.items())
    pred = pd.Series(np.nan, index=sub.index)
    for (s, i), fc in zip(groups.items(), out):
        h = i[(sub["Date"].to_numpy()[i] >= split.holdout_start)]
        pred.iloc[h] = np.exp(fc)
    ho = sub.loc[m["holdout"].loc[sub.index]]
    pred = pred.loc[ho.index].to_numpy()
    return Result("Per-store SARIMAX (ARMA errors)", pd.Series(_closed_to_zero(ho, pred), index=ho.index),
                  time.perf_counter() - t0, len(groups))


def importance(booster: lgb.Booster) -> pd.DataFrame:
    imp = pd.DataFrame({"feature": booster.feature_name(),
                        "gain": booster.feature_importance("gain"),
                        "splits": booster.feature_importance("split")})
    imp["gain_share"] = imp["gain"] / imp["gain"].sum()
    group = {f: g for g, fs in F.FEATURE_GROUPS.items() for f in fs}
    imp["group"] = imp["feature"].map(group)
    return imp.sort_values("gain", ascending=False).reset_index(drop=True)
