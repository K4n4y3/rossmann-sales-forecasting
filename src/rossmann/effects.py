"""Promotion and holiday effects.

Three complementary estimates:

1. **Descriptive** - raw means (confounded by season and weekday; for intuition only).
2. **Fixed-effects regression** (statsmodels OLS) on log sales of open days:
   store fixed effects (within transformation) + weekday, month, year and
   day-of-month dummies, two-way clustered standard errors (store and date).
   ``exp(beta) - 1`` is the percentage effect.  Promo is set chain-wide on a
   fixed bi-weekly calendar (the same for every store on a date), so the
   comparison of promo and non-promo weeks with calendar controls is close to a
   natural experiment.
3. **Model-based counterfactual** - the global LightGBM forecast for the hold-out
   with and without the promo calendar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

XMAS_DAYS = {f"xmas_D-{k}": k for k in range(1, 8)}          # Dec 18..24
POST_XMAS = {f"xmas_D+{k}": k for k in range(2, 7)}          # Dec 27..31


EFFECT_COLS = ["Store", "Date", "Sales", "Customers", "Open", "Promo", "SchoolHoliday", "StateHoliday",
               "StateHoliday_code", "is_test", "is_promo2_month", "days_to_holiday", "days_since_holiday",
               "days_to_christmas", "days_since_christmas", "prev_closed_len", "days_since_closed",
               "next_closed_len", "days_until_closed", "DayOfWeek", "Month", "Year", "Day", "StoreType", "Assortment"]


def _open_days(p: pd.DataFrame) -> pd.DataFrame:
    m = (p["is_test"] == 0) & (p["Open"] == 1) & (p["Sales"] > 0)
    return p.loc[m, [c for c in EFFECT_COLS if c in p]].copy()


# -----------------------------------------------------------------------------
# 1. descriptive
# -----------------------------------------------------------------------------
def descriptive(p: pd.DataFrame) -> dict[str, pd.DataFrame]:
    o = _open_days(p)
    o["SalesPerCustomer"] = o["Sales"] / o["Customers"].replace(0, np.nan)
    out = {}
    out["promo_overall"] = o.groupby("Promo")[["Sales", "Customers", "SalesPerCustomer"]].mean().T \
        .rename(columns={0: "no_promo", 1: "promo"}).assign(uplift_pct=lambda d: 100 * (d.promo / d.no_promo - 1))
    wd = o[o["DayOfWeek"] <= 5].groupby(["DayOfWeek", "Promo"])["Sales"].mean().unstack()
    out["promo_by_weekday"] = wd.rename(columns={0: "no_promo", 1: "promo"}) \
        .assign(uplift_pct=lambda d: 100 * (d.promo / d.no_promo - 1))
    st = o[o["DayOfWeek"] <= 5].groupby(["StoreType", "Promo"], observed=True)["Sales"].mean().unstack()
    out["promo_by_storetype"] = st.rename(columns={0: "no_promo", 1: "promo"}) \
        .assign(uplift_pct=lambda d: 100 * (d.promo / d.no_promo - 1))
    k = p[p["is_test"] == 0]
    hol = k.groupby("StateHoliday", observed=True).agg(
        store_days=("Open", "size"), share_open=("Open", "mean"))
    hol["mean_sales_if_open"] = o.groupby("StateHoliday", observed=True)["Sales"].mean()
    hol.index = hol.index.map({"0": "none", "a": "public", "b": "Easter", "c": "Christmas"})
    out["state_holiday"] = hol
    out["school_holiday"] = o.groupby("SchoolHoliday")["Sales"].mean().to_frame("mean_sales") \
        .assign(ratio=lambda d: d.mean_sales / d.mean_sales.iloc[0])
    return out


# -----------------------------------------------------------------------------
# 2. fixed-effects regression
# -----------------------------------------------------------------------------
def effect_design(o: pd.DataFrame, interactions: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Regressors of interest + calendar controls. Returns (X, names of interest)."""
    xmas_window = o["days_to_christmas"].between(0, 7) | o["days_since_christmas"].between(0, 8)
    X = pd.DataFrame(index=o.index)
    X["Promo"] = o["Promo"]
    X["SchoolHoliday"] = o["SchoolHoliday"]
    X["Promo2 month"] = o["is_promo2_month"]
    X["Open on state holiday"] = (o["StateHoliday_code"] > 0).astype(int)
    for k in (1, 2, 3):
        X[f"{k} day(s) before holiday"] = ((o["days_to_holiday"] == k) & ~xmas_window).astype(int)
        X[f"{k} day(s) after holiday"] = ((o["days_since_holiday"] == k) & ~xmas_window).astype(int)
    for name, k in XMAS_DAYS.items():
        X[name] = (o["days_to_christmas"] == k).astype(int)
    for name, k in POST_XMAS.items():
        X[name] = (o["days_since_christmas"] == k).astype(int)
    X["Reopening (first week after closure >= 5 days)"] = (
        (o["prev_closed_len"] >= 5) & (o["days_since_closed"] <= 7) & ~xmas_window).astype(int)
    X["Last week before closure >= 5 days"] = (
        (o["next_closed_len"] >= 5) & (o["days_until_closed"] <= 7) & ~xmas_window).astype(int)
    interest = list(X.columns)

    if interactions == "weekday":
        for d in range(2, 6):
            X[f"Promo x weekday {d}"] = o["Promo"] * (o["DayOfWeek"] == d)
        interest = ["Promo"] + [c for c in X if c.startswith("Promo x")]
    elif interactions == "storetype":
        for t in ["b", "c", "d"]:
            X[f"Promo x type {t}"] = o["Promo"] * (o["StoreType"].astype(str) == t)
        interest = ["Promo"] + [c for c in X if c.startswith("Promo x")]
    elif interactions == "assortment":
        for t in ["b", "c"]:
            X[f"Promo x assortment {t}"] = o["Promo"] * (o["Assortment"].astype(str) == t)
        interest = ["Promo"] + [c for c in X if c.startswith("Promo x")]

    # calendar controls (reference: Monday, January, 2013, day 1)
    for col, levels in {"DayOfWeek": range(2, 8), "Month": range(2, 13), "Year": (2014, 2015),
                        "Day": range(2, 32)}.items():
        for lv in levels:
            X[f"{col}={lv}"] = (o[col] == lv).astype(int)
    return X.astype("float64"), interest


def fe_ols(p: pd.DataFrame, target: str = "Sales", interactions: str | None = None,
           mask: pd.Series | None = None):
    """Within (store fixed effects) OLS of log(target) with two-way clustered SEs."""
    o = _open_days(p)
    o = o[o["Customers"] > 0]
    if mask is not None:
        o = o[mask.loc[o.index]]
    if target == "Sales":
        y = np.log(o["Sales"])
    elif target == "Customers":
        y = np.log(o["Customers"])
    elif target == "SalesPerCustomer":
        y = np.log(o["Sales"] / o["Customers"])
    else:
        raise ValueError(target)
    X, interest = effect_design(o, interactions)
    X = X.loc[:, X.std() > 0]
    names = list(X.columns)
    g = o["Store"].to_numpy()
    groups = np.column_stack([pd.factorize(o["Store"])[0], pd.factorize(o["Date"])[0]])
    yd = (y - y.groupby(g).transform("mean")).to_numpy()
    Xd = (X - X.groupby(g).transform("mean")).to_numpy()   # within transformation (store fixed effects)
    del X, o
    res = sm.OLS(yd, Xd).fit(cov_type="cluster", cov_kwds={"groups": groups})
    tab = pd.DataFrame({"coef": res.params, "se": res.bse, "p": res.pvalues}, index=names)
    ci = res.conf_int()
    tab["ci_low"], tab["ci_high"] = ci[:, 0], ci[:, 1]
    for c in ["coef", "ci_low", "ci_high"]:
        tab[c.replace("coef", "effect") + "_pct"] = 100 * (np.exp(tab[c]) - 1)
    tab = tab.rename(columns={"ci_low_pct": "ci_low_pct", "ci_high_pct": "ci_high_pct"})
    tab.attrs.update({"n_obs": int(res.nobs), "r2_within": float(res.rsquared),
                      "target": target, "interactions": interactions})
    return tab.loc[[c for c in interest if c in tab.index]], tab, res


def interaction_effects(tab_full: pd.DataFrame, base: str, prefix: str, ref_label: str) -> pd.DataFrame:
    """Total promo effect per group = base + interaction (point estimate)."""
    rows = [{"group": ref_label, "effect_pct": 100 * (np.exp(tab_full.loc[base, "coef"]) - 1)}]
    for c in tab_full.index:
        if c.startswith(prefix):
            b = tab_full.loc[base, "coef"] + tab_full.loc[c, "coef"]
            rows.append({"group": c.replace(prefix, "").strip(), "effect_pct": 100 * (np.exp(b) - 1),
                         "diff_p": tab_full.loc[c, "p"]})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 3. model-based counterfactuals (global LightGBM)
# -----------------------------------------------------------------------------
def promo_counterfactual(booster, ho: pd.DataFrame, feats: list[str]) -> dict:
    """Promo-day uplift implied by the global LightGBM on the hold-out.

    Main scenario (*week flip*): every promo day is turned into a day of a regular
    non-promo week - Promo = 0, promo_day = 0 and, as in 94% of real non-promo
    weekdays, promo in the week before and after.  A tree model cannot extrapolate,
    so the counterfactual must stay inside combinations seen in training.

    Sensitivity (*flag only*): Promo and promo_day set to 0, neighbouring weeks left
    as they are.  On a promo day this gives "no promo last, this and next week", a
    combination that never occurs in the data, so it is reported only to show why
    the in-distribution scenario is needed.
    """
    is_open = ho["Open"].to_numpy() == 1
    base = np.expm1(booster.predict(ho[feats])) * is_open

    flip = ho[feats].copy()
    flip["Promo"], flip["promo_day"] = 0, 0
    flag = flip.copy()
    no_promo_flag = np.expm1(booster.predict(flag)) * is_open
    flip["PromoLastWeek"], flip["PromoNextWeek"] = 1, 1
    no_promo = np.expm1(booster.predict(flip)) * is_open

    pm = (ho["Promo"].to_numpy() == 1) & is_open
    by = pd.DataFrame({"StoreType": ho["StoreType"].astype(str).to_numpy(),
                       "DayOfWeek": ho["DayOfWeek"].to_numpy(),
                       "with": base, "without": no_promo})[pm]
    by_type = by.groupby("StoreType")[["with", "without"]].sum().assign(
        uplift_pct=lambda d: 100 * (d["with"] / d["without"] - 1))
    by_dow = by.groupby("DayOfWeek")[["with", "without"]].sum().assign(
        uplift_pct=lambda d: 100 * (d["with"] / d["without"] - 1))
    inc = base[pm].sum() - no_promo[pm].sum()
    return {
        "promo_day_uplift_pct": float(100 * (base[pm].sum() / no_promo[pm].sum() - 1)),
        "promo_day_uplift_flag_only_pct": float(100 * (base[pm].sum() / no_promo_flag[pm].sum() - 1)),
        "promo_days_incremental_eur": float(inc),
        "promo_days_incremental_share_pct": float(100 * inc / base.sum()),
        "holdout_forecast_eur": float(base.sum()),
        "n_promo_store_days": int(pm.sum()),
        "by_storetype": by_type, "by_weekday": by_dow,
    }


def shap_groups(booster, X: pd.DataFrame, groups: dict[str, list[str]], n: int = 3_000,
                seed: int = 42) -> pd.DataFrame:
    """Mean absolute TreeSHAP contribution (log-sales units) per feature group."""
    Xs = X.sample(min(n, len(X)), random_state=seed)
    contrib = booster.predict(Xs, pred_contrib=True)[:, :-1]
    cols = list(X.columns)
    rows = []
    for g, fs in groups.items():
        idx = [cols.index(f) for f in fs if f in cols]
        if idx:
            rows.append({"group": g, "mean_abs_contrib": float(np.abs(contrib[:, idx].sum(1)).mean())})
    return pd.DataFrame(rows).sort_values("mean_abs_contrib", ascending=False)
