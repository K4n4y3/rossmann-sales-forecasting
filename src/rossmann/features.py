"""Feature engineering.

Everything here is computed on a *complete* daily store x date grid so that
"days since / until" features are correct even across the Jul-Dec 2014 gap.
Only information that is known in advance for a future day is used
(calendar, promo schedule, holiday calendar, planned openings, store master
data).  Target statistics (``store_*``) are computed from the training window
only, via :func:`add_store_stats`, so they never leak hold-out sales.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dateutil.easter import easter

CAP = 30  # cap for all "days to / since" features

MONTH_ABBR = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul",
              8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec"}

CALENDAR = ["DayOfWeek", "Day", "Month", "Year", "WeekOfYear", "DayOfYear", "t"]
PROMO = ["Promo", "promo_day", "PromoNextWeek", "PromoLastWeek",
         "promo2_active", "promo2_weeks", "is_promo2_month"]
HOLIDAY = ["StateHoliday_code", "SchoolHoliday", "days_to_holiday", "days_since_holiday",
           "days_to_christmas", "days_since_christmas", "days_to_easter", "days_since_easter",
           "days_to_school_hol", "days_since_school_hol", "holidays_this_week", "holidays_next_week"]
CLOSURE = ["days_since_closed", "days_until_closed", "prev_closed_len", "next_closed_len"]
STORE_STATIC = ["StoreType_code", "Assortment_code", "log_comp_distance", "comp_open_months",
                "comp_active", "Promo2"]
STORE_STATS = ["store_mean_log", "store_dow_dev", "store_promo_lift", "store_mean_log_cust",
               "store_sunday_open"]

GLOBAL_FEATURES = ["Store"] + CALENDAR + PROMO + HOLIDAY + CLOSURE + STORE_STATIC + STORE_STATS
LOCAL_FEATURES = CALENDAR + PROMO + HOLIDAY + CLOSURE + ["comp_active"]
CATEGORICAL = ["Store", "StoreType_code", "Assortment_code", "StateHoliday_code"]

FEATURE_GROUPS = {"calendar": CALENDAR, "promo": PROMO, "holiday": HOLIDAY,
                  "closure": CLOSURE, "store": STORE_STATIC + STORE_STATS + ["Store"]}


# -----------------------------------------------------------------------------
# helpers working on a panel sorted by (Store, Date) with one row per day
# -----------------------------------------------------------------------------
def _days_since(flag: pd.Series, store: pd.Series, cap: int = CAP) -> np.ndarray:
    idx = pd.Series(np.arange(len(flag), dtype="float64"), index=flag.index)
    last = idx.where(flag.to_numpy()).groupby(store.to_numpy()).ffill()
    return np.minimum((idx - last).fillna(cap).to_numpy(), cap)


def _days_until(flag: pd.Series, store: pd.Series, cap: int = CAP) -> np.ndarray:
    idx = pd.Series(np.arange(len(flag), dtype="float64"), index=flag.index)
    nxt = idx.where(flag.to_numpy()).groupby(store.to_numpy()).bfill()
    return np.minimum((nxt - idx).fillna(cap).to_numpy(), cap)


def _run_position(flag: pd.Series, store: pd.Series) -> np.ndarray:
    """1, 2, 3 ... inside consecutive runs of flag == True, 0 outside."""
    f = flag.astype(int)
    new_run = (f != f.shift()) | (store != store.shift())
    run_id = new_run.cumsum()
    pos = f.groupby(run_id).cumcount() + 1
    return np.where(f == 1, pos, 0)


def _run_length(flag: pd.Series, store: pd.Series) -> pd.Series:
    f = flag.astype(int)
    new_run = (f != f.shift()) | (store != store.shift())
    run_id = new_run.cumsum()
    return f.groupby(run_id).transform("size").where(f == 1)


def _event_distance(dates: pd.Series, event_dates: list[pd.Timestamp], cap: int = CAP):
    """Days to the next and since the last event (calendar based, no store dimension)."""
    ev = np.array(sorted(event_dates), dtype="datetime64[D]")
    d = dates.to_numpy().astype("datetime64[D]")
    pos = np.searchsorted(ev, d, side="left")
    nxt = ev[np.minimum(pos, len(ev) - 1)]
    to = (nxt - d).astype(int)
    to = np.where(pos >= len(ev), cap, to)
    pos_r = np.searchsorted(ev, d, side="right") - 1
    prv = ev[np.maximum(pos_r, 0)]
    since = (d - prv).astype(int)
    since = np.where(pos_r < 0, cap, since)
    return np.minimum(to, cap), np.minimum(since, cap)


# -----------------------------------------------------------------------------
def build_panel(clean: pd.DataFrame, test: pd.DataFrame | None = None,
                store: pd.DataFrame | None = None) -> pd.DataFrame:
    """Observed train (+ test) rows on a full daily grid, with all features.

    Returns only rows that exist in train or test; column ``is_test`` marks
    Kaggle test rows (``Sales`` is NaN there).
    """
    base_cols = ["Store", "Date", "Sales", "Customers", "Open", "Promo", "StateHoliday",
                 "SchoolHoliday"]
    tr = clean[base_cols].copy()
    tr["is_test"] = 0
    tr["Id"] = np.nan
    frames = [tr]
    if test is not None:
        te = test[["Id", "Store", "Date", "Open", "Promo", "StateHoliday", "SchoolHoliday"]].copy()
        te["Sales"], te["Customers"], te["is_test"] = np.nan, np.nan, 1
        frames.append(te[tr.columns])
    obs = pd.concat(frames, ignore_index=True)
    obs["StateHoliday"] = obs["StateHoliday"].astype(str)
    obs["Sales"] = obs["Sales"].astype("float64")
    obs["Customers"] = obs["Customers"].astype("float64")
    obs["Open"] = obs["Open"].astype("float64")

    # --- full grid -----------------------------------------------------------
    stores = np.sort(obs["Store"].unique())
    dates = pd.date_range(obs["Date"].min(), obs["Date"].max(), freq="D")
    grid = pd.MultiIndex.from_product([stores, dates], names=["Store", "Date"]).to_frame(index=False)
    p = grid.merge(obs, on=["Store", "Date"], how="left", indicator=True)
    p["exists"] = p.pop("_merge") == "both"

    # chain-wide calendar values fill the grid where a store has no row
    # (promo is identical for every store on a date; holidays: majority value)
    by_date = obs.groupby("Date").agg(
        promo_d=("Promo", "max"),
        hol_d=("StateHoliday", lambda s: s.value_counts().idxmax()),
    )
    p = p.join(by_date, on="Date")
    p["Promo"] = p["Promo"].fillna(p["promo_d"]).fillna(0).astype("int8")
    p["StateHoliday"] = p["StateHoliday"].fillna(p["hol_d"]).fillna("0")
    p["SchoolHoliday"] = p["SchoolHoliday"].fillna(0).astype("int8")
    p = p.drop(columns=["promo_d", "hol_d"])
    p = p.sort_values(["Store", "Date"]).reset_index(drop=True)
    s = p["Store"]

    # --- calendar --------------------------------------------------------------
    d = p["Date"]
    p["DayOfWeek"] = (d.dt.dayofweek + 1).astype("int8")
    p["Day"] = d.dt.day.astype("int8")
    p["Month"] = d.dt.month.astype("int8")
    p["Year"] = d.dt.year.astype("int16")
    p["WeekOfYear"] = d.dt.isocalendar().week.astype("int8").to_numpy()
    p["DayOfYear"] = d.dt.dayofyear.astype("int16")
    p["t"] = ((d - pd.Timestamp("2013-01-01")).dt.days).astype("int16")  # linear trend

    # --- promo -----------------------------------------------------------------
    p["promo_day"] = _run_position(p["Promo"] == 1, s).astype("int8")
    g = p.groupby("Store")["Promo"]
    p["PromoNextWeek"] = g.shift(-7).fillna(0).astype("int8")
    p["PromoLastWeek"] = g.shift(7).fillna(0).astype("int8")

    # --- holidays ----------------------------------------------------------------
    hol = p["StateHoliday"] != "0"
    p["StateHoliday_code"] = p["StateHoliday"].map({"0": 0, "a": 1, "b": 2, "c": 3}).astype("int8")
    p["IsStateHoliday"] = hol.astype("int8")
    p["days_to_holiday"] = _days_until(hol, s)
    p["days_since_holiday"] = _days_since(hol, s)
    years = range(p["Year"].min() - 1, p["Year"].max() + 2)
    p["days_to_christmas"], p["days_since_christmas"] = _event_distance(
        d, [pd.Timestamp(y, 12, 25) for y in years], cap=60)
    p["days_to_easter"], p["days_since_easter"] = _event_distance(
        d, [pd.Timestamp(easter(y)) for y in years])
    school = p["SchoolHoliday"] == 1
    p["days_to_school_hol"] = _days_until(school, s)
    p["days_since_school_hol"] = _days_since(school, s)
    wk = d.dt.to_period("W-SUN")
    p["holidays_this_week"] = p.groupby([s, wk])["IsStateHoliday"].transform("sum").astype("int8")
    p["holidays_next_week"] = p.groupby("Store")["holidays_this_week"].shift(-7).fillna(0).astype("int8")

    # --- closures (refurbishments, Sundays, holidays) --------------------------
    open_ = p["Open"].fillna(1)  # unknown days (gap, missing) treated as open
    closed = open_ == 0
    p["days_since_closed"] = _days_since(closed, s, cap=60)
    p["days_until_closed"] = _days_until(closed, s, cap=60)
    run_len = _run_length(closed, s)
    p["prev_closed_len"] = run_len.groupby(s.to_numpy()).ffill().fillna(0).clip(upper=60).to_numpy()
    p["next_closed_len"] = run_len.groupby(s.to_numpy()).bfill().fillna(0).clip(upper=60).to_numpy()
    # on an open day the "current" run is open; ffill/bfill give the neighbouring closed runs

    p = p[p["exists"]].drop(columns="exists").reset_index(drop=True)
    p["Open"] = p["Open"].fillna(1).astype("int8")

    if store is not None:
        p = add_store_static(p, store)
    return p


def add_store_static(p: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    st = store.copy()
    st["StoreType_code"] = st["StoreType"].astype(str).map({"a": 0, "b": 1, "c": 2, "d": 3}).astype("int8")
    st["Assortment_code"] = st["Assortment"].astype(str).map({"a": 0, "b": 1, "c": 2}).astype("int8")
    dist = st["CompetitionDistance"].astype("float64")
    st["log_comp_distance"] = np.log1p(dist.fillna(dist.median()))
    keep = ["Store", "StoreType", "Assortment", "StoreType_code", "Assortment_code",
            "log_comp_distance", "CompetitionOpenSinceYear", "CompetitionOpenSinceMonth",
            "Promo2", "Promo2SinceYear", "Promo2SinceWeek", "PromoInterval"]
    p = p.merge(st[keep], on="Store", how="left")

    # competition: months since the competitor opened (NaN if unknown)
    cy = p.pop("CompetitionOpenSinceYear").astype("float64")
    cm = p.pop("CompetitionOpenSinceMonth").astype("float64")
    months = (p["Year"] - cy) * 12 + (p["Month"] - cm)
    p["comp_open_months"] = months.clip(lower=-1, upper=240)
    p["comp_active"] = np.where(months.isna(), 1, (months >= 0).astype(int)).astype("int8")

    # Promo2: continuing promotion, active from (Promo2SinceYear, Promo2SinceWeek)
    py = p.pop("Promo2SinceYear").astype("float64")
    pw = p.pop("Promo2SinceWeek").astype("float64")
    start = pd.to_datetime(py.astype("Int64").astype(str) + "-01-01", errors="coerce") + \
        pd.to_timedelta((pw - 1) * 7, unit="D")
    active = (p["Promo2"] == 1) & (p["Date"] >= start)
    p["promo2_active"] = active.astype("int8")
    p["promo2_weeks"] = np.where(active, (p["Date"] - start).dt.days // 7, -1).astype("int16")
    month_abbr = p["Month"].map(MONTH_ABBR)
    interval = p.pop("PromoInterval").astype(str)
    in_month = [m in iv.split(",") for m, iv in zip(month_abbr, interval)]
    p["is_promo2_month"] = (active & np.array(in_month)).astype("int8")
    return p


def add_store_stats(p: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Per-store target statistics from the training window only (no leakage)."""
    p = p.drop(columns=[c for c in STORE_STATS if c in p], errors="ignore")
    m = train_mask & (p["Open"] == 1) & (p["Sales"] > 0)
    tr = p.loc[m, ["Store", "DayOfWeek", "Promo", "Sales", "Customers"]].copy()
    tr["ls"] = np.log1p(tr["Sales"])
    tr["lc"] = np.log1p(tr["Customers"])
    st = tr.groupby("Store").agg(store_mean_log=("ls", "mean"), store_mean_log_cust=("lc", "mean"))
    promo = tr.groupby(["Store", "Promo"])["ls"].mean().unstack()
    st["store_promo_lift"] = (promo.get(1) - promo.get(0)).fillna(0)
    dow = tr.groupby(["Store", "DayOfWeek"])["ls"].mean().rename("dow_mean").reset_index()
    dow = dow.merge(st["store_mean_log"].reset_index(), on="Store")
    dow["store_dow_dev"] = dow["dow_mean"] - dow["store_mean_log"]
    sun = p.loc[train_mask & (p["DayOfWeek"] == 7)].groupby("Store")["Open"].mean()
    st["store_sunday_open"] = sun

    p = p.merge(st.reset_index(), on="Store", how="left")
    p = p.merge(dow[["Store", "DayOfWeek", "store_dow_dev"]], on=["Store", "DayOfWeek"], how="left")
    p["store_dow_dev"] = p["store_dow_dev"].fillna(0)
    return p
