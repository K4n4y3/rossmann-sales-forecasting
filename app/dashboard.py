"""Rossmann sales forecasting - mini dashboard (Streamlit).

Run from the repository root after `python run_pipeline.py`:

    streamlit run app/dashboard.py

Data source: outputs/rossmann_results.sqlite (written by the pipeline).
Filters: backtest window and period, region (store type / assortment - the data
has no geography), series (stores), models, promo / non-promo days.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DB = Path(__file__).resolve().parents[1] / "outputs" / "rossmann_results.sqlite"
COLORS = {"Actual": "#52514e", "lgbm_global": "#2a78d6", "lgbm_per_store": "#1baf7a", "ridge_per_store": "#eda100",
          "profile": "#e87ba4", "seasonal_naive": "#eb6834", "naive": "#6250d6", "sarimax_per_store": "#008300"}
DEFAULT_MODELS = ["lgbm_global", "lgbm_per_store", "seasonal_naive"]

st.set_page_config(page_title="Rossmann sales forecast", layout="wide")


@st.cache_data(show_spinner=False)
def table(name: str) -> pd.DataFrame:
    with sqlite3.connect(DB) as con:
        return pd.read_sql(f"SELECT * FROM {name}", con)


def scores(df: pd.DataFrame, col: str) -> dict:
    d = df[(df["Sales"] > 0) & df[col].notna()]
    if d.empty:
        return {}
    y, f = d["Sales"].to_numpy(float), d[col].to_numpy(float)
    e = y - f
    return {"RMSPE": float(np.sqrt(np.mean((e / y) ** 2))),
            "sMAPE %": float(100 * np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(f)))),
            "MAPE %": float(100 * np.mean(np.abs(e) / y)),
            "MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "MASE": float(np.mean(np.abs(e) / d["mase_scale"].to_numpy(float))),
            "Bias %": float(100 * (f.sum() - y.sum()) / y.sum()), "Days": int(len(d))}


if not DB.exists():
    st.error(f"{DB} not found - run `python run_pipeline.py` first.")
    st.stop()

pred = table("predictions")
pred["Date"] = pd.to_datetime(pred["Date"])
models = table("models").set_index("column")["model"].to_dict()
metrics = table("metrics")
fold_labels = metrics.drop_duplicates("fold").set_index("fold")["fold_label"].to_dict()

# ----------------------------------------------------------------------------- sidebar
st.sidebar.header("Filters")
fold = st.sidebar.selectbox("Backtest window", list(fold_labels), format_func=lambda f: fold_labels[f])
pf = pred[pred["fold"] == fold]
dmin, dmax = pf["Date"].min().date(), pf["Date"].max().date()
period = st.sidebar.slider("Period", min_value=dmin, max_value=dmax, value=(dmin, dmax), format="DD MMM YYYY")
types = st.sidebar.multiselect("Region: store type", sorted(pf["StoreType"].unique()),
                               default=sorted(pf["StoreType"].unique()))
assort = st.sidebar.multiselect("Region: assortment", sorted(pf["Assortment"].unique()),
                                default=sorted(pf["Assortment"].unique()),
                                format_func=lambda a: {"a": "a - basic", "b": "b - extra", "c": "c - extended"}[a])
pool = pf[pf["StoreType"].isin(types) & pf["Assortment"].isin(assort)]
stores = st.sidebar.multiselect("Series: stores (empty = all stores in the region)", sorted(pool["Store"].unique()))
available = [c for c in models if c in pf and pf[c].notna().any() and not c.startswith("lgbm_global_no")
             and c != "lgbm_global_corrected"]
chosen = st.sidebar.multiselect("Models", available, default=[m for m in DEFAULT_MODELS if m in available],
                                format_func=lambda c: models[c])
days = st.sidebar.radio("Days", ["All open days", "Promo days", "Non-promo days"])

sel = pool[(pool["Date"].dt.date >= period[0]) & (pool["Date"].dt.date <= period[1])]
if stores:
    sel = sel[sel["Store"].isin(stores)]
if days == "Promo days":
    sel = sel[sel["Promo"] == 1]
elif days == "Non-promo days":
    sel = sel[sel["Promo"] == 0]

# ----------------------------------------------------------------------------- header
st.title("Rossmann store sales - forecast dashboard")
st.caption(f"{fold_labels[fold]} · {sel['Store'].nunique():,} stores · {(sel['Sales'] > 0).sum():,} open store-days "
           "selected · forecasts made only from data before the window")
if sel.empty or not chosen:
    st.warning("No data for this selection - widen the filters or pick a model.")
    st.stop()

score_tab = pd.DataFrame({models[c]: scores(sel, c) for c in chosen}).T.sort_values("RMSPE")
best = score_tab.index[0]
k = st.columns([2.2, 1, 1, 1, 1])
k[0].metric("Best model (RMSPE)", best)
k[1].metric("RMSPE", f"{score_tab.loc[best, 'RMSPE']:.3f}")
k[2].metric("sMAPE", f"{score_tab.loc[best, 'sMAPE %']:.1f}%")
k[3].metric("MAE", f"{score_tab.loc[best, 'MAE']:,.0f} EUR")
k[4].metric("MASE", f"{score_tab.loc[best, 'MASE']:.2f}")

t1, t2, t3, t4 = st.tabs(["Forecast vs actual", "Errors", "Future forecast (Aug-Sep 2015)", "Promo & holiday effects"])

with t1:
    daily = sel.groupby("Date")[["Sales", *chosen]].sum(min_count=1)
    fig = go.Figure(go.Scatter(x=daily.index, y=daily["Sales"], name="Actual", line=dict(color=COLORS["Actual"], width=2)))
    for c in chosen:
        fig.add_trace(go.Scatter(x=daily.index, y=daily[c], name=models[c], line=dict(color=COLORS.get(c), width=2)))
    title = f"Store {stores[0]}" if len(stores) == 1 else ("Selected stores, total" if stores else "All stores in the region, total")
    fig.update_layout(title=f"{title}: daily sales, actual vs forecast", hovermode="x unified", height=430,
                      yaxis_title="Sales (EUR)", legend=dict(orientation="h", y=1.12), template="simple_white")
    st.plotly_chart(fig, width="stretch")
    st.subheader("Accuracy for the current selection")
    st.dataframe(score_tab.style.format({"RMSPE": "{:.4f}", "sMAPE %": "{:.2f}", "MAPE %": "{:.2f}", "MAE": "{:,.0f}",
                                         "RMSE": "{:,.0f}", "MASE": "{:.3f}", "Bias %": "{:+.2f}", "Days": "{:,.0f}"}),
                 width="stretch")

with t2:
    c1, c2 = st.columns(2)
    wk = pd.DataFrame({models[c]: sel.groupby("week").apply(lambda g, c=c: scores(g, c).get("RMSPE", np.nan))
                       for c in chosen})
    fig = go.Figure([go.Bar(x=[f"Week {i}" for i in wk.index], y=wk[m], name=m,
                            marker_color=COLORS.get([c for c in chosen if models[c] == m][0])) for m in wk])
    fig.update_layout(title="RMSPE by week of the window", barmode="group", height=380, template="simple_white",
                      legend=dict(orientation="h", y=1.15))
    c1.plotly_chart(fig, width="stretch")
    ref = chosen[0]
    ps = sel.groupby("Store").apply(lambda g: pd.Series(scores(g, ref))).dropna(subset=["RMSPE"])
    c2.markdown(f"**Stores with the largest error - {models[ref]}**")
    c2.dataframe(ps.sort_values("RMSPE", ascending=False).head(15)[["RMSPE", "sMAPE %", "MAE", "Bias %"]]
                 .style.format({"RMSPE": "{:.3f}", "sMAPE %": "{:.1f}", "MAE": "{:,.0f}", "Bias %": "{:+.1f}"}),
                 width="stretch")
    st.markdown("**All models, all windows: quality and training time** (whole chain, from the pipeline)")
    st.dataframe(metrics[["fold", "model", "RMSPE", "sMAPE", "MAE", "RMSE", "MASE", "Bias_%", "train_seconds", "n_models"]]
                 .style.format({"RMSPE": "{:.4f}", "sMAPE": "{:.2f}", "MAE": "{:,.0f}", "RMSE": "{:,.0f}",
                                "MASE": "{:.3f}", "Bias_%": "{:+.2f}", "train_seconds": "{:.1f}"}),
                 width="stretch", hide_index=True)

with t3:
    kf = table("kaggle_forecast")
    kf["Date"] = pd.to_datetime(kf["Date"])
    kf = kf[kf["StoreType"].isin(types)]
    hist = pred[(pred["fold"] == "F1") & pred["StoreType"].isin(types) & pred["Assortment"].isin(assort)]
    if stores:
        kf, hist = kf[kf["Store"].isin(stores)], hist[hist["Store"].isin(stores)]
    else:
        hist = hist[hist["Store"].isin(kf["Store"].unique())]
    if kf.empty:
        st.info("The selected stores are not in the Kaggle test period (it covers 856 of the 1,115 stores).")
    else:
        h = hist.groupby("Date")["Sales"].sum()
        f = kf.groupby("Date")["forecast"].sum()
        fig = go.Figure([go.Scatter(x=h.index, y=h, name="Actual (20 Jun - 31 Jul 2015)", line=dict(color=COLORS["Actual"])),
                         go.Scatter(x=f.index, y=f, name="Global LightGBM forecast (1 Aug - 17 Sep 2015)",
                                    line=dict(color=COLORS["lgbm_global"]))])
        fig.update_layout(title=f"Forecast for {kf['Store'].nunique():,} stores, model retrained on all data",
                          hovermode="x unified", height=430, yaxis_title="Sales (EUR)", template="simple_white",
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Forecast total: {f.sum() / 1e6:,.1f} M EUR over {f.index.nunique()} days. "
                   "Kaggle test period - true sales are not public.")

with t4:
    eff = table("effects").set_index("term")
    rows = ["Promo", "1 day(s) before holiday", "1 day(s) after holiday", "2 day(s) before holiday",
            "2 day(s) after holiday", "SchoolHoliday", "Promo2 month", "Open on state holiday",
            "Reopening (first week after closure >= 5 days)"]
    e = eff.loc[[r for r in rows if r in eff.index]]
    fig = go.Figure(go.Scatter(x=e["effect_pct"], y=e.index, mode="markers", marker=dict(color="#2a78d6", size=10),
                               error_x=dict(type="data", symmetric=False, array=e["ci_high_pct"] - e["effect_pct"],
                                            arrayminus=e["effect_pct"] - e["ci_low_pct"])))
    fig.update_layout(title="Effect on daily sales, % (store fixed-effects regression, 95% CI)", height=420,
                      template="simple_white", margin=dict(l=260))
    fig.update_yaxes(autorange="reversed")
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(fig, width="stretch")
    corr = table("correlations_store_day")[["factor", "spearman_sales", "spearman_customers", "share_of_days_pct"]]
    c2.markdown("**Correlation of external factors with sales** (Spearman, store mean removed)")
    c2.dataframe(corr.style.format({"spearman_sales": "{:+.3f}", "spearman_customers": "{:+.3f}",
                                    "share_of_days_pct": "{:.1f}"}), width="stretch", hide_index=True)
    pw = table("promo_by_weekday")
    pt = table("promo_by_storetype")
    c3, c4 = st.columns(2)
    c3.plotly_chart(go.Figure(go.Bar(x=["Mon", "Tue", "Wed", "Thu", "Fri"], y=pw["effect_pct"], marker_color="#2a78d6"))
                    .update_layout(title="Promo effect by weekday, %", height=320, template="simple_white"),
                    width="stretch")
    c4.plotly_chart(go.Figure(go.Bar(x=[f"Type {g}" for g in pt["group"]], y=pt["effect_pct"], marker_color="#2a78d6"))
                    .update_layout(title="Promo effect by store type, %", height=320, template="simple_white"),
                    width="stretch")
