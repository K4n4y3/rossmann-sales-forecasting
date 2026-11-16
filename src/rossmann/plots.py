"""Figures for the report and slides (static PNG via matplotlib, interactive HTML via Plotly).

Style: one light surface, hairline solid grid, 2 px lines, a fixed categorical
order (blue, orange, aqua), no dual axes, legends for >= 2 series plus
selective direct labels.  Text always uses ink colours, never the series colour.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.ticker import FuncFormatter, PercentFormatter
from plotly.subplots import make_subplots

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY = "#b9b8b0"
SERIES = [BLUE, ORANGE, AQUA]
FONT = ["DejaVu Sans"]

MODEL_ORDER = ["Naive (last observed day)", "Seasonal naive (last same weekday)", "Store x weekday x promo profile",
               "Per-store SARIMAX (ARMA errors)", "Per-store ridge regression",
               "Per-store LightGBM", "Global LightGBM"]
SHORT = {"Naive (last observed day)": "Naive", "Seasonal naive (last same weekday)": "Seasonal naive",
         "Store x weekday x promo profile": "Weekday x promo profile",
         "Per-store SARIMAX (ARMA errors)": "Per-store SARIMAX",
         "Per-store ridge regression": "Per-store ridge",
         "Per-store LightGBM": "Per-store LightGBM",
         "Global LightGBM": "Global LightGBM"}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": FONT, "font.size": 11,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.titlelocation": "left", "axes.titlepad": 24, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 10, "legend.labelcolor": INK2,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def _save(fig, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _subtitle(ax, text):
    ax.text(0, 1.012, text, transform=ax.transAxes, color=INK2, fontsize=10, va="bottom")


def _kfmt(x, _):
    return f"{x / 1e6:.0f}M" if abs(x) >= 1e6 else (f"{x / 1e3:.0f}K" if abs(x) >= 1e4 else f"{x:,.0f}")


def _end_label(ax, x, y, text, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center",
                color=INK2, fontsize=9.5)


# -----------------------------------------------------------------------------
def total_sales_overview(daily: pd.DataFrame, folds: list[tuple], path):
    """Weekly mean sales per open store with the backtest windows shaded."""
    style()
    w = daily.resample("W-SUN").agg(Sales=("Sales", "sum"), n_open=("n_open", "sum"))
    w = w.iloc[1:-1]
    w["per_store"] = w["Sales"] / w["n_open"]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, (s, e, label) in enumerate(folds):
        ax.axvspan(s, e, color=BLUE if i == 0 else GRAY, alpha=0.12 if i == 0 else 0.18, lw=0)
        ax.text(s + (e - s) / 2, 1.0, label, transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=9, color=INK2)
    ax.plot(w.index, w["per_store"], color=BLUE)
    ax.set_ylabel("Mean daily sales per open store (EUR)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_title("Weekly sales per open store, Jan 2013 to Jul 2015")
    _subtitle(ax, "Christmas peaks and a bi-weekly promo rhythm; shaded: 3 backtest hold-outs (F1 = main)")
    ax.set_ylim(bottom=0)
    return _save(fig, path)


def model_comparison(metrics: pd.DataFrame, path, fold: str = "F1"):
    """RMSPE and sMAPE per model on the main hold-out (small multiples, one measure each)."""
    style()
    m = metrics[metrics["fold"] == fold].set_index("model").reindex(
        [x for x in MODEL_ORDER if x in set(metrics.loc[metrics.fold == fold, "model"])])
    m = m[m.index != "Per-store SARIMAX (ARMA errors)"]
    labels = [SHORT[i] for i in m.index]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, col, fmt in [(axes[0], "RMSPE", "{:.3f}"), (axes[1], "sMAPE", "{:.1f}%")]:
        colors = [BLUE if i == "Global LightGBM" else GRAY for i in m.index]
        y = np.arange(len(m))
        ax.barh(y, m[col], height=0.5, color=colors)
        for yi, v in zip(y, m[col]):
            ax.text(v, yi, "  " + fmt.format(v), va="center", color=INK, fontsize=10)
        ax.set_yticks(y, labels)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, m[col].max() * 1.22)
        ax.set_title(col + (" (lower is better)" if col == "RMSPE" else ", % (lower is better)"), fontsize=12, pad=10)
    axes[0].invert_yaxis()
    fig.suptitle("Hold-out accuracy, last 6 weeks (20 Jun - 31 Jul 2015), 1,115 stores",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, path)


def backtest_folds(metrics: pd.DataFrame, path, models=("Global LightGBM", "Per-store LightGBM",
                                                         "Per-store ridge regression",
                                                         "Store x weekday x promo profile")):
    style()
    m = metrics[metrics["model"].isin(models)]
    folds = sorted(m["fold"].unique())
    fig, ax = plt.subplots(figsize=(10, 4.2))
    y = np.arange(len(models))
    for j, f in enumerate(folds):
        v = m[m.fold == f].set_index("model").reindex(models)["RMSPE"]
        ax.scatter(v, y + (j - 1) * 0.18, s=60, color=SERIES[j], edgecolor=SURFACE, linewidth=2,
                   label=m.loc[m.fold == f, "fold_label"].iat[0], zorder=3)
    ax.set_yticks(y, [SHORT[x] for x in models])
    ax.invert_yaxis()
    ax.set_xlabel("RMSPE (lower is better)")
    ax.grid(axis="y", visible=False)
    ax.set_title("The ranking holds on all three 6-week backtest windows")
    _subtitle(ax, "Rolling-origin evaluation, each window forecast from data before it")
    ax.legend(loc="upper right", ncols=1)
    return _save(fig, path)


def per_store_scatter(per_store: pd.DataFrame, path):
    style()
    d = per_store.pivot(index="Store", columns="model", values="RMSPE").dropna(
        subset=["Global LightGBM", "Per-store LightGBM"])
    x, y = d["Per-store LightGBM"], d["Global LightGBM"]
    win = float((y < x).mean())
    fig, ax = plt.subplots(figsize=(6.4, 6))
    lim = float(np.nanpercentile(np.r_[x, y], 99.5)) * 1.05
    ax.plot([0, lim], [0, lim], color=AXIS, lw=1)
    ax.scatter(x.clip(upper=lim), y.clip(upper=lim), s=14, color=BLUE, alpha=0.55, linewidth=0)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Per-store LightGBM - RMSPE per store")
    ax.set_ylabel("Global LightGBM - RMSPE per store")
    ax.text(0.97, 0.04, f"Global better for {win:.0%} of stores\n(points below the diagonal)",
            transform=ax.transAxes, ha="right", va="bottom", color=INK, fontsize=10)
    ax.set_title("Per-store error: global vs per-store model")
    _subtitle(ax, "Hold-out F1, one dot per store (1,115)")
    return _save(fig, path), win


def error_by_week(by_week: pd.DataFrame, path, models=("Store x weekday x promo profile",
                                                       "Per-store LightGBM", "Global LightGBM")):
    style()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = {models[0]: ORANGE, models[1]: AQUA, models[2]: BLUE}
    for mdl in models:
        g = by_week[by_week.model == mdl].sort_values("week")
        ax.plot(g["week"], g["RMSPE"], color=colors[mdl], marker="o", ms=7, mec=SURFACE, mew=2,
                label=SHORT[mdl])
        _end_label(ax, g["week"].iat[-1], g["RMSPE"].iat[-1], SHORT[mdl])
    ax.set_xlabel("Week of the hold-out (1 = 20-26 Jun 2015)")
    ax.set_ylabel("RMSPE")
    ax.set_xlim(0.8, 7.4)
    ax.set_xticks(range(1, 7))
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower left")
    ax.set_title("Error by forecast horizon")
    _subtitle(ax, "No model degrades much over 6 weeks: the drivers (calendar, promo) are known in advance")
    return _save(fig, path)


def holdout_total(pred: pd.DataFrame, path, models=("Global LightGBM", "Per-store LightGBM")):
    style()
    t = pred.groupby("Date")[["Sales", *[f"pred::{m}" for m in models]]].sum()
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(t.index, t["Sales"], color=INK2, lw=2, label="Actual")
    for c, mdl in zip([BLUE, AQUA], models):
        ax.plot(t.index, t[f"pred::{mdl}"], color=c, lw=2, label=SHORT[mdl])
    ax.yaxis.set_major_formatter(FuncFormatter(_kfmt))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_ylabel("Total daily sales, all stores (EUR)")
    ax.set_ylim(0, float(t.to_numpy().max()) * 1.22)
    ax.legend(loc="upper left", ncols=3)
    ax.set_title("Hold-out: total daily sales, actual vs forecast")
    _subtitle(ax, "Sundays are near zero (most stores closed); promo weeks visibly higher")
    return _save(fig, path)


def store_examples(pred: pd.DataFrame, stores: list[int], path, models=("Global LightGBM", "Per-store LightGBM")):
    style()
    fig, axes = plt.subplots(len(stores), 1, figsize=(11, 2.6 * len(stores)), sharex=True)
    for ax, s in zip(np.atleast_1d(axes), stores):
        g = pred[(pred.Store == s) & (pred.Open == 1)].sort_values("Date")
        ax.plot(g["Date"], g["Sales"], color=INK2, lw=2, label="Actual")
        for c, mdl in zip([BLUE, AQUA], models):
            ax.plot(g["Date"], g[f"pred::{mdl}"], color=c, lw=2, label=SHORT[mdl])
        st = g["StoreType"].iat[0]
        ax.set_title(f"Store {s} (type {st})", fontsize=11, pad=8)
        ax.yaxis.set_major_formatter(FuncFormatter(_kfmt))
        ax.set_ylim(bottom=0)
    np.atleast_1d(axes)[0].legend(loc="lower left", ncols=3)
    np.atleast_1d(axes)[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.suptitle("Example stores in the hold-out (open days)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, path)


def coef_plot(tab: pd.DataFrame, path, title: str, subtitle: str, rows: list[str] | None = None,
              labels: dict | None = None, figsize=(9, 5)):
    """Effect in % with 95% CI, one row per regressor."""
    style()
    t = tab.loc[rows] if rows else tab
    lab = [labels.get(i, i) if labels else i for i in t.index]
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(t))
    ax.axvline(0, color=AXIS, lw=1)
    ax.hlines(y, t["ci_low_pct"], t["ci_high_pct"], color=BLUE, lw=2)
    ax.scatter(t["effect_pct"], y, s=64, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    for yi, v, hi in zip(y, t["effect_pct"], t["ci_high_pct"]):
        ax.text(max(v, hi), yi, f"  {v:+.1f}%", va="center", color=INK, fontsize=10)
    ax.set_yticks(y, lab)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    lo, hi = float(t["ci_low_pct"].min()), float(t["ci_high_pct"].max())
    ax.set_xlim(min(lo, 0) - 5, hi + (hi - min(lo, 0)) * 0.18)
    ax.set_xlabel("Effect on daily sales of an open store (95% CI)")
    ax.set_title(title)
    _subtitle(ax, subtitle)
    return _save(fig, path)


def christmas_event(tab: pd.DataFrame, path):
    style()
    names = [f"xmas_D-{k}" for k in range(7, 0, -1)] + [f"xmas_D+{k}" for k in range(2, 7)]
    t = tab.loc[[n for n in names if n in tab.index]]
    days = [18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30, 31]
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.axhline(0, color=AXIS, lw=1)
    ax.vlines(x, t["ci_low_pct"], t["ci_high_pct"], color=BLUE, lw=2)
    ax.scatter(x, t["effect_pct"], s=64, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    for xi, v in zip(x, t["effect_pct"]):
        if abs(v) > 20:
            ax.text(xi + 0.15, v, f"{v:+.0f}%", va="center", color=INK, fontsize=9.5)
    ax.set_xticks(x, [f"{d} Dec" for d in days], rotation=0, fontsize=9)
    ax.axvspan(6.5, 6.5, color=AXIS)
    ax.text(6.5, 0.02, "25-26 Dec closed", transform=ax.get_xaxis_transform(), ha="center",
            color=MUTED, fontsize=9)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_ylabel("Effect vs an ordinary December day")
    ax.set_title("Christmas: demand is pulled into the last shopping days")
    _subtitle(ax, "Fixed-effects regression, store + calendar controls, 95% CI")
    return _save(fig, path)


def promo_heterogeneity(by_dow: pd.DataFrame, by_type: pd.DataFrame, path):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [5, 4]})
    dname = {"1": "Mon", "2": "Tue", "3": "Wed", "4": "Thu", "5": "Fri"}
    for ax, d, lab, title in [(axes[0], by_dow, [dname[str(g)] for g in by_dow["group"]], "By weekday"),
                              (axes[1], by_type, [f"Type {g}" for g in by_type["group"]], "By store type")]:
        x = np.arange(len(d))
        ax.bar(x, d["effect_pct"], width=0.5, color=BLUE)
        for xi, v in zip(x, d["effect_pct"]):
            ax.text(xi, v, f"+{v:.0f}%", ha="center", va="bottom", color=INK, fontsize=10)
        ax.set_xticks(x, lab)
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax.set_ylim(0, d["effect_pct"].max() * 1.18)
        ax.set_title(title, fontsize=12, pad=10)
    axes[0].set_ylabel("Promo effect on daily sales")
    fig.suptitle("Promo uplift is largest on Mondays and smallest in type-b stores",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, path)


def promo_decomposition(eff: dict, path):
    """Promo effect on sales split into traffic (customers) and basket size."""
    style()
    labels = ["Sales", "Customers (traffic)", "Sales per customer (basket)"]
    keys = ["Sales", "Customers", "SalesPerCustomer"]
    v = [eff[k]["effect_pct"] for k in keys]
    lo = [eff[k]["ci_low_pct"] for k in keys]
    hi = [eff[k]["ci_high_pct"] for k in keys]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    y = np.arange(3)
    ax.barh(y, v, height=0.5, color=[BLUE, GRAY, GRAY])
    for yi, val, h in zip(y, v, hi):
        ax.text(val, yi, f"  +{val:.1f}%", va="center", color=INK, fontsize=10.5)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlim(0, max(v) * 1.25)
    ax.set_title("A promo day: more shoppers and bigger baskets")
    _subtitle(ax, "1.20 x 1.15 = 1.38; fixed-effects estimates, 95% CI within +/-1.7 pp")
    return _save(fig, path)


def feature_groups(imp: pd.DataFrame, path, top: int = 12):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [4, 5]})
    g = imp.groupby("group")["gain_share"].sum().sort_values(ascending=False)
    names = {"store": "Store identity & history", "promo": "Promo calendar", "calendar": "Calendar",
             "holiday": "Holidays", "closure": "Openings / closures"}
    y = np.arange(len(g))
    axes[0].barh(y, g.values * 100, height=0.5, color=BLUE)
    for yi, v in zip(y, g.values * 100):
        axes[0].text(v, yi, f"  {v:.0f}%", va="center", color=INK, fontsize=10)
    axes[0].set_yticks(y, [names.get(i, i) for i in g.index])
    axes[0].set_xlim(0, g.max() * 100 * 1.25)
    axes[0].set_title("Share of total gain by feature group", fontsize=12, pad=10)
    t = imp.head(top)
    y = np.arange(len(t))
    axes[1].barh(y, t["gain_share"] * 100, height=0.5, color=GRAY)
    for yi, v in zip(y, t["gain_share"] * 100):
        axes[1].text(v, yi, f"  {v:.1f}%", va="center", color=INK, fontsize=9.5)
    axes[1].set_yticks(y, t["feature"], fontsize=9.5)
    axes[1].set_xlim(0, t["gain_share"].max() * 100 * 1.25)
    axes[1].set_title(f"Top {top} features", fontsize=12, pad=10)
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    fig.suptitle("Global LightGBM: what drives the forecast", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, path)


def ablation(ab: pd.DataFrame, path):
    """RMSPE by day segment for the full model and models without promo / holiday features."""
    style()
    variants = list(ab["variant"].unique())
    segs = list(ab["segment"].unique())
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(segs))
    w = 0.22
    for j, (v, c) in enumerate(zip(variants, SERIES)):
        d = ab[ab.variant == v].set_index("segment").reindex(segs)
        ax.bar(x + (j - 1) * (w + 0.02), d["RMSPE"], width=w, color=c, label=v)
        for xi, val in zip(x + (j - 1) * (w + 0.02), d["RMSPE"]):
            ax.text(xi, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(x, segs)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("RMSPE (backtest window F3, with Easter)")
    ax.legend(loc="upper left", ncols=3)
    ax.set_ylim(0, ab["RMSPE"].max() * 1.25)
    ax.set_title("Without the promo calendar the forecast breaks on promo days")
    _subtitle(ax, "Global LightGBM retrained without feature groups (ablation)")
    return _save(fig, path)


def correlation_bars(store_day: pd.DataFrame, path):
    """Spearman correlation of each external factor with store-demeaned log sales."""
    style()
    d = store_day.sort_values("spearman_sales")
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    y = np.arange(len(d))
    ax.axvline(0, color=AXIS, lw=1)
    colors = [BLUE if f == "Promo day" else GRAY for f in d["factor"]]
    ax.barh(y, d["spearman_sales"], height=0.5, color=colors)
    for yi, v in zip(y, d["spearman_sales"]):
        ax.text(v + (0.01 if v >= 0 else -0.01), yi, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", color=INK, fontsize=10)
    ax.set_yticks(y, d["factor"])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(-0.1, d["spearman_sales"].min() - 0.08), d["spearman_sales"].max() + 0.12)
    ax.set_xlabel("Spearman correlation with daily sales (log, store mean removed)")
    ax.set_title("Promotions are by far the strongest external driver of sales")
    _subtitle(ax, "844k open store-days, 2013-2015; holidays and Christmas week follow")
    return _save(fig, path)


# -----------------------------------------------------------------------------
# interactive (Plotly)
# -----------------------------------------------------------------------------
def _pl_layout(fig, title):
    fig.update_layout(title=dict(text=title, x=0.01), template="simple_white", paper_bgcolor=SURFACE,
                      plot_bgcolor=SURFACE, font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif",
                                                       color=INK2), hovermode="x unified",
                      legend=dict(orientation="h", y=1.08, x=0))
    fig.update_xaxes(showgrid=False, linecolor=AXIS)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS)
    return fig


def interactive_holdout(pred: pd.DataFrame, path, stores: list[int],
                        models=("Global LightGBM", "Per-store LightGBM", "Store x weekday x promo profile")):
    colors = [BLUE, AQUA, ORANGE]
    t = pred.groupby("Date")[["Sales", *[f"pred::{m}" for m in models]]].sum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t.index, y=t["Sales"], name="Actual", line=dict(color=INK2, width=2)))
    for c, m in zip(colors, models):
        fig.add_trace(go.Scatter(x=t.index, y=t[f"pred::{m}"], name=SHORT[m], line=dict(color=c, width=2)))
    n_total = len(fig.data)
    for s in stores:
        g = pred[pred.Store == s].sort_values("Date")
        fig.add_trace(go.Scatter(x=g["Date"], y=g["Sales"], name="Actual", visible=False,
                                 line=dict(color=INK2, width=2)))
        for c, m in zip(colors, models):
            fig.add_trace(go.Scatter(x=g["Date"], y=g[f"pred::{m}"], name=SHORT[m], visible=False,
                                     line=dict(color=c, width=2)))
    k = 1 + len(models)
    buttons = [dict(label="All stores (total)", method="update",
                    args=[{"visible": [i < n_total for i in range(len(fig.data))]}])]
    for j, s in enumerate(stores):
        vis = [n_total + j * k <= i < n_total + (j + 1) * k for i in range(len(fig.data))]
        buttons.append(dict(label=f"Store {s}", method="update", args=[{"visible": vis}]))
    fig.update_layout(updatemenus=[dict(buttons=buttons, x=1.0, xanchor="right", y=1.15)])
    _pl_layout(fig, "Hold-out forecast vs actual (20 Jun - 31 Jul 2015)")
    fig.update_yaxes(title="Sales (EUR)")
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def interactive_correlation_matrix(matrix: pd.DataFrame, path):
    fig = go.Figure(go.Heatmap(z=matrix.to_numpy(), x=list(matrix.columns), y=list(matrix.index), zmin=-1, zmax=1,
                               colorscale=[[0, "#d03b3b"], [0.5, "#f0efec"], [1, BLUE]],
                               text=np.round(matrix.to_numpy(), 2), texttemplate="%{text}",
                               hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
                               colorbar=dict(title="Spearman")))
    _pl_layout(fig, "Correlation of sales, customers and external factors (Spearman, open store-days)")
    fig.update_layout(height=650, hovermode="closest", margin=dict(l=220, b=200))
    fig.update_yaxes(autorange="reversed")
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def interactive_effects(tab: pd.DataFrame, path):
    t = tab.iloc[::-1]
    fig = go.Figure(go.Scatter(
        x=t["effect_pct"], y=t.index, mode="markers", marker=dict(color=BLUE, size=10),
        error_x=dict(type="data", symmetric=False, array=t["ci_high_pct"] - t["effect_pct"],
                     arrayminus=t["effect_pct"] - t["ci_low_pct"], color=BLUE),
        hovertemplate="%{y}: %{x:+.1f}%<extra></extra>"))
    _pl_layout(fig, "Promo and holiday effects on daily sales (store fixed effects, 95% CI)")
    fig.update_layout(hovermode="closest", height=650, margin=dict(l=320))
    fig.update_xaxes(title="Effect on sales (%)", zeroline=True, zerolinecolor=AXIS)
    fig.write_html(path, include_plotlyjs="cdn")
    return path
