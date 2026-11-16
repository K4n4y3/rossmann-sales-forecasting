"""End-to-end pipeline: data -> features -> backtests -> effects -> submission -> figures.

Run from the repository root::

    python -m rossmann.pipeline              # full run (~35 min on 2 CPU cores)
    python -m rossmann.pipeline --quick      # 60 stores, 1 fold - smoke test (~2 min), writes to outputs/quick/

Every stage writes its results to ``outputs/`` so the notebooks can load them.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config, data, effects as E, features as F, metrics as MET, models as M, plots as P

FOLDS = [
    ("F1", "2015-06-20", "F1: 20 Jun - 31 Jul 2015 (main)"),
    ("F2", "2015-05-09", "F2: 9 May - 19 Jun 2015"),
    ("F3", "2015-03-28", "F3: 28 Mar - 8 May 2015 (Easter)"),
]
PRED_KEEP = ["Store", "Date", "Sales", "Open", "Promo", "StoreType", "DayOfWeek", "StateHoliday_code",
             "SchoolHoliday", "days_to_holiday", "days_since_holiday"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -----------------------------------------------------------------------------
def prepare(raw_dir: Path = config.RAW_DIR, rebuild: bool = False, stores: list[int] | None = None):
    config.ensure_dirs()
    clean = data.get_clean(raw_dir, rebuild=rebuild)
    test, store = data.load_test(raw_dir), data.load_store(raw_dir)
    panel_path = config.DATA_DIR / "panel.parquet"
    if panel_path.exists() and not rebuild and stores is None:
        panel = pd.read_parquet(panel_path)
    else:
        panel = F.build_panel(clean, test, store)
        if stores is None:
            panel.to_parquet(panel_path, index=False)
    if stores is not None:
        panel = panel[panel["Store"].isin(stores)].reset_index(drop=True)
        clean = clean[clean["Store"].isin(stores)].reset_index(drop=True)
    return clean, test, store, panel


def segments(pred: pd.DataFrame, col: str) -> pd.DataFrame:
    near_hol = (pred["days_to_holiday"] <= 3) | (pred["days_since_holiday"] <= 3)
    segs = {"All open days": pred["Open"] == 1, "Promo days": pred["Promo"] == 1,
            "Non-promo days": pred["Promo"] == 0, "+/-3 days around a holiday": near_hol,
            "School holiday": pred["SchoolHoliday"] == 1,
            "Stores with 2014 gap": pred["has_gap"], "Stores without gap": ~pred["has_gap"]}
    for t in ["a", "b", "c", "d"]:
        segs[f"Store type {t}"] = pred["StoreType"].astype(str) == t
    rows = []
    for name, m in segs.items():
        g = pred[m]
        if (g["Sales"] > 0).sum():
            rows.append({"segment": name, "RMSPE": MET.rmspe(g["Sales"], g[col]),
                         "sMAPE": MET.smape(g["Sales"], g[col]), "n": int((g["Sales"] > 0).sum())})
    return pd.DataFrame(rows)


def run_fold(panel: pd.DataFrame, fold: tuple, gap_stores: set, n_jobs: int = -1,
             sarimax_stores: list[int] | None = None, extras: bool = False, ablation: bool = False):
    fid, start, label = fold
    split = M.Split(pd.Timestamp(start))
    log(f"{fid}: hold-out {split.holdout_start.date()} - {split.holdout_end.date()}")
    p_fit = F.add_store_stats(panel, panel["Date"] < split.valid_start)
    p_train = F.add_store_stats(panel, panel["Date"] < split.holdout_start)

    results = [M.seasonal_naive(p_train, split), M.profile_baseline(p_train, split)]
    log(f"{fid}: baselines done")
    results.append(M.per_store_ridge(p_train, split))
    log(f"{fid}: per-store ridge done ({results[-1].train_seconds:.0f}s)")
    results.append(M.per_store_lgb(p_train, split, n_jobs=n_jobs))
    log(f"{fid}: per-store LightGBM done ({results[-1].train_seconds:.0f}s, "
        f"{results[-1].info['n_rounds']} trees per store)")
    glob, booster = M.fit_global(p_fit, p_train, split)
    results.append(glob)
    log(f"{fid}: global LightGBM done ({glob.train_seconds:.0f}s, {glob.info['n_trees']} trees)")
    if sarimax_stores:
        results.append(M.per_store_sarimax(p_train, split, sarimax_stores, n_jobs=n_jobs))
        log(f"{fid}: per-store SARIMAX on {len(sarimax_stores)} stores done ({results[-1].train_seconds:.0f}s)")

    ho = p_train.loc[split.masks(p_train)["holdout"], PRED_KEEP].copy()
    ho["StoreType"] = ho["StoreType"].astype(str)
    ho["has_gap"] = ho["Store"].isin(gap_stores)
    ho["fold"] = fid
    ho["week"] = (ho["Date"] - split.holdout_start).dt.days // 7 + 1
    for r in results:
        ho[f"pred::{r.name}"] = r.pred.reindex(ho.index)
    ho["pred::Global LightGBM (corrected)"] = glob.info["pred_corrected"].reindex(ho.index)

    rows = []
    for r in results:
        sub = ho[ho[f"pred::{r.name}"].notna()]
        rows.append({"fold": fid, "fold_label": label, "model": r.name, "n_models": r.n_models,
                     "train_seconds": round(r.train_seconds, 1), "n_eval_days": int((sub.Sales > 0).sum()),
                     **MET.score(sub["Sales"], sub[f"pred::{r.name}"])})
    met = pd.DataFrame(rows)
    info = {"global": {k: v for k, v in glob.info.items() if not isinstance(v, pd.Series)},
            "per_store_lgb": {k: v for k, v in results[3].info.items() if not isinstance(v, pd.Series)},
            "global_corrected_holdout": MET.score(ho["Sales"], ho["pred::Global LightGBM (corrected)"])}
    out = {"pred": ho, "metrics": met, "info": info}

    if sarimax_stores:  # all models compared on the SARIMAX sample
        sub = ho[ho["Store"].isin(sarimax_stores)]
        out["sample_metrics"] = pd.DataFrame(
            [{"model": r.name, **MET.score(sub["Sales"], sub[f"pred::{r.name}"])} for r in results])

    if extras:
        log(f"{fid}: diagnostics (per store, by week, segments, importance, SHAP, counterfactual)")
        model_names = [r.name for r in results if r.name != "Per-store SARIMAX (ARMA errors)"]
        out["per_store"] = pd.concat([MET.score_by(ho, f"pred::{m}", "Store").assign(model=m)
                                      for m in model_names])
        out["by_week"] = pd.concat([MET.score_by(ho, f"pred::{m}", "week").assign(model=m)
                                    for m in model_names])
        out["segments"] = pd.concat([segments(ho, f"pred::{m}").assign(model=m) for m in model_names])
        out["importance"] = M.importance(booster)
        feats = glob.info["features"]
        hx = p_train.loc[ho.index]
        out["shap_groups"] = E.shap_groups(booster, hx.loc[hx["Open"] == 1, feats], F.FEATURE_GROUPS)
        out["promo_cf"] = E.promo_counterfactual(booster, hx, feats)

    if ablation:
        variants = {"Full model": F.GLOBAL_FEATURES,
                    "Without promo features": [f for f in F.GLOBAL_FEATURES
                                               if f not in F.PROMO + ["store_promo_lift"]],
                    "Without holiday features": [f for f in F.GLOBAL_FEATURES if f not in F.HOLIDAY]}
        abl = []
        for v, feats in variants.items():
            if v == "Full model":
                col = "pred::Global LightGBM"
            else:
                r, _ = M.fit_global(p_fit, p_train, split, feats=feats, name=v)
                col = f"pred::{v}"
                ho[col] = r.pred.reindex(ho.index)
                log(f"{fid}: ablation '{v}' done ({r.train_seconds:.0f}s)")
            seg = segments(ho, col)
            abl.append(seg[seg.segment.isin(["All open days", "Promo days", "Non-promo days",
                                             "+/-3 days around a holiday"])].assign(variant=v, fold=fid))
        out["ablation"] = pd.concat(abl)
    del p_fit, p_train, booster
    gc.collect()
    return out


def run_effects(panel: pd.DataFrame) -> dict:
    log("effects: descriptive + fixed-effects regressions")
    known = panel[panel["is_test"] == 0]
    gc.collect()
    out = {"descriptive": E.descriptive(known)}
    tab, full, res = E.fe_ols(known)
    out["fe_main"], out["fe_main_full"] = tab, full
    out["fe_meta"] = dict(tab.attrs)
    dec = {}
    for tgt in ["Sales", "Customers", "SalesPerCustomer"]:
        t = tab if tgt == "Sales" else E.fe_ols(known, tgt)[0]
        dec[tgt] = t.loc["Promo", ["effect_pct", "ci_low_pct", "ci_high_pct"]].to_dict()
        if tgt != "Sales":
            out[f"fe_{tgt}"] = t
    out["promo_decomposition"] = dec
    _, f_dow, _ = E.fe_ols(known, interactions="weekday")
    out["promo_by_weekday"] = E.interaction_effects(f_dow, "Promo", "Promo x weekday ", "1")
    _, f_typ, _ = E.fe_ols(known, interactions="storetype")
    out["promo_by_storetype"] = E.interaction_effects(f_typ, "Promo", "Promo x type ", "a")
    _, f_ass, _ = E.fe_ols(known, interactions="assortment")
    out["promo_by_assortment"] = E.interaction_effects(f_ass, "Promo", "Promo x assortment ", "a")
    return out


def make_submission(panel: pd.DataFrame, n_trees: int, raw_dir: Path) -> pd.DataFrame:
    log(f"submission: global LightGBM on all data, {n_trees} trees")
    import lightgbm as lgb
    known = panel["is_test"] == 0
    p_all = F.add_store_stats(panel, known)
    rows = M.trainable(p_all, known)
    feats = F.GLOBAL_FEATURES
    cats = [c for c in F.CATEGORICAL if c in feats]
    b = lgb.train(M.GLOBAL_PARAMS, lgb.Dataset(p_all.loc[rows, feats], np.log1p(p_all.loc[rows, "Sales"]),
                                               categorical_feature=cats), n_trees)
    te = p_all[p_all["is_test"] == 1]
    pred = np.where(te["Open"] == 1, np.expm1(b.predict(te[feats])), 0)
    sub = pd.DataFrame({"Id": te["Id"].astype(int).to_numpy(), "Sales": pred}).sort_values("Id")
    sample = pd.read_csv(data._find(raw_dir, "sample_submission.csv"))
    sub = sample[["Id"]].merge(sub, on="Id", how="left")
    assert sub["Sales"].notna().all() and len(sub) == len(sample)
    return sub


# -----------------------------------------------------------------------------
def save_all(res: dict, eff: dict | None, folds_out: list[dict]):
    met = pd.concat([f["metrics"] for f in folds_out], ignore_index=True)
    met.to_csv(config.MET_DIR / "backtest_metrics.csv", index=False)
    pd.concat([f["pred"] for f in folds_out]).to_parquet(config.PRED_DIR / "backtest_predictions.parquet",
                                                        index=False)
    f1 = folds_out[0]
    for k in ["per_store", "by_week", "segments", "importance", "shap_groups", "sample_metrics"]:
        if k in f1:
            f1[k].to_csv(config.MET_DIR / f"{k}.csv", index=False)
    if "promo_cf" in f1:
        cf = f1["promo_cf"]
        cf["by_storetype"].to_csv(config.MET_DIR / "promo_counterfactual_by_storetype.csv")
        cf["by_weekday"].to_csv(config.MET_DIR / "promo_counterfactual_by_weekday.csv")
    for f in folds_out:
        if "ablation" in f:
            f["ablation"].to_csv(config.MET_DIR / "ablation.csv", index=False)
    if eff:
        for k, v in eff["descriptive"].items():
            v.to_csv(config.MET_DIR / f"desc_{k}.csv")
        eff["fe_main"].to_csv(config.MET_DIR / "fe_effects_sales.csv")
        eff["fe_main_full"].to_csv(config.MET_DIR / "fe_effects_sales_all_coefficients.csv")
        for k in ["fe_Customers", "fe_SalesPerCustomer", "promo_by_weekday", "promo_by_storetype",
                  "promo_by_assortment"]:
            eff[k].to_csv(config.MET_DIR / f"{k}.csv")
    (config.OUT_DIR / "summary.json").write_text(json.dumps(res, indent=2, default=float))


def make_figures(panel: pd.DataFrame, clean: pd.DataFrame, folds_out: list[dict], eff: dict | None,
                 example_stores: list[int]):
    log("figures")
    fd = config.FIG_DIR
    met = pd.concat([f["metrics"] for f in folds_out], ignore_index=True)
    f1 = folds_out[0]
    daily = clean.groupby("Date").agg(Sales=("Sales", "sum"), n_open=("Open", "sum"))
    spans = [(M.Split(pd.Timestamp(s)).holdout_start, M.Split(pd.Timestamp(s)).holdout_end, fid)
             for fid, s, _ in FOLDS[:len(folds_out)]]
    P.total_sales_overview(daily, spans, fd / "fig01_sales_overview.png")
    P.model_comparison(met, fd / "fig02_model_comparison.png")
    if len(folds_out) > 1:
        P.backtest_folds(met, fd / "fig03_backtest_folds.png")
    _, win = P.per_store_scatter(f1["per_store"], fd / "fig04_per_store_global_vs_local.png")
    P.error_by_week(f1["by_week"], fd / "fig05_error_by_week.png")
    P.holdout_total(f1["pred"], fd / "fig06_holdout_total.png")
    P.store_examples(f1["pred"], example_stores, fd / "fig07_store_examples.png")
    P.feature_groups(f1["importance"], fd / "fig08_feature_importance.png")
    P.interactive_holdout(f1["pred"], fd / "interactive_holdout_forecast.html", example_stores)
    for f in folds_out:
        if "ablation" in f:
            P.ablation(f["ablation"], fd / "fig09_ablation.png")
    if eff:
        tab = eff["fe_main"]
        main_rows = ["Promo", "Reopening (first week after closure >= 5 days)", "1 day(s) after holiday",
                     "1 day(s) before holiday", "2 day(s) before holiday", "2 day(s) after holiday",
                     "3 day(s) after holiday", "SchoolHoliday", "Promo2 month",
                     "Open on state holiday", "Last week before closure >= 5 days"]
        labels = {"SchoolHoliday": "School holiday", "Promo": "Promo day",
                  "Promo2 month": "Promo2 (coupon) month", "Open on state holiday": "Open on a state holiday",
                  "Reopening (first week after closure >= 5 days)": "First week after reopening (closure >= 5 d)",
                  "Last week before closure >= 5 days": "Last week before closure (>= 5 d)"}
        P.coef_plot(tab, fd / "fig10_promo_holiday_effects.png", "Promo and holiday effects on daily sales",
                    "Store fixed effects + weekday, month, year, day-of-month controls; 844k open store-days",
                    rows=[r for r in main_rows if r in tab.index], labels=labels)
        P.christmas_event(tab, fd / "fig11_christmas.png")
        P.promo_heterogeneity(eff["promo_by_weekday"], eff["promo_by_storetype"],
                              fd / "fig12_promo_heterogeneity.png")
        P.promo_decomposition(eff["promo_decomposition"], fd / "fig13_promo_decomposition.png")
        P.interactive_effects(tab, fd / "interactive_effects.html")
    return win


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=config.RAW_DIR, help="folder with the Kaggle CSV files")
    ap.add_argument("--quick", action="store_true", help="smoke test: 60 stores, fold F1 only")
    ap.add_argument("--folds", type=int, default=3, help="number of backtest windows (1-3)")
    ap.add_argument("--sarimax-stores", type=int, default=150, help="sample size for per-store SARIMAX")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--rebuild", action="store_true", help="rebuild clean.parquet / panel.parquet")
    ap.add_argument("--rerun", action="store_true", help="ignore cached fold / effect results in outputs/cache")
    ap.add_argument("--no-effects", action="store_true")
    ap.add_argument("--no-submission", action="store_true")
    a = ap.parse_args(argv)

    t0 = time.perf_counter()
    if a.quick:  # smoke test writes to outputs/quick/ so the full results stay untouched
        config.OUT_DIR = config.OUT_DIR / "quick"
        config.FIG_DIR, config.MET_DIR, config.PRED_DIR = (config.OUT_DIR / "figures", config.OUT_DIR / "metrics",
                                                          config.OUT_DIR / "predictions")
        config.ensure_dirs()
    stores = list(range(1, 61)) if a.quick else None
    nfold = 1 if a.quick else max(1, min(3, a.folds))
    clean, test, store, panel = prepare(a.raw_dir, a.rebuild, stores)
    log(f"panel: {len(panel):,} rows, {panel['Store'].nunique()} stores")
    gap_stores = set(clean.loc[clean["has_gap"], "Store"].unique())

    rng = np.random.default_rng(config.SEED)
    ho_stores = np.sort(panel.loc[(panel.Date >= config.HOLDOUT_START) & (panel.is_test == 0), "Store"].unique())
    sar = sorted(rng.choice(ho_stores, size=min(a.sarimax_stores, len(ho_stores), 10 if a.quick else 10**6),
                            replace=False).tolist()) if a.sarimax_stores else None

    cache = config.OUT_DIR / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    tag = "quick" if a.quick else "full"
    folds_out = []
    for i, fold in enumerate(FOLDS[:nfold]):
        cf = cache / f"{tag}_fold_{fold[0]}.joblib"
        if cf.exists() and not a.rerun:
            log(f"{fold[0]}: loaded from cache ({cf.name})")
            folds_out.append(joblib.load(cf))
            continue
        res_fold = run_fold(panel, fold, gap_stores, a.n_jobs, sarimax_stores=sar if i == 0 else None,
                            extras=(i == 0), ablation=(fold[0] == "F3" or (a.quick and i == 0)))
        joblib.dump(res_fold, cf)
        folds_out.append(res_fold)
        gc.collect()

    eff = None
    if not a.no_effects:
        ce = cache / f"{tag}_effects.joblib"
        if ce.exists() and not a.rerun:
            log("effects: loaded from cache")
            eff = joblib.load(ce)
        else:
            eff = run_effects(panel)
            joblib.dump(eff, ce)
    f1 = folds_out[0]
    res = {
        "data": data.summary(clean),
        "holdout": {"start": str(config.HOLDOUT_START.date()), "end": str(config.LAST_TRAIN_DATE.date()),
                    "days": config.HOLDOUT_DAYS},
        "metrics_F1": f1["metrics"].set_index("model")[["RMSPE", "sMAPE", "MAE", "Bias_%", "train_seconds",
                                                         "n_models"]].to_dict("index"),
        "fold_mean_RMSPE": pd.concat([f["metrics"] for f in folds_out]).groupby("model")["RMSPE"].mean().to_dict(),
        "global_info": f1["info"]["global"], "per_store_lgb_info": f1["info"]["per_store_lgb"],
        "global_corrected_holdout": f1["info"]["global_corrected_holdout"],
        "sarimax_sample_stores": sar,
    }
    if "promo_cf" in f1:
        res["promo_counterfactual"] = {k: v for k, v in f1["promo_cf"].items() if not isinstance(v, pd.DataFrame)}
    if eff:
        res["effects"] = {"meta": eff["fe_meta"], "promo_decomposition": eff["promo_decomposition"],
                          "main": eff["fe_main"][["effect_pct", "ci_low_pct", "ci_high_pct", "p"]].to_dict("index")}

    ex = [s for s in [551, 613, 262] if s in set(panel.Store)][:3] or list(panel.Store.unique()[:3])
    res["per_store_global_win_share"] = make_figures(panel, clean, folds_out, eff, ex)

    if not a.no_submission and not a.quick:
        n_trees = int(round(f1["info"]["global"]["best_iter"] * 1.1))
        sub = make_submission(panel, n_trees, a.raw_dir)
        sub.to_csv(config.OUT_DIR / "submission.csv", index=False)
        res["submission"] = {"rows": len(sub), "n_trees": n_trees, "mean_sales_open": float(sub.loc[sub.Sales > 0, "Sales"].mean())}
    res["runtime_minutes"] = (time.perf_counter() - t0) / 60
    save_all(res, eff, folds_out)
    log(f"done in {res['runtime_minutes']:.1f} min")
    return res


if __name__ == "__main__":
    main()
