"""P5 — Train and compare B0 baseline, Ridge and LightGBM with spatial and random CV.

Target: y = log(price_idr). Metrics after exp() back to Rupiah (R² on log scale).
Spatial CV (primary): GroupKFold by kecamatan_code. Random CV (comparison): KFold.
The same folds are used for every model.

Inputs:  data/processed/model_table.csv, data/external/osm_kecamatan_features.csv,
         data/interim/listings_clean.csv (descriptions for the worst-predictions report)
Outputs: models/ridge.joblib, models/lgbm_point.txt, models/feature_columns.json, models/metadata.json,
         reports/metrics.csv, reports/metrics.md, reports/worst_predictions.md,
         reports/figures/07_predicted_vs_actual.png, reports/figures/08_mape_by_kabupaten.png

Run: python -m src.train
Then re-run python -m src.quantile: this script rewrites metadata.json and metrics.md
from scratch, which drops the quantile (price range) results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.eda import C, _caption, _idr_ticks, _save, _style
from src.evaluate import ape, mape_by, price_band, regression_metrics, summarize_folds, worst_predictions
from src.features import NOT_FEATURES, feature_columns
from src.utils import format_idr, get_md_section, replace_md_section

CVS = ("spatial", "random")


@dataclass
class ModelSpec:
    """One model configuration."""

    name: str
    family: str
    params: dict = field(default_factory=dict)


def model_specs() -> list[ModelSpec]:
    """B0, Ridge for each alpha, LightGBM for each grid point."""
    specs = [ModelSpec("B0 baseline", "baseline")]
    specs += [ModelSpec(f"Ridge alpha={a:g}", "ridge", {"alpha": a}) for a in config.RIDGE_ALPHAS]
    for nl in config.LGBM_GRID["num_leaves"]:
        for mcs in config.LGBM_GRID["min_child_samples"]:
            specs.append(ModelSpec(f"LightGBM leaves={nl} min_child={mcs}", "lgbm",
                                   {"num_leaves": nl, "min_child_samples": mcs}))
    return specs


def baseline_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """B0: median price per land m² from the TRAINING rows only × land_m2.

    Lookup order per test row: its kecamatan (by kecamatan_code) → its kabupaten → overall.
    Under spatial CV the kecamatan is never in the training fold, so kabupaten is used.
    """
    by_kec = train.groupby(config.CV_GROUP_COLUMN)["price_per_land_m2"].median()
    by_kab = train.groupby("kabupaten")["price_per_land_m2"].median()
    overall = train["price_per_land_m2"].median()
    ppl = test[config.CV_GROUP_COLUMN].map(by_kec)
    ppl = ppl.fillna(test["kabupaten"].map(by_kab)).fillna(overall)
    return (ppl * test["land_m2"]).to_numpy()


def ridge_pipeline(alpha: float) -> Pipeline:
    """Median imputer → StandardScaler → Ridge (fitted per fold, so no leakage)."""
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("ridge", Ridge(alpha=alpha, random_state=config.RANDOM_STATE))])


def lgbm_model(params: dict) -> lgb.LGBMRegressor:
    """LightGBM point model with the fixed base settings plus grid params."""
    return lgb.LGBMRegressor(**config.LGBM_BASE_PARAMS, **params)


def fit_predict(spec: ModelSpec, train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Fit on train rows and return predicted price (Rp) for test rows."""
    if spec.family == "baseline":
        return baseline_predict(train, test)
    model = ridge_pipeline(spec.params["alpha"]) if spec.family == "ridge" else lgbm_model(spec.params)
    model.fit(train[features], train["log_price"])
    return np.exp(model.predict(test[features]))


def make_folds(table: pd.DataFrame) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Spatial (GroupKFold by kecamatan_code) and random (KFold) folds, built once."""
    groups = table[config.CV_GROUP_COLUMN]
    spatial = list(GroupKFold(n_splits=config.N_SPLITS).split(table, groups=groups))
    random = list(KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE).split(table))
    for tr, te in spatial:
        assert not set(groups.iloc[tr]) & set(groups.iloc[te]), "kecamatan in both train and test fold"
    return {"spatial": spatial, "random": random}


def cross_validate(spec: ModelSpec, table: pd.DataFrame, features: list[str],
                   folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[list[dict], np.ndarray]:
    """Per-fold metrics and out-of-fold predictions (Rp)."""
    oof = np.full(len(table), np.nan)
    per_fold = []
    for tr, te in folds:
        pred = fit_predict(spec, table.iloc[tr], table.iloc[te], features)
        oof[te] = pred
        per_fold.append(regression_metrics(table["price_idr"].iloc[te], pred))
    return per_fold, oof


def check_no_leakage(features: list[str]) -> None:
    """Fail if any price-derived or ID column is a model input."""
    bad = [f for f in features if "price" in f.lower() or f in NOT_FEATURES]
    if bad:
        raise AssertionError(f"leakage: forbidden feature columns {bad}")


def fig_pred_vs_actual(oof: pd.DataFrame, model_name: str) -> str:
    """07 — out-of-fold predicted vs actual (log-log), spatial CV."""
    fig, ax = plt.subplots(figsize=(6.4, 6))
    lo = min(oof["price_idr"].min(), oof["pred_idr"].min()) * 0.8
    hi = max(oof["price_idr"].max(), oof["pred_idr"].max()) * 1.25
    ax.plot([lo, hi], [lo, hi], color=C["text_secondary"], linewidth=1, zorder=1)
    ax.scatter(oof["price_idr"], oof["pred_idr"], s=10, color=C["series_1"], alpha=0.6,
               edgecolors=C["surface"], linewidths=0.3, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [2e8, 5e8, 1e9, 2e9, 5e9, 1e10]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(_idr_ticks))
        axis.set_minor_formatter(NullFormatter())
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.annotate("perfect prediction", (hi / 1.6, hi / 1.35), fontsize=8, color=C["text_secondary"],
                ha="right", rotation=45, rotation_mode="anchor")
    ax.set_title(f"Predicted vs actual price — {model_name}")
    ax.set_xlabel("Actual asking price (Rp, log scale)")
    ax.set_ylabel("Predicted price (Rp, log scale)")
    _caption(fig, "Out-of-fold predictions under spatial CV: each kecamatan predicted by a model "
                  "that never saw it.")
    return _save(fig, "07_predicted_vs_actual.png")


def fig_mape_by_kabupaten(by_kab: pd.DataFrame, model_name: str, overall: float) -> str:
    """08 — MAPE by kabupaten (spatial CV, out-of-fold)."""
    d = by_kab.sort_values("mape")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh([f"{r.kabupaten}  (n={r.n})" for r in d.itertuples()], d["mape"], color=C["series_1"], height=0.6)
    for i, v in enumerate(d["mape"]):
        ax.annotate(f"{v:.0f}%", (v, i), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=9, color=C["text_secondary"],
                    bbox={"facecolor": C["surface"], "edgecolor": "none", "pad": 1.5})
    ax.axvline(overall, color=C["text_secondary"], linewidth=0.8, zorder=0)
    ax.set_xlim(0, d["mape"].max() * 1.2)
    ax.set_title(f"Average error by kabupaten — {model_name}")
    ax.set_xlabel("Mean absolute percentage error, spatial CV (%)")
    ax.grid(axis="y", visible=False)
    _caption(fig, f"Lower is better. Vertical line: all listings ({overall:.0f}%). "
                  "Small kabupaten (few listings) give noisy estimates.")
    return _save(fig, "08_mape_by_kabupaten.png")


def _md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def _pm(res: dict, cv: str, m: str, fmt: str = "{:.1f}") -> str:
    return f"{fmt.format(res[cv][m + '_mean'])} ± {fmt.format(res[cv][m + '_std'])}"


def write_metrics_md(results: dict[str, dict], chosen: dict[str, str], best: str, by_kab: pd.DataFrame,
                     by_band: pd.DataFrame, n_rows: int, n_groups: int) -> None:
    """reports/metrics.md: comparison table, grids, error analysis, interpretation."""
    def row(name: str, label: str) -> dict:
        r = results[name]
        return {"Model": label, "Spatial MAPE": _pm(r, "spatial", "mape") + "%",
                "Spatial MdAPE": _pm(r, "spatial", "mdape") + "%",
                "Spatial MAE (Rp)": f"{format_idr(int(r['spatial']['mae_mean']))} ± "
                                    f"{format_idr(int(r['spatial']['mae_std']))}",
                "Spatial R² (log)": _pm(r, "spatial", "r2_log", "{:.2f}"),
                "Random MAPE": _pm(r, "random", "mape") + "%"}

    main = pd.DataFrame([row(chosen["baseline"], "B0 baseline (median price/m² × land)"),
                         row(chosen["ridge"], f"M1 {chosen['ridge']}"),
                         row(chosen["lgbm"], f"M2 {chosen['lgbm']}")])

    def grid(family: str) -> pd.DataFrame:
        return pd.DataFrame([{"Config": n, "Spatial MAPE": _pm(r, "spatial", "mape") + "%",
                              "Spatial MdAPE": _pm(r, "spatial", "mdape") + "%",
                              "Random MAPE": _pm(r, "random", "mape") + "%"}
                             for n, r in results.items() if r["family"] == family])

    b, lg = results[chosen["baseline"]], results[chosen["lgbm"]]
    s_b, s_l = b["spatial"]["mape_mean"], lg["spatial"]["mape_mean"]
    r_l = lg["random"]["mape_mean"]
    beats = s_l < s_b
    gap = s_l - r_l
    small = gap < config.SMALL_CV_GAP_POINTS
    interp = (
        f"Under spatial CV the model is always tested on kecamatan it has never seen — the honest setting for "
        f"an app that must price houses in thinly covered areas. LightGBM's error rises from {r_l:.1f}% "
        f"(random CV) to {s_l:.1f}% (spatial CV), a gap of {gap:.1f} points. "
        + (f"The gap is small. Two likely reasons: cleaning removed reposts of the same house, so near-identical "
           f"listings rarely sit on both sides of a split; and the location features describe a kecamatan "
           f"(distances, amenity density) rather than identify it, so they carry over to unseen kecamatan. "
           f"Random CV is still the optimistic number, because neighbors from the same kecamatan help it. "
           if small else
           f"Random CV is optimistic: houses from the same kecamatan — often the same development — land in "
           f"both training and test folds, so the model partly memorizes local price levels. ")
        + (f"LightGBM beats the baseline under spatial CV ({s_l:.1f}% vs {s_b:.1f}% MAPE), so house size, "
           f"rooms and location features add real information beyond 'median price per m² in the kabupaten'. "
           if beats else
           f"LightGBM does **not** beat the baseline under spatial CV ({s_l:.1f}% vs {s_b:.1f}% MAPE). ")
        + "MAPE is pulled up by a few large misses (several are data errors — see worst predictions); the "
          "median error (MdAPE) is the better guide to a typical listing."
    )

    fmt_tbl = lambda d, key: d.assign(mape=d["mape"].round(1), mdape=d["mdape"].round(1)).rename(
        columns={key: key.replace("_", " "), "mape": "MAPE %", "mdape": "MdAPE %"})
    md = (
        "# Model metrics\n\n"
        f"Generated by `python -m src.train`. {n_rows:,} listings, {n_groups} kecamatan groups, "
        f"{config.N_SPLITS}-fold CV, seed {config.RANDOM_STATE}. Target `log(price_idr)`; metrics on the Rupiah "
        "scale (R² on log scale). Values are mean ± standard deviation across folds.\n\n"
        "## Comparison\n\n" + _md(main) + "\n\n"
        "**Spatial CV** = GroupKFold by kecamatan (`kecamatan_code`): test kecamatan are unseen in training. "
        "**Random CV** = shuffled KFold. The same folds are used for every model.\n\n"
        "## Spatial vs random CV\n\n" + interp + "\n\n"
        f"## Error analysis — {best} (spatial CV, out-of-fold)\n\n"
        "### By kabupaten\n\n" + _md(fmt_tbl(by_kab, "kabupaten")) + "\n\n"
        "### By price band\n\n" + _md(fmt_tbl(by_band, "price_band")) + "\n\n"
        "Worst 10 predictions: `reports/worst_predictions.md`. Figures: `07_predicted_vs_actual.png`, "
        "`08_mape_by_kabupaten.png`.\n\n"
        "## Known limitations\n\n"
        "- **Point estimate is a typical price, not an average.** The models predict `log(price)` and the app "
        "converts back with `exp()`, with no smearing correction. That gives roughly the median price for a "
        "house like this one, which sits below the mean when errors are skewed. It is one reason the most "
        "expensive houses (> 5 M) are under-predicted. P6's quantile models report the spread around it.\n"
        "- **Few listings outside Sleman.** Gunungkidul and Kulon Progo have very few listings, so their "
        "error estimates above are noisy.\n"
        "- **Asking prices, not sale prices,** from one site in March–May 2024.\n\n"
        "## Ridge alphas\n\n" + _md(grid("ridge")) + "\n\n"
        "## LightGBM grid\n\n" + _md(grid("lgbm")) + "\n\n"
        f"Chosen by spatial MAPE: **{chosen['ridge']}**, **{chosen['lgbm']}**.\n"
    )
    if config.METRICS_MD.exists():
        old_text = config.METRICS_MD.read_text(encoding="utf-8")
        for heading in config.METRICS_PRESERVED_SECTIONS:
            kept = get_md_section(old_text, heading)
            if kept:
                md = replace_md_section(md, heading, kept)
    config.METRICS_MD.write_text(md, encoding="utf-8")


def write_worst_md(worst: pd.DataFrame, best: str) -> None:
    """reports/worst_predictions.md with description snippets."""
    rows = []
    for r in worst.itertuples():
        desc = str(r.description).replace("\n", " ").replace("|", "/")
        rows.append({"listing_id": r.listing_id, "kecamatan": f"{r.kecamatan} ({r.kabupaten})",
                     "actual": format_idr(int(r.price_idr)), "predicted": format_idr(int(r.pred_idr)),
                     "error": f"{r.ape:.0f}%", "land / building m²": f"{r.land_m2:g} / {r.building_m2:g}",
                     "description": desc[: config.WORST_DESC_CHARS]})
    md = (f"# Worst {len(rows)} predictions — {best}, spatial CV (out-of-fold)\n\n"
          "Error = |predicted − actual| / actual.\n\n" + _md(pd.DataFrame(rows)) + "\n")
    config.WORST_PREDICTIONS_MD.write_text(md, encoding="utf-8")


def main() -> None:
    """Cross-validate all models, analyze errors, fit final models, write outputs."""
    _style()
    table = pd.read_csv(config.MODEL_TABLE, dtype={config.CV_GROUP_COLUMN: str})
    osm = pd.read_csv(config.OSM_FEATURES_CSV)
    features = feature_columns(osm)
    check_no_leakage(features)
    folds = make_folds(table)
    n_groups = table[config.CV_GROUP_COLUMN].nunique()
    print(f"{len(table):,} rows, {len(features)} features, {n_groups} kecamatan groups, {config.N_SPLITS} folds")

    results: dict[str, dict] = {}
    oofs: dict[str, np.ndarray] = {}
    rows = []
    for spec in model_specs():
        res = {"family": spec.family, "params": spec.params}
        for cv in CVS:
            per_fold, oof = cross_validate(spec, table, features, folds[cv])
            res[cv] = summarize_folds(per_fold)
            if cv == "spatial":
                oofs[spec.name] = oof
            rows.append({"model": spec.name, "family": spec.family, "cv": cv, **res[cv]})
        results[spec.name] = res
        print(f"  {spec.name:<36} spatial MAPE {res['spatial']['mape_mean']:5.1f}%  "
              f"MdAPE {res['spatial']['mdape_mean']:5.1f}%   random MAPE {res['random']['mape_mean']:5.1f}%")

    pd.DataFrame(rows).to_csv(config.METRICS_CSV, index=False, encoding="utf-8")

    def best_of(family: str) -> str:
        cands = {n: r for n, r in results.items() if r["family"] == family}
        return min(cands, key=lambda n: cands[n]["spatial"]["mape_mean"])

    chosen = {f: best_of(f) for f in ("baseline", "ridge", "lgbm")}
    best = min(chosen.values(), key=lambda n: results[n]["spatial"]["mape_mean"])
    lgbm_beats = results[chosen["lgbm"]]["spatial"]["mape_mean"] < results[chosen["baseline"]]["spatial"]["mape_mean"]

    clean = pd.read_csv(config.CLEAN_CSV, usecols=["listing_id", "description"])
    oof = table[["listing_id", "kecamatan", "kabupaten", "price_idr", "land_m2", "building_m2"]].copy()
    oof["pred_idr"] = oofs[best]
    oof["ape"] = ape(oof["price_idr"], oof["pred_idr"])
    oof["price_band"] = price_band(oof["price_idr"])
    oof = oof.merge(clean, on="listing_id", how="left")
    by_kab = mape_by(oof, "kabupaten")
    by_band = mape_by(oof, "price_band")
    figs = [fig_pred_vs_actual(oof, best), fig_mape_by_kabupaten(by_kab, best, oof["ape"].mean())]
    write_worst_md(worst_predictions(oof, config.N_WORST_PREDICTIONS), best)
    write_metrics_md(results, chosen, best, by_kab, by_band, len(table), n_groups)

    print(f"\nChosen: {chosen['ridge']} | {chosen['lgbm']} | best overall (spatial MAPE): {best}")
    if not lgbm_beats:
        print("STOP: LightGBM does not beat the baseline under spatial CV — final models NOT fitted. "
              "Report and decide before tuning further.")
        return

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X, y = table[features], table["log_price"]
    ridge = ridge_pipeline(results[chosen["ridge"]]["params"]["alpha"]).fit(X, y)
    joblib.dump(ridge, config.RIDGE_MODEL)
    lgbm = lgbm_model(results[chosen["lgbm"]]["params"]).fit(X, y)
    lgbm.booster_.save_model(str(config.LGBM_POINT_MODEL))
    config.FEATURE_COLUMNS_JSON.write_text(json.dumps(features, indent=2), encoding="utf-8")

    ranges = {c: {"min": float(table[c].min()), "max": float(table[c].max())} for c in config.RANGE_CHECK_COLUMNS}
    meta = {
        "trained_on": date.today().isoformat(),
        "data_source": "Kaggle pramudyadika/yogyakarta-housing-price-ndonesia (Rumah123)",
        "data_date_range": list(config.DATA_DATE_RANGE),
        "n_rows": len(table),
        "n_kecamatan": int(n_groups),
        "target": "log(price_idr)",
        "random_state": config.RANDOM_STATE,
        "cv": {"n_splits": config.N_SPLITS, "spatial": f"GroupKFold by {config.CV_GROUP_COLUMN}",
               "random": "KFold shuffle"},
        "chosen": {"ridge": {"name": chosen["ridge"], **results[chosen["ridge"]]["params"]},
                   "lgbm": {"name": chosen["lgbm"], **config.LGBM_BASE_PARAMS,
                            **results[chosen["lgbm"]]["params"]},
                   "best_overall": best},
        "cv_metrics": {n: {cv: r[cv] for cv in CVS} for n, r in results.items()},
        "training_ranges": ranges,
        "active_flags": list(config.ACTIVE_FLAGS),
        "kabupaten_order": list(config.KABUPATEN_ORDER),
        "feature_columns_file": config.FEATURE_COLUMNS_JSON.name,
    }
    config.METADATA_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved {config.RIDGE_MODEL.name}, {config.LGBM_POINT_MODEL.name}, "
          f"{config.FEATURE_COLUMNS_JSON.name}, {config.METADATA_JSON.name}")
    print(f"Wrote {config.METRICS_MD.relative_to(config.ROOT)}, {config.METRICS_CSV.relative_to(config.ROOT)}, "
          f"{config.WORST_PREDICTIONS_MD.relative_to(config.ROOT)}, figures {', '.join(figs)}")
    print("\n" + config.METRICS_MD.read_text(encoding="utf-8").split("## Spatial vs random CV")[0]
          .split("## Comparison")[1].strip())


if __name__ == "__main__":
    main()
