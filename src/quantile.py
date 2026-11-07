"""P6a — Quantile models for the likely price range (10th, 50th, 90th percentile).

LightGBM with objective="quantile" at each alpha in config.QUANTILES, using the
tree settings chosen for the point model in P5 (models/metadata.json). Evaluated
under the same spatial CV as P5: coverage = share of true prices inside [q10, q90],
width = (q90 − q10) / q50. Crossing quantiles are fixed by sorting per row.
config.INTERVAL_CALIBRATION_FACTOR (default 1.0) widens the log interval around q50;
this script also prints the factor that would reach config.CALIBRATION_GOAL.
P7b U2: per-kabupaten scales (same widening, one scale per kabupaten with enough rows,
the global factor for the rest) are stored in metadata.json "range_scales" and used
by predict_house.

Inputs:  data/processed/model_table.csv, data/external/osm_kecamatan_features.csv,
         models/metadata.json
Outputs: models/lgbm_q10.txt, models/lgbm_q50.txt, models/lgbm_q90.txt,
         metadata.json ("quantile" and "range_scales" blocks), reports/metrics.md ("Price range" section)

Run: python -m src.quantile
Order: always after `python -m src.train` (train rewrites metadata.json and metrics.md,
which removes the quantile block); then src.explain and src.comps.
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src.evaluate import sort_quantiles, widen_log_interval
from src.features import feature_columns
from src.train import CVS, _md, check_no_leakage, make_folds
from src.utils import replace_md_section

QUANTILE_MODEL_PATHS = {0.1: config.LGBM_Q10_MODEL, 0.5: config.LGBM_Q50_MODEL, 0.9: config.LGBM_Q90_MODEL}
METRICS_SECTION = "## Price range (quantile models)"


def chosen_tree_params() -> dict:
    """num_leaves / min_child_samples of the P5 point model."""
    meta = json.loads(config.METADATA_JSON.read_text(encoding="utf-8"))
    chosen = meta["chosen"]["lgbm"]
    return {k: chosen[k] for k in config.LGBM_GRID}


def quantile_model(alpha: float, tree_params: dict) -> lgb.LGBMRegressor:
    """LightGBM quantile regressor: P5 base settings + chosen tree params."""
    params = {**config.LGBM_BASE_PARAMS, **tree_params, "objective": "quantile", "alpha": alpha}
    return lgb.LGBMRegressor(**params)


def interval_metrics(y_idr: np.ndarray, q_idr: np.ndarray, point_idr: np.ndarray | None = None) -> dict[str, float]:
    """Coverage (%) of [q10, q90], share below/above, width as % of q50, point-outside share (%)."""
    y = np.asarray(y_idr, dtype=float)
    lo, mid, hi = q_idr[:, 0], q_idr[:, 1], q_idr[:, 2]
    width = (hi - lo) / mid * 100
    out = {
        "coverage": float(np.mean((y >= lo) & (y <= hi)) * 100),
        "below_low": float(np.mean(y < lo) * 100),
        "above_high": float(np.mean(y > hi) * 100),
        "width_mean": float(np.mean(width)),
        "width_median": float(np.median(width)),
    }
    if point_idr is not None:
        p = np.asarray(point_idr, dtype=float)
        out["point_outside"] = float(np.mean((p < lo) | (p > hi)) * 100)
    return out


def coverage_at(y_log: np.ndarray, q_log: np.ndarray, factor: float) -> float:
    """Share (0–1) of true log prices inside the widened interval."""
    q = widen_log_interval(q_log, factor)
    return float(np.mean((y_log >= q[:, 0]) & (y_log <= q[:, 2])))


def suggest_factor(y_log: np.ndarray, q_log: np.ndarray) -> float:
    """Smallest factor on the config.CALIBRATION_SEARCH grid reaching config.CALIBRATION_GOAL."""
    start, stop, step = config.CALIBRATION_SEARCH
    for f in np.arange(start, stop + step / 2, step):
        if coverage_at(y_log, q_log, f) >= config.CALIBRATION_GOAL:
            return round(float(f), 2)
    return float(stop)


def cross_validate_quantiles(table: pd.DataFrame, features: list[str], folds: list, tree_params: dict,
                             factor: float) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Per-fold interval metrics, out-of-fold sorted log quantiles (raw, before widening), point (Rp)."""
    oof_log = np.full((len(table), len(config.QUANTILES)), np.nan)
    oof_point = np.full(len(table), np.nan)
    per_fold = []
    for tr, te in folds:
        X_tr, y_tr, X_te = table[features].iloc[tr], table["log_price"].iloc[tr], table[features].iloc[te]
        q_log = np.column_stack([quantile_model(a, tree_params).fit(X_tr, y_tr).predict(X_te)
                                 for a in config.QUANTILES])
        n_crossed = int(np.sum(np.any(np.diff(q_log, axis=1) < 0, axis=1)))
        q_log = sort_quantiles(q_log)
        point = np.exp(lgb.LGBMRegressor(**config.LGBM_BASE_PARAMS, **tree_params).fit(X_tr, y_tr).predict(X_te))
        oof_log[te], oof_point[te] = q_log, point
        m = interval_metrics(table["price_idr"].iloc[te].to_numpy(), np.exp(widen_log_interval(q_log, factor)),
                             point)
        m["crossed_rows"] = n_crossed
        per_fold.append(m)
    return per_fold, oof_log, oof_point


def kabupaten_scale(y_log: np.ndarray, q_log: np.ndarray) -> float:
    """U2: smallest scale within config.RANGE_SCALE_CAP reaching config.TARGET_COVERAGE (upper cap if none)."""
    lo, hi = config.RANGE_SCALE_CAP
    step = config.CALIBRATION_SEARCH[2]
    for s in np.arange(lo, hi + step / 2, step):
        if coverage_at(y_log, q_log, s) >= config.TARGET_COVERAGE:
            return round(float(s), 2)
    return float(hi)


def fit_range_scales(y_log: np.ndarray, q_log: np.ndarray, kabupaten: np.ndarray,
                     global_scale: float) -> dict[str, float]:
    """U2: one scale per kabupaten with ≥ MIN_ROWS_FOR_KABUPATEN_CALIBRATION rows; the others get `global_scale`."""
    scales = {}
    for kab in config.KABUPATEN_ORDER:
        m = kabupaten == kab
        enough = m.sum() >= config.MIN_ROWS_FOR_KABUPATEN_CALIBRATION
        scales[kab] = kabupaten_scale(y_log[m], q_log[m]) if enough else float(global_scale)
    return scales


def row_scales(kabupaten: np.ndarray, scales: dict[str, float], global_scale: float) -> np.ndarray:
    """Per-row scale from each row's kabupaten."""
    return np.array([scales.get(k, global_scale) for k in kabupaten], dtype=float)


def _inside(y_log: np.ndarray, q_log: np.ndarray) -> np.ndarray:
    return (y_log >= q_log[:, 0]) & (y_log <= q_log[:, 2])


def _width_pct(q_log: np.ndarray) -> np.ndarray:
    """(q90 − q10) / q50 in % per row."""
    q = np.exp(q_log)
    return (q[:, 2] - q[:, 0]) / q[:, 1] * 100


def range_scale_table(y_log: np.ndarray, q_log: np.ndarray, kabupaten: np.ndarray, scales: dict[str, float],
                      global_scale: float) -> pd.DataFrame:
    """Per kabupaten: rows, coverage raw / before (global factor) / after (own scale), widths before/after."""
    rows = []
    for kab in config.KABUPATEN_ORDER:
        m = kabupaten == kab
        if not m.any():
            continue
        before, after = widen_log_interval(q_log[m], global_scale), widen_log_interval(q_log[m], scales[kab])
        rows.append({
            "kabupaten": kab, "n": int(m.sum()),
            "source": "own" if m.sum() >= config.MIN_ROWS_FOR_KABUPATEN_CALIBRATION else "global",
            "coverage_raw": float(_inside(y_log[m], q_log[m]).mean() * 100),
            "coverage_before": float(_inside(y_log[m], before).mean() * 100),
            "scale": scales[kab],
            "coverage_after": float(_inside(y_log[m], after).mean() * 100),
            "width_before": float(_width_pct(before).mean()),
            "width_after": float(_width_pct(after).mean()),
        })
    return pd.DataFrame(rows)


def nested_scale_check(y_log: np.ndarray, q_log: np.ndarray, kabupaten: np.ndarray, folds: list) -> dict:
    """Honest check of U2: global factor and kabupaten scales chosen on 4 folds' OOF, coverage on the 5th."""
    inside = np.zeros(len(y_log), dtype=bool)
    per_fold = []
    for tr, te in folds:
        g = suggest_factor(y_log[tr], q_log[tr])
        sc = fit_range_scales(y_log[tr], q_log[tr], kabupaten[tr], g)
        ins = _inside(y_log[te], widen_log_interval(q_log[te], row_scales(kabupaten[te], sc, g)))
        inside[te] = ins
        per_fold.append(float(ins.mean() * 100))
    by_kab = {k: float(inside[kabupaten == k].mean() * 100) for k in config.KABUPATEN_ORDER if (kabupaten == k).any()}
    return {"coverage_mean": float(np.mean(per_fold)), "coverage_std": float(np.std(per_fold, ddof=1)),
            "coverage_by_fold": per_fold, "coverage_by_kabupaten": by_kab}


def nested_factor_check(y_log: np.ndarray, oof_log: np.ndarray, folds: list) -> dict:
    """Honest check of a calibration factor: choose it on 4 folds' OOF, measure coverage on the 5th."""
    held_out, factors = [], []
    for tr, te in folds:
        f = suggest_factor(y_log[tr], oof_log[tr])
        factors.append(f)
        held_out.append(coverage_at(y_log[te], oof_log[te], f) * 100)
    return {"factors": factors, "coverage_mean": float(np.mean(held_out)),
            "coverage_std": float(np.std(held_out, ddof=1)), "coverage_by_fold": held_out}


def _summary(per_fold: list[dict]) -> dict[str, float]:
    df = pd.DataFrame(per_fold)
    out = {f"{c}_mean": float(df[c].mean()) for c in df.columns if c != "crossed_rows"}
    out |= {f"{c}_std": float(df[c].std(ddof=1)) for c in df.columns if c != "crossed_rows"}
    out["crossed_rows"] = int(df["crossed_rows"].sum())
    return out


def range_scales_md(table: pd.DataFrame, rs: dict) -> str:
    """metrics.md subsection for the U2 per-kabupaten scales."""
    t = pd.DataFrame({
        "Kabupaten": table["kabupaten"], "Rows": table["n"],
        "Scale": [f"{s:.2f}" + (" (global)" if src == "global" else "") for s, src in zip(table["scale"],
                                                                                         table["source"])],
        "Coverage raw": table["coverage_raw"].map("{:.1f}%".format),
        "Coverage before": table["coverage_before"].map("{:.1f}%".format),
        "Coverage after": table["coverage_after"].map("{:.1f}%".format),
        "Honest (nested)": table["kabupaten"].map(lambda k: f"{rs['nested_check']['coverage_by_kabupaten'][k]:.1f}%"),
        "Mean width before": table["width_before"].map("{:.0f}%".format),
        "Mean width after": table["width_after"].map("{:.0f}%".format),
    })
    a, n = rs["spatial_after"], rs["nested_check"]
    return (
        "### Per-kabupaten range calibration (P7b U2)\n\n"
        f"Each kabupaten with ≥ {config.MIN_ROWS_FOR_KABUPATEN_CALIBRATION} rows gets the smallest scale on the "
        f"log half-widths around q50 that reaches {config.TARGET_COVERAGE:.0%} spatial out-of-fold coverage, "
        f"clipped to {config.RANGE_SCALE_CAP[0]:g}–{config.RANGE_SCALE_CAP[1]:g}; smaller kabupaten keep the global "
        f"factor {rs['global_scale']:g}. \"Before\" = global factor for every row; \"raw\" = no widening. "
        "Scales are stored in `models/metadata.json` (`range_scales`) and applied by `predict_house`.\n\n"
        + _md(t) + "\n\n"
        f"Overall spatial coverage after: **{a['coverage_all']:.1f}%** (fold mean {a['coverage_mean']:.1f} ± "
        f"{a['coverage_std']:.1f}%), mean width {a['width_mean']:.0f}% (median {a['width_median']:.0f}%).\n\n"
        "**\"Coverage after\" is optimistic:** it is measured on the same out-of-fold predictions used to pick "
        "the scales. The \"Honest (nested)\" column chooses the global factor and the scales on 4 folds and "
        f"measures the 5th: {n['coverage_mean']:.1f} ± {n['coverage_std']:.1f}% overall. The drift check (U4) on "
        "new listings is the independent test.\n"
    )


def write_metrics_section(res: dict, by_kab: pd.DataFrame, tree_params: dict, range_table: pd.DataFrame,
                          range_scales: dict) -> None:
    """Replace (or append) the price-range section of reports/metrics.md."""
    lo_t, hi_t = (t * 100 for t in config.COVERAGE_TARGET)
    rows = []
    for cv in CVS:
        r = res[cv]
        rows.append({"CV": cv, "Coverage of [q10, q90]": f"{r['coverage_mean']:.1f} ± {r['coverage_std']:.1f}%",
                     "Below q10": f"{r['below_low_mean']:.1f}%", "Above q90": f"{r['above_high_mean']:.1f}%",
                     "Width (q90 − q10) / q50": f"{r['width_mean_mean']:.0f}% (median {r['width_median_mean']:.0f}%)",
                     "Point estimate outside range": f"{r['point_outside_mean']:.1f}%"})
    s = res["spatial"]["coverage_mean"]
    where = "inside" if lo_t <= s <= hi_t else "below" if s < lo_t else "above"
    factor = res["factor_applied"]
    sug, nested = res["suggested_factor"], res["nested"]
    kab = by_kab.assign(coverage=by_kab["coverage"].round(1), width=by_kab["width"].round(0)).rename(
        columns={"coverage": "coverage %", "width": "mean width %"})
    calib = (
        f"Calibration factor applied: **{factor:g}** "
        + ("(raw quantiles). " if factor == 1 else
           f"(log interval widened around q50; raw quantiles covered {res['spatial']['raw_coverage_all']:.1f}% "
           f"spatial / {res['random']['raw_coverage_all']:.1f}% random). ")
        + f"A factor of {sug:g} would reach {config.CALIBRATION_GOAL:.0%} spatial coverage; chosen on 4 folds and "
          f"tested on the 5th it gives {nested['coverage_mean']:.1f} ± {nested['coverage_std']:.1f}% coverage."
    )
    section = (
        f"{METRICS_SECTION}\n\n"
        f"LightGBM quantile models at {', '.join(f'{q:g}' for q in config.QUANTILES)} with the point model's tree "
        f"settings (`num_leaves={tree_params['num_leaves']}`, `min_child_samples={tree_params['min_child_samples']}`). "
        "The app shows [q10, q90] as the likely range (nominally 80%). Crossing quantiles are sorted per row "
        f"({res['spatial']['crossed_rows']} of {res['n_rows']:,} rows crossed under spatial CV).\n\n"
        + _md(pd.DataFrame(rows)) + "\n\n"
        f"Spatial coverage is **{s:.1f}%**, {where} the {lo_t:.0f}–{hi_t:.0f}% target. Width is the range as a "
        "share of the middle (q50) estimate."
        + (f" The raw ranges are too narrow under both CV schemes (random: "
           f"{res['random']['coverage_mean']:.1f}%), so unseen kecamatan are not the main cause: quantile trees "
           f"fitted on ~{res['n_rows'] * (config.N_SPLITS - 1) // config.N_SPLITS:,} listings follow the "
           "training spread too closely and understate it for new houses." if s < lo_t else "")
        + (" The coverage above is after calibration; the factor was chosen on these same out-of-fold "
           "predictions, so the nested check below is the honest number." if factor != 1 else "")
        + "\n\n" + calib + "\n\n"
        "### Coverage by kabupaten (spatial CV, out-of-fold)\n\n" + _md(kab) + "\n\n"
        + range_scales_md(range_table, range_scales)
    )
    text = replace_md_section(config.METRICS_MD.read_text(encoding="utf-8"), METRICS_SECTION, section)
    config.METRICS_MD.write_text(text, encoding="utf-8")


def main() -> None:
    """Cross-validate quantile models, fit final models, update metadata and metrics.md."""
    table = pd.read_csv(config.MODEL_TABLE, dtype={config.CV_GROUP_COLUMN: str})
    osm = pd.read_csv(config.OSM_FEATURES_CSV)
    features = feature_columns(osm)
    check_no_leakage(features)
    tree_params = chosen_tree_params()
    folds = make_folds(table)
    factor = config.INTERVAL_CALIBRATION_FACTOR
    print(f"{len(table):,} rows, {len(features)} features; quantiles {config.QUANTILES}; tree params {tree_params}; "
          f"calibration factor {factor:g}")

    res: dict = {"n_rows": len(table), "factor_applied": factor}
    y_log = table["log_price"].to_numpy()
    oof_spatial = None
    for cv in CVS:
        per_fold, oof_log, _ = cross_validate_quantiles(table, features, folds[cv], tree_params, factor)
        res[cv] = _summary(per_fold)
        if cv == "spatial":
            oof_spatial = oof_log
        res[cv]["raw_coverage_all"] = coverage_at(y_log, oof_log, 1.0) * 100
        r = res[cv]
        print(f"  {cv:<8} coverage {r['coverage_mean']:5.1f} ± {r['coverage_std']:4.1f}%  "
              f"(below q10 {r['below_low_mean']:.1f}%, above q90 {r['above_high_mean']:.1f}%)  "
              f"width mean {r['width_mean_mean']:.0f}% median {r['width_median_mean']:.0f}%  "
              f"point outside {r['point_outside_mean']:.1f}%  crossed rows {r['crossed_rows']}")

    res["suggested_factor"] = suggest_factor(y_log, oof_spatial)
    res["nested"] = nested_factor_check(y_log, oof_spatial, folds["spatial"])
    sug_q = np.exp(widen_log_interval(oof_spatial, res["suggested_factor"]))
    res["suggested_width_mean"] = float(np.mean((sug_q[:, 2] - sug_q[:, 0]) / sug_q[:, 1] * 100))

    q = np.exp(widen_log_interval(oof_spatial, factor))
    y = table["price_idr"].to_numpy()
    by_kab = (pd.DataFrame({"kabupaten": table["kabupaten"], "inside": (y >= q[:, 0]) & (y <= q[:, 2]),
                            "width": (q[:, 2] - q[:, 0]) / q[:, 1] * 100})
              .groupby("kabupaten").agg(n=("inside", "size"), coverage=("inside", "mean"), width=("width", "mean"))
              .reset_index())
    by_kab["coverage"] *= 100

    kab_arr = table["kabupaten"].to_numpy()
    scales = fit_range_scales(y_log, oof_spatial, kab_arr, factor)
    range_table = range_scale_table(y_log, oof_spatial, kab_arr, scales, factor)
    q_after = widen_log_interval(oof_spatial, row_scales(kab_arr, scales, factor))
    inside_after = _inside(y_log, q_after)
    fold_cov = [float(inside_after[te].mean() * 100) for _, te in folds["spatial"]]
    width_after = _width_pct(q_after)
    range_scales = {
        "target_coverage": config.TARGET_COVERAGE,
        "min_rows_for_kabupaten": config.MIN_ROWS_FOR_KABUPATEN_CALIBRATION,
        "cap": list(config.RANGE_SCALE_CAP),
        "global_scale": factor,
        "by_kabupaten": scales,
        "table": range_table.to_dict("records"),
        "spatial_after": {"coverage_all": float(inside_after.mean() * 100), "coverage_mean": float(np.mean(fold_cov)),
                          "coverage_std": float(np.std(fold_cov, ddof=1)), "width_mean": float(width_after.mean()),
                          "width_median": float(np.median(width_after))},
        "nested_check": nested_scale_check(y_log, oof_spatial, kab_arr, folds["spatial"]),
        "note": "coverage_after is measured on the OOF predictions used to choose the scales (optimistic); "
                "nested_check chooses on 4 folds and tests on the 5th.",
    }

    X = table[features]
    for alpha, path in QUANTILE_MODEL_PATHS.items():
        quantile_model(alpha, tree_params).fit(X, y_log).booster_.save_model(str(path))

    meta = json.loads(config.METADATA_JSON.read_text(encoding="utf-8"))
    meta["quantile"] = {
        "quantiles": list(config.QUANTILES),
        "models": {f"q{int(a * 100)}": p.name for a, p in QUANTILE_MODEL_PATHS.items()},
        "params": {**config.LGBM_BASE_PARAMS, **tree_params, "objective": "quantile"},
        "crossing_fix": "sort per row",
        "calibration_factor": factor,
        "suggested_calibration_factor": res["suggested_factor"],
        "nested_calibration_check": res["nested"],
        "coverage_target": list(config.COVERAGE_TARGET),
        "cv": {cv: res[cv] for cv in CVS},
        "coverage_by_kabupaten_spatial": by_kab.to_dict("records"),
    }
    meta["range_scales"] = range_scales
    config.METADATA_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_metrics_section(res, by_kab, tree_params, range_table, range_scales)

    lo_t, hi_t = (t * 100 for t in config.COVERAGE_TARGET)
    s = res["spatial"]["coverage_mean"]
    print("Coverage by kabupaten (spatial):")
    for r in by_kab.itertuples():
        print(f"  {r.kabupaten:<16} n={r.n:<4} coverage {r.coverage:5.1f}%  width {r.width:4.0f}%")
    n = res["nested"]
    print(f"Suggested calibration factor: {res['suggested_factor']:g} (spatial OOF coverage → "
          f"{config.CALIBRATION_GOAL:.0%}, mean width {res['suggested_width_mean']:.0f}%); nested check "
          f"(choose on 4 folds, test on 5th): {n['coverage_mean']:.1f} ± {n['coverage_std']:.1f}%, factors {n['factors']}")
    print(f"U2 per-kabupaten range scales (target {config.TARGET_COVERAGE:.0%}, cap {config.RANGE_SCALE_CAP}, "
          f"own scale if n >= {config.MIN_ROWS_FOR_KABUPATEN_CALIBRATION}, else global {factor:g}):")
    hon = range_scales["nested_check"]
    for r in range_table.itertuples():
        print(f"  {r.kabupaten:<16} n={r.n:<4} {r.source:<6} coverage raw {r.coverage_raw:5.1f}%  before "
              f"{r.coverage_before:5.1f}%  scale {r.scale:.2f}  after {r.coverage_after:5.1f}%  honest "
              f"{hon['coverage_by_kabupaten'][r.kabupaten]:5.1f}%  width {r.width_before:.0f}% -> {r.width_after:.0f}%")
    a = range_scales["spatial_after"]
    print(f"  overall after: coverage {a['coverage_all']:.1f}% (fold mean {a['coverage_mean']:.1f} ± "
          f"{a['coverage_std']:.1f}%), width mean {a['width_mean']:.0f}% median {a['width_median']:.0f}%; "
          f"honest (nested) {hon['coverage_mean']:.1f} ± {hon['coverage_std']:.1f}%")
    print(f"Saved {', '.join(p.name for p in QUANTILE_MODEL_PATHS.values())}; updated "
          f"{config.METADATA_JSON.name} and {config.METRICS_MD.relative_to(config.ROOT)}")
    if not lo_t <= s <= hi_t:
        print(f"NOTE: spatial coverage {s:.1f}% is outside the {lo_t:.0f}–{hi_t:.0f}% target — needs a decision "
              "(INTERVAL_CALIBRATION_FACTOR in config.py).")


if __name__ == "__main__":
    main()
