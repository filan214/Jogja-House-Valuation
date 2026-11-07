"""P6d — Single entry point for the app: predict_house(inputs, asking_price).

Loads everything once (lazy, module level) from models/ and data/app/ only:
point + quantile LightGBM boosters, feature_columns.json, metadata.json,
osm_kecamatan_features.csv, kecamatan_stats.csv, comps_pool.csv.
Features come from src.features.build_model_features (single source of truth).
App-safe: no shap, matplotlib, osmnx or geopandas.

Example:
    python -c "from src.predict import predict_house; print(predict_house({...}))"
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src.comps import find_comps
from src.evaluate import sort_quantiles, widen_log_interval
from src.explain import explain
from src.features import build_model_features

INPUT_COLUMNS = ("kabupaten", "kecamatan", "land_m2", "building_m2", "bedrooms", "bathrooms", "carport")
COMP_FIELDS = ("kecamatan", "kabupaten", "land_m2", "building_m2", "bedrooms", "bathrooms", "price_idr")


@dataclass(frozen=True)
class Artifacts:
    """Everything predict_house needs, loaded from disk once."""

    point: lgb.Booster
    quantiles: tuple[lgb.Booster, ...]
    feature_columns: list[str]
    metadata: dict
    osm: pd.DataFrame
    stats: pd.DataFrame
    pool: pd.DataFrame


@lru_cache(maxsize=1)
def load_artifacts() -> Artifacts:
    """Load models and lookup files (cached; the app wraps this in st.cache_resource)."""
    booster = lambda p: lgb.Booster(model_file=str(p))
    cols = json.loads(config.FEATURE_COLUMNS_JSON.read_text(encoding="utf-8"))
    point = booster(config.LGBM_POINT_MODEL)
    if point.feature_name() != cols:
        raise ValueError("lgbm_point feature order differs from feature_columns.json")
    return Artifacts(
        point=point,
        quantiles=tuple(booster(p) for p in (config.LGBM_Q10_MODEL, config.LGBM_Q50_MODEL, config.LGBM_Q90_MODEL)),
        feature_columns=cols,
        metadata=json.loads(config.METADATA_JSON.read_text(encoding="utf-8")),
        osm=pd.read_csv(config.APP_OSM_FEATURES_CSV),
        stats=pd.read_csv(config.KECAMATAN_STATS_CSV),
        pool=pd.read_csv(config.COMPS_POOL_CSV),
    )


def _input_frame(inputs: dict) -> pd.DataFrame:
    row = {c: inputs.get(c) for c in INPUT_COLUMNS}
    row["carport"] = row["carport"] if row["carport"] is not None else 0
    row |= {f: bool(inputs.get(f, False)) for f in config.ACTIVE_FLAGS}
    for c in ("land_m2", "building_m2", "bedrooms", "bathrooms", "carport"):
        row[c] = np.nan if row[c] is None else float(row[c])
    return pd.DataFrame([row])


def _fmt(v: float) -> str:
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.1f}"


def input_warnings(inputs: dict, art: Artifacts) -> list[str]:
    """Out-of-training-range inputs and low-data kecamatan."""
    out = []
    for col, r in art.metadata["training_ranges"].items():
        v = inputs.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if not r["min"] <= float(v) <= r["max"]:
            label = config.RANGE_CHECK_LABELS.get(col, col)
            out.append(f"{label} {_fmt(float(v))} is outside the training range "
                       f"({_fmt(r['min'])}–{_fmt(r['max'])}). {config.OUT_OF_RANGE_MESSAGE}")
    hit = art.stats[(art.stats["kecamatan"] == inputs["kecamatan"]) & (art.stats["kabupaten"] == inputs["kabupaten"])]
    n = int(hit["n_listings"].iloc[0]) if not hit.empty else 0
    if n < config.LOW_DATA_KECAMATAN_THRESHOLD:
        out.append(f"{inputs['kecamatan']} has only {n} training listing{'s' if n != 1 else ''} (limited data); "
                   "the estimate leans on similar areas and is less reliable.")
    return out


def range_scale(kabupaten: str, metadata: dict) -> float:
    """Widening of [q10, q90] for this kabupaten (P7b U2 `range_scales`), else the global P6 factor."""
    rs = metadata.get("range_scales")
    if rs:
        return float(rs["by_kabupaten"].get(kabupaten, rs["global_scale"]))
    return float(metadata.get("quantile", {}).get("calibration_factor", 1.0))


def verdict(asking_price: int | None, low: int, high: int) -> str | None:
    """'above' / 'below' the likely range, else 'within'; None without an asking price."""
    if asking_price is None:
        return None
    if asking_price > high:
        return "above"
    if asking_price < low:
        return "below"
    return "within"


def price_adjustment() -> dict | None:
    """P7b U3: {factor, source, as_of} from config.PRICE_INDEX_*, or None when no factor is set.

    The factor is set by hand from published reports (CLAUDE.md); code never estimates it.
    Read at call time so tests can patch config. Raises ValueError for an incomplete setting.
    """
    factor = config.PRICE_INDEX_FACTOR
    if factor is None:
        return None
    if not (isinstance(factor, (int, float)) and factor > 0):
        raise ValueError(f"config.PRICE_INDEX_FACTOR must be a positive number, got {factor!r}")
    if not config.PRICE_INDEX_SOURCE or not config.PRICE_INDEX_AS_OF:
        raise ValueError("config.PRICE_INDEX_FACTOR is set but PRICE_INDEX_SOURCE / PRICE_INDEX_AS_OF are missing")
    return {"factor": float(factor), "source": config.PRICE_INDEX_SOURCE, "as_of": config.PRICE_INDEX_AS_OF}


def predict_house(inputs: dict, asking_price: int | None = None) -> dict:
    """Estimate, likely range, verdict, drivers, comparables and warnings for one house.

    inputs keys: kabupaten, kecamatan, land_m2, building_m2, bedrooms,
                 bathrooms, carport, and flag_* booleans.
    returns: {
      "estimate_2024_idr": int,                   # lgbm_point, 2024 price level
      "low_2024_idr": int, "high_2024_idr": int,  # q10, q90 (sorted, range-scaled), 2024 level
      "estimate_idr", "low_idr", "high_idr": int, # 2024 values × PRICE_INDEX_FACTOR (= 2024 if None)
      "adjustment": {"factor", "source", "as_of"} | None,
      "verdict": "above" | "within" | "below" | None,   # vs the ADJUSTED range
      "drivers": list[dict],            # from explain()
      "comps": list[dict],              # from find_comps(); listing prices stay at 2024 level
      "warnings": list[str],            # out-of-range inputs, low-data kecamatan
    }
    Raises ValueError for unknown kecamatan/kabupaten or non-positive areas.
    """
    art = load_artifacts()
    X = build_model_features(_input_frame(inputs), art.osm)[art.feature_columns]

    estimate = int(round(float(np.exp(art.point.predict(X)[0]))))
    q_log = sort_quantiles(np.array([[b.predict(X)[0] for b in art.quantiles]]))
    q_log = widen_log_interval(q_log, range_scale(inputs["kabupaten"], art.metadata))
    low, high = (int(round(float(np.exp(v)))) for v in (q_log[0, 0], q_log[0, 2]))

    adj = price_adjustment()
    f = adj["factor"] if adj else 1.0
    est_adj, low_adj, high_adj = (int(round(v * f)) for v in (estimate, low, high))

    comps = find_comps(inputs, k=config.N_COMPS, pool=art.pool, centroids=art.osm)
    comp_rows = [{k: (None if pd.isna(v) else (int(v) if k == "price_idr" else v)) for k, v in r.items()}
                 for r in comps[list(COMP_FIELDS)].to_dict("records")]
    return {
        "estimate_2024_idr": estimate,
        "low_2024_idr": low,
        "high_2024_idr": high,
        "estimate_idr": est_adj,
        "low_idr": low_adj,
        "high_idr": high_adj,
        "adjustment": adj,
        "verdict": verdict(asking_price, low_adj, high_adj),
        "drivers": explain(X, booster=art.point),
        "comps": comp_rows,
        "warnings": input_warnings(inputs, art),
    }
