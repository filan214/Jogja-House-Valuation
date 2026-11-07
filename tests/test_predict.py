import numpy as np
import pandas as pd
import pytest

from src import config
from src.comps import find_comps
from src.evaluate import sort_quantiles, widen_log_interval
from src.explain import contributions, explain, grouped_contributions, load_point_booster
from src.predict import load_artifacts, predict_house, verdict

DEPOK = {"kabupaten": "Sleman", "kecamatan": "Depok", "land_m2": 120, "building_m2": 90, "bedrooms": 3,
         "bathrooms": 2, "carport": 1, "flag_shm": True, "flag_complex": True}
KEYS = {"estimate_idr", "low_idr", "high_idr", "estimate_2024_idr", "low_2024_idr", "high_2024_idr",
        "adjustment", "verdict", "drivers", "comps", "warnings"}


def test_typical_depok_house():
    out = predict_house(DEPOK, asking_price=900_000_000)
    assert set(out) == KEYS
    tol = 0.02
    assert out["low_idr"] * (1 - tol) <= out["estimate_idr"] <= out["high_idr"] * (1 + tol)
    assert len(out["comps"]) == config.N_COMPS
    assert all(c["kabupaten"] == "Sleman" for c in out["comps"])
    assert out["verdict"] in {"above", "within", "below"}
    assert 1 <= len(out["drivers"]) <= config.N_DRIVERS
    assert all(set(d) == {"label", "pct_effect"} for d in out["drivers"])
    assert out["warnings"] == []


def test_land_out_of_range_warns():
    out = predict_house({**DEPOK, "land_m2": 10_000})
    assert any("Land area" in w and config.OUT_OF_RANGE_MESSAGE in w for w in out["warnings"])


def test_low_data_kecamatan_warns():
    stats = load_artifacts().stats
    row = stats[(stats["n_listings"] == 0)].iloc[0]
    out = predict_house({**DEPOK, "kecamatan": row["kecamatan"], "kabupaten": row["kabupaten"]})
    assert any("limited data" in w for w in out["warnings"])


def test_unknown_kecamatan_raises():
    with pytest.raises(ValueError):
        predict_house({**DEPOK, "kecamatan": "Atlantis"})


def test_verdict_rules():
    assert verdict(None, 1, 3) is None
    assert verdict(4, 1, 3) == "above"
    assert verdict(0, 1, 3) == "below"
    assert verdict(2, 1, 3) == verdict(1, 1, 3) == verdict(3, 1, 3) == "within"


def test_sort_and_widen_quantiles():
    q = sort_quantiles(np.array([[2.0, 1.0, 3.0]]))
    assert q.tolist() == [[1.0, 2.0, 3.0]]
    assert widen_log_interval(q, 1.0).tolist() == q.tolist()
    assert widen_log_interval(q, 1.5).tolist() == [[0.5, 2.0, 3.5]]


def test_grouped_contributions_sum_to_prediction():
    art = load_artifacts()
    from src.features import build_model_features
    X = build_model_features(pd.DataFrame([DEPOK]), art.osm)[art.feature_columns]
    booster = load_point_booster()
    row = contributions(X, booster).iloc[0]
    groups = grouped_contributions(row)
    assert groups.sum() + row["bias"] == pytest.approx(booster.predict(X, raw_score=True)[0])
    assert set(config.FEATURE_LABELS) >= set(art.feature_columns)
    top = explain(X, booster)
    effects = [abs(np.log1p(d["pct_effect"] / 100)) for d in top]
    assert effects == sorted(effects, reverse=True)


def test_comps_fall_back_to_full_pool():
    art = load_artifacts()
    query = {"kabupaten": "Gunungkidul", "kecamatan": "Wonosari", "land_m2": 150, "building_m2": 80,
             "bedrooms": 2, "bathrooms": None}
    n_gk = int((art.pool["kabupaten"] == "Gunungkidul").sum())
    k = n_gk + 2
    comps = find_comps(query, k=k, pool=art.pool, centroids=art.osm)
    assert len(comps) == k
    assert (comps["kabupaten"] != "Gunungkidul").any()
    same = find_comps(query, k=min(n_gk, 3), pool=art.pool, centroids=art.osm)
    assert (same["kabupaten"] == "Gunungkidul").all()
    assert comps["distance"].is_monotonic_increasing


def test_kabupaten_scale_search_and_cap():
    from src.quantile import coverage_at, fit_range_scales, kabupaten_scale
    rng = np.random.default_rng(config.RANDOM_STATE)
    y = rng.normal(0, 1, 400)
    q = np.column_stack([np.full(400, -0.5), np.zeros(400), np.full(400, 0.5)])
    s = kabupaten_scale(y, q)
    assert config.RANGE_SCALE_CAP[0] <= s <= config.RANGE_SCALE_CAP[1]
    assert coverage_at(y, q, s) >= config.TARGET_COVERAGE or s == config.RANGE_SCALE_CAP[1]
    assert s == config.RANGE_SCALE_CAP[0] or coverage_at(y, q, s - 0.01) < config.TARGET_COVERAGE
    wide = np.column_stack([np.full(400, -9.0), np.zeros(400), np.full(400, 9.0)])
    assert kabupaten_scale(y, wide) == config.RANGE_SCALE_CAP[0]
    kab = np.array(["Sleman"] * 390 + ["Gunungkidul"] * 10)
    scales = fit_range_scales(y, q, kab, global_scale=1.23)
    assert scales["Gunungkidul"] == 1.23 and scales["Kulon Progo"] == 1.23
    assert scales["Sleman"] == kabupaten_scale(y[:390], q[:390])


def test_range_scale_used_by_predict():
    from src.predict import range_scale
    meta = load_artifacts().metadata
    rs = meta["range_scales"]
    assert set(rs["by_kabupaten"]) == set(config.KABUPATEN_ORDER)
    assert range_scale("Sleman", meta) == rs["by_kabupaten"]["Sleman"]
    assert range_scale("Nowhere", meta) == rs["global_scale"]
    assert range_scale("Sleman", {"quantile": {"calibration_factor": 1.4}}) == 1.4
    art = load_artifacts()
    from src.features import build_model_features
    house = {**DEPOK, "kabupaten": "Kota Yogyakarta", "kecamatan": "Gondokusuman"}
    X = build_model_features(pd.DataFrame([house]), art.osm)[art.feature_columns]
    q = sort_quantiles(np.array([[b.predict(X)[0] for b in art.quantiles]]))
    q = np.exp(widen_log_interval(q, rs["by_kabupaten"]["Kota Yogyakarta"]))
    out = predict_house(house)
    assert out["low_idr"] == int(round(q[0, 0])) and out["high_idr"] == int(round(q[0, 2]))


def _set_index(monkeypatch, factor, source="Test index", as_of="2026-09"):
    monkeypatch.setattr(config, "PRICE_INDEX_FACTOR", factor)
    monkeypatch.setattr(config, "PRICE_INDEX_SOURCE", source)
    monkeypatch.setattr(config, "PRICE_INDEX_AS_OF", as_of)


def test_no_factor_adjusted_equals_2024(monkeypatch):
    _set_index(monkeypatch, None, None, None)
    out = predict_house(DEPOK)
    assert out["adjustment"] is None
    for k in ("estimate", "low", "high"):
        assert out[f"{k}_idr"] == out[f"{k}_2024_idr"]


def test_factor_scales_all_outputs(monkeypatch):
    base = predict_house(DEPOK)
    _set_index(monkeypatch, 1.2)
    out = predict_house(DEPOK)
    assert out["adjustment"] == {"factor": 1.2, "source": "Test index", "as_of": "2026-09"}
    for k in ("estimate", "low", "high"):
        assert out[f"{k}_2024_idr"] == base[f"{k}_2024_idr"]
        assert abs(out[f"{k}_idr"] - out[f"{k}_2024_idr"] * 1.2) <= 1


def test_verdict_uses_adjusted_range(monkeypatch):
    _set_index(monkeypatch, 1.2)
    out = predict_house(DEPOK)
    asking = (out["high_2024_idr"] + out["high_idr"]) // 2
    assert verdict(asking, out["low_2024_idr"], out["high_2024_idr"]) == "above"
    assert predict_house(DEPOK, asking_price=asking)["verdict"] == "within"
    below = (out["low_2024_idr"] + out["low_idr"]) // 2
    assert predict_house(DEPOK, asking_price=below)["verdict"] == "below"


@pytest.mark.parametrize("factor, source, as_of", [(0, "x", "2026-09"), (-1.1, "x", "2026-09"), ("1.1", "x", "2026-09"),
                                                   (1.1, None, "2026-09"), (1.1, "x", None)])
def test_incomplete_index_setting_raises(monkeypatch, factor, source, as_of):
    _set_index(monkeypatch, factor, source, as_of)
    with pytest.raises(ValueError):
        predict_house(DEPOK)


def test_real_factor_untouched():
    """Guard: the factor is set by Filan from published reports, never by code (CLAUDE.md)."""
    assert config.PRICE_INDEX_FACTOR is None or (config.PRICE_INDEX_SOURCE and config.PRICE_INDEX_AS_OF)
