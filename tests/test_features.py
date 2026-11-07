import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import (NOT_FEATURES, ROAD_FEATURES, build_model_features, feature_columns, kabupaten_column,
                          keyword_flags, parse_road_km, road_features)

OSM = pd.DataFrame({
    "kecamatan": ["Depok", "Jetis", "Jetis"],
    "kabupaten": ["Sleman", "Bantul", "Kota Yogyakarta"],
    **{c: [1.0, 2.0, 3.0] for c in config.LOCATION_FEATURES},
})


def _listing(**kw) -> pd.DataFrame:
    row = {"kecamatan": "Depok", "kabupaten": "Sleman", "land_m2": 120.0, "building_m2": 90.0,
           "bedrooms": 3.0, "bathrooms": 2.0, "carport": 1,
           "description": "Rumah Baru 2 Lantai Siap Huni SHM Dalam Cluster Dekat UGM"}
    row.update(kw)
    return pd.DataFrame([row])


def test_columns_and_no_leakage():
    X = build_model_features(_listing(), OSM)
    assert list(X.columns) == feature_columns(OSM)
    assert not set(X.columns) & NOT_FEATURES
    assert "price_per_land_m2" not in X.columns and "kecamatan" not in X.columns


def test_size_and_ratio():
    x = build_model_features(_listing(), OSM).iloc[0]
    assert x["log_land"] == pytest.approx(np.log(120))
    assert x["building_to_land_ratio"] == pytest.approx(0.75)


def test_flags_from_text():
    x = build_model_features(_listing(), OSM).iloc[0]
    for f in ("flag_shm", "flag_ready", "flag_new", "flag_two_floor", "flag_complex", "flag_near_campus"):
        assert x[f] == 1, f
    assert x["flag_pool"] == 0 and x["flag_furnished"] == 0


@pytest.mark.parametrize("title, expected", [("Rumah 2 Lt Dekat UGM", 1), ("Rumah 2lt Minimalis", 1),
                                             ("Rumah Lantai 2", 1), ("Rumah 12 Lt", 0)])
def test_two_floor_short_form(title, expected):
    assert keyword_flags(pd.Series([title]))["flag_two_floor"].iloc[0] == expected


def test_app_row_matches_training_row():
    """App form (checkboxes, no description) must give the same features as the text row."""
    train = build_model_features(_listing(), OSM)
    flags = train.iloc[0][list(config.ACTIVE_FLAGS)].astype(bool).to_dict()
    app = pd.DataFrame([{"kecamatan": "Depok", "kabupaten": "Sleman", "land_m2": 120, "building_m2": 90,
                         "bedrooms": 3, "bathrooms": 2, "carport": 1, **flags}])
    pd.testing.assert_frame_equal(build_model_features(app, OSM), train, check_dtype=False)


def test_checkbox_overrides_text():
    x = build_model_features(_listing(flag_pool=True, flag_shm=False), OSM).iloc[0]
    assert x["flag_pool"] == 1 and x["flag_shm"] == 0


def test_minimal_app_row_defaults():
    x = build_model_features(pd.DataFrame([{"kecamatan": "Depok", "kabupaten": "Sleman",
                                            "land_m2": 100, "building_m2": 80}]), OSM).iloc[0]
    assert np.isnan(x["bedrooms"]) and x["carport"] == 0
    assert all(x[f] == 0 for f in config.ACTIVE_FLAGS)


def test_one_hot_kabupaten_and_duplicate_kecamatan_name():
    x = build_model_features(_listing(kecamatan="Jetis", kabupaten="Kota Yogyakarta"), OSM).iloc[0]
    assert x[kabupaten_column("Kota Yogyakarta")] == 1 and x[kabupaten_column("Sleman")] == 0
    assert x["dist_center_km"] == 3.0
    assert sum(x[kabupaten_column(k)] for k in config.KABUPATEN_ORDER) == 1


@pytest.mark.parametrize(
    "kw, msg",
    [
        ({"kecamatan": "Atlantis"}, "no OSM features"),
        ({"kabupaten": "Klaten"}, "unknown kabupaten"),
        ({"land_m2": 0}, "positive"),
        ({"building_m2": -5}, "positive"),
    ],
)
def test_invalid_inputs_raise(kw, msg):
    with pytest.raises(ValueError, match=msg):
        build_model_features(_listing(**kw), OSM)


def test_dropped_flags_not_features():
    cols = feature_columns(OSM)
    assert "flag_hgb" not in cols and "flag_main_road" not in cols


@pytest.mark.parametrize("text, road, km", [
    ("Rumah di Jl. Kaliurang km 8 Sleman", "kaliurang", 8.0),
    ("jalan magelang km. 10,5", "magelang", 10.5),
    ("Rumah Baru Jakal Km 7 Dekat UGM", "kaliurang", 7.0),
    ("Kaliurang Km9 cluster", "kaliurang", 9.0),
    ("Jl Imogiri Barat Km 13", "imogiri", 13.0),
    ("Jl. Palagan KM. 8.5", "palagan", 8.5),
    ("Dekat Jl Paris Km 14", "parangtritis", 14.0),
    ("Jl Gito Gati Km 2", "gito gati", 2.0),
])
def test_parse_road_km(text, road, km):
    r = parse_road_km(pd.Series([text]))
    assert r.loc[0, "road_name"] == road and r.loc[0, "road_km"] == km


@pytest.mark.parametrize("text", ["Rumah 5 km dari UGM", "Jl. Sidomoyo Km 1", "Jl Wates Km 99", "Rumah murah", None])
def test_parse_road_km_no_match(text):
    r = parse_road_km(pd.Series([text]))
    assert r[["road_name", "road_km"]].isna().all(axis=None)


def test_road_features_app_row():
    """App form rows: no road given -> NaN / 0; a given km is used; out of range -> NaN / 0."""
    none = road_features(_listing().drop(columns="description"))
    assert np.isnan(none.loc[0, "road_km"]) and none.loc[0, "has_road_km"] == 0
    given = road_features(_listing(road_name="kaliurang", road_km=9.5))
    assert given.loc[0, "road_km"] == 9.5 and given.loc[0, "has_road_km"] == 1
    bad = road_features(_listing(road_name="kaliurang", road_km=0))
    assert np.isnan(bad.loc[0, "road_km"]) and bad.loc[0, "has_road_km"] == 0


def test_road_features_switch(monkeypatch):
    assert not set(ROAD_FEATURES) & set(feature_columns(OSM, road_km=False))
    assert feature_columns(OSM, road_km=True)[-2:] == ROAD_FEATURES
    monkeypatch.setattr(config, "USE_ROAD_KM_FEATURES", True)
    X = build_model_features(_listing(description="Rumah Jakal km 6,5"), OSM)
    assert X.loc[0, "road_km"] == 6.5 and X.loc[0, "has_road_km"] == 1
    single = build_model_features(_listing().drop(columns="description"), OSM)
    assert list(single.columns) == feature_columns(OSM) and single.loc[0, "has_road_km"] == 0
    assert "road_name" in NOT_FEATURES and "road_name" not in X.columns
