import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from src import config
from src.osm_features import compute_features, osm_kabupaten_name, osm_kecamatan_key


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Kapanéwon Depok", "depok"),
        ("Kemantren Gedongtengen", "gedongtengen"),
        ("Gedong Tengen", "gedongtengen"),
        ("Kecamatan Bambang Lipuro", "bambanglipuro"),
        ("Pakualaman", "pakualaman"),
        (None, ""),
    ],
)
def test_osm_kecamatan_key(name, expected):
    assert osm_kecamatan_key(name) == expected


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Kabupaten Sleman", "Sleman"),
        ("Kota Yogyakarta", "Kota Yogyakarta"),
        ("Kabupaten Gunungkidul", "Gunungkidul"),
        ("Kabupaten Kulon Progo", "Kulon Progo"),
        ("Kulonprogo", "Kulon Progo"),
        ("Sleman", "Sleman"),
        ("Kabupaten Klaten", None),
        (None, None),
    ],
)
def test_osm_kabupaten_name(name, expected):
    assert osm_kabupaten_name(name) == expected


def test_compute_features_counts_and_density():
    tugu = gpd.GeoSeries([Point(config.REF_POINTS["tugu_yogyakarta"][::-1])], crs=config.WGS84_CRS)
    c = tugu.to_crs(config.UTM_CRS).iloc[0]
    sq = box(c.x - 1000, c.y - 1000, c.x + 1000, c.y + 1000)
    kec = gpd.GeoDataFrame({"kabupaten": ["Kota Yogyakarta"], "kecamatan": ["Test"],
                            "kecamatan_code": ["3471999"]}, geometry=[sq], crs=config.UTM_CRS)
    pts = [Point(c.x + 10 * i, c.y) for i in range(5)] + [Point(c.x + 5000, c.y)]
    pois = gpd.GeoDataFrame(
        {"amenity": ["school", "school", "clinic", "university", pd.NA, "school"],
         "shop": [pd.NA, pd.NA, pd.NA, pd.NA, "convenience", pd.NA]},
        geometry=pts, crs=config.UTM_CRS,
    ).to_crs(config.WGS84_CRS)

    f = compute_features(kec, pois).iloc[0]
    assert f["area_km2"] == pytest.approx(4.0)
    assert f["dist_center_km"] == pytest.approx(0.0, abs=0.01)
    assert (f["n_schools"], f["n_health"], f["n_universities"], f["n_retail"], f["n_markets"]) == (2, 1, 1, 1, 0)
    assert f["schools_per_km2"] == pytest.approx(0.5)
    assert f["centroid_lat"] == pytest.approx(config.REF_POINTS["tugu_yogyakarta"][0], abs=1e-4)
