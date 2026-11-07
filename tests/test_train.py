import numpy as np
import pandas as pd

from src import config
from src.evaluate import regression_metrics
from src.train import baseline_predict, check_no_leakage, make_folds

import pytest


def _table(n_kec: int = 10, per: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(config.RANDOM_STATE)
    rows = []
    for k in range(n_kec):
        for i in range(per):
            land = float(rng.integers(60, 400))
            ppl = 5e6 + k * 1e6
            rows.append({config.CV_GROUP_COLUMN: f"34{k:05d}", "kabupaten": "Sleman" if k % 2 else "Bantul",
                         "land_m2": land, "price_per_land_m2": ppl, "price_idr": land * ppl})
    return pd.DataFrame(rows)


def test_baseline_uses_training_rows_only():
    t = _table()
    train, test = t.iloc[:30], t.iloc[30:].copy()
    before = baseline_predict(train, test)
    test["price_per_land_m2"] *= 100
    test["price_idr"] *= 100
    assert np.allclose(before, baseline_predict(train, test))


def test_baseline_fallback_order():
    t = _table()
    train = t[t[config.CV_GROUP_COLUMN] != "3400000"]
    test = t[t[config.CV_GROUP_COLUMN] == "3400000"].head(1)
    kab_median = train.loc[train["kabupaten"] == test["kabupaten"].iloc[0], "price_per_land_m2"].median()
    assert baseline_predict(train, test)[0] == pytest.approx(kab_median * test["land_m2"].iloc[0])
    seen = t.head(1)
    assert baseline_predict(t, seen)[0] == pytest.approx(seen["price_idr"].iloc[0])
    other = test.assign(kabupaten="Gunungkidul")
    assert baseline_predict(train, other)[0] == pytest.approx(train["price_per_land_m2"].median()
                                                              * other["land_m2"].iloc[0])


def test_spatial_folds_never_share_a_kecamatan():
    t = _table()
    folds = make_folds(t)
    assert len(folds["spatial"]) == len(folds["random"]) == config.N_SPLITS
    g = t[config.CV_GROUP_COLUMN]
    for tr, te in folds["spatial"]:
        assert not set(g.iloc[tr]) & set(g.iloc[te])


def test_leakage_guard():
    check_no_leakage(["land_m2", "dist_ugm_km"])
    for bad in (["price_per_land_m2"], ["kecamatan"], ["log_price"]):
        with pytest.raises(AssertionError):
            check_no_leakage(bad)


def test_metrics():
    m = regression_metrics(np.array([100.0, 200.0]), np.array([110.0, 150.0]))
    assert m["mae"] == pytest.approx(30.0)
    assert m["mape"] == pytest.approx(17.5)
    assert m["mdape"] == pytest.approx(17.5)
