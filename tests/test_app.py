"""Streamlit pages driven headlessly with streamlit.testing (no browser needed)."""

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import config

APP = str(config.ROOT / "streamlit_app.py")
PAGES = tuple(str(config.ROOT / p) for p in ("pages/1_Market_map.py", "pages/2_About_the_model.py"))
TIMEOUT = 60
HEAVY = ("osmnx", "geopandas", "shap", "matplotlib")


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    return at


def _submit(at: AppTest, **values) -> AppTest:
    for key in ("kabupaten", "kecamatan"):
        if key in values:
            at.selectbox(key=key).set_value(values.pop(key)).run()
    for key, v in values.items():
        at.number_input(key=key).set_value(v)
    at.button[0].click().run()
    assert not at.exception, at.exception
    return at


def _busiest_kecamatan(kab: str) -> str:
    s = pd.read_csv(config.KECAMATAN_STATS_CSV)
    return s[s["kabupaten"] == kab].sort_values("n_listings", ascending=False)["kecamatan"].iloc[0]


@pytest.mark.parametrize("kab", config.KABUPATEN_ORDER)
def test_valuation_each_kabupaten(kab):
    at = _submit(_app(), kabupaten=kab, kecamatan=_busiest_kecamatan(kab))
    labels = [m.label for m in at.metric]
    assert labels == ["Estimate", "Likely range"]
    assert at.metric[0].value.startswith("Rp ")
    assert not at.error
    assert len(at.dataframe) == 1


def test_verdict_with_asking_price():
    at = _submit(_app(), asking=900_000_000)
    assert [m.label for m in at.metric][-1] == "Verdict"
    assert at.metric[2].value in config.VERDICT_LABELS.values()


def test_invalid_input_shows_error():
    at = _submit(_app(), land_m2=0.0)
    assert any("Land area must be greater than 0" in e.value for e in at.error)
    assert not at.metric


def test_out_of_range_input_warns():
    at = _submit(_app(), land_m2=10_000.0)
    assert any(config.OUT_OF_RANGE_MESSAGE in w.value for w in at.warning)
    assert at.metric


def test_limited_data_label_and_warning():
    s = pd.read_csv(config.KECAMATAN_STATS_CSV)
    low = s[(s["kabupaten"] == "Gunungkidul") & (s["n_listings"] < config.LOW_DATA_KECAMATAN_THRESHOLD)]
    at = _app()
    at.selectbox(key="kabupaten").set_value("Gunungkidul").run()
    assert any("(limited data)" in o for o in at.selectbox(key="kecamatan").options)
    at = _submit(at, kecamatan=low["kecamatan"].iloc[0])
    assert any("limited data" in w.value for w in at.warning)


@pytest.mark.parametrize("page", PAGES)
def test_pages_render(page):
    at = AppTest.from_file(page, default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception


def test_map_metric_switch():
    at = AppTest.from_file(PAGES[0], default_timeout=TIMEOUT).run()
    for label in config.MAP_METRICS:
        at.radio[0].set_value(label).run()
        assert not at.exception, at.exception


def test_app_path_has_no_heavy_imports():
    src = [config.ROOT / "src" / f"{m}.py" for m in ("app_data", "predict", "explain", "comps", "features",
                                                    "evaluate", "utils", "config")]
    for f in (APP, *PAGES, *src):
        top = [ln for ln in Path(f).read_text(encoding="utf-8").splitlines() if re.match(r"(import|from) ", ln)]
        assert not [ln for ln in top if any(h in ln for h in HEAVY)], f
    code = "import sys, src.app_data; print(','.join(m for m in %r if m in sys.modules))" % (HEAVY,)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


def test_caption_without_adjustment(monkeypatch):
    monkeypatch.setattr(config, "PRICE_INDEX_FACTOR", None)
    at = _submit(_app())
    assert any(c.value == config.NO_ADJUSTMENT_CAPTION for c in at.caption)
    assert any(c.value == config.DISCLAIMER for c in at.caption)


def test_caption_with_adjustment(monkeypatch):
    from src.app_data import adjustment_caption
    from src.utils import format_idr
    monkeypatch.setattr(config, "PRICE_INDEX_FACTOR", 1.15)
    monkeypatch.setattr(config, "PRICE_INDEX_SOURCE", "Test index")
    monkeypatch.setattr(config, "PRICE_INDEX_AS_OF", "2026-09")
    at = _submit(_app())
    caps = [c.value for c in at.caption]
    assert any(c.startswith("At 2024 prices: Rp ") and "adjusted +15% using Test index (as of Sep 2026)" in c
               for c in caps), caps
    assert config.DISCLAIMER_ADJUSTED in caps
    res = {"adjustment": {"factor": 0.95, "source": "S", "as_of": "2026-01"}, "estimate_2024_idr": 875_000_000}
    assert adjustment_caption(res) == f"At 2024 prices: {format_idr(875_000_000)} · adjusted −5% using S (as of Jan 2026)"
    about = AppTest.from_file(PAGES[1], default_timeout=TIMEOUT).run()
    assert not about.exception
    text = " ".join(m.value for m in about.markdown)
    assert "city-wide" in text and "Test index" in text
