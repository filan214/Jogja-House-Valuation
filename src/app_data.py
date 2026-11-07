"""P7 — Cached loaders shared by the Streamlit pages.

Models via st.cache_resource, CSV/JSON/GeoJSON via st.cache_data. Reads only
models/ and data/app/. No feature logic here: predictions go through
src.predict.predict_house.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src import config
from src.predict import Artifacts, load_artifacts, predict_house
from src.utils import format_idr


@st.cache_resource(show_spinner="Loading models…")
def artifacts() -> Artifacts:
    """Boosters + lookup tables used by predict_house (loaded once per server)."""
    return load_artifacts()


@st.cache_data
def kecamatan_stats() -> pd.DataFrame:
    """All 78 kecamatan with listing counts and medians."""
    return pd.read_csv(config.KECAMATAN_STATS_CSV, dtype={"kecamatan_code": str})


@st.cache_data
def geojson() -> dict:
    """Simplified kecamatan polygons (feature id = kecamatan_code)."""
    return json.loads(config.KECAMATAN_GEOJSON.read_text(encoding="utf-8"))


@st.cache_data
def metadata() -> dict:
    return json.loads(config.METADATA_JSON.read_text(encoding="utf-8"))


@st.cache_data
def cleaning_summary() -> pd.DataFrame:
    return pd.read_csv(config.CLEANING_SUMMARY_CSV)


@st.cache_data
def drift_summary() -> dict | None:
    """P7b U4 drift-check summary (data/app/drift_summary.json), or None before the sample is scored."""
    if not config.DRIFT_SUMMARY_JSON.exists():
        return None
    return json.loads(config.DRIFT_SUMMARY_JSON.read_text(encoding="utf-8"))


def kecamatan_options(kabupaten: str) -> list[tuple[str, str]]:
    """(kecamatan, display label) for one kabupaten; '(limited data)' below the threshold."""
    s = kecamatan_stats()
    s = s[s["kabupaten"] == kabupaten].sort_values("kecamatan")
    return [(r.kecamatan, r.kecamatan + (" (limited data)" if r.n_listings < config.LOW_DATA_KECAMATAN_THRESHOLD
                                          else "")) for r in s.itertuples()]


def month_label(as_of: str) -> str:
    """'2026-09' -> 'Sep 2026' (other text returned unchanged)."""
    try:
        return f"{pd.Timestamp(str(as_of) + '-01'):%b %Y}"
    except ValueError:
        return str(as_of)


def adjustment_caption(result: dict) -> str:
    """'At 2024 prices: Rp X · adjusted +Y% using <source> (as of <month>)', or the no-adjustment note."""
    adj = result["adjustment"]
    if adj is None:
        return config.NO_ADJUSTMENT_CAPTION
    pct = f"{(adj['factor'] - 1) * 100:+.0f}%".replace("-", "−")
    return (f"At 2024 prices: {format_idr(result['estimate_2024_idr'])} · adjusted {pct} using {adj['source']} "
            f"(as of {month_label(adj['as_of'])})")


def disclaimer() -> str:
    """PRD §10 disclaimer; mentions the city-wide adjustment only when a factor is set."""
    return config.DISCLAIMER_ADJUSTED if config.PRICE_INDEX_FACTOR is not None else config.DISCLAIMER


def limitations() -> tuple[tuple[str, str], ...]:
    """config.LIMITATIONS with item 2 in its adjusted wording when a factor is set."""
    if config.PRICE_INDEX_FACTOR is None:
        return config.LIMITATIONS
    return (config.LIMITATIONS[0], config.LIMITATION_DATE_ADJUSTED, *config.LIMITATIONS[2:])


def predict(inputs: dict, asking_price: int | None) -> dict:
    """predict_house with the cached artifacts warmed up."""
    artifacts()
    return predict_house(inputs, asking_price)
