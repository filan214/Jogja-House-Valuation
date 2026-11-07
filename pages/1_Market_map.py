"""Page 2 — Market map: kecamatan choropleth (Plotly MapLibre tiles, no API token)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.app_data import geojson, kecamatan_stats
from src.utils import format_idr


def _idr_or_dash(v: float) -> str:
    return "—" if pd.isna(v) else format_idr(int(v))


def map_figure(stats: pd.DataFrame, column: str, label: str) -> go.Figure:
    """Choropleth by `column`; price metrics gray out kecamatan below MIN_LISTINGS_FOR_MAP."""
    s = stats.copy()
    s["ppl_txt"] = s["median_price_per_land_m2"].map(_idr_or_dash)
    s["price_txt"] = s["median_price"].map(_idr_or_dash)
    is_count = column == "n_listings"
    enough = s["n_listings"] >= config.MIN_LISTINGS_FOR_MAP
    colored, gray = (s, s.iloc[0:0]) if is_count else (s[enough], s[~enough])
    custom = ["kecamatan", "kabupaten", "ppl_txt", "price_txt", "n_listings"]
    scale = [[i / (len(config.MAP_SEQUENTIAL) - 1), c] for i, c in enumerate(config.MAP_SEQUENTIAL)]

    z = colored[column].astype(float)
    ticks = np.linspace(z.min(), z.max(), 5)
    ticktext = [f"{t:.0f}" for t in ticks] if is_count else [format_idr(int(t)) for t in ticks]
    common = {"geojson": geojson(), "featureidkey": "properties.kecamatan_code",
              "marker": {"line": {"width": 0.6, "color": "#fcfcfb"}}}
    fig = go.Figure(go.Choroplethmap(
        locations=colored["kecamatan_code"], z=z, colorscale=scale, customdata=colored[custom],
        hovertemplate=("<b>%{customdata[0]}</b> (%{customdata[1]})<br>Median price per land m²: %{customdata[2]}"
                       "<br>Median price: %{customdata[3]}<br>Listings: %{customdata[4]}<extra></extra>"),
        colorbar={"title": {"text": label, "side": "right"}, "tickvals": ticks, "ticktext": ticktext,
                  "thickness": 12, "len": 0.75},
        name=label, **common,
    ))
    if len(gray):
        fig.add_trace(go.Choroplethmap(
            locations=gray["kecamatan_code"], z=np.zeros(len(gray)),
            colorscale=[[0, config.MAP_NO_DATA_COLOR], [1, config.MAP_NO_DATA_COLOR]], showscale=False,
            customdata=gray[custom], name="Not enough data",
            hovertemplate=("<b>%{customdata[0]}</b> (%{customdata[1]})<br>Not enough data "
                           "(%{customdata[4]} listings)<extra></extra>"),
            **common,
        ))
    fig.update_layout(map={"style": config.MAP_STYLE, "center": config.MAP_CENTER, "zoom": config.MAP_ZOOM},
                      margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=560)
    return fig


st.title("Market map")
st.write("Listing prices across the kecamatan of DIY, from the cleaned 2024 Rumah123 listings.")
stats = kecamatan_stats()
label = st.radio("Color by", list(config.MAP_METRICS), horizontal=True)
column = config.MAP_METRICS[label]
st.plotly_chart(map_figure(stats, column, label), width="stretch", config={"scrollZoom": False})
if column != "n_listings":
    st.caption(f"Gray: fewer than {config.MIN_LISTINGS_FOR_MAP} listings — not enough data for a median. "
               "Medians are of listing (asking) prices.")

st.subheader("All kecamatan")
table = pd.DataFrame({
    "Kecamatan": stats["kecamatan"], "Kabupaten": stats["kabupaten"], "Listings": stats["n_listings"],
    "Median price per land m² (Rp jt)": stats["median_price_per_land_m2"] / 1e6,
    "Median price (Rp jt)": stats["median_price"] / 1e6,
})
low = table["Listings"] < config.MIN_LISTINGS_FOR_MAP
table.loc[low, ["Median price per land m² (Rp jt)", "Median price (Rp jt)"]] = np.nan
st.dataframe(
    table.sort_values("Listings", ascending=False), hide_index=True, width="stretch",
    column_config={"Median price per land m² (Rp jt)": st.column_config.NumberColumn(format="%.1f"),
                   "Median price (Rp jt)": st.column_config.NumberColumn(format="localized")},
)
st.caption(f"Click a column header to sort. Medians are left empty below {config.MIN_LISTINGS_FOR_MAP} listings. "
           "1 M (miliar) = 1,000 jt (juta).")
