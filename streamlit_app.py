"""Jogja House Valuation — Streamlit entry point and Valuation page.

Run: streamlit run streamlit_app.py
The app never trains and never calls OSM: it loads models/ and data/app/ only.
All predictions go through src.predict.predict_house.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.app_data import adjustment_caption, artifacts, disclaimer, kecamatan_options, metadata, predict
from src.utils import format_idr

ABOUT_PAGE = "pages/2_About_the_model.py"


def _date_range() -> str:
    start, end = (pd.Timestamp(d) for d in config.DATA_DATE_RANGE)
    return f"{start:%d %b} – {end:%d %b %Y}"


def validate(land: float | None, building: float | None, asking: float | None) -> list[str]:
    """Inline error messages for invalid form inputs (empty list = valid)."""
    errors = []
    if land is None or land <= 0:
        errors.append("Land area must be greater than 0 m².")
    if building is None or building <= 0:
        errors.append("Building area must be greater than 0 m².")
    if asking is not None and asking <= 0:
        errors.append("Asking price must be greater than Rp 0, or leave it empty.")
    return errors


def drivers_chart(drivers: list[dict]) -> go.Figure:
    """Horizontal bars of approximate % effect, largest at the top; blue raises, orange lowers."""
    d = list(reversed(drivers))
    x = [r["pct_effect"] for r in d]
    fig = go.Figure(go.Bar(
        x=x, y=[r["label"] for r in d], orientation="h",
        marker_color=[config.DRIVER_COLORS["up" if v >= 0 else "down"] for v in x],
        text=[f"{v:+.0f}%".replace("-", "−") for v in x], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:+.1f}%<extra></extra>",
    ))
    span = max(abs(v) for v in x) * 1.35 or 1
    fig.update_layout(
        height=60 + 44 * len(d), margin={"l": 10, "r": 10, "t": 10, "b": 40},
        xaxis={"title": "Effect on the estimate (%)", "range": [-span, span], "zeroline": True,
               "zerolinecolor": "#8a8984", "ticksuffix": "%", "gridcolor": "#e6e5e0"},
        yaxis={"automargin": True}, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False, font={"size": 13, "color": config.FIG_COLORS["text_secondary"]},
    )
    return fig


def comps_table(comps: list[dict]) -> pd.DataFrame:
    """Similar listings formatted for display."""
    rooms = lambda v: "—" if v is None else f"{v:.0f}"
    return pd.DataFrame([{
        "Kecamatan": c["kecamatan"], "Kabupaten": c["kabupaten"], "Land (m²)": f"{c['land_m2']:,.0f}",
        "Building (m²)": f"{c['building_m2']:,.0f}", "Bedrooms": rooms(c["bedrooms"]),
        "Bathrooms": rooms(c["bathrooms"]), "Price": format_idr(c["price_idr"]),
    } for c in comps])


def show_results(result: dict, asking: int | None) -> None:
    """Metric cards, warnings, drivers chart, comps table."""
    cards = st.columns(2)
    cards[0].metric("Estimate", format_idr(result["estimate_idr"]), border=True)
    cards[1].metric("Likely range", f"{format_idr(result['low_idr'])} – {format_idr(result['high_idr'])}",
                    help="10th–90th percentile: about 8 in 10 similar houses list inside this range.", border=True)
    st.caption(adjustment_caption(result))
    if asking is not None:
        diff = (asking / result["estimate_idr"] - 1) * 100
        st.columns(2)[0].metric(
            "Verdict", config.VERDICT_LABELS[result["verdict"]], border=True,
            delta=f"Asking {format_idr(asking)} ({diff:+.0f}% vs estimate)".replace("-", "−"),
            delta_color="off", delta_arrow="off")
    for w in result["warnings"]:
        st.warning(w)

    st.subheader("Why this estimate")
    st.plotly_chart(drivers_chart(result["drivers"]), width="stretch", config={"displayModeBar": False})
    st.caption("Effect of each factor compared with an average listing in the data. "
               "Blue raises the estimate, orange lowers it.")

    st.subheader("Similar listings")
    st.dataframe(comps_table(result["comps"]), hide_index=True, width="stretch")
    st.caption("Nearest listings by size, rooms and location, from the same kabupaten when it has enough data. "
               "Prices are the original 2024 asking prices.")


def valuation_page() -> None:
    """Page 1 — Valuation form and results."""
    st.title(config.APP_TITLE)
    st.write("Estimate the asking price of a house in Yogyakarta (DIY) from its location, size, rooms "
             "and listing features.")
    st.caption(disclaimer())

    d = config.APP_DEFAULT_INPUTS
    c1, c2 = st.columns(2)
    kabupaten = c1.selectbox("Kabupaten/Kota", config.KABUPATEN_ORDER,
                             index=config.KABUPATEN_ORDER.index(d["kabupaten"]), key="kabupaten")
    options = kecamatan_options(kabupaten)
    names = [o[0] for o in options]
    labels = dict(options)
    kecamatan = c2.selectbox("Kecamatan", names, format_func=labels.get,
                             index=names.index(d["kecamatan"]) if d["kecamatan"] in names else 0, key="kecamatan")

    with st.form("house"):
        a, b = st.columns(2)
        land = a.number_input("Land area (m²)", min_value=0.0, value=d["land_m2"], step=10.0, key="land_m2")
        building = b.number_input("Building area (m²)", min_value=0.0, value=d["building_m2"], step=10.0, key="building_m2")
        r1, r2, r3 = st.columns(3)
        bedrooms = r1.number_input("Bedrooms", 0, config.APP_MAX_ROOMS, d["bedrooms"])
        bathrooms = r2.number_input("Bathrooms", 0, config.APP_MAX_ROOMS, d["bathrooms"])
        carport = r3.number_input("Carport", 0, config.APP_MAX_ROOMS, d["carport"])
        st.write("Listing features")
        fcols = st.columns(2)
        flags = {f: fcols[i % 2].checkbox(config.FLAG_LABELS[f], key=f)
                 for i, f in enumerate(config.ACTIVE_FLAGS)}
        asking = st.number_input("Asking price (Rp, optional)", min_value=0, value=None, step=10_000_000,
                                 placeholder="e.g. 900000000", help="Enables the verdict.", key="asking")
        submitted = st.form_submit_button("Estimate price", type="primary")

    if submitted:
        errors = validate(land, building, asking)
        for e in errors:
            st.error(e)
        if not errors:
            inputs = {"kabupaten": kabupaten, "kecamatan": kecamatan, "land_m2": land, "building_m2": building,
                      "bedrooms": bedrooms, "bathrooms": bathrooms, "carport": carport, **flags}
            asking_int = int(asking) if asking is not None else None
            try:
                result = predict(inputs, asking_int)
            except ValueError as exc:
                st.error(f"Could not estimate this house: {exc}")
            else:
                show_results(result, asking_int)

    st.divider()
    meta = metadata()
    st.caption(f"Data: [{config.DATA_SOURCE_NAME}]({config.DATA_SOURCE_URL}), {_date_range()}; "
               f"{meta['n_rows']:,} listings after cleaning.")
    st.page_link(ABOUT_PAGE, label="How the model works and its limits →")


st.set_page_config(page_title=config.APP_TITLE, layout="wide")
artifacts()
nav = st.navigation([
    st.Page(valuation_page, title="Valuation", default=True),
    st.Page("pages/1_Market_map.py", title="Market map"),
    st.Page(ABOUT_PAGE, title="About the model"),
])
nav.run()
