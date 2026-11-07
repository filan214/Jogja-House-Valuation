"""Page 3 — About the model: data, cleaning, metrics, interval coverage, limitations."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.app_data import cleaning_summary, disclaimer, drift_summary, limitations, metadata, month_label
from src.predict import price_adjustment
from src.utils import format_idr


def _height(n_rows: int) -> int:
    """Dataframe height that shows every row without an inner scroll bar."""
    return 38 + 35 * n_rows

meta = metadata()
cv = meta["cv_metrics"]
q = meta["quantile"]
start, end = (pd.Timestamp(d) for d in meta["data_date_range"])

st.title("About the model")
st.write("How the estimate is made, how accurate it is on areas it has not seen, and where it is weak.")

st.header("Data")
st.markdown(
    f"- **Source:** [{config.DATA_SOURCE_NAME}]({config.DATA_SOURCE_URL})\n"
    f"- **Listing dates:** {start:%d %B} – {end:%d %B %Y}\n"
    f"- **License:** {config.DATA_LICENSE_NOTE}\n"
    f"- **Used for the model:** {meta['n_rows']:,} houses in {meta['n_kecamatan']} kecamatan, "
    f"located by kecamatan (not by street). Location features come from OpenStreetMap.\n"
    f"- **Model trained:** {meta['trained_on']}"
)

st.header("Cleaning")
cs = cleaning_summary()
n_raw = int(cs.loc[cs["drop_reason"] == "_raw", "rows"].iloc[0])
n_clean = int(cs.loc[cs["drop_reason"] == "_clean", "rows"].iloc[0])
st.write(f"**{n_raw:,} raw listings → {n_clean:,} used** ({n_clean / n_raw:.0%}). Every removed row has a reason:")
reasons = cs[~cs["drop_reason"].str.startswith("_")]
st.dataframe(reasons.rename(columns={"label": "Reason", "rows": "Rows", "share_of_raw_pct": "% of raw"})
             [["Reason", "Rows", "% of raw"]], hide_index=True, width="stretch", height=_height(len(reasons)),
             column_config={"% of raw": st.column_config.NumberColumn(format="%.1f")})

st.header("Accuracy")
st.write("Errors are measured with **spatial cross-validation**: the model is tested on whole kecamatan it never "
         "saw in training, the honest setting for an app that prices houses in thinly covered areas. "
         "Random cross-validation is shown for comparison.")
rows = []
for label, name in (("Baseline (median price per land m² × land)", "B0 baseline"),
                    ("Ridge regression", meta["chosen"]["ridge"]["name"]),
                    ("LightGBM (used by the app)", meta["chosen"]["lgbm"]["name"])):
    s, r = cv[name]["spatial"], cv[name]["random"]
    rows.append({"Model": label, "Spatial MAPE": f"{s['mape_mean']:.1f}%", "Spatial MdAPE": f"{s['mdape_mean']:.1f}%",
                 "Spatial MAE": format_idr(int(s["mae_mean"])), "Spatial R² (log)": f"{s['r2_log_mean']:.2f}",
                 "Random MAPE": f"{r['mape_mean']:.1f}%"})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
lg = cv[meta["chosen"]["lgbm"]["name"]]["spatial"]
st.caption(f"MAPE = mean absolute percentage error; MdAPE = median, the typical listing: half of all houses are "
           f"estimated within about {lg['mdape_mean']:.0f}% of their asking price. MAE = mean absolute error.")

st.subheader("Price range")
sp, rs = q["cv"]["spatial"], meta["range_scales"]
after, nested = rs["spatial_after"], rs["nested_check"]
c1, c2, c3 = st.columns(3)
c1.metric("Coverage of the likely range", f"{after['coverage_all']:.0f}%",
          help=f"Target {config.COVERAGE_TARGET[0]:.0%}–{config.COVERAGE_TARGET[1]:.0%}.")
c2.metric("Honest check", f"{nested['coverage_mean']:.0f}%",
          help="Widening chosen on 4 folds, tested on the 5th.")
c3.metric("Typical width", f"{after['width_median']:.0f}% of estimate")
st.write(f"The likely range is the 10th–90th percentile from LightGBM quantile models. Raw quantile ranges held "
         f"only {sp['raw_coverage_all']:.0f}% of true prices under spatial cross-validation, so they are widened "
         f"(in log space) to reach about {rs['target_coverage']:.0%}: each kabupaten with at least "
         f"{rs['min_rows_for_kabupaten']} listings gets its own widening, the others use the overall factor of "
         f"{rs['global_scale']:g}.")
kab = pd.DataFrame(rs["table"])
thin = " and ".join(kab.loc[kab["source"] == "global", "kabupaten"])
kab = pd.DataFrame({
    "Kabupaten": kab["kabupaten"], "Listings": kab["n"],
    "Widening": [f"{s:.2f}" + (" (overall)" if src == "global" else "") for s, src in zip(kab["scale"], kab["source"])],
    "Coverage %": kab["coverage_after"].round(0), "Honest check %": kab["kabupaten"].map(
        nested["coverage_by_kabupaten"]).round(0), "Mean width %": kab["width_after"].round(0),
})
st.dataframe(kab, hide_index=True, width="stretch", height=_height(len(kab)))
st.caption("Coverage is measured on the same predictions used to choose the widening, so it is optimistic; the "
           "honest check chooses on 4 folds and tests on the 5th."
           + (f" {thin} have too few listings to judge." if thin else ""))

st.subheader("Market adjustment")
adj = price_adjustment()
st.write(
    "The listings are from March–May 2024, so every estimate and range is multiplied by one factor that moves "
    "2024 prices to today's market level. "
    + (f"The factor is {adj['factor']:g} ({(adj['factor'] - 1) * 100:+.0f}%), taken from {adj['source']} "
       f"(as of {month_label(adj['as_of'])}); it is set by hand from published reports and never fitted to data. "
       if adj else
       "No factor is set yet, so prices are shown at 2024 levels; when set, it comes by hand from published "
       "Rumah123 Yogyakarta reports and is never fitted to data. ")
    + "The index is city-wide, so differences in local growth are not captured: areas that rose faster than "
      "average are still underestimated, and slower ones slightly overestimated."
)

st.subheader("Check on 2026 listings")
drift = drift_summary()
if drift is None:
    st.caption(f"Not filled yet: {config.DRIFT_TARGET_ROWS[0]}–{config.DRIFT_TARGET_ROWS[1]} current listings, "
               "copied from public listing pages, will be scored here to show how far 2024-based estimates are "
               "from today's asking prices.")
else:
    pct = lambda v, signed=False: (f"{round(v):+d}%" if signed else f"{v:.0f}%").replace("-", "−")
    st.write(f"{drift['n_rows']} current listings seen {drift['date_seen'][0]} to {drift['date_seen'][1]}, "
             "copied from public Rumah123 listing pages and scored with the model. They are only used for this "
             "check, never to set the adjustment. Negative signed error = estimate below the asking price.")
    st.dataframe(pd.DataFrame([{
        "Prices": r["prices"], "Listings": r["n"], "MAPE": pct(r["mape"]), "MdAPE": pct(r["mdape"]),
        "Range coverage": pct(r["coverage"]), "Mean signed error": pct(r["mean_signed_error"], True),
        "Median signed error": pct(r["median_signed_error"], True),
    } for r in drift["overall"]]), hide_index=True, width="stretch")
    st.write("By asking price (adjusted):")
    st.dataframe(pd.DataFrame([{
        "Asking price": r["band"], "Listings": r["n"], "MAPE": pct(r["mape"]),
        "Median signed error": pct(r["median_signed_error"], True), "Range coverage": pct(r["coverage"]),
    } for r in drift["by_price_band"]]), hide_index=True, width="stretch", height=_height(len(drift["by_price_band"])))
    st.write("By kabupaten (adjusted):")
    st.dataframe(pd.DataFrame([{
        "Kabupaten": r["kabupaten"], "Listings": r["n"], "MAPE (adjusted)": pct(r["mape_adjusted"]),
        "Range coverage (adjusted)": pct(r["coverage_adjusted"]),
        "Mean signed error (adjusted)": pct(r["mean_signed_error_adjusted"], True),
    } for r in drift["by_kabupaten"]]), hide_index=True, width="stretch", height=_height(len(drift["by_kabupaten"])))
    st.caption(" ".join(drift["findings"] + [f"Kabupaten with fewer than {config.DRIFT_SMALL_KABUPATEN_N} listings "
                                             "say almost nothing on their own."]))

st.header("Figures")
for name, caption, max_px in config.APP_FIGURES:
    col = st.columns([max_px, max(1, config.APP_FIGURE_FRAME_PX - max_px)])[0]
    col.image(str(config.APP_FIGURES_DIR / name), caption=caption, width="stretch")

st.header("Limitations")
st.markdown("\n".join(f"{i}. **{head}** {body}" for i, (head, body) in enumerate(limitations(), 1)))
st.info(disclaimer())
st.markdown(f"Code and full method: [GitHub — filan214/Jogja-_House-_Valuation]({config.GITHUB_URL})")
