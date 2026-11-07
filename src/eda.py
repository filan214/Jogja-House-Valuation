"""P4b — EDA figures, plain-language findings, and kecamatan stats for the map page.

Inputs:  data/processed/model_table.csv, data/external/osm_kecamatan_features.csv
Outputs: reports/figures/01_…06_*.png (150 dpi), reports/eda_summary.md,
         data/app/kecamatan_stats.csv (all 78 kecamatan; n_listings = 0 where no data)

Run: python -m src.eda
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter

from src import config
from src.utils import format_idr

C = config.FIG_COLORS
JUTA = 1_000_000


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": config.FIG_FONT,
        "font.size": 10,
        "figure.facecolor": C["surface"],
        "axes.facecolor": C["surface"],
        "savefig.facecolor": C["surface"],
        "axes.edgecolor": C["axis"],
        "axes.linewidth": 0.8,
        "axes.labelcolor": C["text_secondary"],
        "axes.titlecolor": C["text"],
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": C["grid"],
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": C["muted"],
        "ytick.color": C["muted"],
        "xtick.labelcolor": C["text_secondary"],
        "ytick.labelcolor": C["text_secondary"],
        "legend.frameon": False,
        "legend.labelcolor": C["text_secondary"],
    })


def _save(fig: plt.Figure, name: str) -> str:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return name


def _idr_ticks(x: float, _pos: int) -> str:
    return format_idr(int(x)).replace("Rp ", "")


def _caption(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, -0.02, text, ha="left", va="top", fontsize=8, color=C["muted"])


def fig_price_distribution(t: pd.DataFrame) -> str:
    """01 — histogram of asking price on a log x-axis."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.logspace(np.log10(t["price_idr"].min()), np.log10(t["price_idr"].max()), 40)
    ax.hist(t["price_idr"], bins=bins, color=C["series_1"], edgecolor=C["surface"], linewidth=1)
    ax.set_xscale("log")
    ticks = [2e8, 5e8, 1e9, 2e9, 5e9, 1e10]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(_idr_ticks))
    ax.xaxis.set_minor_formatter(NullFormatter())
    med = t["price_idr"].median()
    ax.axvline(med, color=C["text"], linewidth=1)
    ax.annotate(f"Median {format_idr(int(med))}", (med, ax.get_ylim()[1] * 0.92), xytext=(6, 0),
                textcoords="offset points", fontsize=9, color=C["text"])
    ax.set_title(f"Asking price distribution ({len(t):,} listings)")
    ax.set_xlabel("Asking price (Rp, log scale)")
    ax.set_ylabel("Listings")
    ax.grid(axis="x", visible=False)
    _caption(fig, "Source: Rumah123 listings, Mar–May 2024, after cleaning.")
    return _save(fig, "01_price_distribution.png")


def fig_ppl_by_kabupaten(t: pd.DataFrame) -> str:
    """02 — boxplot of price per land m² by kabupaten (sorted by median)."""
    order = t.groupby("kabupaten")["price_per_land_m2"].median().sort_values().index.tolist()
    data = [t.loc[t["kabupaten"] == k, "price_per_land_m2"] / JUTA for k in order]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bp = ax.boxplot(data, orientation="horizontal", widths=0.5, patch_artist=True, showfliers=True,
                    medianprops={"color": C["text"], "linewidth": 1.5},
                    boxprops={"facecolor": C["series_1"], "edgecolor": C["series_1"], "alpha": 0.85},
                    whiskerprops={"color": C["muted"]}, capprops={"color": C["muted"]},
                    flierprops={"marker": "o", "markersize": 3, "markerfacecolor": C["context"],
                                "markeredgecolor": "none"})
    del bp
    ax.set_yticks(range(1, len(order) + 1),
                  [f"{k}  (n={int((t['kabupaten'] == k).sum())})" for k in order])
    ax.set_title("Price per m² of land by kabupaten")
    ax.set_xlabel("Asking price per m² of land (Rp juta)")
    ax.grid(axis="y", visible=False)
    _caption(fig, "Box = middle 50% of listings, line = median, dots = outliers. Sorted by median.")
    return _save(fig, "02_price_per_m2_by_kabupaten.png")


def kecamatan_table(t: pd.DataFrame) -> pd.DataFrame:
    """Per-kecamatan counts and medians (listings only)."""
    return (t.groupby(["kecamatan", "kabupaten"])
             .agg(n_listings=("price_idr", "size"), median_price=("price_idr", "median"),
                  median_price_per_land_m2=("price_per_land_m2", "median"))
             .reset_index())


def fig_top_bottom_kecamatan(t: pd.DataFrame) -> str:
    """03 — median price per land m², top and bottom N kecamatan with ≥ MIN_LISTINGS_FOR_MAP listings."""
    k = kecamatan_table(t)
    k = k[k["n_listings"] >= config.MIN_LISTINGS_FOR_MAP].sort_values("median_price_per_land_m2")
    n = config.EDA_TOP_N_KECAMATAN
    top, bottom = k.tail(n), k.head(n)
    xmax = k["median_price_per_land_m2"].max() / JUTA * 1.15
    fig, axes = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True)
    for ax, part, title in ((axes[0], top, f"{n} most expensive"), (axes[1], bottom, f"{n} least expensive")):
        labels = [f"{r.kecamatan} ({r.kabupaten}) · n={r.n_listings}" for r in part.itertuples()]
        ax.barh(labels, part["median_price_per_land_m2"] / JUTA, color=C["series_1"], height=0.7)
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, xmax)
    axes[1].set_xlabel("Median asking price per m² of land (Rp juta)")
    fig.suptitle(f"Kecamatan by median price per m² of land (≥ {config.MIN_LISTINGS_FOR_MAP} listings)",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold", color=C["text"])
    fig.tight_layout()
    _caption(fig, f"{len(k)} kecamatan have ≥ {config.MIN_LISTINGS_FOR_MAP} listings; "
                  "n = listings in that kecamatan.")
    return _save(fig, "03_top_bottom_kecamatan.png")


def fig_land_vs_price(t: pd.DataFrame) -> str:
    """04 — log-log land vs price, one small multiple per kabupaten (highlight over gray context)."""
    order = list(config.KABUPATEN_ORDER)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8), sharex=True, sharey=True)
    flat = axes.ravel()
    for ax, kab in zip(flat, order):
        sel = t["kabupaten"] == kab
        ax.scatter(t.loc[~sel, "land_m2"], t.loc[~sel, "price_idr"], s=6, color=C["context"],
                   alpha=0.5, linewidths=0)
        ax.scatter(t.loc[sel, "land_m2"], t.loc[sel, "price_idr"], s=12, color=C["series_1"],
                   alpha=0.8, edgecolors=C["surface"], linewidths=0.4)
        ax.set_title(f"{kab} (n={int(sel.sum())})", fontsize=10.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax_note = flat[len(order)]
    ax_note.axis("off")
    ax_note.text(0.05, 0.6, "Blue: listings in that kabupaten\nGray: all other listings, for context",
                 fontsize=9.5, color=C["text_secondary"], va="top", transform=ax_note.transAxes)
    for ax in flat[: len(order)]:
        ax.set_yticks([3e8, 1e9, 3e9, 1e10])
        ax.yaxis.set_major_formatter(FuncFormatter(_idr_ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:,.0f}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
    for ax in axes[1]:
        ax.set_xlabel("Land area (m², log scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Asking price (Rp, log scale)")
    axes[0, 2].tick_params(labelbottom=True)
    fig.suptitle("Land area vs asking price", x=0.01, ha="left", fontsize=12, fontweight="semibold",
                 color=C["text"])
    fig.tight_layout()
    return _save(fig, "04_land_vs_price.png")


def flag_effects(t: pd.DataFrame) -> pd.DataFrame:
    """Median price and building size with vs without each active flag."""
    rows = []
    for f in config.ACTIVE_FLAGS:
        on = t[f] == 1
        rows.append({"flag": f, "label": config.FLAG_LABELS[f], "share": on.mean(), "n_with": int(on.sum()),
                     "median_with": t.loc[on, "price_idr"].median(), "median_without": t.loc[~on, "price_idr"].median(),
                     "building_with": t.loc[on, "building_m2"].median(),
                     "building_without": t.loc[~on, "building_m2"].median()})
    return pd.DataFrame(rows).sort_values("median_with")


def fig_flag_effects(t: pd.DataFrame) -> str:
    """05 — dot plot: median price with vs without each keyword."""
    fe = flag_effects(t)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(fe))
    for yi, r in zip(y, fe.itertuples()):
        ax.plot([r.median_without / 1e9, r.median_with / 1e9], [yi, yi], color=C["axis"], linewidth=2, zorder=1)
    ax.scatter(fe["median_without"] / 1e9, y, s=60, color=C["series_1"], label="Without keyword", zorder=2,
               edgecolors=C["surface"], linewidths=1.5)
    ax.scatter(fe["median_with"] / 1e9, y, s=60, color=C["series_2"], label="With keyword", zorder=3,
               edgecolors=C["surface"], linewidths=1.5)
    ax.set_yticks(y, [f"{r.label}  ({r.share:.0%})" for r in fe.itertuples()])
    ax.set_xlabel("Median asking price (Rp miliar)")
    ax.set_title("Median asking price with vs without each listing keyword")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=9)
    _caption(fig, "Share of listings with the keyword in brackets. Differences mostly reflect house size "
                  "and type, not the keyword itself.")
    return _save(fig, "05_flag_effects.png")


def fig_listings_per_kecamatan(t: pd.DataFrame) -> str:
    """06 — listing counts per kecamatan (shows data thinness)."""
    k = kecamatan_table(t).sort_values("n_listings")
    low = k["n_listings"] < config.LOW_DATA_KECAMATAN_THRESHOLD
    fig, ax = plt.subplots(figsize=(8, 9.5))
    labels = [f"{r.kecamatan} ({r.kabupaten})" for r in k.itertuples()]
    ax.barh(labels, k["n_listings"], color=np.where(low, C["context"], C["series_1"]), height=0.7)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:,.0f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    for thr in (config.MIN_LISTINGS_FOR_MAP, config.LOW_DATA_KECAMATAN_THRESHOLD):
        ax.axvline(thr, color=C["text_secondary"], linewidth=0.8)
    for i, v in enumerate(k["n_listings"]):
        ax.annotate(str(v), (v, i), xytext=(3, 0), textcoords="offset points", fontsize=7.5,
                    color=C["text_secondary"], va="center")
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title(f"Listings per kecamatan ({len(k)} of {config.N_KECAMATAN_DIY} kecamatan have data)")
    ax.set_xlabel("Listings (log scale)")
    ax.grid(axis="y", visible=False)
    ax.margins(y=0.01)
    fig.tight_layout()
    _caption(fig, f"Gray: fewer than {config.LOW_DATA_KECAMATAN_THRESHOLD} listings ('limited data' in the app). "
                  f"Vertical lines: {config.MIN_LISTINGS_FOR_MAP} listings (minimum to color the map) and "
                  f"{config.LOW_DATA_KECAMATAN_THRESHOLD}.")
    return _save(fig, "06_listings_per_kecamatan.png")


def findings(t: pd.DataFrame) -> list[str]:
    """Five plain-language findings, computed from the data."""
    kab = t.groupby("kabupaten").agg(n=("price_idr", "size"), ppl=("price_per_land_m2", "median"),
                                     price=("price_idr", "median"))
    kota, bantul, kp = kab.loc["Kota Yogyakarta"], kab.loc["Bantul"], kab.loc["Kulon Progo"]

    bands = pd.cut(t["dist_center_km"], list(config.DIST_BANDS_KM))
    by_band = t.groupby(bands, observed=True)["price_per_land_m2"].agg(["median", "size"])
    near, far = by_band.iloc[0], by_band.iloc[-1]
    b0, b_last = config.DIST_BANDS_KM[1], config.DIST_BANDS_KM[-2]

    slope = np.polyfit(t["log_land"], t["log_price"], 1)[0]
    fe = flag_effects(t).set_index("flag")
    shm, pool = fe.loc["flag_shm"], fe.loc["flag_pool"]

    vc = t.groupby(["kecamatan", "kabupaten"]).size().sort_values(ascending=False)
    top3 = vc.head(3)
    sleman_share = (t["kabupaten"] == "Sleman").mean()
    thin = int((vc < config.LOW_DATA_KECAMATAN_THRESHOLD).sum())

    return [
        f"**Land in the city costs the most.** A square meter of land (with its house) in Kota Yogyakarta "
        f"is listed at a median of {format_idr(int(kota.ppl))}, about {kota.ppl / bantul.ppl:.1f}× Bantul "
        f"({format_idr(int(bantul.ppl))}) and {kota.ppl / kp.ppl:.1f}× Kulon Progo ({format_idr(int(kp.ppl))}).",
        f"**Prices fall with distance from the center.** Within {b0} km of Tugu the median is "
        f"{format_idr(int(near['median']))} per m² of land; more than {b_last} km away it is "
        f"{format_idr(int(far['median']))} — about {near['median'] / far['median']:.1f}× less.",
        f"**Price grows roughly in step with land size.** Across all listings, a house on twice the land is "
        f"listed for about {2 ** slope:.1f}× the price. Land size is the strongest single clue to price.",
        f"**Listing keywords mostly describe the type of house, not extra value.** Listings mentioning an SHM "
        f"certificate have a lower median price ({format_idr(int(shm.median_with))} vs "
        f"{format_idr(int(shm.median_without))}) because they are smaller houses "
        f"({shm.building_with:.0f} vs {shm.building_without:.0f} m² of building). Listings with a pool are "
        f"{pool.median_with / pool.median_without:.1f}× pricier, but also much bigger.",
        f"**The data is concentrated in a few places.** {sleman_share:.0%} of listings are in Sleman, and "
        f"three kecamatan ({', '.join(k for k, _ in top3.index)}) hold {top3.sum() / len(t):.0%} of them. "
        f"{thin} of the {len(vc)} kecamatan with data have fewer than "
        f"{config.LOW_DATA_KECAMATAN_THRESHOLD} listings, and Gunungkidul has only "
        f"{int(kab.loc['Gunungkidul', 'n'])} — estimates there will be rough.",
    ]


def kecamatan_stats(t: pd.DataFrame, osm: pd.DataFrame) -> pd.DataFrame:
    """Stats for all 78 kecamatan (map page); kecamatan without listings get n_listings = 0."""
    k = kecamatan_table(t)
    out = osm[["kecamatan_code", "kecamatan", "kabupaten"]].merge(k, on=["kecamatan", "kabupaten"], how="left")
    out["n_listings"] = out["n_listings"].fillna(0).astype(int)
    out["median_price"] = out["median_price"].round(0)
    out["median_price_per_land_m2"] = out["median_price_per_land_m2"].round(0)
    return out.sort_values("kecamatan_code").reset_index(drop=True)


def main() -> None:
    """Write the six figures, the EDA summary and kecamatan stats."""
    _style()
    t = pd.read_csv(config.MODEL_TABLE)
    osm = pd.read_csv(config.OSM_FEATURES_CSV, dtype={"kecamatan_code": str})

    figs = [fig_price_distribution(t), fig_ppl_by_kabupaten(t), fig_top_bottom_kecamatan(t),
            fig_land_vs_price(t), fig_flag_effects(t), fig_listings_per_kecamatan(t)]

    items = findings(t)
    md = ("# EDA summary\n\nGenerated by `python -m src.eda` from "
          f"`{config.MODEL_TABLE.relative_to(config.ROOT).as_posix()}` ({len(t):,} listings). "
          "Prices are 2024 asking prices, not sale prices.\n\n## Five findings\n\n"
          + "\n".join(f"{i}. {s}" for i, s in enumerate(items, 1))
          + "\n\n## Figures\n\n" + "\n".join(f"- `reports/figures/{f}`" for f in figs) + "\n")
    config.EDA_SUMMARY_MD.write_text(md, encoding="utf-8")

    stats = kecamatan_stats(t, osm)
    config.APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_csv(config.KECAMATAN_STATS_CSV, index=False, encoding="utf-8")

    print(f"Figures: {', '.join(figs)}")
    print(f"Wrote {config.EDA_SUMMARY_MD.relative_to(config.ROOT)} and "
          f"{config.KECAMATAN_STATS_CSV.relative_to(config.ROOT)} ({len(stats)} kecamatan, "
          f"{int((stats['n_listings'] > 0).sum())} with listings, "
          f"{int((stats['n_listings'] >= config.MIN_LISTINGS_FOR_MAP).sum())} with ≥ {config.MIN_LISTINGS_FOR_MAP})")
    for i, s in enumerate(items, 1):
        print(f"  {i}. {s.replace('**', '')}")


if __name__ == "__main__":
    main()
