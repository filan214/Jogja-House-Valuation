"""P7 — Bundle the files the About page needs into data/app/ (offline step).

Writes data/app/cleaning_summary.csv (one row per drop reason, from
data/interim/listings_dropped.csv, plus raw/clean totals) and copies the
config.APP_FIGURES PNGs from reports/figures/ into data/app/figures/, so the app
reads only models/ and data/app/.

Run: python -m src.app_assets   (after src.explain, which draws figure 09)
"""

from __future__ import annotations

import shutil

import pandas as pd

from src import config


def cleaning_summary(dropped: pd.DataFrame, n_clean: int) -> pd.DataFrame:
    """Rows per drop reason (largest first) with a plain-English label and share of raw rows."""
    n_raw = len(dropped) + n_clean
    s = dropped["drop_reason"].value_counts().rename_axis("drop_reason").reset_index(name="rows")
    unknown = set(s["drop_reason"]) - set(config.DROP_REASON_LABELS)
    if unknown:
        raise KeyError(f"no DROP_REASON_LABELS entry for {sorted(unknown)}")
    s["label"] = s["drop_reason"].map(config.DROP_REASON_LABELS)
    s["share_of_raw_pct"] = (s["rows"] / n_raw * 100).round(1)
    totals = pd.DataFrame([{"drop_reason": "_raw", "rows": n_raw, "label": "Raw listings", "share_of_raw_pct": 100.0},
                           {"drop_reason": "_clean", "rows": n_clean, "label": "Listings used by the model",
                            "share_of_raw_pct": round(n_clean / n_raw * 100, 1)}])
    return pd.concat([totals, s], ignore_index=True)


def main() -> None:
    """Write cleaning_summary.csv and copy the About-page figures."""
    dropped = pd.read_csv(config.DROPPED_CSV, usecols=["drop_reason"])
    n_clean = len(pd.read_csv(config.MODEL_TABLE, usecols=["listing_id"]))
    summary = cleaning_summary(dropped, n_clean)
    config.APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.CLEANING_SUMMARY_CSV, index=False, encoding="utf-8")
    print(f"Cleaning summary: {summary.iloc[0]['rows']:,} raw -> {n_clean:,} used, "
          f"{len(summary) - 2} drop reasons -> {config.CLEANING_SUMMARY_CSV.relative_to(config.ROOT)}")

    config.APP_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for name, *_ in config.APP_FIGURES:
        shutil.copyfile(config.FIGURES_DIR / name, config.APP_FIGURES_DIR / name)
    print(f"Copied {len(config.APP_FIGURES)} figures -> {config.APP_FIGURES_DIR.relative_to(config.ROOT)}: "
          f"{', '.join(n for n, *_ in config.APP_FIGURES)}")


if __name__ == "__main__":
    main()
