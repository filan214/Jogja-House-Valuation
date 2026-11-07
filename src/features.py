"""P4a — Model features: the single source of truth for training and the app.

`build_model_features(df_listings, osm_features)` turns listings (or one row of app
form inputs) into the model's feature matrix. Training and the app both call it.

Inputs:  data/interim/listings_clean.csv, data/external/osm_kecamatan_features.csv
Output:  data/processed/model_table.csv
         = features + price_idr + log_price + kecamatan/kabupaten/listing_id (grouping)
           + price_per_land_m2 (EDA/baseline only — never a feature)
           + road_name (EDA only) and road_km / has_road_km (features only if
             config.USE_ROAD_KM_FEATURES; always in the table for the P7b U1 ablation)

Run: python -m src.features
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src import config

SIZE_FEATURES = ["land_m2", "building_m2", "log_land", "log_building", "building_to_land_ratio"]
ROOM_FEATURES = ["bedrooms", "bathrooms", "carport"]
ROAD_FEATURES = ["road_km", "has_road_km"]
NOT_FEATURES = {"price_idr", "log_price", "price_per_land_m2", "kecamatan", "kecamatan_code",
                "listing_id", "listing_url", "description", "area_raw", "kabupaten_raw",
                "road_name"}


def kabupaten_column(kab: str) -> str:
    """One-hot column name: 'Kota Yogyakarta' -> 'kab_kota_yogyakarta'."""
    return "kab_" + kab.lower().replace(" ", "_")


def location_feature_names(osm_features: pd.DataFrame) -> list[str]:
    """Location features present in this OSM table (optional BPS column only if available)."""
    opt = [c for c in config.OPTIONAL_LOCATION_FEATURES if c in osm_features.columns]
    return list(config.LOCATION_FEATURES) + opt


def feature_columns(osm_features: pd.DataFrame, road_km: bool | None = None) -> list[str]:
    """Ordered list of model feature columns (`road_km` defaults to config.USE_ROAD_KM_FEATURES)."""
    use_road = config.USE_ROAD_KM_FEATURES if road_km is None else road_km
    return (
        SIZE_FEATURES
        + ROOM_FEATURES
        + list(config.ACTIVE_FLAGS)
        + [kabupaten_column(k) for k in config.KABUPATEN_ORDER]
        + location_feature_names(osm_features)
        + (ROAD_FEATURES if use_road else [])
    )


def _road_pattern() -> re.Pattern:
    """Regex for '<road> [suffix] km <number>' from config.MAIN_ROADS + ROAD_ALIASES.

    Matches "Jl. Kaliurang km 8", "jalan magelang km. 10,5", "Jakal km 7", "Kaliurang Km9",
    "Jl Imogiri Barat Km 13". The road name must sit right before "km", so distances like
    "5 km dari UGM" never match. A "Jl."/"jalan" prefix is optional (titles often drop it).
    """
    names = sorted(list(config.MAIN_ROADS) + list(config.ROAD_ALIASES), key=len, reverse=True)
    name_alt = "|".join(re.escape(n).replace(r"\ ", r"\s*") for n in names)
    suffix_alt = "|".join(config.ROAD_NAME_SUFFIXES)
    return re.compile(rf"\b(?P<name>{name_alt})(?:\s+(?:{suffix_alt}))?\s*[,.]?\s*km\s*\.?\s*"
                      rf"(?P<km>\d+(?:[.,]\d+)?)(?!\d)", re.IGNORECASE)


ROAD_PATTERN = _road_pattern()


def _canonical_road(name: str) -> str:
    """'Jakal' -> 'kaliurang', 'Gito  Gati' -> 'gito gati'."""
    n = re.sub(r"\s+", " ", name.lower())
    n = config.ROAD_ALIASES.get(n, n)
    return "gito gati" if n == "gitogati" else n


def parse_road_km(description: pd.Series) -> pd.DataFrame:
    """First road km marker per text: road_name (canonical) and road_km (float), NaN if none.

    Comma or dot decimals ("km 8,5" = 8.5). Values outside ROAD_KM_MIN..MAX count as not parsed.
    """
    text = description.fillna("").astype(str)
    found = text.str.extract(ROAD_PATTERN)
    km = pd.to_numeric(found["km"].str.replace(",", ".", regex=False), errors="coerce")
    ok = km.between(config.ROAD_KM_MIN, config.ROAD_KM_MAX)
    return pd.DataFrame({"road_name": found["name"].where(ok).map(_canonical_road, na_action="ignore"),
                         "road_km": km.where(ok).astype(float)}, index=description.index)


def road_features(df: pd.DataFrame) -> pd.DataFrame:
    """road_name, road_km, has_road_km for listings or app inputs.

    If a `road_km` column is present (app form) it is used as given; otherwise the marker is
    parsed from `description`; with neither, road_km = NaN and has_road_km = 0.
    """
    if "road_km" in df.columns:
        km = pd.to_numeric(df["road_km"], errors="coerce").astype(float)
        km = km.where(km.between(config.ROAD_KM_MIN, config.ROAD_KM_MAX))
        name = df["road_name"] if "road_name" in df.columns else pd.Series(np.nan, index=df.index)
        road = pd.DataFrame({"road_name": name.where(km.notna()), "road_km": km}, index=df.index)
    elif "description" in df.columns:
        road = parse_road_km(df["description"])
    else:
        road = pd.DataFrame({"road_name": np.nan, "road_km": np.nan}, index=df.index)
    road["has_road_km"] = road["road_km"].notna().astype(int)
    return road


def keyword_flags(description: pd.Series) -> pd.DataFrame:
    """0/1 flags from listing text using config.KEYWORD_FLAGS (all flags, active or not)."""
    text = description.fillna("").astype(str)
    return pd.DataFrame(
        {k: text.str.contains(p, case=False, regex=True).astype(int) for k, p in config.KEYWORD_FLAGS.items()},
        index=description.index,
    )


def build_model_features(df_listings: pd.DataFrame, osm_features: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix for listings or app inputs.

    Required columns: kecamatan, kabupaten, land_m2, building_m2. Optional: bedrooms,
    bathrooms (NaN allowed), carport (missing → 0). Flags: if a `flag_*` column is
    present (app checkboxes) it is used as 0/1; otherwise it is computed from
    `description` with config.KEYWORD_FLAGS. Road km: see road_features(). Raises ValueError for an unknown
    (kecamatan, kabupaten) pair or non-positive areas. Returns columns in
    `feature_columns()` order, same index as the input.
    """
    df = df_listings.copy()
    for col in ("kecamatan", "kabupaten", "land_m2", "building_m2"):
        if col not in df.columns:
            raise ValueError(f"missing required column: {col}")
    land = pd.to_numeric(df["land_m2"], errors="coerce").astype(float)
    building = pd.to_numeric(df["building_m2"], errors="coerce").astype(float)
    if (land.isna() | (land <= 0) | building.isna() | (building <= 0)).any():
        raise ValueError("land_m2 and building_m2 must be positive numbers")

    out = pd.DataFrame(index=df.index)
    out["land_m2"] = land
    out["building_m2"] = building
    out["log_land"] = np.log(land)
    out["log_building"] = np.log(building)
    out["building_to_land_ratio"] = building / land

    for col in ("bedrooms", "bathrooms"):
        out[col] = pd.to_numeric(df[col], errors="coerce").astype(float) if col in df.columns else np.nan
    out["carport"] = (pd.to_numeric(df["carport"], errors="coerce").fillna(0).astype(float)
                      if "carport" in df.columns else 0.0)

    text_flags = keyword_flags(df["description"]) if "description" in df.columns else None
    for flag in config.ACTIVE_FLAGS:
        if flag in df.columns:
            out[flag] = df[flag].fillna(False).astype(bool).astype(int)
        elif text_flags is not None:
            out[flag] = text_flags[flag]
        else:
            out[flag] = 0

    unknown = set(df["kabupaten"]) - set(config.KABUPATEN_ORDER)
    if unknown:
        raise ValueError(f"unknown kabupaten: {sorted(unknown)}")
    for kab in config.KABUPATEN_ORDER:
        out[kabupaten_column(kab)] = (df["kabupaten"] == kab).astype(int)

    loc_cols = location_feature_names(osm_features)
    osm = osm_features[["kecamatan", "kabupaten"] + loc_cols]
    merged = df[["kecamatan", "kabupaten"]].merge(osm, on=["kecamatan", "kabupaten"], how="left",
                                                    validate="many_to_one")
    merged.index = df.index
    missing = merged[loc_cols[0]].isna()
    if missing.any():
        pairs = sorted(set(zip(df.loc[missing, "kecamatan"], df.loc[missing, "kabupaten"])))
        raise ValueError(f"no OSM features for (kecamatan, kabupaten): {pairs}")
    for c in loc_cols:
        out[c] = merged[c].astype(float)

    road = road_features(df)
    for c in ROAD_FEATURES:
        out[c] = road[c]

    cols = feature_columns(osm_features)
    assert not (set(cols) & NOT_FEATURES), "leakage: forbidden column in feature list"
    return out[cols]


def flag_share_report(listings: pd.DataFrame) -> pd.DataFrame:
    """Share of listings per keyword flag (all flags) and whether it is active."""
    flags = keyword_flags(listings["description"])
    rep = pd.DataFrame({
        "flag": flags.columns,
        "label": [config.FLAG_LABELS.get(f, f) for f in flags.columns],
        "listings": flags.sum().values,
        "share_%": (flags.mean() * 100).round(1).values,
    })
    rep["kept"] = rep["flag"].isin(config.ACTIVE_FLAGS)
    return rep


def road_km_report(listings: pd.DataFrame, n_examples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Counts per road and `n_examples` random examples (description snippet → parsed value)."""
    road = road_features(listings)
    counts = road["road_name"].value_counts().rename_axis("road").reset_index(name="listings")
    hit = road[road["has_road_km"] == 1]
    sample = hit.sample(min(n_examples, len(hit)), random_state=config.RANDOM_STATE)
    snippets = []
    for i in sample.index:
        text = str(listings.at[i, "description"])
        m = ROAD_PATTERN.search(text)
        lo, hi = max(0, m.start() - 15), min(len(text), m.end() + 10)
        snippets.append(("..." if lo else "") + text[lo:hi].strip() + ("..." if hi < len(text) else ""))
    examples = pd.DataFrame({"snippet": snippets, "road_name": sample["road_name"].values,
                             "road_km": sample["road_km"].values})
    return counts, examples


def main() -> None:
    """Build data/processed/model_table.csv and print a summary."""
    listings = pd.read_csv(config.CLEAN_CSV)
    osm = pd.read_csv(config.OSM_FEATURES_CSV)

    rep = flag_share_report(listings)
    print("Keyword flag shares:")
    for r in rep.itertuples():
        print(f"  {r.flag:<18} {r._4:>5.1f}%  ({r.listings:>3})  {'kept' if r.kept else 'DROPPED (< 2%)'}")
    should = set(rep.loc[rep["share_%"] >= config.MIN_FLAG_SHARE * 100, "flag"])
    if should != set(config.ACTIVE_FLAGS):
        print(f"  WARNING: ACTIVE_FLAGS differs from the {config.MIN_FLAG_SHARE:.0%} rule: "
              f"rule keeps {sorted(should)}")

    road = road_features(listings)
    counts, examples = road_km_report(listings, config.ROAD_KM_N_EXAMPLES)
    n_road = int(road["has_road_km"].sum())
    print(f"Road km markers: {n_road} of {len(listings):,} listings ({n_road / len(listings):.1%}); "
          f"km {road['road_km'].min():g}-{road['road_km'].max():g}, median {road['road_km'].median():g}")
    print("  " + ", ".join(f"{r.road} {r.listings}" for r in counts.itertuples()))
    for r in examples.itertuples():
        print(f"  {r.snippet!r:<70} -> {r.road_name} km {r.road_km:g}")

    X = build_model_features(listings, osm)
    code = listings[["kecamatan", "kabupaten"]].merge(
        osm[["kecamatan", "kabupaten", "kecamatan_code"]], on=["kecamatan", "kabupaten"], how="left",
        validate="many_to_one")["kecamatan_code"].astype(str)
    code.index = listings.index
    table = pd.concat(
        [
            listings[["listing_id", "kecamatan", "kabupaten"]],
            code,
            X,
            road[[c for c in ["road_name", *ROAD_FEATURES] if c not in X.columns]],
            listings[["price_idr"]],
            np.log(listings["price_idr"]).rename("log_price"),
            listings[["price_per_land_m2"]],
        ],
        axis=1,
    )
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.MODEL_TABLE, index=False, encoding="utf-8")

    cols = feature_columns(osm)
    nan = X.isna().sum()
    print(f"Model table: {len(table):,} rows, {len(cols)} features -> {config.MODEL_TABLE.relative_to(config.ROOT)}")
    print(f"  NaN per feature (non-zero): {nan[nan > 0].to_dict()}")
    print(f"  kecamatan groups (by kecamatan_code): {table['kecamatan_code'].nunique()}")
    print(f"  price_per_land_m2 in table: {'price_per_land_m2' in table.columns}, in features: "
          f"{'price_per_land_m2' in cols}; kecamatan in table: {'kecamatan' in table.columns}, "
          f"in features: {'kecamatan' in cols}")
    print(f"  road_km/has_road_km in table: True, in features: {'road_km' in cols} "
          f"(config.USE_ROAD_KM_FEATURES = {config.USE_ROAD_KM_FEATURES}); road_name in features: "
          f"{'road_name' in cols}")


if __name__ == "__main__":
    main()
