"""P6c — Comparable listings: k nearest clean listings, same kabupaten first.

Distance = Euclidean on standardized [log_land, log_building, bedrooms, bathrooms,
centroid_lat, centroid_lon] (config.COMPS_DISTANCE_FEATURES). Standardization uses
the whole pool. Missing bedrooms/bathrooms (pool or query) are filled with pool
medians for the distance only; returned rows keep their original values.
Candidates come from the query's kabupaten; if it has fewer than k listings, the
full pool is used. App-safe: numpy + pandas only.

Inputs:  data/interim/listings_clean.csv, data/external/osm_kecamatan_features.csv
Outputs: data/app/comps_pool.csv, data/app/osm_kecamatan_features.csv (copy, so the app
         needs only models/ and data/app/)

Run: python -m src.comps
"""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from src import config

ROOM_COLUMNS = ("bedrooms", "bathrooms")


def build_pool(listings: pd.DataFrame, osm_features: pd.DataFrame) -> pd.DataFrame:
    """Clean listings + kecamatan centroid lat/lon, columns config.COMPS_POOL_COLUMNS."""
    cent = osm_features[["kecamatan", "kabupaten", "centroid_lat", "centroid_lon"]]
    pool = listings.merge(cent, on=["kecamatan", "kabupaten"], how="left", validate="many_to_one")
    if pool["centroid_lat"].isna().any():
        raise ValueError("listings without a kecamatan centroid")
    return pool[list(config.COMPS_POOL_COLUMNS)].reset_index(drop=True)


def _distance_matrix(df: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["log_land"] = np.log(df["land_m2"].astype(float))
    out["log_building"] = np.log(df["building_m2"].astype(float))
    for c in ROOM_COLUMNS:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype(float).fillna(medians[c])
    out["centroid_lat"] = df["centroid_lat"].astype(float)
    out["centroid_lon"] = df["centroid_lon"].astype(float)
    return out[list(config.COMPS_DISTANCE_FEATURES)]


def find_comps(query: dict, k: int = config.N_COMPS, pool: pd.DataFrame | None = None,
               centroids: pd.DataFrame | None = None) -> pd.DataFrame:
    """The k listings most similar to `query`, nearest first, with a `distance` column.

    query keys: kabupaten, kecamatan, land_m2, building_m2, bedrooms, bathrooms
    (rooms may be None). The query's location is its kecamatan centroid, looked up in
    `centroids` (kecamatan, kabupaten, centroid_lat, centroid_lon); default: the app
    copy of the OSM features. `pool` default: data/app/comps_pool.csv.
    """
    if pool is None:
        pool = pd.read_csv(config.COMPS_POOL_CSV)
    if centroids is None:
        centroids = pd.read_csv(config.APP_OSM_FEATURES_CSV)
    hit = centroids[(centroids["kecamatan"] == query["kecamatan"]) & (centroids["kabupaten"] == query["kabupaten"])]
    if hit.empty:
        raise ValueError(f"unknown (kecamatan, kabupaten): ({query['kecamatan']}, {query['kabupaten']})")
    q = pd.DataFrame([{**{c: query.get(c) for c in ("land_m2", "building_m2", *ROOM_COLUMNS)},
                       "centroid_lat": hit["centroid_lat"].iloc[0], "centroid_lon": hit["centroid_lon"].iloc[0]}])

    medians = pool[list(ROOM_COLUMNS)].median()
    P = _distance_matrix(pool, medians)
    mu, sd = P.mean(), P.std(ddof=0).replace(0, 1.0)
    Pz = (P - mu) / sd
    qz = (_distance_matrix(q, medians) - mu) / sd

    same = pool["kabupaten"] == query["kabupaten"]
    cand = same if same.sum() >= k else pd.Series(True, index=pool.index)
    dist = np.sqrt(((Pz[cand] - qz.iloc[0]) ** 2).sum(axis=1))
    idx = dist.sort_values(kind="stable").index[:k]
    return pool.loc[idx].assign(distance=dist.loc[idx].round(3)).reset_index(drop=True)


def main() -> None:
    """Build data/app/comps_pool.csv and print an example query."""
    listings = pd.read_csv(config.CLEAN_CSV)
    osm = pd.read_csv(config.OSM_FEATURES_CSV)
    pool = build_pool(listings, osm)
    config.APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pool.to_csv(config.COMPS_POOL_CSV, index=False, encoding="utf-8")
    shutil.copyfile(config.OSM_FEATURES_CSV, config.APP_OSM_FEATURES_CSV)
    print(f"Copied {config.OSM_FEATURES_CSV.name} -> {config.APP_OSM_FEATURES_CSV.relative_to(config.ROOT)}")
    print(f"Comps pool: {len(pool):,} listings, {pool.groupby(['kecamatan', 'kabupaten']).ngroups} kecamatan "
          f"-> {config.COMPS_POOL_CSV.relative_to(config.ROOT)}")
    print(f"  per kabupaten: {pool['kabupaten'].value_counts().to_dict()}")
    query = {"kabupaten": "Sleman", "kecamatan": "Depok", "land_m2": 120, "building_m2": 90,
             "bedrooms": 3, "bathrooms": 2}
    print(f"Example comps for {query}:")
    print(find_comps(query, pool=pool, centroids=osm).drop(columns=["centroid_lat", "centroid_lon"]).to_string())


if __name__ == "__main__":
    main()
