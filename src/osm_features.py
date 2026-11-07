"""P3b — Kecamatan-level location features from OpenStreetMap.

Inputs:  OSM via osmnx (downloaded once; raw layers cached in cache/*.gpkg)
         data/external/kecamatan_diy.csv, data/interim/listings_clean.csv
         data/external/bps_population.csv (optional)
Outputs: data/external/osm_kecamatan_features.csv (one row per kecamatan, all 78)
         data/app/kecamatan.geojson (simplified polygons: kecamatan, kabupaten, kecamatan_code)

Run: python -m src.osm_features            (uses cache if present)
     python -m src.osm_features --refresh  (re-download from OSM)
"""

from __future__ import annotations

import argparse
import difflib
import time
import unicodedata

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import Point

from src import config
from src.location import match_key
from src.utils import normalize_name

WGS84 = config.WGS84_CRS


def _setup_osmnx() -> None:
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(config.OSM_CACHE_DIR)
    ox.settings.requests_timeout = config.OSM_REQUEST_TIMEOUT_S
    ox.settings.log_console = False


def _with_retry(fn, what: str):
    """Call fn() across config.OVERPASS_URLS with exponential backoff. Raises the last error."""
    last: Exception | None = None
    for url in config.OVERPASS_URLS:
        ox.settings.overpass_url = url
        wait = config.OVERPASS_BACKOFF_S
        for attempt in range(1, config.OVERPASS_RETRIES_PER_URL + 1):
            try:
                print(f"  {what}: {url} (attempt {attempt})")
                return fn()
            except Exception as e:
                last = e
                print(f"    failed: {str(e).splitlines()[0][:120]}")
                if attempt < config.OVERPASS_RETRIES_PER_URL:
                    time.sleep(wait)
                    wait *= 2
    raise RuntimeError(f"All Overpass servers failed for {what}") from last


def _keep(gdf: gpd.GeoDataFrame, cols: list[str]) -> gpd.GeoDataFrame:
    gdf = gdf.reset_index()
    for c in cols:
        if c not in gdf.columns:
            gdf[c] = pd.NA
    out = gdf[cols + ["geometry"]].copy()
    for c in cols:
        out[c] = out[c].astype("string")
    return gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs)


def download_osm(refresh: bool = False) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Return (diy, kabupaten, kecamatan, pois) layers in WGS84, from cache unless refresh."""
    config.OSM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    gp, pp = config.OSM_BOUNDARIES_GPKG, config.OSM_POIS_GPKG
    if not refresh and gp.exists() and pp.exists():
        print(f"Using cached OSM layers: {gp.name}, {pp.name}")
        return (gpd.read_file(gp, layer="diy"), gpd.read_file(gp, layer="kabupaten"),
                gpd.read_file(gp, layer="kecamatan"), gpd.read_file(pp, layer="pois"))

    _setup_osmnx()
    print("Downloading from OSM…")
    diy = ox.geocode_to_gdf(config.OSM_PLACE)[["display_name", "osm_id", "geometry"]]
    poly = diy.geometry.iloc[0]

    def admin(level: str) -> gpd.GeoDataFrame:
        g = _with_retry(lambda: ox.features_from_polygon(
            poly, tags={"boundary": "administrative", "admin_level": level}), f"admin_level={level}")
        g = g[g.geom_type.isin(["Polygon", "MultiPolygon"])]
        return _keep(g, ["element", "id", "name", "admin_level"])

    kab = admin(config.OSM_KABUPATEN_ADMIN_LEVEL)
    kec = admin(config.OSM_KECAMATAN_ADMIN_LEVEL)

    tags: dict[str, list[str]] = {}
    for spec in config.POI_TAGS.values():
        for k, vals in spec.items():
            tags.setdefault(k, []).extend(v for v in vals if v not in tags.get(k, []))
    pois = _with_retry(lambda: ox.features_from_polygon(poly, tags=tags), "POIs")
    pois = _keep(pois, ["element", "id", "name", "amenity", "shop"])
    pois["geometry"] = pois.to_crs(config.UTM_CRS).geometry.representative_point().to_crs(WGS84)

    diy.to_file(gp, layer="diy", driver="GPKG")
    kab.to_file(gp, layer="kabupaten", driver="GPKG")
    kec.to_file(gp, layer="kecamatan", driver="GPKG")
    pois.to_file(pp, layer="pois", driver="GPKG")
    return diy, kab, kec, pois


def verify_ref_points(refresh: bool = False) -> pd.DataFrame:
    """Geocode each reference point on OSM (cached) and report its offset from config.REF_POINTS."""
    path = config.OSM_REF_POINTS_CSV
    if path.exists() and not refresh:
        osm = pd.read_csv(path)
    else:
        _setup_osmnx()
        rows = []
        for key, q in config.REF_POINT_QUERIES.items():
            try:
                g = ox.geocode_to_gdf(q)
                c = g.to_crs(config.UTM_CRS).geometry.iloc[0].centroid
                c = gpd.GeoSeries([c], crs=config.UTM_CRS).to_crs(WGS84).iloc[0]
                rows.append({"ref": key, "query": q, "method": "polygon centroid",
                             "osm_name": g["display_name"].iloc[0], "osm_lat": c.y, "osm_lon": c.x})
                continue
            except Exception:
                pass
            try:
                lat, lon = ox.geocode(q)
                rows.append({"ref": key, "query": q, "method": "point", "osm_name": q, "osm_lat": lat, "osm_lon": lon})
            except Exception as e:
                rows.append({"ref": key, "query": q, "method": "none", "osm_name": f"NOT FOUND: {e}",
                             "osm_lat": None, "osm_lon": None})
        osm = pd.DataFrame(rows)
        osm.to_csv(path, index=False, encoding="utf-8")

    out = []
    for r in osm.itertuples():
        lat, lon = config.REF_POINTS[r.ref]
        if pd.isna(r.osm_lat):
            off = None
        else:
            pts = gpd.GeoSeries([Point(lon, lat), Point(r.osm_lon, r.osm_lat)], crs=WGS84).to_crs(config.UTM_CRS)
            off = pts.iloc[0].distance(pts.iloc[1])
        out.append({"ref": r.ref, "config_lat": lat, "config_lon": lon, "osm_lat": r.osm_lat,
                    "osm_lon": r.osm_lon, "offset_m": off, "method": r.method, "osm_name": r.osm_name})
    return pd.DataFrame(out)


def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def osm_kecamatan_key(name: object) -> str:
    """Match key for an OSM kecamatan name: drop accents and 'Kapanewon'/'Kemantren'/'Kecamatan'."""
    if pd.isna(name):
        return ""
    n = normalize_name(_strip_accents(str(name)))
    for p in config.OSM_KECAMATAN_NAME_PREFIXES:
        if n.startswith(p + " "):
            n = n[len(p) + 1:]
    return match_key(n)


def osm_kabupaten_name(name: object) -> str | None:
    """'Kabupaten Sleman' -> 'Sleman', 'Kota Yogyakarta' -> 'Kota Yogyakarta'; None if not DIY."""
    if pd.isna(name):
        return None
    n = normalize_name(_strip_accents(str(name)))
    for p in ("kabupaten ", "kab ", "kota "):
        if n.startswith(p):
            n = n[len(p):]
    by_key = {match_key(k): v for k, v in config.KABUPATEN_MAP.items()}
    return by_key.get(match_key(n))


def match_osm_kecamatan(kab_osm: gpd.GeoDataFrame, kec_osm: gpd.GeoDataFrame,
                        official: pd.DataFrame) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Attach official (kabupaten, kecamatan) to OSM kecamatan polygons.

    Kabupaten comes from the OSM kabupaten polygon containing each kecamatan's interior
    point (works for duplicate names like Jetis). Returns (matched polygons, match log).
    """
    kab_osm = kab_osm[(kab_osm["element"] == "relation") & (kab_osm["admin_level"] == config.OSM_KABUPATEN_ADMIN_LEVEL)]
    kec_osm = kec_osm[(kec_osm["element"] == "relation") & (kec_osm["admin_level"] == config.OSM_KECAMATAN_ADMIN_LEVEL)]
    kab = kab_osm.assign(kabupaten=kab_osm["name"].map(osm_kabupaten_name)).dropna(subset=["kabupaten"])
    kec = kec_osm.to_crs(config.UTM_CRS).copy()
    pts = gpd.GeoDataFrame(kec[["id"]], geometry=kec.geometry.representative_point(), crs=config.UTM_CRS)
    joined = gpd.sjoin(pts, kab[["kabupaten", "geometry"]].to_crs(config.UTM_CRS), how="left", predicate="within")
    kec["kabupaten"] = joined.groupby(level=0)["kabupaten"].first()
    kec = kec.dropna(subset=["kabupaten"])
    kec["key"] = kec["name"].map(osm_kecamatan_key)

    off = official.assign(key=official["kecamatan"].map(match_key))
    log, rows = [], []
    for o in off.itertuples():
        cands = kec[kec["kabupaten"] == o.kabupaten]
        hit = cands[cands["key"] == o.key]
        how = "exact"
        if hit.empty:
            close = difflib.get_close_matches(o.key, cands["key"].tolist(), n=1, cutoff=config.OSM_FUZZY_MATCH_CUTOFF)
            if close:
                hit, how = cands[cands["key"] == close[0]], f"fuzzy ({close[0]})"
        if hit.empty:
            log.append({"kabupaten": o.kabupaten, "kecamatan": o.kecamatan, "match": "MISSING", "osm_name": None})
            continue
        if len(hit) > 1:
            how += f", {len(hit)} polygons dissolved"
        geom = hit.geometry.union_all()
        rows.append({"kabupaten": o.kabupaten, "kecamatan": o.kecamatan, "kecamatan_code": o.kecamatan_code,
                     "osm_name": "; ".join(hit["name"].astype(str).unique()), "geometry": geom})
        log.append({"kabupaten": o.kabupaten, "kecamatan": o.kecamatan, "match": how,
                    "osm_name": "; ".join(hit["name"].astype(str).unique())})
    matched = gpd.GeoDataFrame(rows, geometry="geometry", crs=config.UTM_CRS)
    used = set().union(*[set(r["osm_name"].split("; ")) for r in rows]) if rows else set()
    unused = kec[~kec["name"].astype(str).isin(used)]
    for u in unused.itertuples():
        log.append({"kabupaten": u.kabupaten, "kecamatan": None, "match": "OSM polygon not matched", "osm_name": u.name})
    return matched, pd.DataFrame(log)


def compute_features(kec: gpd.GeoDataFrame, pois: gpd.GeoDataFrame) -> pd.DataFrame:
    """Area, centroid, distances to reference points, POI counts and densities per kecamatan."""
    kec = kec.to_crs(config.UTM_CRS)
    out = kec[["kabupaten", "kecamatan", "kecamatan_code"]].copy()
    out["area_km2"] = kec.geometry.area / 1e6
    cent_utm = kec.geometry.centroid
    cent = gpd.GeoSeries(cent_utm, crs=config.UTM_CRS).to_crs(WGS84)
    out["centroid_lat"], out["centroid_lon"] = cent.y.values, cent.x.values

    ref = {k: gpd.GeoSeries([Point(lon, lat)], crs=WGS84).to_crs(config.UTM_CRS).iloc[0]
           for k, (lat, lon) in config.REF_POINTS.items()}
    out["dist_center_km"] = cent_utm.distance(ref["tugu_yogyakarta"]) / 1000
    out["dist_ugm_km"] = cent_utm.distance(ref["ugm"]) / 1000
    out["dist_airport_km"] = cent_utm.distance(ref["yia_airport"]) / 1000

    p = pois.to_crs(config.UTM_CRS)
    j = gpd.sjoin(p, kec[["kecamatan_code", "geometry"]], how="inner", predicate="within")
    for feat, spec in config.POI_TAGS.items():
        mask = pd.Series(False, index=j.index)
        for k, vals in spec.items():
            mask |= j[k].isin(vals)
        counts = j[mask].groupby("kecamatan_code").size()
        out[feat] = out["kecamatan_code"].map(counts).fillna(0).astype(int)
        out[f"{feat[2:]}_per_km2"] = out[feat] / out["area_km2"]

    bps = config.BPS_POPULATION_CSV
    if bps.exists():
        pop = pd.read_csv(bps)
        out = out.merge(pop[["kecamatan", "kabupaten", "population"]], on=["kecamatan", "kabupaten"], how="left")
        out["population_density"] = out["population"] / out["area_km2"]
        print(f"BPS population merged ({out['population'].notna().sum()} kecamatan)")
    else:
        print("BPS population file not found — skipped (optional)")
    return out.sort_values("kecamatan_code").reset_index(drop=True)


def write_geojson(kec: gpd.GeoDataFrame) -> float:
    """Write simplified kecamatan polygons for the app; returns file size in MB."""
    g = kec.to_crs(config.UTM_CRS)[["kecamatan", "kabupaten", "kecamatan_code", "geometry"]].copy()
    g["geometry"] = g.geometry.simplify(config.GEOJSON_SIMPLIFY_M, preserve_topology=True)
    g = g.to_crs(WGS84)
    config.APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = config.KECAMATAN_GEOJSON
    path.unlink(missing_ok=True)
    g.to_file(path, driver="GeoJSON", layer_options={"COORDINATE_PRECISION": config.GEOJSON_COORD_DECIMALS})
    return path.stat().st_size / 1e6


def main() -> None:
    """Download (or load cached) OSM data, compute features, write CSV + GeoJSON, print summary."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="re-download OSM data")
    args = ap.parse_args()

    official = pd.read_csv(config.KECAMATAN_DIY_CSV, dtype={"kecamatan_code": str})
    diy, kab_osm, kec_osm, pois = download_osm(args.refresh)
    print(f"OSM layers: {len(kab_osm)} kabupaten-level, {len(kec_osm)} kecamatan-level polygons, {len(pois):,} POIs")

    kec, log = match_osm_kecamatan(kab_osm, kec_osm, official)
    missing = log[log["match"] == "MISSING"]
    fuzzy = log[log["match"].str.startswith("fuzzy")]
    unused = log[log["match"] == "OSM polygon not matched"]
    print(f"Kecamatan matched: {len(kec)}/{len(official)}  (fuzzy: {len(fuzzy)}, missing: {len(missing)}, "
          f"unmatched OSM polygons inside DIY: {len(unused)})")
    for r in pd.concat([fuzzy, missing, unused]).itertuples():
        print(f"  {r.match:<28} {r.kabupaten} / {r.kecamatan} <- OSM '{r.osm_name}'")

    feats = compute_features(kec, pois)
    config.EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    feats.to_csv(config.OSM_FEATURES_CSV, index=False, encoding="utf-8")
    mb = write_geojson(kec)
    print(f"Wrote {config.OSM_FEATURES_CSV.relative_to(config.ROOT)} ({len(feats)} rows), "
          f"{config.KECAMATAN_GEOJSON.relative_to(config.ROOT)} ({mb:.2f} MB, {len(kec)} features)")
    if mb > config.GEOJSON_MAX_MB:
        print(f"  WARNING: GeoJSON above {config.GEOJSON_MAX_MB} MB — raise GEOJSON_SIMPLIFY_M")

    refs = verify_ref_points(args.refresh)
    print("Reference points (config vs OSM):")
    for r in refs.itertuples():
        flag = "" if r.offset_m is None or r.offset_m <= config.REF_POINT_MAX_OFFSET_M else "  <-- OFF, update REF_POINTS"
        off = "n/a" if r.offset_m is None else f"{r.offset_m:,.0f} m"
        print(f"  {r.ref:<16} offset {off:>8}  ({r.method}; OSM {r.osm_lat:.5f}, {r.osm_lon:.5f})  "
              f"{str(r.osm_name)[:60]}{flag}")

    clean = pd.read_csv(config.CLEAN_CSV)
    in_data = set(zip(clean["kabupaten"], clean["kecamatan"]))
    have = set(zip(feats["kabupaten"], feats["kecamatan"]))
    lacking = sorted(in_data - have)
    print(f"Kecamatan in clean data without OSM features: {len(lacking)} {lacking if lacking else ''}")
    print(f"Listings per kabupaten: {clean['kabupaten'].value_counts().to_dict()}")
    vc = clean.groupby(["kecamatan", "kabupaten"]).size()
    print(f"Kecamatan with >= {config.MIN_LISTINGS_FOR_MAP} listings: {int((vc >= config.MIN_LISTINGS_FOR_MAP).sum())}")
    print("POI totals:", {f: int(feats[f].sum()) for f in config.POI_TAGS})


if __name__ == "__main__":
    main()
