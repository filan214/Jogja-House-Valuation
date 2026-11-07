"""All paths, thresholds, and constants for the project.

Thresholds are starting values from IMPLEMENTATION_GUIDE.md §3. Any change
goes in the Deviations log.
"""

from pathlib import Path

RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW_DIR = DATA / "raw"
EXTERNAL_DIR = DATA / "external"
INTERIM_DIR = DATA / "interim"
PROCESSED_DIR = DATA / "processed"
APP_DATA_DIR = DATA / "app"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
OSM_CACHE_DIR = ROOT / "cache"

RAW_CSV = RAW_DIR / "rumah123_yogya_unfiltered.csv"
CLEAN_CSV = INTERIM_DIR / "listings_clean.csv"
DROPPED_CSV = INTERIM_DIR / "listings_dropped.csv"
MODEL_TABLE = PROCESSED_DIR / "model_table.csv"

WILAYAH_ZIP = EXTERNAL_DIR / "IndoArea-19-04-2024.zip"
KECAMATAN_DIY_CSV = EXTERNAL_DIR / "kecamatan_diy.csv"
DESA_DIY_CSV = EXTERNAL_DIR / "desa_diy.csv"
AREA_TO_KECAMATAN_CSV = EXTERNAL_DIR / "area_to_kecamatan.csv"
OSM_FEATURES_CSV = EXTERNAL_DIR / "osm_kecamatan_features.csv"
BPS_POPULATION_CSV = EXTERNAL_DIR / "bps_population.csv"

KECAMATAN_GEOJSON = APP_DATA_DIR / "kecamatan.geojson"
KECAMATAN_STATS_CSV = APP_DATA_DIR / "kecamatan_stats.csv"
COMPS_POOL_CSV = APP_DATA_DIR / "comps_pool.csv"
APP_OSM_FEATURES_CSV = APP_DATA_DIR / "osm_kecamatan_features.csv"
APP_FIGURES_DIR = APP_DATA_DIR / "figures"

RIDGE_MODEL = MODELS_DIR / "ridge.joblib"
LGBM_POINT_MODEL = MODELS_DIR / "lgbm_point.txt"
LGBM_Q10_MODEL = MODELS_DIR / "lgbm_q10.txt"
LGBM_Q50_MODEL = MODELS_DIR / "lgbm_q50.txt"
LGBM_Q90_MODEL = MODELS_DIR / "lgbm_q90.txt"
FEATURE_COLUMNS_JSON = MODELS_DIR / "feature_columns.json"
METADATA_JSON = MODELS_DIR / "metadata.json"

DATA_PROFILE_MD = REPORTS_DIR / "data_profile.md"
CLEANING_REPORT_MD = REPORTS_DIR / "cleaning_report.md"
EDA_SUMMARY_MD = REPORTS_DIR / "eda_summary.md"
METRICS_CSV = REPORTS_DIR / "metrics.csv"
METRICS_MD = REPORTS_DIR / "metrics.md"
WORST_PREDICTIONS_MD = REPORTS_DIR / "worst_predictions.md"

FIGURE_DPI = 150
FIG_COLORS = {
    "series_1": "#2a78d6",
    "series_2": "#eb6834",
    "context": "#c9c8c1",
    "surface": "#fcfcfb",
    "text": "#0b0b0b",
    "text_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}
FIG_FONT = ["Segoe UI", "DejaVu Sans"]
EDA_TOP_N_KECAMATAN = 10
DIST_BANDS_KM = (0, 5, 10, 15, 100)

KABUPATEN_MAP = {
    "yogyakarta": "Kota Yogyakarta",
    "sleman": "Sleman",
    "bantul": "Bantul",
    "kulon progo": "Kulon Progo",
    "gunung kidul": "Gunungkidul",
    "gunungkidul": "Gunungkidul",
}

MIN_PRICE_IDR = 100_000_000
MAX_PRICE_IDR = 25_000_000_000
MIN_LAND_M2 = 20
MAX_LAND_M2 = 5_000
MIN_BUILDING_M2 = 15
MAX_BUILDING_M2 = 2_000
MIN_PRICE_PER_LAND_M2 = 1_000_000
MAX_PRICE_PER_LAND_M2 = 40_000_000
MAX_BUILDING_TO_LAND_RATIO = 3
KOS_MIN_BEDROOMS = 10
DESC_PRICE_MISMATCH_RATIO = 2.0

LAND_COMMERCIAL_REGEX = r"\b(?:dijual tanah|tanah kavling|ruko|gudang)\b"
RENTAL_REGEX = r"\bdisewakan\b|\bsewa\b|\bper tahun\b|/\s*tahun\b"
DESC_PRICE_REGEX = r"(?<![\w.,])(\d+(?:[.,]\d+)?)\s*(miliar|milyar|jutaan|juta|jt|M)(?![\w²])"
DESC_BARE_M_MAX = 100
KOS_REGEX = r"\bkos(?:t|an|kosan|-kosan|\.kosan|²an|2an|t2an|2|tel)?\b"
COMMERCIAL_REGEX = (
    r"\bhotel\b|guest ?house|home ?stay|penginapan|restaurant|restoran|\bresto\b|\bcafe\b|\bkafe\b"
)
APARTMENT_TITLE_REGEX = r"^\s*(?:dijual\s+)?(?:apartemen|apartment)\b"
LAND_TITLE_REGEX = r"^\s*tanah\b"
NON_PRICE_CONTEXT_REGEX = (
    r"cicil|angsur|\bdp\b|per\s*bulan|/\s*bln|/\s*bulan|\bbln\b|booking|\butj\b|kpr"
    r"|diskon|potongan|promo|bonus|cashback|cukup|bayar|akses|jalan|\bjl\b|lebar"
)
DP_CONTEXT_REGEX = r"\bdp\b|uang\s*muka|booking|\butj\b|cicil|angsur"
DESC_PRICE_CONTEXT_CHARS = 25
DP_PRICE_EQUAL_RATIO = 1.1
AMBIGUOUS_KABUPATEN_PARTS = ("yogyakarta",)

DIY_PROVINCE_CODE = "34"
N_KECAMATAN_DIY = 78
FUZZY_MATCH_CUTOFF = 0.8

LOCALITY_SUGGESTIONS: dict[str, tuple[str, str, str]] = {
    "kaliurang": (
        "Ngaglik", "Sleman",
        "Usually 'Jl. Kaliurang' corridor: titles cite km 4–14 (km ~8–14 = Ngaglik; km 4–7 = Depok/Mlati). "
        "Villa listings in the Kaliurang resort itself are in Pakem.",
    ),
    "pogung": ("Mlati", "Sleman", "Pogung (Kidul/Lor/Baru) is in desa Sinduadi, Mlati; titles also cite Palagan/Jongke."),
    "jombor": ("Mlati", "Sleman", "Jombor is in desa Sinduadi, Mlati."),
    "cebongan": ("Mlati", "Sleman", "Pasar Cebongan is in desa Tlogoadi, Mlati; one title says 'Cebongan, Mlati'."),
    "seturan": ("Depok", "Sleman", "Seturan is in desa Caturtunggal, Depok; titles say 'Seturan Depok Sleman'."),
    "nologaten": (
        "Depok", "Sleman",
        "Nologaten is in desa Caturtunggal, Depok. CONFLICT: both titles mention Potorono/Banguntapan (Bantul).",
    ),
    "kulonprogo": (
        "Sentolo", "Kulon Progo",
        "Area is only the kabupaten name. 9 of 11 titles mention Sentolo (one development near the Sentolo toll exit).",
    ),
}

AREA_REVIEW_NOTES: dict[str, str] = {
    "demangan": "CONFLICT: both titles say 'Jogja Selatan' / 'dekat kampus UMY' (Kasihan, Bantul).",
}

OSM_PLACE = "Daerah Istimewa Yogyakarta, Indonesia"
OSM_KABUPATEN_ADMIN_LEVEL = "5"
OSM_KECAMATAN_ADMIN_LEVEL = "6"
OVERPASS_URLS = (
    "https://overpass-api.de/api",
    "https://overpass.private.coffee/api",
    "https://overpass.kumi.systems/api",
)
OVERPASS_RETRIES_PER_URL = 3
OVERPASS_BACKOFF_S = 20
OSM_REQUEST_TIMEOUT_S = 600
OSM_BOUNDARIES_GPKG = OSM_CACHE_DIR / "osm_boundaries_diy.gpkg"
OSM_POIS_GPKG = OSM_CACHE_DIR / "osm_pois_diy.gpkg"
OSM_REF_POINTS_CSV = OSM_CACHE_DIR / "osm_ref_points.csv"
OSM_KECAMATAN_NAME_PREFIXES = ("kecamatan", "kapanewon", "kemantren", "distrik")
OSM_FUZZY_MATCH_CUTOFF = 0.8
POI_TAGS: dict[str, dict[str, list[str]]] = {
    "n_schools": {"amenity": ["school"]},
    "n_universities": {"amenity": ["university", "college"]},
    "n_health": {"amenity": ["hospital", "clinic"]},
    "n_markets": {"amenity": ["marketplace"]},
    "n_retail": {"shop": ["supermarket", "mall", "convenience"]},
}
REF_POINT_QUERIES = {
    "tugu_yogyakarta": "Tugu Yogyakarta, Yogyakarta, Indonesia",
    "ugm": "Universitas Gadjah Mada, Sleman, Indonesia",
    "yia_airport": "Yogyakarta International Airport, Kulon Progo, Indonesia",
}
REF_POINT_MAX_OFFSET_M = 300
GEOJSON_SIMPLIFY_M = 40
GEOJSON_COORD_DECIMALS = 5
GEOJSON_MAX_MB = 2.0

MIN_LISTINGS_FOR_MAP = 5
LOW_DATA_KECAMATAN_THRESHOLD = 10

REF_POINTS = {
    "tugu_yogyakarta": (-7.7829, 110.3671),
    "ugm": (-7.7701, 110.3776),
    "yia_airport": (-7.9003, 110.0583),
}
UTM_CRS = "EPSG:32749"
WGS84_CRS = "EPSG:4326"

KEYWORD_FLAGS: dict[str, str] = {
    "flag_shm": r"\bshm\b|sertifikat hak milik",
    "flag_hgb": r"\b(?:shgb|hgb)\b",
    "flag_ready": r"siap huni",
    "flag_new": r"\b(?:rumah baru|baru gress|brand new)\b",
    "flag_near_campus": r"\b(?:kampus|ugm|uny|uii|upn|uad|umy|atma jaya|amikom|sanata dharma)\b",
    "flag_two_floor": r"\b(?:2|dua) lantai\b|\blantai 2\b|\b2 ?lt\b",
    "flag_complex": r"\b(?:perumahan|cluster|one gate|komplek)\b",
    "flag_main_road": r"\b(?:pinggir jalan|tepi jalan|jalan utama|jalan raya)\b",
    "flag_furnished": r"\b(?:full furnish|furnished|semi furnish)\b",
    "flag_pool": r"kolam renang|swimming pool",
}
MIN_FLAG_SHARE = 0.02
ACTIVE_FLAGS: tuple[str, ...] = (
    "flag_shm",
    "flag_ready",
    "flag_new",
    "flag_near_campus",
    "flag_two_floor",
    "flag_complex",
    "flag_furnished",
    "flag_pool",
)
FLAG_LABELS: dict[str, str] = {
    "flag_shm": "SHM certificate",
    "flag_hgb": "HGB certificate",
    "flag_ready": "Ready to move in",
    "flag_new": "New house",
    "flag_near_campus": "Near a campus",
    "flag_two_floor": "Two floors",
    "flag_complex": "In a housing complex",
    "flag_main_road": "On a main road",
    "flag_furnished": "Furnished",
    "flag_pool": "Swimming pool",
}

KABUPATEN_ORDER: tuple[str, ...] = ("Kota Yogyakarta", "Sleman", "Bantul", "Kulon Progo", "Gunungkidul")
LOCATION_FEATURES: tuple[str, ...] = (
    "dist_center_km",
    "dist_ugm_km",
    "dist_airport_km",
    "schools_per_km2",
    "universities_per_km2",
    "health_per_km2",
    "markets_per_km2",
    "retail_per_km2",
)
OPTIONAL_LOCATION_FEATURES: tuple[str, ...] = ("population_density",)

DISTANCE_LABEL = "Distance to city/campus/airport"
AMENITIES_LABEL = "Neighborhood amenities"
KABUPATEN_LABEL = "Kabupaten/Kota"
FEATURE_LABELS: dict[str, str] = {
    "land_m2": "Land size",
    "log_land": "Land size",
    "building_m2": "Building size",
    "log_building": "Building size",
    "building_to_land_ratio": "Building-to-land ratio",
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "carport": "Carport",
    **{f: FLAG_LABELS[f] for f in ACTIVE_FLAGS},
    **{"kab_" + k.lower().replace(" ", "_"): KABUPATEN_LABEL for k in KABUPATEN_ORDER},
    "dist_center_km": DISTANCE_LABEL,
    "dist_ugm_km": DISTANCE_LABEL,
    "dist_airport_km": DISTANCE_LABEL,
    **{f: AMENITIES_LABEL for f in LOCATION_FEATURES if f.endswith("_per_km2")},
    "population_density": "Population density",
}

PROFILE_EXAMPLES_PER_PATTERN = 3
PROFILE_TOP_AREAS = 40
PROFILE_N_SAMPLES = 10

N_SPLITS = 5
QUANTILES = (0.1, 0.5, 0.9)
CV_GROUP_COLUMN = "kecamatan_code"
DATA_DATE_RANGE = ("2024-03-20", "2024-05-21")

RIDGE_ALPHAS = (0.1, 1.0, 10.0)
LGBM_BASE_PARAMS = {
    "objective": "regression",
    "n_estimators": 600,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}
LGBM_GRID = {"num_leaves": (7, 15, 31), "min_child_samples": (10, 20, 40)}

PRICE_BANDS = (0, 500_000_000, 1_000_000_000, 2_000_000_000, 5_000_000_000, float("inf"))
PRICE_BAND_LABELS = ("< 500 jt", "500 jt–1 M", "1–2 M", "2–5 M", "> 5 M")
N_WORST_PREDICTIONS = 10
SMALL_CV_GAP_POINTS = 5.0
WORST_DESC_CHARS = 90
RANGE_CHECK_COLUMNS = ("land_m2", "building_m2", "bedrooms", "bathrooms", "carport")

COVERAGE_TARGET = (0.75, 0.85)
INTERVAL_CALIBRATION_FACTOR = 1.32
CALIBRATION_SEARCH = (1.0, 2.0, 0.01)
CALIBRATION_GOAL = 0.80
N_DRIVERS = 5
N_COMPS = 5
COMPS_DISTANCE_FEATURES = ("log_land", "log_building", "bedrooms", "bathrooms", "centroid_lat", "centroid_lon")
COMPS_POOL_COLUMNS = ("listing_id", "kecamatan", "kabupaten", "land_m2", "building_m2", "bedrooms",
                      "bathrooms", "price_idr", "centroid_lat", "centroid_lon")
SHAP_MAX_DISPLAY = 15
OUT_OF_RANGE_MESSAGE = "This is outside the range the model has seen; treat the estimate with extra caution."
RANGE_CHECK_LABELS = {"land_m2": "Land area", "building_m2": "Building area", "bedrooms": "Bedrooms",
                      "bathrooms": "Bathrooms", "carport": "Carport"}

APP_TITLE = "Jogja House Valuation"
GITHUB_URL = "https://github.com/filan214/Jogja-_House-_Valuation"
DATA_SOURCE_NAME = "Rumah123 listings via Kaggle (pramudyadika/yogyakarta-housing-price-ndonesia)"
DATA_SOURCE_URL = "https://www.kaggle.com/datasets/pramudyadika/yogyakarta-housing-price-ndonesia"
DATA_LICENSE_NOTE = ("CC BY-NC 4.0 (Attribution-NonCommercial 4.0 International), dataset by Pramudya Dika on "
                     "Kaggle. Used here for a non-commercial portfolio project; the raw data is not redistributed.")
DISCLAIMER_ADJUSTED = (
    "Estimates are based on 2024 Rumah123 listing prices, adjusted to the current market with a city-wide "
    "price index, and are for reference only. They are not an official valuation. For buying, selling, or "
    "financing decisions, consult a licensed appraiser (KJPP, Kantor Jasa Penilai Publik)."
)
DISCLAIMER = (
    "Estimates are based on 2024 Rumah123 listing prices and are for reference only. They are not an "
    "official valuation. For buying, selling, or financing decisions, consult a licensed appraiser "
    "(KJPP, Kantor Jasa Penilai Publik)."
)
LIMITATION_DATE_ADJUSTED = (
    "Data is from March–May 2024.",
    "Estimates are adjusted to the current level with one city-wide index, so areas that grew faster than "
    "average are still underestimated (and slower ones slightly overestimated). The drift check reports how "
    "big the remaining error is.",
)
NO_ADJUSTMENT_CAPTION = "Shown at 2024 prices; no market adjustment set."
LIMITATIONS: tuple[tuple[str, str], ...] = (
    ("Asking price ≠ sale price.", "The model learns what sellers list for; deals usually close lower."),
    ("Data is from March–May 2024.", "Jogja prices have risen since, so estimates likely run low for 2026."),
    ("~2,000 listings, located only by kecamatan.",
     "Two houses in the same kecamatan on very different streets look the same to the model."),
    ("Thin data in some areas", "(e.g. Gunungkidul, Kulon Progo), so estimates there are less reliable."),
    ("Listing text is self-reported", "by agents and may be inaccurate."),
)
VERDICT_LABELS = {"above": "Above typical range", "within": "Within range", "below": "Below typical range"}
APP_DEFAULT_INPUTS = {"kabupaten": "Sleman", "kecamatan": "Depok", "land_m2": 120.0, "building_m2": 90.0,
                      "bedrooms": 3, "bathrooms": 2, "carport": 1}
APP_MAX_ROOMS = 50
CLEANING_SUMMARY_CSV = APP_DATA_DIR / "cleaning_summary.csv"
APP_FIGURES: tuple[tuple[str, str, int], ...] = (
    ("02_price_per_m2_by_kabupaten.png", "Price per m² of land by kabupaten.", 760),
    ("07_predicted_vs_actual.png", "Predicted vs actual price under spatial cross-validation.", 560),
    ("09_shap_summary.png", "What moves the estimate (SHAP values of the point model).", 820),
)
APP_FIGURE_FRAME_PX = 900
DROP_REASON_LABELS = {
    "dup_near": "Repost of the same house (near-duplicate)",
    "dup_listing_id": "Duplicate listing ID",
    "kos": "Boarding house (kos)",
    "kecamatan_unmatched": "Location could not be matched to a kecamatan",
    "non_residential": "Commercial, apartment or land with small building",
    "price_is_dp": "Price field is a down payment",
    "price_out_of_range": f"Price outside Rp {MIN_PRICE_IDR // 10**6} jt – {MAX_PRICE_IDR // 10**9} M",
    "land_title": "Land listing (title starts with “Tanah”)",
    "building_out_of_range": f"Building area outside {MIN_BUILDING_M2}–{MAX_BUILDING_M2:,} m²",
    "land_out_of_range": f"Land area outside {MIN_LAND_M2}–{MAX_LAND_M2:,} m²",
    "building_land_ratio_out_of_range": f"Building more than {MAX_BUILDING_TO_LAND_RATIO}× land area",
    "land_only": "No building area",
    "price_per_land_out_of_range": (f"Price per land m² outside Rp {MIN_PRICE_PER_LAND_M2 // 10**6}–"
                                    f"{MAX_PRICE_PER_LAND_M2 // 10**6} jt"),
    "price_unparseable": "Price could not be read",
    "kabupaten_unknown": "Outside DIY / unknown kabupaten",
    "rental": "Rental listing",
}
MAP_STYLE = "carto-positron"
MAP_CENTER = {"lat": -7.87, "lon": 110.43}
MAP_ZOOM = 9.0
MAP_SEQUENTIAL = ("#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
MAP_NO_DATA_COLOR = "#c9c8c1"
MAP_METRICS = {
    "Median price per land m²": "median_price_per_land_m2",
    "Median price": "median_price",
    "Listing count": "n_listings",
}
DRIVER_COLORS = {"up": "#2a78d6", "down": "#eb6834"}

MAIN_ROADS = ["kaliurang", "magelang", "wates", "solo", "parangtritis", "imogiri",
              "wonosari", "godean", "palagan", "monjali", "bantul", "gito gati"]
ROAD_KM_MIN, ROAD_KM_MAX = 0.5, 40
ROAD_ALIASES = {"jakal": "kaliurang", "paris": "parangtritis"}
ROAD_NAME_SUFFIXES = ("barat", "timur", "utara", "selatan", "raya")
USE_ROAD_KM_FEATURES = False
ROAD_KM_LABEL = "Road km marker"
FEATURE_LABELS.update({"road_km": ROAD_KM_LABEL, "has_road_km": ROAD_KM_LABEL})
ABLATION_ROAD_KM_MD = REPORTS_DIR / "ablation_road_km.md"
ABLATION_ROAD_KM_CSV = REPORTS_DIR / "ablation_road_km.csv"
ROAD_KM_N_EXAMPLES = 10
ABLATION_METRICS_SECTION = "## Ablation: road km feature (P7b U1)"
METRICS_PRESERVED_SECTIONS = (ABLATION_METRICS_SECTION,)

TARGET_COVERAGE = 0.80
MIN_ROWS_FOR_KABUPATEN_CALIBRATION = 40
RANGE_SCALE_CAP = (1.0, 2.0)

PRICE_INDEX_FACTOR = None
PRICE_INDEX_SOURCE = None
PRICE_INDEX_AS_OF = None
DATA_END_DATE = "2024-05"

DRIFT_CSV = EXTERNAL_DIR / "drift_check_2026.csv"
DRIFT_EXAMPLE_CSV = EXTERNAL_DIR / "drift_check_2026_EXAMPLE.csv"
DRIFT_REPORT_MD = REPORTS_DIR / "drift_check.md"
DRIFT_SUMMARY_JSON = APP_DATA_DIR / "drift_summary.json"
DRIFT_COLUMNS: tuple[str, ...] = (
    "date_seen", "url", "kabupaten", "kecamatan", "land_m2", "building_m2", "bedrooms", "bathrooms",
    "carport", "road_name", "road_km", *ACTIVE_FLAGS, "asking_price_idr",
)
DRIFT_TARGET_ROWS = (20, 30)
DRIFT_SMALL_KABUPATEN_N = 5
DRIFT_LOW_BIAS_PCT = -10.0
