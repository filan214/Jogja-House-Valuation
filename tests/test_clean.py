import math

import pandas as pd
import pytest

from src import config
from src.clean import (
    _contains,
    clean,
    extract_listing_id,
    find_price_mentions,
    parse_area,
    parse_location,
    parse_price,
    price_is_dp,
    price_mismatch,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Rp 1,1 Miliar", 1_100_000_000),
        ("Rp 875 Juta", 875_000_000),
        ("Rp 2,45 Miliar", 2_450_000_000),
        ("Rp 1.250.000.000", 1_250_000_000),
        ("Harga Nego", None),
        ("Rp 1,79 Miliar", 1_790_000_000),
        ("Rp 42,1 Miliar", 42_100_000_000),
        ("Rp 2 Miliar", 2_000_000_000),
        ("Rp 7 Juta", 7_000_000),
        ("rp 875 juta", 875_000_000),
        ("  Rp   875   Juta ", 875_000_000),
        ("Total Rp 1,5 Miliar", 1_500_000_000),
        ("Rp 850000000", 850_000_000),
        ("", None),
        (None, None),
        (float("nan"), None),
        ("Rp", None),
    ],
)
def test_parse_price(text, expected):
    assert parse_price(text) == expected


def test_parse_price_returns_int():
    assert isinstance(parse_price("Rp 2,45 Miliar"), int)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("120 m²", 120.0),
        ("1.200 m²", 1200.0),
        ("4222 m²", 4222.0),
        ("120 m2", 120.0),
        ("85,5 m²", 85.5),
        ("2408150000 m²", 2_408_150_000.0),
        ("abc", None),
        ("120", None),
        ("1 ha", None),
        (None, None),
        (float("nan"), None),
    ],
)
def test_parse_area(text, expected):
    assert parse_area(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Depok, Sleman", ("Depok", "Sleman")),
        ("Umbulharjo, Yogyakarta", ("Umbulharjo", "Kota Yogyakarta")),
        ("Karangmojo, Gunung Kidul", ("Karangmojo", "Gunungkidul")),
        ("Sentolo, Kulon Progo", ("Sentolo", "Kulon Progo")),
        ("Condong Catur, Sleman", ("Condong Catur", "Sleman")),
        ("Sleman, Sleman", ("Sleman", "Sleman")),
        ("A, B, Bantul", ("A, B", "Bantul")),
        ("Kuta, Badung", ("Kuta", None)),
        ("Sleman", ("Sleman", None)),
        (None, (None, None)),
    ],
)
def test_parse_location(text, expected):
    assert parse_location(text) == expected


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://www.rumah123.com/properti/sleman/hos17166670/", "hos17166670"),
        ("https://www.rumah123.com/properti/kulon-progo/hos4415395", "hos4415395"),
        ("https://www.rumah123.com/properti/sleman/", None),
        (None, None),
    ],
)
def test_extract_listing_id(url, expected):
    assert extract_listing_id(url) == expected


def test_mentions_plain_price():
    (m,) = find_price_mentions("Rumah Baru 400 Jutaan Tamanan Banguntapan")
    assert m.value_idr == 400_000_000 and not m.non_price_context and not m.dp_context


def test_mentions_miliar_decimal():
    (m,) = find_price_mentions("Rumah Mewah 1,2 M Siap Huni")
    assert m.value_idr == 1_200_000_000


def test_mentions_dot_decimal_jt():
    (m,) = find_price_mentions("Cukup Bayar 5.5jt Gratis Semua Biaya")
    assert m.value_idr == 5_500_000 and m.non_price_context


def test_mentions_meters_ignored():
    assert find_price_mentions("Jalan lebar 6 m dari kampus") == []
    assert find_price_mentions("RUMAH 200 M DARI JALAN RAYA") == []
    assert find_price_mentions("Luas 88 M2 dekat pasar") == []


def test_mentions_road_access_is_non_price():
    (m,) = find_price_mentions("Rumah Scandinavian Dengan Mezanine Akses 7 M Dekat Candi")
    assert m.non_price_context


def test_mentions_dp_context():
    (m,) = find_price_mentions("Rumah Murah Hanya Dp 155 Juta di Besi Jangkang")
    assert m.dp_context and m.non_price_context


def test_price_mismatch_rules():
    assert price_mismatch("Rumah 400 Jutaan di Bantul", 2_000_000_000)
    assert not price_mismatch("Rumah 400 Jutaan di Bantul", 455_000_000)
    assert not price_mismatch("Diskon 200 Juta rumah mewah", 2_170_000_000)
    assert not price_mismatch("Rumah Murah Cicilan 1jt", 200_000_000)
    assert not price_mismatch("Rumah tanpa harga di judul", 500_000_000)


def test_price_is_dp_rules():
    assert price_is_dp("Promo Rumah Siap Huni Dengan Dp 155jt di Ngaglik", 155_000_000)
    assert not price_is_dp("Promo Rumah Siap Huni Dengan Dp 155jt di Ngaglik", 900_000_000)
    assert not price_is_dp("Rumah 250 Juta Siap Kpr", 250_000_000)


def _raw(**overrides) -> dict:
    row = {
        "price": "Rp 1,1 Miliar",
        "nav-link": "https://www.rumah123.com/properti/sleman/hos1/",
        "description": "Rumah Siap Huni Dekat UGM",
        "listing-location": "Depok, Sleman",
        "bed": 3.0,
        "bath": 2.0,
        "carport": 1.0,
        "surface_area": "120 m²",
        "building_area": "100 m²",
    }
    row.update(overrides)
    return row


def _id(n: int) -> str:
    return f"https://www.rumah123.com/properti/sleman/hos{n}/"


@pytest.fixture(scope="module")
def cleaned():
    rows = [
        _raw(),
        _raw(**{"nav-link": _id(2), "price": "Harga Nego"}),
        _raw(**{"nav-link": _id(3), "listing-location": "Kuta, Badung"}),
        _raw(**{"nav-link": _id(4), "building_area": None}),
        _raw(**{"nav-link": _id(5), "description": "Dijual Guest House Dekat Malioboro"}),
        _raw(**{"nav-link": _id(6), "description": "Ruko Strategis", "building_area": "10 m²"}),
        _raw(**{"nav-link": _id(7), "description": "Rumah Disewakan Per Tahun"}),
        _raw(**{"nav-link": _id(8), "price": "Rp 155 Juta", "description": "Hanya Dp 155 Juta"}),
        _raw(**{"nav-link": _id(9), "description": "Kost 12 Kamar Dekat UNY"}),
        _raw(**{"nav-link": _id(10), "bed": 12.0}),
        _raw(**{"nav-link": _id(11), "price": "Rp 69 Juta"}),
        _raw(**{"nav-link": _id(12), "surface_area": "4 m²"}),
        _raw(**{"nav-link": _id(13), "building_area": "2500 m²", "surface_area": "1000 m²"}),
        _raw(**{"nav-link": _id(14), "building_area": "1012 m²", "surface_area": "101 m²"}),
        _raw(**{"nav-link": _id(15), "price": "Rp 130 Juta", "surface_area": "320 m²"}),
        _raw(),
        _raw(**{"nav-link": _id(17), "description": "Rumah Cantik Lain Judul"}),
        _raw(**{"nav-link": _id(18), "description": "Rumah 400 Jutaan Siap Huni",
                "carport": None, "listing-location": "Maguwoharjo, Yogyakarta"}),
        _raw(**{"nav-link": _id(19), "bed": 8.0, "price": "Rp 3,5 Miliar",
                "surface_area": "300 m²", "building_area": "400 m²"}),
        _raw(**{"nav-link": _id(20), "price": "Rp 155 Juta",
                "description": "Rumah Murah Dekat Pasar"}),
        _raw(**{"nav-link": _id(21), "bed": None, "price": "Rp 900 Juta",
                "surface_area": "150 m²"}),
    ]
    raw = pd.DataFrame(rows)
    return raw, *clean(raw)


def test_clean_row_accounting(cleaned):
    raw, clean_df, dropped, _ = cleaned
    assert len(clean_df) + len(dropped) == len(raw)
    assert dropped["drop_reason"].notna().all()


def test_clean_drop_reasons(cleaned):
    _, _, dropped, _ = cleaned
    got = dropped.set_index("raw_row")["drop_reason"].to_dict()
    assert got == {
        1: "price_unparseable",
        2: "kabupaten_unknown",
        3: "land_only",
        4: "non_residential",
        5: "non_residential",
        6: "rental",
        7: "price_is_dp",
        8: "kos",
        9: "kos",
        10: "price_out_of_range",
        11: "land_out_of_range",
        12: "building_out_of_range",
        13: "building_land_ratio_out_of_range",
        14: "price_per_land_out_of_range",
        15: "dup_listing_id",
        16: "dup_near",
        19: "price_is_dp",
    }


def test_clean_kept_rows(cleaned):
    _, clean_df, _, _ = cleaned
    assert clean_df["listing_id"].tolist() == ["hos1", "hos18", "hos19", "hos21"]
    assert pd.isna(clean_df.set_index("listing_id").loc["hos21", "bedrooms"])
    first = clean_df.iloc[0]
    assert first["price_idr"] == 1_100_000_000
    assert first["land_m2"] == 120.0 and first["building_m2"] == 100.0
    assert math.isclose(first["price_per_land_m2"], 1_100_000_000 / 120)
    assert not first["kabupaten_ambiguous"]


def test_clean_flags_and_fills(cleaned):
    _, clean_df, _, _ = cleaned
    row = clean_df.set_index("listing_id").loc["hos18"]
    assert row["carport"] == 0
    assert row["kabupaten"] == "Kota Yogyakarta" and row["kabupaten_ambiguous"]
    assert row["price_mismatch"]


def test_clean_waterfall_consistent(cleaned):
    raw, clean_df, _, wf = cleaned
    assert wf.steps[0]["rows_in"] == len(raw)
    assert wf.steps[-1]["rows_out"] == len(clean_df)
    for prev, nxt in zip(wf.steps, wf.steps[1:]):
        assert prev["rows_out"] == nxt["rows_in"]


def test_config_thresholds_used():
    assert config.KOS_MIN_BEDROOMS == 10
    assert config.MIN_PRICE_PER_LAND_M2 == 1_000_000
    assert config.MAX_BUILDING_TO_LAND_RATIO == 3


@pytest.mark.parametrize(
    "title, apartment, land",
    [
        ("Apartemen Fully Furnished Taman Melati", True, False),
        ("Dijual Apartment Dekat UGM", True, False),
        ("Rumah dekat Apartemen Patra Palagan", False, False),
        ("Tanah Jogja Siap Ajb, Dekat Mirota", False, True),
        ("TANAH MURAH BU, BONUS RUMAH", False, True),
        ("Rumah Di Atas Tanah 200 m2", False, False),
        ("Dijual Tanah Bangunan Sidoarum", False, False),
    ],
)
def test_title_type_rules(title, apartment, land):
    s = pd.Series([title])
    assert _contains(s, config.APARTMENT_TITLE_REGEX).iloc[0] == apartment
    assert _contains(s, config.LAND_TITLE_REGEX).iloc[0] == land
