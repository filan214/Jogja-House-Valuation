"""P2 — Clean raw Rumah123 listings into one row per regular house.

Inputs:  data/raw/rumah123_yogya_unfiltered.csv (read-only)
Outputs: data/interim/listings_clean.csv
         data/interim/listings_dropped.csv  (every dropped row + drop_reason)
         reports/cleaning_report.md         (waterfall: rows in → each step → rows out)

Steps follow IMPLEMENTATION_GUIDE.md P2 1–12, with the P1 deviations (see §6 log).

Run: python -m src.clean
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd

from src import config
from src.utils import format_idr, normalize_name

RENAME = {
    "price": "price_raw",
    "nav-link": "listing_url",
    "description": "description",
    "listing-location": "location_raw",
    "bed": "bedrooms",
    "bath": "bathrooms",
    "carport": "carport",
    "surface_area": "land_raw",
    "building_area": "building_raw",
}

CLEAN_COLUMNS = [
    "listing_id",
    "listing_url",
    "description",
    "area_raw",
    "kabupaten_raw",
    "kabupaten",
    "kabupaten_ambiguous",
    "price_idr",
    "land_m2",
    "building_m2",
    "bedrooms",
    "bathrooms",
    "carport",
    "price_mismatch",
    "price_per_land_m2",
]

NEAR_DUP_KEY = ["area_raw", "kabupaten", "price_idr", "land_m2", "building_m2", "bedrooms", "bathrooms"]

_UNIT_MULTIPLIER = {
    "miliar": Decimal(1_000_000_000),
    "milyar": Decimal(1_000_000_000),
    "m": Decimal(1_000_000_000),
    "juta": Decimal(1_000_000),
    "jutaan": Decimal(1_000_000),
    "jt": Decimal(1_000_000),
}
_PRICE_WORD_RE = re.compile(r"^(\d+(?:,\d+)?)\s*(miliar|milyar|juta|jt|m)$", re.IGNORECASE)
_PRICE_PLAIN_RE = re.compile(r"^\d{1,3}(?:\.\d{3})+$|^\d+$")
_AREA_RE = re.compile(r"^(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?\s*(?:m²|m2)$", re.IGNORECASE)
_LISTING_ID_RE = re.compile(r"/(hos\d+)/?$")

_DESC_PRICE_RE = re.compile(config.DESC_PRICE_REGEX, re.IGNORECASE)
_NON_PRICE_CTX_RE = re.compile(config.NON_PRICE_CONTEXT_REGEX, re.IGNORECASE)
_DP_CTX_RE = re.compile(config.DP_CONTEXT_REGEX, re.IGNORECASE)


def _is_missing(text: object) -> bool:
    return text is None or (isinstance(text, float) and np.isnan(text)) or pd.isna(text)


def parse_price(text: object) -> int | None:
    """Parse a Rumah123 price string to integer Rupiah.

    "Rp 1,1 Miliar" -> 1_100_000_000, "Rp 875 Juta" -> 875_000_000,
    "Rp 1.250.000.000" -> 1_250_000_000. Comma is the decimal separator; dots in
    plain digits are thousand separators. "Rp", spaces and "Total" are ignored.
    Returns None when the text matches no known pattern (e.g. "Harga Nego").
    """
    if _is_missing(text):
        return None
    s = re.sub(r"\brp\.?|\btotal\b", "", str(text), flags=re.IGNORECASE).strip()
    s = re.sub(r"\s+", " ", s)
    m = _PRICE_WORD_RE.match(s)
    try:
        if m:
            value = Decimal(m.group(1).replace(",", ".")) * _UNIT_MULTIPLIER[m.group(2).lower()]
            return int(value)
        if _PRICE_PLAIN_RE.match(s):
            return int(s.replace(".", ""))
    except InvalidOperation:
        return None
    return None


def parse_area(text: object) -> float | None:
    """Parse an area string in m² to float.

    "120 m²" -> 120.0, "1.200 m²" -> 1200.0 (dot = thousands), "85,5 m2" -> 85.5.
    P1 found only "<int> m²" values; other units (ha, are) are not converted.
    Returns None for missing or unparseable values.
    """
    if _is_missing(text):
        return None
    m = _AREA_RE.match(str(text).strip())
    if not m:
        return None
    whole = m.group(1).replace(".", "")
    frac = m.group(2)
    return float(f"{whole}.{frac}") if frac else float(whole)


def parse_location(text: object) -> tuple[str | None, str | None]:
    """Split "Area, Kabupaten" on the last comma -> (area_raw, kabupaten).

    The kabupaten part is normalized and mapped with config.KABUPATEN_MAP
    ("Umbulharjo, Yogyakarta" -> ("Umbulharjo", "Kota Yogyakarta")). An unknown
    or missing kabupaten part gives None in the second position.
    """
    if _is_missing(text):
        return None, None
    s = str(text).strip()
    if "," not in s:
        return (s or None), None
    area, kab = (p.strip() for p in s.rsplit(",", 1))
    return (area or None), config.KABUPATEN_MAP.get(normalize_name(kab))


def kabupaten_part(text: object) -> str | None:
    """Raw text after the last comma of listing-location (stripped), or None."""
    if _is_missing(text) or "," not in str(text):
        return None
    return str(text).rsplit(",", 1)[1].strip() or None


def extract_listing_id(url: object) -> str | None:
    """Extract the Rumah123 listing ID ("hos12345678") from the end of a listing URL."""
    if _is_missing(url):
        return None
    m = _LISTING_ID_RE.search(str(url).strip())
    return m.group(1) if m else None


@dataclass
class PriceMention:
    """A price-like amount found in listing text."""

    text: str
    value_idr: int
    non_price_context: bool
    dp_context: bool


def find_price_mentions(text: object) -> list[PriceMention]:
    """Find amounts like "400 jutaan", "1,2 M", "155jt" in a description.

    Lowercase "m" is treated as meters and skipped; an uppercase bare "M" counts as
    miliar only when the number is below config.DESC_BARE_M_MAX. Each mention
    records whether non-price or down-payment words appear nearby.
    """
    if _is_missing(text):
        return []
    s = str(text)
    out: list[PriceMention] = []
    for m in _DESC_PRICE_RE.finditer(s):
        num, unit = m.group(1), m.group(2)
        if unit == "m":
            continue
        if "," in num:
            value = Decimal(num.replace(",", "."))
        elif re.fullmatch(r"\d{1,3}\.\d{3}", num):
            value = Decimal(num.replace(".", ""))
        else:
            value = Decimal(num)
        if unit == "M" and value >= config.DESC_BARE_M_MAX:
            continue
        lo = max(0, m.start() - config.DESC_PRICE_CONTEXT_CHARS)
        window = s[lo : m.end() + config.DESC_PRICE_CONTEXT_CHARS]
        out.append(
            PriceMention(
                text=m.group(0),
                value_idr=int(value * _UNIT_MULTIPLIER[unit.lower()]),
                non_price_context=bool(_NON_PRICE_CTX_RE.search(window)),
                dp_context=bool(_DP_CTX_RE.search(window)),
            )
        )
    return out


def _ratio(a: float, b: float) -> float:
    """Symmetric ratio max/min (>= 1); inf if either is non-positive."""
    if a <= 0 or b <= 0:
        return float("inf")
    return max(a, b) / min(a, b)


def price_mismatch(description: object, price_idr: int) -> bool:
    """True if a plain-context price in the text differs from price_idr by > DESC_PRICE_MISMATCH_RATIO."""
    return any(
        _ratio(m.value_idr, price_idr) > config.DESC_PRICE_MISMATCH_RATIO
        for m in find_price_mentions(description)
        if not m.non_price_context
    )


def price_is_dp(description: object, price_idr: int) -> bool:
    """True if the price field equals a down-payment/installment amount stated in the text."""
    return any(
        _ratio(m.value_idr, price_idr) <= config.DP_PRICE_EQUAL_RATIO
        for m in find_price_mentions(description)
        if m.dp_context
    )


@dataclass
class Waterfall:
    """Row counts per step and the rows dropped so far."""

    steps: list[dict] = field(default_factory=list)
    dropped: list[pd.DataFrame] = field(default_factory=list)

    def record(self, step: str, rows_in: int, rows_out: int, reasons: dict[str, int] | None = None) -> None:
        self.steps.append(
            {"step": step, "rows_in": rows_in, "dropped": rows_in - rows_out, "rows_out": rows_out,
             "reasons": reasons or {}}
        )


def _drop(df: pd.DataFrame, masks: list[tuple[str, pd.Series]], wf: Waterfall, step: str) -> pd.DataFrame:
    """Drop rows per (reason, mask) in order; first matching reason wins. Records the step."""
    rows_in = len(df)
    reason = pd.Series(pd.NA, index=df.index, dtype="object")
    for name, mask in masks:
        reason = reason.mask(reason.isna() & mask.fillna(False).astype(bool), name)
    gone = reason.notna()
    if gone.any():
        wf.dropped.append(df[gone].assign(drop_reason=reason[gone], drop_step=step))
    kept = df[~gone]
    counts = reason[gone].value_counts().to_dict()
    wf.record(step, rows_in, len(kept), {k: int(v) for k, v in counts.items()})
    return kept


def _contains(s: pd.Series, pattern: str) -> pd.Series:
    return s.fillna("").astype(str).str.contains(pattern, case=False, regex=True)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Waterfall]:
    """Apply cleaning steps 1–12 to the raw listings.

    Returns (clean, dropped, waterfall). `dropped` holds every removed row with
    `drop_reason` and `drop_step`.
    """
    wf = Waterfall()
    n0 = len(df)

    df = df.rename(columns=RENAME).copy()
    df.insert(0, "listing_id", df["listing_url"].map(extract_listing_id))
    wf.record("1. Rename columns, extract listing_id", n0, len(df))

    df["price_idr"] = df["price_raw"].map(parse_price).astype("Int64")
    df = _drop(df, [("price_unparseable", df["price_idr"].isna())], wf, "2. Parse price")

    df["land_m2"] = df["land_raw"].map(parse_area).astype("float")
    df["building_m2"] = df["building_raw"].map(parse_area).astype("float")
    wf.record("3. Parse land/building areas", len(df), len(df))

    loc = df["location_raw"].map(parse_location)
    df["area_raw"] = loc.map(lambda t: t[0])
    df["kabupaten"] = loc.map(lambda t: t[1])
    df["kabupaten_raw"] = df["location_raw"].map(kabupaten_part)
    df["kabupaten_ambiguous"] = (
        df["kabupaten_raw"].fillna("").map(normalize_name).isin(config.AMBIGUOUS_KABUPATEN_PARTS)
    )
    df = _drop(df, [("kabupaten_unknown", df["kabupaten"].isna())], wf, "4. Parse location")

    df["carport"] = df["carport"].fillna(0).astype(int)
    wf.record("5. Fill missing carport with 0", len(df), len(df))

    desc = df["description"]
    no_building = df["building_m2"].isna() | (df["building_m2"] == 0)
    small_building = no_building | (df["building_m2"] < config.MIN_BUILDING_M2)
    df = _drop(
        df,
        [
            ("land_only", no_building),
            ("non_residential", _contains(desc, config.LAND_COMMERCIAL_REGEX) & small_building),
            ("non_residential", _contains(desc, config.COMMERCIAL_REGEX)),
            ("non_residential", _contains(desc, config.APARTMENT_TITLE_REGEX)),
            ("land_title", _contains(desc, config.LAND_TITLE_REGEX)),
            ("rental", _contains(desc, config.RENTAL_REGEX)),
        ],
        wf,
        "6. Remove non-house listings",
    )

    df["price_mismatch"] = [price_mismatch(d, p) for d, p in zip(df["description"], df["price_idr"])]
    wf.record("7a. Flag description–price mismatch (no drop)", len(df), len(df))
    is_dp = pd.Series([price_is_dp(d, p) for d, p in zip(df["description"], df["price_idr"])], index=df.index)
    group_has_dp = is_dp.groupby([df[c] for c in NEAR_DUP_KEY], dropna=False).transform("any")
    dp_sibling = group_has_dp & ~is_dp
    df = _drop(df, [("price_is_dp", is_dp), ("price_is_dp", dp_sibling)], wf,
               "7b. Drop price field = down payment (+ reposts)")

    is_kos = (df["bedrooms"] >= config.KOS_MIN_BEDROOMS) | _contains(df["description"], config.KOS_REGEX)
    df = _drop(df, [("kos", is_kos)], wf, "8. Remove kos")

    ppl = df["price_idr"].astype(float) / df["land_m2"]
    ratio = df["building_m2"] / df["land_m2"]
    df = _drop(
        df,
        [
            ("price_out_of_range", ~df["price_idr"].astype(float).between(config.MIN_PRICE_IDR, config.MAX_PRICE_IDR)),
            ("land_out_of_range", ~df["land_m2"].between(config.MIN_LAND_M2, config.MAX_LAND_M2)),
            ("building_out_of_range", ~df["building_m2"].between(config.MIN_BUILDING_M2, config.MAX_BUILDING_M2)),
            ("building_land_ratio_out_of_range", ~(ratio <= config.MAX_BUILDING_TO_LAND_RATIO)),
            ("price_per_land_out_of_range",
             ~ppl.between(config.MIN_PRICE_PER_LAND_M2, config.MAX_PRICE_PER_LAND_M2)),
        ],
        wf,
        "9. Range checks",
    )

    dup_id = df["listing_id"].notna() & df["listing_id"].duplicated(keep="first")
    df = _drop(df, [("dup_listing_id", dup_id)], wf, "10a. Dedupe on listing_id")
    df = _drop(df, [("dup_near", df.duplicated(subset=NEAR_DUP_KEY, keep="first"))], wf,
               "10b. Dedupe near-duplicates")

    df["price_per_land_m2"] = df["price_idr"].astype(float) / df["land_m2"]
    wf.record("11. Add price_per_land_m2", len(df), len(df))

    clean_df = df[CLEAN_COLUMNS].reset_index(drop=True)
    dropped = pd.concat(wf.dropped) if wf.dropped else pd.DataFrame(columns=["drop_reason", "drop_step"])
    dropped = dropped.sort_index().reset_index(names="raw_row")
    return clean_df, dropped, wf


def _md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in row) + " |")
    return "\n".join(lines)


def build_report(raw: pd.DataFrame, clean_df: pd.DataFrame, dropped: pd.DataFrame, wf: Waterfall) -> str:
    """Markdown cleaning report: waterfall, drop reasons, mismatch examples, summary."""
    steps = pd.DataFrame(wf.steps)
    steps["reasons"] = steps["reasons"].map(lambda d: ", ".join(f"`{k}` {v}" for k, v in d.items()) or "—")

    reasons = dropped["drop_reason"].value_counts().rename_axis("drop_reason").reset_index(name="rows")
    reasons["share_of_raw_%"] = (reasons["rows"] / len(raw) * 100).round(1)

    flagged_dropped = dropped[dropped.get("price_mismatch", pd.Series(False, index=dropped.index))
                              .fillna(False).astype(bool)]
    flagged = pd.concat([
        clean_df[clean_df["price_mismatch"]].assign(status="in clean data"),
        flagged_dropped.assign(status=lambda d: "dropped later: " + d["drop_reason"]),
    ])
    mm_rows = []
    for r in flagged.head(config.PROFILE_N_SAMPLES).itertuples():
        found = [m for m in find_price_mentions(r.description) if not m.non_price_context]
        mm_rows.append({
            "listing_id": r.listing_id,
            "status": r.status,
            "price_idr": format_idr(int(r.price_idr)),
            "text_mention": "; ".join(f"{m.text} ({format_idr(m.value_idr)})" for m in found),
            "land_m2": f"{r.land_m2:g}",
            "building_m2": f"{r.building_m2:g}",
            "description": r.description,
        })
    mm_txt = _md_table(pd.DataFrame(mm_rows)) if mm_rows else "_None._"
    n_flag_clean = int(clean_df["price_mismatch"].sum())
    n_flag_total = len(flagged)

    kab = clean_df["kabupaten"].value_counts().rename_axis("kabupaten").reset_index(name="rows")
    amb = clean_df[clean_df["kabupaten_ambiguous"]]["area_raw"].value_counts()
    amb_txt = ", ".join(f"{a} ({n})" for a, n in amb.items())

    def q(s: pd.Series, money: bool = False) -> str:
        v = s.quantile([0, 0.5, 1]).tolist()
        f = (lambda x: format_idr(int(x))) if money else (lambda x: f"{x:,.0f}")
        return " / ".join(f(x) for x in v)

    summary = pd.DataFrame([
        {"column": "price_idr", "min / median / max": q(clean_df["price_idr"].astype(float), money=True),
         "missing": int(clean_df["price_idr"].isna().sum())},
        {"column": "land_m2", "min / median / max": q(clean_df["land_m2"]),
         "missing": int(clean_df["land_m2"].isna().sum())},
        {"column": "building_m2", "min / median / max": q(clean_df["building_m2"]),
         "missing": int(clean_df["building_m2"].isna().sum())},
        {"column": "bedrooms", "min / median / max": q(clean_df["bedrooms"]),
         "missing": int(clean_df["bedrooms"].isna().sum())},
        {"column": "bathrooms", "min / median / max": q(clean_df["bathrooms"]),
         "missing": int(clean_df["bathrooms"].isna().sum())},
        {"column": "carport", "min / median / max": q(clean_df["carport"]),
         "missing": int(clean_df["carport"].isna().sum())},
        {"column": "price_per_land_m2", "min / median / max": q(clean_df["price_per_land_m2"], money=True),
         "missing": int(clean_df["price_per_land_m2"].isna().sum())},
    ])

    return (
        "# Cleaning report\n\n"
        "Generated by `python -m src.clean`. Rules: IMPLEMENTATION_GUIDE.md P2 plus the P1 entries "
        "in the Deviations log.\n\n"
        f"- Rows in: **{len(raw):,}**\n- Rows out: **{len(clean_df):,}** "
        f"({len(clean_df) / len(raw) * 100:.1f}% kept)\n- Rows dropped: **{len(dropped):,}** "
        "(all in `data/interim/listings_dropped.csv` with `drop_reason`)\n\n"
        "## Waterfall\n\n" + _md_table(steps) + "\n\n"
        "## Drop reasons\n\n" + _md_table(reasons) + "\n\n"
        "## Description–price mismatch (flag only, not dropped)\n\n"
        f"Flagged at step 7a: **{n_flag_total}**; still in clean data: **{n_flag_clean}** "
        "(the rest were dropped by a later step for another reason). "
        "A price mentioned in the title differs from the price field by more than "
        f"{config.DESC_PRICE_MISMATCH_RATIO:g}×, after skipping mentions near installment / DP / KPR / "
        "discount / promo / road-width words.\n\n" + mm_txt + "\n\n"
        "## Clean data summary\n\n" + _md_table(kab) + "\n\n"
        f"- Rows with ambiguous kabupaten part 'Yogyakarta' (provisionally Kota Yogyakarta, "
        f"resolved in P3): **{int(clean_df['kabupaten_ambiguous'].sum())}** — {amb_txt}\n"
        f"- Distinct `area_raw`: **{clean_df['area_raw'].nunique()}**\n\n" + _md_table(summary) + "\n"
    )


def main() -> None:
    """Run cleaning end to end and write outputs."""
    raw = pd.read_csv(config.RAW_CSV)
    clean_df, dropped, wf = clean(raw)

    assert len(clean_df) + len(dropped) == len(raw), "rows lost between clean and dropped"
    assert dropped["drop_reason"].notna().all(), "dropped row without drop_reason"

    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(config.CLEAN_CSV, index=False, encoding="utf-8")
    dropped.to_csv(config.DROPPED_CSV, index=False, encoding="utf-8")
    config.CLEANING_REPORT_MD.write_text(build_report(raw, clean_df, dropped, wf), encoding="utf-8")

    print(f"Rows in: {len(raw):,}  ->  clean: {len(clean_df):,}  dropped: {len(dropped):,}")
    for s in wf.steps:
        if s["dropped"]:
            print(f"  {s['step']:<40} -{s['dropped']:>4}  {s['reasons']}")
    flagged_later = int(dropped["price_mismatch"].fillna(False).astype(bool).sum())
    print(f"price_mismatch flagged: {int(clean_df['price_mismatch'].sum())} in clean data "
          f"(+{flagged_later} flagged then dropped later)")
    print(f"kabupaten_ambiguous: {int(clean_df['kabupaten_ambiguous'].sum())}")
    print(f"Clean rows per kabupaten: {clean_df['kabupaten'].value_counts().to_dict()}")
    print(f"Wrote {config.CLEAN_CSV.relative_to(config.ROOT)}, "
          f"{config.DROPPED_CSV.relative_to(config.ROOT)}, {config.CLEANING_REPORT_MD.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
