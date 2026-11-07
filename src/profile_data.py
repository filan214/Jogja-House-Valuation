"""P1 — Profile the raw Rumah123 CSV and write reports/data_profile.md.

Read-only: no cleaning, no rows dropped, nothing written to data/.
The small value-extraction helpers below exist only to describe the data
(patterns, ranges, duplicates). The real parsers are written in P2 (src/clean.py).

Run: python -m src.profile_data
"""

from __future__ import annotations

import re

import pandas as pd

from src import config
from src.clean import find_price_mentions

COL_PRICE = "price"
COL_URL = "nav-link"
COL_DESC = "description"
COL_LOC = "listing-location"
COL_BED = "bed"
COL_BATH = "bath"
COL_CARPORT = "carport"
COL_LAND = "surface_area"
COL_BUILDING = "building_area"

NEAR_DUP_KEY = [COL_LOC, COL_PRICE, COL_LAND, COL_BUILDING, COL_BED, COL_BATH]

PRICE_PATTERNS: list[tuple[str, str]] = [
    ("Rp X Miliar", r"^Rp\s*\d+(?:,\d+)?\s*Miliar$"),
    ("Rp X Juta", r"^Rp\s*\d+(?:,\d+)?\s*Juta$"),
    ("Rp X Ribu", r"^Rp\s*\d+(?:,\d+)?\s*Ribu$"),
    ("Rp X Triliun", r"^Rp\s*\d+(?:,\d+)?\s*Triliun$"),
    ("Plain digits (dot thousands)", r"^(?:Rp\s*)?\d{1,3}(?:\.\d{3})+$"),
    ("Plain digits", r"^(?:Rp\s*)?\d+$"),
    ("Rp + number + other word", r"^Rp\s*[\d.,]+\s*[A-Za-z]+$"),
]
PRICE_MULTIPLIER = {
    "Rp X Miliar": 1_000_000_000,
    "Rp X Juta": 1_000_000,
    "Rp X Ribu": 1_000,
    "Rp X Triliun": 1_000_000_000_000,
    "Plain digits (dot thousands)": 1,
    "Plain digits": 1,
}

AREA_UNITS: list[tuple[str, str]] = [
    ("m²", r"m²"),
    ("m2", r"\bm2\b"),
    ("ha", r"\bha\b|hektar"),
    ("are", r"\bare\b"),
]
AREA_UNIT_FACTOR = {"m²": 1.0, "m2": 1.0, "ha": 10_000.0, "are": 100.0}

URL_ID_PATTERNS: list[tuple[str, str]] = [
    ("hos<digits>", r"/(hos\d+)/?$"),
    ("other letters+digits", r"/([a-z]+\d+)/?$"),
]
URL_SLUG_RE = re.compile(r"/properti/([^/]+)/")

DESC_KEYWORDS: dict[str, str] = {
    "kos": r"\bkos\b",
    "tanah": r"\btanah\b",
    "ruko": r"\bruko\b",
    "gudang": r"\bgudang\b",
    "disewakan": r"\bdisewakan\b",
    "sewa": r"\bsewa\b",
}
KOS_BROAD_RE = config.KOS_REGEX
EXTRA_KEYWORDS: dict[str, str] = {
    "kos, any spelling (kost, kosan, kos2an, kos²an, koskosan, kostel…)": KOS_BROAD_RE,
    "hotel": r"\bhotel\b",
    "guest house": r"guest ?house",
    "homestay / penginapan": r"home ?stay|penginapan",
    "villa": r"\bvilla\b",
    "restaurant / resto / cafe": r"restaurant|restoran|\bresto\b|\bcafe\b|\bkafe\b",
    "kantor (office)": r"\bkantor\b",
    "toko (shop)": r"\btoko\b",
    "kavling": r"\bkavling\b|\bkapling\b",
}

def md_table(df: pd.DataFrame, index: bool = False) -> str:
    """Render a DataFrame as a GitHub markdown table (no extra dependency)."""
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]

    def cell(v: object) -> str:
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        return str(v).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines)


def _to_float(num: str) -> float:
    """'1,79' -> 1.79; '1.250' -> 1250; '1.5' -> 1.5 (dot + 3 digits = thousands)."""
    if "," in num:
        return float(num.replace(".", "").replace(",", "."))
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", num):
        return float(num.replace(".", ""))
    return float(num)


def classify_price(text: object) -> str:
    """Return the name of the first matching price pattern, or 'No pattern'."""
    if pd.isna(text):
        return "Missing"
    s = str(text).strip()
    for name, pat in PRICE_PATTERNS:
        if re.match(pat, s, flags=re.IGNORECASE):
            return name
    return "No pattern"


def approx_price(text: object, pattern: str) -> float | None:
    """Approximate Rupiah value for profiling stats only."""
    if pattern not in PRICE_MULTIPLIER:
        return None
    m = re.search(r"\d[\d.,]*", str(text))
    if not m:
        return None
    num = m.group(0)
    if pattern.startswith("Plain digits"):
        return float(num.replace(".", ""))
    return float(num.replace(",", ".")) * PRICE_MULTIPLIER[pattern]


def area_unit(text: object) -> str:
    """Return the unit found in an area string."""
    if pd.isna(text):
        return "Missing"
    s = str(text).strip().lower()
    if not re.search(r"\d", s):
        return "No number"
    for name, pat in AREA_UNITS:
        if re.search(pat, s):
            return name
    return "Other / none"


def approx_area(text: object) -> float | None:
    """Approximate m² for profiling stats only."""
    unit = area_unit(text)
    if unit not in AREA_UNIT_FACTOR:
        return None
    m = re.search(r"\d[\d.,]*", str(text))
    return _to_float(m.group(0).rstrip(".,")) * AREA_UNIT_FACTOR[unit] if m else None


def quantiles(s: pd.Series) -> pd.DataFrame:
    """Summary quantiles of a numeric series."""
    qs = [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]
    q = s.dropna().quantile(qs)
    return pd.DataFrame({"quantile": [f"{int(x * 100)}%" for x in qs], "value": q.values})


def fmt_int(x: float) -> str:
    """Integer with comma thousands, for report tables."""
    return f"{x:,.0f}"


def desc_price_mentions(text: str) -> list[dict]:
    """Price-like mentions in a description (shared parser from src.clean)."""
    return [
        {"match": m.text, "value_idr": float(m.value_idr), "non_price_context": m.non_price_context,
         "dp_context": m.dp_context}
        for m in find_price_mentions(text)
    ]


def section_overview(df: pd.DataFrame) -> str:
    """Step 1: rows, columns, dtypes, nulls."""
    info = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "nulls": df.isna().sum().values,
            "null_%": (df.isna().mean() * 100).round(1).values,
            "distinct": df.nunique().values,
        }
    )
    return (
        "## 1. Overview\n\n"
        f"- Rows: **{len(df):,}**\n- Columns: **{df.shape[1]}**\n\n"
        + md_table(info)
        + "\n"
    )


def section_price(df: pd.DataFrame) -> tuple[str, pd.Series]:
    """Step 2: price patterns, examples, unmatched values, approx distribution."""
    pat = df[COL_PRICE].map(classify_price)
    rows = []
    for name, grp in df.groupby(pat, sort=False):
        ex = grp[COL_PRICE].drop_duplicates().head(config.PROFILE_EXAMPLES_PER_PATTERN)
        rows.append({"pattern": name, "count": len(grp), "examples": "; ".join(ex)})
    table = pd.DataFrame(rows).sort_values("count", ascending=False)

    unmatched = df.loc[pat.isin(["No pattern", "Rp + number + other word"]), COL_PRICE]
    unmatched_txt = (
        "\n".join(f"- `{v}` ({n}×)" for v, n in unmatched.value_counts().items())
        if len(unmatched)
        else "_None — every price value matches a known pattern._"
    )

    price = pd.Series(
        [approx_price(t, p) for t, p in zip(df[COL_PRICE], pat)], index=df.index, dtype="float"
    )
    q = quantiles(price)
    q["value"] = q["value"].map(fmt_int)
    below = int((price < config.MIN_PRICE_IDR).sum())
    above = int((price > config.MAX_PRICE_IDR).sum())

    lows = df.loc[price < config.MIN_PRICE_IDR, [COL_PRICE, COL_LAND, COL_BUILDING, COL_DESC]]
    highs = df.loc[price > config.MAX_PRICE_IDR, [COL_PRICE, COL_LAND, COL_BUILDING, COL_DESC]]

    text = (
        "## 2. Price patterns\n\n"
        + md_table(table)
        + "\n\n### Values matching no numeric pattern\n\n"
        + unmatched_txt
        + "\n\n### Approximate price distribution (Rp)\n\n"
        "_Profiling-only conversion: Miliar ×1e9, Juta ×1e6, comma = decimal._\n\n"
        + md_table(q)
        + f"\n\n- Below `MIN_PRICE_IDR` ({fmt_int(config.MIN_PRICE_IDR)}): **{below}**\n"
        f"- Above `MAX_PRICE_IDR` ({fmt_int(config.MAX_PRICE_IDR)}): **{above}**\n"
    )
    if len(lows):
        text += "\n#### Rows below MIN_PRICE_IDR\n\n" + md_table(lows) + "\n"
    if len(highs):
        text += "\n#### Rows above MAX_PRICE_IDR\n\n" + md_table(highs) + "\n"
    return text, price


def section_areas(df: pd.DataFrame, price: pd.Series) -> tuple[str, pd.Series, pd.Series]:
    """Step 3: area units, separators, examples, unparseable, approx ranges."""
    parts = ["## 3. Area patterns (`surface_area` = land, `building_area` = building)\n"]
    parsed = {}
    limits = {
        COL_LAND: (config.MIN_LAND_M2, config.MAX_LAND_M2, "MIN_LAND_M2", "MAX_LAND_M2"),
        COL_BUILDING: (
            config.MIN_BUILDING_M2,
            config.MAX_BUILDING_M2,
            "MIN_BUILDING_M2",
            "MAX_BUILDING_M2",
        ),
    }
    for col in (COL_LAND, COL_BUILDING):
        s = df[col]
        unit = s.map(area_unit)
        rows = []
        for name, grp in s.groupby(unit, sort=False):
            ex = grp.dropna().drop_duplicates().head(config.PROFILE_EXAMPLES_PER_PATTERN)
            rows.append({"unit": name, "count": len(grp), "examples": "; ".join(map(str, ex))})
        table = pd.DataFrame(rows).sort_values("count", ascending=False)

        st = s.dropna().astype(str)
        n_dot_thousands = int(st.str.contains(r"\d\.\d{3}(?!\d)", regex=True).sum())
        n_comma = int(st.str.contains(r"\d,\d", regex=True).sum())
        n_dot_other = int(st.str.contains(r"\d\.\d(?!\d{2}(?!\d))", regex=True).sum())

        val = s.map(approx_area).astype("float")
        parsed[col] = val
        lo, hi, lo_name, hi_name = limits[col]
        unparseable = s[val.isna() & s.notna()]
        q = quantiles(val)
        q["value"] = q["value"].map(lambda v: f"{v:,.0f}")

        parts.append(
            f"### `{col}`\n\n"
            + md_table(table)
            + f"\n\n- Values with dot thousands separator (e.g. `1.200`): **{n_dot_thousands}**\n"
            f"- Values with comma decimal (e.g. `85,5`): **{n_comma}**\n"
            f"- Values with other dot usage: **{n_dot_other}**\n"
            f"- Zero values: **{int((val == 0).sum())}**\n"
            f"- Unparseable (non-missing, no usable unit/number): **{len(unparseable)}**"
            + (f" — {', '.join(f'`{v}`' for v in unparseable.unique()[:20])}" if len(unparseable) else "")
            + "\n\nApproximate distribution (m²):\n\n"
            + md_table(q)
            + f"\n\n- Below `{lo_name}` ({lo}): **{int((val < lo).sum())}**\n"
            f"- Above `{hi_name}` ({hi:,}): **{int((val > hi).sum())}**\n"
        )

    land, building = parsed[COL_LAND], parsed[COL_BUILDING]
    ratio = building / land
    ppl = price / land
    q = quantiles(ppl)
    q["value"] = q["value"].map(fmt_int)
    k = config.MAX_BUILDING_TO_LAND_RATIO
    sus = df.loc[ratio > k, [COL_PRICE, COL_LOC, COL_BED, COL_BATH, COL_LAND, COL_BUILDING, COL_DESC]]
    sus = sus.assign(ratio=ratio[sus.index].round(1))
    ppl_out = (ppl < config.MIN_PRICE_PER_LAND_M2) | (ppl > config.MAX_PRICE_PER_LAND_M2)
    ppl_tbl = df.loc[ppl_out, [COL_PRICE, COL_LOC, COL_LAND, COL_BUILDING, COL_DESC]]
    ppl_tbl = ppl_tbl.assign(price_per_land_m2=ppl[ppl_tbl.index].map(fmt_int))
    parts.append(
        "### Land vs building\n\n"
        f"- building_m2 > land_m2 (multi-storey or error): **{int((ratio > 1).sum())}**\n"
        f"- building_m2 > {k} × land_m2: **{len(sus)}**\n\n"
        f"Rows with building_m2 > {k} × land_m2 (possible typos, kos, or commercial):\n\n"
        + md_table(sus)
        + "\n\n### Approximate price per land m² (Rp/m²)\n\n"
        + md_table(q)
        + f"\n\n- Below `MIN_PRICE_PER_LAND_M2` ({fmt_int(config.MIN_PRICE_PER_LAND_M2)}): "
        f"**{int((ppl < config.MIN_PRICE_PER_LAND_M2).sum())}**\n"
        f"- Above `MAX_PRICE_PER_LAND_M2` ({fmt_int(config.MAX_PRICE_PER_LAND_M2)}): "
        f"**{int((ppl > config.MAX_PRICE_PER_LAND_M2).sum())}**\n\n"
        "Rows outside the price-per-land range:\n\n"
        + md_table(ppl_tbl)
        + "\n"
    )
    return "\n".join(parts), land, building


def section_location(df: pd.DataFrame) -> tuple[str, pd.Series]:
    """Step 4: split listing-location on last comma."""
    loc = df[COL_LOC].astype(str).str.strip()
    has_comma = loc.str.contains(",", regex=False)
    split = loc.str.rsplit(",", n=1, expand=True)
    area = split[0].str.strip()
    kab = split[1].str.strip() if split.shape[1] > 1 else pd.Series(pd.NA, index=df.index)

    kab_counts = kab.fillna("(no comma)").value_counts().rename_axis("kabupaten_part")
    kab_counts = kab_counts.reset_index(name="count")
    kab_counts["in KABUPATEN_MAP"] = kab_counts["kabupaten_part"].map(
        lambda k: "yes" if str(k).lower() in config.KABUPATEN_MAP else "**no**"
    )

    top = (
        area.value_counts()
        .head(config.PROFILE_TOP_AREAS)
        .rename_axis("area_raw")
        .reset_index(name="count")
    )

    slug = df[COL_URL].astype(str).str.extract(URL_SLUG_RE, expand=False)
    cross = pd.crosstab(kab.fillna("(no comma)"), slug.fillna("(none)"))
    cross.index.name = "kabupaten_part \\ url_slug"

    yog = area[kab.str.lower() == "yogyakarta"].value_counts().head(config.PROFILE_TOP_AREAS)
    yog = yog.rename_axis("area_raw (kabupaten part = 'Yogyakarta')").reset_index(name="count")

    kab_names = set(config.KABUPATEN_MAP)
    is_kab_name = area.str.lower().map(lambda a: a.replace(" ", "") in {k.replace(" ", "") for k in kab_names})
    kabname = (
        pd.DataFrame({"area_raw": area[is_kab_name], "kabupaten_part": kab[is_kab_name]})
        .value_counts()
        .reset_index(name="count")
    )

    text = (
        "## 4. Location (`listing-location`, split on last comma)\n\n"
        f"- Rows without a comma: **{int((~has_comma).sum())}**\n"
        f"- Distinct kabupaten parts: **{kab.nunique()}**\n"
        f"- Distinct area values: **{area.nunique()}**\n\n"
        "### Kabupaten part counts\n\n"
        + md_table(kab_counts)
        + "\n\n### Kabupaten part vs URL city slug\n\n"
        + md_table(cross, index=True)
        + f"\n\n### Top {config.PROFILE_TOP_AREAS} area values\n\n"
        + md_table(top)
        + "\n\n### Areas listed with kabupaten part 'Yogyakarta'\n\n"
        "_'Yogyakarta' may mean Kota Yogyakarta or the province (DIY). "
        "Names here that are outside the city signal the ambiguity._\n\n"
        + md_table(yog)
        + "\n\n### Area part equal to a kabupaten/kota name\n\n"
        f"Rows: **{int(is_kab_name.sum())}**. "
        "_'Sleman' and 'Bantul' are also kecamatan names, so these may or may not "
        "mean the kecamatan; 'Kulonprogo'/'Yogyakarta' are not kecamatan._\n\n"
        + md_table(kabname)
        + "\n"
    )
    return text, kab


def section_ids_dups(df: pd.DataFrame) -> str:
    """Step 5: listing ID, city slug, duplicates."""
    url = df[COL_URL].astype(str)
    id_kind = pd.Series("No ID pattern", index=df.index)
    ids = pd.Series(pd.NA, index=df.index, dtype="object")
    for name, pat in reversed(URL_ID_PATTERNS):
        ext = url.str.extract(pat, expand=False)
        id_kind[ext.notna()] = name
        ids[ext.notna()] = ext[ext.notna()]
    kinds = id_kind.value_counts().rename_axis("id_pattern").reset_index(name="count")
    kinds["examples"] = kinds["id_pattern"].map(
        lambda k: "; ".join(url[id_kind == k].head(config.PROFILE_EXAMPLES_PER_PATTERN))
    )
    slug = url.str.extract(URL_SLUG_RE, expand=False)
    slugs = slug.fillna("(none)").value_counts().rename_axis("url_slug").reset_index(name="count")

    exact_extra = int(df.duplicated().sum())
    exact_groups = int(df[df.duplicated(keep=False)].drop_duplicates().shape[0])

    id_dup_mask = ids.notna() & ids.duplicated(keep=False)
    id_extra = int((ids.notna() & ids.duplicated()).sum())
    id_groups = int(ids[id_dup_mask].nunique())
    differing = {}
    if id_groups:
        g = df[id_dup_mask].groupby(ids[id_dup_mask])
        for col in df.columns:
            differing[col] = int((g[col].nunique(dropna=False) > 1).sum())
    diff_txt = (
        ", ".join(f"`{c}` in {n} groups" for c, n in differing.items() if n) or "none (all identical)"
    )

    key = df[NEAR_DUP_KEY].astype(str)
    near_mask = key.duplicated(keep=False)
    near_extra = int(key.duplicated().sum())
    near_groups = int(key[near_mask].drop_duplicates().shape[0])
    near_multi_id = 0
    if near_groups:
        grp_key = key[near_mask].apply(tuple, axis=1)
        near_multi_id = int((ids[near_mask].groupby(grp_key).nunique() > 1).sum())
    ex = df[near_mask].sort_values(NEAR_DUP_KEY).head(config.PROFILE_N_SAMPLES)
    ex = ex.assign(listing_id=ids[ex.index])[["listing_id"] + NEAR_DUP_KEY + [COL_DESC]]

    return (
        "## 5. Listing IDs and duplicates (`nav-link`)\n\n"
        "### ID patterns\n\n"
        + md_table(kinds)
        + "\n\n### City slugs in URL\n\n"
        + md_table(slugs)
        + "\n\n### Duplicates\n\n"
        "| check | groups | extra rows (would be removed) |\n|---|---|---|\n"
        f"| Exact duplicate rows (all 9 columns) | {exact_groups} | {exact_extra} |\n"
        f"| Duplicate listing ID | {id_groups} | {id_extra} |\n"
        f"| Same (location, price, land, building, bed, bath) | {near_groups} | {near_extra} |\n\n"
        f"- Columns that differ inside duplicate-ID groups: {diff_txt}\n"
        f"- Near-duplicate groups containing more than one listing ID "
        f"(same house, different listings/agents): **{near_multi_id}**\n\n"
        f"Examples of near-duplicate groups (first {config.PROFILE_N_SAMPLES} rows, sorted):\n\n"
        + md_table(ex)
        + "\n"
    )


def section_rooms(df: pd.DataFrame) -> str:
    """Step 6: bedroom/bathroom/carport distributions."""
    parts = ["## 6. Bedrooms, bathrooms, carport\n"]
    for col in (COL_BED, COL_BATH, COL_CARPORT):
        vc = df[col].value_counts(dropna=False).sort_index()
        labels = ["NaN" if pd.isna(v) else f"{v:g}" for v in vc.index]
        t = pd.DataFrame([vc.values], columns=labels)
        t.insert(0, col, "rows")
        parts.append(f"### `{col}`\n\n" + md_table(t) + "\n")
    k = config.KOS_MIN_BEDROOMS
    parts.append(
        f"- Rows with bed ≥ {k} (`KOS_MIN_BEDROOMS`): **{int((df[COL_BED] >= k).sum())}**\n"
        f"- Rows with bath ≥ {k}: **{int((df[COL_BATH] >= k).sum())}**\n"
        f"- Rows with bed = 0: **{int((df[COL_BED] == 0).sum())}**, "
        f"bath = 0: **{int((df[COL_BATH] == 0).sum())}**\n"
    )

    kos = df[COL_DESC].astype(str).str.contains(KOS_BROAD_RE, case=False, regex=True)
    rows = []
    for cut in range(5, 13):
        big = df[COL_BED] >= cut
        rows.append({"bed ≥": cut, "rows": int(big.sum()), "with kos keyword": int((big & kos).sum()),
                     "without kos keyword": int((big & ~kos).sum())})
    rows.append({"bed ≥": "any", "rows": len(df), "with kos keyword": int(kos.sum()),
                 "without kos keyword": int((~kos).sum())})
    no_kw = df[(df[COL_BED] >= k) & ~kos].sort_values(COL_BED)
    no_kw = no_kw[[COL_BED, COL_BATH, COL_LAND, COL_BUILDING, COL_PRICE, COL_DESC]]
    parts.append(
        "### Bedroom cutoff vs kos keyword\n\n"
        + md_table(pd.DataFrame(rows))
        + f"\n\nRows with bed ≥ {k} and **no** kos keyword "
        f"(would be dropped by the bedroom rule alone), first {config.PROFILE_TOP_AREAS}:\n\n"
        + md_table(no_kw.head(config.PROFILE_TOP_AREAS))
        + "\n"
    )
    return "\n".join(parts)


def section_description(df: pd.DataFrame) -> str:
    """Step 7: random description examples and keyword counts."""
    desc = df[COL_DESC].astype(str)
    length = desc.str.len()
    sample = df.sample(config.PROFILE_N_SAMPLES, random_state=config.RANDOM_STATE)
    def kw_table(keywords: dict[str, str]) -> pd.DataFrame:
        rows = []
        for name, pat in keywords.items():
            hit = desc.str.contains(pat, case=False, regex=True)
            rows.append({"keyword": name, "regex": f"`{pat}`", "rows": int(hit.sum()),
                         "share_%": round(hit.mean() * 100, 1)})
        return pd.DataFrame(rows)

    return (
        "## 7. Description\n\n"
        "_The `description` column holds the listing **title** (max 255 chars), "
        "not the full listing text._\n\n"
        f"- Length (chars): min {length.min()}, median {length.median():.0f}, max {length.max()}\n"
        f"- Distinct descriptions: **{desc.nunique():,}** of {len(desc):,}\n\n"
        "### Keyword counts (case-insensitive, whole word)\n\n"
        + md_table(kw_table(DESC_KEYWORDS))
        + "\n\n### Extra keywords found while profiling\n\n"
        + md_table(kw_table(EXTRA_KEYWORDS))
        + f"\n\n### {config.PROFILE_N_SAMPLES} random examples (seed {config.RANDOM_STATE})\n\n"
        + md_table(sample[[COL_PRICE, COL_LOC, COL_BED, COL_LAND, COL_BUILDING, COL_DESC]])
        + "\n"
    )


def section_misalignment(df: pd.DataFrame, price: pd.Series) -> str:
    """Step 8: prices mentioned in the description vs the price field."""
    desc = df[COL_DESC].astype(str)
    mentions = desc.map(desc_price_mentions)
    has = mentions.map(len) > 0

    records = []
    for idx in df.index[has]:
        for m in mentions[idx]:
            p = price[idx]
            ratio = (max(m["value_idr"], p) / min(m["value_idr"], p)) if pd.notna(p) and p > 0 else None
            records.append({"row": idx, **m, "price_field": df.at[idx, COL_PRICE],
                            "price_idr": p, "ratio": ratio})
    rec = pd.DataFrame(records)

    lim = config.DESC_PRICE_MISMATCH_RATIO
    if rec.empty:
        return "## 8. Possible column misalignment\n\n_No price mentions found in descriptions._\n"

    rec["mismatch"] = rec["ratio"] > lim
    clean = rec[~rec["non_price_context"]]
    ctx = rec[rec["non_price_context"]]
    rows_mismatch_all = rec.loc[rec["mismatch"], "row"].nunique()
    rows_mismatch_clean = clean.loc[clean["mismatch"], "row"].nunique()
    rows_ctx = ctx["row"].nunique()

    def with_desc(t: pd.DataFrame) -> pd.DataFrame:
        return t.assign(
            description=desc[t["row"]].values,
            desc_value=t["value_idr"].map(fmt_int),
            ratio=t["ratio"].round(2),
        )[["row", "price_field", "match", "desc_value", "ratio", "mismatch", "description"]]

    ex = pd.concat([clean[clean["mismatch"]], clean[~clean["mismatch"]]])
    ex = with_desc(ex.drop_duplicates("row").head(config.PROFILE_N_SAMPLES))
    mm = with_desc(rec[rec["mismatch"]].drop_duplicates("row"))
    mm = mm.assign(non_price_context=rec[rec["mismatch"]].drop_duplicates("row")["non_price_context"].values)

    eq = rec[rec["dp_context"] & (rec["ratio"] <= config.DP_PRICE_EQUAL_RATIO)]
    eq = eq.drop_duplicates("row")
    eq_tbl = with_desc(eq)
    key = df[NEAR_DUP_KEY].astype(str).apply(tuple, axis=1)
    sus_keys = set(key[eq["row"]])
    n_siblings = int(key.isin(sus_keys).sum())

    return (
        "## 8. Possible column misalignment (price in description vs price field)\n\n"
        "_Mention regex: number + `miliar|milyar|juta|jutaan|jt|M` "
        f"(uppercase bare `M` only, and only if the number < {config.DESC_BARE_M_MAX}; "
        "lowercase `m` = meters). **Non-price context** = installment, down payment (DP), KPR, "
        "discount (diskon/potongan), promo, bonus, 'cukup bayar', or road access width "
        f"(akses/jalan/lebar) within {config.DESC_PRICE_CONTEXT_CHARS} chars of the mention._\n\n"
        f"- Rows with ≥ 1 price mention: **{int(has.sum())}** ({has.mean() * 100:.1f}%)\n"
        f"- Rows with a mention in non-price context: **{rows_ctx}**\n"
        f"- Rows with a mention differing from the price field by > {lim:g}× (any context): "
        f"**{rows_mismatch_all}**\n"
        f"- … excluding non-price-context mentions (candidate misalignment): "
        f"**{rows_mismatch_clean}**\n\n"
        f"### {config.PROFILE_N_SAMPLES} examples next to the price field "
        "(plain context; mismatches first)\n\n"
        + md_table(ex)
        + f"\n\n### Every mention > {lim:g}× off the price field\n\n"
        + (md_table(mm) if len(mm) else "_None._")
        + "\n\n### Price field ≈ a down-payment / installment amount in the text\n\n"
        "_Mention next to DP / uang muka / booking / UTJ / cicilan / angsuran, within "
        f"{config.DP_PRICE_EQUAL_RATIO:g}× of the price field. The ratio check above cannot "
        "see these, yet the price field may hold the down payment, not the full price._\n\n"
        f"- Rows: **{len(eq_tbl)}**; rows sharing their (location, price, land, building, bed, "
        f"bath) key: **{n_siblings}**\n\n"
        + (md_table(eq_tbl) if len(eq_tbl) else "_None._")
        + "\n"
    )


def main() -> None:
    """Profile the raw CSV and write the markdown report."""
    df = pd.read_csv(config.RAW_CSV)

    s1 = section_overview(df)
    s2, price = section_price(df)
    s3, land, building = section_areas(df, price)
    s4, kab = section_location(df)
    s5 = section_ids_dups(df)
    s6 = section_rooms(df)
    s7 = section_description(df)
    s8 = section_misalignment(df, price)

    header = (
        "# Data profile — raw Rumah123 listings\n\n"
        f"Source: `{config.RAW_CSV.relative_to(config.ROOT).as_posix()}` "
        "(read-only). Generated by `python -m src.profile_data`. No cleaning applied; "
        "approximate numeric conversions are for description only.\n"
    )
    report = "\n\n".join([header, s1, s2, s3, s4, s5, s6, s7, s8])
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_PROFILE_MD.write_text(report, encoding="utf-8")

    n_price_unmatched = int(df[COL_PRICE].map(classify_price).isin(["No pattern"]).sum())
    print(f"Rows: {len(df):,}  Columns: {df.shape[1]}")
    print(f"Price values matching no pattern: {n_price_unmatched}")
    print(f"Land parsed: {land.notna().sum():,}  Building parsed: {building.notna().sum():,}")
    print(f"Kabupaten parts: {kab.value_counts().to_dict()}")
    print(f"Exact duplicate rows (extra): {int(df.duplicated().sum())}")
    print(f"Rows with bed >= {config.KOS_MIN_BEDROOMS}: {int((df[COL_BED] >= config.KOS_MIN_BEDROOMS).sum())}")
    print(f"Wrote {config.DATA_PROFILE_MD.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
