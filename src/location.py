"""P3a — Map listing area names to official kecamatan.

Inputs:  data/external/IndoArea-19-04-2024.zip (hanifabd/wilayah-indonesia-area, BPS/Kemendagri)
         data/interim/listings_clean.csv
Outputs: data/external/kecamatan_diy.csv   (kabupaten, kecamatan, kecamatan_code) — 78 rows
         data/external/desa_diy.csv        (kabupaten, kecamatan, desa, desa_code)
         data/external/area_to_kecamatan.csv  (draft mapping, reviewed by hand)

Run:   python -m src.location           build lists + draft mapping (keeps reviewed rows)
       python -m src.location --apply   add kecamatan to listings_clean.csv (after review)
"""

from __future__ import annotations

import argparse
import difflib
import json
import zipfile

import pandas as pd

from src import config
from src.utils import normalize_name

MAPPING_COLUMNS = [
    "area_raw", "kabupaten_raw", "kecamatan", "kabupaten", "match_type", "n_listings", "needs_review", "note",
]
MATCH_ORDER = {"auto": 0, "auto_corrected": 1, "suggested": 2, "manual": 3, "unmatched": 4}


def match_key(name: object) -> str:
    """Comparison key: normalize_name with all spaces removed ("Condong Catur" == "Condongcatur")."""
    return normalize_name(str(name)).replace(" ", "") if pd.notna(name) else ""


def official_kabupaten(raw: str) -> str:
    """'KAB. GUNUNGKIDUL' -> 'Gunungkidul', 'KOTA YOGYAKARTA' -> 'Kota Yogyakarta' (via KABUPATEN_MAP)."""
    n = normalize_name(raw)
    for prefix in ("kab ", "kabupaten ", "kota "):
        if n.startswith(prefix):
            n = n[len(prefix):]
    if n not in config.KABUPATEN_MAP:
        raise ValueError(f"Unknown DIY kabupaten in wilayah data: {raw!r}")
    return config.KABUPATEN_MAP[n]


def build_admin_lists() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read DIY kecamatan and desa from the wilayah zip; write both CSVs. Returns (kecamatan, desa)."""
    p = config.DIY_PROVINCE_CODE
    with zipfile.ZipFile(config.WILAYAH_ZIP) as z:
        kabs = json.loads(z.read(f"kabupaten_kota/kab-{p}.json"))
        kec_rows, desa_rows = [], []
        for kab_code, kab_raw in kabs.items():
            kab = official_kabupaten(kab_raw)
            kecs = json.loads(z.read(f"kecamatan/kec-{p}-{kab_code}.json"))
            for kec_code, kec_raw in kecs.items():
                kec = kec_raw.title()
                kec_rows.append({"kabupaten": kab, "kecamatan": kec, "kecamatan_code": f"{p}{kab_code}{kec_code}"})
                desas = json.loads(z.read(f"kelurahan_desa/keldesa-{p}-{kab_code}-{kec_code}.json"))
                for desa_code, desa_raw in desas.items():
                    desa_rows.append({"kabupaten": kab, "kecamatan": kec, "desa": desa_raw.title(),
                                      "desa_code": f"{p}{kab_code}{kec_code}{desa_code}"})
    kec_df = pd.DataFrame(kec_rows).sort_values("kecamatan_code").reset_index(drop=True)
    desa_df = pd.DataFrame(desa_rows).sort_values("desa_code").reset_index(drop=True)
    if len(kec_df) != config.N_KECAMATAN_DIY:
        raise ValueError(f"Expected {config.N_KECAMATAN_DIY} kecamatan, got {len(kec_df)}")
    config.EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    kec_df.to_csv(config.KECAMATAN_DIY_CSV, index=False, encoding="utf-8")
    desa_df.to_csv(config.DESA_DIY_CSV, index=False, encoding="utf-8")
    return kec_df, desa_df


def _row(area: str, kab_raw: str, n: int, kec: str | None, kab: str | None, match_type: str,
         needs_review: bool, note: str = "") -> dict:
    return {"area_raw": area, "kabupaten_raw": kab_raw, "kecamatan": kec, "kabupaten": kab,
            "match_type": match_type, "n_listings": n, "needs_review": needs_review, "note": note}


def match_area(area: str, kab_raw: str, n: int, kec_df: pd.DataFrame, desa_df: pd.DataFrame) -> dict:
    """Map one (area_raw, kabupaten_raw) pair. Order: kecamatan exact → desa exact → fuzzy → locality list."""
    key = match_key(area)
    kab_hint = config.KABUPATEN_MAP.get(normalize_name(kab_raw))
    ambiguous = normalize_name(kab_raw) in config.AMBIGUOUS_KABUPATEN_PARTS
    is_kab_name = key in {match_key(k) for k in config.KABUPATEN_MAP}

    hits = kec_df[kec_df["kecamatan"].map(match_key) == key]
    if len(hits):
        same = hits[hits["kabupaten"] == kab_hint]
        notes, review = [], False
        if is_kab_name:
            notes.append("Area equals a kabupaten name: may mean the whole kabupaten, not this kecamatan.")
            review = True
        if len(same):
            h = same.iloc[0]
            if ambiguous and len(hits) > 1:
                others = ", ".join(f"{r.kecamatan} ({r.kabupaten})" for r in hits[hits["kabupaten"] != kab_hint].itertuples())
                notes.append(f"Kabupaten part 'Yogyakarta' is ambiguous and the name also exists in {others}.")
                review = True
            return _row(area, kab_raw, n, h.kecamatan, h.kabupaten, "auto", review, " ".join(notes))
        if len(hits) == 1:
            h = hits.iloc[0]
            notes.insert(0, f"Kabupaten part '{kab_raw}' corrected to {h.kabupaten}.")
            return _row(area, kab_raw, n, h.kecamatan, h.kabupaten, "auto_corrected", review, " ".join(notes))
        opts = "; ".join(f"{r.kecamatan} ({r.kabupaten})" for r in hits.itertuples())
        return _row(area, kab_raw, n, None, None, "unmatched", True, f"Name matches several kecamatan: {opts}.")

    dh = desa_df[desa_df["desa"].map(match_key) == key]
    if len(dh):
        pref = dh if ambiguous else dh[dh["kabupaten"] == kab_hint]
        pref = pref if len(pref) else dh
        if len(pref) == 1:
            h = pref.iloc[0]
            moved = "" if h.kabupaten == kab_hint else f" Kabupaten part '{kab_raw}' would change to {h.kabupaten}."
            return _row(area, kab_raw, n, h.kecamatan, h.kabupaten, "suggested", True,
                        f"Official desa/kelurahan '{h.desa}' ({h.desa_code}) is in kecamatan {h.kecamatan}.{moved}")
        opts = "; ".join(f"desa {r.desa} → {r.kecamatan} ({r.kabupaten})" for r in pref.itertuples())
        return _row(area, kab_raw, n, None, None, "unmatched", True, f"Several official desa share this name: {opts}.")

    keys = kec_df["kecamatan"].map(match_key)
    close = difflib.get_close_matches(key, keys.tolist(), n=1, cutoff=config.FUZZY_MATCH_CUTOFF)
    if close:
        h = kec_df[keys == close[0]]
        h = h[h["kabupaten"] == kab_hint] if (h["kabupaten"] == kab_hint).any() else h
        if len(h) == 1:
            h = h.iloc[0]
            ratio = difflib.SequenceMatcher(None, key, close[0]).ratio()
            return _row(area, kab_raw, n, h.kecamatan, h.kabupaten, "suggested", True,
                        f"Spelling variant of official kecamatan '{h.kecamatan}' (similarity {ratio:.2f}).")

    if key in config.LOCALITY_SUGGESTIONS:
        kec, kab, note = config.LOCALITY_SUGGESTIONS[key]
        ok = ((kec_df["kecamatan"] == kec) & (kec_df["kabupaten"] == kab)).any()
        if not ok:
            raise ValueError(f"LOCALITY_SUGGESTIONS entry {key!r} names unknown kecamatan {kec} ({kab})")
        return _row(area, kab_raw, n, kec, kab, "suggested", True, f"General knowledge, not official data. {note}")

    return _row(area, kab_raw, n, None, None, "unmatched", True, "No official or known match.")


def build_mapping(clean_df: pd.DataFrame, kec_df: pd.DataFrame, desa_df: pd.DataFrame) -> pd.DataFrame:
    """Draft mapping for every distinct (area_raw, kabupaten_raw) in the clean data."""
    pairs = clean_df.groupby(["area_raw", "kabupaten_raw"], dropna=False).size().reset_index(name="n")
    rows = [match_area(r.area_raw, r.kabupaten_raw, int(r.n), kec_df, desa_df) for r in pairs.itertuples()]
    for row in rows:
        extra = config.AREA_REVIEW_NOTES.get(match_key(row["area_raw"]))
        if extra:
            row["note"] = f"{row['note']} {extra}".strip()
            row["needs_review"] = True
    out = pd.DataFrame(rows, columns=MAPPING_COLUMNS)
    out["_o"] = out["match_type"].map(MATCH_ORDER)
    return out.sort_values(["_o", "needs_review", "n_listings"], ascending=[True, True, False]).drop(columns="_o")


def merge_with_existing(draft: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep rows already in area_to_kecamatan.csv (may hold manual review); add only new pairs.

    `n_listings` is refreshed from the draft. Returns (mapping, n_preserved).
    """
    path = config.AREA_TO_KECAMATAN_CSV
    if not path.exists():
        return draft, 0
    old = pd.read_csv(path, dtype={"kecamatan": "string", "kabupaten": "string", "note": "string"})
    k = ["area_raw", "kabupaten_raw"]
    old = old.drop(columns="n_listings").merge(draft[k + ["n_listings"]], on=k, how="inner")
    new = draft.merge(old[k], on=k, how="left", indicator=True)
    new = new[new["_merge"] == "left_only"].drop(columns="_merge")
    return pd.concat([old[MAPPING_COLUMNS], new[MAPPING_COLUMNS]], ignore_index=True), len(old)


def apply_mapping(df: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add `kecamatan` and corrected `kabupaten` to clean listings. Returns (mapped, dropped).

    Refuses to run while any mapping row still has needs_review = True. Rows whose
    pair has no kecamatan are returned as dropped with drop_reason = "kecamatan_unmatched".
    """
    pending = mapping[mapping["needs_review"].astype(str).str.lower().isin(["true", "1"])]
    if len(pending):
        raise ValueError(f"{len(pending)} mapping rows still need review: "
                         + ", ".join(f"{a} ({k})" for a, k in zip(pending.area_raw, pending.kabupaten_raw)))
    k = ["area_raw", "kabupaten_raw"]
    m = mapping[k + ["kecamatan", "kabupaten"]].rename(columns={"kabupaten": "kabupaten_mapped"})
    out = df.drop(columns=["kecamatan"], errors="ignore").merge(m, on=k, how="left", validate="many_to_one")
    unmatched = out["kecamatan"].isna()
    out.loc[~unmatched, "kabupaten"] = out.loc[~unmatched, "kabupaten_mapped"]
    out = out.drop(columns="kabupaten_mapped")
    cols = [c for c in out.columns if c != "kecamatan"]
    at = cols.index("area_raw") + 1 if "area_raw" in cols else len(cols)
    out = out[cols[:at] + ["kecamatan"] + cols[at:]]
    dropped = out[unmatched].assign(drop_reason="kecamatan_unmatched", drop_step="P3a. Apply kecamatan mapping")
    return out[~unmatched].reset_index(drop=True), dropped


def _summary(mapping: pd.DataFrame) -> None:
    by = mapping.groupby(["match_type", "needs_review"])["n_listings"].agg(["count", "sum"])
    print("Mapping summary (pairs, listings):")
    for (mt, nr), r in by.iterrows():
        print(f"  {mt:<15} needs_review={nr!s:<5}  pairs={r['count']:>3}  listings={r['sum']:>5}")


def main() -> None:
    """Build official lists and the draft mapping, or apply a reviewed mapping with --apply."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="apply reviewed mapping to listings_clean.csv")
    args = ap.parse_args()

    clean_df = pd.read_csv(config.CLEAN_CSV)

    if args.apply:
        mapping = pd.read_csv(config.AREA_TO_KECAMATAN_CSV)
        mapped, dropped = apply_mapping(clean_df, mapping)
        old_dropped = pd.read_csv(config.DROPPED_CSV)
        old_dropped = old_dropped[old_dropped["drop_reason"] != "kecamatan_unmatched"]
        pd.concat([old_dropped, dropped], ignore_index=True).to_csv(config.DROPPED_CSV, index=False, encoding="utf-8")
        mapped.to_csv(config.CLEAN_CSV, index=False, encoding="utf-8")
        print(f"Applied mapping: {len(clean_df):,} -> {len(mapped):,} rows "
              f"({len(dropped)} dropped as kecamatan_unmatched)")
        print(f"Listings per kabupaten: {mapped['kabupaten'].value_counts().to_dict()}")
        vc = mapped.groupby(["kecamatan", "kabupaten"]).size()
        print(f"Kecamatan with listings: {len(vc)}; with >= {config.MIN_LISTINGS_FOR_MAP} listings: "
              f"{int((vc >= config.MIN_LISTINGS_FOR_MAP).sum())}")
        return

    kec_df, desa_df = build_admin_lists()
    print(f"Official lists: {len(kec_df)} kecamatan, {len(desa_df)} desa/kelurahan "
          f"-> {config.KECAMATAN_DIY_CSV.name}, {config.DESA_DIY_CSV.name}")
    draft = build_mapping(clean_df, kec_df, desa_df)
    mapping, kept = merge_with_existing(draft)
    mapping.to_csv(config.AREA_TO_KECAMATAN_CSV, index=False, encoding="utf-8")
    if kept:
        print(f"Kept {kept} existing rows from {config.AREA_TO_KECAMATAN_CSV.name} (review preserved)")
    print(f"Pairs: {len(mapping)}  listings: {int(mapping['n_listings'].sum()):,}")
    _summary(mapping)
    print(f"Wrote {config.AREA_TO_KECAMATAN_CSV.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
