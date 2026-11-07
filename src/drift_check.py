"""P7b U4 — Drift check: score hand-collected 2026 listings with the 2024 model.

Each row of data/external/drift_check_2026.csv (20–30 current listings copied from public listing pages; format in
drift_check_2026_EXAMPLE.csv) goes through src.predict.predict_house. Compares the
asking price with the unadjusted (2024) and the adjusted (× PRICE_INDEX_FACTOR)
outputs: MAPE, MdAPE, range coverage, mean signed error, and the same by kabupaten.

This sample is an evaluation set only: PRICE_INDEX_FACTOR is never tuned on it. If
adjusted estimates are still clearly low, that is reported as a finding, not fixed.

Inputs:  data/external/drift_check_2026.csv, models/, data/app/
Outputs: reports/drift_check.md, data/app/drift_summary.json (About page)
         With no rows: prints "No drift rows yet", writes a short note to
         reports/drift_check.md, removes a stale drift_summary.json, exits cleanly.

Run: python -m src.drift_check [path/to/rows.csv]
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from src import config
from src.predict import predict_house, price_adjustment
from src.utils import format_idr

NUMERIC = ("land_m2", "building_m2", "bedrooms", "bathrooms", "carport", "road_km", "asking_price_idr")
VARIANTS = {"unadjusted": ("2024 prices (unadjusted)", "_2024_idr"), "adjusted": ("Adjusted", "_idr")}


def load_rows(path: Path) -> pd.DataFrame:
    """Read the sheet; the header must have exactly config.DRIFT_COLUMNS (any order)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in config.DRIFT_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in config.DRIFT_COLUMNS]
    if missing or extra:
        raise SystemExit(f"{path.name}: column mismatch (missing {missing}, unexpected {extra}); "
                         f"see {config.DRIFT_EXAMPLE_CSV.name}")
    df = df.apply(lambda s: s.str.strip())
    return df[(df != "").any(axis=1)].reset_index(drop=True)


def _num(v: str) -> float | None:
    return None if v == "" else float(v)


def row_problem(row: pd.Series) -> str | None:
    """Why a row cannot be scored (None = fine). Nothing is guessed or repaired."""
    try:
        pd.Timestamp(row["date_seen"])
    except ValueError:
        return f"date_seen {row['date_seen']!r} is not YYYY-MM-DD"
    if row["kabupaten"] not in config.KABUPATEN_ORDER:
        return f"unknown kabupaten {row['kabupaten']!r}"
    try:
        vals = {c: _num(row[c]) for c in NUMERIC}
    except ValueError:
        return "a numeric column has text (use plain numbers, dot decimals)"
    for c in ("land_m2", "building_m2", "asking_price_idr"):
        if vals[c] is None or vals[c] <= 0:
            return f"{c} must be a number > 0"
    bad_flags = [f for f in config.ACTIVE_FLAGS if row[f] not in ("", "0", "1")]
    if bad_flags:
        return f"flags must be 0, 1 or empty: {bad_flags}"
    for c in ("bedrooms", "bathrooms", "carport"):
        if vals[c] is not None and (vals[c] < 0 or vals[c] != int(vals[c])):
            return f"{c} must be a whole number ≥ 0 (or empty)"
    return p2_rule_problem(vals)


def p2_rule_problem(vals: dict) -> str | None:
    """Same sanity limits as P2 cleaning: a row outside them would not be in the training data either."""
    price, land, building = vals["asking_price_idr"], vals["land_m2"], vals["building_m2"]
    if not config.MIN_PRICE_IDR <= price <= config.MAX_PRICE_IDR:
        return f"asking price {format_idr(int(price))} outside the P2 range"
    if not config.MIN_LAND_M2 <= land <= config.MAX_LAND_M2:
        return f"land_m2 {land:g} outside {config.MIN_LAND_M2}–{config.MAX_LAND_M2:,}"
    if not config.MIN_BUILDING_M2 <= building <= config.MAX_BUILDING_M2:
        return f"building_m2 {building:g} outside {config.MIN_BUILDING_M2}–{config.MAX_BUILDING_M2:,}"
    if building > config.MAX_BUILDING_TO_LAND_RATIO * land:
        return f"building more than {config.MAX_BUILDING_TO_LAND_RATIO}× land (P2 rule)"
    ppl = price / land
    if not config.MIN_PRICE_PER_LAND_M2 <= ppl <= config.MAX_PRICE_PER_LAND_M2:
        return f"price per land m² {format_idr(int(ppl))} outside the P2 range"
    if vals["bedrooms"] is not None and vals["bedrooms"] >= config.KOS_MIN_BEDROOMS:
        return f"{vals['bedrooms']:g} bedrooms (≥ {config.KOS_MIN_BEDROOMS}: treated as kos in P2)"
    return None


def data_checks(df: pd.DataFrame) -> dict:
    """Empty optional values (kept, not guessed) and duplicate URLs."""
    return {
        "missing": {c: int((df[c] == "").sum()) for c in ("bedrooms", "bathrooms", "carport")},
        "duplicate_urls": int(df["url"].duplicated().sum()),
    }


def to_inputs(row: pd.Series) -> dict:
    """predict_house inputs from one sheet row (road_name / road_km are not model inputs)."""
    inputs = {c: _num(row[c]) for c in ("land_m2", "building_m2", "bedrooms", "bathrooms", "carport")}
    inputs |= {"kabupaten": row["kabupaten"], "kecamatan": row["kecamatan"]}
    inputs |= {f: row[f] == "1" for f in config.ACTIVE_FLAGS}
    return inputs


def score(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Predictions per valid row and a list of skipped rows with their reason."""
    rows, skipped = [], []
    dup = df["url"].duplicated() & (df["url"] != "")
    for i, row in df.iterrows():
        problem = "duplicate URL (same listing twice)" if dup[i] else row_problem(row)
        if problem is None:
            try:
                out = predict_house(to_inputs(row))
            except ValueError as exc:
                problem = str(exc)
        if problem:
            skipped.append({"row": int(i) + 2, "url": row["url"], "reason": problem})
            continue
        rows.append({"date_seen": row["date_seen"], "url": row["url"], "kabupaten": row["kabupaten"],
                     "kecamatan": row["kecamatan"], "asking_price_idr": int(float(row["asking_price_idr"])),
                     **{k: out[k] for k in ("estimate_2024_idr", "low_2024_idr", "high_2024_idr",
                                            "estimate_idr", "low_idr", "high_idr")}})
    return pd.DataFrame(rows), skipped


def metrics(scored: pd.DataFrame, suffix: str) -> dict:
    """MAPE, MdAPE, coverage and mean signed error (%) of one variant vs the asking price."""
    ask = scored["asking_price_idr"].astype(float)
    est = scored[f"estimate{suffix}"].astype(float)
    signed = (est - ask) / ask * 100
    inside = (ask >= scored[f"low{suffix}"]) & (ask <= scored[f"high{suffix}"])
    return {"n": int(len(scored)), "mape": float(signed.abs().mean()), "mdape": float(signed.abs().median()),
            "coverage": float(inside.mean() * 100), "mean_signed_error": float(signed.mean()),
            "median_signed_error": float(signed.median())}


def price_band_table(scored: pd.DataFrame) -> list[dict]:
    """Adjusted signed error by asking-price band (same bands as P5 error analysis)."""
    band = pd.cut(scored["asking_price_idr"], bins=list(config.PRICE_BANDS), labels=list(config.PRICE_BAND_LABELS),
                  right=False)
    out = []
    for label in config.PRICE_BAND_LABELS:
        part = scored[band == label]
        if not part.empty:
            m = metrics(part, "_idr")
            out.append({"band": label, "n": m["n"], "mape": m["mape"], "median_signed_error": m["median_signed_error"],
                        "coverage": m["coverage"]})
    return out


def overall_table(scored: pd.DataFrame) -> list[dict]:
    return [{"prices": label, **metrics(scored, suffix)} for label, suffix in VARIANTS.values()]


def kabupaten_table(scored: pd.DataFrame) -> list[dict]:
    out = []
    for kab in config.KABUPATEN_ORDER:
        part = scored[scored["kabupaten"] == kab]
        if part.empty:
            continue
        u, a = metrics(part, "_2024_idr"), metrics(part, "_idr")
        out.append({"kabupaten": kab, "n": u["n"], "mape_unadjusted": u["mape"], "mape_adjusted": a["mape"],
                    "coverage_adjusted": a["coverage"], "mean_signed_error_adjusted": a["mean_signed_error"],
                    "small_n": u["n"] < config.DRIFT_SMALL_KABUPATEN_N})
    return out


def findings(overall: list[dict], adj: dict | None, n: int) -> list[str]:
    """Plain statements for the report; never a fix."""
    a = overall[1]
    out = []
    lo, hi = config.DRIFT_TARGET_ROWS
    if n < lo:
        out.append(f"Only {n} rows (target {lo}–{hi}): treat every number here as rough.")
    if adj is None:
        out.append("No price index factor is set, so adjusted = unadjusted (2024 prices).")
    if a["mean_signed_error"] < config.DRIFT_LOW_BIAS_PCT:
        out.append(f"Adjusted estimates are still clearly low (mean signed error {a['mean_signed_error']:+.1f}%). "
                   "Reported as a finding: PRICE_INDEX_FACTOR is never tuned on this sample.")
    elif a["mean_signed_error"] > -config.DRIFT_LOW_BIAS_PCT:
        out.append(f"Adjusted estimates are clearly high (mean signed error {a['mean_signed_error']:+.1f}%). "
                   "Reported as a finding: PRICE_INDEX_FACTOR is never tuned on this sample.")
    return out


def _pct(v: float, signed: bool = False) -> str:
    return (f"{v:+.1f}%" if signed else f"{v:.1f}%").replace("-", "−")


def write_report(scored: pd.DataFrame, overall: list[dict], by_kab: list[dict], skipped: list[dict],
                 adj: dict | None, notes: list[str], bands: list[dict], checks: dict) -> None:
    """reports/drift_check.md."""
    adj_txt = (f"factor {adj['factor']:g} ({(adj['factor'] - 1) * 100:+.0f}%) from {adj['source']}, as of {adj['as_of']}"
               if adj else "none set (`PRICE_INDEX_FACTOR = None`)")
    lines = [
        "# Drift check: 2026 listings vs the 2024 model (P7b U4)", "",
        f"Generated by `python -m src.drift_check` on {date.today().isoformat()}. {len(scored)} listings scored "
        f"(seen {scored['date_seen'].min()} to {scored['date_seen'].max()}), {len(skipped)} skipped. "
        f"Price adjustment: {adj_txt}. Errors compare the estimate with the **asking** price; signed error "
        "< 0 means the model is below the asking price.", "",
        "This is an evaluation set only: the price index factor is never tuned on it.", "",
        "## Data checks", "",
        f"Rows read: {len(scored) + len(skipped)}; scored: {len(scored)}; skipped: {len(skipped)} (reasons at the end). "
        "Rows outside the P2 cleaning limits (price, areas, building-to-land ratio, price per land m², ≥ "
        f"{config.KOS_MIN_BEDROOMS} bedrooms) are skipped like in training. Duplicate URLs: "
        f"{checks['duplicate_urls']}. Empty optional values, kept as missing (not guessed): "
        + ", ".join(f"{c} {n}" for c, n in checks["missing"].items())
        + " (empty carport counts as 0, as in training).", "",
        "## Overall", "",
        "| Prices | Listings | MAPE | MdAPE | Range coverage | Mean signed error | Median signed error |",
        "|---|---|---|---|---|---|---|",
        *[f"| {r['prices']} | {r['n']} | {_pct(r['mape'])} | {_pct(r['mdape'])} | {_pct(r['coverage'])} | "
          f"{_pct(r['mean_signed_error'], True)} | {_pct(r['median_signed_error'], True)} |" for r in overall], "",
        "## By asking-price band (adjusted)", "",
        "| Band | Listings | MAPE | Median signed error | Range coverage |", "|---|---|---|---|---|",
        *[f"| {r['band']} | {r['n']} | {_pct(r['mape'])} | {_pct(r['median_signed_error'], True)} | "
          f"{_pct(r['coverage'])} |" for r in bands], "",
        "## By kabupaten", "",
        f"Counts are small; rows marked * have fewer than {config.DRIFT_SMALL_KABUPATEN_N} listings and say "
        "almost nothing on their own.", "",
        "| Kabupaten | Listings | MAPE unadjusted | MAPE adjusted | Coverage adjusted | Mean signed error adjusted |",
        "|---|---|---|---|---|---|",
        *[f"| {r['kabupaten']}{' *' if r['small_n'] else ''} | {r['n']} | {_pct(r['mape_unadjusted'])} | "
          f"{_pct(r['mape_adjusted'])} | {_pct(r['coverage_adjusted'])} | "
          f"{_pct(r['mean_signed_error_adjusted'], True)} |" for r in by_kab], "",
    ]
    if notes:
        lines += ["## Findings", "", *[f"- {n}" for n in notes], ""]
    lines += ["## Per listing", "",
              "| Seen | Kabupaten | Kecamatan | Asking | Estimate (2024) | Estimate (adjusted) | Adjusted range | Inside |",
              "|---|---|---|---|---|---|---|---|"]
    for r in scored.itertuples():
        inside = r.low_idr <= r.asking_price_idr <= r.high_idr
        lines.append(f"| {r.date_seen} | {r.kabupaten} | {r.kecamatan} | {format_idr(r.asking_price_idr)} | "
                     f"{format_idr(r.estimate_2024_idr)} | {format_idr(r.estimate_idr)} | "
                     f"{format_idr(r.low_idr)} – {format_idr(r.high_idr)} | {'yes' if inside else 'no'} |")
    if skipped:
        lines += ["", "## Skipped rows (not scored)", "", "| Row | URL | Reason |", "|---|---|---|",
                  *[f"| {s['row']} | {s['url']} | {s['reason']} |" for s in skipped]]
    config.DRIFT_REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rel(p: Path) -> str:
    """Repo-relative POSIX path when inside the repo, else the path as given."""
    p = Path(p)
    return p.relative_to(config.ROOT).as_posix() if p.is_relative_to(config.ROOT) else str(p)


def write_empty_note(path: Path) -> None:
    """reports/drift_check.md when the sheet has no rows yet; remove a stale app summary."""
    config.DRIFT_REPORT_MD.write_text(
        "# Drift check: 2026 listings vs the 2024 model (P7b U4)\n\n"
        f"Not run yet: `{_rel(path)}` "
        f"has no rows. Fill {config.DRIFT_TARGET_ROWS[0]}–{config.DRIFT_TARGET_ROWS[1]} current listings "
        f"(format: `{config.DRIFT_EXAMPLE_CSV.name}`), then run `python -m src.drift_check`.\n",
        encoding="utf-8")
    config.DRIFT_SUMMARY_JSON.unlink(missing_ok=True)


def main(path: Path | None = None) -> None:
    """Score the sheet, write the report and the About-page summary."""
    path = Path(path) if path else config.DRIFT_CSV
    if not path.exists():
        raise SystemExit(f"{path} not found; the template is created in P7b U4 part 1")
    df = load_rows(path)
    if df.empty:
        write_empty_note(path)
        print("No drift rows yet")
        return

    adj = price_adjustment()
    checks = data_checks(df)
    scored, skipped = score(df)
    print(f"{len(df)} rows read, {len(scored)} scored, {len(skipped)} skipped")
    for s in skipped:
        print(f"  skipped row {s['row']}: {s['reason']}")
    if scored.empty:
        write_empty_note(path)
        print("No drift rows could be scored")
        return

    overall, by_kab, bands = overall_table(scored), kabupaten_table(scored), price_band_table(scored)
    notes = findings(overall, adj, len(scored))
    write_report(scored, overall, by_kab, skipped, adj, notes, bands, checks)
    summary = {
        "generated": date.today().isoformat(), "n_rows": len(scored), "n_skipped": len(skipped),
        "date_seen": [scored["date_seen"].min(), scored["date_seen"].max()], "adjustment": adj,
        "overall": overall, "by_kabupaten": by_kab, "by_price_band": bands, "findings": notes,
        "data_checks": checks, "skipped": skipped,
    }
    config.APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.DRIFT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for r in overall:
        print(f"  {r['prices']:<26} MAPE {r['mape']:5.1f}%  MdAPE {r['mdape']:5.1f}%  coverage {r['coverage']:5.1f}%  "
              f"mean signed {r['mean_signed_error']:+6.1f}%  median signed {r['median_signed_error']:+6.1f}%")
    for r in bands:
        print(f"  {r['band']:<12} n={r['n']:<3} MAPE {r['mape']:5.1f}%  median signed {r['median_signed_error']:+6.1f}%  "
              f"coverage {r['coverage']:5.1f}%")
    print(f"  data checks: missing {checks['missing']}, duplicate URLs {checks['duplicate_urls']}")
    for r in by_kab:
        print(f"  {r['kabupaten']:<16} n={r['n']:<3} MAPE adj {r['mape_adjusted']:5.1f}%"
              f"{'  (very small n)' if r['small_n'] else ''}")
    for n in notes:
        print(f"  NOTE: {n}")
    print(f"Wrote {_rel(config.DRIFT_REPORT_MD)}, {_rel(config.DRIFT_SUMMARY_JSON)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
