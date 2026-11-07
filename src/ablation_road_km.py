"""P7b U1 — Ablation: does road_km / has_road_km improve the model?

Runs the same model grid, folds and selection rule as src.train (best per family by
spatial-CV MAPE) twice: without and with the two road km features. CV only — it
writes no models and does not touch models/ or metadata.json. The decision lives in
config.USE_ROAD_KM_FEATURES.

Inputs:  data/processed/model_table.csv (has road_km / has_road_km from src.features),
         data/external/osm_kecamatan_features.csv
Outputs: reports/ablation_road_km.csv, reports/ablation_road_km.md (parsing report + ablation),
         reports/metrics.md ("Ablation: road km feature" section; src.train keeps it on rewrite)

Run: python -m src.ablation_road_km   (after python -m src.features)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.evaluate import ape, summarize_folds
from src.features import ROAD_FEATURES, feature_columns, road_features, road_km_report
from src.train import CVS, check_no_leakage, cross_validate, make_folds, model_specs
from src.utils import format_idr, replace_md_section

VARIANTS = {"without road km": False, "with road km": True}


def run_variant(table: pd.DataFrame, features: list[str], folds: dict) -> tuple[dict, dict]:
    """CV results per model spec and spatial out-of-fold predictions."""
    results, oofs = {}, {}
    for spec in model_specs():
        res = {"family": spec.family, "params": spec.params}
        for cv in CVS:
            per_fold, oof = cross_validate(spec, table, features, folds[cv])
            res[cv] = summarize_folds(per_fold)
            if cv == "spatial":
                oofs[spec.name] = oof
        results[spec.name] = res
    return results, oofs


def best_of(results: dict, family: str) -> str:
    """Model with the lowest spatial MAPE in a family (same rule as src.train)."""
    cands = {n: r for n, r in results.items() if r["family"] == family}
    return min(cands, key=lambda n: cands[n]["spatial"]["mape_mean"])


def summary_rows(table: pd.DataFrame, runs: dict) -> pd.DataFrame:
    """One row per (variant, family): chosen model, spatial MAPE/MdAPE/MAE, random MAPE,
    and spatial MAPE on listings with / without a road km marker."""
    has_road = table["has_road_km"].to_numpy() == 1
    rows = []
    for variant, (results, oofs) in runs.items():
        for family in ("ridge", "lgbm"):
            name = best_of(results, family)
            s, r = results[name]["spatial"], results[name]["random"]
            row_ape = ape(table["price_idr"], pd.Series(oofs[name], index=table.index)).to_numpy()
            rows.append({
                "variant": variant, "family": family, "model": name,
                "spatial_mape": s["mape_mean"], "spatial_mape_std": s["mape_std"],
                "spatial_mdape": s["mdape_mean"], "spatial_mae": s["mae_mean"],
                "spatial_r2_log": s["r2_log_mean"], "random_mape": r["mape_mean"],
                "spatial_mape_road_rows": float(np.mean(row_ape[has_road])),
                "spatial_mape_other_rows": float(np.mean(row_ape[~has_road])),
            })
    return pd.DataFrame(rows)


def same_params_row(table: pd.DataFrame, runs: dict, name: str) -> pd.DataFrame:
    """The currently deployed LightGBM params, scored in both variants (isolates the feature effect)."""
    out = []
    for variant, (results, _) in runs.items():
        s, r = results[name]["spatial"], results[name]["random"]
        out.append({"variant": variant, "model": name, "spatial_mape": s["mape_mean"],
                    "spatial_mdape": s["mdape_mean"], "spatial_mae": s["mae_mean"], "random_mape": r["mape_mean"]})
    return pd.DataFrame(out)


def _table(df: pd.DataFrame) -> str:
    """Markdown rows: variant, model, spatial MAPE (± std), MdAPE, MAE, random MAPE."""
    cols = ["Variant", "Model", "Spatial MAPE", "Spatial MdAPE", "Spatial MAE", "Random MAPE"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples():
        pm = f" ± {r.spatial_mape_std:.1f}" if hasattr(r, "spatial_mape_std") else ""
        lines.append(f"| {r.variant} | {r.model} | {r.spatial_mape:.2f}%{pm} | {r.spatial_mdape:.2f}% | "
                     f"{format_idr(int(r.spatial_mae))} | {r.random_mape:.2f}% |")
    return "\n".join(lines)


def _md_rows(header: str, rows: list[str]) -> str:
    """Markdown table from a header row ('| a | b |') and body rows."""
    n = header.count("|") - 1
    return "\n".join([header, "|" + "---|" * n, *rows])


def decision_text(summary: pd.DataFrame) -> str:
    """One paragraph: the rule, the LightGBM numbers, and the decision recorded in config."""
    lg = summary[summary["family"] == "lgbm"].set_index("variant")
    w, wo = lg.loc["with road km"], lg.loc["without road km"]
    kept = config.USE_ROAD_KM_FEATURES
    return (
        f"Rule (guide P7b U1): keep only if spatial-CV MAPE is equal or better. LightGBM: "
        f"{wo['spatial_mape']:.2f}% without vs {w['spatial_mape']:.2f}% with (fold std ± "
        f"{wo['spatial_mape_std']:.1f}); MdAPE {wo['spatial_mdape']:.2f}% → {w['spatial_mdape']:.2f}%; random MAPE "
        f"{wo['random_mape']:.2f}% → {w['random_mape']:.2f}%; on the listings with a marker "
        f"{wo['spatial_mape_road_rows']:.1f}% → {w['spatial_mape_road_rows']:.1f}%. Every difference is far "
        "inside the fold-to-fold spread.\n\n"
        f"**Decision: {'kept' if kept else 'not kept'}** (`config.USE_ROAD_KM_FEATURES = {kept}`). "
        + ("" if kept else "No measurable gain, and keeping it would add a road + km input to the app that "
           "helps under 10% of houses. The parser stays; `road_name`, `road_km`, `has_road_km` remain in "
           "`model_table.csv` for EDA.")
    )


def metrics_section(summary: pd.DataFrame, n_road: int, n: int) -> str:
    """The section for reports/metrics.md (both ablation rows + decision)."""
    return (
        f"{config.ABLATION_METRICS_SECTION}\n\n"
        f"`road_km` / `has_road_km` parsed from listing text (\"Jl. Kaliurang km 8\", \"Jakal km 7\") for "
        f"{n_road} of {n:,} listings ({n_road / n:.1%}). Same grid, folds and selection rule as above; best "
        "model per family by spatial MAPE within each variant. Written by `python -m src.ablation_road_km`; "
        "parsing details in `reports/ablation_road_km.md`.\n\n"
        + _table(summary) + "\n\n" + decision_text(summary) + "\n"
    )


def write_md(summary: pd.DataFrame, fixed: pd.DataFrame, listings: pd.DataFrame, n_road: int, n: int) -> None:
    """reports/ablation_road_km.md (parsing report + ablation) and the metrics.md section."""
    counts, examples = road_km_report(listings, config.ROAD_KM_N_EXAMPLES)
    road = road_features(listings)
    by_kab = (pd.DataFrame({"kabupaten": listings["kabupaten"], "has": road["has_road_km"]})
              .groupby("kabupaten")["has"].agg(["sum", "size"]))
    lg = summary[summary["family"] == "lgbm"]
    parsing = (
        "Regex built from `config.MAIN_ROADS` + `ROAD_ALIASES` (\"Jakal\" = Kaliurang, \"Paris\" = "
        "Parangtritis). The road name must sit right before \"km\" (\"Jl.\"/\"jalan\" optional; \"Barat\", "
        "\"Timur\", \"Raya\" allowed in between), so distances like \"5 km dari UGM\" never match. Comma or dot "
        f"decimals; values outside {config.ROAD_KM_MIN:g}–{config.ROAD_KM_MAX:g} km are ignored. "
        f"**{n_road} of {n:,} listings ({n_road / n:.1%})** have a marker; km {road['road_km'].min():g}–"
        f"{road['road_km'].max():g}, median {road['road_km'].median():g}."
    )
    md = "\n\n".join([
        "# Road km feature (P7b U1)",
        "## Parsing", parsing,
        _md_rows("| Road | Listings |", [f"| {r.road} | {r.listings} |" for r in counts.itertuples()]),
        _md_rows("| Kabupaten | With marker | Listings |",
                 [f"| {k} | {r['sum']} | {r['size']} |" for k, r in by_kab.iterrows()]),
        f"### {len(examples)} random examples (seed {config.RANDOM_STATE})",
        _md_rows("| Description snippet | Parsed |",
                 [f"| {r.snippet.replace('|', '/')} | {r.road_name} km {r.road_km:g} |"
                  for r in examples.itertuples()]),
        "## Ablation", "### Best model per family", _table(summary),
        "### Same LightGBM params in both variants", _table(fixed),
        "### LightGBM error by subset (spatial CV, out-of-fold)",
        _md_rows("| Variant | Spatial MAPE, rows with marker | Spatial MAPE, other rows |",
                 [f"| {r.variant} | {r.spatial_mape_road_rows:.1f}% | {r.spatial_mape_other_rows:.1f}% |"
                  for r in lg.itertuples()]),
        decision_text(summary),
    ]) + "\n"
    config.ABLATION_ROAD_KM_MD.write_text(md, encoding="utf-8")
    text = replace_md_section(config.METRICS_MD.read_text(encoding="utf-8"), config.ABLATION_METRICS_SECTION,
                              metrics_section(summary, n_road, n))
    config.METRICS_MD.write_text(text, encoding="utf-8")


def main() -> None:
    """Run both variants, print the comparison, write the ablation report."""
    table = pd.read_csv(config.MODEL_TABLE, dtype={config.CV_GROUP_COLUMN: str})
    osm = pd.read_csv(config.OSM_FEATURES_CSV)
    missing = [c for c in ROAD_FEATURES if c not in table.columns]
    if missing:
        raise SystemExit(f"model table lacks {missing}: run python -m src.features first")
    folds = make_folds(table)
    n_road = int(table["has_road_km"].sum())
    print(f"{len(table):,} rows, {n_road} with a road km marker ({n_road / len(table):.1%})")

    runs = {}
    for variant, use_road in VARIANTS.items():
        features = feature_columns(osm, road_km=use_road)
        check_no_leakage(features)
        print(f"  {variant}: {len(features)} features ...")
        runs[variant] = run_variant(table, features, folds)

    summary = summary_rows(table, runs)
    deployed = runs["without road km"][0]
    fixed = same_params_row(table, runs, best_of(deployed, "lgbm"))
    summary.to_csv(config.ABLATION_ROAD_KM_CSV, index=False, encoding="utf-8")
    listings = pd.read_csv(config.CLEAN_CSV)
    write_md(summary, fixed, listings, n_road, len(table))

    pd.set_option("display.width", 200)
    show = summary.assign(spatial_mae=summary["spatial_mae"].map(lambda v: format_idr(int(v))))
    print("\n" + show.drop(columns=["family", "spatial_r2_log"]).round(1).to_string(index=False))
    print("\nSame deployed LightGBM params:")
    print(fixed.assign(spatial_mae=fixed["spatial_mae"].map(lambda v: format_idr(int(v)))).round(1)
          .to_string(index=False))
    print(f"\nDecision in config: USE_ROAD_KM_FEATURES = {config.USE_ROAD_KM_FEATURES}")
    print(f"Wrote {config.ABLATION_ROAD_KM_MD.relative_to(config.ROOT)}, "
          f"{config.ABLATION_ROAD_KM_CSV.relative_to(config.ROOT)}, section in "
          f"{config.METRICS_MD.relative_to(config.ROOT)} (models/ untouched)")


if __name__ == "__main__":
    main()
