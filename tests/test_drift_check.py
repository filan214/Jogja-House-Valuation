"""P7b U4 drift check: template, empty sheet, scoring, skipped rows, About page."""

import json

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import config, drift_check


@pytest.fixture
def out_paths(tmp_path, monkeypatch):
    """Send the report and the app summary to tmp so tests never touch the real files."""
    monkeypatch.setattr(config, "DRIFT_REPORT_MD", tmp_path / "drift_check.md")
    monkeypatch.setattr(config, "DRIFT_SUMMARY_JSON", tmp_path / "drift_summary.json")
    return tmp_path


def _example_rows(tmp_path, **changes) -> "Path":
    """The two EXAMPLE rows without # lines, optionally with edits {(row, col): value}."""
    lines = [ln for ln in config.DRIFT_EXAMPLE_CSV.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    path = tmp_path / "rows.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if changes:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        for (i, col), v in changes.items():
            df.loc[i, col] = v
        df.to_csv(path, index=False)
    return path


def test_template_columns_match_guide_and_model():
    header = pd.read_csv(config.DRIFT_CSV, nrows=0).columns.tolist()
    assert header == list(config.DRIFT_COLUMNS)
    meta = json.loads(config.METADATA_JSON.read_text(encoding="utf-8"))
    assert [c for c in header if c.startswith("flag_")] == meta["active_flags"]
    example = [ln for ln in config.DRIFT_EXAMPLE_CSV.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    assert example[0].split(",") == list(config.DRIFT_COLUMNS) and len(example) == 3


def test_empty_template_exits_cleanly(out_paths, tmp_path, capsys):
    empty = tmp_path / "empty.csv"
    empty.write_text(",".join(config.DRIFT_COLUMNS) + "\n", encoding="utf-8")
    config.DRIFT_SUMMARY_JSON.write_text("{}", encoding="utf-8")
    drift_check.main(empty)
    assert "No drift rows yet" in capsys.readouterr().out
    assert not config.DRIFT_SUMMARY_JSON.exists()
    assert "Not run yet" in config.DRIFT_REPORT_MD.read_text(encoding="utf-8")


def test_scores_rows_without_factor(out_paths, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PRICE_INDEX_FACTOR", None)
    drift_check.main(_example_rows(tmp_path))
    s = json.loads(config.DRIFT_SUMMARY_JSON.read_text(encoding="utf-8"))
    assert s["n_rows"] == 2 and s["n_skipped"] == 0 and s["adjustment"] is None
    unadj, adj = s["overall"]
    assert {k: v for k, v in unadj.items() if k != "prices"} == {k: v for k, v in adj.items() if k != "prices"}
    assert [r["kabupaten"] for r in s["by_kabupaten"]] == ["Sleman", "Bantul"]
    assert all(r["small_n"] for r in s["by_kabupaten"])
    assert any("No price index factor" in f for f in s["findings"])
    md = config.DRIFT_REPORT_MD.read_text(encoding="utf-8")
    assert "## Overall" in md and "## By kabupaten" in md and "Mean signed error" in md


def test_factor_changes_only_adjusted(out_paths, tmp_path, monkeypatch):
    rows = _example_rows(tmp_path)
    df = drift_check.load_rows(rows)
    monkeypatch.setattr(config, "PRICE_INDEX_FACTOR", 1.2)
    monkeypatch.setattr(config, "PRICE_INDEX_SOURCE", "Test index")
    monkeypatch.setattr(config, "PRICE_INDEX_AS_OF", "2026-09")
    scored, _ = drift_check.score(df)
    assert ((scored["estimate_idr"] - scored["estimate_2024_idr"] * 1.2).abs() <= 1).all()
    u, a = drift_check.metrics(scored, "_2024_idr"), drift_check.metrics(scored, "_idr")
    assert a["mean_signed_error"] > u["mean_signed_error"]


def test_low_bias_is_reported_not_fixed():
    overall = [{"mean_signed_error": -30.0}, {"mean_signed_error": config.DRIFT_LOW_BIAS_PCT - 1}]
    notes = drift_check.findings(overall, {"factor": 1.1}, n=25)
    assert any("still clearly low" in n and "never tuned" in n for n in notes)


def test_bad_rows_are_skipped_with_reason(out_paths, tmp_path):
    rows = _example_rows(tmp_path, **{})
    df = drift_check.load_rows(rows)
    bad = pd.concat([df] * 5, ignore_index=True)
    bad["url"] = [f"https://example.com/{i}" for i in range(len(bad))]
    bad.loc[0, "kabupaten"] = "Jakarta"
    bad.loc[1, "kecamatan"] = "Atlantis"
    bad.loc[2, "land_m2"] = "0"
    bad.loc[3, "flag_shm"] = "yes"
    bad.loc[4, "asking_price_idr"] = "1,2 M"
    scored, skipped = drift_check.score(bad)
    assert len(scored) == 5 and len(skipped) == 5
    reasons = " | ".join(s["reason"] for s in skipped)
    for part in ("unknown kabupaten", "Atlantis", "land_m2", "flags", "numeric"):
        assert part in reasons


def test_column_mismatch_stops(tmp_path):
    p = tmp_path / "wrong.csv"
    p.write_text("date_seen,url,price\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        drift_check.load_rows(p)


def test_about_page_shows_summary(out_paths, tmp_path, monkeypatch):
    from src.app_data import drift_summary
    page = str(config.ROOT / "pages" / "2_About_the_model.py")
    drift_summary.clear()
    at = AppTest.from_file(page, default_timeout=60).run()
    assert not at.exception and any("Not filled yet" in c.value for c in at.caption)
    monkeypatch.setattr(config, "PRICE_INDEX_FACTOR", None)
    drift_check.main(_example_rows(tmp_path))
    drift_summary.clear()
    at = AppTest.from_file(page, default_timeout=60).run()
    assert not at.exception
    assert any("current listings seen 2026-10-05" in m.value for m in at.markdown)
    drift_summary.clear()


def test_p2_limits_duplicates_and_bands(out_paths, tmp_path):
    df = drift_check.load_rows(_example_rows(tmp_path))
    rows = pd.concat([df] * 4, ignore_index=True)
    rows["url"] = [f"https://example.com/{i}" for i in range(len(rows))]
    rows.loc[2, "building_m2"] = str(float(rows.loc[2, "land_m2"]) * 4)
    rows.loc[3, "bedrooms"] = str(config.KOS_MIN_BEDROOMS)
    rows.loc[4, "asking_price_idr"] = "50000000"
    rows.loc[5, "bathrooms"] = "1.5"
    rows.loc[6, "url"] = rows.loc[0, "url"]
    scored, skipped = drift_check.score(rows)
    reasons = " | ".join(s["reason"] for s in skipped)
    assert len(skipped) == 5 and len(scored) == 3
    for part in ("3× land", "kos", "P2 range", "whole number", "duplicate URL"):
        assert part in reasons
    assert drift_check.data_checks(rows)["duplicate_urls"] == 1
    bands = drift_check.price_band_table(scored)
    assert sum(b["n"] for b in bands) == len(scored)
    assert {b["band"] for b in bands} <= set(config.PRICE_BAND_LABELS)
