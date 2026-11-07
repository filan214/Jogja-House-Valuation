import pandas as pd
import pytest

from src.location import apply_mapping, match_area, match_key, official_kabupaten

KEC = pd.DataFrame(
    [
        ("Sleman", "Depok", "3404070"),
        ("Sleman", "Mlati", "3404060"),
        ("Sleman", "Sleman", "3404130"),
        ("Sleman", "Seyegan", "3404030"),
        ("Bantul", "Jetis", "3402080"),
        ("Bantul", "Imogiri", "3402090"),
        ("Bantul", "Bambang Lipuro", "3402050"),
        ("Kota Yogyakarta", "Jetis", "3471130"),
        ("Kota Yogyakarta", "Umbulharjo", "3471040"),
        ("Kulon Progo", "Sentolo", "3401060"),
    ],
    columns=["kabupaten", "kecamatan", "kecamatan_code"],
)
DESA = pd.DataFrame(
    [
        ("Sleman", "Depok", "Maguwoharjo", "3404070002"),
        ("Sleman", "Depok", "Condong Catur", "3404070003"),
    ],
    columns=["kabupaten", "kecamatan", "desa", "desa_code"],
)


def m(area: str, kab_raw: str) -> dict:
    return match_area(area, kab_raw, 1, KEC, DESA)


def test_match_key_ignores_spaces_case_dots():
    assert match_key("Condong Catur") == match_key("condongcatur") == "condongcatur"
    assert match_key("Bambanglipuro") == match_key("BAMBANG LIPURO")


@pytest.mark.parametrize(
    "raw, expected",
    [("KAB. GUNUNGKIDUL", "Gunungkidul"), ("KOTA YOGYAKARTA", "Kota Yogyakarta"), ("KAB. KULON PROGO", "Kulon Progo")],
)
def test_official_kabupaten(raw, expected):
    assert official_kabupaten(raw) == expected


def test_auto_exact():
    r = m("Depok", "Sleman")
    assert (r["kecamatan"], r["kabupaten"], r["match_type"], r["needs_review"]) == ("Depok", "Sleman", "auto", False)


def test_auto_space_variant():
    r = m("Bambanglipuro", "Bantul")
    assert r["kecamatan"] == "Bambang Lipuro" and r["match_type"] == "auto"


def test_auto_corrected_other_kabupaten():
    r = m("Imogiri", "Yogyakarta")
    assert (r["kecamatan"], r["kabupaten"], r["match_type"], r["needs_review"]) == (
        "Imogiri", "Bantul", "auto_corrected", False)


def test_ambiguous_name_under_yogyakarta_needs_review():
    r = m("Jetis", "Yogyakarta")
    assert r["kabupaten"] == "Kota Yogyakarta" and r["match_type"] == "auto" and r["needs_review"]


def test_jetis_bantul_is_plain_auto():
    r = m("Jetis", "Bantul")
    assert r["kabupaten"] == "Bantul" and not r["needs_review"]


def test_area_equal_to_kabupaten_name_needs_review():
    r = m("Sleman", "Sleman")
    assert r["kecamatan"] == "Sleman" and r["needs_review"]


def test_desa_match_is_suggestion():
    r = m("Maguwoharjo", "Yogyakarta")
    assert (r["kecamatan"], r["kabupaten"], r["match_type"], r["needs_review"]) == (
        "Depok", "Sleman", "suggested", True)


def test_spelling_variant_is_suggestion():
    r = m("Sayegan", "Sleman")
    assert r["kecamatan"] == "Seyegan" and r["match_type"] == "suggested" and r["needs_review"]


def test_locality_suggestion_never_confirmed():
    r = m("Seturan", "Yogyakarta")
    assert r["kecamatan"] == "Depok" and r["match_type"] == "suggested" and r["needs_review"]
    assert "General knowledge" in r["note"]


def test_unknown_is_unmatched():
    r = m("Atlantis", "Sleman")
    assert r["kecamatan"] is None and r["match_type"] == "unmatched" and r["needs_review"]


def _listings() -> pd.DataFrame:
    return pd.DataFrame({
        "listing_id": ["a", "b", "c"],
        "area_raw": ["Depok", "Imogiri", "Atlantis"],
        "kabupaten_raw": ["Sleman", "Yogyakarta", "Sleman"],
        "kabupaten": ["Sleman", "Kota Yogyakarta", "Sleman"],
    })


def test_apply_mapping_refuses_pending_review():
    mapping = pd.DataFrame([m("Depok", "Sleman"), m("Seturan", "Yogyakarta")])
    with pytest.raises(ValueError, match="need review"):
        apply_mapping(_listings(), mapping)


def test_apply_mapping_corrects_kabupaten_and_drops_unmatched():
    rows = [m("Depok", "Sleman"), m("Imogiri", "Yogyakarta"), m("Atlantis", "Sleman")]
    rows[2]["needs_review"] = False
    mapped, dropped = apply_mapping(_listings(), pd.DataFrame(rows))
    assert mapped.set_index("listing_id")["kecamatan"].to_dict() == {"a": "Depok", "b": "Imogiri"}
    assert mapped.set_index("listing_id").loc["b", "kabupaten"] == "Bantul"
    assert dropped["listing_id"].tolist() == ["c"]
    assert (dropped["drop_reason"] == "kecamatan_unmatched").all()
