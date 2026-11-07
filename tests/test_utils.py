import pytest

from src.utils import format_idr, get_md_section, normalize_name, replace_md_section

MD = "# T\n\n## A\n\na text\n\n## B\n\nb text\n"


def test_get_md_section():
    assert get_md_section(MD, "## A") == "## A\n\na text\n"
    assert get_md_section(MD, "## B") == "## B\n\nb text\n"
    assert get_md_section(MD, "## C") is None


def test_replace_md_section_keeps_neighbors_and_appends():
    assert replace_md_section(MD, "## A", "## A\n\nnew\n") == "# T\n\n## A\n\nnew\n\n## B\n\nb text\n"
    assert replace_md_section(MD, "## C", "## C\n\nc\n").endswith("b text\n\n## C\n\nc\n")
    assert replace_md_section("# T\n", "## B", get_md_section(MD, "## B")) == "# T\n\n## B\n\nb text\n"


@pytest.mark.parametrize(
    "value, expected",
    [
        (875_000_000, "Rp 875 jt"),
        (1_100_000_000, "Rp 1,1 M"),
        (1_050_000_000, "Rp 1,05 M"),
        (2_000_000_000, "Rp 2 M"),
        (920_000_000, "Rp 920 jt"),
        (3_500_000, "Rp 3,5 jt"),
        (750_000, "Rp 750.000"),
        (999_999_999, "Rp 1 M"),
        (-875_000_000, "-Rp 875 jt"),
    ],
)
def test_format_idr(value, expected):
    assert format_idr(value) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  Depok ", "depok"),
        ("Gunung-Kidul", "gunung kidul"),
        ("D.I.  Yogyakarta", "di yogyakarta"),
        ("KULON   PROGO", "kulon progo"),
    ],
)
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected
