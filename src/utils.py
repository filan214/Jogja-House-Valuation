"""Small shared helpers: money formatting, name normalization, Markdown sections."""

import re

_JUTA = 1_000_000
_MILIAR = 1_000_000_000


def _fmt_decimal(value: float, decimals: int) -> str:
    """Round to `decimals`, drop trailing zeros, use a comma as decimal separator."""
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text.replace(".", ",")


def format_idr(value: int) -> str:
    """Format a Rupiah amount for display.

    Rules (comma is the decimal separator):
    - >= 1 miliar: "Rp 1,1 M", "Rp 1,05 M", "Rp 2 M" (up to 2 decimals)
    - >= 100 juta: "Rp 875 jt" (whole juta)
    - >= 1 juta:   "Rp 3,5 jt" (up to 1 decimal)
    - below 1 juta: full amount with dot thousand separators, "Rp 750.000"

    Negative values keep a leading minus sign: "-Rp 875 jt".
    """
    sign = "-" if value < 0 else ""
    amount = abs(float(value))

    miliar = round(amount / _MILIAR, 2)
    if miliar >= 1:
        return f"{sign}Rp {_fmt_decimal(miliar, 2)} M"

    juta = amount / _JUTA
    decimals = 0 if juta >= 100 else 1
    juta = round(juta, decimals)
    if juta >= 1:
        return f"{sign}Rp {_fmt_decimal(juta, decimals)} jt"

    return f"{sign}Rp {round(amount):,}".replace(",", ".")


def normalize_name(s: str) -> str:
    """Normalize a place name for matching.

    Lowercase, strip, delete dots, turn hyphens into spaces, collapse
    whitespace. Example: " Gunung-Kidul " -> "gunung kidul",
    "D.I. Yogyakarta" -> "di yogyakarta".
    """
    s = s.lower().replace(".", "").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


def get_md_section(text: str, heading: str) -> str | None:
    """The `## heading` section of a Markdown text (up to the next `## `), or None."""
    if heading not in text:
        return None
    after = text.split(heading, 1)[1]
    nxt = after.find("\n## ")
    return heading + (after[:nxt] if nxt >= 0 else after).rstrip("\n") + "\n"


def replace_md_section(text: str, heading: str, section: str) -> str:
    """Replace the `## heading` section (up to the next `## `) with `section`, or append it."""
    if heading in text:
        before, after = text.split(heading, 1)
        nxt = after.find("\n## ")
        return before + section.rstrip("\n") + "\n" + (after[nxt:] if nxt >= 0 else "")
    return text.rstrip("\n") + "\n\n" + section
