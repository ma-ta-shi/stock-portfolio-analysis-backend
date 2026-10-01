"""One vocabulary for sector names.

The providers do not share a taxonomy: yfinance/FMP say "Financial Services", "Basic
Materials", "Consumer Cyclical" and "Consumer Defensive", while openbb-tmx (Canadian
tickers) says "Finance", "Materials" and "Media & Telecommunications" (see
data/pipeline.py, `_US_SECTOR_ETF` and `_CA_SECTOR_ETF`, for the strings seen live).
The Macro Economist's design doc keys its sector tables on 11 names, so anything that
looks a sector up there goes through `normalize_sector` first.
"""

CANONICAL_SECTORS = (
    "Financials",
    "Real Estate",
    "Utilities",
    "Energy",
    "Materials",
    "Industrials",
    "Consumer Staples",
    "Consumer Discretionary",
    "Healthcare",
    "Technology",
    "Communication Services",
)

_ALIASES = {
    "finance": "Financials",
    "financial services": "Financials",
    "basic materials": "Materials",
    "consumer defensive": "Consumer Staples",
    "consumer cyclical": "Consumer Discretionary",
    "health care": "Healthcare",
    "information technology": "Technology",
    "media & telecommunications": "Communication Services",
    "telecommunications": "Communication Services",
}
_BY_LOWER = {name.lower(): name for name in CANONICAL_SECTORS}


def normalize_sector(sector: str | None) -> str | None:
    """The canonical sector name for a provider's sector string, or None when the
    string is empty or not recognised (callers fall back to their own default)."""
    key = (sector or "").strip().lower()
    if not key:
        return None
    return _ALIASES.get(key) or _BY_LOWER.get(key)
