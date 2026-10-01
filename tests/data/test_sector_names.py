"""Tests for data/sector_names.py: one vocabulary over the providers' sector spellings."""

import pytest

from data.sector_names import CANONICAL_SECTORS, normalize_sector


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Financial Services", "Financials"),  # yfinance / FMP
        ("Finance", "Financials"),  # openbb-tmx (RY.TO, TD.TO)
        ("Basic Materials", "Materials"),
        ("Materials", "Materials"),
        ("Consumer Cyclical", "Consumer Discretionary"),
        ("Consumer Defensive", "Consumer Staples"),
        ("Media & Telecommunications", "Communication Services"),
        ("Communication Services", "Communication Services"),
        ("Technology", "Technology"),
        ("  energy ", "Energy"),  # case and spacing
    ],
)
def test_provider_spellings_map_to_the_canonical_sector(raw, expected):
    assert normalize_sector(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "  ", "Cryptocurrency"])
def test_missing_or_unknown_sectors_map_to_none(raw):
    assert normalize_sector(raw) is None


def test_every_canonical_name_maps_to_itself():
    assert [normalize_sector(name) for name in CANONICAL_SECTORS] == list(CANONICAL_SECTORS)
