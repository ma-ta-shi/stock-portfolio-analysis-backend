"""Shared fixtures for the live E2E suite (ClickUp 86bb7j0kh).

Gating is two layers: pyproject.toml's `addopts = "-m 'not live'"` keeps
this whole suite out of every default `pytest` run, and this autouse
fixture makes `-m live` itself fail with a clear skip message instead of a
raw API auth traceback when a required key is missing.
"""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

_REQUIRED_ENV_VARS = (
    "SP_FMP_API_KEY",
    "SP_FINNHUB_API_KEY",
    "SP_EDGAR_IDENTITY",
    "SP_FRED_API_KEY",
)


@pytest.fixture(autouse=True)
def _require_live_api_keys():
    missing = [name for name in _REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        pytest.skip(f"Live suite needs real API keys, missing: {', '.join(missing)}")
