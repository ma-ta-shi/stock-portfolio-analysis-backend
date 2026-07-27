import pandas as pd
import pytest

from data.providers.base import MacroDataProvider
from data.providers.boc import BOCMacroDataProvider


@pytest.fixture
def provider():
    return BOCMacroDataProvider()


def _observations(series_id: str, values: list[str]) -> dict:
    return {
        "observations": [
            {"d": f"2026-01-{i + 1:02d}", series_id: {"v": v}} for i, v in enumerate(values)
        ]
    }


def test_provider_is_a_macro_data_provider(provider):
    assert isinstance(provider, MacroDataProvider)


async def test_get_macro_data_matches_abc_signature_no_dates(provider, monkeypatch):
    """The whole point of this fix: callable with only series_ids, like every other
    MacroDataProvider implementation — no start_date/end_date required."""

    async def fake_fetch_json(url, params=None):
        assert "start_date" not in url and "end_date" not in url
        return _observations("FXUSDCAD", ["1.35", "1.36"])

    monkeypatch.setattr(provider, "_fetch_json", fake_fetch_json)

    result = await provider.get_macro_data(["FXUSDCAD"])

    assert isinstance(result["FXUSDCAD"], pd.Series)
    assert list(result["FXUSDCAD"]) == [1.35, 1.36]


async def test_get_macro_data_uses_recent_param(provider, monkeypatch):
    seen_urls = []

    async def fake_fetch_json(url, params=None):
        seen_urls.append(url)
        return _observations("V39079", ["3.5"])

    monkeypatch.setattr(provider, "_fetch_json", fake_fetch_json)

    await provider.get_macro_data(["V39079"])

    assert "recent=10" in seen_urls[0]


async def test_get_macro_data_one_bad_series_does_not_crash_batch(provider, monkeypatch):
    async def flaky_fetch_json(url, params=None):
        if "BADSERIES" in url:
            raise ValueError("no such series")
        return _observations("FXUSDCAD", ["1.35"])

    monkeypatch.setattr(provider, "_fetch_json", flaky_fetch_json)

    result = await provider.get_macro_data(["FXUSDCAD", "BADSERIES"])

    assert "FXUSDCAD" in result
    assert "BADSERIES" not in result


async def test_get_interest_rates_matches_abc_signature_no_dates(provider, monkeypatch):
    """The other half of the fix: callable with zero args."""
    call_count = 0

    async def fake_fetch_json(url, params=None):
        nonlocal call_count
        call_count += 1
        assert "start_date" not in url and "end_date" not in url
        if "V39079" in url:
            return _observations("V39079", ["3.5"])
        return _observations("V80691311", ["5.2"])

    monkeypatch.setattr(provider, "_fetch_json", fake_fetch_json)

    rates = await provider.get_interest_rates()

    assert rates["overnight_rate"] == 3.5
    assert rates["prime_rate"] == 5.2
    assert call_count == 2


async def test_get_interest_rates_uses_recent_one(provider, monkeypatch):
    seen_urls = []

    async def fake_fetch_json(url, params=None):
        seen_urls.append(url)
        return _observations("V39079", ["3.5"])

    monkeypatch.setattr(provider, "_fetch_json", fake_fetch_json)

    await provider.get_interest_rates()

    assert all("recent=1" in url for url in seen_urls)


async def test_get_exchange_rates_unaffected_by_signature_fix(provider, monkeypatch):
    async def fake_fetch_json(url, params=None):
        return _observations("FXUSDCAD", ["1.35"])

    monkeypatch.setattr(provider, "_fetch_json", fake_fetch_json)

    result = await provider.get_exchange_rates("USDCAD")

    assert result["pair"] == "USDCAD"
    assert result["rate"] == 1.35
