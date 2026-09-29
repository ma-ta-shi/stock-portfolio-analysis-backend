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

    # recent=200 (~9 months of business days) — enough history for a 90-day
    # delta on a daily series, not just the latest value (86bbq8rj1).
    assert "recent=200" in seen_urls[0]


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


# --- degradation reporting (86bc997wr): BoC swallows its own failures ---


async def test_a_series_that_raises_is_reported_as_fetch_failed(provider, monkeypatch, parity):
    async def flaky(url, params=None):
        if "BADSERIES" in url:
            raise ValueError("no such series")
        return _observations("FXUSDCAD", ["1.35"])

    monkeypatch.setattr(provider, "_fetch_json", flaky)

    def call():
        async def go():
            result = await provider.get_macro_data(["FXUSDCAD", "BADSERIES"])
            return {k: list(v) for k, v in result.items()}

        return go()

    plain, with_collector, events = await parity(call)

    assert plain == with_collector == {"FXUSDCAD": [1.35]}
    assert [e.key for e in events] == [("boc", "get_macro_data", "fetch_failed")]


async def test_a_series_with_no_valid_observations_is_reported_as_data_missing(
    provider, monkeypatch, collector
):
    async def empty_values(url, params=None):
        return _observations("V39079", [""])  # a blank value is skipped, leaving no data

    monkeypatch.setattr(provider, "_fetch_json", empty_values)

    result = await provider.get_macro_data(["V39079"])

    assert result == {}  # unchanged
    assert [e.key for e in collector.drain()] == [("boc", "get_macro_data", "data_missing")]


async def test_interest_rates_and_exchange_rates_failures_are_reported(
    provider, monkeypatch, collector
):
    async def broken(url, params=None):
        raise ValueError("valet is down")

    monkeypatch.setattr(provider, "_fetch_json", broken)

    assert await provider.get_interest_rates() == {}  # unchanged
    assert await provider.get_exchange_rates("CADUSD") == {
        "pair": "CADUSD",
        "rate": None,
        "date": None,
    }  # unchanged
    assert [e.key for e in collector.drain()] == [
        ("boc", "get_interest_rates", "fetch_failed"),
        ("boc", "get_exchange_rates", "fetch_failed"),
    ]


async def test_healthy_boc_calls_report_nothing(provider, monkeypatch, collector):
    async def ok(url, params=None):
        if "V39079" in url or "V80691311" in url:
            series = "V39079" if "V39079" in url else "V80691311"
            return _observations(series, ["3.5"])
        return _observations("FXUSDCAD", ["1.35"])

    monkeypatch.setattr(provider, "_fetch_json", ok)
    await provider.get_macro_data(["FXUSDCAD"])
    await provider.get_interest_rates()
    await provider.get_exchange_rates("CADUSD")
    assert collector.drain() == []
