import pandas as pd
import pytest
import structlog

from data.providers.fred import DEFAULT_SERIES_IDS, RATE_SERIES, FredMacroDataProvider


@pytest.fixture(autouse=True)
def fred_api_key(monkeypatch):
    monkeypatch.setenv("SP_FRED_API_KEY", "test-key")


@pytest.fixture
def provider():
    return FredMacroDataProvider()


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2026-01-01", periods=len(values), freq="MS"))


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("SP_FRED_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SP_FRED_API_KEY"):
        FredMacroDataProvider()


async def test_get_macro_data_returns_dict_of_series(provider, monkeypatch):
    monkeypatch.setattr(provider._client, "get_series", lambda series_id: _series([1.0, 2.0, 3.0]))

    result = await provider.get_macro_data(["FEDFUNDS", "CPIAUCSL"])

    assert set(result.keys()) == {"FEDFUNDS", "CPIAUCSL"}
    assert all(isinstance(s, pd.Series) for s in result.values())
    assert list(result["FEDFUNDS"]) == [1.0, 2.0, 3.0]


async def test_get_macro_data_normalizes_series_id_case(provider, monkeypatch):
    seen_ids = []
    monkeypatch.setattr(
        provider._client,
        "get_series",
        lambda series_id: seen_ids.append(series_id) or _series([1.0]),
    )

    result = await provider.get_macro_data([" fedfunds "])

    assert seen_ids == ["FEDFUNDS"]
    assert "FEDFUNDS" in result


async def test_get_macro_data_one_bad_series_does_not_crash_batch(provider, monkeypatch):
    def flaky_get_series(series_id):
        if series_id == "BADSERIES":
            raise ValueError("no such series")
        return _series([42.0])

    monkeypatch.setattr(provider._client, "get_series", flaky_get_series)

    with structlog.testing.capture_logs() as logs:
        result = await provider.get_macro_data(["FEDFUNDS", "BADSERIES"])

    assert list(result["FEDFUNDS"]) == [42.0]
    assert result["BADSERIES"].empty
    assert result["BADSERIES"].dtype == "float64"
    assert any(
        log["event"] == "fred_series_fetch_failed" and log["series_id"] == "BADSERIES"
        for log in logs
    )


async def test_get_macro_data_defaults_to_documented_series_set(provider, monkeypatch):
    seen_ids = []
    monkeypatch.setattr(
        provider._client,
        "get_series",
        lambda series_id: seen_ids.append(series_id) or _series([1.0]),
    )

    await provider.get_macro_data()

    assert seen_ids == DEFAULT_SERIES_IDS


async def test_get_macro_data_empty_list_returns_empty_dict_not_defaults(provider, monkeypatch):
    """An explicit [] means "fetch nothing" — must not be treated the same as
    unset/None just because [] is falsy in Python."""
    monkeypatch.setattr(provider._client, "get_series", lambda series_id: _series([1.0]))

    result = await provider.get_macro_data([])

    assert result == {}


async def test_get_interest_rates_returns_expected_keys(provider, monkeypatch):
    monkeypatch.setattr(provider._client, "get_series", lambda series_id: _series([1.111, 2.222]))

    rates = await provider.get_interest_rates()

    assert set(rates.keys()) == set(RATE_SERIES.keys())
    assert rates["fed_funds_effective"] == 2.22


async def test_get_interest_rates_falls_back_to_none_on_failure(provider, monkeypatch):
    def flaky_get_series(series_id):
        if series_id == "DGS10":
            raise ValueError("upstream error")
        return _series([5.0])

    monkeypatch.setattr(provider._client, "get_series", flaky_get_series)

    rates = await provider.get_interest_rates()

    assert rates["treasury_10yr"] is None
    assert rates["fed_funds_effective"] == 5.0


async def test_get_interest_rates_handles_empty_series(provider, monkeypatch):
    monkeypatch.setattr(
        provider._client, "get_series", lambda series_id: pd.Series(dtype="float64")
    )

    rates = await provider.get_interest_rates()

    assert all(value is None for value in rates.values())


async def test_get_exchange_rates_returns_not_supported_status(provider):
    result = await provider.get_exchange_rates("CADUSD")

    assert result["pair"] == "CADUSD"
    assert "Bank of Canada Valet" in result["status"] or "boc.py" in result["status"]
