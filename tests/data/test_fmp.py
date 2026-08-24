import json

import pandas as pd
import pytest
import structlog

from data.providers.base import NewsProvider, StockDataProvider
from data.providers.fmp import FMPDataProvider, _period_to_from_date


# --- Fakes standing in for aiohttp's response/session objects ---


class FakeResponse:
    def __init__(self, status: int, json_data=None, text_data: str | None = None) -> None:
        self.status = status
        self._json_data = json_data
        self._text_data = text_data if text_data is not None else json.dumps(json_data)

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def json(self):
        return self._json_data

    async def text(self) -> str:
        return self._text_data

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self._response = response
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, params: dict | None = None) -> FakeResponse:
        self.calls.append((url, params))
        return self._response


@pytest.fixture(autouse=True)
def fmp_api_key(monkeypatch):
    monkeypatch.setenv("SP_FMP_API_KEY", "test-key")


@pytest.fixture
def provider():
    return FMPDataProvider()


def _wire(provider: FMPDataProvider, response: FakeResponse) -> FakeSession:
    session = FakeSession(response)
    provider.session = session
    return session


# --- Construction ---


def test_provider_is_a_stock_and_news_provider(provider):
    assert isinstance(provider, StockDataProvider)
    assert isinstance(provider, NewsProvider)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("SP_FMP_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SP_FMP_API_KEY"):
        FMPDataProvider()


def test_explicit_api_key_overrides_env(monkeypatch):
    monkeypatch.delenv("SP_FMP_API_KEY", raising=False)
    provider = FMPDataProvider(api_key="explicit-key")
    assert provider._api_key == "explicit-key"


# --- _request: the shared HTTP layer FMP's free-tier quirks live in ---


async def test_request_returns_data_on_success(provider):
    _wire(provider, FakeResponse(200, json_data=[{"symbol": "AAPL"}]))

    data = await provider._request("profile", {"symbol": "AAPL"})

    assert data == [{"symbol": "AAPL"}]


async def test_request_returns_none_and_logs_on_402(provider):
    """Dual-class-share tickers (BRK.B, GOOG) return HTTP 402 with a plain-text body
    despite the application/json content-type header — must not call .json() on it."""
    _wire(provider, FakeResponse(402, text_data="Premium Query Parameter: not available"))

    with structlog.testing.capture_logs() as logs:
        data = await provider._request("historical-price-eod/full", {"symbol": "BRK.B"})

    assert data is None
    assert any(log["event"] == "fmp_symbol_not_available" for log in logs)


async def test_request_returns_none_and_logs_on_empty_list(provider):
    """Invalid tickers, and /profile's way of signaling a paywalled symbol, come
    back as HTTP 200 with an empty list rather than an error status."""
    _wire(provider, FakeResponse(200, json_data=[]))

    with structlog.testing.capture_logs() as logs:
        data = await provider._request("profile", {"symbol": "BRK.B"})

    assert data is None
    assert any(log["event"] == "fmp_empty_response" for log in logs)


async def test_request_raises_on_401_bad_key(provider):
    _wire(provider, FakeResponse(401, text_data='{"Error Message": "Invalid API KEY."}'))

    with pytest.raises(RuntimeError, match="401"):
        await provider._request("profile", {"symbol": "AAPL"})


async def test_request_raises_on_5xx(provider):
    _wire(provider, FakeResponse(500, json_data={"error": "server error"}))

    with pytest.raises(RuntimeError, match="500"):
        await provider._request("profile", {"symbol": "AAPL"})


async def test_request_attaches_api_key_to_query(provider):
    session = _wire(provider, FakeResponse(200, json_data=[{"symbol": "AAPL"}]))

    await provider._request("profile", {"symbol": "AAPL"})

    _, params = session.calls[0]
    assert params["apikey"] == "test-key"
    assert params["symbol"] == "AAPL"


# --- _period_to_from_date ---


@pytest.mark.parametrize(
    "period,expected_days_ago",
    [
        ("1mo", 30),
        ("6mo", 180),
        ("1y", 365),
        ("5y", 1825),
        ("10d", 10),
    ],
)
def test_period_to_from_date_valid_periods(period, expected_days_ago):
    result = pd.Timestamp(_period_to_from_date(period))
    expected = pd.Timestamp.now().normalize() - pd.Timedelta(days=expected_days_ago)
    assert result == expected


def test_period_to_from_date_max_returns_epoch():
    assert _period_to_from_date("max") == "1970-01-01"


def test_period_to_from_date_invalid_raises():
    with pytest.raises(ValueError, match="Unsupported period"):
        _period_to_from_date("banana")


# --- get_price_history ---


async def test_get_price_history_returns_sorted_indexed_dataframe(provider):
    rows = [
        {"symbol": "AAPL", "date": "2026-07-24", "close": 333.0},
        {"symbol": "AAPL", "date": "2026-07-22", "close": 325.0},
        {"symbol": "AAPL", "date": "2026-07-23", "close": 321.0},
    ]
    _wire(provider, FakeResponse(200, json_data=rows))

    df = await provider.get_price_history("AAPL", "1mo", "1d")

    assert list(df.index) == [
        pd.Timestamp("2026-07-22"),
        pd.Timestamp("2026-07-23"),
        pd.Timestamp("2026-07-24"),
    ]


async def test_get_price_history_paywalled_ticker_returns_empty_dataframe_not_crash(provider):
    """BRK.B/GOOG-style dual-class tickers must degrade gracefully, not raise."""
    _wire(provider, FakeResponse(402, text_data="Premium Query Parameter"))

    df = await provider.get_price_history("BRK.B", "1mo", "1d")

    assert isinstance(df, pd.DataFrame)
    assert df.empty


async def test_get_price_history_etf_paywalled_returns_empty_dataframe_not_crash(provider):
    """Not in the original ticket: ETFs (QQQ, GLD, VOO, IWM, XLK, ARKK, DIA all
    confirmed live) hit the same 402 "Special Endpoint" paywall as dual-class
    shares, and much more broadly. SPY was the sole free exception found. Since
    ETFs route through FMP as ordinary non-.TO tickers and are common TFSA/RRSP
    holdings for this platform, this needs the same graceful degradation."""
    _wire(provider, FakeResponse(402, text_data="Premium Query Parameter"))

    df = await provider.get_price_history("QQQ", "1mo", "1d")

    assert isinstance(df, pd.DataFrame)
    assert df.empty


# --- get_company_info ---


async def test_get_company_info_maps_to_normalized_shape(provider):
    _wire(
        provider,
        FakeResponse(
            200,
            json_data=[
                {
                    "symbol": "AAPL",
                    "companyName": "Apple Inc.",
                    "sector": "Technology",
                    "industry": "Consumer Electronics",
                    "marketCap": 3_000_000_000_000,
                    "currency": "USD",
                    "country": "US",
                    "exchange": "NASDAQ",
                }
            ],
        ),
    )

    info = await provider.get_company_info("AAPL")

    assert info == {
        "name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "market_cap": 3_000_000_000_000,
        "currency": "USD",
        "country": "US",
        "primary_exchange": "NASDAQ",
    }


async def test_get_company_info_paywalled_ticker_returns_empty_dict(provider):
    _wire(provider, FakeResponse(200, json_data=[]))

    info = await provider.get_company_info("BRK.B")

    assert info == {}


# --- get_analyst_estimates (86bbdu04a: rewritten from the old raw
# {"symbol": ..., "estimates": [...]} shape to NormalizedAnalystEstimates) ---


async def test_get_analyst_estimates_returns_nearest_future_fiscal_year_eps(provider):
    """Real gap found planning this ticket: FMP's period=annual rows come
    back sorted descending by fiscal-year-end date (furthest-future
    first), confirmed live — estimates[0] is the furthest year, not the
    nearest. Must filter to future rows and take the soonest."""
    session = _wire(
        provider,
        FakeResponse(
            200,
            json_data=[
                {"date": "2099-09-27", "epsAvg": 20.0},  # furthest future
                {"date": "2027-09-27", "epsAvg": 9.5},  # nearest future
                {"date": "2020-09-27", "epsAvg": 4.0},  # past
            ],
        ),
    )

    result = await provider.get_analyst_estimates("AAPL")

    _, params = session.calls[0]
    assert params["period"] == "annual"
    assert result == {"forward_eps": 9.5}


async def test_get_analyst_estimates_no_future_rows_returns_empty_dict(provider):
    """Bare {}, not {"forward_eps": None} — matches get_company_info's/
    get_quote's own "empty dict signals no data" convention, needed for
    Router._is_empty() to correctly recognize this as empty."""
    _wire(provider, FakeResponse(200, json_data=[{"date": "2020-09-27", "epsAvg": 4.0}]))

    result = await provider.get_analyst_estimates("AAPL")

    assert result == {}


async def test_get_analyst_estimates_no_data_returns_empty_dict(provider):
    _wire(provider, FakeResponse(200, json_data=[]))

    result = await provider.get_analyst_estimates("BRK.B")

    assert result == {}


# --- get_analyst_ratings ---


async def test_get_analyst_ratings_returns_snapshot(provider):
    _wire(provider, FakeResponse(200, json_data=[{"symbol": "AAPL", "rating": "B"}]))

    result = await provider.get_analyst_ratings("AAPL")

    assert result["rating"] == "B"


# --- get_earnings_surprises (new, 86bbdu04a) ---


async def test_get_earnings_surprises_filters_realized_rows_and_caps_at_four(provider):
    """FMP's /earnings mixes forward calendar rows (epsActual is None,
    confirmed live) with real historical actual-vs-estimate rows — filter
    to realized only, and cap at 4 (matches yfinance's own hard limit for
    the same field)."""
    rows = [{"date": "2099-01-01", "epsActual": None, "epsEstimated": 2.0}]
    rows += [
        {
            "date": f"202{5 - i}-01-01",  # newest first, matching real FMP order
            "epsActual": 2.0 + i,
            "epsEstimated": 2.0,
            "revenueActual": 1000.0 + i,
            "revenueEstimated": 1000.0,
        }
        for i in range(6)
    ]
    _wire(provider, FakeResponse(200, json_data=rows))

    result = await provider.get_earnings_surprises("AAPL")

    assert len(result) == 4
    assert result[0]["period_end"] == "2025-01-01"
    assert all(r["eps_actual"] is not None for r in result)


async def test_get_earnings_surprises_computes_surprise_pct(provider):
    _wire(
        provider,
        FakeResponse(
            200,
            json_data=[
                {
                    "date": "2026-07-30",
                    "epsActual": 2.02,
                    "epsEstimated": 1.89,
                    "revenueActual": 109417000000,
                    "revenueEstimated": 109038900000,
                }
            ],
        ),
    )

    result = await provider.get_earnings_surprises("AAPL")

    assert result == [
        {
            "period_end": "2026-07-30",
            "eps_actual": 2.02,
            "eps_estimated": 1.89,
            "eps_surprise_pct": pytest.approx((2.02 - 1.89) / 1.89 * 100),
            "revenue_actual": 109417000000,
            "revenue_estimated": 109038900000,
        }
    ]


async def test_get_earnings_surprises_zero_estimate_returns_none_surprise_pct(provider):
    """abs(epsEstimated) would divide by zero — treated as undefined,
    matching how a missing epsEstimated is already handled."""
    _wire(
        provider,
        FakeResponse(
            200, json_data=[{"date": "2024-01-01", "epsActual": 0.5, "epsEstimated": 0.0}]
        ),
    )

    result = await provider.get_earnings_surprises("AAPL")

    assert result[0]["eps_surprise_pct"] is None


async def test_get_earnings_surprises_no_data_returns_empty_list(provider):
    _wire(provider, FakeResponse(200, json_data=[]))

    result = await provider.get_earnings_surprises("AAPL")

    assert result == []


# --- get_earnings_calendar ---


async def test_get_earnings_calendar_filters_bulk_response_by_ticker(provider):
    bulk = [
        {"symbol": "RKT", "date": "2026-08-06"},
        {"symbol": "UBER", "date": "2026-08-05"},
        {"symbol": "SONY", "date": "2026-08-06"},
    ]
    _wire(provider, FakeResponse(200, json_data=bulk))

    result = await provider.get_earnings_calendar("uber")

    assert result == [{"symbol": "UBER", "date": "2026-08-05"}]


async def test_get_earnings_calendar_no_match_returns_empty_list(provider):
    _wire(provider, FakeResponse(200, json_data=[{"symbol": "RKT", "date": "2026-08-06"}]))

    result = await provider.get_earnings_calendar("ZZZZ")

    assert result == []


# --- get_dividend_history ---


async def test_get_dividend_history_filters_by_date_range(provider):
    rows = [
        {"symbol": "AAPL", "date": "2026-05-11", "dividend": 0.27, "paymentDate": "2026-05-14"},
        {"symbol": "AAPL", "date": "2025-11-10", "dividend": 0.25, "paymentDate": "2025-11-13"},
    ]
    _wire(provider, FakeResponse(200, json_data=rows))

    result = await provider.get_dividend_history("AAPL", "2026-01-01", "2026-12-31")

    assert len(result) == 1
    assert result[0] == {
        "ex_date": "2026-05-11",
        "payment_date": "2026-05-14",
        "amount_per_share": 0.27,
    }


async def test_get_dividend_history_no_data_returns_empty_list(provider):
    _wire(provider, FakeResponse(200, json_data=[]))

    result = await provider.get_dividend_history("BRK.B", "2026-01-01", "2026-12-31")

    assert result == []


# --- get_quote / get_ratios_ttm (not on the ABC) ---


async def test_get_quote_maps_to_normalized_shape(provider):
    _wire(
        provider,
        FakeResponse(
            200,
            json_data=[
                {
                    "symbol": "AAPL",
                    "price": 334.96,
                    "marketCap": 4_500_000_000_000,
                    "yearHigh": 344.57,
                    "yearLow": 223.78,
                }
            ],
        ),
    )

    result = await provider.get_quote("AAPL")

    assert result == {
        "current_price": 334.96,
        "market_cap": 4_500_000_000_000,
        "currency": "USD",
        "high_52w": 344.57,
        "low_52w": 223.78,
    }


async def test_get_quote_no_data_returns_empty_dict(provider):
    _wire(provider, FakeResponse(200, json_data=[]))

    result = await provider.get_quote("ZZZZ")

    assert result == {}


async def test_get_quote_null_price_returns_empty_dict(provider):
    """Real bug caught on review: .get("price", 0.0) only guards a missing
    key, not an explicit null in the API response — would have silently
    fabricated a $0.00 quote instead of signaling "no real price"."""
    _wire(provider, FakeResponse(200, json_data=[{"symbol": "ZZZZ", "price": None}]))

    result = await provider.get_quote("ZZZZ")

    assert result == {}


async def test_get_ratios_ttm_returns_first_record(provider):
    _wire(provider, FakeResponse(200, json_data=[{"symbol": "AAPL", "netProfitMarginTTM": 0.27}]))

    result = await provider.get_ratios_ttm("AAPL")

    assert result["netProfitMarginTTM"] == 0.27


# --- Out-of-scope methods ---


@pytest.mark.parametrize(
    "method,args",
    [
        ("get_financials", ("AAPL", "income", "annual")),
        ("get_insider_trading", ("AAPL",)),
        ("get_peers", ("AAPL",)),
        ("get_news", ("AAPL", 7)),
        ("get_analyst_recommendation_trends", ("AAPL",)),
    ],
)
async def test_out_of_scope_methods_raise_not_implemented(provider, method, args):
    with pytest.raises(NotImplementedError):
        await getattr(provider, method)(*args)
