import json
import time

import pytest
import structlog

from data.providers.base import NewsProvider, StockDataProvider
from data.providers.finnhub import FinnhubDataProvider


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
    """Wired with either a single FakeResponse (returned every call) or a list
    (popped in order — needed for the 429-then-succeed retry test)."""

    def __init__(self, responses) -> None:
        self._responses = list(responses) if isinstance(responses, list) else [responses]
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, params: dict | None = None) -> FakeResponse:
        self.calls.append((url, params))
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


@pytest.fixture(autouse=True)
def finnhub_api_key(monkeypatch):
    monkeypatch.setenv("SP_FINNHUB_API_KEY", "test-key")


@pytest.fixture
def provider():
    return FinnhubDataProvider()


def _wire(provider: FinnhubDataProvider, responses) -> FakeSession:
    session = FakeSession(responses)
    provider.session = session
    return session


# --- Construction ---


def test_provider_is_a_news_provider_only(provider):
    assert isinstance(provider, NewsProvider)
    assert not isinstance(provider, StockDataProvider)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("SP_FINNHUB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SP_FINNHUB_API_KEY"):
        FinnhubDataProvider()


def test_explicit_api_key_overrides_env(monkeypatch):
    monkeypatch.delenv("SP_FINNHUB_API_KEY", raising=False)
    provider = FinnhubDataProvider(api_key="explicit-key")
    assert provider._api_key == "explicit-key"


# --- Session lifecycle ---


class FakeClientSession:
    """Stands in for aiohttp.ClientSession() itself, not just a response —
    exercises __aenter__/__aexit__ actually opening/closing an owned session."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


async def test_context_manager_opens_and_closes_owned_session(monkeypatch):
    created = []

    def fake_client_session():
        session = FakeClientSession()
        created.append(session)
        return session

    monkeypatch.setattr("data.providers.finnhub.aiohttp.ClientSession", fake_client_session)

    async with FinnhubDataProvider(api_key="test-key") as provider:
        assert provider.session is created[0]
        assert provider.session.closed is False

    assert created[0].closed is True


async def test_context_manager_does_not_close_externally_provided_session():
    external = FakeClientSession()
    provider = FinnhubDataProvider(api_key="test-key", session=external)

    async with provider:
        pass

    assert external.closed is False


# --- _request ---


async def test_request_returns_data_on_success(provider):
    _wire(provider, FakeResponse(200, json_data=[{"symbol": "AAPL"}]))

    data = await provider._request("company-news", {"symbol": "AAPL"})

    assert data == [{"symbol": "AAPL"}]


async def test_request_returns_empty_list_as_is_not_none(provider):
    """Unlike FMP, an empty list from Finnhub means "no data" (e.g. a quiet
    ticker with no recent news), not a paywall — must not be coerced to None."""
    _wire(provider, FakeResponse(200, json_data=[]))

    data = await provider._request("company-news", {"symbol": "X"})

    assert data == []


async def test_request_returns_none_and_logs_on_403(provider):
    _wire(
        provider,
        FakeResponse(403, text_data='{"error": "You don\'t have access to this resource."}'),
    )

    with structlog.testing.capture_logs() as logs:
        data = await provider._request("news-sentiment", {"symbol": "AAPL"})

    assert data is None
    assert any(log["event"] == "finnhub_forbidden" for log in logs)


async def test_request_raises_on_401_bad_key(provider):
    _wire(provider, FakeResponse(401, text_data='{"error": "Invalid API key"}'))

    with pytest.raises(RuntimeError, match="401"):
        await provider._request("stock/peers", {"symbol": "AAPL"})


async def test_request_retries_once_on_429_then_succeeds(provider, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", lambda *_: _noop())
    session = _wire(
        provider,
        [
            FakeResponse(429, text_data="rate limited"),
            FakeResponse(200, json_data=[{"symbol": "AAPL"}]),
        ],
    )

    data = await provider._request("stock/peers", {"symbol": "AAPL"})

    assert data == [{"symbol": "AAPL"}]
    assert len(session.calls) == 2


async def test_request_raises_after_second_429(provider, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", lambda *_: _noop())
    _wire(provider, FakeResponse(429, text_data="rate limited"))

    with pytest.raises(RuntimeError, match="429"):
        await provider._request("stock/peers", {"symbol": "AAPL"})


async def _noop():
    return None


async def test_request_attaches_api_key_as_token_param(provider):
    session = _wire(provider, FakeResponse(200, json_data=[]))

    await provider._request("company-news", {"symbol": "AAPL"})

    _, params = session.calls[0]
    assert params["token"] == "test-key"
    assert params["symbol"] == "AAPL"


# --- get_news ---


async def test_get_news_maps_fields_and_excludes_sentiment_score(provider):
    now_ts = time.time()
    rows = [
        {
            "headline": "Apple hits new high",
            "summary": "...",
            "source": "Yahoo",
            "url": "https://example.com/1",
            "datetime": now_ts,
        }
    ]
    _wire(provider, FakeResponse(200, json_data=rows))

    result = await provider.get_news("AAPL", 7)

    assert len(result) == 1
    assert result[0]["headline"] == "Apple hits new high"
    assert "sentiment_score" not in result[0]


async def test_get_news_filters_out_articles_older_than_days(provider):
    now_ts = time.time()
    old_ts = now_ts - (400 * 86400)
    rows = [
        {"headline": "Recent", "summary": "", "source": "S", "url": "u1", "datetime": now_ts},
        {"headline": "Old", "summary": "", "source": "S", "url": "u2", "datetime": old_ts},
    ]
    _wire(provider, FakeResponse(200, json_data=rows))

    result = await provider.get_news("AAPL", 7)

    assert len(result) == 1
    assert result[0]["headline"] == "Recent"


async def test_get_news_no_data_returns_empty_list(provider):
    _wire(provider, FakeResponse(200, json_data=[]))

    result = await provider.get_news("X", 7)

    assert result == []


async def test_get_news_ca_ticker_raises_without_calling_api(provider):
    """Confirmed live: Finnhub 403s on every .TO ticker across every endpoint
    used here, not just /stock/peers. Without this guard a misrouted .TO
    ticker would silently look like "no news today" via the generic 403
    handling — must fail loudly instead."""
    session = _wire(provider, FakeResponse(200, json_data=["should not be reached"]))

    with pytest.raises(NotImplementedError, match="Canadian coverage"):
        await provider.get_news("RY.TO", 7)

    assert session.calls == []


# --- get_analyst_recommendation_trends ---


async def test_get_analyst_recommendation_trends_renames_to_snake_case(provider):
    rows = [
        {
            "symbol": "AAPL",
            "period": "2026-07-01",
            "strongBuy": 13,
            "buy": 23,
            "hold": 16,
            "sell": 2,
            "strongSell": 0,
        }
    ]
    _wire(provider, FakeResponse(200, json_data=rows))

    result = await provider.get_analyst_recommendation_trends("AAPL")

    assert result == [
        {
            "period": "2026-07-01",
            "strong_buy": 13,
            "buy": 23,
            "hold": 16,
            "sell": 2,
            "strong_sell": 0,
        }
    ]


async def test_get_analyst_recommendation_trends_no_data_returns_empty_list(provider):
    _wire(provider, FakeResponse(200, json_data=[]))

    result = await provider.get_analyst_recommendation_trends("ZZZZ")

    assert result == []


async def test_get_analyst_recommendation_trends_ca_ticker_raises_without_calling_api(provider):
    session = _wire(provider, FakeResponse(200, json_data=["should not be reached"]))

    with pytest.raises(NotImplementedError, match="Canadian coverage"):
        await provider.get_analyst_recommendation_trends("SHOP.TO")

    assert session.calls == []


# --- get_peers ---


async def test_get_peers_excludes_self_and_applies_limit(provider):
    _wire(provider, FakeResponse(200, json_data=["AAPL", "DELL", "SNDK", "WDC", "HPE", "NTAP"]))

    result = await provider.get_peers("AAPL", limit=3)

    assert result == ["DELL", "SNDK", "WDC"]


async def test_get_peers_ca_ticker_raises_without_calling_api(provider):
    session = _wire(provider, FakeResponse(200, json_data=["should not be reached"]))

    with pytest.raises(NotImplementedError, match="US-only"):
        await provider.get_peers("SHOP.TO")

    assert session.calls == []


async def test_get_peers_etf_returns_empty_list(provider):
    _wire(provider, FakeResponse(200, json_data=[]))

    result = await provider.get_peers("QQQ")

    assert result == []


# --- get_general_news ---


async def test_get_general_news_passes_category_param(provider):
    session = _wire(provider, FakeResponse(200, json_data=[{"headline": "Market news"}]))

    result = await provider.get_general_news()

    _, params = session.calls[0]
    assert params["category"] == "general"
    assert result == [{"headline": "Market news"}]


# --- get_metric ---


async def test_get_metric_returns_flat_metric_dict(provider):
    _wire(
        provider,
        FakeResponse(
            200, json_data={"metric": {"beta": 1.1}, "metricType": "all", "symbol": "AAPL"}
        ),
    )

    result = await provider.get_metric("AAPL")

    assert result == {"beta": 1.1}


async def test_get_metric_no_data_returns_empty_dict(provider):
    _wire(provider, FakeResponse(403, text_data="forbidden"))

    result = await provider.get_metric("ZZZZ")

    assert result == {}


# --- get_earnings_calendar ---


async def test_get_earnings_calendar_queries_by_symbol_within_a_window(provider):
    """86bbpgrjv: the request must carry `symbol` (plus from/to). Without
    it Finnhub returns a bulk feed capped at ~1500 rows that omits the
    requested ticker, so the old client-side filter matched nothing."""
    payload = {"earningsCalendar": [{"symbol": "DDD", "date": "2026-11-02", "epsEstimate": -0.04}]}
    session = _wire(provider, FakeResponse(200, json_data=payload))

    result = await provider.get_earnings_calendar("DDD")

    _, params = session.calls[0]
    assert params["symbol"] == "DDD"
    assert "from" in params and "to" in params
    assert result == [{"symbol": "DDD", "date": "2026-11-02", "epsEstimate": -0.04}]


async def test_get_earnings_calendar_filters_response_by_ticker(provider):
    """Client-side filter kept as a cheap guard against a stray row."""
    payload = {"earningsCalendar": [{"symbol": "RKT"}, {"symbol": "UBER"}, {"symbol": "SONY"}]}
    _wire(provider, FakeResponse(200, json_data=payload))

    result = await provider.get_earnings_calendar("uber")

    assert result == [{"symbol": "UBER"}]


async def test_get_earnings_calendar_no_ticker_returns_all_rows(provider):
    payload = {"earningsCalendar": [{"symbol": "RKT"}, {"symbol": "UBER"}]}
    _wire(provider, FakeResponse(200, json_data=payload))

    result = await provider.get_earnings_calendar()

    assert result == [{"symbol": "RKT"}, {"symbol": "UBER"}]


async def test_get_earnings_calendar_no_data_returns_empty_list(provider):
    _wire(provider, FakeResponse(403, text_data="forbidden"))

    result = await provider.get_earnings_calendar("UBER")

    assert result == []
