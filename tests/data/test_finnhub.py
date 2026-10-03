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


# --- degradation reporting (86bc997wr) ---


async def test_403_is_reported_as_not_covered_and_changes_nothing(provider, collector):
    _wire(provider, FakeResponse(403, text_data='{"error": "You don\'t have access."}'))

    data = await provider._request("news-sentiment", {"symbol": "AAPL"})

    assert data is None  # unchanged
    assert [e.key for e in collector.drain()] == [("finnhub", "news-sentiment", "not_covered")]


async def test_a_successful_finnhub_request_reports_nothing(provider, collector):
    _wire(provider, FakeResponse(200, json_data=[{"headline": "x"}]))
    assert await provider._request("company-news", {"symbol": "AAPL"}) == [{"headline": "x"}]
    assert collector.drain() == []


async def test_a_429_that_recovers_reports_nothing(provider, collector, monkeypatch):
    """A retried rate limit recovers on its own; recording it would be noise."""
    responses = [FakeResponse(429, text_data="slow down"), FakeResponse(200, json_data=[1])]
    _wire(provider, responses)
    monkeypatch.setattr("data.providers.finnhub.asyncio.sleep", lambda *_: _noop())
    assert await provider._request("company-news", {"symbol": "AAPL"}) == [1]
    assert collector.drain() == []


# --- get_news(thorough=True): the ~250-row cap on one response (BB-023) ---


class _BusySession:
    """A Finnhub that, like the real one, answers any from/to window with only the newest
    ~250 matching rows. `per_day` is how many articles each day (0 = today) really has."""

    CAP = 250

    def __init__(self, per_day: dict[int, int], fail_windows: int = 0) -> None:
        from datetime import date, datetime, timedelta

        self.calls: list[dict] = []
        self._fail_left = fail_windows
        self.rows: list[dict] = []
        today = date.today()
        n = 0
        for offset, count in per_day.items():
            day = today - timedelta(days=offset)
            for i in range(count):
                n += 1
                stamp = datetime(day.year, day.month, day.day, 23, 59) - timedelta(minutes=i)
                self.rows.append(
                    {
                        "headline": f"Story {n}",
                        "summary": "",
                        "source": "Yahoo",
                        "url": f"https://x/{n}",
                        "datetime": stamp.timestamp(),
                        "_day": day.isoformat(),
                    }
                )

    def get(self, url: str, params: dict | None = None):
        self.calls.append(params)
        window = [r for r in self.rows if params["from"] <= r["_day"] <= params["to"]]
        window.sort(key=lambda r: r["datetime"], reverse=True)
        if self._fail_left and params["from"] != params["to"] and len(self.calls) > 1:
            self._fail_left -= 1
            return FakeResponse(500, json_data=None)
        return FakeResponse(200, json_data=[{k: v for k, v in r.items() if k != "_day"} for r in window[: self.CAP]])


def _days_covered(articles) -> int:
    return len({a["published_at"][:10] for a in articles})


async def test_get_news_default_is_one_request_even_when_the_answer_is_capped(provider):
    """Peers and every other caller keep the single request."""
    session = _BusySession({d: 100 for d in range(0, 30)})
    provider.session = session

    result = await provider.get_news("MSFT", 30)

    assert len(session.calls) == 1
    assert len(result) == 250  # the cut-off answer, as before


async def test_thorough_a_complete_window_costs_one_request(provider):
    session = _BusySession({d: 3 for d in range(0, 30)})  # 90 articles: nothing cut off
    provider.session = session

    result = await provider.get_news("RDDT", 30, thorough=True)

    assert len(session.calls) == 1
    assert len(result) == 90


async def test_thorough_a_capped_window_is_re_asked_in_three_day_windows(provider):
    """MSFT-like: ~100 articles a day. One request sees about 2 days; the windows cover
    the month, in a fixed number of requests that does not grow with the volume."""
    session = _BusySession({d: 100 for d in range(0, 30)})
    provider.session = session

    result = await provider.get_news("MSFT", 30, thorough=True)

    # 1 probe (newest ~2.5 days) + one request per 3 days over the ~29 days it did not reach
    assert len(session.calls) == 1 + 10
    assert _days_covered(result) >= 28
    assert len({a["url"] for a in result}) == len(result)  # the probe's rows are not repeated


async def test_thorough_call_count_is_bounded_whatever_the_volume(provider):
    small = _BusySession({d: 20 for d in range(0, 30)})  # 600 articles, capped
    huge = _BusySession({d: 400 for d in range(0, 30)})  # 12,000 articles
    provider.session = small
    await provider.get_news("A", 30, thorough=True)
    provider.session = huge
    await provider.get_news("B", 30, thorough=True)

    assert len(small.calls) <= len(huge.calls) <= 12  # never more than probe + 11 windows


async def test_thorough_a_mid_volume_ticker_that_nearly_fits_costs_two_requests(provider):
    """KO-like: ~10 a day, ~310 in the month. The first answer already reaches back almost
    to the start of the window, so only the missing tail is re-asked (it used to be 12)."""
    session = _BusySession({d: 10 for d in range(0, 31)})
    provider.session = session

    result = await provider.get_news("KO", 30, thorough=True)

    assert len(session.calls) == 2
    assert _days_covered(result) >= 30
    assert len({a["url"] for a in result}) == len(result)


async def test_thorough_drops_the_same_story_under_a_second_url_or_headline_spelling(provider):
    now = time.time()
    rows = [
        {"headline": "Apple, Inc. beats!", "summary": "", "source": "A", "url": "u1", "datetime": now},
        {"headline": "apple inc beats", "summary": "", "source": "B", "url": "u2", "datetime": now - 5},
        {"headline": "Other", "summary": "", "source": "A", "url": "u1", "datetime": now - 9},
        {"headline": "Different", "summary": "", "source": "C", "url": "u3", "datetime": now - 20},
    ]
    capped = rows + [
        {"headline": f"Filler {i}", "summary": "", "source": "A", "url": f"f{i}", "datetime": now - 30 - i}
        for i in range(240)
    ]
    session = FakeSession(FakeResponse(200, json_data=capped))
    provider.session = session

    result = await provider.get_news("AAPL", 30, thorough=True)

    headlines = [a["headline"] for a in result]
    assert "Apple, Inc. beats!" in headlines and "apple inc beats" not in headlines  # same headline
    assert "Other" not in headlines  # same url as the first
    assert "Different" in headlines


async def test_thorough_a_failed_window_is_reported_and_the_rest_still_count(provider):
    from data.degradation import DegradationCollector, reset_collector, set_collector

    session = _BusySession({d: 100 for d in range(0, 30)}, fail_windows=2)
    provider.session = session
    collector = DegradationCollector()
    token = set_collector(collector)
    try:
        result = await provider.get_news("MSFT", 30, thorough=True)
    finally:
        reset_collector(token)

    assert _days_covered(result) >= 22  # two 3-day windows lost, the rest kept
    events = collector.drain()
    assert [e.key for e in events] == [("finnhub", "get_news_windows", "fetch_failed")]
    assert "2 of 10" in events[0].message


async def test_thorough_treats_the_cap_threshold_as_cut_off_and_one_below_as_complete(provider):
    complete = _BusySession({0: 239})
    provider.session = complete
    await provider.get_news("A", 30, thorough=True)
    assert len(complete.calls) == 1  # 239 rows: not suspected

    cut_off = _BusySession({0: 240})
    provider.session = cut_off
    await provider.get_news("B", 30, thorough=True)
    assert len(cut_off.calls) > 1  # 240 rows: re-asked
