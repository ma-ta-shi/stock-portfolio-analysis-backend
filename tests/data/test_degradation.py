"""Tests for data/degradation.py and its hook points (86bc997wr, step 6): data or
model calls that quietly went wrong get recorded, and recording them changes
NOTHING about what the providers return, log, retry or raise.

Router hooks are exercised with the REAL Router and fake providers, the same
way tests/data/test_router.py does. Hermetic: no network, no Ollama.
"""

import asyncio
import json

import pytest

from data.degradation import (
    AUTH_FAILED,
    DATA_MISSING,
    EMPTY_AFTER_FAILURE,
    FETCH_FAILED,
    LINK_FAILED,
    LLM_REQUEST_FAILED,
    NOT_COVERED,
    SENTIMENT_UNSCORED,
    DegradationCollector,
    classify,
    current_collector,
    report,
    reset_collector,
    set_collector,
)
from data.precompute.filing_summarizer import summarize_filing_section
from data.precompute.sentiment import _score_article, summarize_news
from data.providers.router import US_CHAINS, Router


class FakeProvider:
    def __init__(self, **methods) -> None:
        for name, fn in methods.items():
            setattr(self, name, fn)


def _ok(value):
    async def method(*args, **kwargs):
        return value

    return method


def _raises(exc):
    async def method(*args, **kwargs):
        raise exc

    return method


@pytest.fixture
def collector():
    """A collector set as the current one for the duration of a test."""
    c = DegradationCollector()
    token = set_collector(c)
    yield c
    reset_collector(token)


def _us_quote_router(first_method, second_method) -> tuple[Router, str, str]:
    """A real US Router whose get_quote chain is (first, second) providers."""
    first, second = US_CHAINS["get_quote"][:2]
    providers = {
        "fmp": FakeProvider(),
        "edgartools": FakeProvider(),
        "finnhub": FakeProvider(),
        "yfinance": FakeProvider(),
    }
    providers[first] = FakeProvider(get_quote=first_method)
    providers[second] = FakeProvider(get_quote=second_method)
    return Router(ticker="AAPL", **providers), first, second


# --- the collector -------------------------------------------------------------------


def test_the_same_event_is_one_event_with_a_count():
    c = DegradationCollector()
    for _ in range(250):
        c.record("ollama", "sentiment_score", LLM_REQUEST_FAILED, "timed out")
    events = c.drain()
    assert len(events) == 1
    assert events[0].count == 250 and events[0].message == "timed out"
    assert events[0].fingerprint == "data:ollama:sentiment_score:llm_request_failed"


def test_different_providers_ops_and_kinds_are_separate_events():
    c = DegradationCollector()
    c.record("fmp", "get_quote", LINK_FAILED, "a")
    c.record("yfinance", "get_quote", LINK_FAILED, "b")
    c.record("fmp", "get_news", LINK_FAILED, "c")
    c.record("fmp", "get_quote", EMPTY_AFTER_FAILURE, "d")
    assert len(c.drain()) == 4


def test_draining_returns_everything_once_and_starts_over():
    c = DegradationCollector()
    c.record("fmp", "get_quote", LINK_FAILED, "a")
    assert len(c.drain()) == 1
    assert c.drain() == []
    c.record("fmp", "get_quote", LINK_FAILED, "again")
    assert c.drain()[0].count == 1  # a fresh event, not a continuation


def test_recording_never_raises_on_hostile_input():
    class Hostile(Exception):
        def __str__(self):
            raise RuntimeError("no str for you")

    c = DegradationCollector()
    c.record("p", "op", LINK_FAILED, Hostile(), exc=Hostile(), context=object())  # type: ignore[arg-type]
    c.record(None, None, None, None)  # type: ignore[arg-type]
    c.drain()  # and draining does not raise either


def test_an_event_carries_the_exception_type_and_a_bounded_message():
    c = DegradationCollector()
    c.record("fmp", "get_quote", LINK_FAILED, "x" * 5000, exc=ValueError("v"))
    event = c.drain()[0]
    assert event.exc_type == "ValueError" and len(event.message) == 500


@pytest.mark.parametrize(
    "provider, op, kind, context, expected",
    [
        # A bad or expired key needs a human.
        ("fmp", "get_quote", AUTH_FAILED, None, "high"),
        # No provider produced data and one raised: an outage may be hiding it.
        ("fmp+yfinance", "get_quote", EMPTY_AFTER_FAILURE, None, "medium"),
        # A provider raised but the chain moved on and something answered.
        ("fmp", "get_quote", LINK_FAILED, None, "low"),
        ("finnhub", "get_news_crosslisted_us", LINK_FAILED, None, "low"),
        # Unscored articles: by how many, since a stray timeout is normal.
        ("ollama", "sentiment", SENTIMENT_UNSCORED, {"unscored": 248, "total": 248}, "high"),
        ("ollama", "sentiment", SENTIMENT_UNSCORED, {"unscored": 125, "total": 250}, "high"),
        ("ollama", "sentiment", SENTIMENT_UNSCORED, {"unscored": 30, "total": 250}, "medium"),
        ("ollama", "sentiment", SENTIMENT_UNSCORED, {"unscored": 25, "total": 250}, "medium"),
        ("ollama", "sentiment", SENTIMENT_UNSCORED, {"unscored": 5, "total": 250}, "low"),
        # Cannot tell how many: assume the worst rather than hide it.
        ("ollama", "sentiment", SENTIMENT_UNSCORED, None, "high"),
        ("ollama", "sentiment", SENTIMENT_UNSCORED, {"unscored": 1, "total": 0}, "high"),
        # The per-call sentiment failures are detail; sentiment_unscored carries the weight.
        ("ollama", "sentiment_score", LLM_REQUEST_FAILED, None, "low"),
        # A failed filing summary is a whole document the Stock Researcher lacks.
        ("ollama", "filing_summary", LLM_REQUEST_FAILED, {"section": "MDA"}, "medium"),
        # Below the Router (86bc997wr). not_covered is low only where a fallback link
        # follows FMP in US_CHAINS, so the data still arrives.
        ("fmp", "quote", NOT_COVERED, None, "low"),
        ("fmp", "analyst-estimates", NOT_COVERED, None, "low"),
        # ... and medium where nothing follows: ratios-ttm is FMP-only, and Finnhub is
        # the only source for US news, peers and recommendation trends.
        ("fmp", "ratios-ttm", NOT_COVERED, None, "medium"),
        ("finnhub", "company-news", NOT_COVERED, None, "medium"),
        ("fmp", "some-new-endpoint", NOT_COVERED, None, "medium"),
        ("fred", "get_macro_data", FETCH_FAILED, None, "medium"),
        ("statcan", "get_unemployment_rate", DATA_MISSING, None, "medium"),
        # Not listed: visible until someone decides.
        ("newprovider", "some_op", "some_new_kind", None, "medium"),
    ],
)
def test_severity_follows_what_the_degradation_means(provider, op, kind, context, expected):
    assert classify(provider, op, kind, context) == expected


def test_every_fmp_endpoint_marked_as_having_a_fallback_really_has_one_in_us_chains():
    """The low-severity allowlist claims a link follows FMP for each endpoint. If a
    chain in router.py loses its fallback, the decline stops being harmless and this
    fails, instead of the loss silently hiding as a low count."""
    from data.degradation import _FMP_ENDPOINTS_WITH_FALLBACK

    method_for_endpoint = {
        "quote": "get_quote",
        "dividends": "get_dividend_history",
        "historical-price-eod/full": "get_price_history",
        "profile": "get_company_info",
        "analyst-estimates": "get_analyst_estimates",
        "earnings": "get_earnings_surprises",
        "earnings-calendar": "get_earnings_calendar",
    }
    assert set(method_for_endpoint) == set(_FMP_ENDPOINTS_WITH_FALLBACK)  # none unmapped
    for endpoint, method in method_for_endpoint.items():
        chain = US_CHAINS[method]
        assert chain[0] == "fmp" and len(chain) >= 2, (endpoint, method, chain)


def test_the_collector_applies_the_policy_using_the_events_own_context():
    c = DegradationCollector()
    c.record("fmp", "get_quote", AUTH_FAILED, "401")
    c.record("ollama", "sentiment", SENTIMENT_UNSCORED, "n", context={"unscored": 5, "total": 250})
    c.record(
        "ollama", "sentiment", SENTIMENT_UNSCORED, "n", context={"unscored": 200, "total": 250}
    )
    severities = {(e.provider, e.kind): e.severity for e in c.drain()}
    # Deduplicated per (provider, op, kind): the FIRST occurrence's context decided.
    assert severities == {("fmp", AUTH_FAILED): "high", ("ollama", SENTIMENT_UNSCORED): "low"}


# --- the context variable --------------------------------------------------------------


def test_reporting_with_no_collector_is_a_silent_no_op():
    assert current_collector() is None
    report("fmp", "get_quote", LINK_FAILED, "nobody is listening")  # must not raise


def test_reporting_reaches_the_current_collector_and_stops_after_reset():
    c = DegradationCollector()
    token = set_collector(c)
    report("fmp", "get_quote", LINK_FAILED, "a")
    reset_collector(token)
    report("fmp", "get_quote", LINK_FAILED, "b")  # nobody listening again
    assert current_collector() is None
    assert c.drain()[0].count == 1


async def test_tasks_spawned_inside_a_run_report_into_the_same_collector(collector):
    """DataPipeline.prepare() gathers provider calls; each task inherits the
    ContextVar, so all of them land in the one collector the orchestrator set."""

    async def one_call():
        report("yfinance", "get_price_history", LINK_FAILED, "boom")

    await asyncio.gather(one_call(), one_call(), one_call())
    events = collector.drain()
    assert len(events) == 1 and events[0].count == 3


async def test_a_collector_left_set_by_one_run_does_not_leak_into_the_next():
    first = DegradationCollector()
    token = set_collector(first)
    report("fmp", "get_quote", LINK_FAILED, "run 1")
    reset_collector(token)

    second = DegradationCollector()
    token = set_collector(second)
    report("fmp", "get_quote", LINK_FAILED, "run 2")
    reset_collector(token)

    assert first.drain()[0].message == "run 1"
    assert second.drain()[0].message == "run 2"


def test_report_swallows_a_collector_that_itself_breaks(monkeypatch, collector):
    def broken(*_a, **_k):
        raise RuntimeError("collector bug")

    monkeypatch.setattr(DegradationCollector, "record", broken)
    report("fmp", "get_quote", LINK_FAILED, "x")  # must not raise


# --- the Router hook (real Router, fake providers) -----------------------------------------


async def test_a_link_that_raises_is_recorded_and_the_next_link_still_answers(collector):
    router, first, second = _us_quote_router(
        _raises(RuntimeError("fmp is down")), _ok({"symbol": "AAPL", "price": 100})
    )

    result = await router.get_quote("AAPL")

    assert result == {"symbol": "AAPL", "price": 100}
    assert router.sources_used["get_quote"] == second
    events = collector.drain()
    assert [(e.provider, e.op, e.kind, e.exc_type) for e in events] == [
        (first, "get_quote", LINK_FAILED, "RuntimeError")
    ]
    assert events[0].message == "fmp is down"


async def test_every_link_failing_records_each_failure_and_that_no_data_resulted(collector):
    router, first, second = _us_quote_router(_raises(RuntimeError("a")), _raises(ValueError("b")))

    result = await router.get_quote("AAPL")

    assert result == {}  # unchanged: the getter still collapses to empty
    events = collector.drain()
    kinds = sorted((e.provider, e.kind) for e in events)
    assert kinds == sorted(
        [(first, LINK_FAILED), (second, LINK_FAILED), (f"{first}+{second}", EMPTY_AFTER_FAILURE)]
    )


async def test_a_confirmed_empty_result_records_nothing(collector):
    async def empty(*_a, **_k):
        return {}

    router, _, _ = _us_quote_router(empty, empty)

    assert await router.get_quote("AAPL") == {}
    assert collector.drain() == []


async def test_a_provider_that_does_not_cover_the_request_records_nothing(collector):
    router, _, second = _us_quote_router(_raises(NotImplementedError()), _ok({"price": 1}))

    assert await router.get_quote("AAPL") == {"price": 1}
    assert collector.drain() == []


async def test_a_401_still_fails_loud_and_is_also_recorded(collector):
    router, first, _ = _us_quote_router(
        _raises(RuntimeError("HTTP 401 bad key")), _ok({"price": 1})
    )

    with pytest.raises(RuntimeError, match="401"):
        await router.get_quote("AAPL")

    events = collector.drain()
    assert [(e.provider, e.kind) for e in events] == [(first, AUTH_FAILED)]


async def test_a_non_runtime_exception_is_recorded_with_its_type(collector):
    router, first, _ = _us_quote_router(_raises(KeyError("price")), _ok({"price": 1}))

    await router.get_quote("AAPL")

    assert collector.drain()[0].exc_type == "KeyError"


async def _outcome(first_method, second_method):
    router, _, _ = _us_quote_router(first_method, second_method)
    try:
        result = await router.get_quote("AAPL")
    except RuntimeError as exc:
        result = f"raised: {exc}"
    return result, dict(router.sources_used)


@pytest.mark.parametrize(
    "first, second",
    [
        (_raises(RuntimeError("down")), _ok({"price": 1})),
        (_raises(RuntimeError("down")), _raises(ValueError("also down"))),
        (_ok({}), _ok({})),
        (_raises(NotImplementedError()), _ok({"price": 2})),
        (_raises(RuntimeError("HTTP 401")), _ok({"price": 3})),
        (_ok({"price": 4}), _raises(RuntimeError("never reached"))),
    ],
)
async def test_results_are_identical_with_and_without_a_collector(first, second):
    """The whole point: reporting must not change what any provider call returns,
    which link answered, or whether it raised."""
    without = await _outcome(first, second)
    token = set_collector(DegradationCollector())
    try:
        with_collector = await _outcome(first, second)
    finally:
        reset_collector(token)
    assert with_collector == without


async def test_a_broken_collector_does_not_change_router_behavior(monkeypatch):
    def broken(*_a, **_k):
        raise RuntimeError("collector bug")

    monkeypatch.setattr(DegradationCollector, "record", broken)
    token = set_collector(DegradationCollector())
    try:
        result, used = await _outcome(_raises(RuntimeError("down")), _ok({"price": 1}))
    finally:
        reset_collector(token)
    assert result == {"price": 1} and "get_quote" in used


async def test_a_failed_cross_listed_news_merge_is_recorded_and_ca_news_is_still_returned(
    collector,
):
    router = Router(
        ticker="SHOP.TO",
        openbb_tmx=FakeProvider(get_news=_ok([{"url": "https://ca.example/a", "headline": "h"}])),
        yfinance_news=FakeProvider(get_news=_ok([])),
        yfinance=FakeProvider(),
        finnhub=FakeProvider(get_news=_raises(RuntimeError("finnhub down"))),
    )

    articles = await router.get_news("SHOP.TO", 30)

    assert articles == [{"url": "https://ca.example/a", "headline": "h"}]
    events = collector.drain()
    assert [(e.provider, e.op, e.kind) for e in events] == [
        ("finnhub", "get_news_crosslisted_us", LINK_FAILED)
    ]


# --- the precompute LLM hooks ----------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status, json_data=None, raise_timeout=False):
        self.status = status
        self._json_data = json_data
        self._raise_timeout = raise_timeout

    async def __aenter__(self):
        if self._raise_timeout:
            raise asyncio.TimeoutError()
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._json_data


class _FakeSession:
    def __init__(self, response):
        self._response = response

    def post(self, url, json, timeout=None):
        return self._response


async def test_a_sentiment_call_that_gets_an_http_error_is_recorded_and_still_returns_none(
    collector,
):
    result = await _score_article(_FakeSession(_FakeResponse(500)), "headline", "text")

    assert result is None
    event = collector.drain()[0]
    assert (event.provider, event.op, event.kind) == (
        "ollama",
        "sentiment_score",
        LLM_REQUEST_FAILED,
    )
    assert event.message == "HTTP 500"


async def test_a_sentiment_call_that_times_out_is_recorded_with_its_exception_type(collector):
    session = _FakeSession(_FakeResponse(200, raise_timeout=True))

    assert await _score_article(session, "headline", "text") is None

    assert collector.drain()[0].exc_type == "TimeoutError"


async def test_a_successful_sentiment_call_records_nothing(collector):
    body = {"message": {"content": json.dumps({"sentiment": "positive"})}}

    assert await _score_article(_FakeSession(_FakeResponse(200, body)), "h", "t") == "positive"

    assert collector.drain() == []


def _article(id_):
    return {
        "id": id_,
        "date": "2026-08-20",
        "headline": "h",
        "source": "Reuters",
        "quality_tier": "primary",
        "text": "body",
        "url": "https://x.example/a",
    }


async def test_unscored_articles_are_counted_even_though_sentiment_source_reads_as_success(
    monkeypatch, collector
):
    async def fake_batch(session, items, **kwargs):
        return ["positive", None, None, "negative", None]

    async def fake_none(*args, **kwargs):
        return None

    monkeypatch.setattr("data.precompute.sentiment._score_batch", fake_batch)
    monkeypatch.setattr("data.precompute.sentiment._score_article", fake_none)

    result = await summarize_news([_article(f"N{i}") for i in range(5)])

    assert result["sentiment_source"] == "local_llm"  # unchanged: this is the misleading part
    assert [a["sentiment"] for a in result["articles"]] == [
        "positive",
        None,
        None,
        "negative",
        None,
    ]
    event = collector.drain()[0]
    assert (event.provider, event.op, event.kind) == ("ollama", "sentiment", SENTIMENT_UNSCORED)
    assert event.context == {"unscored": 3, "total": 5}
    assert event.message == "3 of 5 articles could not be scored"
    assert event.severity == "high"  # 60% of the articles: the Sentiment agent is half blind


async def test_fully_scored_articles_record_nothing(monkeypatch, collector):
    async def fake_batch(session, items, **kwargs):
        return ["neutral"] * len(items)

    monkeypatch.setattr("data.precompute.sentiment._score_batch", fake_batch)

    await summarize_news([_article("N1"), _article("N2")])

    assert collector.drain() == []


async def test_a_filing_summary_http_error_is_recorded_with_its_section(collector):
    result = await summarize_filing_section(_FakeSession(_FakeResponse(503)), "text", "MDA")

    assert result is None
    event = collector.drain()[0]
    assert (event.provider, event.op, event.kind) == (
        "ollama",
        "filing_summary",
        LLM_REQUEST_FAILED,
    )
    assert event.context == {"section": "MDA"} and "HTTP 503" in event.message


async def test_a_filing_summary_timeout_is_recorded(collector):
    session = _FakeSession(_FakeResponse(200, raise_timeout=True))

    assert await summarize_filing_section(session, "text", "Business") is None

    event = collector.drain()[0]
    assert event.exc_type == "TimeoutError" and event.context == {"section": "Business"}


async def test_an_empty_filing_section_is_not_a_degradation(collector):
    assert (
        await summarize_filing_section(_FakeSession(_FakeResponse(200, {})), "   ", "MDA") is None
    )
    assert collector.drain() == []


# --- which symbol an event was for (ledger BB-041) ------------------------------------


def test_symbols_accumulate_on_one_deduplicated_event_without_a_lone_symbol_key():
    c = DegradationCollector()
    for symbol in ("WMT", "PEP", "WMT", "KO"):
        c.record("finnhub", "get_news", LINK_FAILED, "boom", context={"symbol": symbol})
    (event,) = c.drain()

    assert event.count == 4
    assert event.context["symbols"] == ["WMT", "PEP", "KO"]  # distinct, in first-seen order
    assert "symbol" not in event.context


def test_the_symbol_list_is_capped():
    c = DegradationCollector()
    for i in range(25):
        c.record("finnhub", "get_news", LINK_FAILED, "boom", context={"symbol": f"T{i}"})
    (event,) = c.drain()
    assert len(event.context["symbols"]) == 10 and event.count == 25


def test_an_event_with_no_symbol_has_no_symbols_key():
    c = DegradationCollector()
    c.record("fred", "get_macro_data", LINK_FAILED, "boom", context={"series": "CPIAUCSL"})
    (event,) = c.drain()
    assert event.context == {"series": "CPIAUCSL"}


@pytest.mark.asyncio
async def test_router_events_say_which_ticker_the_failed_call_was_for(collector):
    """A peer's failed news call and the stock's own call looked identical before."""
    router, first, second = _us_quote_router(_raises(RuntimeError("boom")), _ok({}))

    assert await router.get_quote("PEP") == {}  # every link empty, one of them raised

    events = {e.kind: e for e in collector.drain()}
    assert events[LINK_FAILED].context["symbols"] == ["PEP"]
    assert events[EMPTY_AFTER_FAILURE].context["symbols"] == ["PEP"]
