import json
from types import SimpleNamespace

import pandas as pd
import pytest
import structlog

from data.providers.base import NewsProvider, StockDataProvider
from data.providers.router import (
    CA_CHAINS,
    US_CHAINS,
    _DEFAULT_PEERS_PATH,
    Router,
    _is_empty,
    is_canadian,
    is_canadian_ticker,
)


# --- is_canadian / is_canadian_ticker ---


def test_is_canadian_ticker_recognizes_tsx_suffix():
    assert is_canadian_ticker("SHOP.TO") is True


def test_is_canadian_ticker_recognizes_tsxv_suffix():
    """Confirmed live: a .TO-only check misses TSXV listings like PLAN.V."""
    assert is_canadian_ticker("PLAN.V") is True


def test_is_canadian_ticker_bare_ticker_is_not_canadian():
    assert is_canadian_ticker("AAPL") is False


def test_is_canadian_prefers_stock_over_ticker_string():
    """T (bare) is AT&T (NYSE/USD); T.TO is TELUS (TSX/CAD) — a real
    confirmed collision. When a Stock record is available it must win over
    ticker-suffix guessing, even if a ticker string is also passed."""
    stock = SimpleNamespace(primary_exchange="NYSE", currency="USD")
    assert is_canadian(stock, ticker="T.TO") is False


def test_is_canadian_stock_tsxv_is_canadian():
    stock = SimpleNamespace(primary_exchange="TSXV", currency="USD")
    assert is_canadian(stock) is True


def test_is_canadian_stock_cad_currency_is_canadian():
    stock = SimpleNamespace(primary_exchange="OTHER", currency="CAD")
    assert is_canadian(stock) is True


def test_is_canadian_falls_back_to_ticker_when_no_stock():
    assert is_canadian(ticker="T.TO") is True
    assert is_canadian(ticker="T") is False


def test_is_canadian_raises_without_stock_or_ticker():
    with pytest.raises(ValueError):
        is_canadian()


def test_is_canadian_ticker_classifies_tsx_composite_index():
    """^GSPTSE carries no .TO/.V suffix — a suffix-only check misses it
    (86bawptw7). Resolved via the explicit index table, not inference."""
    assert is_canadian_ticker("^GSPTSE") is True


def test_is_canadian_ticker_classifies_sp500_index():
    assert is_canadian_ticker("^GSPC") is False


# --- Fakes for provider chain testing ---


class FakeProvider:
    """Generic fake exposing whichever methods are passed in as async
    callables, so chains can be exercised without touching real APIs."""

    def __init__(self, **methods) -> None:
        for name, fn in methods.items():
            setattr(self, name, fn)


def _ok(value):
    async def method(*args, **kwargs):
        return value

    return method


def _tracked_empty():
    """Returns (calls, method): method is an async fake that records every
    invocation into `calls` and returns an empty dict — real "not called"
    proof for a chain-order assertion. A raising fake doesn't work for this:
    _try_chain's own `except Exception` catches and logs any failure
    (including AssertionError) and falls through to the next link exactly
    like a real provider error would, so the test would pass whether or not
    the provider was actually skipped."""
    calls: list[tuple] = []

    async def method(*args, **kwargs):
        calls.append(args)
        return {}

    return calls, method


def _raises(exc):
    async def method(*args, **kwargs):
        raise exc

    return method


# --- Router construction is branch-scoped (no eager API-key requirement) ---


def test_ca_router_does_not_construct_key_gated_us_providers():
    """Constructing a CA router must not require SP_FMP_API_KEY/SP_FINNHUB_API_KEY
    — those providers raise RuntimeError in __init__ if the key is missing, and
    a CA-only lookup's chains never reference them."""
    router = Router(ticker="SHOP.TO", openbb_tmx=FakeProvider(), yfinance_news=FakeProvider())
    assert "fmp" not in router._providers
    assert "edgartools" not in router._providers
    assert "finnhub" not in router._providers
    assert "openbb_tmx" in router._providers


def test_us_router_does_not_construct_ca_only_providers():
    router = Router(
        ticker="AAPL",
        fmp=FakeProvider(),
        edgartools=FakeProvider(),
        finnhub=FakeProvider(),
    )
    assert "openbb_tmx" not in router._providers
    assert "yfinance_news" not in router._providers
    assert "fmp" in router._providers


def test_router_is_a_stock_data_provider_and_news_provider():
    router = Router(
        ticker="AAPL", fmp=FakeProvider(), edgartools=FakeProvider(), finnhub=FakeProvider()
    )
    assert isinstance(router, StockDataProvider)
    assert isinstance(router, NewsProvider)


# --- _try_chain semantics ---


async def test_chain_returns_first_non_empty_result():
    router = Router(
        ticker="AAPL",
        fmp=FakeProvider(get_quote=_ok({"symbol": "AAPL", "price": 100})),
        edgartools=FakeProvider(),
        finnhub=FakeProvider(),
    )
    result = await router.get_quote("AAPL")
    assert result == {"symbol": "AAPL", "price": 100}


async def test_chain_records_source_used_on_success():
    router = Router(
        ticker="AAPL",
        fmp=FakeProvider(get_quote=_ok({"symbol": "AAPL", "price": 100})),
        edgartools=FakeProvider(),
        finnhub=FakeProvider(),
    )
    await router.get_quote("AAPL")
    assert router.sources_used["get_quote"] == "fmp"


async def test_chain_records_source_used_after_fallback():
    router = Router(
        ticker="AAPL",
        fmp=FakeProvider(get_quote=_ok({})),
        edgartools=FakeProvider(),
        finnhub=FakeProvider(),
        yfinance=FakeProvider(get_quote=_ok({"symbol": "AAPL", "price": 100})),
    )
    await router.get_quote("AAPL")
    assert router.sources_used["get_quote"] == "yfinance"


async def test_index_price_history_override_skips_straight_to_yfinance():
    """^GSPTSE's price history must never try openbb_tmx: confirmed live
    (2026-09-15) it always raises EmptyDataError for index symbols — a
    guaranteed-wasted call CA_CHAINS's normal ordering would otherwise
    make on every benchmark fetch (86bawptw7). The "typical" construction
    pattern — Router built directly for the benchmark ticker, matching how
    the ticket's own live verification exercised this."""
    openbb_calls, openbb_get_price_history = _tracked_empty()
    router = Router(
        ticker="^GSPTSE",
        openbb_tmx=FakeProvider(get_price_history=openbb_get_price_history),
        yfinance=FakeProvider(get_price_history=_ok(pd.DataFrame({"close": [35000]}))),
        yfinance_news=FakeProvider(),
    )
    result = await router.get_price_history("^GSPTSE", "1y", "1d")
    assert openbb_calls == [], "openbb_tmx.get_price_history must never be invoked for ^GSPTSE"
    assert list(result["close"]) == [35000]
    assert router.sources_used["get_price_history"] == "yfinance"


async def test_index_price_history_override_applies_even_when_router_built_for_a_different_ticker():
    """The override must key off the ticker actually passed to
    get_price_history(), not the ticker Router was constructed with — those
    can differ if one Router instance is reused across tickers (e.g. a
    stock's own Router asked to also fetch a benchmark index's price
    history). Constructed here for a real CA stock, RY.TO, then called with
    ^GSPTSE — must still skip openbb_tmx. Edge case beyond the "typical
    usage" test above: this is the one that actually catches a regression
    to construction-time-only resolution, since the two tickers differ."""
    openbb_calls, openbb_get_price_history = _tracked_empty()
    router = Router(
        ticker="RY.TO",
        openbb_tmx=FakeProvider(get_price_history=openbb_get_price_history),
        yfinance=FakeProvider(get_price_history=_ok(pd.DataFrame({"close": [35000]}))),
        yfinance_news=FakeProvider(),
    )
    result = await router.get_price_history("^GSPTSE", "1y", "1d")
    assert openbb_calls == [], "openbb_tmx.get_price_history must never be invoked for ^GSPTSE"
    assert list(result["close"]) == [35000]
    assert router.sources_used["get_price_history"] == "yfinance"


@pytest.mark.parametrize(
    "ticker", ["XLK", "XLF", "XLV", "XLP", "XLY", "XLI", "XLE", "XLB", "XLRE", "XLU", "XLC"]
)
async def test_sector_etf_price_history_override_skips_fmp(ticker):
    """Every SPDR US sector ETF (86bawpty3's sector_etf_price resolution)
    must never try FMP: confirmed live every one 402s on FMP's free tier
    (the same ETF paywall CLAUDE.md already documents), a guaranteed-wasted
    call US_CHAINS's normal ordering would otherwise make on every
    sector-relative-strength fetch for a US stock."""
    fmp_calls, fmp_get_price_history = _tracked_empty()
    router = Router(
        ticker=ticker,
        fmp=FakeProvider(get_price_history=fmp_get_price_history),
        edgartools=FakeProvider(),
        finnhub=FakeProvider(),
        yfinance=FakeProvider(get_price_history=_ok(pd.DataFrame({"close": [100]}))),
    )
    result = await router.get_price_history(ticker, "1y", "1d")
    assert fmp_calls == [], f"fmp.get_price_history must never be invoked for {ticker}"
    assert list(result["close"]) == [100]
    assert router.sources_used["get_price_history"] == "yfinance"


async def test_chain_falls_back_on_empty_result():
    router = Router(
        ticker="AAPL",
        fmp=FakeProvider(get_quote=_ok({})),
        edgartools=FakeProvider(),
        finnhub=FakeProvider(),
        yfinance=FakeProvider(get_quote=_ok({"symbol": "AAPL", "price": 100})),
    )
    result = await router.get_quote("AAPL")
    assert result == {"symbol": "AAPL", "price": 100}


async def test_chain_falls_back_on_not_implemented():
    router = Router(
        ticker="AAPL",
        fmp=FakeProvider(get_company_info=_raises(NotImplementedError("nope"))),
        edgartools=FakeProvider(),
        finnhub=FakeProvider(),
        yfinance=FakeProvider(get_company_info=_ok({"symbol": "AAPL"})),
    )
    result = await router.get_company_info("AAPL")
    assert result == {"symbol": "AAPL"}


async def test_chain_skips_provider_missing_the_method_entirely():
    """The chain must skip a provider lacking the method via getattr rather
    than raising AttributeError. Originally used get_quote for this (real
    openbb_tmx genuinely had no get_quote at all), but CA_CHAINS["get_quote"]
    no longer lists openbb_tmx as primary (fixed 2026-08-04, 86bb7j0kh — it
    was never actually reachable there, so that scenario stopped exercising
    the getattr-skip path once the table was corrected). Uses
    get_price_history instead — still a real 2-link CA chain
    (openbb_tmx -> yfinance) — to keep this mechanism genuinely covered."""
    router = Router(
        ticker="SHOP.TO",
        openbb_tmx=FakeProvider(),  # deliberately no get_price_history
        yfinance=FakeProvider(get_price_history=_ok(pd.DataFrame({"close": [80]}))),
        yfinance_news=FakeProvider(),
    )
    result = await router.get_price_history("SHOP.TO", "1y", "1d")
    assert list(result["close"]) == [80]


async def test_chain_reraises_auth_shaped_runtime_error():
    router = Router(
        ticker="AAPL",
        fmp=FakeProvider(get_quote=_raises(RuntimeError("401 Unauthorized"))),
        edgartools=FakeProvider(),
        finnhub=FakeProvider(),
    )
    with pytest.raises(RuntimeError, match="401"):
        await router.get_quote("AAPL")


async def test_chain_logs_and_continues_on_generic_exception():
    router = Router(
        ticker="AAPL",
        fmp=FakeProvider(get_quote=_raises(ValueError("boom"))),
        edgartools=FakeProvider(),
        finnhub=FakeProvider(),
        yfinance=FakeProvider(get_quote=_ok({"symbol": "AAPL", "price": 5})),
    )
    with structlog.testing.capture_logs() as logs:
        result = await router.get_quote("AAPL")
    assert result == {"symbol": "AAPL", "price": 5}
    assert any(log["event"] == "router_chain_call_failed" for log in logs)


async def test_chain_exhausted_returns_canonical_empty():
    router = Router(
        ticker="AAPL",
        fmp=FakeProvider(get_analyst_estimates=_ok({})),
        yfinance=FakeProvider(get_analyst_estimates=_ok({})),  # US chain is [fmp, yfinance]
        edgartools=FakeProvider(),
        finnhub=FakeProvider(),
    )
    result = await router.get_analyst_estimates("AAPL")
    assert result == {}


# --- get_financials: yfinance-sourced results get correct_alignment ---


async def _misaligned_df() -> pd.DataFrame:
    old_col = pd.Timestamp.now() - pd.DateOffset(months=24)
    return pd.DataFrame({old_col: [1, 2]})


async def test_get_financials_applies_correct_alignment_on_yfinance_fallback():
    router = Router(
        ticker="AAPL",
        fmp=FakeProvider(),
        edgartools=FakeProvider(get_financials=_raises(NotImplementedError())),
        finnhub=FakeProvider(),
        yfinance=FakeProvider(get_financials=lambda *a, **k: _misaligned_df()),
    )
    df, source = await router.get_financials("AAPL", "income", "annual")
    assert (pd.Timestamp.now() - df.columns[0]).days / 30 < 18
    assert source == "yfinance"


async def test_get_financials_does_not_touch_edgartools_sourced_result():
    fixed_col = pd.Timestamp.now() - pd.DateOffset(months=30)
    edgar_df = pd.DataFrame({fixed_col: [1, 2]})
    router = Router(
        ticker="AAPL",
        fmp=FakeProvider(),
        edgartools=FakeProvider(get_financials=_ok(edgar_df)),
        finnhub=FakeProvider(),
    )
    df, source = await router.get_financials("AAPL", "income", "annual")
    assert df.columns[0] == fixed_col  # untouched — only yfinance results get corrected
    assert source == "edgartools"


async def test_get_financials_returns_empty_dataframe_when_chain_exhausted():
    router = Router(
        ticker="AAPL",
        fmp=FakeProvider(),
        edgartools=FakeProvider(get_financials=_ok(pd.DataFrame())),
        finnhub=FakeProvider(),
        yfinance=FakeProvider(get_financials=_ok(pd.DataFrame())),
    )
    df, source = await router.get_financials("AAPL", "income", "annual")
    assert df.empty


# --- get_financials: source exposure (86bbb001k) — covers what the generic
# chain-loop tests above no longer can, since get_financials returns
# (DataFrame, source) rather than a bare DataFrame ---


async def test_get_financials_ca_chain_single_link_source():
    router = Router(
        ticker="SHOP.TO",
        fmp=FakeProvider(),
        edgartools=FakeProvider(),
        finnhub=FakeProvider(),
        yfinance=FakeProvider(get_financials=_ok(pd.DataFrame({pd.Timestamp.now(): [1]}))),
    )
    df, source = await router.get_financials("SHOP.TO", "income", "annual")
    assert not df.empty
    assert source == "yfinance"


async def test_get_financials_ca_chain_exhausted_returns_none_source():
    router = Router(
        ticker="SHOP.TO",
        fmp=FakeProvider(),
        edgartools=FakeProvider(),
        finnhub=FakeProvider(),
        yfinance=FakeProvider(get_financials=_ok(pd.DataFrame())),
    )
    df, source = await router.get_financials("SHOP.TO", "income", "annual")
    assert df.empty
    assert source is None


# --- CA get_peers: static data/peers.json, not a live provider ---


def test_shipped_peers_json_is_well_formed():
    """The committed data/peers.json — not a tmp fixture. Every value must
    be a non-empty list of uppercase ticker strings, and no ticker may list
    itself as its own peer (it would collide in compute_peer_comparison)."""
    data = json.loads(_DEFAULT_PEERS_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and data
    for anchor_ticker, peer_list in data.items():
        assert isinstance(peer_list, list) and peer_list, anchor_ticker
        assert all(isinstance(p, str) and p == p.upper() for p in peer_list), anchor_ticker
        assert anchor_ticker.upper() not in {p.upper() for p in peer_list}, anchor_ticker
        assert len(peer_list) == len(set(peer_list)), f"{anchor_ticker} has duplicate peers"


def test_shipped_peers_json_gives_every_entry_at_least_two_peers():
    """< 2 peers trips the Fundamental/Researcher "limited peer data"
    penalty, which is the whole reason this file exists (audit S6). Any
    entry worth adding must clear that bar."""
    data = json.loads(_DEFAULT_PEERS_PATH.read_text(encoding="utf-8"))
    thin = {t: p for t, p in data.items() if len(p) < 2}
    assert not thin, f"entries with <2 peers defeat the purpose: {thin}"


async def test_ca_get_peers_reads_the_real_shipped_file():
    """End to end through Router against the committed file, no fixture."""
    router = Router(ticker="RY.TO", openbb_tmx=FakeProvider(), yfinance_news=FakeProvider())
    assert await router.get_peers("RY.TO", limit=3) == ["TD.TO", "BNS.TO", "BMO.TO"]
    # a cross-listed name intentionally carries US peers (audit 86bbr4azz logic)
    assert await router.get_peers("SHOP.TO", limit=3) == ["XYZ", "WIX", "GDDY"]


async def test_ca_get_peers_reads_static_file(tmp_path):
    peers_file = tmp_path / "peers.json"
    peers_file.write_text(json.dumps({"RY.TO": ["TD.TO", "BNS.TO", "BMO.TO"]}))
    router = Router(
        ticker="RY.TO",
        openbb_tmx=FakeProvider(),
        yfinance_news=FakeProvider(),
        peers_path=peers_file,
    )
    result = await router.get_peers("RY.TO", limit=2)
    assert result == ["TD.TO", "BNS.TO"]


async def test_ca_get_peers_unknown_ticker_returns_empty_list(tmp_path):
    peers_file = tmp_path / "peers.json"
    peers_file.write_text(json.dumps({"RY.TO": ["TD.TO"]}))
    router = Router(
        ticker="XYZ.TO",
        openbb_tmx=FakeProvider(),
        yfinance_news=FakeProvider(),
        peers_path=peers_file,
    )
    assert await router.get_peers("XYZ.TO") == []


async def test_ca_get_peers_empty_file_returns_empty_list(tmp_path):
    peers_file = tmp_path / "peers.json"
    peers_file.write_text("")
    router = Router(
        ticker="RY.TO",
        openbb_tmx=FakeProvider(),
        yfinance_news=FakeProvider(),
        peers_path=peers_file,
    )
    assert await router.get_peers("RY.TO") == []


async def test_ca_get_peers_missing_file_returns_empty_list(tmp_path):
    router = Router(
        ticker="RY.TO",
        openbb_tmx=FakeProvider(),
        yfinance_news=FakeProvider(),
        peers_path=tmp_path / "does_not_exist.json",
    )
    assert await router.get_peers("RY.TO") == []


async def test_ca_get_peers_invalid_json_returns_empty_list(tmp_path):
    peers_file = tmp_path / "peers.json"
    peers_file.write_text("{not valid json")
    router = Router(
        ticker="RY.TO",
        openbb_tmx=FakeProvider(),
        yfinance_news=FakeProvider(),
        peers_path=peers_file,
    )
    with structlog.testing.capture_logs() as logs:
        result = await router.get_peers("RY.TO")
    assert result == []
    assert any(log["event"] == "router_peers_json_invalid" for log in logs)


async def test_us_get_peers_uses_finnhub_chain():
    router = Router(
        ticker="AAPL",
        fmp=FakeProvider(),
        edgartools=FakeProvider(),
        finnhub=FakeProvider(get_peers=_ok(["MSFT", "GOOG"])),
    )
    result = await router.get_peers("AAPL")
    assert result == ["MSFT", "GOOG"]


# --- get_ratios_ttm: carried forward from us_equity.py, FMP-only cross-check ---


async def test_get_ratios_ttm_us_calls_fmp():
    router = Router(
        ticker="AAPL",
        fmp=FakeProvider(get_ratios_ttm=_ok({"pe": 30})),
        edgartools=FakeProvider(),
        finnhub=FakeProvider(),
    )
    assert await router.get_ratios_ttm("AAPL") == {"pe": 30}


async def test_get_ratios_ttm_ca_returns_empty_without_fmp():
    router = Router(ticker="SHOP.TO", openbb_tmx=FakeProvider(), yfinance_news=FakeProvider())
    assert await router.get_ratios_ttm("SHOP.TO") == {}


# --- get_filings: CA-only new capability (86bbpggr5), no US equivalent ---


async def test_get_filings_ca_calls_openbb_tmx():
    router = Router(
        ticker="RY.TO",
        openbb_tmx=FakeProvider(get_filings=_ok([{"report_type": "MD&A"}])),
        yfinance_news=FakeProvider(),
    )
    assert await router.get_filings("RY.TO") == [{"report_type": "MD&A"}]


async def test_get_filings_us_returns_empty_no_chain():
    router = Router(
        ticker="AAPL", fmp=FakeProvider(), edgartools=FakeProvider(), finnhub=FakeProvider()
    )
    assert await router.get_filings("AAPL") == []


# --- get_short_interest: not on the ABC, yfinance .info is the only source ---


async def test_get_short_interest_us_calls_yfinance():
    router = Router(
        ticker="AAPL",
        yfinance=FakeProvider(get_short_interest=_ok({"short_interest_pct": 0.8})),
        fmp=FakeProvider(),
        edgartools=FakeProvider(),
        finnhub=FakeProvider(),
    )
    assert await router.get_short_interest("AAPL") == {"short_interest_pct": 0.8}


async def test_get_short_interest_ca_also_calls_yfinance():
    router = Router(
        ticker="SHOP.TO",
        yfinance=FakeProvider(get_short_interest=_ok({"days_to_cover": 4.0})),
        openbb_tmx=FakeProvider(),
        yfinance_news=FakeProvider(),
    )
    assert await router.get_short_interest("SHOP.TO") == {"days_to_cover": 4.0}


# --- CA get_analyst_recommendation_trends routes to the yfinance news class ---


async def test_ca_get_analyst_recommendation_trends_uses_yfinance_news():
    router = Router(
        ticker="SHOP.TO",
        openbb_tmx=FakeProvider(),
        yfinance_news=FakeProvider(get_analyst_recommendation_trends=_ok([{"period": "0m"}])),
    )
    result = await router.get_analyst_recommendation_trends("SHOP.TO")
    assert result == [{"period": "0m"}]


# --- CA + US news merge for cross-listed tickers (86bbr4azz) ---
# SHOP.TO -> SHOP and CNR.TO -> CNI are real entries in the shipped
# ca_us_crosslisting.json — used as-is rather than mocked, so these also
# double as a live check that the real map still resolves the way the
# merge logic assumes. ZZZZ.TO is the file's existing convention for "not
# in any map" (see test_is_crosslisted_false_for_unmapped_ticker).


def _ca_router(**overrides) -> Router:
    """Not _make_router below: that helper hardcodes ticker="SHOP.TO" for
    every is_ca=True router, one ticker per branch. These tests need three
    different real CA tickers (SHOP.TO, ZZZZ.TO, CNR.TO) to exercise three
    different real crosslisting-map outcomes, so the ticker has to be a
    per-test override, not a fixed default."""
    kwargs = {"yfinance": FakeProvider(), "yfinance_news": FakeProvider()}
    kwargs.update(overrides)
    return Router(**kwargs)


async def test_cross_listed_ca_ticker_merges_ca_and_us_news():
    router = _ca_router(
        ticker="SHOP.TO",
        openbb_tmx=FakeProvider(get_news=_ok([{"headline": "CA wire", "url": "https://ca/1"}])),
        finnhub=FakeProvider(get_news=_ok([{"headline": "US journalism", "url": "https://us/1"}])),
    )
    result = await router.get_news("SHOP.TO", 30)
    assert {a["headline"] for a in result} == {"CA wire", "US journalism"}


async def test_non_cross_listed_ca_ticker_stays_ca_only_and_never_calls_finnhub():
    calls = []

    async def _tracked_get_news(ticker, days):
        calls.append(ticker)
        return [{"headline": "should never be reached"}]

    router = _ca_router(
        ticker="ZZZZ.TO",
        openbb_tmx=FakeProvider(get_news=_ok([{"headline": "CA only", "url": "https://ca/1"}])),
        finnhub=FakeProvider(get_news=_tracked_get_news),
    )
    result = await router.get_news("ZZZZ.TO", 30)
    assert [a["headline"] for a in result] == ["CA only"]
    assert calls == [], "a non-cross-listed CA ticker must never call Finnhub"


async def test_merge_dedupes_on_overlapping_url():
    router = _ca_router(
        ticker="SHOP.TO",
        openbb_tmx=FakeProvider(get_news=_ok([{"headline": "Story", "url": "https://shared/1"}])),
        finnhub=FakeProvider(
            get_news=_ok([{"headline": "Same story via Finnhub", "url": "https://shared/1"}])
        ),
    )
    result = await router.get_news("SHOP.TO", 30)
    assert len(result) == 1


async def test_merge_does_not_collide_articles_with_no_url():
    """The house news shape defaults url to "" when a provider has none —
    two unrelated articles that both lack one must not dedupe together."""
    router = _ca_router(
        ticker="SHOP.TO",
        openbb_tmx=FakeProvider(get_news=_ok([{"headline": "CA, no url", "url": ""}])),
        finnhub=FakeProvider(get_news=_ok([{"headline": "US, no url", "url": ""}])),
    )
    result = await router.get_news("SHOP.TO", 30)
    assert len(result) == 2


async def test_finnhub_failure_degrades_to_ca_only_no_exception():
    router = _ca_router(
        ticker="SHOP.TO",
        openbb_tmx=FakeProvider(get_news=_ok([{"headline": "CA only", "url": "https://ca/1"}])),
        finnhub=FakeProvider(get_news=_raises(RuntimeError("SP_FINNHUB_API_KEY is not set"))),
    )
    result = await router.get_news("SHOP.TO", 30)
    assert [a["headline"] for a in result] == ["CA only"]


async def test_merge_resolves_the_real_us_ticker_not_a_suffix_strip():
    """CNR.TO's real US symbol is CNI — bare "CNR" is Core Natural
    Resources, an unrelated company."""
    calls = []

    async def _tracked_get_news(us_ticker, days):
        calls.append(us_ticker)
        return [{"headline": "CNI news", "url": "https://us/1"}]

    router = _ca_router(
        ticker="CNR.TO",
        openbb_tmx=FakeProvider(get_news=_ok([{"headline": "CA wire", "url": "https://ca/1"}])),
        finnhub=FakeProvider(get_news=_tracked_get_news),
    )
    await router.get_news("CNR.TO", 30)
    assert calls == ["CNI"]


async def test_us_ticker_get_news_unaffected_by_ca_merge_logic():
    router = Router(
        ticker="AAPL",
        fmp=FakeProvider(),
        edgartools=FakeProvider(),
        finnhub=FakeProvider(get_news=_ok([{"headline": "US news", "url": "https://us/1"}])),
    )
    result = await router.get_news("AAPL", 7)
    assert result == [{"headline": "US news", "url": "https://us/1"}]


# --- __aenter__/__aexit__ propagate to session-owning providers only ---


async def test_context_manager_propagates_to_session_owning_providers():
    calls = []

    class SessionProvider:
        async def __aenter__(self):
            calls.append("enter")
            return self

        async def __aexit__(self, *exc):
            calls.append("exit")

    router = Router(
        ticker="AAPL",
        fmp=SessionProvider(),
        edgartools=FakeProvider(),
        finnhub=FakeProvider(),
    )
    async with router:
        pass
    assert calls == ["enter", "exit"]


# --- Systematic coverage: every method in both real chain tables ---
#
# The tests above exercise the shared _try_chain mechanism through a handful
# of representative methods. These walk US_CHAINS/CA_CHAINS themselves (the
# real tables the router uses, not a hand-maintained copy) so every method
# gets both a "first provider succeeds" and a "chain exhausted -> canonical
# empty" case, and a typo'd method_name string passed to _try_chain would
# show up as a KeyError here even for methods no other test happens to call.

_METHOD_ARGS: dict[str, tuple] = {
    "get_price_history": ("AAPL", "1y", "1d"),
    "get_financials": ("AAPL", "income", "annual"),
    "get_dividend_history": ("AAPL", "2024-01-01", "2024-12-31"),
    "get_quote": ("AAPL",),
    "get_company_info": ("AAPL",),
    "get_business_summary": ("AAPL",),
    "get_analyst_estimates": ("AAPL",),
    "get_analyst_ratings": ("AAPL",),
    "get_earnings_surprises": ("AAPL",),
    "get_insider_trading": ("AAPL", 90),
    "get_earnings_calendar": ("AAPL",),
    "get_peers": ("AAPL", 5),
    "get_news": ("AAPL", 7),
    "get_analyst_recommendation_trends": ("AAPL",),
    "get_ratios_ttm": ("AAPL",),
    "get_filings": ("AAPL", 20),
    "get_short_interest": ("AAPL",),
}

_METHOD_EMPTY_TYPE: dict[str, type] = {
    "get_price_history": pd.DataFrame,
    "get_financials": pd.DataFrame,
    "get_dividend_history": list,
    "get_quote": dict,
    "get_company_info": dict,
    "get_business_summary": type(None),
    "get_analyst_estimates": dict,
    "get_analyst_ratings": dict,
    "get_earnings_surprises": list,
    "get_insider_trading": list,
    "get_earnings_calendar": list,
    "get_peers": list,
    "get_news": list,
    "get_analyst_recommendation_trends": list,
    "get_ratios_ttm": dict,
    "get_filings": list,
    "get_short_interest": dict,
}

# Recent timestamp column so get_financials samples are a correct_alignment no-op.
_METHOD_SAMPLE_VALUE: dict[str, object] = {
    "get_price_history": pd.DataFrame({"close": [1.0]}),
    "get_financials": pd.DataFrame({pd.Timestamp.now(): [1]}),
    "get_dividend_history": [{"date": "2024-01-01", "amount": 0.5}],
    "get_quote": {"symbol": "AAPL", "price": 100},
    "get_company_info": {"symbol": "AAPL"},
    "get_business_summary": "A leading widget maker.",
    "get_analyst_estimates": {"forward_eps": 1.0},
    "get_analyst_ratings": {
        "consensus_rating": "buy",
        "num_analysts": 11,
        "target_mean": 307.76,
        "buy_count": 8,
        "hold_count": 3,
        "sell_count": 0,
    },
    "get_earnings_surprises": [{"period_end": "2026-07-30", "eps_actual": 2.02}],
    "get_insider_trading": [{"insider_name": "Jane"}],
    "get_earnings_calendar": [{"date": "2024-01-01"}],
    "get_peers": ["MSFT"],
    "get_news": [{"headline": "x"}],
    "get_analyst_recommendation_trends": [{"period": "0m"}],
    "get_ratios_ttm": {"pe": 10},
    "get_filings": [{"filing_date": "2026-08-01", "report_type": "MD&A"}],
    "get_short_interest": {
        "short_interest_pct": 0.8,
        "days_to_cover": 2.19,
        "shares_short": 116_327_753,
        "shares_short_prior_month": 146_547_784,
        "as_of_date": "2026-08-14",
        "prior_month_date": "2026-07-15",
    },
}

_US_BRANCH_KEYS = ["fmp", "edgartools", "finnhub"]
_CA_BRANCH_KEYS = ["openbb_tmx", "yfinance_news"]


def _make_router(is_ca: bool, **method_impls: dict[str, object]) -> Router:
    """Every provider key the branch needs gets a bare FakeProvider() override
    so construction never touches a real API key. method_impls overrides
    specific provider keys with real method implementations."""
    kwargs = {key: FakeProvider() for key in (_CA_BRANCH_KEYS if is_ca else _US_BRANCH_KEYS)}
    kwargs["yfinance"] = FakeProvider()
    for key, methods in method_impls.items():
        kwargs[key] = FakeProvider(**methods)
    return Router(ticker="SHOP.TO" if is_ca else "AAPL", **kwargs)


def _values_equal(a, b) -> bool:
    if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
        return a.equals(b)
    return a == b


# get_peers is special-cased on the CA branch (static file, not a live
# provider chain) — covered by its own dedicated tests above instead.
# get_financials returns (DataFrame, source) since 86bbb001k, not a bare
# DataFrame — the generic loop's isinstance(result, _METHOD_EMPTY_TYPE[method])
# check doesn't fit its richer contract; covered by its own dedicated tests
# instead (see "get_financials: source exposure" below).
# get_news stays in this generic table and still exercises _try_chain
# correctly, but its 86bbr4azz CA/US merge behavior is NOT covered here:
# _METHOD_ARGS uses "AAPL" for every method regardless of branch, and
# "AAPL" is never a crosslisting-map key, so the merge branch is always a
# no-op below. That's exercised by the dedicated tests above instead — if
# _METHOD_ARGS["get_news"] ever changes to a real cross-listed CA ticker,
# these generic tests would start depending on the real crosslisting map.
_CA_CHAIN_METHODS = [m for m in CA_CHAINS if m not in ("get_peers", "get_financials")]
_US_CHAIN_METHODS = [m for m in US_CHAINS if m != "get_financials"]


@pytest.mark.parametrize("method", sorted(_US_CHAIN_METHODS))
async def test_us_method_exhausted_chain_returns_canonical_empty(method):
    router = _make_router(is_ca=False)
    result = await getattr(router, method)(*_METHOD_ARGS[method])
    assert isinstance(result, _METHOD_EMPTY_TYPE[method])
    assert _is_empty(result)


@pytest.mark.parametrize("method", sorted(m for m in _US_CHAIN_METHODS if US_CHAINS[m]))
async def test_us_method_first_provider_in_chain_succeeds(method):
    first_key = US_CHAINS[method][0]
    value = _METHOD_SAMPLE_VALUE[method]
    router = _make_router(is_ca=False, **{first_key: {method: _ok(value)}})
    result = await getattr(router, method)(*_METHOD_ARGS[method])
    assert _values_equal(result, value)


@pytest.mark.parametrize("method", sorted(_CA_CHAIN_METHODS))
async def test_ca_method_exhausted_chain_returns_canonical_empty(method):
    router = _make_router(is_ca=True)
    result = await getattr(router, method)(*_METHOD_ARGS[method])
    assert isinstance(result, _METHOD_EMPTY_TYPE[method])
    assert _is_empty(result)


@pytest.mark.parametrize("method", sorted(m for m in _CA_CHAIN_METHODS if CA_CHAINS[m]))
async def test_ca_method_first_provider_in_chain_succeeds(method):
    first_key = CA_CHAINS[method][0]
    value = _METHOD_SAMPLE_VALUE[method]
    router = _make_router(is_ca=True, **{first_key: {method: _ok(value)}})
    result = await getattr(router, method)(*_METHOD_ARGS[method])
    assert _values_equal(result, value)


@pytest.mark.parametrize("method", sorted(m for m in _US_CHAIN_METHODS if len(US_CHAINS[m]) > 1))
async def test_us_method_falls_back_past_empty_first_provider(method):
    chain = US_CHAINS[method]
    value = _METHOD_SAMPLE_VALUE[method]
    router = _make_router(
        is_ca=False,
        **{
            chain[0]: {method: _ok(_METHOD_EMPTY_TYPE[method]())},
            chain[1]: {method: _ok(value)},
        },
    )
    result = await getattr(router, method)(*_METHOD_ARGS[method])
    assert _values_equal(result, value)


@pytest.mark.parametrize("method", sorted(m for m in _CA_CHAIN_METHODS if len(CA_CHAINS[m]) > 1))
async def test_ca_method_falls_back_past_empty_first_provider(method):
    chain = CA_CHAINS[method]
    value = _METHOD_SAMPLE_VALUE[method]
    router = _make_router(
        is_ca=True,
        **{
            chain[0]: {method: _ok(_METHOD_EMPTY_TYPE[method]())},
            chain[1]: {method: _ok(value)},
        },
    )
    result = await getattr(router, method)(*_METHOD_ARGS[method])
    assert _values_equal(result, value)


def test_every_stockdataprovider_and_newsprovider_method_has_both_chains():
    """Guards against a method being added to the ABC/router without a
    matching entry in one or both static chain tables."""
    router_methods = {
        name for name in dir(Router) if not name.startswith("_") and callable(getattr(Router, name))
    } - {"get_ratios_ttm", "get_filings", "get_short_interest"}  # not on either ABC, checked below
    abc_methods = {
        name
        for base in (StockDataProvider, NewsProvider)
        for name in dir(base)
        if not name.startswith("_")
    }
    assert abc_methods <= router_methods
    for method in abc_methods:
        assert method in US_CHAINS, f"{method} missing from US_CHAINS"
        assert method in CA_CHAINS, f"{method} missing from CA_CHAINS"
    assert "get_ratios_ttm" in US_CHAINS
    assert "get_ratios_ttm" in CA_CHAINS
    assert "get_filings" in US_CHAINS
    assert "get_filings" in CA_CHAINS
    assert "get_short_interest" in US_CHAINS
    assert "get_short_interest" in CA_CHAINS
