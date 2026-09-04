"""Pass 1 — per-provider completeness (ClickUp 86bb7j0kh).

Calls each provider adapter DIRECTLY (never through Router — that's pass
2's job in test_router_chains.py) against a deliberately edge-case-heavy
ticker matrix. Assertions target what's actually checkable given there's no
field-level doc contract (see the ClickUp plan): correct return TYPE per
the ABC/type hint (the exact bug class that already slipped past mocked
tests — get_insider_trading returning a DataFrame instead of list[dict]),
and non-emptiness for tickers expected to have coverage vs. clean empty for
tickers expected not to.

Baseline sweep reuses tickers.US_LARGE_CAP / tickers.CA_CROSSLISTED
(already broadly representative) instead of an arbitrary anchor pair —
every method with a real primary implementation gets exercised against all
4 tickers in one of these two categories. This is exactly the sweep that
would have caught openbb_tmx.py's pre-fix dead ends (get_analyst_estimates/
get_insider_trading/get_earnings_calendar/get_news all silently
NotImplementedError for every CA ticker) had it existed before this
ticket. Edge-case categories below are each tested only against the
specific method(s) their own "why" is about — see tickers.py.
"""

import pandas as pd
import pytest

from data.providers.edgartools import EdgarToolsDataProvider
from data.providers.finnhub import FinnhubDataProvider
from data.providers.fmp import FMPDataProvider
from data.providers.openbb_tmx import OpenBBTMXProvider
from data.providers.router import CA_CHAINS, US_CHAINS
from data.providers.yfinance import NewsProvider as YFinanceNewsProvider
from data.providers.yfinance import YFinanceDataProvider

import tickers as t

pytestmark = pytest.mark.live

# Methods that are inherently allowed to be empty even for real, working
# coverage — excluded from the strict "must be non-empty" baseline
# assertion (still gets a correct-type check). get_earnings_calendar: a
# forward-looking ~90-day window legitimately has no entry for most
# tickers most days (RY.TO not reporting this week isn't a coverage gap).
# get_dividend_history: not every large-cap pays a dividend (confirmed
# live 2026-08-04 — MRNA genuinely doesn't); a baseline category mixing
# payers and non-payers can't assert non-empty here without knowing which
# is which per ticker.
_INHERENTLY_SPARSE_METHODS = {"get_earnings_calendar", "get_dividend_history"}

# Narrower than _INHERENTLY_SPARSE_METHODS: a specific (ticker, method) pair
# confirmed live to hit a real openbb-tmx upstream data-quality defect, not
# a coverage gap — SU.TO's filings history has (as of 2026-09) a row with a
# null `description`, which fails Pydantic validation for the WHOLE batch
# inside openbb-tmx's own TmxCompanyFilingsData model, in BOTH get_filings'
# narrow and wide fetch windows (86bbpggr5 — confirmed both tiers raise the
# same OpenBBError). OpenBBTMXProvider.get_filings() degrades this to a
# clean [] rather than crashing (see its own docstring/tests), which is the
# right behavior — but that means SU.TO can never pass the blanket
# non-empty assertion below for this one method, through no fault of the
# fallback logic. Excluded here by exact pair, not by loosening the
# assertion for every ticker (RY.TO/SHOP.TO/CVE.TO still get the real
# check). Revisit if openbb-tmx fixes the upstream row.
_KNOWN_EMPTY_TICKER_METHOD_PAIRS = {("SU.TO", "get_filings")}

_LIST_DICT_METHODS = {
    "get_insider_trading",
    "get_earnings_calendar",
    "get_dividend_history",
    "get_news",
    "get_analyst_recommendation_trends",
    "get_filings",
}
# get_peers returns list[str] per the ABC (StockDataProvider.get_peers),
# not list[dict] — confirmed live 2026-08-04 (Finnhub returns real ticker
# strings like ['DELL', 'SNDK', ...]); a separate category from the other
# list-returning methods, which all return list[dict].
_LIST_STR_METHODS = {"get_peers"}
_DICT_METHODS = {"get_company_info", "get_analyst_estimates", "get_analyst_ratings"}
_DATAFRAME_METHODS = {"get_price_history", "get_financials"}

_ARG_BUILDERS = {
    "get_price_history": lambda ticker: (ticker, "1y", "1d"),
    "get_financials": lambda ticker: (ticker, "income", "annual"),
    "get_company_info": lambda ticker: (ticker,),
    "get_analyst_estimates": lambda ticker: (ticker,),
    "get_analyst_ratings": lambda ticker: (ticker,),
    "get_insider_trading": lambda ticker: (ticker,),
    "get_peers": lambda ticker: (ticker,),
    "get_earnings_calendar": lambda ticker: (ticker,),
    "get_dividend_history": lambda ticker: (ticker, "2024-01-01", "2026-08-04"),
    "get_news": lambda ticker: (ticker, 30),
    "get_analyst_recommendation_trends": lambda ticker: (ticker,),
    "get_filings": lambda ticker: (ticker,),
}

# peers_json isn't a live provider (router-only static file, already
# covered by test_router.py's fakes) — deliberately excluded here.
_PROVIDER_FACTORIES = {
    "fmp": FMPDataProvider,
    "yfinance": YFinanceDataProvider,
    "edgartools": EdgarToolsDataProvider,
    "openbb_tmx": OpenBBTMXProvider,
    "finnhub": FinnhubDataProvider,
    "yfinance_news": YFinanceNewsProvider,
}


async def _call(provider, method_name: str, ticker: str):
    args = _ARG_BUILDERS[method_name](ticker)
    if hasattr(provider, "__aenter__"):
        async with provider:
            return await getattr(provider, method_name)(*args)
    return await getattr(provider, method_name)(*args)


def _assert_correct_type(method_name: str, result) -> None:
    """The exact class of bug already found in yfinance.get_insider_trading
    (DataFrame where list[dict] is promised) — type-check first, always."""
    if method_name in _DATAFRAME_METHODS:
        assert isinstance(result, pd.DataFrame), f"{method_name} must return a DataFrame"
    elif method_name in _LIST_DICT_METHODS:
        assert isinstance(result, list), f"{method_name} must return a list, got {type(result)}"
        assert all(isinstance(row, dict) for row in result), (
            f"{method_name} must return list[dict], got a list with non-dict rows"
        )
    elif method_name in _LIST_STR_METHODS:
        assert isinstance(result, list), f"{method_name} must return a list, got {type(result)}"
        assert all(isinstance(row, str) for row in result), (
            f"{method_name} must return list[str], got a list with non-str rows"
        )
    elif method_name in _DICT_METHODS:
        assert isinstance(result, dict), f"{method_name} must return a dict, got {type(result)}"


def _is_empty(result) -> bool:
    if isinstance(result, pd.DataFrame):
        return result.empty
    if isinstance(result, (list, dict)):
        return len(result) == 0
    return result is None


# ---------- Baseline sweep: every method with a real primary provider,
# against the two broadly-representative categories ----------

_US_PRIMARY_METHODS = [
    (method, chain[0]) for method, chain in US_CHAINS.items() if chain and method in _ARG_BUILDERS
]
_CA_PRIMARY_METHODS = [
    (method, chain[0])
    for method, chain in CA_CHAINS.items()
    if chain and chain[0] != "peers_json" and method in _ARG_BUILDERS
]


@pytest.mark.parametrize("method,provider_key", _US_PRIMARY_METHODS)
@pytest.mark.parametrize("ticker", t.US_LARGE_CAP)
async def test_us_large_cap_baseline_sweep(ticker, method, provider_key):
    provider = _PROVIDER_FACTORIES[provider_key]()
    result = await _call(provider, method, ticker)
    _assert_correct_type(method, result)
    if method in _INHERENTLY_SPARSE_METHODS:
        return
    assert not _is_empty(result), (
        f"{provider_key}.{method}({ticker}) returned empty — expected real "
        f"coverage for a US large-cap baseline ticker"
    )


@pytest.mark.parametrize("method,provider_key", _CA_PRIMARY_METHODS)
@pytest.mark.parametrize("ticker", t.CA_CROSSLISTED)
async def test_ca_crosslisted_baseline_sweep(ticker, method, provider_key):
    provider = _PROVIDER_FACTORIES[provider_key]()
    result = await _call(provider, method, ticker)
    _assert_correct_type(method, result)
    if method in _INHERENTLY_SPARSE_METHODS:
        return
    if (ticker, method) in _KNOWN_EMPTY_TICKER_METHOD_PAIRS:
        return
    assert not _is_empty(result), (
        f"{provider_key}.{method}({ticker}) returned empty — expected real "
        f"coverage for a CA cross-listed baseline ticker"
    )


# ---------- US micro-cap / thin financials ----------
# Already the sparsest cases in test_edgartools.py's market-cap-spread test
# — reused, not reinvented. Confirms real (if sparse) data, not a crash.


@pytest.mark.parametrize("ticker", t.US_MICRO_CAP)
async def test_us_micro_cap_financials_do_not_crash(ticker):
    provider = EdgarToolsDataProvider()
    result = await provider.get_financials(ticker, "income", "annual")
    assert isinstance(result, pd.DataFrame)
    assert not result.empty, (
        f"edgartools.get_financials({ticker}) should have at least a thin statement"
    )


# ---------- US ETF — FMP paywall, confirmed clean (not a crash) at the
# provider level. Fallback engagement is pass 2's job. ----------


@pytest.mark.parametrize("ticker", t.US_ETF)
async def test_fmp_paywalled_etf_fails_clean_not_crash(ticker):
    async with FMPDataProvider() as fmp:
        price = await fmp.get_price_history(ticker, "1y", "1d")
        dividends = await fmp.get_dividend_history(ticker, "2024-01-01", "2026-08-04")
    assert isinstance(price, pd.DataFrame) and price.empty
    assert dividends == []


# ---------- US dual-class/ADR — real finding 2026-08-04: FMP's paywall
# here is per-ticker, not blanket. BABA returns real data (251 rows) on
# every endpoint tested; only GOOG is actually paywalled. fmp.py's own
# docstring's claim ("paywalled outright... across every per-symbol
# endpoint") doesn't hold for BABA — don't assert a specific outcome here,
# just that neither ticker crashes. ----------


@pytest.mark.parametrize("ticker", t.US_DUAL_CLASS_ADR)
async def test_us_dual_class_adr_does_not_crash(ticker):
    async with FMPDataProvider() as fmp:
        price = await fmp.get_price_history(ticker, "1y", "1d")
    assert isinstance(price, pd.DataFrame)


# ---------- Invalid / delisted tickers — must fail clean, never raise ----------


async def test_invalid_ticker_fails_clean_across_us_providers():
    async with FMPDataProvider() as fmp:
        fmp_price = await fmp.get_price_history(t.INVALID_TICKER, "1y", "1d")
        fmp_info = await fmp.get_company_info(t.INVALID_TICKER)
    yf_price = await YFinanceDataProvider().get_price_history(t.INVALID_TICKER, "1y", "1d")
    assert isinstance(fmp_price, pd.DataFrame) and fmp_price.empty
    assert fmp_info == {}
    assert isinstance(yf_price, pd.DataFrame) and yf_price.empty


async def test_delisted_ticker_fails_clean_not_crash():
    """Distinct failure path from INVALID_TICKER's "never existed" — this
    one existed once. Real finding 2026-08-04: FMP's /profile for a
    delisted ticker (ATVI) returns real but STALE data (address, historic
    averageVolume, etc.) from before the delisting — not empty, not an
    error. That's a legitimate, distinct third outcome (neither "clean
    empty" nor "crash") worth knowing about, not something to force into
    either bucket; the only real assertion here is "doesn't raise"."""
    async with FMPDataProvider() as fmp:
        info = await fmp.get_company_info(t.DELISTED_TICKER)
    assert isinstance(info, dict)
    yf_price = await YFinanceDataProvider().get_price_history(t.DELISTED_TICKER, "1y", "1d")
    assert isinstance(yf_price, pd.DataFrame)


# ---------- CA multi-class shares / REIT trust units — core
# price/financials/dividend trio, the shape that broke the crosslisting
# resolver's naive ticker-guess logic ----------


@pytest.mark.parametrize("ticker", (*t.CA_MULTI_CLASS, *t.CA_REIT_TRUST_UNIT))
async def test_ca_hyphenated_suffix_tickers_get_real_core_data(ticker):
    """Price + financials + dividends. get_dividend_history was previously
    excluded here (raised TypeError for every CA ticker — its result's
    index wasn't a DatetimeIndex) — fixed 2026-08-04 (86bb7j0kh: filter on
    the real `ex_dividend_date` column instead), re-added now that it
    actually works. Not asserting non-empty on dividends specifically since
    not every multi-class/REIT ticker necessarily pays a distribution in
    the exact window queried — just that it returns cleanly."""
    price = await OpenBBTMXProvider().get_price_history(ticker, "1y", "1d")
    financials = await YFinanceDataProvider().get_financials(ticker, "income", "annual")
    dividends = await OpenBBTMXProvider().get_dividend_history(ticker, "2015-01-01", "2026-08-04")
    assert isinstance(price, pd.DataFrame) and not price.empty
    assert isinstance(financials, pd.DataFrame) and not financials.empty
    assert isinstance(dividends, list)


# ---------- TSXV ----------
# Real finding, 2026-08-04, bigger than originally assumed: openbb_tmx.
# get_price_history doesn't just fail clean for a TMX-uncovered ticker —
# it RAISES openbb's own EmptyDataError instead of returning an empty
# DataFrame. And it's not a PLAN.V-specific gap: live-checked all 6 known
# TSXV tickers (PLAN.V plus the 5 confirmed-SEC-cross-listed names —
# SGML.V, MOON.V, GRZ.V, SCZ.V, MKO.V) and EVERY ONE raises the same way.
# SEC cross-listing status (a completely different data source) doesn't
# predict openbb-tmx price coverage — the original "SGML.V is a confirmed-
# covered positive case" assumption was wrong; corrected here rather than
# left stale. yfinance covers all 6 with real data (252 rows each) — it's
# the only real TSXV price source in practice today, despite being listed
# second in CA_CHAINS.


async def test_tsxv_tickers_all_raise_from_openbb_tmx_but_yfinance_covers_them():
    from openbb_core.provider.utils.errors import EmptyDataError

    tsxv_tickers = (t.CA_TSXV_UNCOVERED, t.CA_TSXV_COVERED, t.LOW_ANALYST_COVERAGE)
    for ticker in tsxv_tickers:
        with pytest.raises(EmptyDataError):
            await OpenBBTMXProvider().get_price_history(ticker, "1y", "1d")
        fallback_result = await YFinanceDataProvider().get_price_history(ticker, "1y", "1d")
        assert isinstance(fallback_result, pd.DataFrame) and not fallback_result.empty, (
            f"yfinance should cover {ticker} even though openbb_tmx doesn't"
        )


# ---------- Low analyst coverage ----------
# Real finding 2026-08-04, corrected from the original assumption: MKO.V
# has yfinance's numberOfAnalystOpinions=1 (a DIFFERENT data source), but
# TMX's own consensus endpoint has ZERO entries for it — confirmed live,
# result.results == []. That's what caught the .to_df()-raises-on-empty
# bug fixed in openbb_tmx.py (get_analyst_estimates now checks
# result.results before calling .to_df()). This test now exercises the
# genuinely-empty path specifically: a graceful {}, not a crash, and not
# fabricated "sparse but real" data.


async def test_low_analyst_coverage_ticker_returns_graceful_empty_not_a_crash():
    result = await OpenBBTMXProvider().get_analyst_estimates(t.LOW_ANALYST_COVERAGE)
    assert result == {}


# ---------- Dividend edge cases ----------
# get_dividend_history's "no dividend" path is never explicitly asserted
# on today — that's the US-side assertion below. The CA-side test here
# used to document a real, severe, pre-existing bug (raised TypeError for
# every CA ticker — its result's index wasn't a DatetimeIndex); fixed
# 2026-08-04 (86bb7j0kh: filter on the real `ex_dividend_date` column
# instead) — now confirms the fix live rather than the bug.


async def test_ca_dividend_payer_returns_real_non_empty_history():
    """RY.TO is a decades-long consistent dividend payer — real
    confirmation the 2026-08-04 fix (filter on ex_dividend_date, not the
    index) works against live data, not just the mocked unit tests."""
    result = await OpenBBTMXProvider().get_dividend_history(
        t.DIVIDEND_PAYER_CA, "2015-01-01", "2026-08-04"
    )
    assert isinstance(result, list) and len(result) > 0


async def test_zero_dividend_stock_returns_explicit_empty_list():
    async with FMPDataProvider() as fmp:
        result = await fmp.get_dividend_history(t.DIVIDEND_NONE_US, "2024-01-01", "2026-08-04")
    assert result == []
