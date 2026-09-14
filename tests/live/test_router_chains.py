"""Pass 2 — router fallback-chain verification (ClickUp 86bb7j0kh).

test_router.py already has 89 unit tests confirming the fallback MECHANISM
against fake providers (empty -> fallback, NotImplementedError -> skip,
etc.). This file verifies the chains actually engage under real conditions
— only methods with more than one link can demonstrate a fallback at all
(see US_CHAINS/CA_CHAINS in router.py); single-link methods are covered by
test_provider_completeness.py instead, there's no fallback to verify.

Uses router._try_chain() directly (white-box) to capture which provider
key actually served each result, per the ticket's own instruction.

Also covers the ambiguous bare-vs-.TO ticker collisions from the matrix —
these are fundamentally about Router's is_canadian_ticker() routing
decision, not a single provider's behavior, so they belong here rather
than in test_provider_completeness.py.
"""

import pandas as pd
import pytest

from data.precompute.news_id_assignment import assign_news_ids
from data.providers.router import Router
from data.providers.yfinance import YFinanceDataProvider

import tickers as t

pytestmark = pytest.mark.live


def _is_empty(result) -> bool:
    if isinstance(result, pd.DataFrame):
        return result.empty
    if isinstance(result, (list, dict)):
        return len(result) == 0
    return result is None


def _latest_close(price_df: pd.DataFrame) -> float:
    """Different sources use different close-column casing — yfinance:
    "Close", fmp/openbb_tmx: "close" (confirmed live 2026-08-04, e.g. bare
    "T" resolves via FMP not yfinance since AT&T isn't paywalled, so a
    fixed "Close" assumption breaks unpredictably depending on which link
    in the chain actually served the request)."""
    for col in ("Close", "close"):
        if col in price_df.columns:
            return price_df[col].iloc[-1]
    raise KeyError(f"no close-price column found in {list(price_df.columns)}")


# ---------- US chains — confirmed forcing tickers ----------
# fmp.py's own docstring confirms ETFs 402 on price/quote/dividends (but
# NOT /profile). Real finding 2026-08-04: the dual-class-share paywall is
# per-ticker, not blanket as the docstring implies — GOOG's /profile now
# returns real FMP data (served by "fmp", not a fallback), while BRK.B
# still forces the yfinance fallback. Use BRK.B, not GOOG, to force
# get_company_info's fallback specifically.


@pytest.mark.parametrize("ticker", t.US_ETF)
@pytest.mark.parametrize(
    "method,args",
    [
        ("get_price_history", ("1y", "1d")),
        # Wide date range, not a recent window: ARKK's distributions are
        # infrequent (confirmed live 2026-08-04 — last one 2023-09-08, none
        # since) — a narrow recent range would legitimately return empty
        # regardless of provider coverage, which isn't what this test means
        # to check.
        ("get_dividend_history", ("2015-01-01", "2026-08-04")),
        ("get_quote", ()),
    ],
)
async def test_us_etf_forces_yfinance_fallback(ticker, method, args):
    async with Router(ticker=ticker) as router:
        result, source = await router._try_chain(method, ticker, *args)
    assert not _is_empty(result), f"{method}({ticker}) should have real data via fallback"
    assert source == "yfinance", (
        f"{method}({ticker}) should fall back to yfinance (FMP paywalls ETFs) — "
        f"actually served by {source!r}"
    )


async def test_us_dual_class_forces_yfinance_fallback_on_company_info():
    async with Router(ticker=t.US_DUAL_CLASS_STILL_PAYWALLED) as router:
        result, source = await router._try_chain(
            "get_company_info", t.US_DUAL_CLASS_STILL_PAYWALLED
        )
    assert not _is_empty(result)
    assert source == "yfinance", (
        f"get_company_info({t.US_DUAL_CLASS_STILL_PAYWALLED}) should fall back to yfinance — "
        f"actually served by {source!r}"
    )


async def test_google_dual_class_company_info_now_served_by_fmp_directly():
    """Real finding 2026-08-04: contradicts fmp.py's docstring claim that
    dual-class tickers 402 on every endpoint — GOOG's /profile currently
    works. Documents the actual current behavior rather than assuming the
    docstring is still accurate; if this starts failing, GOOG has gone
    back to being paywalled and the docstring was right after all."""
    async with Router(ticker="GOOG") as router:
        result, source = await router._try_chain("get_company_info", "GOOG")
    assert not _is_empty(result)
    assert source == "fmp"


# ---------- US chains — no confirmed forcing ticker found; these verify
# the PRIMARY engages correctly for a normal ticker (chain integrity),
# not fallback specifically. edgartools.get_financials was live-checked
# against BABA (a 20-F foreign private issuer) during planning and
# succeeded anyway — no real US ticker found this session that fails
# edgartools cleanly enough to force this particular fallback. ----------


@pytest.mark.parametrize("ticker", t.US_LARGE_CAP)
async def test_us_financials_chain_primary_engages(ticker):
    async with Router(ticker=ticker) as router:
        result, source = await router._try_chain("get_financials", ticker, "income", "annual")
    assert not _is_empty(result)
    assert source == "edgartools"


async def test_us_earnings_calendar_chain_primary_engages():
    async with Router(ticker="AAPL") as router:
        result, source = await router._try_chain("get_earnings_calendar", "AAPL")
    assert not _is_empty(result)
    assert source == "fmp"


# ---------- CA chains ----------
# get_price_history: PLAN.V forces openbb_tmx to raise (EmptyDataError,
# not a clean empty — see test_provider_completeness.py's TSXV test) —
# this is the real live test of whether _try_chain's generic exception
# handler actually recovers and falls back to yfinance. Real finding
# 2026-08-04: router.py's own logger.warning(..., exc_info=True) call
# crashed with UnicodeEncodeError on this Windows console when logging
# that exception (reproduced via a raw script; did NOT reproduce under
# pytest's own output capture, likely because pytest's captured stream
# doesn't hit the same cp1252-default-console path a raw `python`/
# `uvicorn` run does) — fixed at the real app's entrypoint
# (src/api/main.py: sys.stdout/stderr.reconfigure(encoding="utf-8")), not
# here, since this test never imports api.main and structlog has no
# central config elsewhere to hook into.


async def test_ca_price_history_survives_openbb_tmx_exception_and_falls_back():
    async with Router(ticker=t.CA_TSXV_UNCOVERED) as router:
        result, source = await router._try_chain(
            "get_price_history", t.CA_TSXV_UNCOVERED, "1y", "1d"
        )
    assert not _is_empty(result)
    assert source == "yfinance"


async def test_ca_price_history_honors_period_not_the_250_bar_default():
    """86bbq7dkv: openbb_tmx.get_price_history() used to ignore `period` and
    return OpenBB's ~250-bar default regardless, silently starving
    weekly_trend and the 3yr drawdown. A 5y request must now return the
    full span."""
    async with Router(ticker="RY.TO") as router:
        result, source = await router._try_chain("get_price_history", "RY.TO", "5y", "1d")
    assert source == "openbb_tmx"
    assert len(result) > 500, f"5y request returned only {len(result)} bars — the old 250-cap bug"
    span_days = (result.index.max() - result.index.min()).days
    assert span_days > 365 * 3, f"5y request spans only {span_days} days"


async def test_price_history_cross_provider_close_alignment():
    """86bbq7dkv: after aligning yfinance to auto_adjust=False, openbb-tmx
    and the yfinance adapter return the same (split-adjusted,
    dividend-unadjusted) close for a CA name."""
    async with Router(ticker="RY.TO") as router:
        tmx, _ = await router._try_chain("get_price_history", "RY.TO", "1y", "1d")
    yf_df = await YFinanceDataProvider().get_price_history("RY.TO", "1y", "1d")

    tmx_close = {
        (d.date() if hasattr(d, "date") else d): v for d, v in tmx["close"].items()
    }
    yf_close = {d.date(): v for d, v in yf_df["Close"].items()}
    shared = sorted(set(tmx_close) & set(yf_close))
    assert len(shared) > 100
    for day in (shared[0], shared[len(shared) // 2], shared[-1]):
        a, b = tmx_close[day], yf_close[day]
        assert abs(a - b) / a < 0.005, f"{day}: openbb-tmx {a} vs yfinance {b} — adjustment mismatch"


async def test_ca_news_house_shape_survives_id_assignment():
    """86bbqh23f: openbb_tmx.get_news() used to return OpenBB's raw
    `date`/`title`/`excerpt` shape, so assign_news_ids() (which keys off
    `published_at`) silently dropped 100% of Canadian articles. This is
    the end-to-end guard: real openbb-tmx CA news → real house shape →
    non-zero survivors. RY.TO, not SHOP.TO — SHOP has no TMX news
    coverage (tracked on 86bbr4azz)."""
    async with Router(ticker="RY.TO") as router:
        raw, source = await router._try_chain("get_news", "RY.TO", 30)
    assert source == "openbb_tmx"
    assert not _is_empty(raw), "RY.TO should have real openbb-tmx news coverage"
    assert all(
        set(row) == {"headline", "summary", "source", "url", "published_at"} for row in raw
    ), f"get_news must return the house shape, got keys {sorted(raw[0])}"

    survivors = assign_news_ids(raw)
    # The bug was a 100% wipeout; > 0 is the real guard. Exact 1:1 survival is
    # covered deterministically by test_openbb_tmx_house_shape_article_survives —
    # asserting it here too would make this live test flaky on a single
    # malformed upstream date.
    assert len(survivors) > 0, "every CA article was dropped by assign_news_ids — the 86bbqh23f bug"
    assert all(s["date"] is not None and s["headline"] for s in survivors)


# --- CA + US news merge for cross-listed tickers (86bbr4azz) ---
# Uses router.get_news() itself, not _try_chain — this is what exercises
# the merge, unlike the single-provider check above.

_WIRE_KEYWORDS = ("newswire", "quotemedia", "business wire")


def _is_wire(article: dict) -> bool:
    return any(k in article["source"].lower() for k in _WIRE_KEYWORDS)


async def test_ca_news_merge_ry_has_both_wire_and_independent_journalism():
    async with Router(ticker="RY.TO") as router:
        result = await router.get_news("RY.TO", 30)
    assert any(_is_wire(a) for a in result), "expected CA wire coverage to survive the merge"
    assert any(not _is_wire(a) for a in result), (
        "expected independent (non-wire) journalism from the Finnhub side of the merge"
    )


async def test_ca_news_merge_shop_gets_real_coverage_where_tmx_has_none():
    """SHOP.TO returns 0 articles from openbb-tmx alone (no TMX news
    coverage for this name, confirmed live) — the sharpest before/after
    case for this ticket. Post-merge it must have real, non-wire
    (Finnhub-sourced) articles."""
    async with Router(ticker="SHOP.TO") as router:
        result = await router.get_news("SHOP.TO", 30)
    assert result, "SHOP.TO should have real news via the Finnhub merge even with 0 TMX coverage"
    assert any(not _is_wire(a) for a in result)


async def test_ca_news_merge_cnr_resolves_to_the_real_us_ticker():
    """CNR.TO's US symbol is CNI (Canadian National Railway) — a naive
    suffix strip would query Finnhub for "CNR" (Core Natural Resources,
    an unrelated company). Confirms the merge fetched the right company's
    news, not just that it fetched something."""
    async with Router(ticker="CNR.TO") as router:
        result = await router.get_news("CNR.TO", 30)
    non_wire = [a for a in result if not _is_wire(a)]
    assert non_wire, "expected Finnhub-sourced articles in the CNR.TO merge"
    combined_text = " ".join(a["headline"] + " " + a["summary"] for a in non_wire).lower()
    assert "canadian national" in combined_text or "cn rail" in combined_text or " cni" in (
        " " + combined_text
    ), "merged articles should be about Canadian National Railway, not a different CNR company"


async def test_ca_quote_chain_always_serves_from_yfinance():
    """Real finding 2026-08-04: OpenBBTMXProvider has no get_quote method
    at all (confirmed via hasattr) — CA_CHAINS used to list it as the
    primary anyway, which _try_chain's `getattr(provider, method_name,
    None)` silently skipped every time, same as a NotImplementedError
    would. Fixed: CA_CHAINS["get_quote"] is now just ["yfinance"] — the
    table matches reality instead of listing a phantom primary."""
    async with Router(ticker=t.CA_CROSSLISTED[0]) as router:
        result, source = await router._try_chain("get_quote", t.CA_CROSSLISTED[0])
    assert not _is_empty(result)
    assert source == "yfinance"


# ---------- asset_type classification (86bbpk6uf part 1) ----------
# Verifies each provider's own classification signal reaches company_info
# through the real Router path: FMP isEtf/isFund on the US chain, openbb-tmx
# issue_type on the CA chain (single-link, no fallback).


@pytest.mark.parametrize(
    "ticker,expected",
    [
        ("AAPL", "equity"),
        (t.US_ETF[0], "etf"),
        (t.CA_CROSSLISTED[0], "equity"),
        (t.CA_ETF_SUFFIXED[0], "etf"),
    ],
)
async def test_company_info_asset_type_reaches_through_router(ticker, expected):
    async with Router(ticker=ticker) as router:
        result, source = await router._try_chain("get_company_info", ticker)
    assert not _is_empty(result), f"get_company_info({ticker}) returned empty via {source!r}"
    assert result["asset_type"] == expected, (
        f"get_company_info({ticker}) served by {source!r} classified "
        f"{result['asset_type']!r}, expected {expected!r}"
    )


# ---------- Ambiguous bare-vs-.TO ticker collisions ----------
# Router-level, not a single-provider concern — is_canadian_ticker()'s
# suffix check is what decides which chain (and therefore which company)
# a bare ticker resolves to.


async def test_bare_us_ticker_and_dotted_ca_ticker_resolve_to_different_companies():
    """T (NYSE: AT&T) vs T.TO (TSX: TELUS) — confirmed different
    companies; is_canadian_ticker must route them to different chains.

    Uses get_price_history, not get_company_info: the upstream openbb-tmx
    symbol-sort KeyError that used to break CA get_company_info was fixed
    2026-08-04 (86bb7j0kh, the bare-symbol transform — RY.TO/etc. resolve
    fine now, see test_company_info_asset_type_reaches_through_router
    above), but get_price_history's real fallback link still makes it the
    cleaner two-different-companies check."""
    async with Router(ticker=t.AMBIGUOUS_US_TICKER) as us_router:
        us_price, _ = await us_router._try_chain(
            "get_price_history", t.AMBIGUOUS_US_TICKER, "1mo", "1d"
        )
    async with Router(ticker=t.AMBIGUOUS_CA_TICKER) as ca_router:
        ca_price, _ = await ca_router._try_chain(
            "get_price_history", t.AMBIGUOUS_CA_TICKER, "1mo", "1d"
        )
    assert not _is_empty(us_price)
    assert not _is_empty(ca_price)
    us_close = _latest_close(us_price)
    ca_close = _latest_close(ca_price)
    assert us_close != pytest.approx(ca_close, rel=0.01), (
        f"AT&T (${us_close}) and TELUS (${ca_close}) should be different prices"
    )


async def test_bare_vgro_is_a_silent_collision_not_a_clean_empty():
    """Confirmed live in prior research: bare VGRO doesn't fail clean like
    other bare Canadian ETF tickers (ZQQ/XGD/XEI) — yfinance matches it to
    a different real security. This asserts the mismatch is still
    happening, since "clean failure" is the wrong expectation here.

    bare VGRO routes to yfinance (capitalized columns, e.g. "Close") and
    VGRO.TO routes to openbb_tmx (lowercase columns, e.g. "close") — two
    different providers with different schemas, confirmed live 2026-08-04;
    pull each source's own close-price column rather than assuming a
    shared column name."""
    async with Router(ticker=t.VGRO_BARE_COLLISION) as bare_router:
        bare_result, _ = await bare_router._try_chain(
            "get_price_history", t.VGRO_BARE_COLLISION, "1mo", "1d"
        )
    async with Router(ticker="VGRO.TO") as real_router:
        real_result, _ = await real_router._try_chain("get_price_history", "VGRO.TO", "1mo", "1d")
    assert not _is_empty(bare_result), "bare VGRO returns SOME data (the collision), not empty"
    bare_close = _latest_close(bare_result)
    real_close = _latest_close(real_result)
    assert bare_close != pytest.approx(real_close, rel=0.01), (
        f"bare VGRO (${bare_close}) and VGRO.TO (${real_close}) should be different "
        f"securities' prices — if this now matches, the collision may be fixed upstream"
    )


async def test_ca_etf_bare_forms_other_than_vgro_fail_clean():
    """Unlike VGRO, ZQQ/XGD's bare forms are confirmed to fail clean
    (empty), not collide with an unrelated security."""
    for bare_ticker in t.CA_ETF_BARE:
        if bare_ticker == t.VGRO_BARE_COLLISION:
            continue
        async with Router(ticker=bare_ticker) as router:
            result, _ = await router._try_chain("get_price_history", bare_ticker, "1mo", "1d")
        assert _is_empty(result), f"bare {bare_ticker} was expected to fail clean, got real data"
