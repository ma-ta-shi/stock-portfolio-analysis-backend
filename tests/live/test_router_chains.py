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

from data.providers.router import Router

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


# ---------- Ambiguous bare-vs-.TO ticker collisions ----------
# Router-level, not a single-provider concern — is_canadian_ticker()'s
# suffix check is what decides which chain (and therefore which company)
# a bare ticker resolves to.


async def test_bare_us_ticker_and_dotted_ca_ticker_resolve_to_different_companies():
    """T (NYSE: AT&T) vs T.TO (TSX: TELUS) — confirmed different
    companies; is_canadian_ticker must route them to different chains.

    Uses get_price_history, not get_company_info: real finding 2026-08-04,
    CA get_company_info is currently broken for real tickers by an
    upstream openbb-tmx library bug (see test_provider_completeness.py's
    baseline sweep — RY.TO/etc. all raise a KeyError inside openbb_tmx's
    own symbol-sorting logic), with no fallback since it's a single-link
    CA chain. get_price_history has a real fallback link and isn't
    affected, so it's the reliable way to confirm two different
    companies here."""
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
