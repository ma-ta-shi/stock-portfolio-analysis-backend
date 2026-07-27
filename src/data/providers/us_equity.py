import pandas as pd
import structlog

from data.providers.base import NewsProvider, StockDataProvider
from data.providers.fmp import FMPDataProvider
from data.providers.yfinance import YFinanceDataProvider

logger = structlog.get_logger(__name__)


def _is_empty(value) -> bool:
    """Uniform emptiness check across dict / list / DataFrame / None — FMP and
    yfinance signal "no data" with different falsy shapes per method."""
    if value is None:
        return True
    if isinstance(value, (dict, list)):
        return len(value) == 0
    if isinstance(value, pd.DataFrame):
        return value.empty
    return False


class USEquityDataProvider(StockDataProvider, NewsProvider):
    """Composite router for US-listed tickers: tries FMPDataProvider first,
    falls back to YFinanceDataProvider whenever FMP has no data for a symbol
    or is unreachable.

    Why this exists: FMP's free tier paywalls (HTTP 402) ETFs and dual-class
    shares across every per-symbol endpoint — confirmed live against QQQ,
    GLD, VOO, IWM, XLK, ARKK, DIA, BRK.B, GOOG (see fmp.py's class
    docstring). Verified live (2026-07-27) that yfinance has full, free
    coverage of price history, dividends, and quotes for all of those plus
    small/mid-cap names FMP handles fine — see
    docs/technical/financial-data-api-research.md for the verification data.

    Scope, deliberately narrow: get_price_history, get_dividend_history,
    get_quote, and get_company_info fall back to yfinance. The first three
    are what the confirmed FMP gap actually blocks and what was verified
    live. get_company_info was added on a second review after live testing
    against real candidate portfolio tickers (2026-07-27) turned up obscure/
    thinly-traded names (e.g. a low-profile micro-cap, a niche themed ETF)
    that plausibly sit outside FMP's free-tier universe entirely, distinct
    from the ETF/dual-class paywall — evidence-driven, not speculative.
    get_ratios_ttm stays FMP-only (see below). get_analyst_estimates,
    get_analyst_ratings, and get_earnings_calendar also stay FMP-only rather
    than growing a yfinance fallback: FMP and yfinance return incompatible
    shapes for all three (different field names, and yfinance's own earnings
    calendar returns a single dict, not the list[dict] the shared ABC
    declares), so a fallback here means inventing and maintaining a merged
    schema with no real consumer yet to validate it against — and the
    ETFs this fix targets don't have EPS/revenue estimates or an earnings
    calendar in any meaningful sense (they're baskets, not companies that
    report earnings). Add a fallback for these if and when an agent actually
    needs analyst data for a dual-class share FMP can't serve; don't build it
    speculatively ahead of that need.

    Design (for the three methods that do fall back): try FMP first — it's
    the better-structured, more stable source when it works, and 250
    calls/day of headroom is otherwise going mostly unused — and fall back to
    yfinance on ANY empty/failed FMP result, not just known ETF/dual-class
    tickers. This needs no static ticker classification list to maintain, and
    it also covers FMP outages (5xx), not just paywall gaps. The cost is one
    wasted round-trip to FMP for tickers that always fail there (e.g. every
    call for a known ETF) — accepted as negligible given the multi-hour/day
    cache TTLs upstream of this provider.

    LIMITATIONS AND CONCERNS — read before wiring this into new call sites
    or assuming it closes the gap completely:

    1. Concentration risk. This makes yfinance/Yahoo the SOLE source for any
       ticker FMP can't serve — there is no third fallback. yfinance is an
       unofficial, unauthenticated scraper of the Yahoo Finance website; if
       Yahoo tightens scraping defenses, ETF and dual-class coverage breaks
       with no safety net, whereas ordinary US stocks still have FMP as a
       working primary.

    2. Rate-limit exposure. Every fallback call adds to yfinance's already
       rate-limited request volume (see yfinance.py's documented 2s-sleep /
       backoff policy, sized originally for ~30 Canadian stocks). Routing
       all US ETF/dual-class price/dividend/quote traffic here too increases
       429 risk under batch load. A live batch test at ~20-symbol scale
       (mixed US common, dual-class, ETF, and Canadian tickers, 2s pacing)
       produced zero 429s, but that is not proof of safety at full production
       universe scale or under concurrent/parallel batch runs — only proof
       it isn't *immediately* broken at the scale tested.

    3. No cross-source reconciliation. If FMP and yfinance ever disagree on
       price/dividend/quote data, this returns whichever one answered first
       (FMP wins on any non-empty result). It does not average, vote, or
       flag discrepancies between providers.

    4. get_ratios_ttm, get_analyst_estimates, get_analyst_ratings, and
       get_earnings_calendar have no yfinance fallback (see class docstring
       above) — for ETFs/dual-class these simply return FMP's empty result
       (`{}` or `[]`) rather than degrading gracefully to a second source.

    5. This class only routes non-`.TO` (US) tickers — `openbb_tmx.py` and
       `edgartools.py` (Canadian prices/dividends/quotes, US fundamentals)
       are separate, not-yet-built pieces of work; not something this fix
       touches or needs to.

    6. Ticker-suffix dispatch is not enforced anywhere in code yet — it's a
       documented convention (see docs/technical/financial-data-api-
       research.md's routing rule), not a real dispatcher. Two confirmed
       live risks for whoever builds that dispatcher: (a) "T" (AT&T, NYSE,
       USD) and "T.TO" (TELUS, TSX, CAD) are completely different companies
       — dropping or inferring the suffix silently fetches the wrong one;
       (b) the `.TO`-only detection rule misses TSX Venture Exchange
       listings, which use `.V` (e.g. `PLAN.V` = Progressive Planet
       Solutions Inc. — bare `PLAN` and `PLAN.TO` both resolve to nothing).
       The rule needs to check `.V` as well as `.TO`.

    7. No bulk quote. get_quote here still calls one symbol at a time through
       whichever provider answers, matching FMP's existing single-symbol
       limitation. yfinance does support genuine multi-symbol bulk download
       (yf.download), confirmed working live for a mixed US+CA basket in one
       call — but wiring that in is a separate future optimization, not part
       of this fallback router.
    """

    def __init__(
        self,
        fmp: FMPDataProvider | None = None,
        yf_provider: YFinanceDataProvider | None = None,
    ) -> None:
        self._fmp = fmp or FMPDataProvider()
        self._yf = yf_provider or YFinanceDataProvider()

    async def __aenter__(self) -> "USEquityDataProvider":
        await self._fmp.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._fmp.__aexit__(exc_type, exc_val, exc_tb)

    async def _try_fmp(self, fmp_coro):
        """Runs an FMP call and treats a RuntimeError as a fallback trigger,
        except for 401 (bad API key) — that's a config error, not a
        per-symbol condition, and should fail loudly rather than silently
        degrade every single call to yfinance."""
        try:
            return await fmp_coro
        except RuntimeError as e:
            if "401" in str(e):
                raise
            logger.warning("fmp_call_failed_falling_back_to_yfinance", error=str(e))
            return None

    # --- The three methods the confirmed FMP gap actually blocks ---

    async def get_price_history(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        data = await self._try_fmp(self._fmp.get_price_history(ticker, period, interval))
        if not _is_empty(data):
            return data
        logger.info("us_equity_fallback_to_yfinance", ticker=ticker, method="get_price_history")
        return await self._yf.get_price_history(ticker, period, interval)

    async def get_quote(self, ticker: str) -> dict:
        """Not on StockDataProvider — used internally for the Portfolio
        Optimizer's bulk price refresh, one symbol per call either way."""
        data = await self._try_fmp(self._fmp.get_quote(ticker))
        if not _is_empty(data):
            return data
        logger.info("us_equity_fallback_to_yfinance", ticker=ticker, method="get_quote")
        return await self._yf.get_quote(ticker)

    async def get_dividend_history(self, ticker: str, from_date: str, to_date: str) -> list[dict]:
        """Normalized shape regardless of source: list[dict] with keys
        {"date": "YYYY-MM-DD", "dividend": float}."""
        data = await self._try_fmp(self._fmp.get_dividend_history(ticker, from_date, to_date))
        if not _is_empty(data):
            return [{"date": row.get("date"), "dividend": row.get("dividend")} for row in data]
        logger.info("us_equity_fallback_to_yfinance", ticker=ticker, method="get_dividend_history")
        df = await self._yf.get_dividend_history(ticker, from_date, to_date)
        if _is_empty(df):
            return []
        return [
            {"date": row["Date"].strftime("%Y-%m-%d"), "dividend": float(row["Dividend"])}
            for _, row in df.iterrows()
        ]

    async def get_company_info(self, ticker: str) -> dict:
        """FMP's /profile is confirmed live to work even for ETFs/dual-class
        (see class docstring) — but that was only verified against
        well-known symbols. Real testing against obscure/thinly-traded
        tickers (e.g. small foreign-listed names) surfaced cases where FMP's
        free-tier universe plausibly doesn't carry a profile at all, distinct
        from the ETF/dual-class paywall — so this one keeps a fallback
        despite the class's general "narrow scope" policy. FMP's shape
        passes through unchanged; the yfinance fallback is tagged
        `"source": "yfinance"` since its field names (`"Company Name"`,
        `"Sector"`, ...) don't match FMP's (`companyName`, `sector`, ...)."""
        data = await self._try_fmp(self._fmp.get_company_info(ticker))
        if not _is_empty(data):
            return data
        logger.info("us_equity_fallback_to_yfinance", ticker=ticker, method="get_company_info")
        yf_result = await self._yf.get_company_info(ticker)
        if not yf_result:
            return {}
        return {"source": "yfinance", **yf_result}

    # --- FMP-only, no fallback (see class docstring, points 1 and 4) ---

    async def get_analyst_estimates(self, ticker: str) -> dict:
        return await self._fmp.get_analyst_estimates(ticker)

    async def get_analyst_ratings(self, ticker: str) -> dict:
        return await self._fmp.get_analyst_ratings(ticker)

    async def get_earnings_calendar(self, ticker: str) -> list[dict]:
        return await self._fmp.get_earnings_calendar(ticker)

    async def get_ratios_ttm(self, ticker: str) -> dict:
        """Not on StockDataProvider. Cross-check only, FMP-only."""
        return await self._fmp.get_ratios_ttm(ticker)

    # --- Out of scope here too — same routing rule as FMPDataProvider ---

    async def get_financials(self, ticker: str, statement: str, period: str) -> pd.DataFrame:
        return await self._fmp.get_financials(ticker, statement, period)

    async def get_insider_trading(self, ticker: str, days: int = 90) -> list[dict]:
        return await self._fmp.get_insider_trading(ticker, days)

    async def get_peers(self, ticker: str, limit: int = 5) -> list[str]:
        return await self._fmp.get_peers(ticker, limit)

    async def get_news(self, ticker: str, days: int) -> list[dict]:
        return await self._fmp.get_news(ticker, days)

    async def get_analyst_recommendation_trends(self, ticker: str) -> list[dict]:
        return await self._fmp.get_analyst_recommendation_trends(ticker)
