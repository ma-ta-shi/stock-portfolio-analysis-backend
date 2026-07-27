import os
import re

import aiohttp
import pandas as pd
import structlog

from data.providers.base import NewsProvider, StockDataProvider

from dotenv import load_dotenv
load_dotenv()

logger = structlog.get_logger(__name__)

BASE_URL = "https://financialmodelingprep.com/stable"

# Verified live against a real free-tier key (2026-07-26 — see ClickUp 86bagzcu5):
# FMP has fully migrated off /api/v3/... onto /stable/... (query-param style).
# Do not add /v3 paths here.

_PERIOD_RE = re.compile(r"^(\d+)(d|mo|y)$")
_PERIOD_UNIT_DAYS = {"d": 1, "mo": 30, "y": 365}


def _period_to_from_date(period: str) -> str:
    """FMP's EOD endpoint takes from/to dates, not yfinance-style period tokens
    (StockDataProvider's ABC signature is shared across both providers)."""
    if period.lower() == "max":
        return "1970-01-01"
    match = _PERIOD_RE.match(period.strip().lower())
    if not match:
        raise ValueError(
            f"Unsupported period '{period}'. Use e.g. '1mo', '6mo', '1y', '5y', or 'max'."
        )
    count, unit = match.groups()
    days = int(count) * _PERIOD_UNIT_DAYS[unit]
    return (pd.Timestamp.now().normalize() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")


class FMPDataProvider(StockDataProvider, NewsProvider):
    """US-stock prices, profiles, and analyst data via FMP's free tier.

    Free-tier gaps confirmed live (2026-07-26/27, ClickUp 86bagzcu5) — not assumed:
    - Dual/multi-class-share tickers (BRK.B, GOOG) are paywalled outright (HTTP 402)
      across every per-symbol endpoint, not just price history. `_request()` treats
      this as "no data for this ticker" rather than a fatal error.
    - ETFs/index funds are paywalled the same way, and much more broadly than the
      dual-class-share case — this is NOT in the original ticket, found by testing
      QQQ, GLD, VOO, IWM, XLK, ARKK, DIA: all HTTP 402 on price history/quote/
      dividends, only `/profile` works. SPY is the sole free exception (likely
      FMP's demo ticker). Since ETFs are core TFSA/RRSP holdings for this platform's
      target users, this is a real coverage gap for US-listed ETFs (which route
      through FMP as ordinary non-.TO tickers) — not a hypothetical. No fallback
      exists yet; flag before this ships if ETF holdings are expected soon.
      (Canadian ETFs like ZQQ/XEI/XGD show the same 402 pattern when queried with
      a .TO suffix, but shouldn't reach this provider at all per the .TO routing
      rule — openbb-tmx/yfinance own those regardless of this gap.)
    - No bulk quote — `quote` and every other per-symbol endpoint here take exactly
      one symbol per call.
    - No news at any endpoint (HTTP 402 "Restricted Endpoint") — Finnhub is the sole
      US news source; `get_news` and `get_analyst_recommendation_trends` here raise
      NotImplementedError.
    - `get_financials`, `get_insider_trading`, and `get_peers` are also out of scope
      here — edgartools.py and finnhub.py own those per the routing rule.
    """

    def __init__(
        self, api_key: str | None = None, session: aiohttp.ClientSession | None = None
    ) -> None:
        self._api_key = api_key or os.environ.get("SP_FMP_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "SP_FMP_API_KEY is not set. Register for a free key at "
                "https://financialmodelingprep.com and set it in the environment."
            )
        self.session = session
        self._owns_session = session is None

    async def __aenter__(self) -> "FMPDataProvider":
        if self._owns_session:
            self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._owns_session and self.session:
            await self.session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None:
            self.session = aiohttp.ClientSession()
            self._owns_session = True
        return self.session

    async def _request(self, path: str, params: dict) -> list | dict | None:
        """GET {BASE_URL}/{path} with the API key attached.

        Returns parsed JSON on success. Returns None for the two "no data for this
        symbol" conditions confirmed live — HTTP 402 on a paywalled/unsupported
        symbol, or HTTP 200 with an empty list (invalid ticker, or /profile's way of
        saying the same paywalled-symbol thing) — a per-call condition, not a fatal
        error. Raises for genuine failures (401 bad key, 429, 5xx).
        """
        session = await self._get_session()
        query = {**params, "apikey": self._api_key}
        async with session.get(f"{BASE_URL}/{path}", params=query) as response:
            if response.status == 402:
                # Despite the application/json content-type header, 402 bodies are
                # plain text ("Premium Query Parameter: ..." / "Restricted Endpoint:
                # ..."), not JSON — .json() would raise here.
                body = await response.text()
                logger.warning(
                    "fmp_symbol_not_available", path=path, params=params, body=body[:200]
                )
                return None
            if response.status == 401:
                body = await response.text()
                raise RuntimeError(f"FMP API key rejected (401): {body[:200]}")
            response.raise_for_status()
            data = await response.json()
            if isinstance(data, list) and not data:
                logger.warning("fmp_empty_response", path=path, params=params)
                return None
            return data

    async def get_price_history(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        # FMP's free-tier EOD endpoint is daily-only regardless of `interval` — kept
        # in the signature for StockDataProvider compatibility with yfinance, which
        # does support intraday intervals.
        from_date = _period_to_from_date(period)
        to_date = pd.Timestamp.now().normalize().strftime("%Y-%m-%d")
        data = await self._request(
            "historical-price-eod/full", {"symbol": ticker, "from": from_date, "to": to_date}
        )
        if data is None:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").set_index("date")

    async def get_company_info(self, ticker: str) -> dict:
        data = await self._request("profile", {"symbol": ticker})
        return data[0] if data else {}

    async def get_analyst_estimates(self, ticker: str) -> dict:
        # `period` is a required param — FMP 400s without it.
        data = await self._request("analyst-estimates", {"symbol": ticker, "period": "annual"})
        return {"symbol": ticker.upper(), "estimates": data or []}

    async def get_analyst_ratings(self, ticker: str) -> dict:
        # Renamed from /rating. A current-period snapshot (letter grade + DCF/ROE/
        # ROA/D-E/P-E/P-B sub-scores) — not a historical trend.
        data = await self._request("ratings-snapshot", {"symbol": ticker})
        return data[0] if data else {}

    async def get_earnings_calendar(self, ticker: str) -> list[dict]:
        """FMP's /earnings-calendar is bulk-only (no symbol filter, renamed from
        earning_calendar) — fetch a forward window and filter client-side."""
        from_date = pd.Timestamp.now().normalize().strftime("%Y-%m-%d")
        to_date = (pd.Timestamp.now().normalize() + pd.Timedelta(days=90)).strftime("%Y-%m-%d")
        data = await self._request("earnings-calendar", {"from": from_date, "to": to_date})
        if not data:
            return []
        ticker_upper = ticker.upper()
        return [row for row in data if row.get("symbol", "").upper() == ticker_upper]

    async def get_dividend_history(self, ticker: str, from_date: str, to_date: str) -> list[dict]:
        # Renamed from historical-price-full/stock_dividend/{symbol}. Filtered
        # client-side too, in case the API's own from/to filtering is inconsistent.
        data = await self._request(
            "dividends", {"symbol": ticker, "from": from_date, "to": to_date}
        )
        if not data:
            return []
        return [row for row in data if from_date <= row.get("date", "") <= to_date]

    async def get_quote(self, ticker: str) -> dict:
        """Not on StockDataProvider — used internally for the Portfolio Optimizer's
        bulk price refresh. No bulk quote on the free tier (comma-separated `symbol`
        returns HTTP 402), so callers must loop this one ticker at a time."""
        data = await self._request("quote", {"symbol": ticker})
        return data[0] if data else {}

    async def get_ratios_ttm(self, ticker: str) -> dict:
        """Not on StockDataProvider. Cross-check only — edgartools.py stays primary
        for fundamentals per the routing rule; don't wire this into it."""
        data = await self._request("ratios-ttm", {"symbol": ticker})
        return data[0] if data else {}

    # Out of scope for FMP's free tier — see class docstring.
    async def get_financials(self, ticker: str, statement: str, period: str) -> pd.DataFrame:
        raise NotImplementedError(
            "Financial statements are edgartools.py's job — see CLAUDE.md routing rule"
        )

    async def get_insider_trading(self, ticker: str, days: int = 90) -> list[dict]:
        raise NotImplementedError("Insider trading (Form 4) is edgartools.py's job")

    async def get_peers(self, ticker: str, limit: int = 5) -> list[str]:
        raise NotImplementedError("Peers are Finnhub's job — GET /stock/peers")

    async def get_news(self, ticker: str, days: int) -> list[dict]:
        raise NotImplementedError(
            "News is paid-only on FMP's free tier (confirmed live, HTTP 402 on every "
            "news endpoint) — Finnhub is the sole US news source"
        )

    async def get_analyst_recommendation_trends(self, ticker: str) -> list[dict]:
        raise NotImplementedError(
            "FMP only provides a current-period ratings snapshot (get_analyst_ratings), "
            "not a historical trend"
        )
