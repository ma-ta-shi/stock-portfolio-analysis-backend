import asyncio
import os
import re
from datetime import datetime, timedelta

import aiohttp
import pandas as pd
import structlog

from data.degradation import FETCH_FAILED, NOT_COVERED
from data.degradation import report as report_degradation
from data.providers.base import NewsProvider

logger = structlog.get_logger(__name__)

BASE_URL = "https://finnhub.io/api/v1"

# Confirmed live against a real free-tier key (2026-07-26 — see ClickUp 86bagzcv5).

# /company-news returns at most ~250 of the NEWEST rows for the window asked, whatever its
# width (BB-023, measured live 2026-09-30: MSFT 65-106 articles a day, so a "180 day"
# request returned 2 days; KO 28 days; even a 7-day window is capped for MSFT and AAPL).
# A response at or above this size may have been cut off.
_NEWS_CAP_SUSPECT = 240
# The window width when re-asking a busy name is the number of days that fit under the cap at
# the density the first answer showed, but never narrower than this. A capped 3-day window of
# a ~100-a-day name returns its newest ~2.5 days, so the month is covered ~83% of days and
# evenly, which serves a trend better than 100% of only the newest 25 days. A quieter name
# (~10 a day) fits a ~20-day window under the cap, so its missing tail is one request.
_NEWS_WINDOW_DAYS = 3
_NEWS_WINDOW_TARGET_ROWS = 150  # per window: headroom under the ~250 cap for busier days
_NEWS_WINDOW_CONCURRENCY = 3  # well inside the free tier's 60 calls a minute

_RECOMMENDATION_RENAME = {
    "period": "period",
    "strongBuy": "strong_buy",
    "buy": "buy",
    "hold": "hold",
    "sell": "sell",
    "strongSell": "strong_sell",
}


def _reject_ca_ticker(ticker: str) -> None:
    """Confirmed live (2026-07-27, all 15 tickers in the mock watchlist's CA
    subset) that every Finnhub endpoint used here 403s on `.TO` symbols — not
    just /stock/peers. Without this guard, a misrouted `.TO` ticker (e.g. a
    future router.py bug) would silently come back as "no news today" via the
    generic 403-to-empty-list handling in `_request`, indistinguishable from a
    genuinely quiet US ticker. Failing loudly here instead makes a routing bug
    visible immediately rather than masking it as an empty result."""
    if ticker.upper().endswith(".TO"):
        raise NotImplementedError(
            f"Finnhub has no Canadian coverage (confirmed 403 on '{ticker}') — "
            "this ticker should never have reached FinnhubDataProvider"
        )


class FinnhubDataProvider(NewsProvider):
    """US news, analyst recommendation trends, and peer identification via
    Finnhub's free tier.

    Deliberately does NOT inherit StockDataProvider despite implementing
    get_peers (which lives on that ABC) — Finnhub is a supplemental source
    for one StockDataProvider method, not a drop-in provider for the other
    eight. Stubbing those with NotImplementedError just to satisfy the ABC
    would be noise with no caller. FMPDataProvider/EdgarToolsDataProvider
    earn the full StockDataProvider inheritance because they ARE pluggable
    US-stock backends; this class isn't.

    More load-bearing than originally planned: fmp.py's news endpoints are
    fully paywalled on the free tier (ClickUp 86bagzcu5), so this is the
    SOLE US news source, not a supplement.

    Confirmed gap — no sentiment scoring on free tier: `/news-sentiment`
    returns HTTP 403, and the working `/company-news` endpoint has no
    sentiment field at all. Do not expect a `sentiment_score` key anywhere
    in this file's output; there isn't one. Gap 2 in
    financial-data-api-research.md (raw text to the Sentiment Analyst LLM,
    VADER as a local quantitative backup) now applies to US news too, not
    just Canadian.

    Confirmed NOT paywalled per-ticker like FMP: tested `/company-news`
    against every ticker that 402'd on FMP's free tier (GME, BB, ACB, CLF,
    X, UUUU, PLUG, FCEL, GOOG, BRK.B, GSAT) — all returned real data (X came
    back an empty list — no recent news, not a block).
    """

    def __init__(
        self, api_key: str | None = None, session: aiohttp.ClientSession | None = None
    ) -> None:
        self._api_key = api_key or os.environ.get("SP_FINNHUB_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "SP_FINNHUB_API_KEY is not set. Register for a free key at "
                "https://finnhub.io and set it in the environment."
            )
        self.session = session
        self._owns_session = session is None

    async def __aenter__(self) -> "FinnhubDataProvider":
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

    async def _request(self, path: str, params: dict, _retried: bool = False) -> list | dict | None:
        """GET {BASE_URL}/{path} with the API key attached.

        Unlike fmp.py, an empty list here is a genuine "no data" answer
        (e.g. no recent news for a quiet ticker), not a paywall signal —
        confirmed live, so it's returned as-is rather than coerced to None.
        HTTP 403 ("no access to this resource" — e.g. a symbol/endpoint
        combination outside free-tier scope) is treated the same way FMP
        treats 402: logged, returns None, not a fatal error. HTTP 429 gets
        one basic retry after a short sleep — confirmed no throttling
        across ~20 rapid test calls at the documented 60/min limit, so this
        is a safety net, not routine call budgeting. Raises for genuine
        failures (401 bad key, repeated 429, 5xx).
        """
        session = await self._get_session()
        query = {**params, "token": self._api_key}
        async with session.get(f"{BASE_URL}/{path}", params=query) as response:
            if response.status == 401:
                body = await response.text()
                raise RuntimeError(f"Finnhub API key rejected (401): {body[:200]}")
            if response.status == 403:
                body = await response.text()
                logger.warning("finnhub_forbidden", path=path, params=params, body=body[:200])
                report_degradation(
                    "finnhub", path, NOT_COVERED, f"HTTP 403: {body[:200]}", context={"path": path, "symbol": params.get("symbol")}
                )
                return None
            if response.status == 429:
                if _retried:
                    response.raise_for_status()
                logger.warning("finnhub_rate_limited_retrying", path=path, params=params)
                await asyncio.sleep(1)
                return await self._request(path, params, _retried=True)
            response.raise_for_status()
            return await response.json()

    async def _news_window(self, ticker: str, start, end) -> list[dict]:
        data = await self._request(
            "company-news",
            {"symbol": ticker, "from": start.isoformat(), "to": end.isoformat()},
        )
        return data or []

    async def _news_windows(self, ticker: str, from_date, to_date, width_days: int) -> list[dict]:
        """Raw rows for every `width_days` window from `from_date` to `to_date` (inclusive
        on both ends, as Finnhub treats from/to). A window that fails is skipped and
        reported once; the rest still count."""
        windows = []
        start = from_date
        while start <= to_date:
            end = min(start + timedelta(days=width_days - 1), to_date)
            windows.append((start, end))
            start = end + timedelta(days=1)
        semaphore = asyncio.Semaphore(_NEWS_WINDOW_CONCURRENCY)

        async def bounded(window):
            async with semaphore:
                return await self._news_window(ticker, *window)

        results = await asyncio.gather(*(bounded(w) for w in windows), return_exceptions=True)
        rows: list[dict] = []
        failed = 0
        for window, result in zip(windows, results, strict=True):
            if isinstance(result, Exception):
                failed += 1
                logger.warning("finnhub_news_window_failed", ticker=ticker, window=str(window), error=str(result))
                continue
            rows.extend(result)
        if failed:
            report_degradation(
                "finnhub",
                "get_news_windows",
                FETCH_FAILED,
                f"{failed} of {len(windows)} news windows failed",
                context={"symbol": ticker},
            )
        return rows

    async def get_news(self, ticker: str, days: int, *, thorough: bool = False) -> list[dict]:
        """Confirmed path is /company-news, not /stock/company-news.

        One request, which Finnhub cuts off at the ~250 newest rows. `thorough=True`
        (the main stock's own news, never a peer's two headlines) notices when the
        answer looks cut off and re-asks, in 3-day windows, only for the part of the
        span the first answer did NOT reach (it holds the newest ~250 rows, so everything
        older than its oldest row is missing), so a busy ticker's month is covered instead
        of its last few days (BB-023). A window that came back complete costs exactly one
        request, as before; a mid-volume ticker whose first answer already reached nearly
        the start of the window costs two."""
        _reject_ca_ticker(ticker)
        to_date = datetime.now().date()
        from_date = to_date - timedelta(days=days)
        data = await self._request(
            "company-news",
            {"symbol": ticker, "from": from_date.isoformat(), "to": to_date.isoformat()},
        )
        if not data:
            return []
        if thorough and len(data) >= _NEWS_CAP_SUSPECT:
            reached = [
                datetime.fromtimestamp(item["datetime"]).date()
                for item in data
                if item.get("datetime") is not None
            ]
            # Rows run newest first and the cut falls on the old end, so the span still
            # missing is from_date up to the oldest day the probe reached (inclusive: that
            # day was only partly returned). No dates at all: re-ask the whole span.
            missing_to = min(reached) if reached else to_date
            if missing_to >= from_date:
                days_reached = max(1, (to_date - missing_to).days + 1)
                rows_per_day = len(data) / days_reached
                width = max(_NEWS_WINDOW_DAYS, int(_NEWS_WINDOW_TARGET_ROWS / rows_per_day))
                data = [*data, *await self._news_windows(ticker, from_date, missing_to, width)]
        # Belt-and-suspenders cutoff on `datetime`, same pattern as yfinance.py's
        # get_news — the from/to query params should already cover this, but
        # filtering client-side too costs nothing and guards against any
        # server-side date-boundary looseness.
        cutoff = (datetime.now() - timedelta(days=days)).timestamp()
        articles = []
        seen_urls: set[str] = set()
        seen_headlines: set[str] = set()
        for item in data:
            published_at = item.get("datetime")
            if published_at is None or published_at < cutoff:
                continue
            if thorough:
                # The probe and the windows overlap, and the same story is syndicated
                # under several URLs: drop repeats by URL, then by normalised headline.
                url = item.get("url") or ""
                headline_key = re.sub(r"[^a-z0-9]+", " ", (item.get("headline") or "").lower()).strip()
                if (url and url in seen_urls) or (headline_key and headline_key in seen_headlines):
                    continue
                seen_urls.add(url)
                seen_headlines.add(headline_key)
            articles.append(
                {
                    "headline": item.get("headline", "N/A"),
                    "summary": item.get("summary", ""),
                    "source": item.get("source", "N/A"),
                    "url": item.get("url", "N/A"),
                    "published_at": datetime.fromtimestamp(published_at).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    # No `sentiment_score` — free tier has none, see class docstring.
                }
            )
        return articles

    async def get_analyst_recommendation_trends(self, ticker: str) -> list[dict]:
        """Real monthly buy/hold/sell/strongBuy/strongSell trend across
        several periods — unlike FMP's single-period ratings-snapshot."""
        _reject_ca_ticker(ticker)
        data = await self._request("stock/recommendation", {"symbol": ticker})
        if not data:
            return []
        return [
            {
                _RECOMMENDATION_RENAME[key]: value
                for key, value in row.items()
                if key in _RECOMMENDATION_RENAME
            }
            for row in data
        ]

    async def get_peers(self, ticker: str, limit: int = 5) -> list[str]:
        """US stocks only. Canadian peers come from the static
        data/peers.json (Gap 1) — confirmed live that Finnhub 403s on .TO
        symbols here, so this is checked up front rather than surfaced as a
        confusing "forbidden" result."""
        if ticker.upper().endswith(".TO"):
            raise NotImplementedError(
                "Finnhub peers are US-only — Canadian peers come from data/peers.json (Gap 1)"
            )
        data = await self._request("stock/peers", {"symbol": ticker})
        if not data:
            return []
        ticker_upper = ticker.upper()
        peers = [symbol for symbol in data if symbol.upper() != ticker_upper]
        return peers[:limit]

    async def get_general_news(self, category: str = "general") -> list[dict]:
        """Not on NewsProvider — no symbol needed. Used by the Macro
        Economist and the home page, not per-ticker Sentiment Analyst runs."""
        data = await self._request("news", {"category": category})
        return data or []

    async def get_metric(self, ticker: str) -> dict:
        """Not on StockDataProvider. Supplemental cross-check only — 52-week
        high/low, beta, and other precomputed metrics — same role FMP's
        get_ratios_ttm plays. Returns the flat "metric" dict, not the
        wrapper envelope Finnhub sends it in."""
        data = await self._request("stock/metric", {"symbol": ticker, "metric": "all"})
        if not data:
            return {}
        return data.get("metric", {})

    async def get_earnings_calendar(
        self, ticker: str | None = None, days_ahead: int = 90
    ) -> list[dict]:
        """Not on the ABC — FMP already covers this; confirmed available
        here too as a fallback, and the only working source in this
        method's US chain for mid/small caps (FMP's free tier omits them
        from its bulk feed — 86bbpgrjv recon: FMP `[]` for DDD/CROX).

        Must pass `symbol` (86bbpgrjv): without it Finnhub returns a bulk
        feed capped at ~1500 rows and the requested ticker is simply not in
        it, so the old client-side filter matched nothing for every ticker
        tried. With `symbol` + `from`/`to` it returns that ticker's next
        event(s) directly. The client-side filter stays as a cheap guard."""
        from_date = pd.Timestamp.now().normalize().date()
        to_date = from_date + timedelta(days=days_ahead)
        params = {"from": from_date.isoformat(), "to": to_date.isoformat()}
        if ticker is not None:
            params["symbol"] = ticker
        data = await self._request("calendar/earnings", params)
        rows = (data or {}).get("earningsCalendar", [])
        if ticker is None:
            return rows
        ticker_upper = ticker.upper()
        return [row for row in rows if row.get("symbol", "").upper() == ticker_upper]
