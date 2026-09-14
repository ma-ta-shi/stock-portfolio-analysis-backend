import os

import aiohttp
import pandas as pd
import structlog

from data.providers.base import (
    NewsProvider,
    NormalizedAnalystEstimates,
    NormalizedCompanyInfo,
    NormalizedDividendRecord,
    NormalizedQuote,
    StockDataProvider,
    period_to_from_date,
)

from dotenv import load_dotenv

load_dotenv()

logger = structlog.get_logger(__name__)

BASE_URL = "https://financialmodelingprep.com/stable"

# Verified live against a real free-tier key (2026-07-26 — see ClickUp 86bagzcu5):
# FMP has fully migrated off /api/v3/... onto /stable/... (query-param style).
# Do not add /v3 paths here.

_EARNINGS_SURPRISE_LOOKBACK = 4  # matches yfinance's own hard 4-row limit for the same field


class FMPDataProvider(StockDataProvider, NewsProvider):
    """US-stock prices, profiles, and analyst data via FMP's free tier.

    Free-tier gaps confirmed live (2026-07-26/27, ClickUp 86bagzcu5; re-verified
    and revised 2026-08-04, ClickUp 86bb7j0kh) — not assumed:
    - The paywall boundary is per-ticker and drifts over time, not a fixed,
      exhaustive list — treat every example below as "confirmed paywalled/open
      as of its own date," not a permanent classification. Confirmed live
      2026-08-04: BRK.B is still paywalled on get_company_info, but GOOG's
      /profile now returns real data directly (no longer paywalled there,
      despite still being a dual-class ticker) — FMP's own tier policy
      evidently changed since the original 2026-07-26 finding. Also newly
      confirmed 2026-08-04: FTNT and ALB (ordinary US large-caps, not ETFs or
      dual-class shares) 402 on get_price_history — the paywall is broader and
      more arbitrary than "ETFs + dual-class shares," the original two
      categories found.
    - ETFs/index funds are paywalled broadly — found by testing QQQ, GLD, VOO,
      IWM, XLK, ARKK, DIA: all HTTP 402 on price history/quote/dividends, only
      `/profile` works. SPY is the sole free exception (likely FMP's demo
      ticker). Since ETFs are core TFSA/RRSP holdings for this platform's
      target users, this was a real coverage gap for US-listed ETFs — router.py
      (`US_CHAINS`) now falls back to yfinance for get_price_history/
      get_dividend_history/get_quote/get_company_info, confirmed live working
      2026-08-04; this docstring previously said "no fallback exists yet,"
      which is now stale.
      (Canadian ETFs like ZQQ/XEI/XGD show the same 402 pattern when queried with
      a .TO suffix, but shouldn't reach this provider at all per the .TO routing
      rule — openbb-tmx/yfinance own those regardless of this gap.)
    - `_request()` treats a 402 as "no data for this ticker" rather than a
      fatal error, on every endpoint here — dual-class, ETF, or otherwise.
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
        from_date = period_to_from_date(period)
        to_date = pd.Timestamp.now().normalize().strftime("%Y-%m-%d")
        data = await self._request(
            "historical-price-eod/full", {"symbol": ticker, "from": from_date, "to": to_date}
        )
        if data is None:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").set_index("date")

    async def get_company_info(self, ticker: str) -> NormalizedCompanyInfo:
        """Maps FMP's raw /profile response onto NormalizedCompanyInfo
        (86bbb001k) — real FMP field names confirmed via openbb_fmp's own
        /profile wrapper (openbb_fmp/models/equity_profile.py's alias dict)
        and cross-checked against a live-observed field in
        tests/live/test_provider_completeness.py. The seven rendered fields
        are present directly, no derivation needed; asset_type (86bbpk6uf)
        is derived from FMP's isEtf/isFund booleans."""
        data = await self._request("profile", {"symbol": ticker})
        if not data:
            return {}
        raw = data[0]
        # asset_type (86bbpk6uf part 1): FMP's /profile carries isEtf/isFund
        # booleans. isFund covers both mutual funds and closed-end funds
        # (FMP doesn't distinguish them) -> "other". A US preferred has both
        # false -> "equity" (a documented gap, rare for this watchlist).
        asset_type = "etf" if raw.get("isEtf") else ("other" if raw.get("isFund") else "equity")
        # .get(key) or "" (not .get(key, "")) — FMP can return an explicit
        # null for these fields, not just omit the key; .get(key, "") only
        # covers the omitted case and would silently store None in a str field.
        return NormalizedCompanyInfo(
            name=raw.get("companyName") or "",
            sector=raw.get("sector") or "",
            industry=raw.get("industry") or "",
            market_cap=raw.get("marketCap"),
            currency=raw.get("currency") or "",
            country=raw.get("country") or "",
            primary_exchange=raw.get("exchange") or "",
            asset_type=asset_type,
        )

    async def get_analyst_estimates(self, ticker: str) -> NormalizedAnalystEstimates:
        """Maps FMP's raw /analyst-estimates response onto
        NormalizedAnalystEstimates (86bbdu04a). `period=annual` rows come
        back sorted descending by fiscal-year-end date (confirmed live:
        furthest-future year first) — estimates[0] is NOT the nearest
        forecast, it's the furthest one. Filter to future rows and take
        the soonest fiscal-year-end.

        Bare {} on no data, not {"forward_eps": None} — see
        NormalizedAnalystEstimates's docstring in base.py for why."""
        # `period` is a required param — FMP 400s without it.
        data = await self._request("analyst-estimates", {"symbol": ticker, "period": "annual"})
        if not data:
            return {}
        today = pd.Timestamp.now().normalize()
        future_rows = [row for row in data if row.get("date") and pd.Timestamp(row["date"]) > today]
        if not future_rows:
            return {}
        # date only picks the row here, it isn't returned
        nearest = min(future_rows, key=lambda row: row["date"])
        forward_eps = nearest.get("epsAvg")
        return {"forward_eps": forward_eps} if forward_eps is not None else {}

    async def get_analyst_ratings(self, ticker: str) -> dict:
        # Renamed from /rating. A current-period snapshot (letter grade + DCF/ROE/
        # ROA/D-E/P-E/P-B sub-scores) — not a historical trend.
        data = await self._request("ratings-snapshot", {"symbol": ticker})
        return data[0] if data else {}

    async def get_earnings_surprises(self, ticker: str) -> list[dict]:
        """FMP's /earnings (not /earnings-surprises, confirmed live 404, nor
        /earnings-surprises-bulk, confirmed live paywalled) is symbol-filtered
        and includes both forward calendar rows (epsActual is None) and real
        historical actual-vs-estimate rows — filter to the realized ones."""
        data = await self._request("earnings", {"symbol": ticker})
        realized = [row for row in (data or []) if row.get("epsActual") is not None]
        return [
            {
                "period_end": row.get("date"),
                "eps_actual": row.get("epsActual"),
                "eps_estimated": row.get("epsEstimated"),
                "eps_surprise_pct": (
                    (row["epsActual"] - row["epsEstimated"]) / abs(row["epsEstimated"]) * 100
                    if row.get("epsEstimated")
                    else None
                ),
                "revenue_actual": row.get("revenueActual"),
                "revenue_estimated": row.get("revenueEstimated"),
            }
            # realized is already newest-first, confirmed live
            for row in realized[:_EARNINGS_SURPRISE_LOOKBACK]
        ]

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

    async def get_dividend_history(
        self, ticker: str, from_date: str, to_date: str
    ) -> list[NormalizedDividendRecord]:
        """Maps FMP's raw /dividends response onto NormalizedDividendRecord
        (86bbb001k) — real FMP field names confirmed live 2026-08-07 against
        AAPL: date, dividend, paymentDate. Renamed from
        historical-price-full/stock_dividend/{symbol}. Filtered client-side
        too, in case the API's own from/to filtering is inconsistent."""
        data = await self._request(
            "dividends", {"symbol": ticker, "from": from_date, "to": to_date}
        )
        if not data:
            return []
        # Real bug caught in a later sweep: .get(key, default) only guards a
        # missing key, not an explicit null in the API response. Worse than
        # the silent-None-in-str-field pattern found elsewhere — the old
        # filter line would have crashed with a TypeError (comparing None
        # to a string with <=) instead of just producing bad data.
        return [
            NormalizedDividendRecord(
                ex_date=row.get("date") or "",
                payment_date=row.get("paymentDate"),
                amount_per_share=row.get("dividend") or 0.0,
            )
            for row in data
            if from_date <= (row.get("date") or "") <= to_date
        ]

    async def get_quote(self, ticker: str) -> NormalizedQuote:
        """Not on StockDataProvider — used internally for the Portfolio Optimizer's
        bulk price refresh, and (86bbb17pw) as the price_info source for
        fundamentals.py. No bulk quote on the free tier (comma-separated `symbol`
        returns HTTP 402), so callers must loop this one ticker at a time.

        Maps onto NormalizedQuote — real FMP raw field names confirmed live
        2026-08-10 against AAPL: price, marketCap, yearHigh, yearLow. FMP's
        /quote has no currency field at all — real, disclosed gap; FMP is
        US-only in this codebase's routing (CLAUDE.md), so hardcoding "USD"
        here is the same safe, established pattern as openbb_tmx hardcoding
        "CAD" in get_company_info() (86bbb001k)."""
        data = await self._request("quote", {"symbol": ticker})
        if not data:
            return {}
        raw = data[0]
        price = raw.get("price")
        # Real bug caught on review: .get("price", 0.0) only guards a missing
        # key, not an explicit null — and would have silently fabricated a
        # $0.00 quote instead of signaling "no real price," inconsistent
        # with yfinance.get_quote()'s own "no valid price -> empty dict"
        # behavior below.
        if price is None:
            return {}
        return NormalizedQuote(
            current_price=price,
            market_cap=raw.get("marketCap"),
            currency="USD",
            high_52w=raw.get("yearHigh"),
            low_52w=raw.get("yearLow"),
        )

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
