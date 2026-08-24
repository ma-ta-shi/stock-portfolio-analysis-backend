"""OpenBB TMX (Toronto Stock Exchange) provider for Canadian stock data.

OpenBB is a free, open-source market data library. This provider leverages
openbb.equity.fundamental and openbb.equity.price for Canadian stocks.

Coverage: Canadian stocks (tickers ending in .TO)
Limitations: Free tier has quotas and some endpoints require paid accounts

get_analyst_estimates/get_insider_trading/get_earnings_calendar/get_news
were previously unverified NotImplementedError stubs (their comments
claimed "not supported by TMX" with no live check). Live-verified 2026-08-04
(ClickUp 86bb7j0kh) against real `openbb` calls that all four are actually
real, working `tmx`-provider endpoints — CLAUDE.md's Data Providers section
already listed "calendar, news" as openbb-tmx coverage; the code just never
matched. Only get_peers is a genuine gap (obb.equity.compare.peers has no
tmx provider) — that one stays NotImplementedError.

Known gap, documented not fixed (86bb7j0kh): get_price_history has ZERO
real TSXV (.V) coverage — live-tested 6 real TSXV tickers (including ones
previously believed covered from unrelated SEC-cross-listing research),
every one raises openbb's own EmptyDataError. This looks like a genuine
upstream data-source limitation, not something fixable in this file.
router.py's CA_CHAINS already lists yfinance as the get_price_history
fallback, which does cover TSXV — no code change made here since the
existing fallback already handles it correctly (once the router's own
exception-logging crash — see router.py — doesn't get in the way first).
"""

import asyncio
from datetime import datetime, timedelta

import pandas as pd
from openbb import obb
from data.providers.base import (
    StockDataProvider,
    NewsProvider,
    NormalizedCompanyInfo,
    NormalizedDividendRecord,
)


class OpenBBTMXProvider(StockDataProvider, NewsProvider):
    """OpenBB (TMX extension) provider for Canadian equities.

    Covers price history, fundamentals, company info, dividends, analyst
    estimates, insider trading, earnings calendar, and news via the
    `openbb-tmx` data provider. Only peers is NOT supported by TMX
    (no tmx provider on obb.equity.compare.peers) — use the router's
    static peers_json fallback for that instead.
    """

    PROVIDER = "tmx"

    # ---------- StockDataProvider ----------

    async def get_price_history(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        result = await asyncio.to_thread(
            obb.equity.price.historical,
            symbol=ticker,
            interval=interval,
            provider=self.PROVIDER,
        )
        return result.to_df()

    async def get_financials(self, ticker: str, statement: str, period: str) -> pd.DataFrame:
        # statement: "income" | "balance" | "cash"
        fetcher = {
            "income": obb.equity.fundamental.income,
            "balance": obb.equity.fundamental.balance,
            "cash": obb.equity.fundamental.cash,
        }.get(statement)

        if fetcher is None:
            raise ValueError(f"Unknown statement type: {statement}")

        result = await asyncio.to_thread(
            fetcher, symbol=ticker, period=period, provider=self.PROVIDER
        )
        return result.to_df()

    async def get_company_info(self, ticker: str) -> dict:
        """Fixed 2026-08-04 (86bb7j0kh) — real bug, confirmed live and
        upstream, in openbb-tmx's own equity_profile.py: its internal
        `symbol_to_index[d["symbol"]]` sort keys the lookup dict off the
        REQUESTED symbol string, but TMX's returned data always uses its
        own bare/dotted convention (e.g. "RY", "RCI.B") — a mismatched
        request format (this codebase's "RY.TO"/"RCI-B") makes the lookup
        raise KeyError *during the fetch itself*, before any Python code
        here runs (confirmed: reading `.results` instead of `.to_df()`
        does NOT help, unlike get_news/get_earnings_calendar's bugs — this
        one happens too early for that workaround). The only real fix is
        requesting the exact format TMX itself returns: strip the .TO/.V
        suffix and swap hyphens for periods, same transform already used
        in get_earnings_calendar — confirmed live for both a plain ticker
        (RY) and a multi-class/trust-unit one (RCI.B, CAR.UN).

        Maps onto NormalizedCompanyInfo (86bbb001k). Two real, confirmed
        gaps (live-verified against RY 2026-08-07, not guessed): OpenBB's
        standard EquityInfo model has no market_cap or currency field at
        all (TMX's own provider extension only adds email/issue_type/
        shares_outstanding/shares_escrow/shares_total/dividend_frequency —
        still neither), and hq_country/inc_country are both None even for
        a plain, unambiguous TSX primary listing like RY — not just an
        edge case. currency is safe to hardcode "CAD" (this codebase's own
        _is_canadian_stock() already treats TSX/TSXV-listed == CAD as a
        given); market_cap and country are left as their empty/None
        sentinels — real, disclosed data gaps, not derived from a second
        API call. industry uses industry_category ("Banking") over the
        finer industry_group ("Diversified Banks") — closer to the
        single-level granularity FMP/yfinance's own `industry` field
        represents."""
        bare_symbol = ticker.removesuffix(".TO").removesuffix(".V").replace("-", ".")
        result = await asyncio.to_thread(
            obb.equity.profile, symbol=bare_symbol, provider=self.PROVIDER
        )
        if not result.results:
            return {}
        raw = result.results[0].model_dump()
        return NormalizedCompanyInfo(
            name=raw.get("name") or "",
            sector=raw.get("sector") or "",
            industry=raw.get("industry_category") or "",
            market_cap=None,
            currency="CAD",
            country=raw.get("hq_country") or raw.get("inc_country") or "",
            primary_exchange=raw.get("stock_exchange") or "",
        )

    async def get_analyst_estimates(self, ticker: str) -> dict:
        """Live-verified 2026-08-04 (86bb7j0kh): one row per ticker —
        target_high/low/mean/consensus, buy/sell/hold_ratings,
        consensus_action. Covers both estimates and ratings in one call
        (get_analyst_ratings below reuses this rather than duplicating it).

        Must check `result.results` before calling `.to_df()` — confirmed
        live (MKO.V, zero analyst coverage) that `.to_df()` itself raises
        OpenBBError("Results not found") on a genuinely empty result rather
        than returning an empty DataFrame; `df.empty` is never reached."""
        result = await asyncio.to_thread(
            obb.equity.estimates.consensus, symbol=ticker, provider=self.PROVIDER
        )
        if not result.results:
            return {}
        df = result.to_df()
        return df.iloc[0].to_dict() if not df.empty else {}

    async def get_analyst_ratings(self, ticker: str) -> dict:
        """Deliberately returns the identical superset dict as
        get_analyst_estimates, not a narrower ratings-only shape — TMX has
        one consensus snapshot with both target-price and rating-breakdown
        fields, not two separate endpoints like FMP's analyst-estimates/
        ratings-snapshot split. No real caller consumes this yet
        (research_sources.py is still a stub), so this wasn't narrowed to
        guess at a shape nothing needs — revisit if a future caller
        specifically wants a ratings-only dict without target-price
        fields mixed in."""
        return await self.get_analyst_estimates(ticker)

    async def get_insider_trading(self, ticker: str, days: int = 90) -> list[dict]:
        """Live-verified 2026-08-04 (86bb7j0kh): real data, but a quarterly
        aggregate rollup per owner (period="three_months" etc.) — TMX never
        populates the shared schema's transaction_date/filing_date fields,
        unlike edgartools' per-transaction Form 4 data. `days` is NOT
        honored here — there's no date to filter against, unlike the
        method's own signature implies (matching edgartools' `days`-
        filtered contract). Every record is tagged `days_filter_applied:
        False` so a future caller can tell from the data itself, not just
        this docstring, that `days` was silently ignored rather than
        actually applied."""
        result = await asyncio.to_thread(
            obb.equity.ownership.insider_trading, symbol=ticker, provider=self.PROVIDER
        )
        if not result.results:
            return []
        df = result.to_df()
        if df.empty:
            return []
        records = df.to_dict("records")
        for record in records:
            record["days_filter_applied"] = False
        return records

    async def get_peers(self, ticker: str, limit: int = 5) -> list[str]:
        # Confirmed live 2026-08-04: obb.equity.compare.peers genuinely has
        # no tmx provider (unlike the other four methods here) — use
        # router.py's static peers_json fallback instead.
        raise NotImplementedError("Peer comparison is not available via openbb-tmx.")

    async def get_earnings_calendar(self, ticker: str) -> list[dict]:
        """Live-verified 2026-08-04 (86bb7j0kh): real data, but bulk (no
        symbol filter on the TMX endpoint itself) — same shape as fmp.py's
        get_earnings_calendar, filtered client-side. TMX's own symbol
        format uses periods for sub-classes (e.g. "DIR.UN", "RCI.B") where
        this codebase uses hyphens + .TO suffix (e.g. "DIR-UN.TO") — strip
        the suffix and swap hyphens back to periods to match."""
        result = await asyncio.to_thread(obb.equity.calendar.earnings, provider=self.PROVIDER)
        if not result.results:
            return []
        df = result.to_df()
        if df.empty:
            return []
        bare_symbol = ticker.removesuffix(".TO").removesuffix(".V").replace("-", ".")
        return df[df["symbol"] == bare_symbol].to_dict("records")

    async def get_dividend_history(
        self, ticker: str, from_date: str, to_date: str
    ) -> list[NormalizedDividendRecord]:
        """Fixed 2026-08-04 (86bb7j0kh) — real bug, confirmed live: `.to_df()`
        does NOT set a DatetimeIndex here despite index="date" being the
        default. TMX's dividend model's date field is named
        `ex_dividend_date`, not `date`, so the index falls back to a plain
        RangeIndex and the old `df.index >= from_date` comparison raised
        TypeError for every CA ticker. Filter on the real column instead.

        Maps onto NormalizedDividendRecord (86bbb001k) — confirmed live
        2026-08-07 that .to_df() already exposes clean, standardized
        columns (ex_dividend_date, amount, payment_date), unlike
        get_company_info()'s raw GraphQL keys — just a rename, no derived/
        missing fields here."""
        result = await asyncio.to_thread(
            obb.equity.fundamental.dividends,
            symbol=ticker,
            provider=self.PROVIDER,
        )
        if not result.results:
            return []
        df = result.to_df()
        if df.empty or "ex_dividend_date" not in df.columns:
            return []
        dates = pd.to_datetime(df["ex_dividend_date"])
        mask = (dates >= from_date) & (dates <= to_date)
        return [
            NormalizedDividendRecord(
                ex_date=str(row["ex_dividend_date"]),
                payment_date=str(row["payment_date"]) if pd.notna(row.get("payment_date")) else None,
                amount_per_share=float(row["amount"]),
            )
            for row in df.loc[mask].to_dict("records")
        ]

    # ---------- NewsProvider ----------

    async def get_news(self, ticker: str, days: int) -> list[dict]:
        """Live-verified 2026-08-04 (86bb7j0kh): real articles, with a real
        `date` field on each result object — but NOT surfaced by
        `.to_df()`'s default columns, so this reads `.results` directly
        (pydantic models) rather than going through the DataFrame.

        `limit` is required — confirmed live TMX returns zero results
        without it, `start_date` is silently ignored (accepted by the
        shared OpenBB signature but not honored by the tmx provider) — so
        `days` filtering below is client-side over a fixed-size fetch, not
        a true date-bounded query. limit=50 (bumped from an initial 20,
        confirmed live: RY.TO alone had >20 articles within a plausible
        30-90 day window) reduces but doesn't eliminate the truncation
        risk for unusually newsy tickers or long `days` windows — no
        server-side fix available via this endpoint."""
        result = await asyncio.to_thread(
            obb.news.company, symbol=ticker, provider=self.PROVIDER, limit=50
        )
        if not result.results:
            return []
        cutoff = datetime.now() - timedelta(days=days)
        articles = []
        for item in result.results:
            article_date = item.date.replace(tzinfo=None) if item.date else None
            if article_date is not None and article_date < cutoff:
                continue
            articles.append(item.model_dump())
        return articles

    async def get_analyst_recommendation_trends(self, ticker: str) -> list[dict]:
        # Not supported by TMX provider.
        raise NotImplementedError("Recommendation trends are not available via openbb-tmx.")
