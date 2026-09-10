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

Updated 86bbdu04a: get_analyst_estimates specifically no longer calls
obb.equity.estimates.consensus at all — that endpoint (still real and
working) has no forward-EPS field of any kind, which is the only thing
this method's normalized contract needs, so there's nothing there worth
fetching. The real consensus call moved to get_analyst_ratings instead,
which still needs it for price-target/rating-breakdown fields.

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
import structlog
from openbb import obb
from openbb_core.app.model.abstract.error import OpenBBError
from data.providers.base import (
    StockDataProvider,
    NewsProvider,
    NormalizedAnalystEstimates,
    NormalizedCompanyInfo,
    NormalizedDividendRecord,
    period_to_from_date,
)

logger = structlog.get_logger(__name__)


class OpenBBTMXProvider(StockDataProvider, NewsProvider):
    """OpenBB (TMX extension) provider for Canadian equities.

    Covers price history, fundamentals, company info, dividends, analyst
    ratings, insider trading, earnings calendar, and news via the
    `openbb-tmx` data provider. Two real gaps: peers (no tmx provider on
    obb.equity.compare.peers — use the router's static peers_json fallback)
    and analyst *estimates* specifically (TMX's consensus endpoint has no
    forward-EPS field — get_analyst_estimates() always returns {} without
    calling the API; get_analyst_ratings() owns the real consensus call).
    """

    PROVIDER = "tmx"

    # ---------- StockDataProvider ----------

    async def get_price_history(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        # `start_date` (not `period`) — the tmx provider ignores `period` and
        # otherwise caps at ~250 bars (~1y) regardless, which silently starved
        # weekly_trend and the 3yr drawdown (86bbq7dkv). Confirmed live: with
        # start_date, RY.TO returns the full requested span (250 -> 753 bars at 3y).
        result = await asyncio.to_thread(
            obb.equity.price.historical,
            symbol=ticker,
            interval=interval,
            provider=self.PROVIDER,
            start_date=period_to_from_date(period),
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

    async def get_analyst_estimates(self, ticker: str) -> NormalizedAnalystEstimates:
        """TMX's consensus endpoint (obb.equity.estimates.consensus) has
        no forward-EPS field of any kind — confirmed live 2026-08-04
        (86bb7j0kh) and re-confirmed reviewing 86bbdu04a. Nothing to fetch,
        so this doesn't call the API at all — get_analyst_ratings() below
        is the one that still needs the real call, for its price-target/
        rating fields. Bare {} on no data — see NormalizedAnalystEstimates's
        docstring in base.py for why."""
        return {}

    async def get_analyst_ratings(self, ticker: str) -> dict:
        """Real consensus fetch, previously delegated-to via
        get_analyst_estimates() (TMX has one consensus snapshot with both
        target-price and rating-breakdown fields, not two separate
        endpoints like FMP's split) — moved here directly (86bbdu04a),
        since get_analyst_estimates() no longer has any use for the raw
        row itself. No real caller consumes this yet (research_sources.py
        is still a stub), so the shape stays the raw OpenBB row, not
        narrowed to a guessed ratings-only shape.

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

    async def get_earnings_surprises(self, ticker: str) -> list[dict]:
        raise NotImplementedError("Earnings-surprise history is not available via openbb-tmx.")

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
                payment_date=str(row["payment_date"])
                if pd.notna(row.get("payment_date"))
                else None,
                amount_per_share=float(row["amount"]),
            )
            for row in df.loc[mask].to_dict("records")
        ]

    # ---------- Not on any ABC — new capability (86bbpggr5) ----------

    _MATERIAL_FILING_TYPES = frozenset(
        {
            "Annual information form",
            "Management information circular",
            "MD&A",
            "Interim financial statements/report",
            "Audited annual financial statements",
            "Annual report",
            "Material change report",
        }
    )

    _FALLBACK_LOOKBACK_DAYS = 400

    async def get_filings(self, ticker: str, limit: int = 20) -> list[dict]:
        """New capability (86bbpggr5) — not on StockDataProvider/NewsProvider,
        same pattern as get_quote/get_ratios_ttm in router.py. No US
        equivalent; router.py's US_CHAINS carries an explicit [] entry.

        Distinct from edgartools.py's Company.get_filings(form=...,
        amendments=...) despite the shared method name — that's an
        edgartools object method for SEC EDGAR form-filtered filings
        (used by ca_crosslisting.py/edgartools.py); this is TMX's own CA
        regulatory-filings feed. No behavioral overlap.

        Materiality filter: TMX's report_type vocabulary is fixed/bounded
        (confirmed live), heavily dominated by prospectus/consent-letter/
        AGM-administrative noise. Filtered via an explicit KEEP allowlist
        of exact strings, not substring matching — the keep and drop
        vocabularies share words ("Material change report" vs. "Other
        material contract(s)"), so no single safe keyword exists.
        "Other material contract(s)" is excluded (a disclosure-threshold
        legal term, not analytical content; episodic). "News release" is
        excluded (already covered by get_news(); would let PR cadence,
        not regulatory cadence, drive latest_filing_age_days in 86ban0x1u).
        An unseen report_type is excluded by default — a visible "no
        material filing this window" gap, not a silent false positive.

        Two-tier fetch: TMX's default (no date params) window was observed
        live as ~16 weeks (not a documented API guarantee, just what one
        live pull returned) — cheap, and normally sufficient since MD&A/
        interim financials recur ~quarterly regardless of the exact window
        size. AIF/circular/material change report are annual/event-driven
        (confirmed: RY's AIF appeared exactly 3x across 3.75 years), so
        only retry with an explicit ~400-day start_date if tier 1 finds
        zero material rows — this trigger only depends on "zero material
        rows found," not on knowing the exact tier-1 window size.

        `limit` is applied client-side only — confirmed live the tmx
        provider silently ignores an API-level limit= (swallowed into
        **kwargs, no effect on row count).
        """
        bare_symbol = ticker.removesuffix(".TO").removesuffix(".V").replace("-", ".")

        async def _fetch(**kwargs) -> pd.DataFrame:
            # Confirmed live (SU.TO, a mapped watchlist ticker): a single
            # filing row with a null `description` fails Pydantic
            # validation for the WHOLE batch inside openbb-tmx's own model
            # (TmxCompanyFilingsData), raising OpenBBError before a result
            # object even comes back — this is not the already-handled
            # "genuinely empty result" OpenBBError seen in
            # get_analyst_ratings above, it's a data-quality defect in one
            # row poisoning the entire response. Treated as "no usable data
            # from this call," which flows into the two-tier retry below
            # the same as a clean empty result would. Logged (this file
            # otherwise has no structlog usage — every other defensive
            # workaround here is silent) because unlike those, this one
            # can't be distinguished from "TMX API genuinely degraded" by
            # its symptoms alone; losing that signal entirely if this ever
            # spreads past the one confirmed ticker isn't worth the noise
            # avoided by staying silent.
            try:
                result = await asyncio.to_thread(
                    obb.equity.fundamental.filings,
                    symbol=bare_symbol,
                    provider=self.PROVIDER,
                    **kwargs,
                )
            except OpenBBError as e:
                logger.warning(
                    "get_filings_openbb_error", ticker=ticker, symbol=bare_symbol, error=str(e)
                )
                return pd.DataFrame()
            if not result.results:
                return pd.DataFrame()
            return result.to_df()

        df = await _fetch()
        material = self._filter_material_filings(df)

        if material.empty:
            wide_start = (datetime.now() - timedelta(days=self._FALLBACK_LOOKBACK_DAYS)).strftime(
                "%Y-%m-%d"
            )
            df = await _fetch(start_date=wide_start)
            material = self._filter_material_filings(df)

        if material.empty:
            return []

        sort_key = pd.to_datetime(material["filing_date"])
        material = (
            material.assign(_sort_key=sort_key)
            .sort_values("_sort_key", ascending=False)
            .drop(columns="_sort_key")
        )
        return material.head(limit).to_dict("records")

    def _filter_material_filings(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "report_type" not in df.columns:
            return pd.DataFrame()
        return df[df["report_type"].isin(self._MATERIAL_FILING_TYPES)]

    # ---------- NewsProvider ----------

    async def get_news(self, ticker: str, days: int) -> list[dict]:
        """Live-verified 2026-08-04 (86bb7j0kh): real articles, with a real
        `date` field on each result object — but NOT surfaced by
        `.to_df()`'s default columns, so this reads `.results` directly
        (pydantic models) rather than going through the DataFrame.

        Returns the house news shape — `headline` / `summary` / `source` /
        `url` / `published_at` — matching `finnhub.py::get_news`, not
        OpenBB's raw `title` / `excerpt` / `date` field names. The only
        consumer, `precompute/news_id_assignment.py::assign_news_ids()`,
        keys off `published_at`; before this mapping every Canadian article
        was silently dropped (86bbqh23f). `summary` is `""` in practice:
        `excerpt` and `body` are `None` on every Canadian article
        (headline-only — a QuoteMedia/TMX limitation, not a bug here; it
        belongs in the reliability discount). `published_at` carries
        OpenBB's `America/New_York`-local wall clock with tz stripped —
        consistent within this CA-only list; a CA+US merge (86bbr4azz) must
        reconcile it against Finnhub's box-local strings.

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
            articles.append(
                {
                    "headline": item.title or "",
                    "summary": item.excerpt or "",
                    "source": item.source or "",
                    "url": item.url or "",
                    "published_at": (
                        article_date.strftime("%Y-%m-%d %H:%M:%S") if article_date else None
                    ),
                }
            )
        return articles

    async def get_analyst_recommendation_trends(self, ticker: str) -> list[dict]:
        # Not supported by TMX provider.
        raise NotImplementedError("Recommendation trends are not available via openbb-tmx.")
