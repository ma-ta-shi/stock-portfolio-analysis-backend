"""DataPipeline.prepare() — final DataBundle assembly (86bawpty3).

The integration point for the whole DataBundle Assembly epic: fetches every
provider input and runs every precompute module needed to build one
DataBundle for one analysis run. Real call graph and every design decision
below are documented in full in the 86bawpty3 plan; this docstring only
covers what a reader of the code itself needs.
"""

import asyncio
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.tables.stock import Stock
from data.precompute import fundamentals, risk_metrics, sentiment, technicals
from data.precompute.canadian_data_flags import build_canadian_data_flags
from data.precompute.macro_sources import compute_macro_sources
from data.precompute.news_id_assignment import assign_news_ids
from data.precompute.research_sources import build_research_sources
from data.precompute.tax_metrics import build_tax_metrics_field
from data.providers.boc import BOCMacroDataProvider
from data.providers.finnhub import FinnhubDataProvider
from data.providers.fred import FredMacroDataProvider
from data.providers.router import Router
from data.providers.stats_canada import StatsCanadaProvider
from data.schemas.common import StockRef
from data.schemas.context import AnalysisContext
from data.schemas.data_bundle import DataBundle, build_data_freshness, get_benchmark

# US sector -> SPDR sector ETF, keyed on the real yfinance/FMP-style sector
# strings (confirmed live 2026-09-15, e.g. RY.TO -> "Finance" vs JPM ->
# "Financial Services" - the two markets do not share a taxonomy).
_US_SECTOR_ETF: dict[str, str] = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Defensive": "XLP",
    "Consumer Cyclical": "XLY",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}

# CA sector -> iShares TSX-listed sector ETF, keyed on the real openbb-tmx-
# style sector strings confirmed live the same session.
_CA_SECTOR_ETF: dict[str, str] = {
    "Finance": "XFN.TO",
    "Energy": "XEG.TO",
    "Materials": "XMA.TO",
    "Technology": "XIT.TO",
    "Industrials": "XID.TO",
    "Real Estate": "XRE.TO",
    "Utilities": "XUT.TO",
    "Healthcare": "XHC.TO",
    "Media & Telecommunications": "XTC.TO",
}

# Covers both real consumers of this window: fundamentals.py's
# dividend_growth_5yr (needs 5y) and tax_metrics.py's compute_trailing_
# dividend (needs 365 days) - 6y gives both a buffer.
_DIVIDEND_HISTORY_YEARS = 6


def resolve_sector_etf(sector: str | None, is_ca: bool) -> str | None:
    """Sector-relative-strength ETF for technicals.py's sector_etf_price
    input. An unmapped/unrecognized sector resolves to None, not a raise -
    matching every other precompute module's graceful-degradation
    convention in this codebase."""
    table = _CA_SECTOR_ETF if is_ca else _US_SECTOR_ETF
    return table.get(sector) if sector else None


class StockNotFoundError(Exception):
    pass


class DataPipeline:
    async def prepare(
        self, stock_id: UUID, context: AnalysisContext, db: AsyncSession
    ) -> DataBundle:
        stock = (
            await db.execute(select(Stock).where(Stock.stock_id == stock_id))
        ).scalar_one_or_none()
        if stock is None:
            raise StockNotFoundError(f"no stock found for stock_id={stock_id}")

        stock_ref = StockRef.from_stock(stock)
        ticker = stock_ref.ticker
        benchmark_ticker = get_benchmark(stock_ref)
        # Captured once, up front, and reused everywhere "now" is needed
        # below (dividend window, technicals' as_of, macro's as_of,
        # data_freshness/data_vintage) - date.today() would read the local
        # system date, which can disagree with this UTC date by a day near
        # midnight, and repeated datetime.now() calls risk visibly
        # different timestamps for what should be one shared "as of" point
        # for this whole run.
        fetched_at = datetime.now(UTC)

        async with Router(stock=stock) as router:
            is_ca = router.is_ca

            # Early, shared prerequisites - needed by later steps, not just
            # passed straight through.
            company_info, analyst_consensus = await asyncio.gather(
                router.get_company_info(ticker),
                router.get_analyst_ratings(ticker),
            )

            # News fetched once; the same id-assigned list feeds both
            # sentiment and research_sources so they cite articles under the
            # same N{n} ids.
            raw_news = await router.get_news(ticker, days=180)
            id_assigned_articles = assign_news_ids(raw_news)

            # Fundamentals inputs. normalize_financials() is a method on the
            # market's own adapter (yfinance for CA, edgartools for US) that
            # does its own internal fetching - it does not consume
            # Router.get_financials()'s chain/fallback output. Reuses the
            # instance Router already constructed for this ticker's market
            # rather than building a second one.
            fin_provider = router._providers["yfinance" if is_ca else "edgartools"]
            dividend_from = (
                fetched_at.date() - timedelta(days=365 * _DIVIDEND_HISTORY_YEARS)
            ).isoformat()
            dividend_to = fetched_at.date().isoformat()

            (
                fin,
                price_info,
                dividend_history,
                peer_tickers,
                analyst_estimates,
                earnings_surprises,
            ) = await asyncio.gather(
                fin_provider.normalize_financials(ticker),
                router.get_quote(ticker),
                router.get_dividend_history(ticker, dividend_from, dividend_to),
                router.get_peers(ticker),
                router.get_analyst_estimates(ticker),
                router.get_earnings_surprises(ticker),
            )

            # All peers fetched concurrently, not one at a time - each
            # normalize_financials() call is itself several sub-requests
            # (yfinance/edgartools), and rate limiting against the
            # underlying APIs is already the provider libraries' own job
            # (pyrate_limiter), not something this loop needs to hand-roll.
            peer_results = await asyncio.gather(
                *(
                    asyncio.gather(fin_provider.normalize_financials(t), router.get_quote(t))
                    for t in peer_tickers
                )
            )
            peer_data = [
                (peer_ticker, peer_fin, peer_quote)
                for peer_ticker, (peer_fin, peer_quote) in zip(peer_tickers, peer_results)
            ]

            fundamentals_result = fundamentals.compute_all(
                fin=fin,
                price_info=price_info,
                dividend_history=dividend_history,
                peer_data=peer_data,
                analyst_estimates=analyst_estimates,
                earnings_surprises=earnings_surprises,
            )

            # Technicals needs the subject's own price history plus a
            # benchmark's and (if resolvable) a sector ETF's - each on its
            # own Router instance, since Router.sources_used is keyed by
            # method name alone and one instance can't serve three
            # get_price_history calls without silently overwriting entries.
            # yfinance/openbb_tmx (session-less, no __aenter__/__aexit__) are
            # reused from the subject Router to avoid duplicate provider
            # construction; fmp/finnhub (aiohttp-session-owning) are
            # deliberately NOT shared across separate `async with` blocks -
            # double-entering one shared session-owning provider's
            # __aenter__ would silently replace its session out from under
            # the other Router still using it.
            reusable_providers = {}
            if "yfinance" in router._providers:
                reusable_providers["yfinance"] = router._providers["yfinance"]
            if "openbb_tmx" in router._providers:
                reusable_providers["openbb_tmx"] = router._providers["openbb_tmx"]

            sector_ticker = resolve_sector_etf(company_info.get("sector"), is_ca)

            # risk_metrics.py's own module docstring: "max_drawdown_3yr_pct/
            # recovery_3yr_days can only resolve if raw_price covers >=3
            # years of history... request a >=3y period from
            # router.get_price_history() when building this module's
            # inputs, or the 3yr fields will silently and permanently read
            # None." raw_price/benchmark_price both feed risk_metrics below,
            # so both need this longer window (5y for buffer, matching the
            # dividend-history buffer above). sector_etf_price only feeds
            # technicals.py's relative-strength calc, which only looks back
            # 3 months (technicals.py's own _relative_performance) - 1y is
            # already more than enough there.
            async def _fetch_benchmark_price():
                async with Router(stock=stock, **reusable_providers) as benchmark_router:
                    price = await benchmark_router.get_price_history(benchmark_ticker, "5y", "1d")
                    return price, benchmark_router.sources_used.get("get_price_history")

            async def _fetch_sector_etf_price():
                if sector_ticker is None:
                    return None, None
                async with Router(stock=stock, **reusable_providers) as sector_router:
                    price = await sector_router.get_price_history(sector_ticker, "1y", "1d")
                    return price, sector_router.sources_used.get("get_price_history")

            (
                raw_price,
                (benchmark_price, benchmark_source),
                (sector_etf_price, sector_source),
                earnings_calendar,
            ) = await asyncio.gather(
                router.get_price_history(ticker, "5y", "1d"),
                _fetch_benchmark_price(),
                _fetch_sector_etf_price(),
                router.get_earnings_calendar(ticker),
            )

            technicals_result = technicals.compute_all(
                raw_price=raw_price,
                benchmark_price=benchmark_price,
                sector_etf_price=sector_etf_price,
                earnings_calendar=earnings_calendar,
                news_ids=id_assigned_articles,
                as_of=fetched_at.date(),
            )

            # Sentiment cluster: sentiment + research_sources both consume
            # the same id-assigned article list; canadian_data_flags must
            # follow both research_sources and macro_sources, and is only
            # ever built for a CA stock (DataBundle's own validator requires
            # None for a US stock).
            sentiment_result, research_sources_bundle, insider_transactions = await asyncio.gather(
                sentiment.summarize_news(id_assigned_articles),
                build_research_sources(ticker, id_assigned_articles, stock),
                router.get_insider_trading(ticker),
            )

            fred = FredMacroDataProvider()
            stats_canada_cm = StatsCanadaProvider() if is_ca else nullcontext(None)
            async with (
                BOCMacroDataProvider() as boc,
                FinnhubDataProvider() as macro_finnhub,
                stats_canada_cm as stats_canada,
            ):
                macro_sources = await compute_macro_sources(
                    company_info.get("sector"),
                    is_ca,
                    context.timeline,
                    fred,
                    boc,
                    macro_finnhub,
                    stats_canada,
                    as_of=fetched_at,
                )

            canadian_data_flags = None
            if is_ca:
                canadian_data_flags = build_canadian_data_flags(
                    research_sources_bundle, macro_sources, analyst_consensus
                )

            analyst_recommendation_trends = None
            if not is_ca:
                analyst_recommendation_trends = await router.get_analyst_recommendation_trends(
                    ticker
                )
            short_interest = await router.get_short_interest(ticker)

            # sources_used merge-then-stamp: merge every Router's
            # sources_used into one dict first, prefixing the benchmark/
            # sector-ETF get_price_history entries to avoid colliding with
            # the subject's own, then stamp build_data_freshness() exactly
            # once with one shared timestamp - calling it once per Router
            # would risk three different `at` values.
            combined_sources = dict(router.sources_used)
            if benchmark_source:
                combined_sources["get_price_history:benchmark"] = benchmark_source
            if sector_source:
                combined_sources["get_price_history:sector_etf"] = sector_source
            data_freshness = build_data_freshness(combined_sources, at=fetched_at)

            # Minimal stand-in for build_tax_metrics_field()'s bundle param -
            # it only ever reads these three attributes (confirmed by
            # reading its full body). A real DataBundle can't be used here
            # since tax_metrics is itself one of DataBundle's own fields and
            # the model is frozen - no construct-then-fill-in-after path.
            # Matches this codebase's own established convention for this
            # exact situation (test_tax_metrics.py's _fake_bundle()).
            tax_metrics_input = SimpleNamespace(
                company_info=company_info,
                dividend_history=dividend_history,
                price_info=price_info,
            )
            tax_metrics = build_tax_metrics_field(ticker, context.account_type, tax_metrics_input)

            return DataBundle(
                stock=stock_ref,
                context=context,
                company_info=company_info,
                research_sources=research_sources_bundle,
                valuation_metrics=fundamentals_result["valuation_metrics"],
                growth_metrics=fundamentals_result["growth_metrics"],
                profitability_metrics=fundamentals_result["profitability_metrics"],
                balance_sheet_metrics=fundamentals_result["balance_sheet_metrics"],
                dividend_info=fundamentals_result["dividend_info"],
                dividend_history=dividend_history,
                peer_metrics=fundamentals_result["peer_metrics"],
                price_info=price_info,
                technical_indicators=technicals_result["technical_indicators"],
                support_resistance=technicals_result["support_resistance"],
                trend_structure=technicals_result["trend_structure"],
                multi_timeframe=technicals_result["multi_timeframe"],
                pattern_metrics=technicals_result["pattern_metrics"],
                market_context=technicals_result["market_context"],
                liquidity_flags=technicals_result["liquidity_flags"],
                earnings_proximity=technicals_result["earnings_proximity"],
                price_position=technicals_result["price_position"],
                is_current=technicals_result["is_current"],
                days_old=technicals_result["days_old"],
                preflight_warnings=technicals_result["preflight_warnings"],
                news_with_sentiment=sentiment_result["articles"],
                sentiment_source=sentiment_result["sentiment_source"],
                analyst_consensus=analyst_consensus,
                analyst_recommendation_trends=analyst_recommendation_trends,
                insider_activity={"transactions": insider_transactions},
                short_interest=short_interest,
                peer_sentiment=[],
                canadian_data_flags=canadian_data_flags,
                macro_sources=macro_sources,
                risk_metrics=risk_metrics.compute_all(
                    raw_price=raw_price,
                    benchmark_price=benchmark_price,
                    benchmark_ticker=benchmark_ticker,
                    currency=company_info.get("currency", stock.currency),
                ),
                tax_metrics=tax_metrics,
                benchmark_ticker=benchmark_ticker,
                data_freshness=data_freshness,
                data_vintage=fetched_at,
            )
