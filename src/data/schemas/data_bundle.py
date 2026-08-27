"""DataBundle contract model + get_benchmark() (ClickUp 86bawp863).

Full field spec: docs/technical/data-pipeline.md §4 "DataBundle", "Benchmark
selection". The valuation_metrics/technical_indicators/etc. dicts stay as
dict for this pass per the layout ticket's (86bawp7yx) call — the doc
doesn't give them a fully specified shape yet.
"""

from datetime import datetime
from typing import Literal

from pydantic import NonNegativeInt, model_validator

from data.schemas.base import ContractModel
from data.schemas.canadian_data_flags import CanadianDataFlags
from data.schemas.common import StockRef
from data.schemas.context import AnalysisContext
from data.schemas.macro_sources_bundle import MacroSourcesBundle
from data.schemas.research_sources_bundle import ResearchSourcesBundle


def _is_canadian_stock(stock: StockRef) -> bool:
    """Same check as data.providers.router.is_canadian() — deliberately
    NOT imported from there (same reasoning as StockLike in common.py:
    keep data.schemas independent of data.providers, which transitively
    imports every provider adapter). router.py's own docstring says this
    check and get_benchmark() must "stay in sync" — if CA-detection logic
    ever changes, update both places."""
    return stock.exchange in ("TSX", "TSXV") or stock.currency == "CAD"


def get_benchmark(stock: StockRef) -> str:
    """Used everywhere a benchmark is needed: technical analyst relative
    performance, risk advisor beta, feedback engine relative scoring.

    Takes StockRef, not the Stock ORM model the doc's own snippet shows —
    every real call site operates on an assembled DataBundle (bundle.stock
    is already a StockRef), and StockRef is this package's canonical
    lightweight stock representation throughout the pipeline layer. A
    caller holding a live ORM Stock instead can get a StockRef via
    StockRef.from_stock() first."""
    return "^GSPTSE" if _is_canadian_stock(stock) else "^GSPC"


class DataBundle(ContractModel):
    """All pre-computed data needed for one analysis run. Populated by
    DataPipeline.prepare() (not yet built). Each agent receives only the
    slice of the bundle relevant to its role.

    Type definition only — the model_validator below (canadian_data_flags
    presence must match whether the stock is CA-listed) is a data-validity
    invariant, not business/scoring logic, same reasoning as the other
    three contracts in this package.
    """

    stock: StockRef
    context: AnalysisContext

    # --- Stock Researcher (Pass 1) ---
    company_info: (
        dict  # name, sector (GICS), industry (GICS), market_cap, currency, country,
        # primary_exchange — matches providers.base.NormalizedCompanyInfo (86bbb001k);
        # was previously documented as "exchange", which no payload template ever used
    )
    research_sources: ResearchSourcesBundle

    # --- Fundamental Analyst (Pass 1) ---
    valuation_metrics: dict  # P/E, P/B, P/S, EV/EBITDA, PEG
    growth_metrics: dict
    profitability_metrics: dict
    balance_sheet_metrics: dict  # incl. health_rating: healthy/adequate/stressed
    dividend_info: dict
    dividend_history: list[dict]  # [{ex_date, payment_date, amount_per_share}]
    peer_metrics: dict
    price_info: dict

    # --- Technical Analyst (Pass 1) ---
    technical_indicators: dict
    support_resistance: dict
    trend_structure: dict
    multi_timeframe: dict
    pattern_metrics: dict
    market_context: dict
    liquidity_flags: dict
    earnings_proximity: dict
    price_position: dict
    is_current: bool
    days_old: NonNegativeInt
    preflight_warnings: list[str]

    # --- Sentiment Analyst (Pass 1) ---
    news_with_sentiment: list[dict] | None  # precompute/sentiment.py (86ban0wf4); populated
    # for both markets now — no longer Finnhub/US-only, see sentiment_source below
    sentiment_source: Literal["local_llm"] | None  # 86ban0wf4: market-agnostic companion to
    # CanadianDataFlags.sentiment_source, which structurally can't cover US stocks (that field
    # stays None there by DataBundle's own validator). None only when news_with_sentiment is
    # None/empty — no articles were scored. Single-value literal, not CanadianDataFlags' 3-value
    # enum, because "finnhub"/"keyword" are confirmed-dead paths for both markets as of this
    # ticket — flag to whoever owns CanadianDataFlags's schema if those should be removed too.
    analyst_consensus: dict
    analyst_recommendation_trends: list[dict] | None  # Finnhub: US only
    insider_activity: dict
    short_interest: dict | None
    peer_sentiment: list[dict]
    canadian_data_flags: CanadianDataFlags | None

    # --- Macro Economist (Pass 1) ---
    macro_sources: MacroSourcesBundle

    # --- Risk Advisor (Pass 2) ---
    risk_metrics: dict

    # --- Metadata ---
    benchmark_ticker: str
    data_freshness: dict
    data_vintage: datetime  # used by Portfolio Optimizer to normalize for market-level
    # moves between analysis dates in multi-day batches

    @model_validator(mode="after")
    def _check_canadian_data_flags_presence(self) -> "DataBundle":
        """Doc states this as a "check before reading" comment on the
        field — made a real constraint instead of a convention agents
        have to remember."""
        is_ca = _is_canadian_stock(self.stock)
        if is_ca and self.canadian_data_flags is None:
            raise ValueError("canadian_data_flags must be set for a CA/TSX-listed stock")
        if not is_ca and self.canadian_data_flags is not None:
            raise ValueError("canadian_data_flags must be None for a non-CA stock")
        return self

    @model_validator(mode="after")
    def _check_benchmark_ticker_matches_stock(self) -> "DataBundle":
        """Real gap caught on review: get_benchmark() exists specifically
        to compute this field, but nothing enforced benchmark_ticker
        actually agreeing with it — a US stock paired with "^GSPTSE" (or
        the reverse) would have constructed silently."""
        expected = get_benchmark(self.stock)
        if self.benchmark_ticker != expected:
            raise ValueError(
                f"benchmark_ticker ({self.benchmark_ticker!r}) does not match "
                f"get_benchmark(stock) ({expected!r})"
            )
        return self

    @model_validator(mode="after")
    def _check_us_only_sentiment_fields_are_none_for_ca_stocks(self) -> "DataBundle":
        """Real gap caught on review: the doc comments say "Finnhub: US
        stocks only" / "Finnhub: US only" on these two fields, and this is
        a confirmed hard technical fact this session (finnhub.py's
        _reject_ca_ticker raises NotImplementedError for any .TO ticker),
        not just a note — a CA stock can never have real values here. One-
        directional only: a US stock legitimately CAN still be None (a
        Finnhub call can fail for any ticker), so that direction isn't
        constrained.

        news_with_sentiment's clause removed (ClickUp 86ban0wf4): its "US-
        only" premise no longer holds — precompute/sentiment.py now scores
        both markets via a local LLM, not Finnhub (which never had CA
        sentiment and, confirmed this ticket, doesn't have US sentiment
        either). analyst_recommendation_trends is untouched — a genuinely
        separate, still-accurate Finnhub-only constraint."""
        if _is_canadian_stock(self.stock):
            if self.analyst_recommendation_trends is not None:
                raise ValueError(
                    "analyst_recommendation_trends must be None for a CA stock (Finnhub US-only)"
                )
        return self

    @model_validator(mode="after")
    def _check_sentiment_source_matches_news_with_sentiment(self) -> "DataBundle":
        """86ban0wf4: sentiment_source and news_with_sentiment are both
        produced by the same precompute/sentiment.py call and must agree
        on whether any scoring happened — same cross-field-invariant
        reasoning as _check_benchmark_ticker_matches_stock above."""
        has_articles = bool(self.news_with_sentiment)
        has_source = self.sentiment_source is not None
        if has_articles != has_source:
            raise ValueError(
                "sentiment_source must be set iff news_with_sentiment is non-empty "
                f"(news_with_sentiment={'set' if has_articles else 'empty/None'}, "
                f"sentiment_source={self.sentiment_source!r})"
            )
        return self
