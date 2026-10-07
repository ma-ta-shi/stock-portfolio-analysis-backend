"""DataBundle contract model + get_benchmark() (ClickUp 86bawp863).

Full field spec: docs/technical/data-pipeline.md §4 "DataBundle", "Benchmark
selection". The valuation_metrics/technical_indicators/etc. dicts stay as
dict for this pass per the layout ticket's (86bawp7yx) call — the doc
doesn't give them a fully specified shape yet.
"""

from datetime import UTC, datetime
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


def build_data_freshness(
    sources_used: dict[str, str], *, at: datetime | None = None
) -> dict[str, str]:
    """Per-source fetch timestamp for audit (86bawptw7). Caller
    (DataPipeline.prepare(), not yet built) passes a Router's sources_used
    once all of that Router's calls for one bundle are done. One shared
    timestamp for every entry, not a true per-call timestamp: no provider or
    the router implements any caching today, so every call in one prepare()
    run happens inline, back-to-back, in the same fetch pass — there is no
    real timing difference between entries for audit to preserve. If a
    caching layer is added later, that layer is the right place to add
    genuine per-call timestamps (it would need to timestamp at the cache
    hit/miss point anyway, which Router doesn't do now).

    data_vintage needs no helper of its own — it's this same `at` timestamp,
    captured once by DataPipeline.prepare() when it fetches the primary
    stock's price history and passed into both places."""
    stamp = (at or datetime.now(UTC)).isoformat()
    return {method: stamp for method in sources_used}


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
        # primary_exchange, asset_type — matches providers.base.NormalizedCompanyInfo
        # (86bbb001k); was previously documented as "exchange", which no payload
        # template ever used. asset_type (86bbpk6uf) is routing-only, not rendered
        # in any agent payload.
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
    missing_fields: list[str]  # fundamentals.py::compute_all()'s own return -- "bucket.key"
    # entries for every None value across valuation_metrics/growth_metrics/
    # profitability_metrics/balance_sheet_metrics (86bbwachy Phase 4). Computed there
    # already, previously discarded before reaching DataBundle -- persisted here rather
    # than re-derived. Does NOT cover dividend_info/peer_metrics/analyst_consensus
    # (compute_all()'s own scan never touched those buckets). Of those three,
    # pass1_fundamental_analyst.py's own input_field_coverage tracks peer_metrics
    # separately (a direct dict-value check, not missing_fields-derived); dividend_info
    # and analyst_consensus aren't tracked there at all -- dividend_regularity="none" is
    # itself an honest "no dividend" answer, not a gap, and analyst_consensus is already
    # tracked by Sentiment Analyst's own input_field_coverage.
    currency_mismatch: dict | None  # {financials_currency, quote_currency, converted, usd_cad} when a
    # CA-listed company reports in a different currency than its quote (e.g. ATD.TO
    # reports in USD). The statements are converted into the quote currency at the Bank of
    # Canada rate (data/precompute/currency.py), so the multiples are right; `converted` False
    # means no rate or an unsupported pair and the multiples were dropped. A quality flag, not a presence flag -- deliberately NOT folded
    # into input_field_coverage's {field: bool} map, which only answers "was this
    # present," not "is this present value trustworthy."
    not_applicable: dict | None = None  # {reason: "REIT", fields: [...]}: metrics that do not exist
    # for this kind of company and came out empty, so the payloads say "not applicable" instead of a data gap.
    metric_profile: dict | None = None  # {group: "financials"|"capital_intensive"|"pre_profit"|"standard", hidden: [...],
    # lens: str, limits: str}: what this stock is judged on (fundamentals.py::metric_group). The Fundamental payload and
    # Pass 2 view leave the hidden metrics out entirely instead of marking them N/A.
    latest_financials_period_end: str | None  # ISO "YYYY-MM-DD", the newest quarter's
    # own period_end (fundamentals.py::compute_all()'s own new return key, 86bbummwp
    # Tier 2) -- was fetched (NormalizedFinancials.quarters[0]["period_end"]) but never
    # forwarded past compute_all() before this. The only real freshness signal
    # Fundamental Analyst has anywhere; None when quarters is empty.

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
    sentiment_source: Literal["local_llm"] | None  # 86ban0wf4: the one real field for this -
    # CanadianDataFlags.sentiment_source was removed entirely (86bawptx4) rather than kept as a
    # 3-value duplicate, since it would always hold this exact same value for a CA stock (one
    # precompute/sentiment.py call scores both markets, no CA-specific branch). None only when
    # news_with_sentiment is None/empty — no articles were scored.
    # BB-023: how much news the window really held, so the Sentiment Analyst can be told
    # "N fetched, M scored, K shown" instead of mistaking a sample for the whole.
    # {"window_days": int, "fetched": int}; None when not recorded (older callers).
    news_coverage: dict | None = None
    analyst_consensus: dict
    analyst_recommendation_trends: list[dict] | None  # monthly rating distribution: Finnhub (US), yfinance (CA)
    # Dated rating changes and price-target moves, newest first (yfinance upgrades_downgrades, both markets);
    # summarised in code by precompute/sentiment_signals.py. None on older callers, [] when the name has none.
    analyst_rating_changes: list[dict] | None = None
    insider_activity: dict
    short_interest: dict | None
    canadian_data_flags: CanadianDataFlags | None

    # --- Macro Economist (Pass 1) ---
    macro_sources: (
        MacroSourcesBundle  # 86bawptxa: precompute/macro_sources.py::compute_macro_sources()
        # is already the complete builder - a caller just passes its output straight through,
        # no further assembly logic needed here (unlike research_sources/canadian_data_flags,
        # which combine multiple inputs at this layer)
    )

    # --- Risk Advisor (Pass 2) ---
    risk_metrics: dict

    # Tax Strategist (Pass 2) has no DataBundle field of its own (removed 86bc8efvb,
    # was `tax_metrics: str | None`): it depends on account_state and
    # user_tax_profile, neither of which DataBundle carries (both are per-user data,
    # not derivable from `stock` alone), and DataBundle is frozen — a field that
    # can't be correctly computed at construction time has no business being one.
    # pass2_tax_strategist.py now calls build_tax_metrics_field() directly, fresh,
    # every call — the only real consumer this field ever had.

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
