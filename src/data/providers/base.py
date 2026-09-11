import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, TypedDict

import pandas as pd

_PERIOD_RE = re.compile(r"^(\d+)(d|mo|y)$")
_PERIOD_UNIT_DAYS = {"d": 1, "mo": 30, "y": 365}


def period_to_from_date(period: str) -> str:
    """Convert a period token ('6mo', '1y', '5y', 'max', ...) to an ISO
    'YYYY-MM-DD' start date. Providers whose price endpoint takes a date
    range rather than a yfinance-style period token (openbb-tmx, FMP) use
    this so the whole StockDataProvider price surface accepts one period
    vocabulary. The 30-/365-day month/year approximation is ~3 days short
    at '10y' — immaterial for the indicator/drawdown windows this feeds."""
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


class NormalizedQuote(TypedDict):
    """Canonical get_quote() shape, needed by precompute/fundamentals.py's
    price_info parameter (86bbb17pw). Confirmed live 2026-08-10: both FMP's
    /quote and yfinance's fast_info already include 52-week high/low
    directly (yearHigh/yearLow, year_high/year_low) — no derivation from
    get_price_history() needed, unlike this ticket's original assumption.
    openbb_tmx has no get_quote() method at all (router.py's CA_CHAINS
    already routes get_quote to yfinance only), so only fmp.py/yfinance.py
    need this mapping."""

    current_price: float
    market_cap: float | None
    currency: str
    high_52w: float | None
    low_52w: float | None


class NormalizedDividendRecord(TypedDict):
    """Canonical get_dividend_history() record shape, matching
    DataBundle.dividend_history's own documented comment
    (# [{ex_date, payment_date, amount_per_share}]). Real, confirmed gap
    found auditing fundamentals.py's full data need (86bbb001k, same
    session as NormalizedCompanyInfo/NormalizedFinancials): yfinance's
    get_dividend_history() declared -> list[dict] but every code path
    actually returned a pd.DataFrame — a type-contract violation, not just
    an unnormalized shape, that would have silently iterated column-name
    strings instead of dividend records downstream. payment_date is a real,
    disclosed gap for yfinance specifically — its raw .dividends Series has
    only the ex-dividend date, no separate payment date (unlike FMP/TMX,
    both confirmed live to have a real paymentDate/payment_date field)."""

    ex_date: str  # ISO date "YYYY-MM-DD"
    payment_date: str | None
    amount_per_share: float


class NormalizedInsiderTransaction(TypedDict):
    """Canonical get_insider_trading() record shape (86bbwha5r). Three very
    different raw shapes collapse into this: edgartools' Form 4 data is
    already transactional (date/shares/price/code per filing), yfinance's
    CA data is transactional too but has no code system (see below),
    openbb-tmx's CA data is a quarterly PER-OWNER AGGREGATE with no date
    and no per-transaction detail at all — CA_CHAINS lists yfinance first
    specifically because it's the only CA source with real transactions;
    openbb-tmx only serves when yfinance returns empty for a ticker
    (confirmed live: happens for real names, e.g. AEM.TO, BCE.TO).

    yfinance's raw `.insider_transactions` has a `Transaction` column that
    is an empty string on every row checked across 8 CA tickers — the
    actual free-text description is in the `Text` column instead. Getting
    this backwards was caught before it shipped; if a future adapter
    change ever touches this again, verify against a live pull, not the
    visual column alignment of a printed DataFrame (that's what caused
    the original mix-up).

    Roughly half of yfinance's CA rows (verified across RY.TO/SHOP.TO/
    WCN.TO) have a blank `Text` — not rare filler, real transactions
    (often large `Issuer`-position buybacks) that yfinance's scraper never
    attached a description to. They carry no `value` either (confirmed:
    blank `Text` and missing `value` correlate perfectly on every ticker
    checked) so they can't be classified purchase vs. sale from any field
    this adapter has access to. They still belong in the output —
    `is_issuer`/`shares` are real — they just land in transaction_type
    "other" like any other non-directional row, silently (no log; a
    missing description isn't the same failure as an unrecognized one).

    No `role`/title field, though one is available for free on the CA
    side: yfinance's raw `Position` ("Director of Issuer", "Senior
    Officer of Issuer") is exactly the field the Sentiment Analyst
    prompt's payload template wanted before it removed `{role}` for lack
    of a source (2026-08-30). Deliberately not added here — nothing
    would read it yet (research_sources.py doesn't exist), and populating
    a field nothing consumes is the orphan-field problem 86bbun08w exists
    to prevent. Worth wiring in (from `Position`) whenever the Sentiment
    prompt's payload assembly is next touched."""

    date: str | None  # ISO "YYYY-MM-DD"; None for the openbb-tmx aggregate (no date at all)
    insider_name: str
    is_issuer: bool  # the "insider" is the company itself (a buyback), not a person
    transaction_type: Literal["purchase", "sale", "exercise", "gift", "buyback", "other"]
    shares: float | None
    value: float | None  # trade value in local currency; None when the source doesn't report it


class NormalizedCompanyInfo(TypedDict):
    """Canonical get_company_info() shape, confirmed by grepping every agent
    prompt that references DataBundle.company_info (86bbb001k) — all 10
    (Bull/Bear/Sentiment/Macro/Technical/Stock Researcher/Fundamental/Risk
    Advisor/CIO/Shadow CIO) render the first seven fields, no variation;
    asset_type (added 86bbpk6uf) is routing-only, not rendered anywhere.
    Each adapter (fmp.py/yfinance.py/openbb_tmx.py) maps its own raw
    get_company_info() response into this shape before returning. A
    TypedDict is a plain dict at runtime, so router.py's _is_empty()/
    _try_chain() need no changes to keep working with it.
    """

    name: str
    sector: str  # GICS
    industry: str  # GICS — documented in data-pipeline.md's DataBundle sketch but
    # not consumed by any agent prompt yet (peer-matching is static/
    # API-driven, not industry-filtered) — kept since every provider
    # already returns it and dropping it would just mean re-adding later
    market_cap: float | None
    currency: str  # "CAD" | "USD"
    country: str
    primary_exchange: str  # NOT "exchange" — every payload template across
    # all 10 consumer prompts renders {primary_exchange}, never {exchange}
    asset_type: Literal["equity", "etf", "other"]  # NOT consumed by any agent
    # prompt (same as industry above) — added for pipeline routing (86bbpk6uf
    # part 2): a gate rejects "etf"/"other" before a run so the equity-only
    # agents never score a fund/index/preferred. Each adapter maps its own
    # provider's classification (yfinance quoteType / FMP isEtf|isFund /
    # openbb-tmx issue_type); "other" absorbs funds, indexes, preferreds,
    # warrants. No provider distinguishes closed-end fund from mutual fund,
    # so finer values aren't populatable.


class NormalizedAnalystEstimates(TypedDict):
    """Canonical get_analyst_estimates() shape (86bbdu04a). One field,
    deliberately — the only real consumer, confirmed against the live
    Fundamental Analyst prompt (not just the ticket), is
    compute_valuation_metrics()'s forward_pe (current_price / forward_eps).
    A fiscal-year-end/period field was considered and cut: FMP's
    /analyst-estimates rows do carry one, but nothing downstream reads it
    — same "don't build past the real consumer" reasoning behind deferring
    get_analyst_ratings() normalization (86bbdu04a scope decision), applied
    to this type's own fields too, not just the sibling method.

    Real bug caught via live integration check: every adapter (fmp.py/
    openbb_tmx.py/yfinance.py) must return a bare {} when forward_eps
    can't be found — never {"forward_eps": None}. router.py's _is_empty()
    treats any non-empty dict as a success regardless of its values, so a
    1-key dict with a None value would make a fallback chain behind this
    method never actually fall through — the CA chain silently never
    tried yfinance behind openbb_tmx until this was caught. Same
    dict-is-a-dict-at-runtime consideration as NormalizedCompanyInfo
    above, but load-bearing here specifically because this type has only
    one field, so a null value can't be distinguished from a "no data"
    signal by shape alone the way a partially-populated multi-field dict
    still could be."""

    forward_eps: float | None


@dataclass
class NormalizedFinancials:
    """Canonical get_financials() shape (86bbb001k), produced by
    yfinance.py's/edgartools.py's normalize_financials() and consumed by
    precompute/fundamentals.py — kept as a plain dataclass, not a Pydantic
    contract, matching DataBundle's own convention of leaving precompute-
    internal shapes as dict/dataclass rather than a validated schema.
    """

    quarters: list[dict]  # newest first, each: {period_end, revenue, net_income,
    # net_income_common, eps, operating_income, interest_expense, tax_expense,
    # depreciation_amortization, dividends_paid, shares_outstanding,
    # cost_of_revenue, operating_cash_flow, capital_expenditures}
    # cost_of_revenue/operating_cash_flow/capital_expenditures added beyond
    # the original ticket spec — needed for gross_margin and free_cash_flow,
    # both live Fundamental Analyst prompt core fields the original shape
    # couldn't actually compute. net_income_common (income to common after
    # preferred dividends) is optional — present from yfinance, absent from
    # edgartools; fundamentals.py falls back to net_income when it's missing.
    annual: list[dict]  # same shape as quarters, 3+ years for CAGR
    balance_sheet: dict  # latest only (not per-period): total_assets,
    # total_liabilities, total_equity, total_debt, cash_and_equivalents,
    # current_assets, current_liabilities
    currency: str  # "CAD" | "USD" — the statement (financial) currency, which
    # for a Canadian-listed USD-reporter (ATD.TO, NTR.TO, ...) is NOT the quote
    # currency; compute_all() flags that mismatch as currency_mismatch (86bbxucf0)

    ttm: dict | None = None  # trailing-twelve-month aggregates, keyed like
    # quarters[] (revenue, net_income, net_income_common, operating_income,
    # operating_cash_flow, capital_expenditures, depreciation_amortization,
    # dividends_paid — NOT eps). None means "no explicit TTM, derive from
    # quarters[:4]" (the yfinance path). edgartools sets it (86bbxuj9e) because
    # its quarterly history is too shallow/gapped for a 4-quarter sum — it is
    # computed by YTD algebra: latest-10-K FY total + latest-10-Q YTD minus the
    # prior-year YTD. fundamentals.py reads it via _ttm_metric().


class StockDataProvider(ABC):
    """Abstract base for stock-centric financial data (FMP, yfinance)."""

    @abstractmethod
    async def get_price_history(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Daily/weekly OHLCV, **split-adjusted, dividend-unadjusted** — the
        series a retail user sees on their broker's chart, and the basis
        every technical indicator and drawdown metric downstream is written
        against. openbb-tmx and FMP are natively this; yfinance is forced to
        it with auto_adjust=False (86bbq7dkv). Total return (price +
        dividends) is a separate concern, computed from get_dividend_history.

        `period`: the portable vocabulary is `1mo`/`3mo`/`6mo`/`1y`/`2y`/`5y`/
        `10y`/`max` — valid natively in yfinance and via period_to_from_date
        in the date-range providers. `interval`: `1d` or `1wk`."""
        ...
    @abstractmethod
    async def get_financials(self, ticker: str, statement: str, period: str) -> pd.DataFrame:
        """Individual adapters (fmp.py, yfinance.py, edgartools.py, openbb_tmx.py)
        return a bare pd.DataFrame, matching this signature exactly.
        Router.get_financials() is a deliberate, documented exception —
        it returns tuple[pd.DataFrame, str | None] instead, exposing which
        provider answered so callers can dispatch to the right adapter's
        normalize_financials() (86bbb001k). Found during a later sweep and
        flagged here rather than "fixed" back to match this signature —
        this divergence is intentional, not an oversight."""
        ...
    @abstractmethod
    async def get_company_info(self, ticker: str) -> NormalizedCompanyInfo: ...
    @abstractmethod
    async def get_analyst_estimates(self, ticker: str) -> NormalizedAnalystEstimates: ...
    @abstractmethod
    async def get_analyst_ratings(self, ticker: str) -> dict: ...
    @abstractmethod
    async def get_earnings_surprises(self, ticker: str) -> list[dict]:
        """Historical actual-vs-estimate earnings, newest first, each:
        {period_end: str, eps_actual: float | None, eps_estimated: float | None,
        eps_surprise_pct: float | None, revenue_actual: float | None,
        revenue_estimated: float | None}. No revenue_surprise_pct — neither
        real source (FMP, yfinance) precomputes one, and fundamentals.py's
        documented gap (86bbdu04a) doesn't need it. revenue_actual/
        revenue_estimated stay None on any source that can't provide them
        (yfinance)."""
        ...
    @abstractmethod
    async def get_insider_trading(
        self, ticker: str, days: int = 90
    ) -> list[NormalizedInsiderTransaction]:
        """See NormalizedInsiderTransaction's own docstring for the real
        shape differences this hides (86bbwha5r) — openbb-tmx's CA path in
        particular is a quarterly aggregate, not real transactions, and is
        the CA fallback only (CA_CHAINS prefers yfinance)."""
        ...
    @abstractmethod
    async def get_peers(self, ticker: str, limit: int = 5) -> list[str]: ...
    @abstractmethod
    async def get_earnings_calendar(self, ticker: str) -> list[dict]: ...
    @abstractmethod
    async def get_dividend_history(
        self, ticker: str, from_date: str, to_date: str
    ) -> list[NormalizedDividendRecord]:
        """Fetch historical dividends for total return calculations.
        Used by the checkpoint evaluator to compute actual total return
        (price change + dividends) rather than price-only return.
        Source:
          - Canadian stocks: openbb-tmx .equity.fundamental.dividends
          - US stocks: FMP /api/v3/historical-price-full/stock_dividend/{ticker}"""
        ...


class NewsProvider(ABC):
    """Abstract base for news data (FMP news, Finnhub news with sentiment)."""
    
    @abstractmethod
    async def get_news(self, ticker: str, days: int) -> list[dict]: ...
    @abstractmethod
    async def get_analyst_recommendation_trends(self, ticker: str) -> list[dict]: ...


class MacroDataProvider(ABC):
    """Abstract base for macroeconomic data (FRED, Bank of Canada Valet)."""
    
    @abstractmethod
    async def get_macro_data(self, series_ids: list[str]) -> dict[str, pd.Series]: ...
    @abstractmethod
    async def get_interest_rates(self) -> dict: ...
    @abstractmethod
    async def get_exchange_rates(self, pair: str = "CADUSD") -> dict: ...