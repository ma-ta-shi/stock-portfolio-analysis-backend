from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypedDict

import pandas as pd


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


class NormalizedCompanyInfo(TypedDict):
    """Canonical get_company_info() shape, confirmed by grepping every agent
    prompt that references DataBundle.company_info (86bbb001k) — all 10
    (Bull/Bear/Sentiment/Macro/Technical/Stock Researcher/Fundamental/Risk
    Advisor/CIO/Shadow CIO) use exactly these fields, no variation. Each
    adapter (fmp.py/yfinance.py/openbb_tmx.py) maps its own raw
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


class StockDataProvider(ABC):
    """Abstract base for stock-centric financial data (FMP, yfinance)."""

    @abstractmethod
    async def get_price_history(self, ticker: str, period: str, interval: str) -> pd.DataFrame: ...
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
    async def get_insider_trading(self, ticker: str, days: int = 90) -> list[dict]: ...
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