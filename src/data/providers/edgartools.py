import asyncio
import os

import pandas as pd
import structlog
from edgar import Company, set_identity

from data.providers.base import NormalizedFinancials, StockDataProvider

logger = structlog.get_logger(__name__)

_EDGAR_IDENTITY_ENV = "SP_EDGAR_IDENTITY"
_identity = os.environ.get(_EDGAR_IDENTITY_ENV)
if _identity:
    set_identity(_identity)
# If unset, edgartools raises its own error on first EDGAR request rather than
# here at import — see financial-data-api-research.md §3.

# (period, statement) -> Financials accessor. Mirrors yfinance.py's MAPPING pattern.
STATEMENT_MAP = {
    "income": lambda financials: financials.income_statement,
    "balance": lambda financials: financials.balance_sheet,
    "cashflow": lambda financials: financials.cashflow_statement,  # no underscore
}

# XBRL concept maps for normalize_financials() (86bbb001k) — confirmed live against
# AAPL's get_quarterly_financials()/get_financials() (2026-08-07), not guessed.
# Real, disclosed limitation: XBRL concept tags can legitimately vary by filer (GAAP
# allows more than one valid tag for some line items, e.g. revenue) — a company using
# a different concept than AAPL's just gets None for that field, same graceful-
# degradation behavior as a genuinely missing row.
_INCOME_CONCEPTS = {
    "revenue": "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax",
    "cost_of_revenue": "us-gaap_CostOfGoodsAndServicesSold",
    "operating_income": "us-gaap_OperatingIncomeLoss",
    "net_income": "us-gaap_NetIncomeLoss",
    "tax_expense": "us-gaap_IncomeTaxExpenseBenefit",
    "eps": "us-gaap_EarningsPerShareDiluted",
    "shares_outstanding": "us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding",
    "interest_expense": "us-gaap_InterestExpense",  # unconfirmed live — AAPL doesn't
    # report this concept at all (minimal net debt); real for debt-heavy filers.
}
_CASHFLOW_CONCEPTS = {
    "depreciation_amortization": "us-gaap_DepreciationDepletionAndAmortization",
    "dividends_paid": "us-gaap_PaymentsOfDividends",
    "operating_cash_flow": "us-gaap_NetCashProvidedByUsedInOperatingActivities",
    "capital_expenditures": "us-gaap_PaymentsToAcquirePropertyPlantAndEquipment",
}
# total_debt has no single XBRL concept — derived by summing current + non-current
# term debt, confirmed live (AAPL: LongTermDebtCurrent + LongTermDebtNoncurrent).
_BALANCE_CONCEPTS = {
    "total_assets": "us-gaap_Assets",
    "total_liabilities": "us-gaap_Liabilities",
    "total_equity": "us-gaap_StockholdersEquity",
    "cash_and_equivalents": "us-gaap_CashAndCashEquivalentsAtCarryingValue",
    "current_assets": "us-gaap_AssetsCurrent",
    "current_liabilities": "us-gaap_LiabilitiesCurrent",
}
_DEBT_CONCEPTS = ("us-gaap_LongTermDebtCurrent", "us-gaap_LongTermDebtNoncurrent")


_META_COLUMNS = {
    "concept", "label", "standard_concept", "level", "abstract", "dimension",
    "is_breakdown", "dimension_axis", "dimension_member", "dimension_member_label",
    "dimension_label", "balance", "weight", "preferred_sign", "parent_concept",
    "parent_abstract_concept",
}


def _period_columns(df: pd.DataFrame, *, exclude_ytd: bool) -> list:
    """Non-metadata columns, i.e. the actual period value columns.

    Real, confirmed finding: get_quarterly_financials()'s income statement
    exposes BOTH a true single-quarter column ("... (Q3)") and a
    cumulative-since-fiscal-year-start column ("... (YTD)") for the same
    period end — a standard SEC 10-Q reporting convention, not an
    edgartools quirk. exclude_ytd=True (used for the quarters[] grain)
    drops the YTD columns to avoid silently reporting cumulative figures
    as if they were single-quarter. The cashflow statement's quarterly
    columns are ALL "(YTD)" — no true per-quarter column exists at all —
    so exclude_ytd=True on cashflow correctly yields zero usable columns;
    callers must not fetch cashflow for the quarters[] grain at all (see
    normalize_financials()). Annual columns are uniformly "(FY)", no
    filtering needed there."""
    columns = [c for c in df.columns if c not in _META_COLUMNS]
    if exclude_ytd:
        columns = [c for c in columns if "(YTD)" not in str(c)]
    return columns


def _extract_concept(df: pd.DataFrame, concept: str, column) -> float | None:
    """Consolidated top-line value for a concept — filters to dimension==False
    (real, confirmed finding: edgartools repeats a concept once per segment/
    product breakdown alongside the consolidated figure; dimension==False is
    the consolidated row). Takes the first match if more than one non-
    dimensional row shares a concept (observed on AAPL for NetIncomeLoss,
    which also labels a retained-earnings roll-forward row identically)."""
    if df.empty or "dimension" not in df.columns or column not in df.columns:
        return None
    matches = df[(df["concept"] == concept) & (df["dimension"] == False)]  # noqa: E712
    if matches.empty:
        return None
    value = matches.iloc[0][column]
    return None if pd.isna(value) else float(value)


def _sum_concepts(df: pd.DataFrame, concepts: tuple[str, ...], column) -> float | None:
    """Sums multiple concepts (e.g. current + non-current debt) into one
    derived field. None only if every component is missing — a company
    reporting only one of the two still gets a real partial total."""
    values = [v for c in concepts if (v := _extract_concept(df, c, column)) is not None]
    return sum(values) if values else None

# Gap 5b (financial-data-api-research.md) describes a "fewer than 5 key line items"
# XBRL-quality check for small-caps. Deliberately not implemented: live-tested
# against ~25 real small/micro-cap and pre-revenue tickers (including deliberately
# thin filers like pre-revenue biotechs) and it never fired below 3-of-5 on any of
# them — while a literal "all 5 required" reading misfired on JPM, Visa, and P&G
# (mega-caps whose sector legitimately has no GrossProfit/OperatingIncome line,
# not an XBRL tagging problem).
#
# get_financials() returning no data outright (no computed Financials object —
# e.g. no XBRL-tagged filings) used to trigger an internal fallback to
# yfinance here. Moved out to router.py (ClickUp 86bb4758g) — "provider
# adapters stay dumb, router owns every completeness check." This class no
# longer constructs or calls another provider; it returns an empty DataFrame
# and lets the caller decide what to do next.


class EdgarToolsDataProvider(StockDataProvider):
    """US-stock fundamentals and insider transactions, sourced directly from SEC
    EDGAR (edgartools) — no ticker whitelist, no tier restriction, unlike FMP's
    free-tier annual-only fundamentals. Covers only the US-stock routing branch:
    prices, profile, estimates, ratings, peers, earnings calendar, and dividends
    are fmp.py's job, not this provider's."""

    async def get_financials(self, ticker: str, statement: str, period: str) -> pd.DataFrame:
        key = statement.lower()
        if key not in STATEMENT_MAP:
            raise ValueError("Invalid statement. Use 'income', 'balance', or 'cashflow'")

        company = await asyncio.to_thread(Company, ticker)
        get_financials_fn = (
            company.get_financials
            if period.lower() == "annual"
            else company.get_quarterly_financials
        )
        financials = await asyncio.to_thread(get_financials_fn)
        if financials is None:
            logger.warning(
                "edgar_financials_missing", ticker=ticker, statement=statement, period=period
            )
            return pd.DataFrame()

        statement_fn = STATEMENT_MAP[key](financials)
        df = await asyncio.to_thread(lambda: statement_fn().to_dataframe())
        # Real, latent gap this surfaced (86bbb001k): the Financials container
        # can exist while one specific statement isn't XBRL-tagged (e.g. a
        # filer missing a cash flow statement) — to_dataframe() returns None
        # in that case, violating this method's own -> pd.DataFrame contract.
        return df if df is not None else pd.DataFrame()

    async def _income_periods(self, ticker: str, period: str) -> list[dict]:
        """One entry per period (newest first). period="quarterly" excludes
        YTD-cumulative columns (see _period_columns); cashflow-derived fields
        are only populated for period="annual" — edgartools' quarterly
        cashflow statement has no true per-quarter column at all (confirmed
        live 2026-08-07), so quarters[] leaves them None rather than
        silently reporting cumulative YTD figures as single-quarter — a
        real, disclosed gap, not fixed here (de-cumulation would need an
        extra fetch + fiscal-year-boundary handling for one field)."""
        income = await self.get_financials(ticker, "income", period)
        if income.empty:
            return []
        cashflow = (
            await self.get_financials(ticker, "cashflow", period)
            if period == "annual"
            else pd.DataFrame()
        )
        columns = _period_columns(income, exclude_ytd=(period == "quarterly"))
        periods = []
        for column in columns:
            # column labels look like "2026-06-27 (Q3)" / "2025-09-27 (FY)" —
            # strip the annotation so period_end is a clean date string,
            # consistent with yfinance.py's normalize_financials() (real
            # inconsistency caught on review: both feed the same
            # NormalizedFinancials.quarters[]/annual[] field).
            row = {"period_end": str(column).split(" ")[0]}
            for field, concept in _INCOME_CONCEPTS.items():
                row[field] = _extract_concept(income, concept, column)
            for field, concept in _CASHFLOW_CONCEPTS.items():
                row[field] = (
                    _extract_concept(cashflow, concept, column) if not cashflow.empty else None
                )
            periods.append(row)
        return periods

    async def normalize_financials(self, ticker: str) -> NormalizedFinancials:
        """Builds NormalizedFinancials (86bbb001k). balance_sheet is the
        single latest quarterly balance (not per-period, per
        NormalizedFinancials' own "latest only" design)."""
        quarters = await self._income_periods(ticker, "quarterly")
        annual = await self._income_periods(ticker, "annual")

        balance = await self.get_financials(ticker, "balance", "quarterly")
        balance_sheet: dict = {}
        if not balance.empty:
            balance_columns = _period_columns(balance, exclude_ytd=False)
            if balance_columns:
                latest_column = balance_columns[0]
                for field, concept in _BALANCE_CONCEPTS.items():
                    balance_sheet[field] = _extract_concept(balance, concept, latest_column)
                balance_sheet["total_debt"] = _sum_concepts(
                    balance, _DEBT_CONCEPTS, latest_column
                )

        return NormalizedFinancials(
            quarters=quarters,
            annual=annual,
            balance_sheet=balance_sheet,
            currency="USD",  # edgartools/SEC EDGAR covers US filers only
        )

    async def get_insider_trading(self, ticker: str, days: int = 90) -> list[dict]:
        """Transactional-level Form 4 data (date, shares, price, insider name) —
        not the aggregate-only data openbb-tmx gives for Canadian stocks."""
        company = await asyncio.to_thread(Company, ticker)
        filings = await asyncio.to_thread(lambda: company.get_filings(form="4").head(20))
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)

        records: list[dict] = []
        for filing in filings:
            try:
                form4 = await asyncio.to_thread(filing.obj)
                df = await asyncio.to_thread(form4.to_dataframe)
            except Exception:
                logger.warning(
                    "edgar_form4_parse_failed",
                    ticker=ticker,
                    accession=filing.accession_no,
                    exc_info=True,
                )
                continue
            if df is None or df.empty or "Date" not in df.columns:
                continue
            for _, row in df[df["Date"] >= cutoff].iterrows():
                records.append(
                    {
                        "date": row["Date"].strftime("%Y-%m-%d"),
                        "shares": float(row["Shares"]) if pd.notna(row["Shares"]) else None,
                        "price": float(row["Price"]) if pd.notna(row["Price"]) else None,
                        "insider_name": row.get("Insider"),
                        "transaction_type": row.get("Transaction Type"),
                        "code": row.get("Code"),
                    }
                )
        return records

    # Out of scope for edgartools — SEC EDGAR has no price, profile, estimates,
    # ratings, peer, earnings-calendar, or dividend data. That's fmp.py's job
    # per the documented routing rule; fake it here rather than fabricating it.
    async def get_price_history(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        raise NotImplementedError("Price history is out of scope for edgartools — use fmp.py")

    async def get_company_info(self, ticker: str) -> dict:
        raise NotImplementedError("Company info is out of scope for edgartools — use fmp.py")

    async def get_analyst_estimates(self, ticker: str) -> dict:
        raise NotImplementedError("Analyst estimates are out of scope for edgartools — use fmp.py")

    async def get_analyst_ratings(self, ticker: str) -> dict:
        raise NotImplementedError("Analyst ratings are out of scope for edgartools — use fmp.py")

    async def get_earnings_surprises(self, ticker: str) -> list[dict]:
        raise NotImplementedError("Earnings surprises are out of scope for edgartools — use fmp.py")

    async def get_peers(self, ticker: str, limit: int = 5) -> list[str]:
        raise NotImplementedError("Peers are out of scope for edgartools — use fmp.py")

    async def get_earnings_calendar(self, ticker: str) -> list[dict]:
        raise NotImplementedError("Earnings calendar is out of scope for edgartools — use fmp.py")

    async def get_dividend_history(self, ticker: str, from_date: str, to_date: str) -> list[dict]:
        raise NotImplementedError("Dividend history is out of scope for edgartools — use fmp.py")
