import asyncio
import os

import pandas as pd
import structlog
from edgar import Company, set_identity

from data.providers.base import StockDataProvider
from data.providers.yfinance import YFinanceDataProvider, correct_alignment

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

# Gap 5b (financial-data-api-research.md) describes a "fewer than 5 key line items"
# XBRL-quality check for small-caps. Deliberately not implemented: live-tested
# against ~25 real small/micro-cap and pre-revenue tickers (including deliberately
# thin filers like pre-revenue biotechs) and it never fired below 3-of-5 on any of
# them — while a literal "all 5 required" reading misfired on JPM, Visa, and P&G
# (mega-caps whose sector legitimately has no GrossProfit/OperatingIncome line,
# not an XBRL tagging problem). The only fallback trigger kept is get_financials()
# returning None outright (no computed Financials object — e.g. no XBRL-tagged
# filings), which is real and cheap to check.


class EdgarToolsDataProvider(StockDataProvider):
    """US-stock fundamentals and insider transactions, sourced directly from SEC
    EDGAR (edgartools) — no ticker whitelist, no tier restriction, unlike FMP's
    free-tier annual-only fundamentals. Covers only the US-stock routing branch:
    prices, profile, estimates, ratings, peers, earnings calendar, and dividends
    are fmp.py's job, not this provider's."""

    def __init__(self, fallback: YFinanceDataProvider | None = None) -> None:
        self._fallback = fallback or YFinanceDataProvider()

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
            return await self._fallback_financials(ticker, statement, period)

        statement_fn = STATEMENT_MAP[key](financials)
        return await asyncio.to_thread(lambda: statement_fn().to_dataframe())

    async def _fallback_financials(self, ticker: str, statement: str, period: str) -> pd.DataFrame:
        df = await self._fallback.get_financials(ticker, statement, period)
        return correct_alignment(df)

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

    async def get_peers(self, ticker: str, limit: int = 5) -> list[str]:
        raise NotImplementedError("Peers are out of scope for edgartools — use fmp.py")

    async def get_earnings_calendar(self, ticker: str) -> list[dict]:
        raise NotImplementedError("Earnings calendar is out of scope for edgartools — use fmp.py")

    async def get_dividend_history(self, ticker: str, from_date: str, to_date: str) -> list[dict]:
        raise NotImplementedError("Dividend history is out of scope for edgartools — use fmp.py")
