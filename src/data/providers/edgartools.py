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

# XBRL concept maps for normalize_financials() (86bbb001k, deepened in 86bbxuj9e).
# Each field maps to an ORDERED list of candidate concept names — first present wins.
# GAAP allows more than one valid tag for many line items and it varies by filer
# (verified live: XOM/JPM tag revenue as "Revenues" not "RevenueFromContract...",
# HD tags capex as "PaymentsToAcquireProductiveAssets"). Names are stored WITHOUT
# the "us-gaap_" prefix; _extract_concept() tries both the prefixed form (the raw
# 10-Q/10-K XBRL frames) and the bare form (the high-level multi-period annual
# frame). A field whose every candidate is absent gets None — the same graceful
# degradation as a genuinely missing row (e.g. oil majors have no OperatingIncomeLoss).
_INCOME_CONCEPTS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "cost_of_revenue": ["CostOfGoodsAndServicesSold", "CostOfRevenue"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    # income to common after preferred dividends — present for preferred-heavy
    # filers (banks), absent otherwise; fundamentals.py falls back to net_income.
    "net_income_common": ["NetIncomeLossAvailableToCommonStockholdersBasic"],
    "tax_expense": ["IncomeTaxExpenseBenefit"],
    "eps": ["EarningsPerShareDiluted"],
    "shares_outstanding": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
    ],
    "interest_expense": [
        "InterestExpense",
        "InterestExpenseNonoperating",
        "InterestAndDebtExpense",
    ],
}
_CASHFLOW_CONCEPTS = {
    "depreciation_amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationDepletionAndAmortizationPropertyPlantAndEquipment",
        "Depreciation",
        "CostOfServicesDepreciation",
    ],
    "dividends_paid": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capital_expenditures": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
}
_BALANCE_CONCEPTS = {
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "total_equity": [
        "StockholdersEquity",
        # includes non-controlling interests — close enough for D/E and ROE,
        # and the only equity line PG / CAT / UNH tag.
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "cash_and_equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashCashEquivalentsAndShortTermInvestments",
    ],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
}
# total_debt has no single XBRL concept and the tags overlap (a filer can tag
# both a noncurrent portion AND a combined total), so _total_debt() buckets
# them: (noncurrent + current), else a combined-total tag, plus short-term
# borrowings on top. First match within each bucket — never sum across.
_DEBT_NONCURRENT = ("LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations")
_DEBT_CURRENT = (
    "LongTermDebtCurrent",
    "LongTermDebtAndCapitalLeaseObligationsCurrent",
    "DebtCurrent",
)
_DEBT_COMBINED_TOTAL = (
    "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
    "DebtLongtermAndShorttermCombinedAmount",
    "LongTermDebt",  # taxonomy: includes both current and noncurrent
)
_DEBT_SHORT_TERM = ("ShortTermBorrowings",)

# Fields carried on NormalizedFinancials.ttm (86bbxuj9e). No eps — see finding 5
# in the plan: US per-quarter/Q4 EPS is unreliable, so P/E takes the net-income path.
_TTM_FIELDS = (
    "revenue",
    "cost_of_revenue",
    "net_income",
    "net_income_common",
    "operating_income",
    "operating_cash_flow",
    "capital_expenditures",
    "depreciation_amortization",
    "dividends_paid",
)


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


def _column_date(column) -> "pd.Timestamp | None":
    """The date embedded in a statement column label ('2026-06-27 (YTD)')."""
    if column is None:
        return None
    try:
        return pd.Timestamp(str(column).split(" ")[0])
    except (ValueError, TypeError):
        return None


def _sorted_ytd_columns(df: pd.DataFrame) -> list:
    """The '(YTD)' value columns, newest period-end first — the two most recent
    are the current and prior-year year-to-date figures a 10-Q reports."""
    if df is None or df.empty:
        return []
    ytd = [c for c in df.columns if c not in _META_COLUMNS and "(YTD)" in str(c)]
    return sorted(ytd, key=lambda c: _column_date(c) or pd.Timestamp.min, reverse=True)


def _fiscal_month_day(company) -> tuple[int, int]:
    """(month, day) from Company.fiscal_year_end ('MMDD'); Dec 31 fallback.
    Only used to give annual[] period_end an approximate date — nothing
    computes on it."""
    raw = getattr(company, "fiscal_year_end", None)
    if isinstance(raw, str) and len(raw) == 4 and raw.isdigit():
        mm, dd = int(raw[:2]), int(raw[2:])
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            return mm, dd
    return 12, 31


def _candidate_names(concept: str):
    """Both forms of an XBRL concept name: bare (the high-level multi-period
    frame indexes rows this way) and us-gaap_-prefixed (the raw 10-Q/10-K
    to_dataframe() frames). Try bare first — it's also how a non-us-gaap
    namespace would appear."""
    if concept.startswith("us-gaap_"):
        return (concept, concept.removeprefix("us-gaap_"))
    return (concept, f"us-gaap_{concept}")


def _extract_concept(df: pd.DataFrame, concepts, column) -> float | None:
    """First present value across an ordered list of candidate concepts, from
    either frame shape:

    - raw 10-Q/10-K XBRL (`to_dataframe()`): 'concept' is a column, rows repeat
      once per segment/product breakdown, and 'dimension'==False marks the
      consolidated row. First non-dimensional match wins (AAPL labels a
      retained-earnings roll-forward row 'NetIncomeLoss' too).
    - high-level multi-period frame (`income_statement(as_dataframe=True)`):
      'concept' is the index, one consolidated row per concept, no 'dimension'.

    `concepts` may be a single string or a list (first candidate that resolves
    to a non-NaN value wins). Prefix-agnostic via _candidate_names()."""
    if df is None or df.empty or column not in df.columns:
        return None
    if isinstance(concepts, str):
        concepts = [concepts]
    concept_is_column = "concept" in df.columns
    has_dimension = "dimension" in df.columns
    for concept in concepts:
        for name in _candidate_names(concept):
            if concept_is_column:
                rows = df[df["concept"] == name]
                if has_dimension:
                    rows = rows[rows["dimension"] == False]  # noqa: E712
                if rows.empty:
                    continue
                value = rows.iloc[0][column]
            else:
                if name not in df.index:
                    continue
                value = df.loc[name, column]
                if isinstance(value, pd.Series):  # duplicate index label
                    value = value.iloc[0]
            if not pd.isna(value):
                return float(value)
    return None


def _total_debt(df: pd.DataFrame, column) -> float | None:
    """Total debt, avoiding the double-count that a flat sum over overlapping
    XBRL debt tags would cause. Base = (noncurrent + current), or a
    combined-total tag when a filer only reports one; plus short-term
    borrowings. None only when the filer tags no debt at all."""
    noncurrent = _extract_concept(df, list(_DEBT_NONCURRENT), column)
    current = _extract_concept(df, list(_DEBT_CURRENT), column)
    if noncurrent is not None or current is not None:
        base = (noncurrent or 0.0) + (current or 0.0)
    else:
        base = _extract_concept(df, list(_DEBT_COMBINED_TOTAL), column)
    short_term = _extract_concept(df, list(_DEBT_SHORT_TERM), column)
    if base is None:
        return short_term
    return base + (short_term or 0.0)


def _fy_columns(df: pd.DataFrame) -> list:
    """A raw 10-K statement's (FY) value columns, most recent fiscal-year end
    first — don't trust to_dataframe() column order for the [0] pick."""
    if df is None or df.empty:
        return []
    fy = [c for c in df.columns if c not in _META_COLUMNS and "(FY)" in str(c)]
    return sorted(fy, key=lambda c: _column_date(c) or pd.Timestamp.min, reverse=True)


def _quarters_from(income: pd.DataFrame) -> list[dict]:
    """quarters[] from the latest 10-Q's income statement: the discrete
    single-quarter column(s), never the YTD ones. Cash-flow fields stay None —
    edgartools' quarterly cash-flow statement is YTD-only, no per-quarter
    column. Newest first."""
    if income is None or income.empty:
        return []
    rows = []
    for column in _period_columns(income, exclude_ytd=True):
        row = {"period_end": str(column).split(" ")[0]}
        for field, concepts in _INCOME_CONCEPTS.items():
            row[field] = _extract_concept(income, concepts, column)
        for field in _CASHFLOW_CONCEPTS:
            row[field] = None
        rows.append(row)
    return rows


def _annual_periods(inc: pd.DataFrame, cf: pd.DataFrame, fiscal_end: tuple[int, int]) -> list[dict]:
    """annual[] from the high-level multi-period frame, most recent fiscal year
    first, trimmed at the first fiscal-year gap so _cagr's fixed offset stays
    honest. period_end is an approximate date from the fiscal-year-end month/day
    — nothing computes on it."""
    if inc is None or inc.empty:
        return []
    mm, dd = fiscal_end
    fy_cols = sorted(
        (c for c in inc.columns if str(c).startswith("FY ")),
        key=lambda c: int(str(c).split()[1]),
        reverse=True,
    )
    rows, prev_year = [], None
    for column in fy_cols:
        year = int(str(column).split()[1])
        if prev_year is not None and year != prev_year - 1:
            break
        prev_year = year
        row = {"period_end": f"{year:04d}-{mm:02d}-{dd:02d}"}
        for field, concepts in _INCOME_CONCEPTS.items():
            row[field] = _extract_concept(inc, concepts, column)
        for field, concepts in _CASHFLOW_CONCEPTS.items():
            row[field] = _extract_concept(cf, concepts, column)
        rows.append(row)
    return rows


def _latest_balance(balance: pd.DataFrame) -> dict:
    """balance_sheet dict from the latest 10-Q's raw-XBRL balance sheet (the
    high-level canonical view drops debt line items)."""
    out: dict = {}
    if balance is None or balance.empty:
        return out
    columns = _period_columns(balance, exclude_ytd=False)
    if not columns:
        return out
    column = columns[0]
    for field, concepts in _BALANCE_CONCEPTS.items():
        out[field] = _extract_concept(balance, concepts, column)
    out["total_debt"] = _total_debt(balance, column)
    # not every filer tags us-gaap_Liabilities; assets - equity is exact
    assets, equity = out.get("total_assets"), out.get("total_equity")
    if out.get("total_liabilities") is None and assets is not None and equity is not None:
        out["total_liabilities"] = assets - equity
    return out


def _compute_ttm(
    ki: pd.DataFrame, kc: pd.DataFrame, qi: pd.DataFrame, qc: pd.DataFrame
) -> dict | None:
    """Trailing-twelve-month aggregates by YTD algebra:

        ttm[field] = FY_prior_full + YTD_current - YTD_prior_year

    FY_prior_full is the latest 10-K's own (FY) column (from the 10-K
    directly, not a labelled multi-period frame, to dodge edgartools'
    fiscal-year label ambiguity). YTD_current / YTD_prior are the latest
    10-Q's two "(YTD)" columns (the 10-Q carries the prior-year YTD as its
    comparative). Per field, None unless all three resolve.

    The algebra is only used when the 10-Q's YTD provably extends past the
    10-K's fiscal-year end; otherwise (no 10-Q, a stale 10-Q, Q4 season, or
    unparseable dates) ttm = FY directly. Returns None when neither filing
    yields anything."""
    if (ki is None or ki.empty) and (qi is None or qi.empty):
        return None

    fy_i_col = next(iter(_fy_columns(ki)), None)
    fy_c_col = next(iter(_fy_columns(kc)), None)
    ytd_i = _sorted_ytd_columns(qi)
    ytd_c = _sorted_ytd_columns(qc)

    fy_end = _column_date(fy_i_col)
    ytd_end = _column_date(ytd_i[0]) if ytd_i else None
    can_roll_forward = bool(len(ytd_i) >= 2 and fy_end and ytd_end and ytd_end > fy_end)

    ttm: dict = {}
    for field in _TTM_FIELDS:
        in_income = field in _INCOME_CONCEPTS
        concepts = (_INCOME_CONCEPTS if in_income else _CASHFLOW_CONCEPTS)[field]
        fy_df, fy_col = (ki, fy_i_col) if in_income else (kc, fy_c_col)
        q_df, q_ytd = (qi, ytd_i) if in_income else (qc, ytd_c)

        fy_val = _extract_concept(fy_df, concepts, fy_col) if fy_col else None
        # dividends_paid is cash-timing sensitive — a payment that shifts across
        # a quarter boundary breaks the YTD subtraction (KO: 2 payments in the
        # current YTD, 1 in the prior). It's also the slowest-changing line, so
        # the last full fiscal year is the better estimate.
        if field == "dividends_paid" or not can_roll_forward or len(q_ytd) < 2:
            ttm[field] = fy_val
            continue
        cur = _extract_concept(q_df, concepts, q_ytd[0])
        prior = _extract_concept(q_df, concepts, q_ytd[1])
        ttm[field] = (
            fy_val + cur - prior
            if fy_val is not None and cur is not None and prior is not None
            else None
        )
    return ttm if any(v is not None for v in ttm.values()) else None

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

    async def normalize_financials(self, ticker: str) -> NormalizedFinancials:
        """Builds NormalizedFinancials (86bbb001k, deepened in 86bbxuj9e).

        - quarters[]: the latest 10-Q's discrete quarter + its prior-year
          comparative (2 entries) — for point-in-time metrics. Quarterly
          cash-flow fields stay None (edgartools' quarterly cash-flow
          statement is YTD-only, no discrete column).
        - annual[]: up to 8 fiscal years from the high-level multi-period
          statement API (Company.income_statement(period='annual')), which
          is reliable where the quarterly one is not — for CAGR / YoY.
        - balance_sheet: the latest quarterly balance, raw XBRL (the
          high-level balance view drops debt line items).
        - ttm: trailing-twelve-month aggregates by YTD algebra (see
          _compute_ttm) — the depth fix that lets _ttm consumers (P/E,
          P/S, PEG, ROE, payout, FCF) resolve for every filer."""
        company = await asyncio.to_thread(Company, ticker)
        qf = await asyncio.to_thread(company.get_quarterly_financials)  # latest 10-Q
        kf = await asyncio.to_thread(company.get_financials)  # latest 10-K

        # Fetch each statement DataFrame once (cheap to reuse, not to re-parse).
        qi = await self._statement_df(qf, "income")
        qc = await self._statement_df(qf, "cashflow")
        qb = await self._statement_df(qf, "balance")
        ki = await self._statement_df(kf, "income")
        kc = await self._statement_df(kf, "cashflow")
        annual_inc, annual_cf = await self._highlevel_annual(company)

        return NormalizedFinancials(
            quarters=_quarters_from(qi),
            annual=_annual_periods(annual_inc, annual_cf, _fiscal_month_day(company)),
            balance_sheet=_latest_balance(qb),
            currency="USD",  # edgartools/SEC EDGAR covers US filers only
            ttm=_compute_ttm(ki, kc, qi, qc),
        )

    @staticmethod
    async def _statement_df(financials, key: str) -> pd.DataFrame:
        """(financials.income_statement | balance_sheet | cashflow_statement)()
        .to_dataframe(), off the event loop, None/error -> empty (a Financials
        container can exist while one statement isn't XBRL-tagged)."""
        if financials is None:
            return pd.DataFrame()
        try:
            accessor = STATEMENT_MAP[key](financials)
            df = await asyncio.to_thread(lambda: accessor().to_dataframe())
        except Exception:
            logger.warning("edgar_statement_df_failed", statement=key, exc_info=True)
            return pd.DataFrame()
        return df if df is not None else pd.DataFrame()

    @staticmethod
    async def _highlevel_annual(company) -> tuple["pd.DataFrame | None", "pd.DataFrame | None"]:
        """Company.income_statement / cashflow_statement(period='annual') — the
        multi-period canonical frame. Degrades to (None, None) on any library
        error rather than failing normalize_financials."""
        try:
            inc = await asyncio.to_thread(
                lambda: company.income_statement(periods=8, period="annual", as_dataframe=True)
            )
            cf = await asyncio.to_thread(
                lambda: company.cashflow_statement(periods=8, period="annual", as_dataframe=True)
            )
        except Exception:
            logger.warning("edgar_highlevel_annual_failed", exc_info=True)
            return None, None
        return inc, cf

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
