import pandas as pd
import pytest
import structlog

from data.providers.base import StockDataProvider
from data.providers.edgartools import EdgarToolsDataProvider


# --- Fakes standing in for the edgar (edgartools) library's object graph ---


class FakeStatement:
    def __init__(self, df: pd.DataFrame | None) -> None:
        self._df = df

    def to_dataframe(self) -> pd.DataFrame | None:
        return self._df


class FakeFinancials:
    def __init__(self, income_df=None, balance_df=None, cashflow_df=None) -> None:
        self._income_df = income_df
        self._balance_df = balance_df
        self._cashflow_df = cashflow_df

    def income_statement(self) -> FakeStatement:
        return FakeStatement(self._income_df)

    def balance_sheet(self) -> FakeStatement:
        return FakeStatement(self._balance_df)

    def cashflow_statement(self) -> FakeStatement:
        return FakeStatement(self._cashflow_df)


class FakeForm4:
    def __init__(self, df: pd.DataFrame | None, raises: bool = False) -> None:
        self._df = df
        self._raises = raises

    def to_dataframe(self) -> pd.DataFrame | None:
        if self._raises:
            raise ValueError("malformed Form 4 XML")
        return self._df


class FakeFiling:
    def __init__(self, accession_no: str, form4_df=None, raises: bool = False) -> None:
        self.accession_no = accession_no
        self._form4_df = form4_df
        self._raises = raises

    def obj(self) -> FakeForm4:
        return FakeForm4(self._form4_df, raises=self._raises)


class FakeFilingsResult(list):
    def head(self, n: int) -> "FakeFilingsResult":
        return FakeFilingsResult(self[:n])


class FakeCompany:
    def __init__(
        self,
        annual=None,
        quarterly=None,
        annual_income_df=None,
        annual_cashflow_df=None,
        fiscal_year_end="1231",
        filings: list[FakeFiling] | None = None,
        filings_by_form: dict[str, list] | None = None,
    ) -> None:
        self._annual = annual  # FakeFinancials for the latest 10-K (FY totals -> ttm)
        self._quarterly = quarterly  # FakeFinancials for the latest 10-Q
        self._annual_income_df = annual_income_df  # high-level multi-period frame
        self._annual_cashflow_df = annual_cashflow_df
        self.fiscal_year_end = fiscal_year_end
        self._filings = filings or []
        self._filings_by_form = filings_by_form or {}

    def get_financials(self) -> FakeFinancials | None:
        return self._annual

    def get_quarterly_financials(self) -> FakeFinancials | None:
        return self._quarterly

    def income_statement(self, periods=4, period="annual", as_dataframe=False):
        return self._annual_income_df

    def cashflow_statement(self, periods=4, period="annual", as_dataframe=False):
        return self._annual_cashflow_df

    def get_filings(self, form: str | None = None, amendments: bool = True) -> FakeFilingsResult:
        if form is not None and form in self._filings_by_form:
            return FakeFilingsResult(self._filings_by_form[form])
        return FakeFilingsResult(self._filings)


def _highlevel_df(rows: list[tuple[str, float | list[float]]], columns: list[str]) -> pd.DataFrame:
    """Concept-INDEXED frame, as Company.income_statement(as_dataframe=True)
    returns it: bare concept names (no 'us-gaap_' prefix), one row per concept,
    no 'dimension' column, one value column per fiscal period ('FY 2025')."""
    data = {}
    for i, col in enumerate(columns):
        data[col] = [(v[i] if isinstance(v, list) else v) for _, v in rows]
    return pd.DataFrame(data, index=[r[0] for r in rows])


# --- Fixtures ---


def _income_df(concepts: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"concept": concepts, "label": concepts})


FULL_INCOME_CONCEPTS = [
    "us-gaap_Revenues",
    "us-gaap_GrossProfit",
    "us-gaap_OperatingIncomeLoss",
    "us-gaap_NetIncomeLoss",
    "us-gaap_EarningsPerShareBasic",
]

# A thin/partial statement — e.g. a bank with no GrossProfit/OperatingIncomeLoss
# line — no longer triggers a fallback (see module docstring in edgartools.py for
# why); it should just pass through as-is.
THIN_INCOME_CONCEPTS = ["us-gaap_Revenues", "us-gaap_NetIncomeLoss"]


@pytest.fixture
def provider(monkeypatch):
    return EdgarToolsDataProvider()


def _patch_company(monkeypatch, company_factory):
    monkeypatch.setattr("data.providers.edgartools.Company", company_factory)


# --- get_financials ---


def test_provider_is_a_stock_data_provider(provider):
    assert isinstance(provider, StockDataProvider)


async def test_get_financials_annual_income_returns_full_dataframe(provider, monkeypatch):
    fake = FakeCompany(annual=FakeFinancials(income_df=_income_df(FULL_INCOME_CONCEPTS)))
    _patch_company(monkeypatch, lambda ticker: fake)

    df = await provider.get_financials("AAPL", "income", "annual")

    assert list(df["concept"]) == FULL_INCOME_CONCEPTS


async def test_get_financials_quarterly_uses_quarterly_accessor(provider, monkeypatch):
    fake = FakeCompany(
        quarterly=FakeFinancials(balance_df=pd.DataFrame({"concept": ["us-gaap_Assets"]}))
    )
    _patch_company(monkeypatch, lambda ticker: fake)

    df = await provider.get_financials("AAPL", "balance", "quarterly")

    assert list(df["concept"]) == ["us-gaap_Assets"]


async def test_get_financials_invalid_statement_raises(provider, monkeypatch):
    _patch_company(monkeypatch, lambda ticker: FakeCompany())

    with pytest.raises(ValueError, match="Invalid statement"):
        await provider.get_financials("AAPL", "notarealstatement", "annual")


async def test_get_financials_thin_statement_passes_through_without_fallback(provider, monkeypatch):
    """A partial income statement (e.g. a bank with no GrossProfit line) is not,
    by itself, a reason to distrust the data — see edgartools.py module docstring."""
    fake = FakeCompany(annual=FakeFinancials(income_df=_income_df(THIN_INCOME_CONCEPTS)))
    _patch_company(monkeypatch, lambda ticker: fake)

    df = await provider.get_financials("JPM", "income", "annual")

    assert list(df["concept"]) == THIN_INCOME_CONCEPTS


async def test_get_financials_none_financials_returns_empty_dataframe(provider, monkeypatch):
    """get_financials()/get_quarterly_financials() can return None outright —
    not just a thin statement. This class no longer falls back to yfinance
    itself (ClickUp 86bb4758g — "provider adapters stay dumb, router owns
    every completeness check") — it returns an empty DataFrame and lets the
    caller (router.py) decide whether to try another source."""
    fake = FakeCompany(annual=None)
    _patch_company(monkeypatch, lambda ticker: fake)

    with structlog.testing.capture_logs() as logs:
        df = await provider.get_financials("NODATA", "income", "annual")

    assert df.empty
    assert any(log["event"] == "edgar_financials_missing" for log in logs)


async def test_get_financials_single_missing_statement_returns_empty_dataframe(
    provider, monkeypatch
):
    """Real gap found via 86bbb001k: the Financials container can exist
    while one specific statement isn't XBRL-tagged (e.g. a filer missing a
    cash flow statement) — to_dataframe() returns None in that case, which
    used to propagate straight through get_financials(), violating its own
    -> pd.DataFrame contract."""
    fake = FakeCompany(annual=FakeFinancials(income_df=_income_df(FULL_INCOME_CONCEPTS)))
    _patch_company(monkeypatch, lambda ticker: fake)

    df = await provider.get_financials("AAPL", "cashflow", "annual")

    assert isinstance(df, pd.DataFrame)
    assert df.empty


# --- normalize_financials (86bbb001k) ---


def _concept_df(rows: list[tuple[str, str, bool, float]], period_col: str) -> pd.DataFrame:
    """rows: (concept, label, dimension, value). Mirrors edgartools'
    real to_dataframe() shape — a flat table with concept/label/dimension
    columns plus one column per period, not index-based like yfinance."""
    return pd.DataFrame(
        {
            "concept": [r[0] for r in rows],
            "label": [r[1] for r in rows],
            "dimension": [r[2] for r in rows],
            period_col: [r[3] for r in rows],
        }
    )


async def test_normalize_financials_uses_consolidated_row_not_segment_breakdown(
    provider, monkeypatch
):
    """The raw 10-Q frame (quarters[]) repeats a concept once for the
    consolidated total and again per segment/product breakdown — dimension
    == False is the consolidated row. A naive first-match-by-concept lookup
    with rows ordered breakdown-first would pick a segment slice instead."""
    income = _concept_df(
        [
            ("us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax", "iPhone", True, 500.0),
            (
                "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax",
                "Net sales",
                False,
                1000.0,
            ),
        ],
        "2026-06-27 (Q3)",
    )
    fake = FakeCompany(quarterly=FakeFinancials(income_df=income))
    _patch_company(monkeypatch, lambda ticker: fake)

    result = await provider.normalize_financials("AAPL")

    assert result.quarters[0]["revenue"] == 1000.0


async def test_normalize_financials_period_end_strips_quarter_annotation(provider, monkeypatch):
    """Real inconsistency caught on review: edgartools' column labels carry
    a "(Q3)"/"(FY)" annotation that yfinance.py's normalize_financials()
    doesn't produce (it emits a clean ISO date) — both feed the same
    NormalizedFinancials.quarters[]/annual[] period_end field, so edgartools
    must strip it too rather than leaking the annotation through."""
    income = _concept_df(
        [("us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax", "Net sales", False, 300.0)],
        "2026-06-27 (Q3)",
    )
    fake = FakeCompany(quarterly=FakeFinancials(income_df=income))
    _patch_company(monkeypatch, lambda ticker: fake)

    result = await provider.normalize_financials("AAPL")

    assert result.quarters[0]["period_end"] == "2026-06-27"


async def test_normalize_financials_quarterly_excludes_ytd_columns(provider, monkeypatch):
    """Real, confirmed finding: get_quarterly_financials()'s income statement
    exposes both a true single-quarter column ("(Q3)") and a
    cumulative-since-fiscal-year-start column ("(YTD)") for the same period
    end — a standard 10-Q convention. quarters[] must use the Q-only
    column, not silently report the cumulative figure as single-quarter."""
    income = _concept_df(
        [("us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax", "Net sales", False, 300.0)],
        "2026-06-27 (Q3)",
    )
    income["2026-06-27 (YTD)"] = [900.0]
    fake = FakeCompany(quarterly=FakeFinancials(income_df=income))
    _patch_company(monkeypatch, lambda ticker: fake)

    result = await provider.normalize_financials("AAPL")

    assert len(result.quarters) == 1
    assert result.quarters[0]["revenue"] == 300.0  # the Q-only figure, not 900 (YTD)


async def test_normalize_financials_quarterly_cashflow_fields_are_none(provider, monkeypatch):
    """Real, disclosed gap: edgartools' quarterly cashflow statement has no
    true per-quarter column at all (every column is "(YTD)") — quarters[]
    leaves cashflow-derived fields None rather than reporting a cumulative
    YTD figure as if it were single-quarter."""
    income = _concept_df(
        [("us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax", "Net sales", False, 300.0)],
        "2026-06-27 (Q3)",
    )
    cashflow = _concept_df(
        [("us-gaap_PaymentsOfDividends", "Dividends paid", False, -60.0)],
        "2026-06-27 (YTD)",
    )
    fake = FakeCompany(quarterly=FakeFinancials(income_df=income, cashflow_df=cashflow))
    _patch_company(monkeypatch, lambda ticker: fake)

    result = await provider.normalize_financials("AAPL")

    assert result.quarters[0]["dividends_paid"] is None
    assert result.quarters[0]["operating_cash_flow"] is None


async def test_normalize_financials_annual_from_highlevel_multiperiod_api(provider, monkeypatch):
    """annual[] comes from Company.income_statement(period='annual') — the
    high-level multi-period frame (concept-indexed, bare names, one row per
    concept). Deep enough for _cagr; income and cash-flow both populate;
    trimmed at the first fiscal-year gap."""
    cols = ["FY 2025", "FY 2024", "FY 2023", "FY 2022"]
    inc = _highlevel_df(
        [
            ("Revenues", [1000.0, 900.0, 800.0, 700.0]),
            ("NetIncomeLoss", [120.0, 100.0, 90.0, 80.0]),
        ],
        cols,
    )
    cf = _highlevel_df([("PaymentsOfDividendsCommonStock", [-60.0, -55.0, -50.0, -45.0])], cols)
    fake = FakeCompany(annual_income_df=inc, annual_cashflow_df=cf)
    _patch_company(monkeypatch, lambda ticker: fake)

    result = await provider.normalize_financials("AAPL")

    assert [a["period_end"][:4] for a in result.annual] == ["2025", "2024", "2023", "2022"]
    assert result.annual[0]["revenue"] == 1000.0
    assert result.annual[0]["dividends_paid"] == -60.0


async def test_normalize_financials_annual_trims_at_fiscal_year_gap(provider, monkeypatch):
    cols = ["FY 2025", "FY 2024", "FY 2022"]  # 2023 missing
    inc = _highlevel_df([("Revenues", [1000.0, 900.0, 700.0])], cols)
    fake = FakeCompany(annual_income_df=inc)
    _patch_company(monkeypatch, lambda ticker: fake)

    result = await provider.normalize_financials("AAPL")

    assert [a["period_end"][:4] for a in result.annual] == ["2025", "2024"]


async def test_normalize_financials_total_debt_sums_current_and_noncurrent(provider, monkeypatch):
    """total_debt has no single XBRL concept — derived by summing current +
    non-current term debt, confirmed live against AAPL's real balance sheet."""
    balance = _concept_df(
        [
            ("us-gaap_Assets", "Total assets", False, 5000.0),
            ("us-gaap_LongTermDebtCurrent", "Term debt, current", False, 100.0),
            ("us-gaap_LongTermDebtNoncurrent", "Term debt, non-current", False, 700.0),
        ],
        "2026-06-27",
    )
    fake = FakeCompany(quarterly=FakeFinancials(balance_df=balance))
    _patch_company(monkeypatch, lambda ticker: fake)

    result = await provider.normalize_financials("AAPL")

    assert result.balance_sheet["total_debt"] == 800.0
    assert result.balance_sheet["total_assets"] == 5000.0


async def test_normalize_financials_none_financials_returns_empty_periods(provider, monkeypatch):
    fake = FakeCompany(annual=None, quarterly=None)
    _patch_company(monkeypatch, lambda ticker: fake)

    result = await provider.normalize_financials("NODATA")

    assert result.quarters == []
    assert result.annual == []
    assert result.balance_sheet == {}
    assert result.currency == "USD"
    assert result.ttm is None


# --- ttm via YTD algebra (86bbxuj9e) ---


def _q_income_with_ytd(concept_ytds: list[tuple[str, float, float]]) -> pd.DataFrame:
    """A 10-Q income frame with a discrete (Q3) column plus current and
    prior-year (YTD) columns. concept_ytds: (concept, ytd_current, ytd_prior)."""
    return pd.DataFrame(
        {
            "concept": [c for c, _, _ in concept_ytds],
            "label": [c for c, _, _ in concept_ytds],
            "dimension": [False] * len(concept_ytds),
            "2026-06-27 (Q3)": [cur / 3 for _, cur, _ in concept_ytds],  # rough per-quarter
            "2026-06-27 (YTD)": [cur for _, cur, _ in concept_ytds],
            "2025-06-28 (YTD)": [pri for _, _, pri in concept_ytds],
        }
    )


async def test_compute_ttm_ytd_algebra(provider, monkeypatch):
    """ttm[field] = FY_prior_full + YTD_current - YTD_prior, from the latest
    10-K's FY column and the latest 10-Q's two YTD columns."""
    fy = _concept_df(
        [
            ("us-gaap_NetIncomeLoss", "Net income", False, 1000.0),
            ("us-gaap_Revenues", "Revenue", False, 4000.0),
        ],
        "2025-09-27 (FY)",
    )
    q_income = _q_income_with_ytd([("us-gaap_NetIncomeLoss", 600.0, 550.0), ("us-gaap_Revenues", 2400.0, 2200.0)])
    fake = FakeCompany(
        annual=FakeFinancials(income_df=fy),
        quarterly=FakeFinancials(income_df=q_income),
    )
    _patch_company(monkeypatch, lambda ticker: fake)

    result = await provider.normalize_financials("AAPL")

    assert result.ttm["net_income"] == pytest.approx(1000.0 + 600.0 - 550.0)
    assert result.ttm["revenue"] == pytest.approx(4000.0 + 2400.0 - 2200.0)


async def test_compute_ttm_falls_to_fy_when_no_newer_10q(provider, monkeypatch):
    """No latest 10-Q -> the 10-K's FY total IS the trailing twelve months."""
    fy = _concept_df([("us-gaap_NetIncomeLoss", "Net income", False, 1000.0)], "2025-09-27 (FY)")
    fake = FakeCompany(annual=FakeFinancials(income_df=fy), quarterly=None)
    _patch_company(monkeypatch, lambda ticker: fake)

    result = await provider.normalize_financials("AAPL")

    assert result.ttm["net_income"] == 1000.0


async def test_compute_ttm_field_none_when_a_component_missing(provider, monkeypatch):
    """With a 10-Q present, a field needs all three of FY / YTD_cur / YTD_prior
    to resolve — a partial set gives None for that field, not a guess."""
    fy = _concept_df(
        [
            ("us-gaap_NetIncomeLoss", "Net income", False, 1000.0),
            ("us-gaap_Revenues", "Revenue", False, 4000.0),
        ],
        "2025-09-27 (FY)",
    )
    # 10-Q YTD has net income but NOT revenue
    q_income = _q_income_with_ytd([("us-gaap_NetIncomeLoss", 600.0, 550.0)])
    fake = FakeCompany(
        annual=FakeFinancials(income_df=fy), quarterly=FakeFinancials(income_df=q_income)
    )
    _patch_company(monkeypatch, lambda ticker: fake)

    result = await provider.normalize_financials("AAPL")

    assert result.ttm["net_income"] == pytest.approx(1000.0 + 600.0 - 550.0)
    assert result.ttm["revenue"] is None  # FY revenue exists but no YTD revenue to roll it forward


# --- concept fallback lists (86bbxuj9e / 86bbq04wm) ---


async def test_normalize_financials_revenue_concept_fallback(provider, monkeypatch):
    """XOM/JPM tag revenue as 'Revenues', not 'RevenueFromContract...' — the
    fallback list resolves it (both quarters[] and the high-level annual)."""
    q_income = _concept_df(
        [("us-gaap_Revenues", "Total revenues", False, 300.0)], "2026-06-27 (Q3)"
    )
    ann = _highlevel_df([("Revenues", [1200.0, 1100.0])], ["FY 2025", "FY 2024"])
    fake = FakeCompany(quarterly=FakeFinancials(income_df=q_income), annual_income_df=ann)
    _patch_company(monkeypatch, lambda ticker: fake)

    result = await provider.normalize_financials("XOM")

    assert result.quarters[0]["revenue"] == 300.0
    assert result.annual[0]["revenue"] == 1200.0


async def test_normalize_financials_net_income_common_picked_up_for_banks(provider, monkeypatch):
    q_income = _concept_df(
        [
            ("us-gaap_NetIncomeLoss", "Net income", False, 100.0),
            ("us-gaap_NetIncomeLossAvailableToCommonStockholdersBasic", "NI to common", False, 94.0),
        ],
        "2026-06-27 (Q3)",
    )
    fake = FakeCompany(quarterly=FakeFinancials(income_df=q_income))
    _patch_company(monkeypatch, lambda ticker: fake)

    result = await provider.normalize_financials("JPM")

    assert result.quarters[0]["net_income_common"] == 94.0


async def test_normalize_financials_no_net_income_common_concept_is_none(provider, monkeypatch):
    q_income = _concept_df(
        [("us-gaap_NetIncomeLoss", "Net income", False, 100.0)], "2026-06-27 (Q3)"
    )
    fake = FakeCompany(quarterly=FakeFinancials(income_df=q_income))
    _patch_company(monkeypatch, lambda ticker: fake)

    result = await provider.normalize_financials("AAPL")

    assert result.quarters[0]["net_income_common"] is None


# --- get_insider_trading ---


def _form4_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


async def test_get_insider_trading_filters_by_days_and_maps_fields(provider, monkeypatch):
    now = pd.Timestamp.now()
    recent_row = {
        "Date": now - pd.Timedelta(days=5),
        "Shares": 100,
        "Price": 25.5,
        "Insider": "Jane Doe",
        "Transaction Type": "Purchase",
        "Code": "P",
    }
    old_row = {**recent_row, "Date": now - pd.Timedelta(days=400), "Insider": "Old Insider"}
    filing = FakeFiling("0001-1", form4_df=_form4_df([recent_row, old_row]))
    fake = FakeCompany(filings=[filing])
    _patch_company(monkeypatch, lambda ticker: fake)

    result = await provider.get_insider_trading("AAPL", days=90)

    assert len(result) == 1
    assert result[0]["insider_name"] == "Jane Doe"
    assert result[0]["shares"] == 100.0
    assert result[0]["price"] == 25.5
    assert result[0]["transaction_type"] == "Purchase"


async def test_get_insider_trading_skips_unparseable_filing_and_logs(provider, monkeypatch):
    now = pd.Timestamp.now()
    good_row = {
        "Date": now,
        "Shares": 50,
        "Price": 10.0,
        "Insider": "Good Filing",
        "Transaction Type": "Sale",
        "Code": "S",
    }
    broken_filing = FakeFiling("0001-broken", raises=True)
    good_filing = FakeFiling("0001-good", form4_df=_form4_df([good_row]))
    fake = FakeCompany(filings=[broken_filing, good_filing])
    _patch_company(monkeypatch, lambda ticker: fake)

    with structlog.testing.capture_logs() as logs:
        result = await provider.get_insider_trading("AAPL", days=90)

    assert len(result) == 1
    assert result[0]["insider_name"] == "Good Filing"
    assert any(
        log["event"] == "edgar_form4_parse_failed" and log["accession"] == "0001-broken"
        for log in logs
    )


async def test_get_insider_trading_no_filings_returns_empty_list(provider, monkeypatch):
    fake = FakeCompany(filings=[])
    _patch_company(monkeypatch, lambda ticker: fake)

    result = await provider.get_insider_trading("AAPL", days=90)

    assert result == []


# --- 5 US stocks of varying market cap, per ticket's unit-test requirement ---


@pytest.mark.parametrize(
    "ticker,concepts",
    [
        ("AAPL", FULL_INCOME_CONCEPTS),  # mega-cap
        ("JPM", THIN_INCOME_CONCEPTS),  # mega-cap bank — no COGS model, partial by design
        ("DELL", FULL_INCOME_CONCEPTS),  # mid-cap
        ("PLUG", THIN_INCOME_CONCEPTS),  # small-cap
        ("GRPN", ["us-gaap_Revenues"]),  # micro-cap, sparse statement
    ],
)
async def test_get_financials_across_market_caps(provider, monkeypatch, ticker, concepts):
    fake = FakeCompany(annual=FakeFinancials(income_df=_income_df(concepts)))
    _patch_company(monkeypatch, lambda t, _fake=fake: _fake)

    df = await provider.get_financials(ticker, "income", "annual")

    assert list(df["concept"]) == concepts


# --- Out-of-scope methods ---


@pytest.mark.parametrize(
    "method,args",
    [
        ("get_price_history", ("AAPL", "1y", "1d")),
        ("get_company_info", ("AAPL",)),
        ("get_analyst_estimates", ("AAPL",)),
        ("get_analyst_ratings", ("AAPL",)),
        ("get_earnings_surprises", ("AAPL",)),
        ("get_peers", ("AAPL",)),
        ("get_earnings_calendar", ("AAPL",)),
        ("get_dividend_history", ("AAPL", "2024-01-01", "2024-12-31")),
    ],
)
async def test_out_of_scope_methods_raise_not_implemented(provider, method, args):
    with pytest.raises(NotImplementedError):
        await getattr(provider, method)(*args)
