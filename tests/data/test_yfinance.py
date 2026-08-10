import pandas as pd
import pytest

from data.providers.yfinance import YFinanceDataProvider


class FakeTicker:
    """Stands in for yf.Ticker(ticker) — exposes exactly the attributes
    yfinance.py reads (.info, .quarterly_financials, .financials, etc.)."""

    def __init__(
        self,
        info: dict | None = None,
        quarterly_financials: pd.DataFrame | None = None,
        financials: pd.DataFrame | None = None,
        quarterly_balance_sheet: pd.DataFrame | None = None,
        balance_sheet: pd.DataFrame | None = None,
        quarterly_cash_flow: pd.DataFrame | None = None,
        cashflow: pd.DataFrame | None = None,
        dividends: pd.Series | None = None,
    ) -> None:
        self.info = info or {}
        self.quarterly_financials = _empty_if_none(quarterly_financials)
        self.financials = _empty_if_none(financials)
        self.quarterly_balance_sheet = _empty_if_none(quarterly_balance_sheet)
        self.balance_sheet = _empty_if_none(balance_sheet)
        self.quarterly_cash_flow = _empty_if_none(quarterly_cash_flow)
        self.cashflow = _empty_if_none(cashflow)
        self.dividends = dividends if dividends is not None else pd.Series(dtype=float)


def _empty_if_none(df: pd.DataFrame | None) -> pd.DataFrame:
    return df if df is not None else pd.DataFrame()


@pytest.fixture
def provider():
    return YFinanceDataProvider()


def _patch_ticker(monkeypatch, ticker_factory) -> None:
    monkeypatch.setattr("data.providers.yfinance.yf.Ticker", ticker_factory)


# --- get_company_info ---


async def test_get_company_info_maps_to_normalized_shape(provider, monkeypatch):
    info = {
        "longName": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "marketCap": 3_000_000_000_000,
        "currency": "USD",
        "country": "United States",
        "exchange": "NMS",
    }
    _patch_ticker(monkeypatch, lambda ticker: FakeTicker(info=info))

    result = await provider.get_company_info("AAPL")

    assert result == {
        "name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "market_cap": 3_000_000_000_000,
        "currency": "USD",
        "country": "United States",
        "primary_exchange": "NMS",
    }


async def test_get_company_info_missing_fields_default_gracefully(provider, monkeypatch):
    _patch_ticker(monkeypatch, lambda ticker: FakeTicker(info={}))

    result = await provider.get_company_info("ZZZZ")

    assert result["name"] == ""
    assert result["market_cap"] is None


# --- normalize_financials ---


def _period_end() -> pd.Timestamp:
    return pd.Timestamp.now().normalize()


async def test_normalize_financials_maps_income_and_cashflow_rows(provider, monkeypatch):
    period = _period_end()
    income = pd.DataFrame(
        {period: [1000.0, 900.0, 300.0, 250.0, 2.5, 50.0, 400.0]},
        index=[
            "Total Revenue",
            "Cost Of Revenue",
            "Operating Income",
            "Net Income",
            "Diluted EPS",
            "Tax Provision",
            "Diluted Average Shares",
        ],
    )
    cashflow = pd.DataFrame(
        {period: [120.0, 60.0, -40.0, 500.0]},
        index=[
            "Depreciation And Amortization",
            "Cash Dividends Paid",
            "Capital Expenditure",
            "Operating Cash Flow",
        ],
    )
    balance = pd.DataFrame(
        {period: [5000.0, 3000.0, 2000.0, 1000.0, 800.0, 1500.0, 700.0]},
        index=[
            "Total Assets",
            "Total Liabilities Net Minority Interest",
            "Stockholders Equity",
            "Total Debt",
            "Cash And Cash Equivalents",
            "Current Assets",
            "Current Liabilities",
        ],
    )

    def fake_ticker(ticker):
        return FakeTicker(
            info={"currency": "USD"},
            quarterly_financials=income,
            financials=income,
            quarterly_cash_flow=cashflow,
            cashflow=cashflow,
            quarterly_balance_sheet=balance,
            balance_sheet=balance,
        )

    _patch_ticker(monkeypatch, fake_ticker)

    result = await provider.normalize_financials("AAPL")

    assert result.currency == "USD"
    assert len(result.quarters) == 1
    q = result.quarters[0]
    assert q["revenue"] == 1000.0
    assert q["cost_of_revenue"] == 900.0
    assert q["operating_income"] == 300.0
    assert q["net_income"] == 250.0
    assert q["eps"] == 2.5
    assert q["interest_expense"] is None  # real row absence, not an error
    assert q["depreciation_amortization"] == 120.0
    assert q["dividends_paid"] == 60.0
    assert q["capital_expenditures"] == -40.0
    assert q["operating_cash_flow"] == 500.0
    assert result.balance_sheet == {
        "total_assets": 5000.0,
        "total_liabilities": 3000.0,
        "total_equity": 2000.0,
        "total_debt": 1000.0,
        "cash_and_equivalents": 800.0,
        "current_assets": 1500.0,
        "current_liabilities": 700.0,
    }
    # annual uses the same fixture DataFrame here, so it round-trips too
    assert len(result.annual) == 1
    assert result.annual[0]["revenue"] == 1000.0


async def test_normalize_financials_empty_income_returns_empty_periods(provider, monkeypatch):
    _patch_ticker(monkeypatch, lambda ticker: FakeTicker(info={"currency": "USD"}))

    result = await provider.normalize_financials("ZZZZ")

    assert result.quarters == []
    assert result.annual == []
    assert result.balance_sheet == {}


async def test_normalize_financials_shifts_cashflow_columns_with_income_when_misaligned(
    provider, monkeypatch
):
    """_periods() ties cashflow's correct_alignment() decision to income's
    own (_correct_alignment_like), rather than each DataFrame deciding
    independently. On reflection this scenario isn't actually reachable in
    practice — the merge already requires cashflow's columns to exactly
    match income's for any row to be extracted at all (_extract_row returns
    None otherwise), so if their raw dates already match, an independent
    per-DataFrame decision would reach the same shift/no-shift answer
    regardless. Keeping the shared-decision form anyway as the more
    obviously-correct shape (readers shouldn't have to reason through that
    argument to trust it), and this test as a plain correctness check that
    a genuinely misaligned period still round-trips."""
    misaligned_period = pd.Timestamp.now() - pd.DateOffset(months=24)  # triggers the shift
    income = pd.DataFrame({misaligned_period: [1000.0]}, index=["Total Revenue"])
    cashflow = pd.DataFrame({misaligned_period: [60.0]}, index=["Cash Dividends Paid"])

    def fake_ticker(ticker):
        return FakeTicker(
            info={"currency": "USD"},
            quarterly_financials=income,
            financials=income,
            quarterly_cash_flow=cashflow,
            cashflow=cashflow,
        )

    _patch_ticker(monkeypatch, fake_ticker)

    result = await provider.normalize_financials("AAPL")

    assert result.quarters[0]["revenue"] == 1000.0
    assert result.quarters[0]["dividends_paid"] == 60.0


async def test_normalize_financials_duplicate_row_label_does_not_crash(provider, monkeypatch):
    """Real crash risk caught on review: df.loc[row_label, column] returns
    a Series, not a scalar, when the index has a duplicate label — pd.isna()/
    float() would raise "truth value of a Series is ambiguous" instead of
    gracefully returning None. Unverified whether yfinance ever actually
    produces a duplicate-labeled statement in practice, but cheap to guard."""
    period = _period_end()
    income = pd.DataFrame(
        {period: [1000.0, 999.0]},
        index=["Total Revenue", "Total Revenue"],  # duplicate label
    )

    def fake_ticker(ticker):
        return FakeTicker(info={"currency": "USD"}, quarterly_financials=income, financials=income)

    _patch_ticker(monkeypatch, fake_ticker)

    result = await provider.normalize_financials("AAPL")

    assert result.quarters[0]["revenue"] == 1000.0


# --- get_dividend_history ---


async def test_get_dividend_history_maps_to_normalized_shape_and_filters_by_date(
    provider, monkeypatch
):
    """Real, confirmed bug fixed here: this method declared -> list[dict]
    but every code path actually returned a pd.DataFrame — a type-contract
    violation, not just an unnormalized shape."""
    dividends = pd.Series(
        [0.24, 0.25, 0.27],
        index=pd.DatetimeIndex(["2024-08-12", "2025-05-11", "2026-05-11"], tz="America/New_York"),
    )
    _patch_ticker(monkeypatch, lambda ticker: FakeTicker(dividends=dividends))

    result = await provider.get_dividend_history("AAPL", "2025-01-01", "2026-12-31")

    assert isinstance(result, list)
    assert all(isinstance(row, dict) for row in result)
    assert result == [
        {"ex_date": "2025-05-11", "payment_date": None, "amount_per_share": 0.25},
        {"ex_date": "2026-05-11", "payment_date": None, "amount_per_share": 0.27},
    ]


async def test_get_dividend_history_no_dividends_returns_empty_list(provider, monkeypatch):
    _patch_ticker(monkeypatch, lambda ticker: FakeTicker(dividends=pd.Series(dtype=float)))

    result = await provider.get_dividend_history("ZZZZ", "2025-01-01", "2026-12-31")

    assert result == []
