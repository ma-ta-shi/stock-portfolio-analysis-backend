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
        self, annual=None, quarterly=None, filings: list[FakeFiling] | None = None
    ) -> None:
        self._annual = annual
        self._quarterly = quarterly
        self._filings = filings or []

    def get_financials(self) -> FakeFinancials | None:
        return self._annual

    def get_quarterly_financials(self) -> FakeFinancials | None:
        return self._quarterly

    def get_filings(self, form: str | None = None) -> FakeFilingsResult:
        return FakeFilingsResult(self._filings)


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
        ("get_peers", ("AAPL",)),
        ("get_earnings_calendar", ("AAPL",)),
        ("get_dividend_history", ("AAPL", "2024-01-01", "2024-12-31")),
    ],
)
async def test_out_of_scope_methods_raise_not_implemented(provider, method, args):
    with pytest.raises(NotImplementedError):
        await getattr(provider, method)(*args)
