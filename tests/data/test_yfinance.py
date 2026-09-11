import pandas as pd
import pytest
import structlog

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
        fast_info: dict | None = None,
        raises_on_fast_info: bool = False,
        insider_transactions: pd.DataFrame | None = None,
        calendar: dict | None = None,
        earnings_history: pd.DataFrame | None = None,
        price_history: pd.DataFrame | None = None,
    ) -> None:
        self._price_history = _empty_if_none(price_history)
        self.history_calls: list[dict] = []
        self.info = info or {}
        self.quarterly_financials = _empty_if_none(quarterly_financials)
        self.financials = _empty_if_none(financials)
        self.quarterly_balance_sheet = _empty_if_none(quarterly_balance_sheet)
        self.balance_sheet = _empty_if_none(balance_sheet)
        self.quarterly_cash_flow = _empty_if_none(quarterly_cash_flow)
        self.cashflow = _empty_if_none(cashflow)
        self.dividends = dividends if dividends is not None else pd.Series(dtype=float)
        self._fast_info = fast_info or {}
        self._raises_on_fast_info = raises_on_fast_info
        self.insider_transactions = _empty_if_none(insider_transactions)
        self.calendar = calendar if calendar is not None else {}
        self.earnings_history = _empty_if_none(earnings_history)

    @property
    def fast_info(self) -> dict:
        if self._raises_on_fast_info:
            raise KeyError("quoteType not found")  # real yfinance failure mode
        return self._fast_info

    def history(self, *args, **kwargs) -> pd.DataFrame:
        self.history_calls.append({"args": args, "kwargs": kwargs})
        return self._price_history


def _empty_if_none(df: pd.DataFrame | None) -> pd.DataFrame:
    return df if df is not None else pd.DataFrame()


@pytest.fixture
def provider():
    return YFinanceDataProvider()


def _patch_ticker(monkeypatch, ticker_factory) -> None:
    monkeypatch.setattr("data.providers.yfinance.yf.Ticker", ticker_factory)


# --- get_price_history (86bbq7dkv) ---


async def test_get_price_history_requests_unadjusted_bar(provider, monkeypatch):
    """auto_adjust=False so `Close` is the raw split-adjusted, NOT
    dividend-back-adjusted, price — same basis as openbb-tmx / FMP."""
    fake = FakeTicker(price_history=pd.DataFrame({"Close": [1.0, 2.0]}))
    _patch_ticker(monkeypatch, lambda ticker: fake)
    await provider.get_price_history("AAPL", "5y", "1d")
    assert fake.history_calls[0]["kwargs"] == {
        "period": "5y",
        "interval": "1d",
        "auto_adjust": False,
    }


async def test_get_price_history_drops_adj_close_column(provider, monkeypatch):
    """The extra `Adj Close` column auto_adjust=False adds is dropped —
    the output column set stays what it was before this change."""
    raw = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [1.0],
            "Low": [1.0],
            "Close": [1.0],
            "Adj Close": [0.95],
            "Volume": [100],
            "Dividends": [0.0],
            "Stock Splits": [0.0],
        }
    )
    _patch_ticker(monkeypatch, lambda ticker: FakeTicker(price_history=raw))
    result = await provider.get_price_history("AAPL", "1y", "1d")
    assert "Adj Close" not in result.columns
    assert list(result.columns) == [
        "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"
    ]
    assert result["Close"].iloc[0] == 1.0  # the raw close, not the 0.95 adjusted one


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
        "quoteType": "EQUITY",
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
        "asset_type": "equity",
    }


@pytest.mark.parametrize(
    "quote_type,expected",
    [
        ("EQUITY", "equity"),
        ("ETF", "etf"),
        ("MUTUALFUND", "other"),
        ("INDEX", "other"),
        ("CURRENCY", "other"),
        (None, "equity"),  # past the longName guard, a missing quoteType -> equity
    ],
)
async def test_get_company_info_maps_asset_type_from_quote_type(
    provider, monkeypatch, quote_type, expected
):
    """86bbpk6uf part 1: yfinance's own quoteType drives asset_type.
    NB: yfinance labels every closed-end fund "EQUITY" — a documented gap
    the US path covers via FMP's isFund."""
    info = {"longName": "Some Fund", "quoteType": quote_type}
    _patch_ticker(monkeypatch, lambda ticker: FakeTicker(info=info))

    result = await provider.get_company_info("SOME")

    assert result["asset_type"] == expected


async def test_get_company_info_no_real_data_returns_empty_dict(provider, monkeypatch):
    """Real gap caught in a final integration-level review: a genuinely
    invalid ticker's .info isn't literally {} — confirmed live it returns
    a near-empty dict with one unrelated key ({'trailingPegRatio': None}).
    Without the `if not info.get("longName")` guard, this would have built
    a full 8-key NormalizedCompanyInfo with everything blank, which
    router.py's _is_empty() (len(dict) == 0) can never detect as empty —
    breaking the fallback chain's ability to recognize total failure."""
    _patch_ticker(monkeypatch, lambda ticker: FakeTicker(info={"trailingPegRatio": None}))

    result = await provider.get_company_info("ZZZZINVALID")

    assert result == {}


async def test_get_company_info_missing_secondary_fields_default_gracefully(provider, monkeypatch):
    """A real ticker with `longName` present but some other fields missing
    still returns a partial NormalizedCompanyInfo, not {} — the guard is
    specifically "is there any real data at all," not "is every field set."""
    _patch_ticker(monkeypatch, lambda ticker: FakeTicker(info={"longName": "Some Co"}))

    result = await provider.get_company_info("SOME")

    assert result["name"] == "Some Co"
    assert result["market_cap"] is None
    assert result["sector"] == ""
    assert result["asset_type"] == "equity"  # no quoteType in .info -> equity


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


async def test_normalize_financials_maps_net_income_common(provider, monkeypatch):
    """net_income_common ("Net Income Common Stockholders") is the P/E
    denominator fundamentals.py prefers for preferred-heavy names."""
    period = _period_end()
    income = pd.DataFrame(
        {period: [1000.0, 250.0, 235.0]},
        index=["Total Revenue", "Net Income", "Net Income Common Stockholders"],
    )

    def fake_ticker(ticker):
        return FakeTicker(info={"currency": "USD"}, quarterly_financials=income, financials=income)

    _patch_ticker(monkeypatch, fake_ticker)

    result = await provider.normalize_financials("RY.TO")

    assert result.quarters[0]["net_income_common"] == 235.0


async def test_normalize_financials_currency_is_financial_currency_not_trading(provider, monkeypatch):
    """A Canadian-listed, USD-reporting company (ATD.TO, NTR.TO, ...) has
    info["currency"] == "CAD" but statement line items in USD. NormalizedFinancials
    must carry the financial currency so compute_all()'s guard catches the
    mismatch instead of producing FX-distorted ratios."""
    period = _period_end()
    income = pd.DataFrame({period: [1000.0, 250.0]}, index=["Total Revenue", "Net Income"])

    def fake_ticker(ticker):
        return FakeTicker(
            info={"currency": "CAD", "financialCurrency": "USD"},
            quarterly_financials=income,
            financials=income,
        )

    _patch_ticker(monkeypatch, fake_ticker)

    result = await provider.normalize_financials("ATD.TO")

    assert result.currency == "USD"


async def test_normalize_financials_currency_falls_back_when_financial_currency_absent(
    provider, monkeypatch
):
    period = _period_end()
    income = pd.DataFrame({period: [1000.0, 250.0]}, index=["Total Revenue", "Net Income"])

    def fake_ticker(ticker):
        return FakeTicker(info={"currency": "CAD"}, quarterly_financials=income, financials=income)

    _patch_ticker(monkeypatch, fake_ticker)

    result = await provider.normalize_financials("RY.TO")

    assert result.currency == "CAD"


async def test_normalize_financials_drops_all_none_leading_period(provider, monkeypatch):
    """yfinance materialises a column for the newest period before the filing
    data lands — an all-None placeholder that would poison _ttm()'s trailing
    window. _periods() drops any period with neither revenue nor net income."""
    newest = pd.Timestamp.now().normalize()
    older = newest - pd.DateOffset(months=3)
    income = pd.DataFrame(
        {newest: [None, None], older: [1000.0, 250.0]},
        index=["Total Revenue", "Net Income"],
    )

    def fake_ticker(ticker):
        return FakeTicker(info={"currency": "USD"}, quarterly_financials=income, financials=income)

    _patch_ticker(monkeypatch, fake_ticker)

    result = await provider.normalize_financials("ENB.TO")

    assert len(result.quarters) == 1
    assert result.quarters[0]["revenue"] == 1000.0


async def test_normalize_financials_keeps_single_real_period(provider, monkeypatch):
    """The drop-empty guard must not eat a lone genuine period."""
    period = _period_end()
    income = pd.DataFrame({period: [1000.0, 250.0]}, index=["Total Revenue", "Net Income"])

    def fake_ticker(ticker):
        return FakeTicker(info={"currency": "USD"}, quarterly_financials=income, financials=income)

    _patch_ticker(monkeypatch, fake_ticker)

    result = await provider.normalize_financials("AAPL")

    assert len(result.quarters) == 1


async def test_normalize_financials_skips_empty_leading_balance_column(provider, monkeypatch):
    """Same recent-filer lag on the balance sheet — the newest column can be
    an all-None placeholder while columns[1] carries the real data. Take the
    newest column that actually has equity or assets, not blindly columns[0]."""
    period = _period_end()
    newest = period
    older = period - pd.DateOffset(months=3)
    balance = pd.DataFrame(
        {
            newest: [None, None, None],
            older: [5000.0, 2000.0, 800.0],
        },
        index=["Total Assets", "Stockholders Equity", "Cash And Cash Equivalents"],
    )
    income = pd.DataFrame({newest: [1000.0, 250.0]}, index=["Total Revenue", "Net Income"])

    def fake_ticker(ticker):
        return FakeTicker(
            info={"currency": "USD"},
            quarterly_financials=income,
            financials=income,
            quarterly_balance_sheet=balance,
            balance_sheet=balance,
        )

    _patch_ticker(monkeypatch, fake_ticker)

    result = await provider.normalize_financials("ENB.TO")

    assert result.balance_sheet["total_equity"] == 2000.0
    assert result.balance_sheet["total_assets"] == 5000.0
    assert result.balance_sheet["cash_and_equivalents"] == 800.0


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


# --- get_quote ---


async def test_get_quote_maps_to_normalized_shape(provider, monkeypatch):
    fast_info = {
        "lastPrice": 306.23,
        "marketCap": 4_469_175_901_736.45,
        "currency": "USD",
        "yearHigh": 344.57,
        "yearLow": 223.78,
    }
    _patch_ticker(monkeypatch, lambda ticker: FakeTicker(fast_info=fast_info))

    result = await provider.get_quote("AAPL")

    assert result == {
        "current_price": 306.23,
        "market_cap": 4_469_175_901_736.45,
        "currency": "USD",
        "high_52w": 344.57,
        "low_52w": 223.78,
    }


async def test_get_quote_invalid_ticker_returns_empty_dict(provider, monkeypatch):
    """Real failure mode: yfinance's fast_info raises KeyError for an
    invalid/delisted ticker rather than returning empty."""
    _patch_ticker(monkeypatch, lambda ticker: FakeTicker(raises_on_fast_info=True))

    result = await provider.get_quote("ZZZZ")

    assert result == {}


async def test_get_quote_missing_price_returns_empty_dict(provider, monkeypatch):
    _patch_ticker(monkeypatch, lambda ticker: FakeTicker(fast_info={}))

    result = await provider.get_quote("ZZZZ")

    assert result == {}


async def test_get_quote_null_currency_defaults_to_empty_string_not_none(provider, monkeypatch):
    """Real bug caught on review: confirmed live that FastInfo.get(key,
    default) only falls back to `default` for a genuinely unrecognized key,
    not when a recognized key's own value is None (e.g. ^GSPC has a real
    fast_info.market_cap of None, and .get("marketCap", "X") still returns
    None, not "X"). currency=fi.get("currency", "") would have silently
    stored None in a field typed str whenever currency is genuinely
    unavailable — same bug pattern already fixed once for get_company_info()."""
    fast_info = {"lastPrice": 100.0, "currency": None}
    _patch_ticker(monkeypatch, lambda ticker: FakeTicker(fast_info=fast_info))

    result = await provider.get_quote("SOME.TICKER")

    assert result["currency"] == ""


# --- get_insider_trading ---


def _insider_df(rows: list[dict]) -> pd.DataFrame:
    """Builds a frame with yfinance's real column set (86bbwha5r) — every
    row gets a default for columns the test doesn't care about, matching
    real .insider_transactions shape (Shares/Value/Text/Insider/Position/
    Transaction/Start Date/Ownership)."""
    now = pd.Timestamp.now()
    defaults = {
        "Shares": 100,
        "Value": 1000.0,
        "Text": "",
        "Insider": "Jane Doe",
        "Position": "Director of Issuer",
        "Transaction": "",  # real yfinance data: always empty, never the phrase field
        "Start Date": now - pd.Timedelta(days=5),
        "Ownership": "D",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


async def test_get_insider_trading_maps_normalized_fields_within_window(provider, monkeypatch):
    """Real, confirmed bug fixed here: declared -> list[dict] but every
    code path actually returned a pd.DataFrame. 86bbwha5r: now normalizes
    into NormalizedInsiderTransaction, classifying `Text` (not the always-
    empty `Transaction` column — a real mix-up caught before shipping)."""
    now = pd.Timestamp.now()
    df = _insider_df(
        [
            {
                "Text": "Disposition in the public market at price 199.35 per share.",
                "Insider": "Jane Doe",
                "Shares": 100,
                "Value": 19935.0,
                "Start Date": now - pd.Timedelta(days=5),
            },
            {
                "Text": "Disposition in the public market at price 100.00 per share.",
                "Insider": "Old Insider",
                "Start Date": now - pd.Timedelta(days=400),
            },
        ]
    )
    _patch_ticker(monkeypatch, lambda ticker: FakeTicker(insider_transactions=df))

    result = await provider.get_insider_trading("AAPL", days=90)

    assert len(result) == 1
    assert result[0] == {
        "date": (now - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        "insider_name": "Jane Doe",
        "is_issuer": False,
        "transaction_type": "sale",
        "shares": 100.0,
        "value": 19935.0,
    }


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Exercise of options at price 72.08 per share.", "exercise"),
        ("Disposition in the public market at price 1.00 per share.", "sale"),
        ("Disposition under a purchase/ownership plan at price 1.00 per share.", "sale"),
        ("Sale at price 170.00 per share.", "sale"),  # WCN.TO's distinct phrasing, same meaning
        ("Purchase at price 152.24 per share.", "purchase"),  # WCN.TO's counterpart, caught live
        ("Acquisition in the public market at price 1.00 per share.", "purchase"),
        ("Acquisition under a purchase/ownership plan at price 1.00 per share.", "purchase"),
        ("Redemption, retraction, cancelation, repurchase at price 1.00 per share.", "buyback"),
        ("Stock Gift at price 0.00 per share.", "gift"),
        ("Grant of rights at price 1.00 per share.", "other"),  # not a market transaction
        ("Change in nature of ownership at price 1.00 per share.", "other"),
    ],
)
async def test_get_insider_trading_classifies_real_text_phrases(
    provider, monkeypatch, text, expected
):
    """Every phrase here was observed live across 8 real CA tickers spanning
    6 sectors (86bbwha5r) — not invented."""
    df = _insider_df([{"Text": text}])
    _patch_ticker(monkeypatch, lambda ticker: FakeTicker(insider_transactions=df))

    result = await provider.get_insider_trading("RY.TO")

    assert result[0]["transaction_type"] == expected


async def test_get_insider_trading_unrecognized_nonempty_text_logs_and_lands_other(
    provider, monkeypatch
):
    df = _insider_df([{"Text": "Some brand new phrase never seen before."}])
    _patch_ticker(monkeypatch, lambda ticker: FakeTicker(insider_transactions=df))

    with structlog.testing.capture_logs() as logs:
        result = await provider.get_insider_trading("RY.TO")

    assert result[0]["transaction_type"] == "other"
    assert any(log["event"] == "yfinance_unrecognized_insider_text" for log in logs)


async def test_get_insider_trading_blank_text_lands_other_without_logging(provider, monkeypatch):
    """86bbwha5r: confirmed live that a blank Text is common (often the
    majority of real rows, e.g. RY.TO/SHOP.TO/WCN.TO), not rare filler —
    it must still appear in the output, just uncounted as "unrecognized"
    since there was never a phrase to fail to recognize."""
    df = _insider_df([{"Text": "", "Shares": 350000, "Position": "Issuer", "Value": None}])
    _patch_ticker(monkeypatch, lambda ticker: FakeTicker(insider_transactions=df))

    with structlog.testing.capture_logs() as logs:
        result = await provider.get_insider_trading("RY.TO")

    assert len(result) == 1
    assert result[0]["transaction_type"] == "other"
    assert result[0]["shares"] == 350000.0
    assert result[0]["value"] is None
    assert not any(log["event"] == "yfinance_unrecognized_insider_text" for log in logs)


async def test_get_insider_trading_is_issuer_flags_buybacks(provider, monkeypatch):
    df = _insider_df(
        [
            {
                "Text": "Redemption, retraction, cancelation, repurchase at price 1.00 per share.",
                "Insider": "Royal Bank of Canada",
                "Position": "Issuer",
            }
        ]
    )
    _patch_ticker(monkeypatch, lambda ticker: FakeTicker(insider_transactions=df))

    result = await provider.get_insider_trading("RY.TO")

    assert result[0]["is_issuer"] is True
    assert result[0]["transaction_type"] == "buyback"


async def test_get_insider_trading_no_data_returns_empty_list(provider, monkeypatch):
    _patch_ticker(monkeypatch, lambda ticker: FakeTicker(insider_transactions=pd.DataFrame()))

    result = await provider.get_insider_trading("ZZZZ")

    assert result == []


async def test_get_insider_trading_no_date_column_returns_empty_list(provider, monkeypatch):
    """Never observed live across 8 real tickers — every real row has
    Start Date or Date. 86bbwha5r: changed from returning malformed raw
    dict rows to a clean [] so the router falls back to openbb_tmx instead
    of serving rows with no `date` field at all."""
    df = pd.DataFrame({"Insider": ["Jane Doe"], "Shares": [100]})
    _patch_ticker(monkeypatch, lambda ticker: FakeTicker(insider_transactions=df))

    result = await provider.get_insider_trading("AAPL")

    assert result == []


# --- get_earnings_calendar ---


async def test_get_earnings_calendar_returns_list_of_dicts(provider, monkeypatch):
    """Real, confirmed bug fixed here: declared -> list[dict] but every
    code path actually returned a single dict, never a list."""
    _patch_ticker(
        monkeypatch,
        lambda ticker: FakeTicker(
            calendar={
                "Earnings Date": [pd.Timestamp("2026-11-05")],
                "Earnings Average": 1.5,
                "Earnings High": 1.7,
                "Earnings Low": 1.3,
                "Revenue Average": 90_000_000_000,
            }
        ),
    )

    result = await provider.get_earnings_calendar("AAPL")

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["Upcoming Earnings Dates"] == ["2026-11-05"]
    assert result[0]["EPS Estimate"] == 1.5


async def test_get_earnings_calendar_no_data_returns_empty_list(provider, monkeypatch):
    _patch_ticker(monkeypatch, lambda ticker: FakeTicker(calendar={}))

    result = await provider.get_earnings_calendar("ZZZZ")

    assert result == []


# --- get_analyst_estimates (new, 86bbdu04a) ---


async def test_get_analyst_estimates_reads_forward_eps_from_info(provider, monkeypatch):
    """yfinance doesn't expose forward EPS via analyst_price_targets or
    recommendations — it's a plain .info field, confirmed live for both
    AAPL and RY.TO."""
    _patch_ticker(monkeypatch, lambda ticker: FakeTicker(info={"forwardEps": 9.50771}))

    result = await provider.get_analyst_estimates("AAPL")

    assert result == {"forward_eps": 9.50771}


async def test_get_analyst_estimates_missing_forward_eps_returns_empty_dict(provider, monkeypatch):
    """Bare {}, not {"forward_eps": None} — matches get_company_info's/
    get_quote's own "empty dict signals no data" convention, needed for
    Router._is_empty() to correctly recognize this as empty."""
    _patch_ticker(monkeypatch, lambda ticker: FakeTicker(info={}))

    result = await provider.get_analyst_estimates("ZZZZ")

    assert result == {}


# --- get_earnings_surprises (new, 86bbdu04a) ---


async def test_get_earnings_surprises_maps_history_to_normalized_shape(provider, monkeypatch):
    """Period comes from the DataFrame's "quarter" index, not a column —
    confirmed live. surprisePercent is a fraction (e.g. 0.0452), not a
    percent — must be scaled by 100 to match FMP's convention."""
    eh = pd.DataFrame(
        {
            "epsActual": [1.85, 2.84],
            "epsEstimate": [1.76993, 2.6708],
            "epsDifference": [0.08, 0.17],
            "surprisePercent": [0.0452, 0.0634],
        },
        index=pd.Index(
            [pd.Timestamp("2025-09-30"), pd.Timestamp("2025-12-31")], name="quarter"
        ),
    )
    _patch_ticker(monkeypatch, lambda ticker: FakeTicker(earnings_history=eh))

    result = await provider.get_earnings_surprises("AAPL")

    assert result == [
        {
            "period_end": "2025-09-30",
            "eps_actual": 1.85,
            "eps_estimated": 1.76993,
            "eps_surprise_pct": pytest.approx(4.52),
            "revenue_actual": None,
            "revenue_estimated": None,
        },
        {
            "period_end": "2025-12-31",
            "eps_actual": 2.84,
            "eps_estimated": 2.6708,
            "eps_surprise_pct": pytest.approx(6.34),
            "revenue_actual": None,
            "revenue_estimated": None,
        },
    ]


async def test_get_earnings_surprises_values_are_native_float_not_numpy(provider, monkeypatch):
    """Real gap caught via live integration check: earnings_history's
    cells are numpy.float64, a DataFrame artifact — a plain `==`
    comparison against Python floats doesn't catch this (numpy.float64
    compares equal to float by value), so this checks the exact type
    instead. numpy.float64 isn't always JSON-serializable downstream and
    nothing else in this codebase's provider layer lets it leak through."""
    eh = pd.DataFrame(
        {
            "epsActual": [1.85],
            "epsEstimate": [1.76993],
            "epsDifference": [0.08],
            "surprisePercent": [0.0452],
        },
        index=pd.Index([pd.Timestamp("2025-09-30")], name="quarter"),
    )
    _patch_ticker(monkeypatch, lambda ticker: FakeTicker(earnings_history=eh))

    result = await provider.get_earnings_surprises("AAPL")

    assert type(result[0]["eps_actual"]) is float
    assert type(result[0]["eps_estimated"]) is float
    assert type(result[0]["eps_surprise_pct"]) is float


async def test_get_earnings_surprises_no_data_returns_empty_list(provider, monkeypatch):
    _patch_ticker(monkeypatch, lambda ticker: FakeTicker(earnings_history=None))

    result = await provider.get_earnings_surprises("ZZZZ")

    assert result == []
