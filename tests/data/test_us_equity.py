import pandas as pd
import pytest

from data.providers.base import NewsProvider, StockDataProvider
from data.providers.us_equity import USEquityDataProvider


# --- Fakes standing in for FMPDataProvider / YFinanceDataProvider ---


class FakeFMP:
    def __init__(self):
        self.price_history = pd.DataFrame()
        self.quote = {}
        self.dividend_history = []
        self.analyst_estimates = {"symbol": "TEST", "estimates": []}
        self.analyst_ratings = {}
        self.earnings_calendar = []
        self.company_info = {"symbol": "TEST"}
        self.ratios_ttm = {}
        self.raise_error: Exception | None = None
        self.calls: list[str] = []

    async def _maybe_raise(self, name):
        self.calls.append(name)
        if self.raise_error:
            raise self.raise_error

    async def get_price_history(self, ticker, period, interval):
        await self._maybe_raise("get_price_history")
        return self.price_history

    async def get_quote(self, ticker):
        await self._maybe_raise("get_quote")
        return self.quote

    async def get_dividend_history(self, ticker, from_date, to_date):
        await self._maybe_raise("get_dividend_history")
        return self.dividend_history

    async def get_analyst_estimates(self, ticker):
        await self._maybe_raise("get_analyst_estimates")
        return self.analyst_estimates

    async def get_analyst_ratings(self, ticker):
        await self._maybe_raise("get_analyst_ratings")
        return self.analyst_ratings

    async def get_earnings_calendar(self, ticker):
        await self._maybe_raise("get_earnings_calendar")
        return self.earnings_calendar

    async def get_company_info(self, ticker):
        await self._maybe_raise("get_company_info")
        return self.company_info

    async def get_ratios_ttm(self, ticker):
        await self._maybe_raise("get_ratios_ttm")
        return self.ratios_ttm

    async def get_financials(self, ticker, statement, period):
        raise NotImplementedError("Financial statements are edgartools.py's job")

    async def get_insider_trading(self, ticker, days=90):
        raise NotImplementedError("Insider trading is edgartools.py's job")

    async def get_peers(self, ticker, limit=5):
        raise NotImplementedError("Peers are Finnhub's job")

    async def get_news(self, ticker, days):
        raise NotImplementedError("News is paid-only on FMP's free tier")

    async def get_analyst_recommendation_trends(self, ticker):
        raise NotImplementedError("FMP has no historical trend endpoint")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeYFinance:
    """Only the methods the composite actually falls back to —
    get_analyst_estimates/get_analyst_ratings/get_earnings_calendar are
    FMP-only with no fallback (see USEquityDataProvider's docstring), so
    there's nothing for this fake to stand in for there."""

    def __init__(self):
        self.price_history = pd.DataFrame({"Close": [1.0, 2.0]})
        self.quote = {"symbol": "TEST", "price": 42.0}
        self.dividend_df = pd.DataFrame(columns=["Date", "Dividend"])
        self.company_info = {"Symbol": "TEST", "Company Name": "Test Co"}
        self.calls: list[str] = []

    async def get_price_history(self, ticker, period, interval):
        self.calls.append("get_price_history")
        return self.price_history

    async def get_quote(self, ticker):
        self.calls.append("get_quote")
        return self.quote

    async def get_company_info(self, ticker):
        self.calls.append("get_company_info")
        return self.company_info

    async def get_dividend_history(self, ticker, from_date, to_date):
        self.calls.append("get_dividend_history")
        return self.dividend_df


@pytest.fixture
def fmp():
    return FakeFMP()


@pytest.fixture
def yf():
    return FakeYFinance()


@pytest.fixture
def provider(fmp, yf):
    return USEquityDataProvider(fmp=fmp, yf_provider=yf)


def test_provider_is_a_stock_and_news_provider(provider):
    assert isinstance(provider, StockDataProvider)
    assert isinstance(provider, NewsProvider)


# --- get_price_history ---


async def test_price_history_returns_fmp_data_when_present(provider, fmp, yf):
    fmp.price_history = pd.DataFrame({"Close": [100.0]})

    result = await provider.get_price_history("AAPL", "1mo", "1d")

    assert list(result["Close"]) == [100.0]
    assert yf.calls == []


async def test_price_history_falls_back_to_yfinance_when_fmp_empty(provider, fmp, yf):
    """This is the ETF/dual-class case: FMP returns an empty DataFrame (its
    graceful degradation for a 402), so yfinance must be tried next."""
    fmp.price_history = pd.DataFrame()

    result = await provider.get_price_history("QQQ", "1mo", "1d")

    assert list(result["Close"]) == [1.0, 2.0]
    assert yf.calls == ["get_price_history"]


async def test_price_history_falls_back_to_yfinance_on_fmp_outage(provider, fmp, yf):
    """A 5xx/network failure (not just a paywall) should also fail over."""
    fmp.raise_error = RuntimeError("FMP API returned 503")

    result = await provider.get_price_history("MSFT", "1mo", "1d")

    assert list(result["Close"]) == [1.0, 2.0]
    assert yf.calls == ["get_price_history"]


async def test_price_history_propagates_401_without_falling_back(provider, fmp, yf):
    """A bad API key is a config error, not a per-symbol gap — must not be
    silently swallowed by falling back to yfinance for every call."""
    fmp.raise_error = RuntimeError("FMP API key rejected (401): Invalid API KEY.")

    with pytest.raises(RuntimeError, match="401"):
        await provider.get_price_history("MSFT", "1mo", "1d")

    assert yf.calls == []


# --- get_quote ---


async def test_quote_returns_fmp_data_when_present(provider, fmp, yf):
    fmp.quote = {"symbol": "AAPL", "price": 334.96}

    result = await provider.get_quote("AAPL")

    assert result == {"symbol": "AAPL", "price": 334.96}
    assert yf.calls == []


async def test_quote_falls_back_to_yfinance_when_fmp_empty(provider, fmp, yf):
    fmp.quote = {}

    result = await provider.get_quote("VOO")

    assert result == {"symbol": "TEST", "price": 42.0}
    assert yf.calls == ["get_quote"]


# --- get_dividend_history ---


async def test_dividend_history_normalizes_fmp_shape(provider, fmp):
    fmp.dividend_history = [{"symbol": "AAPL", "date": "2026-05-11", "dividend": 0.27}]

    result = await provider.get_dividend_history("AAPL", "2026-01-01", "2026-12-31")

    assert result == [{"date": "2026-05-11", "dividend": 0.27}]


async def test_dividend_history_falls_back_and_normalizes_yfinance_dataframe(provider, fmp, yf):
    fmp.dividend_history = []
    yf.dividend_df = pd.DataFrame({"Date": [pd.Timestamp("2026-06-01")], "Dividend": [1.5]})

    result = await provider.get_dividend_history("QQQ", "2026-01-01", "2026-12-31")

    assert result == [{"date": "2026-06-01", "dividend": 1.5}]


async def test_dividend_history_both_sources_empty_returns_empty_list(provider, fmp, yf):
    """GLD-style case: no dividends is a legitimate answer from both
    providers, not a crash."""
    fmp.dividend_history = []
    yf.dividend_df = pd.DataFrame(columns=["Date", "Dividend"])

    result = await provider.get_dividend_history("GLD", "2026-01-01", "2026-12-31")

    assert result == []


# --- get_company_info ---


async def test_company_info_returns_fmp_data_when_present(provider, fmp, yf):
    fmp.company_info = {"symbol": "AAPL", "companyName": "Apple Inc."}

    result = await provider.get_company_info("AAPL")

    assert result == {"symbol": "AAPL", "companyName": "Apple Inc."}
    assert yf.calls == []


async def test_company_info_falls_back_and_tags_source(provider, fmp, yf):
    """Added on review after live testing against obscure/thinly-traded
    real candidate tickers surfaced cases plausibly outside FMP's free-tier
    universe entirely — distinct from the ETF/dual-class paywall case."""
    fmp.company_info = {}
    yf.company_info = {"Symbol": "CRE", "Company Name": "Some Obscure Co"}

    result = await provider.get_company_info("CRE")

    assert result == {"source": "yfinance", "Symbol": "CRE", "Company Name": "Some Obscure Co"}


async def test_company_info_no_data_anywhere_returns_empty_dict(provider, fmp, yf):
    fmp.company_info = {}
    yf.company_info = {}

    result = await provider.get_company_info("ZZZZ")

    assert result == {}


# --- FMP-only methods: no yfinance fallback, ever ---


async def test_ratios_ttm_never_falls_back(provider, fmp, yf):
    fmp.ratios_ttm = {}

    result = await provider.get_ratios_ttm("QQQ")

    assert result == {}
    assert yf.calls == []


async def test_analyst_estimates_never_falls_back(provider, fmp, yf):
    fmp.analyst_estimates = {"symbol": "QQQ", "estimates": []}

    result = await provider.get_analyst_estimates("QQQ")

    assert result == {"symbol": "QQQ", "estimates": []}
    assert yf.calls == []


async def test_analyst_ratings_never_falls_back(provider, fmp, yf):
    fmp.analyst_ratings = {}

    result = await provider.get_analyst_ratings("QQQ")

    assert result == {}
    assert yf.calls == []


async def test_earnings_calendar_never_falls_back(provider, fmp, yf):
    fmp.earnings_calendar = []

    result = await provider.get_earnings_calendar("QQQ")

    assert result == []
    assert yf.calls == []


# --- Out-of-scope methods: unchanged routing rule ---


@pytest.mark.parametrize(
    "method,args",
    [
        ("get_financials", ("AAPL", "income", "annual")),
        ("get_insider_trading", ("AAPL",)),
        ("get_peers", ("AAPL",)),
        ("get_news", ("AAPL", 7)),
        ("get_analyst_recommendation_trends", ("AAPL",)),
    ],
)
async def test_out_of_scope_methods_raise_not_implemented(provider, method, args):
    with pytest.raises(NotImplementedError):
        await getattr(provider, method)(*args)
