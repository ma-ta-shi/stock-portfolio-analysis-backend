import asyncio
from unittest.mock import MagicMock, patch
from data.providers.boc import OpenBBTMXProvider
import pandas as pd
import pytest


# ---------- Helpers ----------

def make_obb_result(df: pd.DataFrame) -> MagicMock:
    """Mimic an OpenBB `OBBject`-like result whose .to_df() returns df."""
    result = MagicMock()
    result.to_df.return_value = df
    return result


@pytest.fixture
def provider():
    return OpenBBTMXProvider()


# ---------- get_price_history ----------

@pytest.mark.asyncio
async def test_get_price_history_is_coroutine(provider):
    """Method must be awaitable (async), per base class contract."""
    coro = provider.get_price_history(ticker="SHOP", period="1y", interval="1d")
    assert asyncio.iscoroutine(coro)
    coro.close()  # avoid "never awaited" warning since we're not running it


@pytest.mark.asyncio
async def test_get_price_history_calls_correct_obb_endpoint(provider):
    """Must call obb.equity.price.historical, not some other fetcher."""
    fake_df = pd.DataFrame({"close": [1.0, 2.0]})
    with patch("openbb.obb.equity.price.historical", return_value=make_obb_result(fake_df)) as mock_call:
        await provider.get_price_history(ticker="SHOP", period="1y", interval="1d")
        mock_call.assert_called_once()


@pytest.mark.asyncio
async def test_get_price_history_uses_tmx_provider(provider):
    """Must route through the 'tmx' provider specifically."""
    fake_df = pd.DataFrame({"close": [1.0]})
    with patch("openbb.obb.equity.price.historical", return_value=make_obb_result(fake_df)) as mock_call:
        await provider.get_price_history(ticker="SHOP", period="1y", interval="1d")
        _, kwargs = mock_call.call_args
        assert kwargs["provider"] == "tmx"


@pytest.mark.asyncio
async def test_get_price_history_passes_ticker_and_interval(provider):
    """Must forward ticker as `symbol` and interval unchanged."""
    fake_df = pd.DataFrame({"close": [1.0]})
    with patch("openbb.obb.equity.price.historical", return_value=make_obb_result(fake_df)) as mock_call:
        await provider.get_price_history(ticker="RY", period="6mo", interval="1wk")
        _, kwargs = mock_call.call_args
        assert kwargs["symbol"] == "RY"
        assert kwargs["interval"] == "1wk"


@pytest.mark.asyncio
async def test_get_price_history_returns_dataframe(provider):
    """Return type must be a pandas DataFrame."""
    fake_df = pd.DataFrame({"close": [1.0, 2.0]})
    with patch("openbb.obb.equity.price.historical", return_value=make_obb_result(fake_df)):
        result = await provider.get_price_history(ticker="SHOP", period="1y", interval="1d")
        assert isinstance(result, pd.DataFrame)


@pytest.mark.asyncio
async def test_get_price_history_returns_underlying_data_unchanged(provider):
    """DataFrame content must match what obb returned (no silent mutation)."""
    fake_df = pd.DataFrame({"close": [10.5, 11.2]})
    with patch("openbb.obb.equity.price.historical", return_value=make_obb_result(fake_df)):
        result = await provider.get_price_history(ticker="SHOP", period="1y", interval="1d")
        pd.testing.assert_frame_equal(result, fake_df)


# ---------- get_financials ----------

@pytest.mark.asyncio
async def test_get_financials_routes_income_statement(provider):
    """statement='income' must call obb.equity.fundamental.income."""
    fake_df = pd.DataFrame({"revenue": [100]})
    with patch("openbb.obb.equity.fundamental.income", return_value=make_obb_result(fake_df)) as mock_call:
        await provider.get_financials(ticker="SHOP", statement="income", period="annual")
        mock_call.assert_called_once()


@pytest.mark.asyncio
async def test_get_financials_routes_balance_statement(provider):
    """statement='balance' must call obb.equity.fundamental.balance."""
    fake_df = pd.DataFrame({"assets": [100]})
    with patch("openbb.obb.equity.fundamental.balance", return_value=make_obb_result(fake_df)) as mock_call:
        await provider.get_financials(ticker="SHOP", statement="balance", period="annual")
        mock_call.assert_called_once()


@pytest.mark.asyncio
async def test_get_financials_routes_cash_statement(provider):
    """statement='cash' must call obb.equity.fundamental.cash."""
    fake_df = pd.DataFrame({"operating_cf": [100]})
    with patch("openbb.obb.equity.fundamental.cash", return_value=make_obb_result(fake_df)) as mock_call:
        await provider.get_financials(ticker="SHOP", statement="cash", period="annual")
        mock_call.assert_called_once()


@pytest.mark.asyncio
async def test_get_financials_unknown_statement_raises_value_error(provider):
    """Unknown statement type must raise ValueError, not silently fetch something."""
    with pytest.raises(ValueError):
        await provider.get_financials(ticker="SHOP", statement="cashflowzz", period="annual")


@pytest.mark.asyncio
async def test_get_financials_returns_dataframe(provider):
    """Return type must be a pandas DataFrame."""
    fake_df = pd.DataFrame({"revenue": [100]})
    with patch("openbb.obb.equity.fundamental.income", return_value=make_obb_result(fake_df)):
        result = await provider.get_financials(ticker="SHOP", statement="income", period="annual")
        assert isinstance(result, pd.DataFrame)


# ---------- get_company_info ----------

@pytest.mark.asyncio
async def test_get_company_info_calls_profile_endpoint(provider):
    """Must call obb.equity.profile."""
    fake_df = pd.DataFrame([{"name": "Shopify Inc.", "sector": "Tech"}])
    with patch("openbb.obb.equity.profile", return_value=make_obb_result(fake_df)) as mock_call:
        await provider.get_company_info(ticker="SHOP")
        mock_call.assert_called_once()


@pytest.mark.asyncio
async def test_get_company_info_returns_dict(provider):
    """Return type must be a plain dict (single-row profile)."""
    fake_df = pd.DataFrame([{"name": "Shopify Inc.", "sector": "Tech"}])
    with patch("openbb.obb.equity.profile", return_value=make_obb_result(fake_df)):
        result = await provider.get_company_info(ticker="SHOP")
        assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_get_company_info_returns_first_row_only(provider):
    """If multiple rows are returned, only the first row's data should be used."""
    fake_df = pd.DataFrame([
        {"name": "Row One"},
        {"name": "Row Two"},
    ])
    with patch("openbb.obb.equity.profile", return_value=make_obb_result(fake_df)):
        result = await provider.get_company_info(ticker="SHOP")
        assert result["name"] == "Row One"


# ---------- get_dividend_history ----------

@pytest.mark.asyncio
async def test_get_dividend_history_calls_dividends_endpoint(provider):
    """Must call obb.equity.fundamental.dividends."""
    fake_df = pd.DataFrame(
        {"amount": [0.5, 0.5]},
        index=pd.to_datetime(["2023-01-01", "2023-06-01"]),
    )
    with patch("openbb.obb.equity.fundamental.dividends", return_value=make_obb_result(fake_df)) as mock_call:
        await provider.get_dividend_history(ticker="SHOP", from_date="2023-01-01", to_date="2023-12-31")
        mock_call.assert_called_once()


@pytest.mark.asyncio
async def test_get_dividend_history_filters_by_date_range(provider):
    """Rows outside [from_date, to_date] must be excluded."""
    fake_df = pd.DataFrame(
        {"amount": [0.5, 0.5, 0.5]},
        index=pd.to_datetime(["2022-01-01", "2023-06-01", "2024-01-01"]),
    )
    with patch("openbb.obb.equity.fundamental.dividends", return_value=make_obb_result(fake_df)):
        result = await provider.get_dividend_history(
            ticker="SHOP", from_date="2023-01-01", to_date="2023-12-31"
        )
        assert len(result) == 1


@pytest.mark.asyncio
async def test_get_dividend_history_returns_list_of_dicts(provider):
    """Return type must be list[dict], per base class contract."""
    fake_df = pd.DataFrame(
        {"amount": [0.5]},
        index=pd.to_datetime(["2023-06-01"]),
    )
    with patch("openbb.obb.equity.fundamental.dividends", return_value=make_obb_result(fake_df)):
        result = await provider.get_dividend_history(
            ticker="SHOP", from_date="2023-01-01", to_date="2023-12-31"
        )
        assert isinstance(result, list)
        assert all(isinstance(row, dict) for row in result)


# ---------- Unsupported methods (should raise NotImplementedError) ----------

@pytest.mark.asyncio
async def test_get_analyst_estimates_not_implemented(provider):
    with pytest.raises(NotImplementedError):
        await provider.get_analyst_estimates(ticker="SHOP")


@pytest.mark.asyncio
async def test_get_analyst_ratings_not_implemented(provider):
    with pytest.raises(NotImplementedError):
        await provider.get_analyst_ratings(ticker="SHOP")


@pytest.mark.asyncio
async def test_get_insider_trading_not_implemented(provider):
    with pytest.raises(NotImplementedError):
        await provider.get_insider_trading(ticker="SHOP")


@pytest.mark.asyncio
async def test_get_peers_not_implemented(provider):
    with pytest.raises(NotImplementedError):
        await provider.get_peers(ticker="SHOP")


@pytest.mark.asyncio
async def test_get_earnings_calendar_not_implemented(provider):
    with pytest.raises(NotImplementedError):
        await provider.get_earnings_calendar(ticker="SHOP")


@pytest.mark.asyncio
async def test_get_news_not_implemented(provider):
    with pytest.raises(NotImplementedError):
        await provider.get_news(ticker="SHOP", days=7)


@pytest.mark.asyncio
async def test_get_analyst_recommendation_trends_not_implemented(provider):
    with pytest.raises(NotImplementedError):
        await provider.get_analyst_recommendation_trends(ticker="SHOP")