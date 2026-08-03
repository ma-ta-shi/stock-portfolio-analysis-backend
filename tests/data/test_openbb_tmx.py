import asyncio
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import data.providers.openbb_tmx as tmx_module
from data.providers.openbb_tmx import OpenBBTMXProvider


# ---------- Helpers ----------

def make_obb_result(df: pd.DataFrame) -> MagicMock:
    """Mimic an OpenBB `OBBject`-like result whose .to_df() returns df."""
    result = MagicMock()
    result.to_df.return_value = df
    return result


@pytest.fixture
def provider():
    return OpenBBTMXProvider()


@pytest.fixture
def mock_obb():
    """Replace the `obb` name inside the provider module with a MagicMock.

    We patch the module-level symbol (not the dotted openbb.* path) because
    OpenBB's `obb` object resolves its command tree dynamically, so patching
    a nested dotted path doesn't reliably intercept calls made from inside
    the provider. Swapping the whole `obb` reference does.
    """
    with patch.object(tmx_module, "obb") as mock:
        yield mock


# ---------- get_price_history ----------

@pytest.mark.asyncio
async def test_get_price_history_is_coroutine(provider):
    """Method must be awaitable (async), per base class contract."""
    coro = provider.get_price_history(ticker="SHOP", period="1y", interval="1d")
    assert asyncio.iscoroutine(coro)
    coro.close()  # avoid "never awaited" warning since we're not running it


@pytest.mark.asyncio
async def test_get_price_history_calls_correct_obb_endpoint(provider, mock_obb):
    """Must call obb.equity.price.historical, not some other fetcher."""
    fake_df = pd.DataFrame({"close": [1.0, 2.0]})
    mock_obb.equity.price.historical.return_value = make_obb_result(fake_df)

    await provider.get_price_history(ticker="SHOP", period="1y", interval="1d")

    mock_obb.equity.price.historical.assert_called_once()


@pytest.mark.asyncio
async def test_get_price_history_uses_tmx_provider(provider, mock_obb):
    """Must route through the 'tmx' provider specifically."""
    fake_df = pd.DataFrame({"close": [1.0]})
    mock_obb.equity.price.historical.return_value = make_obb_result(fake_df)

    await provider.get_price_history(ticker="SHOP", period="1y", interval="1d")

    _, kwargs = mock_obb.equity.price.historical.call_args
    assert kwargs["provider"] == "tmx"


@pytest.mark.asyncio
async def test_get_price_history_passes_ticker_and_interval(provider, mock_obb):
    """Must forward ticker as `symbol` and interval unchanged."""
    fake_df = pd.DataFrame({"close": [1.0]})
    mock_obb.equity.price.historical.return_value = make_obb_result(fake_df)

    await provider.get_price_history(ticker="RY", period="6mo", interval="1wk")

    _, kwargs = mock_obb.equity.price.historical.call_args
    assert kwargs["symbol"] == "RY"
    assert kwargs["interval"] == "1wk"


@pytest.mark.asyncio
async def test_get_price_history_returns_dataframe(provider, mock_obb):
    """Return type must be a pandas DataFrame."""
    fake_df = pd.DataFrame({"close": [1.0, 2.0]})
    mock_obb.equity.price.historical.return_value = make_obb_result(fake_df)

    result = await provider.get_price_history(ticker="SHOP", period="1y", interval="1d")

    assert isinstance(result, pd.DataFrame)


@pytest.mark.asyncio
async def test_get_price_history_returns_underlying_data_unchanged(provider, mock_obb):
    """DataFrame content must match what obb returned (no silent mutation)."""
    fake_df = pd.DataFrame({"close": [10.5, 11.2]})
    mock_obb.equity.price.historical.return_value = make_obb_result(fake_df)

    result = await provider.get_price_history(ticker="SHOP", period="1y", interval="1d")

    pd.testing.assert_frame_equal(result, fake_df)


# ---------- get_financials ----------

@pytest.mark.asyncio
async def test_get_financials_routes_income_statement(provider, mock_obb):
    """statement='income' must call obb.equity.fundamental.income."""
    fake_df = pd.DataFrame({"revenue": [100]})
    mock_obb.equity.fundamental.income.return_value = make_obb_result(fake_df)

    await provider.get_financials(ticker="SHOP", statement="income", period="annual")

    mock_obb.equity.fundamental.income.assert_called_once()


@pytest.mark.asyncio
async def test_get_financials_routes_balance_statement(provider, mock_obb):
    """statement='balance' must call obb.equity.fundamental.balance."""
    fake_df = pd.DataFrame({"assets": [100]})
    mock_obb.equity.fundamental.balance.return_value = make_obb_result(fake_df)

    await provider.get_financials(ticker="SHOP", statement="balance", period="annual")

    mock_obb.equity.fundamental.balance.assert_called_once()


@pytest.mark.asyncio
async def test_get_financials_routes_cash_statement(provider, mock_obb):
    """statement='cash' must call obb.equity.fundamental.cash."""
    fake_df = pd.DataFrame({"operating_cf": [100]})
    mock_obb.equity.fundamental.cash.return_value = make_obb_result(fake_df)

    await provider.get_financials(ticker="SHOP", statement="cash", period="annual")

    mock_obb.equity.fundamental.cash.assert_called_once()


@pytest.mark.asyncio
async def test_get_financials_unknown_statement_raises_value_error(provider, mock_obb):
    """Unknown statement type must raise ValueError, not silently fetch something."""
    with pytest.raises(ValueError):
        await provider.get_financials(ticker="SHOP", statement="cashflowzz", period="annual")


@pytest.mark.asyncio
async def test_get_financials_returns_dataframe(provider, mock_obb):
    """Return type must be a pandas DataFrame."""
    fake_df = pd.DataFrame({"revenue": [100]})
    mock_obb.equity.fundamental.income.return_value = make_obb_result(fake_df)

    result = await provider.get_financials(ticker="SHOP", statement="income", period="annual")

    assert isinstance(result, pd.DataFrame)


# ---------- get_company_info ----------

@pytest.mark.asyncio
async def test_get_company_info_calls_profile_endpoint(provider, mock_obb):
    """Must call obb.equity.profile."""
    fake_df = pd.DataFrame([{"name": "Shopify Inc.", "sector": "Tech"}])
    mock_obb.equity.profile.return_value = make_obb_result(fake_df)

    await provider.get_company_info(ticker="SHOP")

    mock_obb.equity.profile.assert_called_once()


@pytest.mark.asyncio
async def test_get_company_info_returns_dict(provider, mock_obb):
    """Return type must be a plain dict (single-row profile)."""
    fake_df = pd.DataFrame([{"name": "Shopify Inc.", "sector": "Tech"}])
    mock_obb.equity.profile.return_value = make_obb_result(fake_df)

    result = await provider.get_company_info(ticker="SHOP")

    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_get_company_info_returns_first_row_only(provider, mock_obb):
    """If multiple rows are returned, only the first row's data should be used."""
    fake_df = pd.DataFrame([
        {"name": "Row One"},
        {"name": "Row Two"},
    ])
    mock_obb.equity.profile.return_value = make_obb_result(fake_df)

    result = await provider.get_company_info(ticker="SHOP")

    assert result["name"] == "Row One"


# ---------- get_dividend_history ----------

@pytest.mark.asyncio
async def test_get_dividend_history_calls_dividends_endpoint(provider, mock_obb):
    """Must call obb.equity.fundamental.dividends."""
    fake_df = pd.DataFrame(
        {"amount": [0.5, 0.5]},
        index=pd.to_datetime(["2023-01-01", "2023-06-01"]),
    )
    mock_obb.equity.fundamental.dividends.return_value = make_obb_result(fake_df)

    await provider.get_dividend_history(ticker="SHOP", from_date="2023-01-01", to_date="2023-12-31")

    mock_obb.equity.fundamental.dividends.assert_called_once()


@pytest.mark.asyncio
async def test_get_dividend_history_filters_by_date_range(provider, mock_obb):
    """Rows outside [from_date, to_date] must be excluded."""
    fake_df = pd.DataFrame(
        {"amount": [0.5, 0.5, 0.5]},
        index=pd.to_datetime(["2022-01-01", "2023-06-01", "2024-01-01"]),
    )
    mock_obb.equity.fundamental.dividends.return_value = make_obb_result(fake_df)

    result = await provider.get_dividend_history(
        ticker="SHOP", from_date="2023-01-01", to_date="2023-12-31"
    )

    assert len(result) == 1


@pytest.mark.asyncio
async def test_get_dividend_history_returns_list_of_dicts(provider, mock_obb):
    """Return type must be list[dict], per base class contract."""
    fake_df = pd.DataFrame(
        {"amount": [0.5]},
        index=pd.to_datetime(["2023-06-01"]),
    )
    mock_obb.equity.fundamental.dividends.return_value = make_obb_result(fake_df)

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