import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from data.providers.openbb_tmx import OpenBBTMXProvider
import pandas as pd
import pytest


# ---------- Helpers ----------


def make_obb_result(df: pd.DataFrame) -> MagicMock:
    """Mimic an OpenBB `OBBject`-like result whose .to_df() returns df.
    Also sets `.results` to a non-empty placeholder list so the
    `if not result.results: return ...` guards (added 2026-08-04,
    86bb7j0kh — real live bug: `.to_df()`/the old index-based logic both
    raise/misbehave on a genuinely empty OBBject) don't short-circuit
    methods that still go through .to_df() for non-empty fixtures."""
    result = MagicMock()
    result.to_df.return_value = df
    result.results = [MagicMock()] if not df.empty else []
    return result


def make_results_result(records: list[dict]) -> MagicMock:
    """Mimic an OpenBB `OBBject`-like result for methods that read
    `.results` directly (bypassing `.to_df()`) — get_company_info (2026-
    08-04 fix: works around a real upstream openbb-tmx KeyError bug in its
    own .to_df() sort logic) and get_news (original design: the date field
    isn't surfaced by .to_df() at all)."""
    result = MagicMock()
    items = []
    for record in records:
        item = MagicMock()
        item.model_dump.return_value = record
        items.append(item)
    result.results = items
    return result


@pytest.fixture
def provider():
    return OpenBBTMXProvider()


@pytest.fixture
def mock_obb(monkeypatch):
    """openbb's real `obb` singleton returns a NEW router-proxy object on
    every attribute access — confirmed live 2026-08-04 (`obb.equity.price is
    obb.equity.price` is False). That silently breaks the standard
    `patch("openbb.obb.equity.price.historical", ...)` pattern: the patch
    lands on one throwaway proxy instance, and the real code's own separate
    `obb.equity.price` access creates a different instance and hits the live
    API instead (this is what was happening here before — every test in
    this file was silently making real network calls, not verified until
    checked live). Patching the single `obb` name this module imports, with
    a plain MagicMock (which DOES cache child attribute access), is the only
    pattern that actually intercepts the call."""
    mock = MagicMock()
    monkeypatch.setattr("data.providers.openbb_tmx.obb", mock)
    return mock


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
async def test_get_financials_unknown_statement_raises_value_error(provider):
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
    mock_obb.equity.profile.return_value = make_results_result(
        [{"name": "Shopify Inc.", "sector": "Tech"}]
    )
    await provider.get_company_info(ticker="SHOP")
    mock_obb.equity.profile.assert_called_once()


@pytest.mark.asyncio
async def test_get_company_info_strips_suffix_and_converts_to_tmx_dot_form(provider, mock_obb):
    """Real bug fixed 2026-08-04 (86bb7j0kh), upstream in openbb-tmx's own
    equity_profile.py: its internal symbol_to_index sort keys off the
    REQUESTED symbol, but TMX's response always uses its own bare/dotted
    convention — a mismatched request format (this codebase's "RCI-B.TO")
    makes the fetch itself raise KeyError, confirmed live. Must request
    the TMX-native form ("RCI.B") instead — same transform already used
    in get_earnings_calendar."""
    mock_obb.equity.profile.return_value = make_results_result([{"name": "Rogers Class B"}])
    await provider.get_company_info(ticker="RCI-B.TO")
    _, kwargs = mock_obb.equity.profile.call_args
    assert kwargs["symbol"] == "RCI.B"


@pytest.mark.asyncio
async def test_get_company_info_returns_dict(provider, mock_obb):
    """Return type must be a plain dict (single-row profile). Reads
    `.results` directly rather than `.to_df()` — get_news/get_earnings_
    calendar both read .results too, but for a different reason (data not
    surfaced by .to_df()'s columns); here it's just the natural shape once
    .to_df()'s buggy internal sort is avoided by requesting the right
    symbol format in the first place (see the transform test above)."""
    mock_obb.equity.profile.return_value = make_results_result(
        [{"name": "Shopify Inc.", "sector": "Tech"}]
    )
    result = await provider.get_company_info(ticker="SHOP")
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_get_company_info_returns_first_result_only(provider, mock_obb):
    """If multiple results are returned, only the first one's data should be used."""
    mock_obb.equity.profile.return_value = make_results_result(
        [{"name": "Row One"}, {"name": "Row Two"}]
    )
    result = await provider.get_company_info(ticker="SHOP")
    assert result["name"] == "Row One"


@pytest.mark.asyncio
async def test_get_company_info_maps_to_normalized_shape(provider, mock_obb):
    """86bbb001k: real, confirmed gaps for TMX — no market_cap/currency field
    exists anywhere in OpenBB's EquityInfo model or TMX's own extension of
    it (confirmed live against RY 2026-08-07); currency is safe to hardcode
    "CAD" (any TSX/TSXV listing trades in CAD), market_cap is left None
    (disclosed gap, not derived via a second API call). industry uses
    industry_category (coarser, closer to how FMP/yfinance's own `industry`
    field reads) over the finer industry_group."""
    mock_obb.equity.profile.return_value = make_results_result(
        [
            {
                "name": "Royal Bank of Canada",
                "sector": "Finance",
                "industry_category": "Banking",
                "industry_group": "Diversified Banks",
                "stock_exchange": "TSX",
                "hq_country": None,
                "inc_country": "CA",
            }
        ]
    )
    result = await provider.get_company_info(ticker="RY")
    assert result == {
        "name": "Royal Bank of Canada",
        "sector": "Finance",
        "industry": "Banking",
        "market_cap": None,
        "currency": "CAD",
        "country": "CA",
        "primary_exchange": "TSX",
    }


@pytest.mark.asyncio
async def test_get_company_info_empty_results_returns_empty_dict(provider, mock_obb):
    """Real bug fixed 2026-08-04 (86bb7j0kh): the old .to_df()-based code
    would raise on a genuinely empty result rather than returning {}."""
    mock_obb.equity.profile.return_value = make_results_result([])
    result = await provider.get_company_info(ticker="ZZZZ")
    assert result == {}


# ---------- get_dividend_history ----------


def _dividend_df(dates: list[str]) -> pd.DataFrame:
    """Real TMX dividend shape, confirmed live 2026-08-04 (86bb7j0kh): a
    plain RangeIndex with an `ex_dividend_date` COLUMN, not a DatetimeIndex
    — the old fixture (DatetimeIndex, no ex_dividend_date column) didn't
    match reality and masked the real TypeError this session found live."""
    return pd.DataFrame({"ex_dividend_date": dates, "amount": [0.5] * len(dates)})


@pytest.mark.asyncio
async def test_get_dividend_history_calls_dividends_endpoint(provider, mock_obb):
    """Must call obb.equity.fundamental.dividends."""
    mock_obb.equity.fundamental.dividends.return_value = make_obb_result(
        _dividend_df(["2023-01-01", "2023-06-01"])
    )
    await provider.get_dividend_history(ticker="SHOP", from_date="2023-01-01", to_date="2023-12-31")
    mock_obb.equity.fundamental.dividends.assert_called_once()


@pytest.mark.asyncio
async def test_get_dividend_history_filters_by_date_range(provider, mock_obb):
    """Rows outside [from_date, to_date] must be excluded — filtered on the
    real `ex_dividend_date` column (2026-08-04 fix), not the index."""
    mock_obb.equity.fundamental.dividends.return_value = make_obb_result(
        _dividend_df(["2022-01-01", "2023-06-01", "2024-01-01"])
    )
    result = await provider.get_dividend_history(
        ticker="SHOP", from_date="2023-01-01", to_date="2023-12-31"
    )
    assert len(result) == 1


@pytest.mark.asyncio
async def test_get_dividend_history_returns_list_of_dicts(provider, mock_obb):
    """Return type must be list[dict], per base class contract."""
    mock_obb.equity.fundamental.dividends.return_value = make_obb_result(
        _dividend_df(["2023-06-01"])
    )
    result = await provider.get_dividend_history(
        ticker="SHOP", from_date="2023-01-01", to_date="2023-12-31"
    )
    assert isinstance(result, list)
    assert all(isinstance(row, dict) for row in result)


@pytest.mark.asyncio
async def test_get_dividend_history_empty_results_returns_empty_list(provider, mock_obb):
    """Real bug fixed 2026-08-04 (86bb7j0kh): raised TypeError comparing a
    RangeIndex against string dates for every real CA ticker — this
    confirms the genuinely-empty-results path is now also handled."""
    mock_obb.equity.fundamental.dividends.return_value = make_obb_result(pd.DataFrame())
    result = await provider.get_dividend_history(
        ticker="ZZZZ", from_date="2023-01-01", to_date="2023-12-31"
    )
    assert result == []


# ---------- get_analyst_estimates / get_analyst_ratings / get_earnings_surprises ----------
# Live-verified 2026-08-04 (ClickUp 86bb7j0kh): obb.equity.estimates.consensus
# is a real tmx-provider endpoint, not unsupported as the old stub claimed.
# Re-verified 2026-08-17 (86bbdu04a): that consensus endpoint has no
# forward-EPS field at all, so get_analyst_estimates() no longer calls it —
# get_analyst_ratings() (below) is now the one that owns the real fetch.


@pytest.mark.asyncio
async def test_get_analyst_estimates_never_calls_the_api(provider, mock_obb):
    """86bbdu04a: TMX consensus has no forward-EPS field of any kind
    (confirmed live) — nothing to fetch, so this must not call the API at
    all, unlike the old delegating implementation. Bare {}, not
    {"forward_eps": None} — matches get_company_info's/get_quote's own
    "empty dict signals no data" convention, needed for Router._is_empty()
    to let the CA chain fall through to yfinance."""
    result = await provider.get_analyst_estimates(ticker="RY.TO")
    mock_obb.equity.estimates.consensus.assert_not_called()
    assert result == {}


@pytest.mark.asyncio
async def test_get_analyst_ratings_calls_consensus_endpoint(provider, mock_obb):
    """86bbdu04a: get_analyst_ratings() now owns the real consensus fetch
    directly (previously reached only via get_analyst_estimates()'s
    delegation, before that method's shape changed)."""
    fake_df = pd.DataFrame([{"symbol": "RY", "target_consensus": 277.24, "buy_ratings": 7}])
    mock_obb.equity.estimates.consensus.return_value = make_obb_result(fake_df)
    await provider.get_analyst_ratings(ticker="RY.TO")
    _, kwargs = mock_obb.equity.estimates.consensus.call_args
    assert kwargs["symbol"] == "RY.TO"
    assert kwargs["provider"] == "tmx"


@pytest.mark.asyncio
async def test_get_analyst_ratings_returns_first_row_as_dict(provider, mock_obb):
    fake_df = pd.DataFrame([{"symbol": "RY", "target_consensus": 277.24}])
    mock_obb.equity.estimates.consensus.return_value = make_obb_result(fake_df)
    result = await provider.get_analyst_ratings(ticker="RY.TO")
    assert result == {"symbol": "RY", "target_consensus": 277.24}


@pytest.mark.asyncio
async def test_get_analyst_ratings_empty_result_returns_empty_dict(provider, mock_obb):
    mock_obb.equity.estimates.consensus.return_value = make_obb_result(pd.DataFrame())
    result = await provider.get_analyst_ratings(ticker="ZZZZ.TO")
    assert result == {}


@pytest.mark.asyncio
async def test_get_earnings_surprises_not_implemented(provider):
    with pytest.raises(NotImplementedError):
        await provider.get_earnings_surprises(ticker="RY.TO")


# ---------- get_insider_trading ----------
# Live-verified 2026-08-04: real tmx endpoint, but a quarterly aggregate
# rollup with no per-transaction date — `days` is intentionally not applied.


@pytest.mark.asyncio
async def test_get_insider_trading_calls_ownership_endpoint(provider, mock_obb):
    fake_df = pd.DataFrame([{"owner_name": "Ross, Bruce", "period": "three_months"}])
    mock_obb.equity.ownership.insider_trading.return_value = make_obb_result(fake_df)
    await provider.get_insider_trading(ticker="RY.TO")
    _, kwargs = mock_obb.equity.ownership.insider_trading.call_args
    assert kwargs["symbol"] == "RY.TO"
    assert kwargs["provider"] == "tmx"


@pytest.mark.asyncio
async def test_get_insider_trading_returns_list_of_dicts(provider, mock_obb):
    fake_df = pd.DataFrame([{"owner_name": "Ross, Bruce"}, {"owner_name": "McLaughlin, Neil"}])
    mock_obb.equity.ownership.insider_trading.return_value = make_obb_result(fake_df)
    result = await provider.get_insider_trading(ticker="RY.TO")
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(row, dict) for row in result)


@pytest.mark.asyncio
async def test_get_insider_trading_empty_result_returns_empty_list(provider, mock_obb):
    mock_obb.equity.ownership.insider_trading.return_value = make_obb_result(pd.DataFrame())
    result = await provider.get_insider_trading(ticker="ZZZZ.TO")
    assert result == []


# ---------- get_earnings_calendar ----------
# Live-verified 2026-08-04: real tmx endpoint, but bulk (no symbol filter
# server-side) — filtered client-side, same pattern as fmp.py.


@pytest.mark.asyncio
async def test_get_earnings_calendar_calls_bulk_endpoint_without_symbol(provider, mock_obb):
    """The TMX endpoint itself takes no symbol param — confirm we don't
    pass one (that would silently be ignored by obb, misleading to read)."""
    fake_df = pd.DataFrame([{"symbol": "RY", "report_date": "2026-08-03"}])
    mock_obb.equity.calendar.earnings.return_value = make_obb_result(fake_df)
    await provider.get_earnings_calendar(ticker="RY.TO")
    _, kwargs = mock_obb.equity.calendar.earnings.call_args
    assert "symbol" not in kwargs
    assert kwargs["provider"] == "tmx"


@pytest.mark.asyncio
async def test_get_earnings_calendar_filters_to_requested_symbol(provider, mock_obb):
    fake_df = pd.DataFrame(
        [
            {"symbol": "RY", "report_date": "2026-08-03"},
            {"symbol": "AC", "report_date": "2026-08-03"},
        ]
    )
    mock_obb.equity.calendar.earnings.return_value = make_obb_result(fake_df)
    result = await provider.get_earnings_calendar(ticker="RY.TO")
    assert len(result) == 1
    assert result[0]["symbol"] == "RY"


@pytest.mark.asyncio
async def test_get_earnings_calendar_converts_hyphenated_class_suffix_to_tmx_dot_form(
    provider, mock_obb
):
    """This codebase's DIR-UN.TO must match TMX's own "DIR.UN" symbol
    convention — confirmed live 2026-08-04."""
    fake_df = pd.DataFrame([{"symbol": "DIR.UN", "report_date": "2026-08-03"}])
    mock_obb.equity.calendar.earnings.return_value = make_obb_result(fake_df)
    result = await provider.get_earnings_calendar(ticker="DIR-UN.TO")
    assert len(result) == 1


@pytest.mark.asyncio
async def test_get_earnings_calendar_no_match_returns_empty_list(provider, mock_obb):
    fake_df = pd.DataFrame([{"symbol": "AC", "report_date": "2026-08-03"}])
    mock_obb.equity.calendar.earnings.return_value = make_obb_result(fake_df)
    result = await provider.get_earnings_calendar(ticker="RY.TO")
    assert result == []


# ---------- get_news ----------
# Live-verified 2026-08-04: real tmx endpoint with a real `date` field on
# each result object — not surfaced by .to_df(), so this reads .results
# directly rather than through the DataFrame helper used elsewhere.


def _fake_news_result(articles: list[dict]) -> MagicMock:
    result = MagicMock()
    items = []
    for article in articles:
        item = MagicMock()
        item.date = article.get("date")
        item.model_dump.return_value = article
        items.append(item)
    result.results = items
    return result


@pytest.mark.asyncio
async def test_get_news_calls_company_news_endpoint(provider, mock_obb):
    mock_obb.news.company.return_value = _fake_news_result([])
    await provider.get_news(ticker="RY.TO", days=30)
    _, kwargs = mock_obb.news.company.call_args
    assert kwargs["symbol"] == "RY.TO"
    assert kwargs["provider"] == "tmx"


@pytest.mark.asyncio
async def test_get_news_filters_by_days(provider, mock_obb):
    now = datetime.now()
    recent = {"date": now - timedelta(days=5), "title": "Recent"}
    old = {"date": now - timedelta(days=400), "title": "Old"}
    mock_obb.news.company.return_value = _fake_news_result([recent, old])
    result = await provider.get_news(ticker="RY.TO", days=90)
    assert len(result) == 1
    assert result[0]["title"] == "Recent"


@pytest.mark.asyncio
async def test_get_news_empty_result_returns_empty_list(provider, mock_obb):
    mock_obb.news.company.return_value = _fake_news_result([])
    result = await provider.get_news(ticker="ZZZZ.TO", days=30)
    assert result == []


# ---------- Unsupported methods (should raise NotImplementedError) ----------
# get_peers: confirmed live 2026-08-04 that obb.equity.compare.peers
# genuinely has no tmx provider — a real gap, not a missed implementation.


@pytest.mark.asyncio
async def test_get_peers_not_implemented(provider):
    with pytest.raises(NotImplementedError):
        await provider.get_peers(ticker="SHOP")


@pytest.mark.asyncio
async def test_get_analyst_recommendation_trends_not_implemented(provider):
    with pytest.raises(NotImplementedError):
        await provider.get_analyst_recommendation_trends(ticker="SHOP")
