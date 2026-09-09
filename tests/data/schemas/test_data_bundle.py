import json
from datetime import datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from data.schemas.canadian_data_flags import CanadianDataFlags
from data.schemas.common import StockRef
from data.schemas.context import AnalysisContext
from data.schemas.data_bundle import DataBundle, get_benchmark
from data.schemas.macro_sources_bundle import MacroSourcesBundle
from data.schemas.research_sources_bundle import ManagementSignals, ResearchSourcesBundle


def _us_stock() -> StockRef:
    return StockRef(stock_id=uuid4(), ticker="AAPL", currency="USD", exchange="NASDAQ")


def _ca_stock() -> StockRef:
    return StockRef(stock_id=uuid4(), ticker="RY.TO", currency="CAD", exchange="TSX")


def _context() -> AnalysisContext:
    return AnalysisContext(account_type="tfsa", timeline="medium_term")


def _empty_research_sources() -> ResearchSourcesBundle:
    return ResearchSourcesBundle(
        filing_digests=[],
        transcript_excerpts=[],
        news_items=[],
        peer_blocks=[],
        management_signals=ManagementSignals(
            c_suite_changes_12mo=0,
            changes_detail="",
            insider_net_direction_90d=None,
            buyback_activity="",
            dividend_activity="",
        ),
        dual_class_flag=False,
        cik_verified=False,
        sedar_filing_available=False,
        latest_filing_age_days=None,
        latest_transcript_age_days=None,
        latest_news_age_days=None,
        transcript_count=0,
        news_item_count=0,
        missing_sources_list=[],
    )


def _empty_macro_sources() -> MacroSourcesBundle:
    return MacroSourcesBundle(
        fed_funds_rate=None,
        treasury_2y=None,
        treasury_5y=None,
        treasury_10y=None,
        cpi=None,
        core_cpi=None,
        gdp=None,
        unemployment=None,
        vix=None,
        cad_usd_fred=None,
        wti_crude=None,
        us_cpi_yoy=None,
        us_core_cpi_yoy=None,
        us_gdp_qoq=None,
        us_gdp_4q_trend=None,
        vix_30d_avg=None,
        boc_rate=None,
        cad_usd=None,
        canada_bond_2y=None,
        canada_bond_5y=None,
        canada_bond_10y=None,
        canada_cpi=None,
        policy_rate_90d_delta_bp=None,
        cpi_3m_delta_pp=None,
        cad_usd_90d_change_pct=None,
        commodity_90d_change_pct=None,
        unemployment_6m_delta=None,
        boc_rate_90d_delta_bp=None,
        rate_trend=None,
        boc_rate_trend=None,
        cpi_trend=None,
        cad_trend=None,
        vix_regime=None,
        us_curve_shape=None,
        ca_curve_shape=None,
        sector_commodity_relevant=False,
        sector_commodity_name=None,
        sector_commodity_level=None,
        sector_commodity_direction=None,
        sector_commodity_age_days=None,
        bond_yields_available=False,
        usd_revenue_exposure_pct=None,
        cb_commentary_items=[],
        cb_commentary_count=0,
        cb_stance_note=None,
        policy_rate_age_days=None,
        cpi_age_days=None,
        gdp_age_days=None,
        unemployment_age_days=None,
        cad_usd_age_days=None,
        vix_age_days=None,
        sector_commodity_age_days_reliability=None,
        statcan_unemployment_ca=None,
        ca_unemployment_6m_delta=None,
        statcan_housing_starts=None,
        statcan_retail_sales_yoy=None,
        statcan_age_days=None,
        ca_cpi_yoy=None,
        ca_cpi_3m_delta=None,
        ca_cpi_trend=None,
        ca_gdp_qoq=None,
        ca_gdp_4q_trend=None,
    )


def _bundle(**overrides) -> DataBundle:
    defaults = dict(
        stock=_us_stock(),
        context=_context(),
        company_info={"name": "Apple Inc.", "exchange": "NASDAQ"},
        research_sources=_empty_research_sources(),
        valuation_metrics={},
        growth_metrics={},
        profitability_metrics={},
        balance_sheet_metrics={},
        dividend_info={},
        dividend_history=[],
        peer_metrics={},
        price_info={},
        technical_indicators={},
        support_resistance={},
        trend_structure={},
        multi_timeframe={},
        pattern_metrics={},
        market_context={},
        liquidity_flags={},
        earnings_proximity={},
        price_position={},
        is_current=True,
        days_old=0,
        preflight_warnings=[],
        news_with_sentiment=None,
        sentiment_source=None,
        analyst_consensus={},
        analyst_recommendation_trends=None,
        insider_activity={},
        short_interest=None,
        peer_sentiment=[],
        canadian_data_flags=None,
        macro_sources=_empty_macro_sources(),
        risk_metrics={},
        benchmark_ticker="^GSPC",
        data_freshness={},
        data_vintage=datetime(2026, 8, 5),
    )
    return DataBundle(**{**defaults, **overrides})


def _ca_flags() -> CanadianDataFlags:
    return CanadianDataFlags(
        sentiment_source="local_llm",
        has_transcript=False,
        transcript_source=None,
        analyst_count=0,
        news_article_count=0,
        news_sources=[],
        sedar_filing_available=False,
        statcan_available=False,
    )


# ---------- get_benchmark ----------


def test_get_benchmark_us_stock_returns_sp500():
    assert get_benchmark(_us_stock()) == "^GSPC"


def test_get_benchmark_tsx_stock_returns_tsx_composite():
    assert get_benchmark(_ca_stock()) == "^GSPTSE"


def test_get_benchmark_cad_currency_non_tsx_exchange_still_returns_tsx_composite():
    """Matches router.is_canadian()'s own logic: currency=CAD alone is
    sufficient, independent of the exchange field."""
    stock = StockRef(stock_id=uuid4(), ticker="XYZ", currency="CAD", exchange="OTHER")
    assert get_benchmark(stock) == "^GSPTSE"


def test_get_benchmark_tsxv_stock_returns_tsx_composite():
    stock = StockRef(stock_id=uuid4(), ticker="PLAN.V", currency="CAD", exchange="TSXV")
    assert get_benchmark(stock) == "^GSPTSE"


# ---------- DataBundle: valid construction ----------


def test_valid_construction_us_stock():
    bundle = _bundle()
    assert bundle.stock.ticker == "AAPL"
    assert bundle.canadian_data_flags is None


def test_valid_construction_ca_stock_with_flags():
    bundle = _bundle(
        stock=_ca_stock(),
        canadian_data_flags=_ca_flags(),
        benchmark_ticker="^GSPTSE",
    )
    assert bundle.canadian_data_flags is not None


# ---------- canadian_data_flags <-> stock consistency ----------


def test_rejects_ca_stock_with_none_flags():
    with pytest.raises(ValidationError):
        _bundle(stock=_ca_stock(), canadian_data_flags=None)


def test_rejects_us_stock_with_real_flags():
    with pytest.raises(ValidationError):
        _bundle(stock=_us_stock(), canadian_data_flags=_ca_flags())


def test_rejects_cad_currency_stock_with_none_flags():
    """Currency alone triggers the CA branch, same as get_benchmark()."""
    stock = StockRef(stock_id=uuid4(), ticker="XYZ", currency="CAD", exchange="OTHER")
    with pytest.raises(ValidationError):
        _bundle(stock=stock, canadian_data_flags=None)


# ---------- benchmark_ticker <-> stock consistency (real gap caught on
# review — previously unenforced) ----------


def test_rejects_us_stock_with_ca_benchmark():
    with pytest.raises(ValidationError):
        _bundle(stock=_us_stock(), benchmark_ticker="^GSPTSE")


def test_rejects_ca_stock_with_us_benchmark():
    with pytest.raises(ValidationError):
        _bundle(
            stock=_ca_stock(),
            canadian_data_flags=_ca_flags(),
            benchmark_ticker="^GSPC",
        )


def test_rejects_garbage_benchmark_ticker():
    with pytest.raises(ValidationError):
        _bundle(benchmark_ticker="not-a-real-benchmark")


# ---------- Finnhub US-only sentiment fields must be None for CA stocks
# (real gap caught on review — previously unenforced) ----------


def test_ca_stock_with_real_news_with_sentiment_is_valid():
    """Inverted from a reject-test (ClickUp 86ban0wf4): news_with_sentiment
    is no longer Finnhub/US-only — precompute/sentiment.py now scores both
    markets via a local LLM. See _check_us_only_sentiment_fields_are_none_
    for_ca_stocks's updated docstring in data_bundle.py."""
    bundle = _bundle(
        stock=_ca_stock(),
        canadian_data_flags=_ca_flags(),
        benchmark_ticker="^GSPTSE",
        news_with_sentiment=[{"headline": "test"}],
        sentiment_source="local_llm",
    )
    assert bundle.news_with_sentiment == [{"headline": "test"}]


def test_rejects_ca_stock_with_real_analyst_recommendation_trends():
    with pytest.raises(ValidationError):
        _bundle(
            stock=_ca_stock(),
            canadian_data_flags=_ca_flags(),
            benchmark_ticker="^GSPTSE",
            analyst_recommendation_trends=[{"period": "0m"}],
        )


def test_us_stock_may_have_none_sentiment_fields_too():
    """One-directional rule: a US stock can still legitimately be None
    here (a Finnhub call can fail for any ticker) — only CA-with-real-
    values is rejected."""
    bundle = _bundle(news_with_sentiment=None, analyst_recommendation_trends=None)
    assert bundle.news_with_sentiment is None


def test_us_stock_with_real_sentiment_fields_is_valid():
    bundle = _bundle(
        news_with_sentiment=[{"headline": "test"}],
        analyst_recommendation_trends=[{"period": "0m"}],
        sentiment_source="local_llm",
    )
    assert bundle.news_with_sentiment == [{"headline": "test"}]


# ---------- sentiment_source <-> news_with_sentiment consistency (86ban0wf4) ----------


def test_rejects_sentiment_source_set_with_no_news_with_sentiment():
    with pytest.raises(ValidationError):
        _bundle(news_with_sentiment=None, sentiment_source="local_llm")


def test_rejects_news_with_sentiment_set_with_no_sentiment_source():
    with pytest.raises(ValidationError):
        _bundle(news_with_sentiment=[{"headline": "test"}], sentiment_source=None)


def test_rejects_sentiment_source_set_with_empty_news_with_sentiment_list():
    """Empty list, not just None, counts as "nothing was scored"."""
    with pytest.raises(ValidationError):
        _bundle(news_with_sentiment=[], sentiment_source="local_llm")


def test_both_none_is_valid():
    bundle = _bundle(news_with_sentiment=None, sentiment_source=None)
    assert bundle.sentiment_source is None


# ---------- base behavior ----------


def test_rejects_negative_days_old():
    with pytest.raises(ValidationError):
        _bundle(days_old=-1)


def test_is_frozen():
    bundle = _bundle()
    with pytest.raises(ValidationError):
        bundle.is_current = False


def test_forbids_extra_fields():
    with pytest.raises(ValidationError):
        _bundle(macro_us={"legacy": True})


def test_nested_bundles_are_the_real_typed_models():
    bundle = _bundle()
    assert isinstance(bundle.research_sources, ResearchSourcesBundle)
    assert isinstance(bundle.macro_sources, MacroSourcesBundle)
    assert isinstance(bundle.stock, StockRef)
    assert isinstance(bundle.context, AnalysisContext)


# ---------- model_dump() round-trip (ClickUp 86bawp88h) ----------
# The highest-value check in the whole package: DataBundle nests every
# other contract in this module (StockRef, AnalysisContext,
# ResearchSourcesBundle, MacroSourcesBundle, CanadianDataFlags) — this
# confirms none of them collapse into an opaque blob or lose data on a
# dump/reload cycle, which matters once a bundle is cached or logged via
# structlog rather than only ever passed in-process.


def test_us_bundle_round_trips_through_model_dump():
    bundle = _bundle()
    dumped = bundle.model_dump()
    assert isinstance(dumped["stock"], dict)
    assert isinstance(dumped["research_sources"], dict)
    assert isinstance(dumped["macro_sources"], dict)
    assert dumped["canadian_data_flags"] is None
    assert DataBundle.model_validate(dumped) == bundle


def test_ca_bundle_round_trips_through_model_dump():
    """Exercises the CanadianDataFlags-populated branch specifically —
    the field that's None for the US bundle above."""
    bundle = _bundle(
        stock=_ca_stock(),
        canadian_data_flags=_ca_flags(),
        benchmark_ticker="^GSPTSE",
    )
    dumped = bundle.model_dump()
    assert isinstance(dumped["canadian_data_flags"], dict)
    assert dumped["canadian_data_flags"]["sentiment_source"] == "local_llm"
    assert DataBundle.model_validate(dumped) == bundle


def test_us_bundle_json_mode_dump_is_json_serializable_and_round_trips():
    bundle = _bundle()
    dumped = bundle.model_dump(mode="json")
    json.dumps(dumped)  # raises if any nested UUID/datetime survived as a non-primitive
    assert dumped["stock"]["stock_id"] == str(bundle.stock.stock_id)
    assert isinstance(dumped["data_vintage"], str)
    assert DataBundle.model_validate(dumped) == bundle
