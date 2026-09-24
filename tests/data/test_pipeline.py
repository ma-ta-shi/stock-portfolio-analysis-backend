from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd
import pytest

import data.pipeline as pipeline_module
from data.pipeline import DataPipeline, StockNotFoundError, resolve_sector_etf
from data.schemas.canadian_data_flags import CanadianDataFlags
from data.schemas.context import AnalysisContext
from data.schemas.macro_sources_bundle import MacroSourcesBundle
from data.schemas.research_sources_bundle import ManagementSignals, ResearchSourcesBundle

# Captured at collection time, before any fixture runs - patched_precompute
# replaces pipeline_module.technicals.compute_all with a fake, and since
# that's the same module object everywhere, grabbing "the real function"
# after the fixture has already patched it would just return the fake.
_REAL_TECHNICALS_COMPUTE_ALL = pipeline_module.technicals.compute_all


# --- resolve_sector_etf: every mapped entry in both tables, plus unmapped ---


@pytest.mark.parametrize(
    "sector,expected",
    [
        ("Technology", "XLK"),
        ("Financial Services", "XLF"),
        ("Healthcare", "XLV"),
        ("Consumer Defensive", "XLP"),
        ("Consumer Cyclical", "XLY"),
        ("Industrials", "XLI"),
        ("Energy", "XLE"),
        ("Basic Materials", "XLB"),
        ("Real Estate", "XLRE"),
        ("Utilities", "XLU"),
        ("Communication Services", "XLC"),
    ],
)
def test_resolve_sector_etf_us_every_mapped_sector(sector, expected):
    assert resolve_sector_etf(sector, is_ca=False) == expected


@pytest.mark.parametrize(
    "sector,expected",
    [
        ("Finance", "XFN.TO"),
        ("Energy", "XEG.TO"),
        ("Materials", "XMA.TO"),
        ("Technology", "XIT.TO"),
        ("Industrials", "XID.TO"),
        ("Real Estate", "XRE.TO"),
        ("Utilities", "XUT.TO"),
        ("Healthcare", "XHC.TO"),
        ("Media & Telecommunications", "XTC.TO"),
    ],
)
def test_resolve_sector_etf_ca_every_mapped_sector(sector, expected):
    assert resolve_sector_etf(sector, is_ca=True) == expected


def test_resolve_sector_etf_unmapped_sector_returns_none():
    assert resolve_sector_etf("Some Unheard-Of Sector", is_ca=False) is None
    assert resolve_sector_etf("Some Unheard-Of Sector", is_ca=True) is None


def test_resolve_sector_etf_none_sector_returns_none():
    assert resolve_sector_etf(None, is_ca=False) is None


def test_resolve_sector_etf_us_sector_not_valid_in_ca_table():
    """The two markets don't share a taxonomy - a US-style sector string
    must not accidentally resolve against the CA table."""
    assert resolve_sector_etf("Financial Services", is_ca=True) is None


# --- DataPipeline.prepare() ---


def _price_df(n=5):
    return pd.DataFrame({"close": [100.0 + i for i in range(n)]})


class _FakeFinancialsProvider:
    def __init__(self, fin):
        self._fin = fin

    async def normalize_financials(self, ticker):
        return self._fin


class _FakeRouter:
    """Stands in for data.pipeline.Router. Every instance shares the same
    canned responses (set on the class before each test) so both the
    subject Router and the benchmark/sector-ETF Routers pipeline.py
    constructs return consistent data."""

    is_ca: bool = False
    fin: object = SimpleNamespace()
    quote: dict = {}
    dividend_history: list = []
    peers: list = []
    price_history: object = None
    instances: list = []
    price_history_calls: list = []

    def __init__(self, stock=None, ticker=None, **provider_overrides):
        self.sources_used = {"get_price_history": "yfinance"}
        self._providers = {
            "yfinance": _FakeFinancialsProvider(self.fin),
            "edgartools": _FakeFinancialsProvider(self.fin),
        }
        type(self).instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def get_company_info(self, ticker):
        return {
            "sector": "Technology" if not self.is_ca else "Finance",
            "currency": "USD" if not self.is_ca else "CAD",
        }

    async def get_analyst_ratings(self, ticker):
        return {"consensus": "buy"}

    async def get_news(self, ticker, days):
        return []

    async def get_quote(self, ticker):
        return self.quote

    async def get_dividend_history(self, ticker, from_date, to_date):
        return self.dividend_history

    async def get_peers(self, ticker):
        return self.peers

    async def get_analyst_estimates(self, ticker):
        return {}

    async def get_earnings_surprises(self, ticker):
        return []

    async def get_price_history(self, ticker, period, interval):
        type(self).price_history_calls.append((ticker, period, interval))
        return self.price_history

    async def get_earnings_calendar(self, ticker):
        return []

    async def get_insider_trading(self, ticker):
        return []

    async def get_analyst_recommendation_trends(self, ticker):
        return [{"period": "2026-08", "buy": 5}]

    async def get_short_interest(self, ticker):
        return {"short_percent_of_float": 1.2}


def _fake_fundamentals_compute_all(**kwargs):
    return {
        "valuation_metrics": {"pe_ratio": 20.0},
        "growth_metrics": {"revenue_growth_yoy": 0.1},
        "profitability_metrics": {"gross_margin": 0.4},
        "balance_sheet_metrics": {"health_rating": "healthy"},
        "dividend_info": {"dividend_yield": None},
        "peer_metrics": {},
    }


def _fake_technicals_compute_all(**kwargs):
    return {
        "technical_indicators": {},
        "support_resistance": {},
        "trend_structure": {},
        "multi_timeframe": {},
        "pattern_metrics": {},
        "market_context": {},
        "liquidity_flags": {},
        "earnings_proximity": {},
        "price_position": {},
        "is_current": True,
        "days_old": 0,
        "preflight_warnings": [],
    }


def _fake_risk_metrics_compute_all(**kwargs):
    return {"beta": 1.1, "sharpe_ratio": 0.8}


async def _fake_summarize_news(articles):
    return {"articles": [], "sentiment_source": None}


async def _fake_build_research_sources(ticker, id_assigned_articles, stock=None):
    return ResearchSourcesBundle(
        filing_digests=[],
        transcript_excerpts=[],
        news_items=[],
        peer_blocks=[],
        management_signals=ManagementSignals(
            c_suite_changes_12mo=None,
            changes_detail="",
            insider_net_direction_90d=None,
            buyback_activity="",
            dividend_activity="",
        ),
        peer_names={},
        dual_class_flag=False,
        cik_verified=False,
        has_filing_digest=False,
        missing_sources_list=[],
        latest_filing_age_days=None,
        latest_transcript_age_days=None,
        latest_news_age_days=None,
        transcript_count=0,
        news_item_count=0,
    )


def _fake_build_canadian_data_flags(research_sources, macro_sources, analyst_consensus):
    return CanadianDataFlags(
        analyst_count=0,
        news_article_count=0,
        news_sources=[],
        sedar_filing_available=False,
        statcan_available=False,
    )


async def _fake_compute_macro_sources(
    sector, is_canadian_stock, timeline, fred, boc, finnhub, stats_canada=None, as_of=None
):
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
        rate_trend=None,
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
        statcan_unemployment_ca=None,
        statcan_housing_starts=None,
        statcan_retail_sales_yoy=None,
        statcan_age_days=None,
        ca_cpi_yoy=None,
        ca_cpi_3m_delta=None,
        ca_cpi_trend=None,
        ca_gdp_qoq=None,
        ca_gdp_4q_trend=None,
        us_cpi_yoy=None,
        us_core_cpi_yoy=None,
        us_gdp_qoq=None,
        us_gdp_4q_trend=None,
        vix_30d_avg=None,
        boc_rate_90d_delta_bp=None,
        boc_rate_trend=None,
        ca_unemployment_6m_delta=None,
    )


def _fake_build_tax_metrics_field(ticker, account_type, bundle, account_state=None, **kwargs):
    return f"tax metrics for {ticker} ({account_type})"


def _fake_assign_news_ids(articles):
    return []


@pytest.fixture
def patched_precompute(monkeypatch):
    monkeypatch.setattr(pipeline_module, "Router", _FakeRouter)
    monkeypatch.setattr(pipeline_module.fundamentals, "compute_all", _fake_fundamentals_compute_all)
    monkeypatch.setattr(pipeline_module.technicals, "compute_all", _fake_technicals_compute_all)
    monkeypatch.setattr(pipeline_module.risk_metrics, "compute_all", _fake_risk_metrics_compute_all)
    monkeypatch.setattr(pipeline_module.sentiment, "summarize_news", _fake_summarize_news)
    monkeypatch.setattr(pipeline_module, "build_research_sources", _fake_build_research_sources)
    monkeypatch.setattr(
        pipeline_module, "build_canadian_data_flags", _fake_build_canadian_data_flags
    )
    monkeypatch.setattr(pipeline_module, "compute_macro_sources", _fake_compute_macro_sources)
    monkeypatch.setattr(pipeline_module, "build_tax_metrics_field", _fake_build_tax_metrics_field)
    monkeypatch.setattr(pipeline_module, "assign_news_ids", _fake_assign_news_ids)
    monkeypatch.setattr(
        pipeline_module,
        "FredMacroDataProvider",
        lambda: SimpleNamespace(),
    )

    class _FakeAsyncCtxProvider:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    monkeypatch.setattr(pipeline_module, "BOCMacroDataProvider", lambda: _FakeAsyncCtxProvider())
    monkeypatch.setattr(pipeline_module, "FinnhubDataProvider", lambda: _FakeAsyncCtxProvider())
    monkeypatch.setattr(pipeline_module, "StatsCanadaProvider", lambda: _FakeAsyncCtxProvider())
    _FakeRouter.instances = []
    _FakeRouter.price_history_calls = []
    yield
    _FakeRouter.instances = []
    _FakeRouter.price_history_calls = []


def _make_stock(is_ca: bool):
    if is_ca:
        return SimpleNamespace(
            stock_id=uuid4(),
            canonical_ticker="RY.TO",
            company_name="Royal Bank of Canada",
            primary_exchange="TSX",
            currency="CAD",
            sector="Finance",
            industry="Banks",
            isin=None,
        )
    return SimpleNamespace(
        stock_id=uuid4(),
        canonical_ticker="AAPL",
        company_name="Apple Inc.",
        primary_exchange="NASDAQ",
        currency="USD",
        sector="Technology",
        industry="Consumer Electronics",
        isin=None,
    )


class _FakeResult:
    def __init__(self, stock):
        self._stock = stock

    def scalar_one_or_none(self):
        return self._stock


class _FakeDB:
    def __init__(self, stock):
        self._stock = stock

    async def execute(self, query):
        return _FakeResult(self._stock)


async def test_prepare_us_stock_populates_every_field(patched_precompute):
    stock = _make_stock(is_ca=False)
    _FakeRouter.is_ca = False
    _FakeRouter.fin = SimpleNamespace(currency="USD", quarters=[])
    _FakeRouter.quote = {
        "current_price": 150.0,
        "market_cap": 1e12,
        "currency": "USD",
        "high_52w": 160.0,
        "low_52w": 100.0,
    }
    _FakeRouter.dividend_history = [
        {"ex_date": "2026-01-01", "payment_date": "2026-01-15", "amount_per_share": 0.5}
    ]
    _FakeRouter.peers = ["MSFT"]
    _FakeRouter.price_history = _price_df()

    db = _FakeDB(stock)
    context = AnalysisContext(account_type="trading", timeline="medium_term")
    bundle = await DataPipeline().prepare(stock.stock_id, context, db)

    assert bundle.stock.ticker == "AAPL"
    assert bundle.benchmark_ticker == "^GSPC"
    assert bundle.canadian_data_flags is None
    assert bundle.analyst_recommendation_trends is not None
    assert bundle.short_interest == {"short_percent_of_float": 1.2}
    assert bundle.insider_activity == {"transactions": []}
    assert bundle.peer_sentiment == []
    assert bundle.tax_metrics == "tax metrics for AAPL (trading)"
    assert bundle.data_freshness["get_price_history"]
    assert bundle.data_freshness["get_price_history:benchmark"]
    assert bundle.data_freshness["get_price_history:sector_etf"]
    assert isinstance(bundle.data_vintage, datetime)
    assert bundle.news_with_sentiment == []
    assert bundle.sentiment_source is None
    # Real bug caught on review: raw_price/benchmark_price were originally
    # fetched with only "1y", but risk_metrics.py's own module docstring
    # requires >=3y of raw_price for max_drawdown_3yr_pct/recovery_3yr_days
    # to resolve at all instead of silently and permanently reading None.
    assert ("AAPL", "5y", "1d") in _FakeRouter.price_history_calls
    assert ("^GSPC", "5y", "1d") in _FakeRouter.price_history_calls
    # Locks in the 3-separate-Router-instances design (subject, benchmark,
    # sector ETF) - reusing one Router across all three get_price_history
    # calls would silently overwrite sources_used entries, the exact
    # collision bug this design avoids.
    assert len(_FakeRouter.instances) == 3


async def test_prepare_ca_stock_populates_every_field(patched_precompute):
    stock = _make_stock(is_ca=True)
    _FakeRouter.is_ca = True
    _FakeRouter.fin = SimpleNamespace(currency="CAD", quarters=[])
    _FakeRouter.quote = {
        "current_price": 120.0,
        "market_cap": 2e11,
        "currency": "CAD",
        "high_52w": 130.0,
        "low_52w": 90.0,
    }
    _FakeRouter.dividend_history = [
        {"ex_date": "2026-01-01", "payment_date": "2026-01-15", "amount_per_share": 1.0}
    ]
    _FakeRouter.peers = ["TD.TO"]
    _FakeRouter.price_history = _price_df()

    db = _FakeDB(stock)
    context = AnalysisContext(account_type="tfsa", timeline="long_term")
    bundle = await DataPipeline().prepare(stock.stock_id, context, db)

    assert bundle.stock.ticker == "RY.TO"
    assert bundle.benchmark_ticker == "^GSPTSE"
    assert bundle.canadian_data_flags is not None
    assert bundle.analyst_recommendation_trends is None
    assert bundle.tax_metrics == "tax metrics for RY.TO (tfsa)"
    assert bundle.data_freshness["get_price_history:sector_etf"]
    assert ("RY.TO", "5y", "1d") in _FakeRouter.price_history_calls
    assert ("^GSPTSE", "5y", "1d") in _FakeRouter.price_history_calls
    assert len(_FakeRouter.instances) == 3


async def test_prepare_stock_not_found_raises(patched_precompute):
    db = _FakeDB(None)
    context = AnalysisContext(account_type="trading", timeline="medium_term")
    with pytest.raises(StockNotFoundError):
        await DataPipeline().prepare(uuid4(), context, db)


def _ohlcv_with_zero_volume_day(n: int = 30) -> pd.DataFrame:
    """Real OHLCV shape (lowercase columns, matching
    tests/data/precompute/test_technicals.py's own _ohlcv convention), with
    the last bar's volume zeroed - a real trigger for technicals.py's own
    _preflight_warnings()."""
    idx = pd.bdate_range("2023-01-02", periods=n)
    close = [100.0 + i * 0.1 for i in range(n)]
    volume = [1_000_000] * n
    volume[-1] = 0
    return pd.DataFrame(
        {
            "open": [c - 0.1 for c in close],
            "high": [c + 0.5 for c in close],
            "low": [c - 0.5 for c in close],
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


async def test_prepare_propagates_a_real_preflight_warning(patched_precompute, monkeypatch):
    """Real gap from the original ticket text: every other test fakes
    technicals.compute_all entirely, so nothing has confirmed a real
    technical anomaly actually survives assembly into the final bundle.
    This test keeps technicals.compute_all real (undoing patched_precompute's
    fake for this one attribute) and feeds it real price history with a
    zero-volume day."""
    monkeypatch.setattr(pipeline_module.technicals, "compute_all", _REAL_TECHNICALS_COMPUTE_ALL)

    stock = _make_stock(is_ca=False)
    _FakeRouter.is_ca = False
    _FakeRouter.fin = SimpleNamespace(currency="USD", quarters=[])
    _FakeRouter.quote = {
        "current_price": 150.0,
        "market_cap": 1e12,
        "currency": "USD",
        "high_52w": 160.0,
        "low_52w": 100.0,
    }
    _FakeRouter.dividend_history = []
    _FakeRouter.peers = []
    _FakeRouter.price_history = _ohlcv_with_zero_volume_day()

    db = _FakeDB(stock)
    context = AnalysisContext(account_type="trading", timeline="medium_term")
    bundle = await DataPipeline().prepare(stock.stock_id, context, db)

    assert any("zero-volume" in w for w in bundle.preflight_warnings)
