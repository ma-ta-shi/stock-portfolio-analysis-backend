import itertools
from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd
import pytest

import data.pipeline as pipeline_module
from agents.capture import CaptureContext
# Mapper-reachability for constructing a real LLMCall() (this file's own
# Phase 3 tests) is handled once, globally, by tests/conftest.py -- see
# its own comment for why.
from api.tables.llm_calls import LLMCall
from data.pipeline import DataPipeline, StockNotFoundError, _precompute_llm_call_rows, resolve_sector_etf
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
    # Class-level so a test can see which provider handled which ticker, and make one fail.
    calls: list = []
    failing: set = set()

    def __init__(self, fin, name="fake"):
        self._fin = fin
        self._name = name

    async def normalize_financials(self, ticker):
        type(self).calls.append((self._name, ticker))
        if ticker in type(self).failing:
            raise RuntimeError(f"Company not found: {ticker!r}")
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
    news: list = []
    news_calls: list = []
    price_history: object = None
    instances: list = []
    price_history_calls: list = []

    def __init__(self, stock=None, ticker=None, **provider_overrides):
        self.sources_used = {"get_price_history": "yfinance"}
        self._providers = {
            "yfinance": _FakeFinancialsProvider(self.fin, "yfinance"),
            "edgartools": _FakeFinancialsProvider(self.fin, "edgartools"),
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

    async def get_news(self, ticker, days, thorough=False):
        type(self).news_calls.append((ticker, days, thorough))
        return self.news

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
        "missing_fields": [],
        "currency_mismatch": None,
        "latest_financials_period_end": "2026-06-30",
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


async def _fake_summarize_news(articles, **kwargs):
    return {"articles": [], "sentiment_source": None}


async def _fake_build_research_sources(ticker, id_assigned_articles, stock=None, **kwargs):
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
    _FakeRouter.news = []
    _FakeRouter.news_calls = []
    _FakeFinancialsProvider.calls = []
    _FakeFinancialsProvider.failing = set()
    yield
    _FakeRouter.instances = []
    _FakeRouter.price_history_calls = []
    _FakeFinancialsProvider.calls = []
    _FakeFinancialsProvider.failing = set()


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
        self.added: list = []  # 86bbwachy Phase 3 -- prepare()'s own db.add() calls land here

    async def execute(self, query):
        return _FakeResult(self._stock)

    def add(self, obj) -> None:
        self.added.append(obj)


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


# ---------- capture / llm_calls (86bbwachy Phase 3) ----------


def test_precompute_llm_call_rows_maps_capture_entries_to_llm_call_rows():
    """Direct unit test of the field mapping, independent of prepare()'s
    own wiring -- mirrors services/orchestrator.py's own _llm_call_rows,
    not shared with it (see this function's own docstring for why), so it
    gets its own equivalent coverage here."""
    capture = CaptureContext(
        run_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", ticker="AAPL", seq_counter=itertools.count()
    )
    capture.call_log = [
        {
            "call_site": "precompute:sentiment",
            "seq": 0,
            "context_tag": "analysis",
            "model": "gpt-oss:20b",
            "options_json": {"num_ctx": 8192},
            "attempt": 1,
            "total_duration_s": 5.0,
            "prompt_eval_count": 40,
            "eval_count": 8,
            "thinking_chars": 0,
            "finish_reason": "stop",
            "parsed_ok": True,
            "parse_error": None,
            "prompt_path": "2026-09-24/AAPL_aaaaaaaa/0_precompute-sentiment.1.prompt.txt",
            "context_path": None,
            "response_path": "2026-09-24/AAPL_aaaaaaaa/0_precompute-sentiment.1.response.json",
        }
    ]

    rows = _precompute_llm_call_rows(capture)

    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row, LLMCall)
    assert row.run_id == capture.run_id
    assert row.call_site == "precompute:sentiment"
    assert row.agent_pass == "precompute"
    assert row.seq == 0
    assert row.model == "gpt-oss:20b"
    assert row.latency_ms == 5000
    assert row.prompt_tokens == 40
    assert row.completion_tokens == 8
    assert row.parsed_ok is True
    assert row.agent_output_id is None  # non-agent call -- no AgentOutput row to link to
    assert row.auto_trimmed is False


async def test_prepare_without_run_id_adds_no_llm_call_rows(patched_precompute):
    """Default behavior, no run_id/seq_counter passed -- matches every
    other test in this file exactly, confirming the new Phase 3 wiring
    doesn't change prepare()'s pre-existing default posture."""
    stock = _make_stock(is_ca=False)
    _FakeRouter.is_ca = False
    _FakeRouter.fin = SimpleNamespace(currency="USD", quarters=[])
    _FakeRouter.quote = {
        "current_price": 150.0, "market_cap": 1e12, "currency": "USD",
        "high_52w": 160.0, "low_52w": 100.0,
    }
    _FakeRouter.dividend_history = []
    _FakeRouter.peers = []
    _FakeRouter.price_history = _price_df()

    db = _FakeDB(stock)
    context = AnalysisContext(account_type="trading", timeline="medium_term")
    await DataPipeline().prepare(stock.stock_id, context, db)

    assert db.added == []


async def test_prepare_with_run_id_adds_llm_call_rows_from_both_precompute_sources(
    patched_precompute, monkeypatch
):
    """Real end-to-end wiring check: fakes that actually populate
    capture.call_log (unlike patched_precompute's own default fakes, which
    accept and ignore capture) stand in for sentiment.summarize_news and
    build_research_sources, confirming prepare() reads capture.call_log
    back and db.add()s a real LLMCall row per entry from BOTH sources --
    not just one, and not silently dropped."""

    async def _fake_summarize_news_with_capture(articles, *, capture=None):
        if capture is not None:
            capture.call_log.append(
                {"call_site": "precompute:sentiment", "seq": next(capture.seq_counter), "model": "gpt-oss:20b"}
            )
        # sentiment_source must be None when news_with_sentiment is empty
        # (DataBundle's own cross-field validator) -- matches the
        # already-established _fake_summarize_news's own return shape.
        return {"articles": [], "sentiment_source": None}

    async def _fake_build_research_sources_with_capture(
        ticker, id_assigned_articles, stock=None, *, capture=None
    ):
        if capture is not None:
            capture.call_log.append(
                {"call_site": "precompute:filing_mda", "seq": next(capture.seq_counter), "model": "gpt-oss:20b"}
            )
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

    monkeypatch.setattr(pipeline_module.sentiment, "summarize_news", _fake_summarize_news_with_capture)
    monkeypatch.setattr(pipeline_module, "build_research_sources", _fake_build_research_sources_with_capture)

    stock = _make_stock(is_ca=False)
    _FakeRouter.is_ca = False
    _FakeRouter.fin = SimpleNamespace(currency="USD", quarters=[])
    _FakeRouter.quote = {
        "current_price": 150.0, "market_cap": 1e12, "currency": "USD",
        "high_52w": 160.0, "low_52w": 100.0,
    }
    _FakeRouter.dividend_history = []
    _FakeRouter.peers = []
    _FakeRouter.price_history = _price_df()

    db = _FakeDB(stock)
    context = AnalysisContext(account_type="trading", timeline="medium_term")
    run_id = uuid4()
    await DataPipeline().prepare(
        stock.stock_id, context, db, run_id=run_id, seq_counter=itertools.count()
    )

    assert len(db.added) == 2
    assert all(isinstance(row, LLMCall) for row in db.added)
    assert {row.call_site for row in db.added} == {"precompute:sentiment", "precompute:filing_mda"}
    assert all(row.run_id == run_id for row in db.added)
    assert all(row.agent_pass == "precompute" for row in db.added)
    # Both sources shared the SAME seq counter, not two independently
    # 0-based ones -- confirms the run-wide sequencing this ticket's own
    # design requires (see AnalysisOrchestrator.run()'s own comment on why).
    assert {row.seq for row in db.added} == {0, 1}


# --- peers: the financials source follows the PEER's market; one bad peer never fails the run (BB-030) ---


def _capture_peer_data(monkeypatch):
    seen = {}

    def fake_compute_all(**kwargs):
        seen["peer_data"] = kwargs["peer_data"]
        return _fake_fundamentals_compute_all(**kwargs)

    monkeypatch.setattr(pipeline_module.fundamentals, "compute_all", fake_compute_all)
    return seen


def _prepare_inputs(*, subject_is_ca: bool, peers: list[str]):
    _FakeRouter.is_ca = subject_is_ca
    _FakeRouter.fin = SimpleNamespace(currency="CAD" if subject_is_ca else "USD", quarters=[])
    _FakeRouter.quote = {"current_price": 10.0, "market_cap": 1e9, "currency": "USD"}
    _FakeRouter.price_history = _price_df()
    _FakeRouter.dividend_history = []
    _FakeRouter.peers = peers
    stock = _make_stock(is_ca=subject_is_ca)
    context = AnalysisContext(account_type="tfsa", timeline="medium_term")
    return stock, context


@pytest.mark.parametrize(
    "subject_is_ca, peer, expected_provider",
    [
        (False, "PEP", "edgartools"),  # US subject, US peer: unchanged
        (False, "PRMW.TO", "yfinance"),  # US subject, Canadian peer: the KO crash
        (True, "TRP.TO", "yfinance"),  # Canadian subject, Canadian peer: unchanged
        (True, "WIX", "yfinance"),  # Canadian subject, US peer: unchanged (the common case)
    ],
)
async def test_each_peer_uses_the_financials_source_for_its_own_market(
    patched_precompute, monkeypatch, subject_is_ca, peer, expected_provider
):
    seen = _capture_peer_data(monkeypatch)
    stock, context = _prepare_inputs(subject_is_ca=subject_is_ca, peers=[peer])

    await DataPipeline().prepare(stock.stock_id, context, _FakeDB(stock))

    peer_calls = [call for call in _FakeFinancialsProvider.calls if call[1] == peer]
    assert peer_calls == [(expected_provider, peer)]
    assert [p[0] for p in seen["peer_data"]] == [peer]


@pytest.mark.parametrize("subject_is_ca, bad_peer", [(False, "PRMW.TO"), (True, "WIX")])
async def test_a_peer_that_cannot_be_fetched_is_dropped_and_reported_not_fatal(
    patched_precompute, monkeypatch, subject_is_ca, bad_peer
):
    from data.degradation import DegradationCollector, reset_collector, set_collector

    seen = _capture_peer_data(monkeypatch)
    stock, context = _prepare_inputs(subject_is_ca=subject_is_ca, peers=["GOOD", bad_peer])
    _FakeFinancialsProvider.failing = {bad_peer}
    collector = DegradationCollector()
    token = set_collector(collector)
    try:
        await DataPipeline().prepare(stock.stock_id, context, _FakeDB(stock))
    finally:
        reset_collector(token)

    assert [p[0] for p in seen["peer_data"]] == ["GOOD"]  # the run finished without the bad peer
    events = collector.drain()
    assert [e.key for e in events] == [("pipeline", "peer_financials", "fetch_failed")]
    assert events[0].context["symbols"] == [bad_peer]
    assert bad_peer in events[0].message


async def test_the_subjects_own_financials_failing_still_fails_the_run(patched_precompute):
    """Only PEER fetches are isolated: no financials for the stock itself is a real failure."""
    stock, context = _prepare_inputs(subject_is_ca=False, peers=[])
    _FakeFinancialsProvider.failing = {"AAPL"}

    with pytest.raises(RuntimeError, match="Company not found"):
        await DataPipeline().prepare(stock.stock_id, context, _FakeDB(stock))


# --- news: fetch the window, hand each agent a bounded selection (BB-023) ---


def _raw_busy_news(days: int = 10, per_day: int = 40) -> list[dict]:
    from datetime import timedelta

    base = datetime(2026, 9, 28, 12, 0, 0)
    articles = []
    for d in range(days):
        for i in range(per_day):
            articles.append(
                {
                    "headline": f"Story {d}-{i}",
                    "summary": "",
                    "source": "Yahoo" if i % 3 else "Benzinga",
                    "url": f"https://x/{d}/{i}",
                    "published_at": (base - timedelta(days=d, minutes=i)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                }
            )
    return articles


async def test_the_main_stocks_news_is_fetched_for_the_whole_window_and_the_peers_keep_one_request(
    patched_precompute,
):
    stock, context = _prepare_inputs(subject_is_ca=False, peers=[])

    await DataPipeline().prepare(stock.stock_id, context, _FakeDB(stock))

    assert _FakeRouter.news_calls == [("AAPL", 30, True)]  # 30 days, thorough


async def test_each_agent_gets_a_bounded_selection_not_everything_fetched(
    patched_precompute, monkeypatch
):
    from data.precompute.news_id_assignment import assign_news_ids

    monkeypatch.setattr(pipeline_module, "assign_news_ids", assign_news_ids)  # the fixture stubs it out
    seen = {}

    async def spy_summarize(articles, **kwargs):
        seen["scored"] = list(articles)
        return {"articles": [{**a, "sentiment": "neutral"} for a in articles], "sentiment_source": "local_llm"}

    async def spy_research(ticker, articles, stock=None, **kwargs):
        seen["researcher"] = list(articles)
        return await _fake_build_research_sources(ticker, articles, stock=stock, **kwargs)

    def spy_technicals(**kwargs):
        seen["technicals_news"] = list(kwargs["news_ids"])
        return _fake_technicals_compute_all(**kwargs)

    monkeypatch.setattr(pipeline_module.sentiment, "summarize_news", spy_summarize)
    monkeypatch.setattr(pipeline_module, "build_research_sources", spy_research)
    monkeypatch.setattr(pipeline_module.technicals, "compute_all", spy_technicals)
    stock, context = _prepare_inputs(subject_is_ca=False, peers=[])
    _FakeRouter.news = _raw_busy_news(days=10, per_day=40)  # 400 articles

    bundle = await DataPipeline().prepare(stock.stock_id, context, _FakeDB(stock))

    assert len(seen["technicals_news"]) == 400  # price-gap explanations still see every date
    assert len(seen["scored"]) == 10 * 6  # six a day: bounded by days, not by volume
    assert len(seen["researcher"]) == 10 * 2
    shown = [a for a in bundle.news_with_sentiment if a["shown"]]
    assert len(bundle.news_with_sentiment) == 60 and len(shown) == 20
    assert bundle.news_coverage == {"window_days": 30, "fetched": 400}
    # the ids the agents cite are the same article everywhere
    scored_ids = {a["id"] for a in seen["scored"]}
    assert {a["id"] for a in shown} <= scored_ids


async def test_a_quiet_ticker_keeps_all_its_news(patched_precompute, monkeypatch):
    from data.precompute.news_id_assignment import assign_news_ids

    async def scoring(articles, **kwargs):
        return {"articles": [{**a, "sentiment": "neutral"} for a in articles], "sentiment_source": "local_llm"}

    monkeypatch.setattr(pipeline_module, "assign_news_ids", assign_news_ids)
    monkeypatch.setattr(pipeline_module.sentiment, "summarize_news", scoring)
    stock, context = _prepare_inputs(subject_is_ca=False, peers=[])
    _FakeRouter.news = _raw_busy_news(days=5, per_day=1)

    bundle = await DataPipeline().prepare(stock.stock_id, context, _FakeDB(stock))

    assert len(bundle.news_with_sentiment) == 5
    assert bundle.news_coverage == {"window_days": 30, "fetched": 5}
