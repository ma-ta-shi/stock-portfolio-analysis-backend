"""Tests for services/orchestrator.py (86bbuhjup) -- the two-pass pipeline
sequencing, gates, and DB persistence. Runner classes are mocked (patched at
their orchestrator-imported name) since exercising the real retry/validation
loop is already covered by test_base.py and each runner's own tests; these
tests are about SEQUENCING and PERSISTENCE, not LLM call mechanics.
DataPipeline.prepare() and the benchmark Router fetch are also mocked -- no
network calls, no real Ollama.
"""
from contextlib import ExitStack
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agents.base import OllamaUnavailable
from api.database import Base
from api.tables.agent_outputs import AgentOutput
from api.tables.analysis_runs import AnalysisRun, RunStatus
from api.tables.llm_calls import LLMCall
from api.tables.prediction_checkpoints import PredictionCheckpoint  # noqa: F401 -- Prediction's mapper needs this reachable
from api.tables.predictions import Prediction
from api.tables.recommendations import Recommendation
from api.tables.shadow_predictions import ShadowPrediction
from api.tables.stock import Stock
from api.tables.user_profile import UserProfile  # noqa: F401 -- AnalysisRun.user_id FK needs this reachable
from services.orchestrator import AnalysisOrchestrator, _add_agent_output_and_calls, _market_cap_bucket
from sqlalchemy import select


async def _make_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # autoflush=False, expire_on_commit=False -- matching api.database's
    # REAL AsyncSessionLocal config exactly, not just expire_on_commit. An
    # earlier version of this fixture only matched the second setting; the
    # mismatch on autoflush masked/altered exactly which of two related
    # session-state bugs a test reproduced (see test_shadow_cio_db_write_
    # failure_does_not_break_the_session_for_later_writes's own docstring).
    return async_sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)()


async def _make_run(session, **overrides) -> AnalysisRun:
    stock = Stock(
        stock_id=uuid4(), canonical_ticker="AAPL", company_name="Apple Inc.",
        primary_exchange="NASDAQ", currency="USD", sector="Technology",
    )
    session.add(stock)
    await session.commit()
    defaults = dict(
        run_id=uuid4(), user_id=uuid4(), stock_id=stock.stock_id,
        account_type="trading", timeline="medium_term", triggered_by="manual",
        status=RunStatus.QUEUED, llm_config={},
    )
    run = AnalysisRun(**{**defaults, **overrides})
    session.add(run)
    await session.commit()
    return run


def _fake_bundle(**overrides) -> SimpleNamespace:
    defaults = dict(
        stock=SimpleNamespace(ticker="AAPL", currency="USD", exchange="NASDAQ", stock_id=uuid4()),
        company_info={"name": "Apple Inc.", "sector": "Technology", "country": "US",
                      "asset_type": "equity", "market_cap": 3_000_000_000_000},
        context=SimpleNamespace(account_type="trading", timeline="medium_term"),
        data_vintage=datetime(2026, 9, 23, tzinfo=UTC),
        benchmark_ticker="^GSPC",
        price_info={"current_price": 220.0, "currency": "USD"},
        dividend_history=[],
        risk_metrics={"beta": 1.1},
        valuation_metrics={"pe_ratio": 28.0},
        growth_metrics={"revenue_growth_yoy": 0.1, "revenue_growth_3yr_cagr": 0.09},
        profitability_metrics={"margin_trend": "stable", "roe": 0.3, "net_margin": 0.25,
                                "operating_margin": 0.3, "fcf_to_net_income": 1.0},
        balance_sheet_metrics={"debt_to_equity": 1.5},
        peer_metrics={"sector_medians": {"sector_median_pe": 25.0}},
        technical_indicators={"volatility_regime_derived": "normal"},
        support_resistance={"nearest_support": 210.0, "nearest_resistance": 230.0},
        macro_sources=SimpleNamespace(
            rate_trend="pausing", cpi_trend="falling", cad_trend="stable",
            sector_commodity_direction=None, vix_regime="low",
        ),
    )
    return SimpleNamespace(**{**defaults, **overrides})


def _completed(**fields) -> dict:
    return {"analysis_confidence": "high", "confidence": 70, **fields}


def _patch_pass1(session_scope=True):
    """Returns a dict of patch targets -> AsyncMock for all 5 Pass 1 runners,
    each returning a real, agent_completed()-true result."""
    return {
        "services.orchestrator.StockResearcherRunner": _completed(),
        "services.orchestrator.FundamentalAnalystRunner": _completed(),
        "services.orchestrator.TechnicalAnalystRunner": _completed(),
        "services.orchestrator.SentimentAnalystRunner": _completed(),
        "services.orchestrator.MacroEconomistRunner": _completed(),
    }


class _StubRunner:
    """Stand-in for a runner class -- constructing it returns an instance
    whose .run()/.run_stage_b() return the given canned (result, errors),
    and whose last_timing/current_agent/session match BaseRunner's real
    shape."""

    def __init__(
        self, result=None, errors=None, stage_b_result=None, stage_b_errors=None,
        raises=None, stage_b_raises=None, session=None,
    ):
        self._result = result
        self._errors = errors or []
        self._stage_b_result = stage_b_result
        self._stage_b_errors = stage_b_errors or []
        self._raises = raises
        self._stage_b_raises = stage_b_raises
        self.last_timing = {"total_duration_s": 1.0, "eval_count": 100}
        self.last_stage_a_context = [1, 2, 3] if stage_b_result is not None else None
        self.current_agent = None
        # BaseRunner's real per-attempt call record list (86bbwachy Phase 2)
        # -- empty by default since these tests only assert on
        # AgentOutput/Recommendation/Prediction rows, not on llm_calls rows
        # themselves; _llm_call_rows() iterates this directly and would
        # AttributeError on a stub that doesn't have it at all.
        self.call_log = []
        # run_id/ticker/seq_counter: stamped post-construction by
        # AnalysisOrchestrator._prime_runner() in real code; defaulted here
        # so a stub used without going through the orchestrator (none
        # currently) still has the attributes.
        self.run_id = None
        self.ticker = None
        self.seq_counter = None
        # _llm_calls_consumed: a real BaseRunner defaults this itself (see
        # that class's own __init__) since it indexes call_log, which
        # BaseRunner also owns -- _prime_runner deliberately does NOT set
        # it. _StubRunner isn't a BaseRunner subclass, so it needs its own
        # copy of that same default.
        self._llm_calls_consumed = 0
        # None by default, matching BaseRunner's own pre-first-call state --
        # exercises _close_runner()'s real "session is None -> no-op"
        # branch. Tests that need to assert a session was actually closed
        # (not just no-op'd) pass an AsyncMock here instead.
        self.session = session

    def __call__(self, *a, **kw):
        return self

    async def run(self, *a, **kw):
        if self._raises:
            raise self._raises
        return self._result, self._errors

    async def run_stage_b(self, *a, **kw):
        if self._stage_b_raises:
            raise self._stage_b_raises
        return self._stage_b_result, self._stage_b_errors


@pytest.mark.parametrize(
    "market_cap, expected",
    [
        (None, None),
        (3_000_000_000_000, "large"),
        (10_000_000_000, "large"),  # exact boundary -- large is inclusive
        (9_999_999_999, "mid"),
        (2_000_000_000, "mid"),  # exact boundary -- mid is inclusive
        (1_999_999_999, "small"),
        (300_000_000, "small"),  # exact boundary -- small is inclusive
        (299_999_999, "micro"),
        (1, "micro"),
        (0, None),  # non-positive is bad data, not a real "micro" classification
        (-100, None),
    ],
)
def test_market_cap_bucket(market_cap, expected):
    assert _market_cap_bucket(market_cap) == expected


@pytest.mark.asyncio
async def test_two_stage_runner_does_not_duplicate_llm_calls_rows():
    """Regression test for a real bug caught on a live AAPL run
    (2026-09-24): CIORunner (and RiskAdvisorRunner) is ONE shared instance
    across Stage A and Stage B, and BaseRunner.call_log accumulates across
    both -- nothing ever clears it. orchestrator.py calls
    _add_agent_output_and_calls once per stage; without _llm_calls_consumed
    tracking which call_log entries were already turned into rows, the
    second call re-processed Stage A's own entries too, producing a real
    duplicate llm_calls row in the database (same seq/call_site, one with
    agent_output_id set from the first write, one blank from the second).
    This test drives that exact two-call pattern against a fake runner
    whose call_log grows between calls, the way the real BaseRunner's does,
    and asserts every seq appears exactly once.
    """
    session = await _make_session()
    run = await _make_run(session)

    class _FakeTwoStageRunner:
        def __init__(self):
            self.call_log = []
            self.run_id = run.run_id
            self.ticker = "AAPL"
            self.seq_counter = None
            self._llm_calls_consumed = 0
            self.last_timing = {"total_duration_s": 1.0, "eval_count": 100}

    runner = _FakeTwoStageRunner()

    # Stage A: one real attempt lands in call_log, then gets written.
    runner.call_log.append({"seq": 0, "call_site": "agent:cio_stage_a", "attempt": 1})
    _add_agent_output_and_calls(
        session, run, "cio_stage_a", "synthesis", {"stock_outlook": "neutral"}, [], runner
    )
    await session.commit()

    # Stage B: call_log now holds Stage A's entry PLUS Stage B's own new
    # one -- the real shape after BaseRunner keeps accumulating and nothing
    # resets it between the two orchestrator-level writes.
    runner.call_log.append({"seq": 1, "call_site": "agent:cio_stage_b", "attempt": 1})
    _add_agent_output_and_calls(
        session, run, "cio_stage_b", "synthesis", {"synthesis_narrative": "n"}, [], runner
    )
    await session.commit()

    rows = (await session.execute(select(LLMCall).where(LLMCall.run_id == run.run_id))).scalars().all()
    seqs = sorted(r.seq for r in rows)
    assert seqs == [0, 1], f"expected exactly one row per seq, got {seqs}"

    by_seq = {r.seq: r for r in rows}
    assert by_seq[0].agent_output_id is not None  # Stage A's own winning attempt
    assert by_seq[1].agent_output_id is not None  # Stage B's own winning attempt


@pytest.mark.asyncio
async def test_full_pipeline_happy_path_creates_recommendation_and_prediction():
    session = await _make_session()
    run = await _make_run(session)
    bundle = _fake_bundle()

    cio_stage_a = _completed(
        stock_outlook="somewhat_bullish", expected_return_tier="outperform",
        thesis_summary="Strong momentum.", key_decision_factors=[{"factor": "growth"}],
    )
    cio_stage_b = {
        "synthesis_narrative": "Buy on strength.",
        "position_sizing_recommendation": "3-5%",
        "expected_return_tier": "outperform",
        "tax_summary": {}, "risk_profile_summary": {},
    }
    shadow_result = _completed(stock_outlook="neutral", expected_return_tier="market_perform")

    with (
        patch("services.orchestrator.DataPipeline") as MockPipeline,
        patch("services.orchestrator.StockResearcherRunner", _StubRunner(_completed())),
        patch("services.orchestrator.FundamentalAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.TechnicalAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.SentimentAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.MacroEconomistRunner", _StubRunner(_completed())),
        patch("services.orchestrator.BullAdvocateRunner", _StubRunner(_completed(recommendation="bullish"))),
        patch("services.orchestrator.BearAdvocateRunner", _StubRunner(_completed(recommendation="bearish"))),
        patch("services.orchestrator.TaxStrategistRunner", _StubRunner(_completed(tax_profile={}))),
        patch(
            "services.orchestrator.RiskAdvisorRunner",
            _StubRunner(_completed(risk_profile={}), stage_b_result={"position_size_recommendation": "3-5%"}),
        ),
        patch("services.orchestrator.CIORunner", _StubRunner(cio_stage_a, stage_b_result=cio_stage_b)),
        patch("services.orchestrator.ShadowCIORunner", _StubRunner(shadow_result)),
        patch("services.orchestrator.Router") as MockRouter,
    ):
        MockPipeline.return_value.prepare = AsyncMock(return_value=bundle)
        router_instance = AsyncMock()
        router_instance.get_quote = AsyncMock(return_value={"current_price": 5800.0})
        MockRouter.return_value.__aenter__ = AsyncMock(return_value=router_instance)
        MockRouter.return_value.__aexit__ = AsyncMock(return_value=False)

        await AnalysisOrchestrator().run(run, session)

    assert run.status == RunStatus.COMPLETED
    assert run.completed_at is not None
    assert run.disagreement_score is not None

    # Context tags (86bbwachy Phase 1) -- populated from the bundle, not left
    # null on a real, otherwise-successful run.
    assert run.exchange == "NASDAQ"
    assert run.currency == "USD"
    assert run.instrument_type == "equity"
    assert run.market_cap_bucket == "large"  # $3T AAPL fixture
    assert run.market_regime is None  # nothing computes this yet -- see analysis_runs.py

    rec = (await session.execute(select(Recommendation).where(Recommendation.run_id == run.run_id))).scalar_one()
    assert rec.stock_outlook_direction == "somewhat_bullish"
    assert rec.position_size_suggestion == "3-5%"
    assert rec.synthesis_narrative == "Buy on strength."
    assert rec.expected_return_tier == "moderate"  # outperform -> moderate, lossy 5->4 tier map

    pred = (await session.execute(select(Prediction).where(Prediction.recommendation_id == rec.recommendation_id))).scalar_one()
    assert pred.price_at_recommendation == 220.0
    assert pred.benchmark_price_at_recommendation == 5800.0

    shadow = (await session.execute(select(ShadowPrediction).where(ShadowPrediction.analysis_run_id == run.run_id))).scalar_one()
    assert shadow.shadow_outlook_direction == "neutral"

    agent_outputs = (await session.execute(select(AgentOutput).where(AgentOutput.run_id == run.run_id))).scalars().all()
    # 5 pass1 + 4 pass2 + cio(synthesis, written twice: stage A and stage B) + shadow_cio = 12
    assert len(agent_outputs) == 12
    assert all(row.status == "completed" for row in agent_outputs)


@pytest.mark.asyncio
async def test_gate1_failure_stops_before_pass2():
    session = await _make_session()
    run = await _make_run(session)
    bundle = _fake_bundle()

    with (
        patch("services.orchestrator.DataPipeline") as MockPipeline,
        patch("services.orchestrator.StockResearcherRunner", _StubRunner(None, ["failed"])),
        patch("services.orchestrator.FundamentalAnalystRunner", _StubRunner(None, ["failed"])),
        patch("services.orchestrator.TechnicalAnalystRunner", _StubRunner(None, ["failed"])),
        patch("services.orchestrator.SentimentAnalystRunner", _StubRunner(None, ["failed"])),
        patch("services.orchestrator.MacroEconomistRunner", _StubRunner(None, ["failed"])),
        patch("services.orchestrator.BullAdvocateRunner") as MockBull,
    ):
        MockPipeline.return_value.prepare = AsyncMock(return_value=bundle)
        await AnalysisOrchestrator().run(run, session)
        MockBull.assert_not_called()

    assert run.status == RunStatus.FAILED
    assert run.error_log[0]["stage"] == "gate1"

    agent_outputs = (await session.execute(select(AgentOutput).where(AgentOutput.run_id == run.run_id))).scalars().all()
    assert len(agent_outputs) == 5  # all 5 pass1 rows still written despite the gate failure
    assert all(row.status == "failed" for row in agent_outputs)


@pytest.mark.asyncio
async def test_gate2_failure_skips_cio_entirely():
    """SETTLED 2026-09-02 (audit E78): a Gate 2 failure must skip the CIO
    entirely, not run it with a warnings label."""
    session = await _make_session()
    run = await _make_run(session)
    bundle = _fake_bundle()

    with (
        patch("services.orchestrator.DataPipeline") as MockPipeline,
        patch("services.orchestrator.StockResearcherRunner", _StubRunner(_completed())),
        patch("services.orchestrator.FundamentalAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.TechnicalAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.SentimentAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.MacroEconomistRunner", _StubRunner(_completed())),
        patch("services.orchestrator.BullAdvocateRunner", _StubRunner(None, ["bull failed"])),
        patch("services.orchestrator.BearAdvocateRunner", _StubRunner(_completed(recommendation="bearish"))),
        patch("services.orchestrator.TaxStrategistRunner", _StubRunner(_completed(tax_profile={}))),
        patch("services.orchestrator.RiskAdvisorRunner", _StubRunner(_completed(risk_profile={}))),
        patch("services.orchestrator.CIORunner") as MockCIO,
    ):
        MockPipeline.return_value.prepare = AsyncMock(return_value=bundle)
        await AnalysisOrchestrator().run(run, session)
        MockCIO.assert_not_called()

    assert run.status == RunStatus.FAILED
    assert run.error_log[0]["stage"] == "gate2"


@pytest.mark.asyncio
async def test_one_pass1_agent_exception_does_not_abort_the_others():
    """Per-agent containment (d): a real, unexpected exception in one Pass 1
    agent must degrade that agent, not the whole gather -- matching the
    EmptyDataError incident this requirement exists to prevent."""
    session = await _make_session()
    run = await _make_run(session)
    bundle = _fake_bundle()

    with (
        patch("services.orchestrator.DataPipeline") as MockPipeline,
        patch("services.orchestrator.StockResearcherRunner", _StubRunner(raises=ValueError("boom"))),
        patch("services.orchestrator.FundamentalAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.TechnicalAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.SentimentAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.MacroEconomistRunner", _StubRunner(_completed())),
        patch("services.orchestrator.BullAdvocateRunner", _StubRunner(_completed(recommendation="bullish"))),
        patch("services.orchestrator.BearAdvocateRunner", _StubRunner(_completed(recommendation="bearish"))),
        patch("services.orchestrator.TaxStrategistRunner", _StubRunner(_completed(tax_profile={}))),
        patch("services.orchestrator.RiskAdvisorRunner", _StubRunner(_completed(risk_profile={}))),
        patch(
            "services.orchestrator.CIORunner",
            _StubRunner(
                _completed(stock_outlook="neutral", expected_return_tier="market_perform",
                           thesis_summary="t", key_decision_factors=[]),
                stage_b_result={
                    "synthesis_narrative": "n", "position_sizing_recommendation": "1-2%",
                    "expected_return_tier": "market_perform", "tax_summary": {}, "risk_profile_summary": {},
                },
            ),
        ),
        patch("services.orchestrator.ShadowCIORunner", _StubRunner(None, ["shadow failed"])),
        patch("services.orchestrator.Router") as MockRouter,
    ):
        MockPipeline.return_value.prepare = AsyncMock(return_value=bundle)
        router_instance = AsyncMock()
        router_instance.get_quote = AsyncMock(return_value={"current_price": 5800.0})
        MockRouter.return_value.__aenter__ = AsyncMock(return_value=router_instance)
        MockRouter.return_value.__aexit__ = AsyncMock(return_value=False)

        await AnalysisOrchestrator().run(run, session)

    # RSRCH failed, but the other 4 Pass 1 agents + the whole rest of the
    # pipeline still ran to completion.
    assert run.status == RunStatus.COMPLETED
    outputs = (await session.execute(select(AgentOutput).where(AgentOutput.run_id == run.run_id))).scalars().all()
    rsrch_row = next(r for r in outputs if r.agent_name == "RSRCH")
    assert rsrch_row.status == "failed"
    fund_row = next(r for r in outputs if r.agent_name == "FUND")
    assert fund_row.status == "completed"


@pytest.mark.asyncio
async def test_risk_stage_b_exception_does_not_destroy_stage_a_result():
    """Real bug this locks in, found during a 2026-09-23 review while
    deduplicating _run_pass1/_run_pass2's near-identical try/except
    wrappers: the pre-fix code wrapped Risk Advisor's Stage A call AND its
    Stage B call in the SAME try/except, so a non-Ollama exception from
    Stage B (a bug, a validation crash, anything) discarded Stage A's own
    already-successful result entirely -- the exact failure mode the tax
    passthrough check's own isolation was already fixed for earlier this
    session, just missed for Risk's Stage B at the time."""
    session = await _make_session()
    run = await _make_run(session)
    bundle = _fake_bundle()

    cio_stage_a = _completed(
        stock_outlook="neutral", expected_return_tier="market_perform",
        thesis_summary="t", key_decision_factors=[],
    )
    cio_stage_b = {
        "synthesis_narrative": "n", "position_sizing_recommendation": "1-2%",
        "expected_return_tier": "market_perform", "tax_summary": {}, "risk_profile_summary": {},
    }

    with (
        patch("services.orchestrator.DataPipeline") as MockPipeline,
        patch("services.orchestrator.StockResearcherRunner", _StubRunner(_completed())),
        patch("services.orchestrator.FundamentalAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.TechnicalAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.SentimentAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.MacroEconomistRunner", _StubRunner(_completed())),
        patch("services.orchestrator.BullAdvocateRunner", _StubRunner(_completed(recommendation="bullish"))),
        patch("services.orchestrator.BearAdvocateRunner", _StubRunner(_completed(recommendation="bearish"))),
        patch("services.orchestrator.TaxStrategistRunner", _StubRunner(_completed(tax_profile={}))),
        patch(
            "services.orchestrator.RiskAdvisorRunner",
            # stage_b_result must be non-None so the stub's own
            # last_stage_a_context becomes truthy (matching a real Stage A
            # success) -- stage_b_raises then fires instead of returning it.
            _StubRunner(
                _completed(risk_profile={"beta": 1.1}),
                stage_b_result={"stop_loss_suggestion": 100.0},
                stage_b_raises=ValueError("stage b bug"),
            ),
        ),
        patch("services.orchestrator.CIORunner", _StubRunner(cio_stage_a, stage_b_result=cio_stage_b)),
        patch("services.orchestrator.ShadowCIORunner", _StubRunner(_completed(stock_outlook="neutral"))),
        patch("services.orchestrator.Router") as MockRouter,
    ):
        MockPipeline.return_value.prepare = AsyncMock(return_value=bundle)
        router_instance = AsyncMock()
        router_instance.get_quote = AsyncMock(return_value={"current_price": 5800.0})
        MockRouter.return_value.__aenter__ = AsyncMock(return_value=router_instance)
        MockRouter.return_value.__aexit__ = AsyncMock(return_value=False)

        await AnalysisOrchestrator().run(run, session)

    # The whole run still completes -- Risk's Stage B bug degrades only
    # Risk's own stage_b data, never the run overall.
    assert run.status == RunStatus.COMPLETED

    outputs = (await session.execute(select(AgentOutput).where(AgentOutput.run_id == run.run_id))).scalars().all()
    risk_row = next(r for r in outputs if r.agent_name == "risk")
    # Stage A's real result must survive: still "completed", still carrying
    # its own real risk_profile data.
    assert risk_row.status == "completed"
    assert risk_row.structured_output["risk_profile"] == {"beta": 1.1}
    # Stage B never ran to completion -- no stage_b key was added, and the
    # failure is recorded rather than silently swallowed.
    assert "stage_b" not in risk_row.structured_output
    assert "stage b bug" in risk_row.error_detail


@pytest.mark.asyncio
async def test_ollama_unavailable_aborts_the_whole_run():
    """Environment-level failure must propagate, not be contained per-agent
    -- agents/base.py's own _retry_loop docstring explains why."""
    session = await _make_session()
    run = await _make_run(session)
    bundle = _fake_bundle()

    with (
        patch("services.orchestrator.DataPipeline") as MockPipeline,
        patch("services.orchestrator.StockResearcherRunner", _StubRunner(raises=OllamaUnavailable("down"))),
        patch("services.orchestrator.FundamentalAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.TechnicalAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.SentimentAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.MacroEconomistRunner", _StubRunner(_completed())),
    ):
        MockPipeline.return_value.prepare = AsyncMock(return_value=bundle)
        with pytest.raises(OllamaUnavailable):
            await AnalysisOrchestrator().run(run, session)

    assert run.status == RunStatus.FAILED
    assert "Ollama unavailable" in run.error_log[0]["error"]

    # Real bug this locks in: asyncio.gather() propagates the first raised
    # exception immediately, orphaning the other still-running tasks AND
    # skipping the AgentOutput write loop entirely if OllamaUnavailable is
    # allowed to escape a single agent's coroutine uncaught. The other 4
    # agents here completed successfully BEFORE/ALONGSIDE the one that
    # raised -- their call-log must survive regardless, which is the whole
    # point of per-agent containment (d).
    outputs = (await session.execute(select(AgentOutput).where(AgentOutput.run_id == run.run_id))).scalars().all()
    assert len(outputs) == 5, "all 5 Pass 1 agents must be recorded, not just the one that raised"
    rsrch_row = next(r for r in outputs if r.agent_name == "RSRCH")
    assert rsrch_row.status == "failed"
    assert "down" in rsrch_row.error_detail
    for agent_id in ("FUND", "TECH", "SENT", "MACRO"):
        row = next(r for r in outputs if r.agent_name == agent_id)
        assert row.status == "completed"


async def test_ollama_unavailable_mid_pass2_still_writes_completed_agents():
    """Same bug, Pass 2 side -- covers _run_bull_bear_tax's and _run_risk's
    independently-fixed OllamaUnavailable handling."""
    session = await _make_session()
    run = await _make_run(session)
    bundle = _fake_bundle()

    with (
        patch("services.orchestrator.DataPipeline") as MockPipeline,
        patch("services.orchestrator.StockResearcherRunner", _StubRunner(_completed())),
        patch("services.orchestrator.FundamentalAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.TechnicalAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.SentimentAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.MacroEconomistRunner", _StubRunner(_completed())),
        patch("services.orchestrator.BullAdvocateRunner", _StubRunner(raises=OllamaUnavailable("down"))),
        patch("services.orchestrator.BearAdvocateRunner", _StubRunner(_completed(recommendation="bearish"))),
        patch("services.orchestrator.TaxStrategistRunner", _StubRunner(_completed(tax_profile={}))),
        patch("services.orchestrator.RiskAdvisorRunner", _StubRunner(_completed(risk_profile={}))),
    ):
        MockPipeline.return_value.prepare = AsyncMock(return_value=bundle)
        with pytest.raises(OllamaUnavailable):
            await AnalysisOrchestrator().run(run, session)

    assert run.status == RunStatus.FAILED
    outputs = (await session.execute(select(AgentOutput).where(AgentOutput.run_id == run.run_id))).scalars().all()
    pass2_rows = [r for r in outputs if r.agent_pass == "pass2"]
    assert len(pass2_rows) == 4, "all 4 Pass 2 agents must be recorded, not just the one that raised"
    for agent_id in ("bear", "tax", "risk"):
        row = next(r for r in pass2_rows if r.agent_name == agent_id)
        assert row.status == "completed"


def _pass1_pass2_all_completed_patches():
    """The 9 Pass 1/Pass 2 patches every CIO-focused test below needs,
    factored out since only the CIO runner itself varies between them."""
    return [
        patch("services.orchestrator.StockResearcherRunner", _StubRunner(_completed())),
        patch("services.orchestrator.FundamentalAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.TechnicalAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.SentimentAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.MacroEconomistRunner", _StubRunner(_completed())),
        patch("services.orchestrator.BullAdvocateRunner", _StubRunner(_completed(recommendation="bullish"))),
        patch("services.orchestrator.BearAdvocateRunner", _StubRunner(_completed(recommendation="bearish"))),
        patch("services.orchestrator.TaxStrategistRunner", _StubRunner(_completed(tax_profile={}))),
        patch("services.orchestrator.RiskAdvisorRunner", _StubRunner(_completed(risk_profile={}))),
    ]


@pytest.mark.asyncio
async def test_cio_session_closed_when_stage_a_raises_ollama_unavailable():
    """Real leak this locks in: the pre-fix code only closed cio_runner's
    session on the success path, right before the Stage B completeness
    check -- every early-exit path (an exception, or Stage A itself never
    completing) left the session open until Python's GC eventually caught
    it, producing the "Unclosed client session" warning confirmed live
    during this port's own smoke test."""
    session = await _make_session()
    run = await _make_run(session)
    bundle = _fake_bundle()
    cio_session_mock = AsyncMock()

    with ExitStack() as stack:
        MockPipeline = stack.enter_context(patch("services.orchestrator.DataPipeline"))
        for p in _pass1_pass2_all_completed_patches():
            stack.enter_context(p)
        stack.enter_context(patch(
            "services.orchestrator.CIORunner",
            _StubRunner(raises=OllamaUnavailable("down"), session=cio_session_mock),
        ))
        MockPipeline.return_value.prepare = AsyncMock(return_value=bundle)
        with pytest.raises(OllamaUnavailable):
            await AnalysisOrchestrator().run(run, session)

    cio_session_mock.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_cio_session_closed_when_stage_a_never_completes():
    """Stage A returning a real (but validation-failed-empty) result is not
    an exception at all -- the pre-fix code `return`ed right after marking
    the run FAILED, before ever reaching the close call further down."""
    session = await _make_session()
    run = await _make_run(session)
    bundle = _fake_bundle()
    cio_session_mock = AsyncMock()

    with ExitStack() as stack:
        MockPipeline = stack.enter_context(patch("services.orchestrator.DataPipeline"))
        for p in _pass1_pass2_all_completed_patches():
            stack.enter_context(p)
        stack.enter_context(patch(
            "services.orchestrator.CIORunner",
            _StubRunner(None, ["stage a never validated"], session=cio_session_mock),
        ))
        MockPipeline.return_value.prepare = AsyncMock(return_value=bundle)
        await AnalysisOrchestrator().run(run, session)

    assert run.status == RunStatus.FAILED
    cio_session_mock.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_cio_session_closed_when_stage_b_raises_ollama_unavailable():
    session = await _make_session()
    run = await _make_run(session)
    bundle = _fake_bundle()
    cio_session_mock = AsyncMock()

    with ExitStack() as stack:
        MockPipeline = stack.enter_context(patch("services.orchestrator.DataPipeline"))
        for p in _pass1_pass2_all_completed_patches():
            stack.enter_context(p)
        stack.enter_context(patch(
            "services.orchestrator.CIORunner",
            _StubRunner(
                _completed(stock_outlook="neutral", expected_return_tier="market_perform",
                           thesis_summary="t", key_decision_factors=[]),
                stage_b_raises=OllamaUnavailable("down"),
                session=cio_session_mock,
            ),
        ))
        MockPipeline.return_value.prepare = AsyncMock(return_value=bundle)
        with pytest.raises(OllamaUnavailable):
            await AnalysisOrchestrator().run(run, session)

    cio_session_mock.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_shadow_cio_failure_does_not_fail_the_primary_run():
    session = await _make_session()
    run = await _make_run(session)
    bundle = _fake_bundle()

    cio_stage_a = _completed(
        stock_outlook="neutral", expected_return_tier="market_perform",
        thesis_summary="t", key_decision_factors=[],
    )
    cio_stage_b = {
        "synthesis_narrative": "n", "position_sizing_recommendation": "1-2%",
        "expected_return_tier": "market_perform", "tax_summary": {}, "risk_profile_summary": {},
    }

    class _RaisingShadowRunner(_StubRunner):
        def __call__(self, *a, **kw):
            return self

        async def run(self, *a, **kw):
            raise RuntimeError("shadow blew up")

    with (
        patch("services.orchestrator.DataPipeline") as MockPipeline,
        patch("services.orchestrator.StockResearcherRunner", _StubRunner(_completed())),
        patch("services.orchestrator.FundamentalAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.TechnicalAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.SentimentAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.MacroEconomistRunner", _StubRunner(_completed())),
        patch("services.orchestrator.BullAdvocateRunner", _StubRunner(_completed(recommendation="bullish"))),
        patch("services.orchestrator.BearAdvocateRunner", _StubRunner(_completed(recommendation="bearish"))),
        patch("services.orchestrator.TaxStrategistRunner", _StubRunner(_completed(tax_profile={}))),
        patch("services.orchestrator.RiskAdvisorRunner", _StubRunner(_completed(risk_profile={}))),
        patch("services.orchestrator.CIORunner", _StubRunner(cio_stage_a, stage_b_result=cio_stage_b)),
        patch("services.orchestrator.ShadowCIORunner", _RaisingShadowRunner()),
        patch("services.orchestrator.Router") as MockRouter,
    ):
        MockPipeline.return_value.prepare = AsyncMock(return_value=bundle)
        router_instance = AsyncMock()
        router_instance.get_quote = AsyncMock(return_value={"current_price": 5800.0})
        MockRouter.return_value.__aenter__ = AsyncMock(return_value=router_instance)
        MockRouter.return_value.__aexit__ = AsyncMock(return_value=False)

        await AnalysisOrchestrator().run(run, session)

    assert run.status == RunStatus.COMPLETED
    shadow_rows = (await session.execute(select(ShadowPrediction).where(ShadowPrediction.analysis_run_id == run.run_id))).scalars().all()
    assert shadow_rows == []


@pytest.mark.asyncio
async def test_shadow_cio_db_write_failure_does_not_break_the_session_for_later_writes():
    """The real, narrower risk _run_shadow_cio's own db.rollback() fix
    guards against: not just "shadow raised", but "shadow raised AFTER a
    db.add()+commit() was already attempted and failed" -- SQLAlchemy
    leaves an async session in a rolled-back/inactive transaction state
    after a failed commit, and the run.status=COMPLETED commit immediately
    afterward would itself raise without an explicit rollback() first,
    turning a non-blocking shadow failure into a primary-run failure."""
    session = await _make_session()
    run = await _make_run(session)
    bundle = _fake_bundle()

    cio_stage_a = _completed(
        stock_outlook="neutral", expected_return_tier="market_perform",
        thesis_summary="t", key_decision_factors=[],
    )
    cio_stage_b = {
        "synthesis_narrative": "n", "position_sizing_recommendation": "1-2%",
        "expected_return_tier": "market_perform", "tax_summary": {}, "risk_profile_summary": {},
    }
    shadow_result = _completed(stock_outlook="somewhat_bearish", expected_return_tier="underperform")

    # Must be a GENUINE DB-level failure, not a synthesized exception raised
    # before the real commit ever runs -- an earlier version of this test
    # did exactly that (raised IntegrityError manually, calling real_commit()
    # only in the non-failing branch) and passed even with the rollback()
    # fix reverted, because the real session's transaction was never
    # actually touched, so there was nothing to roll back and nothing this
    # test was really proving. This intercepts session.add() instead and
    # corrupts the pending ShadowPrediction's NOT NULL column right before
    # it's flushed, so the REAL aiosqlite commit raises a REAL
    # IntegrityError and genuinely leaves the session's transaction broken.
    real_add = session.add

    def _corrupt_shadow_prediction_on_add(obj):
        if isinstance(obj, ShadowPrediction):
            obj.primary_confidence = None
        return real_add(obj)

    session.add = _corrupt_shadow_prediction_on_add

    with (
        patch("services.orchestrator.DataPipeline") as MockPipeline,
        patch("services.orchestrator.StockResearcherRunner", _StubRunner(_completed())),
        patch("services.orchestrator.FundamentalAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.TechnicalAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.SentimentAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.MacroEconomistRunner", _StubRunner(_completed())),
        patch("services.orchestrator.BullAdvocateRunner", _StubRunner(_completed(recommendation="bullish"))),
        patch("services.orchestrator.BearAdvocateRunner", _StubRunner(_completed(recommendation="bearish"))),
        patch("services.orchestrator.TaxStrategistRunner", _StubRunner(_completed(tax_profile={}))),
        patch("services.orchestrator.RiskAdvisorRunner", _StubRunner(_completed(risk_profile={}))),
        patch("services.orchestrator.CIORunner", _StubRunner(cio_stage_a, stage_b_result=cio_stage_b)),
        patch("services.orchestrator.ShadowCIORunner", _StubRunner(shadow_result)),
        patch("services.orchestrator.Router") as MockRouter,
    ):
        MockPipeline.return_value.prepare = AsyncMock(return_value=bundle)
        router_instance = AsyncMock()
        router_instance.get_quote = AsyncMock(return_value={"current_price": 5800.0})
        MockRouter.return_value.__aenter__ = AsyncMock(return_value=router_instance)
        MockRouter.return_value.__aexit__ = AsyncMock(return_value=False)

        # Must not raise: without the rollback() fix, this failing commit
        # would leave the session unusable for the run.status=COMPLETED
        # commit that follows, and THAT commit raising would escape this
        # call entirely (no try/except wraps it in run()).
        await AnalysisOrchestrator().run(run, session)

    assert run.status == RunStatus.COMPLETED
    # The corrupted row's own commit genuinely failed and was rolled back --
    # it must not exist, confirming the failure was real, not swallowed.
    shadow_rows = (await session.execute(select(ShadowPrediction).where(ShadowPrediction.analysis_run_id == run.run_id))).scalars().all()
    assert shadow_rows == []


@pytest.mark.asyncio
async def test_data_pipeline_failure_marks_run_failed_and_reraises():
    session = await _make_session()
    run = await _make_run(session)

    with patch("services.orchestrator.DataPipeline") as MockPipeline:
        MockPipeline.return_value.prepare = AsyncMock(side_effect=ValueError("no data"))
        with pytest.raises(ValueError):
            await AnalysisOrchestrator().run(run, session)

    assert run.status == RunStatus.FAILED
    assert run.error_log[0]["stage"] == "data_pipeline"
