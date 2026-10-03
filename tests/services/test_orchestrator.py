"""Tests for services/orchestrator.py (86bbuhjup) -- the two-pass pipeline
sequencing, gates, and DB persistence. Runner classes are mocked (patched at
their orchestrator-imported name) since exercising the real retry/validation
loop is already covered by test_base.py and each runner's own tests; these
tests are about SEQUENCING and PERSISTENCE, not LLM call mechanics.
DataPipeline.prepare() and the benchmark Router fetch are also mocked -- no
network calls, no real Ollama.
"""
import asyncio
import re
from contextlib import ExitStack
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
import structlog
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agents.base import OllamaUnavailable
from api.database import Base
from data.degradation import LINK_FAILED, LLM_REQUEST_FAILED, current_collector
from data.degradation import report as report_degradation
# PredictionCheckpoint/UserProfile mapper-reachability (neither is used
# directly in this file) is handled once, globally, by tests/conftest.py --
# see its own comment for why.
from api.tables.agent_outputs import AgentOutput
from api.tables.analysis_runs import AnalysisRun, RunStatus
from api.tables.error_records import ErrorRecord
from api.tables.llm_calls import LLMCall
from api.tables.predictions import Prediction
from api.tables.recommendations import Recommendation
from api.tables.run_quality_summary import RunQualitySummary
from api.tables.shadow_predictions import ShadowPrediction
from api.tables.stock import Stock
from api.tables.user_profile import UserProfile
from services.orchestrator import (
    AnalysisOrchestrator,
    _add_agent_output_and_calls,
    _build_tax_inputs,
    _market_cap_bucket,
)
from sqlalchemy import select, text


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
        peer_metrics={"industry_benchmark": None},
        insider_activity={"transactions": []},
        analyst_consensus={"consensus_rating": "buy", "target_mean": 250.0},
        short_interest=None,
        analyst_rating_changes=None,
        not_applicable=None,
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
    whose .run()/.run_stage_b() (CIO only) return the given canned (result, errors),
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
        self.stage_b_calls = []  # positional args of every run_stage_b() call
        self.last_timing = {"total_duration_s": 1.0, "eval_count": 100}
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
        # last_field_coverage: 86bbwachy Phase 4 -- set by the 7 in-scope
        # real runners' own run(), None for the rest (including every
        # runner these tests stub out, none of which are in that 7).
        # _agent_output_row() reads it via getattr(..., None), so this
        # isn't strictly required for existing tests to pass, but kept
        # explicit here to match every other optional attribute's own
        # documented-default style in this class.
        self.last_field_coverage = None
        # last_stale_data/last_anomalies/last_data_coverage: 86bbummwp Tier 2
        # -- same optional-attribute pattern as last_field_coverage above,
        # None by default for every runner these tests stub out.
        self.last_stale_data = None
        self.last_anomalies = None
        self.last_data_coverage = None
        # last_data_quality_assessment: 86bbummwp Tier 3 -- same pattern.
        self.last_data_quality_assessment = None
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
        self.stage_b_calls.append(a)
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
    # model is a hard requirement on LLMCall.from_call_log_entry (real
    # producers always set it) -- included here for the same reason.
    runner.call_log.append(
        {"seq": 0, "call_site": "agent:cio_stage_a", "attempt": 1, "model": "gpt-oss:20b"}
    )
    _add_agent_output_and_calls(
        session, run, "cio_stage_a", "synthesis", {"stock_outlook": "neutral"}, [], runner
    )
    await session.commit()

    # Stage B: call_log now holds Stage A's entry PLUS Stage B's own new
    # one -- the real shape after BaseRunner keeps accumulating and nothing
    # resets it between the two orchestrator-level writes.
    runner.call_log.append(
        {"seq": 1, "call_site": "agent:cio_stage_b", "attempt": 1, "model": "gpt-oss:20b"}
    )
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


async def test_agent_output_persists_input_field_coverage_when_runner_sets_it():
    """86bbwachy Phase 4: a runner that stamps last_field_coverage (the 7
    in-scope agents' own run(), after build_user_message() returns) gets it
    persisted onto the AgentOutput row. Real producer shape: a runner
    in this ticket's own scope."""
    session = await _make_session()
    run = await _make_run(session)
    runner = _StubRunner(result={"recommendation": "bullish", "confidence": 70})
    runner.last_field_coverage = {"dividend_context": True, "business_description": False}

    _add_agent_output_and_calls(session, run, "FUND", "pass1", runner._result, [], runner)
    await session.commit()

    row = (
        await session.execute(select(AgentOutput).where(AgentOutput.run_id == run.run_id))
    ).scalar_one()
    assert row.input_field_coverage == {"dividend_context": True, "business_description": False}


async def test_agent_output_input_field_coverage_null_for_out_of_scope_agent():
    """Bull/Bear/CIO/Shadow CIO never set last_field_coverage -- confirms
    the column stays null (not an empty dict) for them, matching the
    column's own documented nullability."""
    session = await _make_session()
    run = await _make_run(session)
    runner = _StubRunner(result={"recommendation": "bullish", "confidence": 70})
    assert runner.last_field_coverage is None  # the _StubRunner default itself

    _add_agent_output_and_calls(session, run, "bull", "pass2", runner._result, [], runner)
    await session.commit()

    row = (
        await session.execute(select(AgentOutput).where(AgentOutput.run_id == run.run_id))
    ).scalar_one()
    assert row.input_field_coverage is None


async def test_agent_output_persists_mechanical_flags_when_runner_sets_them():
    """86bbummwp Tier 2 + Tier 3: a runner that stamps last_stale_data/
    last_anomalies/last_data_coverage/last_data_quality_assessment gets each
    persisted onto its own AgentOutput column, the same getattr/None-default
    pattern as input_field_coverage above."""
    session = await _make_session()
    run = await _make_run(session)
    runner = _StubRunner(result={"analysis_confidence": "medium"})
    runner.last_stale_data = ["price"]
    runner.last_anomalies = [">25% gap on 2026-09-10 with no corresponding news item"]
    runner.last_data_coverage = {"present": ["news_block"], "absent": ["short_interest"]}
    runner.last_data_quality_assessment = "low"

    _add_agent_output_and_calls(session, run, "TECH", "pass1", runner._result, [], runner)
    await session.commit()

    row = (
        await session.execute(select(AgentOutput).where(AgentOutput.run_id == run.run_id))
    ).scalar_one()
    assert row.stale_data == ["price"]
    assert row.anomalies == [">25% gap on 2026-09-10 with no corresponding news item"]
    assert row.data_coverage == {"present": ["news_block"], "absent": ["short_interest"]}
    assert row.data_quality_assessment == "low"


async def test_agent_output_mechanical_flags_null_for_agent_without_that_check():
    """A runner that never sets last_stale_data/last_anomalies/
    last_data_quality_assessment (in production, every Pass 2/CIO/Shadow CIO
    runner, plus any future Pass 1 agent that hasn't implemented a given
    check yet -- as of 86bbummwp Tier 3, all 5 Pass 1 agents implement all 4)
    must read back None for the checks it doesn't implement -- NOT an empty
    list/string, which would read as "checked, found nothing" instead of
    "not implemented"."""
    session = await _make_session()
    run = await _make_run(session)
    runner = _StubRunner(result={"analysis_confidence": "high"})
    runner.last_data_coverage = {"present": ["peers_block"], "absent": []}
    assert runner.last_stale_data is None
    assert runner.last_anomalies is None
    assert runner.last_data_quality_assessment is None

    _add_agent_output_and_calls(session, run, "FUND", "pass1", runner._result, [], runner)
    await session.commit()

    row = (
        await session.execute(select(AgentOutput).where(AgentOutput.run_id == run.run_id))
    ).scalar_one()
    assert row.stale_data is None
    assert row.anomalies is None
    assert row.data_quality_assessment is None
    assert row.data_coverage == {"present": ["peers_block"], "absent": []}


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
        "expected_return_tier": "outperform",
        "tax_summary": {},
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
            _StubRunner(_completed(risk_profile={})),
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
    assert rec.position_size_suggestion is None  # sizing removed 2026-10-01; column kept, never written
    assert rec.synthesis_narrative == "Buy on strength."
    assert rec.expected_return_tier == "outperform"  # 86bbt1kpj: 5-tier vocabulary stored as-is, no lossy DB collapse

    pred = (await session.execute(select(Prediction).where(Prediction.recommendation_id == rec.recommendation_id))).scalar_one()
    assert pred.price_at_recommendation == 220.0
    assert pred.benchmark_price_at_recommendation == 5800.0

    shadow = (await session.execute(select(ShadowPrediction).where(ShadowPrediction.analysis_run_id == run.run_id))).scalar_one()
    assert shadow.shadow_outlook_direction == "neutral"
    assert shadow.primary_expected_return_tier == "outperform"
    assert shadow.shadow_expected_return_tier == "market_perform"
    assert shadow.primary_cio_outlook_distance == 1  # somewhat_bullish(1) vs neutral(0)
    assert shadow.high_divergence is False
    assert shadow.divergence_magnitude == "minor"

    agent_outputs = (await session.execute(select(AgentOutput).where(AgentOutput.run_id == run.run_id))).scalars().all()
    # 5 pass1 + 4 pass2 + cio(synthesis, written twice: stage A and stage B) + shadow_cio = 12
    assert len(agent_outputs) == 12
    assert all(row.status == "completed" for row in agent_outputs)


# --- 86bc8efvb: real UserProfile -> account_state/user_tax_profile wiring ---


def test_build_tax_inputs_none_when_no_profile():
    assert _build_tax_inputs(None) == (None, None)


def test_build_tax_inputs_converts_dollars_to_cents_and_passes_as_of():
    profile = UserProfile(
        user_id=uuid4(),
        display_name="test",
        province="ON",
        income_annual=95_000.0,
        tfsa_room_remaining=7_500.0,
        tfsa_room_as_of=datetime(2026, 9, 1),
    )
    account_state, user_tax_profile = _build_tax_inputs(profile)
    assert account_state["tfsa_room_remaining_cents"] == 750_000
    assert account_state["tfsa_room_as_of"] == datetime(2026, 9, 1)
    assert "rrsp_room_remaining_cents" not in account_state  # unset fields aren't fabricated
    assert user_tax_profile == {"province": "ON", "income_annual": 95_000.0}


def test_build_tax_inputs_returns_none_not_empty_dict_when_profile_has_nothing_set():
    """A UserProfile row existing at all doesn't guarantee it has any of the
    relevant fields set -- an all-unset profile should read the same as no
    profile, not as a present-but-empty dict."""
    profile = UserProfile(user_id=uuid4(), display_name="test")
    assert _build_tax_inputs(profile) == (None, None)


@pytest.mark.asyncio
async def test_full_pipeline_real_user_profile_reaches_tax_runner():
    """The real code path this ticket exists to wire, not just the isolated
    precompute/agent units: a real UserProfile row, keyed to run.user_id,
    reaches TaxStrategistRunner.run() as account_state/user_tax_profile --
    exercised through the actual orchestrator, not a direct call to
    _build_tax_inputs() alone."""
    session = await _make_session()
    user_id = uuid4()
    session.add(
        UserProfile(
            user_id=user_id,
            display_name="test",
            province="ON",
            income_annual=95_000.0,
            tfsa_room_remaining=7_500.0,
            tfsa_room_as_of=datetime(2026, 9, 1),
        )
    )
    await session.commit()
    run = await _make_run(session, user_id=user_id)
    bundle = _fake_bundle()

    class _RecordingTaxRunner(_StubRunner):
        """Same as _StubRunner, but records the kwargs .run() was actually
        called with."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.run_calls = []

        async def run(self, *a, **kw):
            self.run_calls.append(kw)
            return await super().run(*a, **kw)

    tax_stub = _RecordingTaxRunner(_completed(tax_profile={}))

    cio_stage_a = _completed(
        stock_outlook="neutral", expected_return_tier="market_perform",
        thesis_summary="Mixed signals.", key_decision_factors=[{"factor": "valuation"}],
    )
    cio_stage_b = {
        "synthesis_narrative": "Hold.",
        "expected_return_tier": "market_perform", "tax_summary": {},
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
        patch("services.orchestrator.TaxStrategistRunner", tax_stub),
        patch(
            "services.orchestrator.RiskAdvisorRunner",
            _StubRunner(_completed(risk_profile={})),
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

    assert len(tax_stub.run_calls) == 1
    call_kwargs = tax_stub.run_calls[0]
    assert call_kwargs["user_tax_profile"] == {"province": "ON", "income_annual": 95_000.0}
    assert call_kwargs["account_state"]["tfsa_room_remaining_cents"] == 750_000


@pytest.mark.asyncio
async def test_strong_underperform_expected_return_tier_persists_without_lossy_collapse():
    """86bbt1kpj: before this fix, _RETURN_TIER_TO_RECOMMENDATION_VOCAB collapsed
    both underperform and strong_underperform to "minimal", destroying the one
    distinction the ticket exists to preserve. Asserts the real 5-tier value
    survives end-to-end through _create_recommendation and _run_shadow_cio, not
    just at the (now-removed) shim boundary."""
    session = await _make_session()
    run = await _make_run(session)
    bundle = _fake_bundle()

    cio_stage_a = _completed(
        stock_outlook="bearish", expected_return_tier="strong_underperform",
        thesis_summary="Deteriorating fundamentals.", key_decision_factors=[{"factor": "margins"}],
    )
    cio_stage_b = {
        "synthesis_narrative": "Exit position.",
        "expected_return_tier": "strong_underperform",
        "tax_summary": {},
    }
    shadow_result = _completed(stock_outlook="somewhat_bearish", expected_return_tier="underperform")

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
            _StubRunner(_completed(risk_profile={})),
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

    rec = (await session.execute(select(Recommendation).where(Recommendation.run_id == run.run_id))).scalar_one()
    assert rec.expected_return_tier == "strong_underperform"  # not "minimal"

    shadow = (await session.execute(select(ShadowPrediction).where(ShadowPrediction.analysis_run_id == run.run_id))).scalar_one()
    assert shadow.primary_expected_return_tier == "strong_underperform"
    assert shadow.shadow_expected_return_tier == "underperform"
    assert shadow.primary_cio_outlook_distance == 1  # bearish(-2) vs somewhat_bearish(-1)
    assert shadow.high_divergence is False
    assert shadow.divergence_magnitude == "minor"


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
                    "synthesis_narrative": "n",
                    "expected_return_tier": "market_perform", "tax_summary": {},
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
    """Same bug, Pass 2 side -- covers _run_bull_bear_tax's and the Risk Advisor's
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
        "synthesis_narrative": "n",
        "expected_return_tier": "market_perform", "tax_summary": {},
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
        "synthesis_narrative": "n",
        "expected_return_tier": "market_perform", "tax_summary": {},
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


# ---------- run_quality_summary (86bbwachy Phase 5) ----------


async def _summary_for(session, run) -> RunQualitySummary:
    return (
        await session.execute(select(RunQualitySummary).where(RunQualitySummary.run_id == run.run_id))
    ).scalar_one()


@pytest.mark.asyncio
async def test_run_quality_summary_written_on_happy_path():
    session = await _make_session()
    run = await _make_run(session)
    bundle = _fake_bundle()

    cio_stage_a = _completed(
        stock_outlook="somewhat_bullish", expected_return_tier="outperform",
        thesis_summary="Strong momentum.", key_decision_factors=[{"factor": "growth"}],
    )
    cio_stage_b = {
        "synthesis_narrative": "Buy on strength.",
        "expected_return_tier": "outperform",
        "tax_summary": {},
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
            _StubRunner(_completed(risk_profile={})),
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

    summary = await _summary_for(session, run)
    assert summary.wall_clock_ms >= 0
    assert summary.gate1_passed is True
    assert summary.gate2_passed is True
    assert summary.stock_outlook == "somewhat_bullish"
    assert summary.overall_confidence == 70
    # _StubRunner's call_log defaults to [] -- no real llm_calls rows exist
    # for this fixture, so the aggregate counts are honestly zero, not
    # missing data. See test_run_quality_summary_llm_execution_stats_reflect_real_llm_calls
    # for the dedicated aggregation-math test against populated call_log entries.
    assert summary.total_calls == 0
    assert summary.total_llm_ms == 0
    assert summary.slowest_call_ms is None

    # None of this fixture's Pass 2/CIO/Shadow CIO results set key_factors/
    # risks/narrative (_completed() doesn't add them) -- real regression
    # coverage for the exclusion sets caught on a live MSFT run: without
    # them, EVERY one of bull/bear/tax/risk/cio_stage_a/cio_stage_b/
    # shadow_cio showed up in agents_with_empty_risks on every real run
    # regardless of actual quality, since none of those agents' own
    # validators ever check for a "risks" field at all.
    assert "bull" not in summary.agents_with_empty_risks
    assert "bear" not in summary.agents_with_empty_risks
    assert "tax" not in summary.agents_with_empty_risks
    assert "risk" not in summary.agents_with_empty_risks
    assert "cio_stage_a" not in summary.agents_with_empty_key_factors
    assert "cio_stage_b" not in summary.agents_with_empty_key_factors
    assert "shadow_cio" not in summary.agents_with_empty_key_factors
    assert "cio_stage_a" not in summary.agents_with_empty_narrative
    assert "shadow_cio" not in summary.agents_with_empty_narrative
    # But the 5 Pass 1 agents genuinely have none of these three fields in
    # this fixture, and Pass 1 IS where all three are real, validated
    # fields -- they must still show up, not get swept away by the fix.
    assert set(summary.agents_with_empty_key_factors) >= {"RSRCH", "FUND", "TECH", "SENT", "MACRO"}
    assert set(summary.agents_with_empty_risks) >= {"RSRCH", "FUND", "TECH", "SENT", "MACRO"}


@pytest.mark.asyncio
async def test_run_quality_summary_written_on_gate1_failure():
    """A Gate 1 failure never reaches Pass 2/CIO -- gate2_*/stock_outlook
    must stay null, not some stale or fabricated value."""
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
    ):
        MockPipeline.return_value.prepare = AsyncMock(return_value=bundle)
        await AnalysisOrchestrator().run(run, session)

    summary = await _summary_for(session, run)
    assert summary.gate1_passed is False
    assert summary.gate1_reason
    assert summary.gate2_passed is None
    assert summary.gate2_reason is None
    assert summary.stock_outlook is None
    assert summary.overall_confidence is None


@pytest.mark.asyncio
async def test_run_quality_summary_written_when_data_pipeline_raises():
    """The one exit that never reaches gate1_check() at all -- both gate
    fields, and everything downstream, must stay null, and a row still
    gets written (a near-empty row is still a real, distinct signal)."""
    session = await _make_session()
    run = await _make_run(session)

    with patch("services.orchestrator.DataPipeline") as MockPipeline:
        MockPipeline.return_value.prepare = AsyncMock(side_effect=ValueError("no data"))
        with pytest.raises(ValueError):
            await AnalysisOrchestrator().run(run, session)

    summary = await _summary_for(session, run)
    assert summary.gate1_passed is None
    assert summary.gate2_passed is None
    assert summary.stock_outlook is None
    assert summary.total_calls == 0
    assert summary.wall_clock_ms >= 0


@pytest.mark.asyncio
async def test_run_quality_summary_write_failure_does_not_mask_original_exception():
    """Matches the Shadow CIO precedent (test_shadow_cio_db_write_failure_
    does_not_break_the_session_for_later_writes): a bug in this SECONDARY
    write must never replace or swallow whatever _run_pipeline itself
    raised."""
    session = await _make_session()
    run = await _make_run(session)

    with (
        patch("services.orchestrator.DataPipeline") as MockPipeline,
        patch(
            "services.orchestrator.AnalysisOrchestrator._write_run_quality_summary",
            AsyncMock(side_effect=RuntimeError("summary write bug")),
        ),
    ):
        MockPipeline.return_value.prepare = AsyncMock(side_effect=ValueError("no data"))
        with pytest.raises(ValueError):
            await AnalysisOrchestrator().run(run, session)

    # The original ValueError propagated, not the summary write's RuntimeError.
    assert run.status == RunStatus.FAILED
    summary_rows = (
        await session.execute(select(RunQualitySummary).where(RunQualitySummary.run_id == run.run_id))
    ).scalars().all()
    assert summary_rows == []  # the write never succeeded, and that's fine -- it's non-fatal


@pytest.mark.asyncio
async def test_run_quality_summary_degenerate_output_flags():
    """Empty key_factors/risks/narrative on a stored AgentOutput row --
    entirely new logic (86bbwachy Phase 5), no existing precedent -- must
    be reported by agent_name, and an agent with real content must NOT be
    flagged."""
    session = await _make_session()
    run = await _make_run(session)
    bundle = _fake_bundle()

    with (
        patch("services.orchestrator.DataPipeline") as MockPipeline,
        patch(
            "services.orchestrator.StockResearcherRunner",
            _StubRunner(_completed(key_factors=[{"factor": "real"}], risks=[{"risk": "real"}], narrative="A real narrative.")),
        ),
        patch("services.orchestrator.FundamentalAnalystRunner", _StubRunner(_completed())),  # no key_factors/risks/narrative at all
        patch("services.orchestrator.TechnicalAnalystRunner", _StubRunner(_completed(key_factors=[], risks=[], narrative=""))),
        patch("services.orchestrator.SentimentAnalystRunner", _StubRunner(None, ["failed"])),
        patch("services.orchestrator.MacroEconomistRunner", _StubRunner(None, ["failed"])),
        # Gate 1 passes on RSRCH/FUND/TECH alone -- Pass 2 still runs, so
        # Bull/Bear/Tax/Risk need real stubs too, not just Pass 1's. Bull
        # fails here deliberately (matching test_gate2_failure_skips_cio_
        # entirely's own pattern) so Gate 2 fails and CIO/Shadow CIO never
        # run -- bounds the mocking surface to exactly what this test needs.
        patch("services.orchestrator.BullAdvocateRunner", _StubRunner(None, ["bull failed"])),
        patch("services.orchestrator.BearAdvocateRunner", _StubRunner(_completed(recommendation="bearish"))),
        patch("services.orchestrator.TaxStrategistRunner", _StubRunner(_completed(tax_profile={}))),
        patch("services.orchestrator.RiskAdvisorRunner", _StubRunner(_completed(risk_profile={}))),
    ):
        MockPipeline.return_value.prepare = AsyncMock(return_value=bundle)
        await AnalysisOrchestrator().run(run, session)

    assert run.status == RunStatus.FAILED  # Gate 2 failure -- confirms the bounded mocking worked

    summary = await _summary_for(session, run)
    assert "RSRCH" not in summary.agents_with_empty_key_factors
    assert "RSRCH" not in summary.agents_with_empty_risks
    assert "RSRCH" not in summary.agents_with_empty_narrative
    # FUND (no fields at all) and TECH (explicitly empty) COMPLETED with thin
    # output: flagged by name.
    assert set(summary.agents_with_empty_key_factors) >= {"FUND", "TECH"}
    assert set(summary.agents_with_empty_risks) >= {"FUND", "TECH"}
    assert set(summary.agents_with_empty_narrative) >= {"FUND", "TECH"}
    # SENT and MACRO FAILED outright (no output at all). This test originally
    # asserted they were listed here too, which described a failure as thin
    # output; a failed agent is reported as a failure (error_log / error_records)
    # and kept out of these lists (86bc997wr).
    for empties in (
        summary.agents_with_empty_key_factors,
        summary.agents_with_empty_risks,
        summary.agents_with_empty_narrative,
    ):
        assert not {"SENT", "MACRO"} & set(empties)


@pytest.mark.asyncio
async def test_run_quality_summary_llm_execution_stats_reflect_real_llm_calls():
    """Dedicated aggregation-math test against a runner with real call_log
    entries -- the happy-path fixture's _StubRunner defaults to an empty
    call_log, so this is the only test that actually exercises the
    SELECT-and-aggregate logic against real llm_calls rows."""
    session = await _make_session()
    run = await _make_run(session)
    bundle = _fake_bundle()

    rsrch_runner = _StubRunner(_completed())
    rsrch_runner.call_log = [
        {"seq": 0, "call_site": "agent:RSRCH", "attempt": 1, "model": "gpt-oss:20b",
         "total_duration_s": 2.0, "prompt_eval_count": 500, "eval_count": 100,
         "thinking_chars": 50, "finish_reason": "stop", "validator_passed": False},
        {"seq": 1, "call_site": "agent:RSRCH", "attempt": 2, "model": "gpt-oss:20b",
         "total_duration_s": 3.5, "prompt_eval_count": 500, "eval_count": 120,
         "thinking_chars": 60, "finish_reason": "length", "validator_passed": True},
    ]
    tech_runner = _StubRunner(_completed())
    tech_runner.call_log = [
        {"seq": 2, "call_site": "agent:TECH", "attempt": 1, "model": "gpt-oss:20b",
         "total_duration_s": 1.0, "prompt_eval_count": 300, "eval_count": 80,
         "thinking_chars": 10, "finish_reason": "empty_content", "validator_passed": None},
    ]

    with (
        patch("services.orchestrator.DataPipeline") as MockPipeline,
        patch("services.orchestrator.StockResearcherRunner", rsrch_runner),
        patch("services.orchestrator.FundamentalAnalystRunner", _StubRunner(None, ["failed"])),
        patch("services.orchestrator.TechnicalAnalystRunner", tech_runner),
        patch("services.orchestrator.SentimentAnalystRunner", _StubRunner(None, ["failed"])),
        patch("services.orchestrator.MacroEconomistRunner", _StubRunner(None, ["failed"])),
        # Gate 1 passes on RSRCH/TECH alone -- Pass 2 still runs; Bull fails
        # to trigger Gate 2 failure and skip CIO/Shadow CIO, same bounded-
        # mocking approach as test_run_quality_summary_degenerate_output_flags.
        patch("services.orchestrator.BullAdvocateRunner", _StubRunner(None, ["bull failed"])),
        patch("services.orchestrator.BearAdvocateRunner", _StubRunner(_completed(recommendation="bearish"))),
        patch("services.orchestrator.TaxStrategistRunner", _StubRunner(_completed(tax_profile={}))),
        patch("services.orchestrator.RiskAdvisorRunner", _StubRunner(_completed(risk_profile={}))),
    ):
        MockPipeline.return_value.prepare = AsyncMock(return_value=bundle)
        await AnalysisOrchestrator().run(run, session)

    summary = await _summary_for(session, run)
    assert summary.total_calls == 3
    assert summary.retry_calls == 1  # only RSRCH's attempt=2 row
    assert summary.total_llm_ms == round((2.0 + 3.5 + 1.0) * 1000)
    assert summary.slowest_call_ms == round(3.5 * 1000)
    assert summary.total_prompt_tokens == 500 + 500 + 300
    assert summary.total_completion_tokens == 100 + 120 + 80
    assert summary.total_thinking_chars == 50 + 60 + 10
    assert summary.truncated_calls == 1  # finish_reason == "length"
    assert summary.empty_content_calls == 1  # finish_reason == "empty_content"
    assert summary.validator_failures == 1  # validator_passed is False (not None)


# --- error_records (86bc997wr) ------------------------------------------------
# Every failure the orchestrator handles queues a note; run() flushes them once,
# at the very end, into error_records. These drive REAL run() end to end (same
# stubbing as the tests above) and read error_records back.

_CIO_A = dict(
    stock_outlook="neutral", expected_return_tier="market_perform",
    thesis_summary="t", key_decision_factors=[],
)
_CIO_B = {
    "synthesis_narrative": "n",
    "expected_return_tier": "market_perform", "tax_summary": {},
}


def _default_stubs() -> dict:
    """Every runner completing cleanly. Tests override just the one under test."""
    return {
        "StockResearcherRunner": _StubRunner(_completed()),
        "FundamentalAnalystRunner": _StubRunner(_completed()),
        "TechnicalAnalystRunner": _StubRunner(_completed()),
        "SentimentAnalystRunner": _StubRunner(_completed()),
        "MacroEconomistRunner": _StubRunner(_completed()),
        "BullAdvocateRunner": _StubRunner(_completed(recommendation="bullish")),
        "BearAdvocateRunner": _StubRunner(_completed(recommendation="bearish")),
        "TaxStrategistRunner": _StubRunner(_completed(tax_profile={})),
        "RiskAdvisorRunner": _StubRunner(_completed(risk_profile={})),
        "CIORunner": _StubRunner(_completed(**_CIO_A), stage_b_result=_CIO_B),
        "ShadowCIORunner": _StubRunner(
            _completed(stock_outlook="somewhat_bearish", expected_return_tier="underperform")
        ),
    }


async def _run_with(session, run, stubs=None, *, prepare=None, quote=None) -> None:
    """Runs the real AnalysisOrchestrator.run() with the given runner stubs
    layered over _default_stubs(). Lets the caller's own exception propagate."""
    merged = {**_default_stubs(), **(stubs or {})}
    with ExitStack() as stack:
        MockPipeline = stack.enter_context(patch("services.orchestrator.DataPipeline"))
        for name, stub in merged.items():
            stack.enter_context(patch(f"services.orchestrator.{name}", stub))
        MockRouter = stack.enter_context(patch("services.orchestrator.Router"))
        MockPipeline.return_value.prepare = prepare or AsyncMock(return_value=_fake_bundle())
        router_instance = AsyncMock()
        router_instance.get_quote = quote or AsyncMock(return_value={"current_price": 5800.0})
        MockRouter.return_value.__aenter__ = AsyncMock(return_value=router_instance)
        MockRouter.return_value.__aexit__ = AsyncMock(return_value=False)
        await AnalysisOrchestrator().run(run, session)


async def _error_rows(session) -> list[ErrorRecord]:
    return list((await session.execute(select(ErrorRecord))).scalars())


def _kinds(rows) -> list[tuple]:
    """(component, error_type, agent_name, dedup_subtype), sorted -- what each
    row says, independent of the order they were flushed in."""
    return sorted(
        (r.component, r.error_type, r.agent_name or "", r.dedup_subtype or "") for r in rows
    )


class _RaisingShadowRunner(_StubRunner):
    async def run(self, *a, **kw):
        raise RuntimeError("shadow blew up")


@pytest.mark.asyncio
async def test_error_records_a_clean_run_writes_nothing():
    session = await _make_session()
    run = await _make_run(session)

    await _run_with(session, run)

    assert run.status == RunStatus.COMPLETED
    assert await _error_rows(session) == []


@pytest.mark.asyncio
async def test_error_records_data_pipeline_failure_has_ticker_context_and_traceback():
    """The most common real failure happens BEFORE any DataBundle exists, so the
    ticker has to come from the Stock row, not the bundle. Exactly one row: the
    exception re-raises out of _run_pipeline, and run()'s catch-all must not
    record it a second time."""
    session = await _make_session()
    run = await _make_run(session)

    with pytest.raises(ValueError):
        await _run_with(session, run, prepare=AsyncMock(side_effect=ValueError("no data")))

    rows = await _error_rows(session)
    assert _kinds(rows) == [
        ("data_pipeline", "prepare_exception", "", "ValueError@services/orchestrator.py:_run_pipeline")
    ]
    row = rows[0]
    assert row.severity == "high" and row.status == "open"
    assert row.stock_ticker == "AAPL"
    assert row.run_id == run.run_id and row.user_id == run.user_id
    assert row.message == "ValueError: no data"
    assert "ValueError: no data" in row.stack_trace
    assert row.context_json == {
        "account_type": "trading", "timeline": "medium_term", "stage": "data_pipeline",
    }
    # The existing per-run record is untouched.
    assert run.status == RunStatus.FAILED
    assert run.error_log == [{"stage": "data_pipeline", "error": "no data"}]


@pytest.mark.asyncio
async def test_error_records_ollama_down_in_pass1_is_one_row_not_one_per_agent():
    session = await _make_session()
    run = await _make_run(session)
    down = OllamaUnavailable("Cannot reach Ollama at http://localhost:19999")

    with pytest.raises(OllamaUnavailable):
        await _run_with(session, run, {
            "StockResearcherRunner": _StubRunner(raises=down),
            "FundamentalAnalystRunner": _StubRunner(raises=down),
            "TechnicalAnalystRunner": _StubRunner(raises=down),
        })

    rows = await _error_rows(session)
    assert _kinds(rows) == [("llm", "ollama_connection_failed", "", "pass1")]
    assert rows[0].severity == "critical"
    assert "localhost:19999" in rows[0].message
    assert run.status == RunStatus.FAILED and run.error_log[0]["stage"] == "pass1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stubs, stage",
    [
        ({"BullAdvocateRunner": _StubRunner(raises=OllamaUnavailable("down"))}, "pass2"),
        ({"CIORunner": _StubRunner(raises=OllamaUnavailable("down"))}, "cio_stage_a"),
        (
            {"CIORunner": _StubRunner(_completed(**_CIO_A), stage_b_raises=OllamaUnavailable("down"))},
            "cio_stage_b",
        ),
    ],
)
async def test_error_records_ollama_down_after_pass1_names_the_stage(stubs, stage):
    session = await _make_session()
    run = await _make_run(session)

    with pytest.raises(OllamaUnavailable):
        await _run_with(session, run, stubs)

    assert _kinds(await _error_rows(session)) == [("llm", "ollama_connection_failed", "", stage)]


@pytest.mark.asyncio
async def test_error_records_gate1_failure_records_each_failed_agent_and_the_gate():
    session = await _make_session()
    run = await _make_run(session)
    dead = {
        n: _StubRunner(None, ["failed"])
        for n in (
            "StockResearcherRunner", "FundamentalAnalystRunner", "TechnicalAnalystRunner",
            "SentimentAnalystRunner", "MacroEconomistRunner",
        )
    }

    await _run_with(session, run, dead)

    assert run.status == RunStatus.FAILED and run.error_log[0]["stage"] == "gate1"
    kinds = _kinds(await _error_rows(session))
    assert ("orchestrator", "gate_failed", "", "gate1") in kinds
    agent_rows = [k for k in kinds if k[0] == "agent"]
    assert sorted(k[2] for k in agent_rows) == ["FUND", "MACRO", "RSRCH", "SENT", "TECH"]
    assert all(k[1] == "all_retries_exhausted" for k in agent_rows)
    assert len(kinds) == 6


@pytest.mark.asyncio
async def test_error_records_gate2_failure_records_the_failed_advocate_and_the_gate():
    session = await _make_session()
    run = await _make_run(session)

    await _run_with(session, run, {"BullAdvocateRunner": _StubRunner(None, ["bull failed"])})

    assert run.status == RunStatus.FAILED and run.error_log[0]["stage"] == "gate2"
    assert _kinds(await _error_rows(session)) == [
        ("agent", "all_retries_exhausted", "bull", "bull:bull failed"),
        ("orchestrator", "gate_failed", "", "gate2"),
    ]


@pytest.mark.asyncio
async def test_error_records_an_agent_exception_is_recorded_with_its_traceback():
    session = await _make_session()
    run = await _make_run(session)

    await _run_with(session, run, {"StockResearcherRunner": _StubRunner(raises=ValueError("boom"))})

    assert run.status == RunStatus.COMPLETED  # one Pass 1 agent failing does not fail the run
    rows = await _error_rows(session)
    assert _kinds(rows) == [
        ("agent", "agent_exception", "RSRCH", "ValueError@services/orchestrator.py:_run_contained")
    ]
    assert rows[0].severity == "high" and "boom" in rows[0].stack_trace


@pytest.mark.asyncio
async def test_error_records_cio_stage_a_that_never_completes():
    session = await _make_session()
    run = await _make_run(session)

    await _run_with(session, run, {"CIORunner": _StubRunner(None, ["stage a never validated"])})

    assert run.status == RunStatus.FAILED
    rows = await _error_rows(session)
    assert _kinds(rows) == [
        ("agent", "all_retries_exhausted", "cio_stage_a", "cio_stage_a:stage a never validated")
    ]
    assert "stage a never validated" in rows[0].message


@pytest.mark.asyncio
async def test_error_records_cio_stage_b_that_never_completes():
    session = await _make_session()
    run = await _make_run(session)
    cio = _StubRunner(_completed(**_CIO_A), stage_b_result=None, stage_b_errors=["b never validated"])

    await _run_with(session, run, {"CIORunner": cio})

    assert run.status == RunStatus.FAILED
    assert _kinds(await _error_rows(session)) == [
        ("agent", "all_retries_exhausted", "cio_stage_b", "cio_stage_b:b never validated")
    ]


@pytest.mark.asyncio
async def test_error_records_cio_output_missing_required_fields():
    session = await _make_session()
    run = await _make_run(session)
    # agent_completed() is satisfied by any non-None field, but Recommendation
    # needs stock_outlook -- the run fails loudly instead of a bare KeyError.
    cio = _StubRunner(_completed(thesis_summary="t"), stage_b_result=_CIO_B)

    await _run_with(session, run, {"CIORunner": cio})

    assert run.status == RunStatus.FAILED and run.error_log[0]["stage"] == "recommendation"
    assert _kinds(await _error_rows(session)) == [
        ("agent", "output_validation_failed", "", "missing:stock_outlook")
    ]


@pytest.mark.asyncio
async def test_error_records_shadow_exception_is_non_blocking_but_recorded():
    session = await _make_session()
    run = await _make_run(session)

    await _run_with(session, run, {"ShadowCIORunner": _RaisingShadowRunner()})

    assert run.status == RunStatus.COMPLETED
    rows = await _error_rows(session)
    assert _kinds(rows) == [
        ("agent", "agent_exception", "shadow_cio", "RuntimeError@services/orchestrator.py:_run_shadow_cio")
    ]
    assert rows[0].severity == "medium" and "shadow blew up" in rows[0].stack_trace


@pytest.mark.asyncio
async def test_error_records_shadow_that_does_not_complete():
    session = await _make_session()
    run = await _make_run(session)

    await _run_with(session, run, {"ShadowCIORunner": _StubRunner(None, ["shadow failed"])})

    assert run.status == RunStatus.COMPLETED
    assert _kinds(await _error_rows(session)) == [
        ("agent", "all_retries_exhausted", "shadow_cio", "did_not_complete")
    ]


@pytest.mark.asyncio
async def test_error_records_shadow_with_no_outlook_to_compare():
    session = await _make_session()
    run = await _make_run(session)
    # Completed by agent_completed()'s standard, but no stock_outlook.
    shadow = _StubRunner(_completed(expected_return_tier="underperform"))

    await _run_with(session, run, {"ShadowCIORunner": shadow})

    assert run.status == RunStatus.COMPLETED
    assert _kinds(await _error_rows(session)) == [
        ("agent", "output_validation_failed", "shadow_cio", "missing_outlook")
    ]


@pytest.mark.asyncio
async def test_error_records_run_quality_summary_write_failure():
    session = await _make_session()
    run = await _make_run(session)

    with patch(
        "services.orchestrator.AnalysisOrchestrator._write_run_quality_summary",
        AsyncMock(side_effect=RuntimeError("summary write bug")),
    ):
        await _run_with(session, run)

    assert run.status == RunStatus.COMPLETED
    rows = await _error_rows(session)
    assert _kinds(rows) == [
        ("orchestrator", "pipeline_exception", "", "RuntimeError@services/orchestrator.py:_run_guarded")
    ]
    assert rows[0].context_json["stage"] == "quality_summary"
    assert rows[0].severity == "low"


@pytest.mark.asyncio
async def test_error_records_an_unhandled_exception_is_recorded_once_and_reraises():
    """An unguarded write (_create_prediction) raising leaves the run
    non-terminal -- run() records it, once, but does not itself fix the status
    (the background task's safety net does; tested in test_analysis_routes)."""
    session = await _make_session()
    run = await _make_run(session)

    with patch(
        "services.orchestrator.AnalysisOrchestrator._create_prediction",
        AsyncMock(side_effect=RuntimeError("prediction blew up")),
    ):
        with pytest.raises(RuntimeError, match="prediction blew up"):
            await _run_with(session, run)

    rows = await _error_rows(session)
    assert _kinds(rows) == [
        ("orchestrator", "pipeline_exception", "", "RuntimeError@services/orchestrator.py:_run_pipeline")
    ]
    assert rows[0].severity == "critical" and "prediction blew up" in rows[0].stack_trace
    assert run.status == RunStatus.SYNTHESIS_RUNNING


@pytest.mark.asyncio
async def test_error_records_write_failure_changes_nothing_about_the_run():
    """The table is genuinely gone, so the flush really fails. The run's status,
    its error_log, and the exception it raises must all be exactly what they
    are when recording works -- the original error is never masked."""
    session = await _make_session()
    run = await _make_run(session)
    await session.execute(text("DROP TABLE error_records"))
    await session.commit()

    with structlog.testing.capture_logs() as logs:
        with pytest.raises(ValueError, match="no data"):
            await _run_with(session, run, prepare=AsyncMock(side_effect=ValueError("no data")))

    assert run.status == RunStatus.FAILED
    assert run.error_log == [{"stage": "data_pipeline", "error": "no data"}]
    assert any(log["event"] == "error_record_write_failed" for log in logs)


@pytest.mark.asyncio
async def test_error_records_note_survives_a_poisoned_run_session():
    """Same real IntegrityError trick as the shadow write-failure test above:
    the run session's transaction is genuinely broken mid-run. The note for it
    is queued with plain values and flushed through an independent session,
    so it still lands and the run still finishes."""
    session = await _make_session()
    run = await _make_run(session)
    real_add = session.add

    def _corrupt_shadow_prediction_on_add(obj):
        if isinstance(obj, ShadowPrediction):
            obj.primary_confidence = None
        return real_add(obj)

    session.add = _corrupt_shadow_prediction_on_add

    await _run_with(session, run)

    assert run.status == RunStatus.COMPLETED
    rows = await _error_rows(session)
    assert _kinds(rows) == [
        ("agent", "agent_exception", "shadow_cio", "IntegrityError@services/orchestrator.py:_run_shadow_cio")
    ]


# --- run context on log lines; failed agents; error_log redaction (86bc997wr) ---


@pytest.mark.asyncio
async def test_run_binds_its_context_to_logging_and_unbinds_afterward():
    """Every log line emitted during a run -- from any layer -- must carry the
    run's identity, and nothing may leak into the next run served by the same
    process. Asserted on structlog's contextvars directly: capture_logs()
    replaces the processor chain, so it cannot show merged context."""
    session = await _make_session()
    run = await _make_run(session)
    seen: list[dict] = []

    class _ContextCapturingRunner(_StubRunner):
        async def run(self, *a, **kw):
            seen.append(dict(structlog.contextvars.get_contextvars()))
            return await super().run(*a, **kw)

    await _run_with(session, run, {"StockResearcherRunner": _ContextCapturingRunner(_completed())})

    assert seen, "the stubbed agent never ran"
    assert seen[0] == {
        "run_id": str(run.run_id), "account_type": "trading",
        "timeline": "medium_term", "ticker": "AAPL",
    }
    assert structlog.contextvars.get_contextvars() == {}


@pytest.mark.asyncio
async def test_run_unbinds_its_logging_context_even_when_the_run_fails():
    session = await _make_session()
    run = await _make_run(session)

    with pytest.raises(ValueError):
        await _run_with(session, run, prepare=AsyncMock(side_effect=ValueError("no data")))

    assert structlog.contextvars.get_contextvars() == {}


@pytest.mark.asyncio
async def test_quality_summary_does_not_list_a_failed_agent_as_having_empty_fields():
    """A failed agent has empty fields because it never produced anything, not
    because its answer was thin. A real failed run (Ollama down) used to show
    all five Pass 1 agents 'with empty key_factors', which read like a
    data-quality problem instead of the failure it was."""
    session = await _make_session()
    run = await _make_run(session)

    await _run_with(session, run, {"FundamentalAnalystRunner": _StubRunner(None, ["failed"])})

    summary = await _summary_for(session, run)
    listed = set(summary.agents_with_empty_key_factors or [])
    assert "FUND" not in listed  # it failed: reported as a failure, not as "empty"
    assert "RSRCH" in listed  # it completed without key_factors: still flagged


@pytest.mark.asyncio
async def test_a_run_where_every_pass1_agent_failed_has_no_empty_field_noise():
    session = await _make_session()
    run = await _make_run(session)
    dead = {
        n: _StubRunner(None, ["failed"])
        for n in (
            "StockResearcherRunner", "FundamentalAnalystRunner", "TechnicalAnalystRunner",
            "SentimentAnalystRunner", "MacroEconomistRunner",
        )
    }

    await _run_with(session, run, dead)

    summary = await _summary_for(session, run)
    assert summary.agents_with_empty_key_factors is None
    assert summary.agents_with_empty_risks is None
    assert summary.agents_with_empty_narrative is None


@pytest.mark.asyncio
async def test_error_log_never_stores_a_credential_from_an_exception():
    """analysis_runs.error_log is persisted. A provider exception can carry a
    request URL with an API key in it (open ticket 86bbq7dmj)."""
    session = await _make_session()
    run = await _make_run(session)
    leaky = ValueError("GET https://example.test/v3/profile/AAPL?apikey=SECRET123&x=1 failed")

    with pytest.raises(ValueError):
        await _run_with(session, run, prepare=AsyncMock(side_effect=leaky))

    stored = run.error_log[0]["error"]
    assert "SECRET123" not in stored
    assert "apikey=[REDACTED]" in stored and "x=1" in stored


@pytest.mark.asyncio
async def test_error_log_redacts_agent_error_strings_for_cio_failures():
    session = await _make_session()
    run = await _make_run(session)
    leaky_errors = ["upstream call failed: https://example.test/x?token=HUNTER2 returned 500"]

    await _run_with(session, run, {"CIORunner": _StubRunner(None, leaky_errors)})

    assert run.status == RunStatus.FAILED and run.error_log[0]["stage"] == "cio_stage_a"
    assert "HUNTER2" not in run.error_log[0]["error"]
    assert "token=[REDACTED]" in run.error_log[0]["error"]


# --- self-contained error rows: attempts + evidence pointers (86bc997wr) --------


def _attempt(seq: int, attempt: int, call_site: str = "agent:fund", **extra) -> dict:
    """A runner.call_log entry as BaseRunner writes it (the fields
    LLMCall.from_call_log_entry needs, plus what a diagnosis needs)."""
    slug = call_site.replace(":", "-")
    return {
        "seq": seq, "attempt": attempt, "call_site": call_site, "model": "gpt-oss:20b",
        "total_duration_s": 8.27, "prompt_eval_count": 2143, "eval_count": 380,
        "finish_reason": "stop", "parsed_ok": True, "validator_passed": False,
        "validator_errors": ["narrative: too long (1913 chars, max 1080)"],
        "prompt_path": f"2026-09-29\\AAPL_abcd1234\\{seq}_{slug}.{attempt}.prompt.txt",
        "response_path": f"2026-09-29\\AAPL_abcd1234\\{seq}_{slug}.{attempt}.response.json",
        **extra,
    }


@pytest.mark.asyncio
async def test_a_failed_agents_error_row_carries_its_attempts_and_evidence_pointers():
    """The row must be enough to start fixing from: which attempt failed and how,
    the validator's own words, and where the raw prompt and response were saved.
    The matching agent_outputs row is run_id + agent_name, so no id is needed."""
    session = await _make_session()
    run = await _make_run(session)
    fund = _StubRunner(None, ["narrative: too long (1913 chars, max 1080)"])
    fund.call_log = [
        _attempt(1, 1),
        _attempt(
            2, 2, validator_errors=["upstream https://example.test/x?apikey=SECRET123 returned 500"]
        ),
    ]

    await _run_with(session, run, {"FundamentalAnalystRunner": fund})

    assert run.status == RunStatus.COMPLETED  # one Pass 1 agent failing does not fail the run
    rows = await _error_rows(session)
    assert _kinds(rows) == [
        ("agent", "all_retries_exhausted", "FUND", "FUND:narrative: too long (# chars, max #)")
    ]
    context = rows[0].context_json
    assert context["stage"] == "pass1"
    assert context["call_site"] == "agent:fund"
    assert context["artifact_dir"] == "2026-09-29/AAPL_abcd1234"
    assert [c["attempt"] for c in context["calls"]] == [1, 2]
    assert context["calls"][0]["validator_errors"] == ["narrative: too long (1913 chars, max 1080)"]
    assert context["calls"][1]["prompt_path"].endswith("2_agent-fund.2.prompt.txt")
    assert context["calls"][1]["response_path"].endswith("2_agent-fund.2.response.json")
    assert "SECRET123" not in str(context)
    # The run's own account/timeline context is still there.
    assert context["account_type"] == "trading" and context["timeline"] == "medium_term"


@pytest.mark.asyncio
async def test_two_agents_failing_the_same_rule_are_two_fingerprints_not_one():
    """Grouping keys on the agent as well as the rule, so FUND's problem and
    TECH's problem are never merged into one recurring failure."""
    session = await _make_session()
    run = await _make_run(session)

    await _run_with(session, run, {
        "FundamentalAnalystRunner": _StubRunner(None, ["narrative: too long (1913 chars, max 1080)"]),
        "TechnicalAnalystRunner": _StubRunner(None, ["narrative: too long (1351 chars, max 1080)"]),
    })

    assert _kinds(await _error_rows(session)) == [
        ("agent", "all_retries_exhausted", "FUND", "FUND:narrative: too long (# chars, max #)"),
        ("agent", "all_retries_exhausted", "TECH", "TECH:narrative: too long (# chars, max #)"),
    ]


@pytest.mark.asyncio
async def test_a_cio_failure_row_carries_the_cio_attempts_too():
    session = await _make_session()
    run = await _make_run(session)
    cio = _StubRunner(None, ["stage a never validated"])
    cio.call_log = [_attempt(9, 1, call_site="agent:cio_stage_a", validator_errors=["thesis_summary: too long"])]

    await _run_with(session, run, {"CIORunner": cio})

    rows = await _error_rows(session)
    assert run.status == RunStatus.FAILED and len(rows) == 1
    assert rows[0].context_json["call_site"] == "agent:cio_stage_a"
    assert rows[0].context_json["calls"][0]["validator_errors"] == ["thesis_summary: too long"]


@pytest.mark.asyncio
async def test_a_failed_agent_with_no_recorded_calls_still_gets_a_row_without_call_detail():
    """Connection errors and timeouts happen before any call is logged, so a
    failed agent can legitimately have an empty call_log."""
    session = await _make_session()
    run = await _make_run(session)

    await _run_with(session, run, {"MacroEconomistRunner": _StubRunner(None, ["timed out"])})

    rows = await _error_rows(session)
    assert len(rows) == 1
    assert "calls" not in rows[0].context_json and "artifact_dir" not in rows[0].context_json
    assert rows[0].context_json["errors"] == ["timed out"]


# --- a cancelled run must not be left stuck (86bc997wr) --------------------------


@pytest.mark.asyncio
async def test_a_cancelled_run_is_ended_and_recorded_and_the_cancellation_still_propagates():
    """Ctrl+C or a server restart cancels the task mid-run. CancelledError is a
    BaseException, so none of the `except Exception` handlers saw it and the run
    stayed non-terminal forever, blocking its ticker (two real ones did)."""
    session = await _make_session()
    run = await _make_run(session)
    started = asyncio.Event()

    class _HangingRunner(_StubRunner):
        async def run(self, *a, **kw):
            started.set()
            await asyncio.sleep(3600)

    task = asyncio.create_task(
        _run_with(session, run, {"StockResearcherRunner": _HangingRunner(_completed())})
    )
    await asyncio.wait_for(started.wait(), timeout=10)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):  # never swallowed
        await task

    await session.refresh(run)
    assert run.status == RunStatus.FAILED
    assert run.error_log[0]["stage"] == "cancelled"
    rows = await _error_rows(session)
    assert _kinds(rows) == [("orchestrator", "run_cancelled", "", "cancelled")]
    assert rows[0].stock_ticker == "AAPL" and rows[0].run_id == run.run_id
    assert structlog.contextvars.get_contextvars() == {}


async def _make_file_session(tmp_path):
    """Like _make_session, but an on-disk SQLite FILE. An in-memory database is one
    shared connection, so a second 'independent' session can never contend with
    the run's own session for the write lock; a file can, and that is the case
    that matters on a real machine."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'run.db').as_posix()}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)()


@pytest.mark.asyncio
async def test_a_run_cancelled_while_its_own_session_holds_a_write_lock_is_still_ended(tmp_path):
    """Regression for a real review finding. The cancel handler writes from a
    SECOND connection. If the run's own session still holds a write transaction
    when it is cancelled (it was mid-write), that second connection would wait
    out SQLite's lock timeout and fail, leaving the run stuck non-terminal and
    the cancel note lost. The handler must release the run's session first.

    The stub also leaves an uncommitted status change on the run: if that were
    ever committed afterward it would overwrite the FAILED the handler wrote."""
    session = await _make_file_session(tmp_path)
    run = await _make_run(session)
    run_id = run.run_id  # read now: after the rollback every attribute on `run` is expired
    started = asyncio.Event()

    class _LockHoldingRunner(_StubRunner):
        async def run(self, *a, **kw):
            run.status = RunStatus.PASS2_RUNNING  # dirty, never committed
            await session.execute(text("UPDATE analysis_runs SET triggered_by = 'x'"))
            started.set()  # the run's own session now holds the write lock
            await asyncio.sleep(3600)

    task = asyncio.create_task(
        _run_with(session, run, {"StockResearcherRunner": _LockHoldingRunner(_completed())})
    )
    await asyncio.wait_for(started.wait(), timeout=10)
    began = asyncio.get_running_loop().time()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=30)
    elapsed = asyncio.get_running_loop().time() - began

    # Not held up by SQLite's ~5s lock wait, which is what the bug looked like.
    assert elapsed < 4, f"cancel cleanup took {elapsed:.1f}s: it waited on a lock"
    fresh = async_sessionmaker(bind=session.bind, expire_on_commit=False)()
    stored = (
        await fresh.execute(select(AnalysisRun).where(AnalysisRun.run_id == run_id))
    ).scalar_one()
    assert stored.status == RunStatus.FAILED  # and the dirty PASS2_RUNNING did not win
    assert stored.error_log[0]["stage"] == "cancelled"
    assert stored.triggered_by == "manual"  # the uncommitted write was rolled back
    rows = (await fresh.execute(select(ErrorRecord))).scalars().all()
    assert _kinds(rows) == [("orchestrator", "run_cancelled", "", "cancelled")]
    await fresh.close()


@pytest.mark.asyncio
async def test_a_cancelled_run_does_not_attempt_a_quality_summary():
    """Its counters are incomplete and its session was just rolled back; the run
    is already recorded as cancelled, with a reason."""
    session = await _make_session()
    run = await _make_run(session)
    started = asyncio.Event()

    class _HangingRunner(_StubRunner):
        async def run(self, *a, **kw):
            started.set()
            await asyncio.sleep(3600)

    task = asyncio.create_task(
        _run_with(session, run, {"StockResearcherRunner": _HangingRunner(_completed())})
    )
    await asyncio.wait_for(started.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert (await session.execute(select(RunQualitySummary))).scalars().all() == []


# --- degradation events reach error_records (86bc997wr, step 6) ------------------------------


async def _prepare_that_reports(*_args, **_kwargs):
    """A prepare() during which the data layer quietly skipped a failed provider
    (twice) and an LLM sub-call got no answer, then returned a normal bundle."""
    report_degradation("yfinance", "get_price_history", LINK_FAILED, "boom", exc=RuntimeError("boom"))
    report_degradation("yfinance", "get_price_history", LINK_FAILED, "boom again")
    report_degradation("ollama", "sentiment_score", LLM_REQUEST_FAILED, "timed out")
    return _fake_bundle()


@pytest.mark.asyncio
async def test_degradation_reported_during_prepare_becomes_error_rows_with_counts():
    session = await _make_session()
    run = await _make_run(session)

    await _run_with(session, run, prepare=_prepare_that_reports)

    assert run.status == RunStatus.COMPLETED  # the run carried on, as designed
    rows = await _error_rows(session)
    assert _kinds(rows) == [
        ("data_pipeline", "link_failed", "", "data:yfinance:get_price_history:link_failed"),
        ("data_pipeline", "llm_request_failed", "", "data:ollama:sentiment_score:llm_request_failed"),
    ]
    by_type = {r.error_type: r for r in rows}
    link = by_type["link_failed"]
    assert link.occurrence_count == 2  # one row, counted, not two rows
    assert link.severity == "low" and link.stock_ticker == "AAPL"
    assert link.message == "boom"  # the first occurrence's message
    assert link.context_json["stage"] == "data"
    assert link.context_json["provider"] == "yfinance"
    assert link.context_json["op"] == "get_price_history"
    assert link.context_json["exc_type"] == "RuntimeError"
    assert link.context_json["account_type"] == "trading"
    assert by_type["llm_request_failed"].occurrence_count == 1


@pytest.mark.asyncio
async def test_degradation_reported_before_prepare_raises_is_still_recorded():
    async def prepare_that_reports_then_fails(*_a, **_k):
        report_degradation("fmp", "get_quote", LINK_FAILED, "fmp down")
        raise ValueError("no data")

    session = await _make_session()
    run = await _make_run(session)

    with pytest.raises(ValueError):
        await _run_with(session, run, prepare=prepare_that_reports_then_fails)

    rows = await _error_rows(session)
    assert sorted(r.error_type for r in rows) == ["link_failed", "prepare_exception"]


@pytest.mark.asyncio
async def test_degradation_reported_after_prepare_returned_is_recorded_too():
    """The benchmark quote for the prediction is fetched long after prepare()."""

    async def quote_that_falls_back(*_a, **_k):
        report_degradation("yfinance", "get_quote", LINK_FAILED, "benchmark quote failed once")
        return {"current_price": 5800.0}

    session = await _make_session()
    run = await _make_run(session)

    await _run_with(session, run, quote=quote_that_falls_back)

    assert run.status == RunStatus.COMPLETED
    assert _kinds(await _error_rows(session)) == [
        ("data_pipeline", "link_failed", "", "data:yfinance:get_quote:link_failed")
    ]


@pytest.mark.asyncio
async def test_the_collector_is_unset_after_a_run_whether_it_succeeds_or_fails():
    session = await _make_session()
    run = await _make_run(session)
    await _run_with(session, run)
    assert current_collector() is None

    session2 = await _make_session()
    run2 = await _make_run(session2)
    with pytest.raises(ValueError):
        await _run_with(session2, run2, prepare=AsyncMock(side_effect=ValueError("no data")))
    assert current_collector() is None


@pytest.mark.asyncio
async def test_one_runs_degradation_never_leaks_into_the_next_runs_rows():
    session = await _make_session()
    run = await _make_run(session)
    await _run_with(session, run, prepare=_prepare_that_reports)

    session2 = await _make_session()
    run2 = await _make_run(session2)
    await _run_with(session2, run2)  # a clean run

    assert await _error_rows(session2) == []


# ---------- prompt fingerprints (86bc997wr, ledger BB-045) ----------


async def _stored_llm_config(session, run) -> dict:
    """Read llm_config back from the database, not from the in-memory run object, so the
    test proves it was really committed."""
    return (
        await session.execute(select(AnalysisRun.llm_config).where(AnalysisRun.run_id == run.run_id))
    ).scalar_one()


@pytest.mark.asyncio
async def test_a_run_records_which_prompt_text_it_used_even_when_it_fails_early():
    session = await _make_session()
    run = await _make_run(session, llm_config={"model": "gpt-oss:20b"})

    with patch("services.orchestrator.DataPipeline") as MockPipeline:
        MockPipeline.return_value.prepare = AsyncMock(side_effect=ValueError("no data"))
        with pytest.raises(ValueError):
            await AnalysisOrchestrator().run(run, session)

    config = await _stored_llm_config(session, run)
    assert config["model"] == "gpt-oss:20b"  # the existing entry is kept
    assert re.fullmatch(r"[0-9a-f]{12}", config["prompts_hash"])
    assert "technical_analyst/v1.txt" in config["prompt_hashes"]
    assert "tax_strategist/canadian_tax_rules_reference.md" in config["prompt_hashes"]
    assert re.fullmatch(r"[0-9a-f]{12}", config["code_hash"])


@pytest.mark.asyncio
async def test_changing_a_prompt_between_two_runs_changes_the_recorded_hash(monkeypatch):
    first = await _make_session()
    run_one = await _make_run(first)
    second = await _make_session()
    run_two = await _make_run(second)

    with patch("services.orchestrator.DataPipeline") as MockPipeline:
        MockPipeline.return_value.prepare = AsyncMock(side_effect=ValueError("no data"))
        with pytest.raises(ValueError):
            await AnalysisOrchestrator().run(run_one, first)
        monkeypatch.setattr(
            "services.orchestrator.prompt_fingerprints",
            lambda: {"files": {"technical_analyst/v1.txt": "edited000000"}, "combined": "edited000000"},
        )
        with pytest.raises(ValueError):
            await AnalysisOrchestrator().run(run_two, second)

    one = await _stored_llm_config(first, run_one)
    two = await _stored_llm_config(second, run_two)
    assert one["prompts_hash"] != two["prompts_hash"]
    assert two["prompt_hashes"] == {"technical_analyst/v1.txt": "edited000000"}


@pytest.mark.asyncio
async def test_a_problem_hashing_the_prompts_never_stops_the_run_or_the_code_hash():
    session = await _make_session()
    run = await _make_run(session, llm_config={"model": "gpt-oss:20b"})

    with (
        patch("services.orchestrator.prompt_fingerprints", side_effect=OSError("disk gone")),
        patch("services.orchestrator.DataPipeline") as MockPipeline,
    ):
        MockPipeline.return_value.prepare = AsyncMock(side_effect=ValueError("no data"))
        with pytest.raises(ValueError, match="no data"):  # the normal failure, not the OSError
            await AnalysisOrchestrator().run(run, session)

    config = await _stored_llm_config(session, run)
    assert "prompts_hash" not in config and "prompt_hashes" not in config
    assert config["model"] == "gpt-oss:20b" and re.fullmatch(r"[0-9a-f]{12}", config["code_hash"])


@pytest.mark.asyncio
async def test_a_problem_hashing_the_code_never_stops_the_run_or_the_prompt_hash():
    session = await _make_session()
    run = await _make_run(session, llm_config={"model": "gpt-oss:20b"})

    with (
        patch("services.orchestrator.code_fingerprint", side_effect=OSError("disk gone")),
        patch("services.orchestrator.DataPipeline") as MockPipeline,
    ):
        MockPipeline.return_value.prepare = AsyncMock(side_effect=ValueError("no data"))
        with pytest.raises(ValueError, match="no data"):
            await AnalysisOrchestrator().run(run, session)

    config = await _stored_llm_config(session, run)
    assert "code_hash" not in config and re.fullmatch(r"[0-9a-f]{12}", config["prompts_hash"])


@pytest.mark.asyncio
async def test_if_neither_hash_can_be_taken_the_run_is_left_exactly_as_it_was():
    session = await _make_session()
    run = await _make_run(session, llm_config={"model": "gpt-oss:20b"})

    with (
        patch("services.orchestrator.prompt_fingerprints", side_effect=OSError("gone")),
        patch("services.orchestrator.code_fingerprint", side_effect=OSError("gone")),
        patch("services.orchestrator.DataPipeline") as MockPipeline,
    ):
        MockPipeline.return_value.prepare = AsyncMock(side_effect=ValueError("no data"))
        with pytest.raises(ValueError, match="no data"):
            await AnalysisOrchestrator().run(run, session)

    assert await _stored_llm_config(session, run) == {"model": "gpt-oss:20b"}


@pytest.mark.asyncio
async def test_the_cio_stage_b_gets_no_risk_input_and_no_position_size_is_stored():
    """Position sizing and the Risk Advisor's stage B were removed 2026-10-01: Risk feeds the CIO's
    stage A only, and the stored recommendation carries no position size."""
    session = await _make_session()
    run = await _make_run(session)
    bundle = _fake_bundle()
    cio_stage_a = _completed(
        stock_outlook="neutral", expected_return_tier="market_perform",
        thesis_summary="t", key_decision_factors=[],
    )
    cio_stage_b = {"synthesis_narrative": "n", "expected_return_tier": "market_perform", "tax_summary": {}}
    cio = _StubRunner(cio_stage_a, stage_b_result=cio_stage_b)
    risk = _StubRunner(_completed(risk_profile={"beta": 1.1}))

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
        patch("services.orchestrator.RiskAdvisorRunner", risk),
        patch("services.orchestrator.CIORunner", cio),
        patch("services.orchestrator.ShadowCIORunner", _StubRunner(_completed(stock_outlook="neutral"))),
        patch("services.orchestrator.Router") as MockRouter,
    ):
        MockPipeline.return_value.prepare = AsyncMock(return_value=bundle)
        router_instance = AsyncMock()
        router_instance.get_quote = AsyncMock(return_value={"current_price": 5800.0})
        MockRouter.return_value.__aenter__ = AsyncMock(return_value=router_instance)
        MockRouter.return_value.__aexit__ = AsyncMock(return_value=False)

        await AnalysisOrchestrator().run(run, session)

    assert run.status == RunStatus.COMPLETED
    assert not hasattr(risk, "stage_b_calls") or risk.stage_b_calls == []  # Risk makes no stage B call
    assert len(cio.stage_b_calls) == 1
    # run_stage_b(bundle, stage_a_result, tax_result, account_type): no Risk argument
    assert len(cio.stage_b_calls[0]) == 4
    rec = (await session.execute(select(Recommendation).where(Recommendation.run_id == run.run_id))).scalar_one()
    assert rec.position_size_suggestion is None


@pytest.mark.asyncio
async def test_the_cio_account_result_carries_the_tax_agents_comparable_cost():
    """The Tax Strategist runs once per account; the CIO's account-specific result is what the
    Portfolio Optimizer reads, so the cost figures are copied into its tax_summary by code."""
    session = await _make_session()
    run = await _make_run(session)
    bundle = _fake_bundle()
    cio_stage_a = _completed(
        stock_outlook="neutral", expected_return_tier="market_perform",
        thesis_summary="t", key_decision_factors=[],
    )
    cio_stage_b = {"synthesis_narrative": "n", "expected_return_tier": "market_perform", "tax_summary": {}}
    tax = _StubRunner(_completed(tax_profile={
        "dividend_yield_pct": 2.4, "annual_tax_drag_pct": 0.72, "effective_after_tax_yield_pct": 1.71,
    }))

    with (
        patch("services.orchestrator.DataPipeline") as MockPipeline,
        patch("services.orchestrator.StockResearcherRunner", _StubRunner(_completed())),
        patch("services.orchestrator.FundamentalAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.TechnicalAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.SentimentAnalystRunner", _StubRunner(_completed())),
        patch("services.orchestrator.MacroEconomistRunner", _StubRunner(_completed())),
        patch("services.orchestrator.BullAdvocateRunner", _StubRunner(_completed(recommendation="bullish"))),
        patch("services.orchestrator.BearAdvocateRunner", _StubRunner(_completed(recommendation="bearish"))),
        patch("services.orchestrator.TaxStrategistRunner", tax),
        patch("services.orchestrator.RiskAdvisorRunner", _StubRunner(_completed(risk_profile={}))),
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

    outputs = (await session.execute(select(AgentOutput).where(AgentOutput.run_id == run.run_id))).scalars().all()
    cio_row = next(r for r in outputs if r.agent_name == "cio_stage_b")
    summary = cio_row.structured_output["tax_summary"]
    assert summary["dividend_yield_pct"] == 2.4
    assert summary["annual_tax_drag_pct"] == 0.72
    assert summary["effective_after_tax_yield_pct"] == 1.71
