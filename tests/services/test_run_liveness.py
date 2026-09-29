"""Tests for services/run_liveness.py (86bc997wr): deciding a run is stale and
ending it without ever touching one that is still working. Hermetic: in-memory
SQLite, an injected clock, no network, no Ollama.
"""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api.database import Base
from api.tables.agent_outputs import AgentOutput
from api.tables.analysis_runs import AnalysisRun, RunStatus
from api.tables.error_records import ErrorRecord
from api.tables.llm_calls import LLMCall
from api.tables.stock import Stock
from services.run_liveness import (
    DEFAULT_STALE_MINUTES,
    assess,
    end_run_if_stale,
    end_run_now,
    list_non_terminal_runs,
    load_snapshot,
    stale_minutes,
)

# Naive UTC, like every timestamp SQLite hands back.
NOW = datetime(2026, 9, 29, 12, 0, 0)


def _ago(minutes: float) -> datetime:
    return NOW - timedelta(minutes=minutes)


# --- the pure rule -----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _default_threshold(monkeypatch):
    monkeypatch.delenv("SP_STALE_RUN_MINUTES", raising=False)


def test_the_default_limit_is_ninety_minutes():
    # Pinned so a change is deliberate. The reasoning (twice the ~45 minutes a run
    # could be silent if every call hung to its timeout) lives in run_liveness's
    # docstring; this test does not, and cannot, prove it -- real timings would.
    assert DEFAULT_STALE_MINUTES == 90
    assert stale_minutes() == 90


@pytest.mark.parametrize(
    "raw, expected",
    [("10", 10), ("120", 120), ("0", 90), ("-5", 90), ("abc", 90), ("", 90), ("2.5", 90)],
)
def test_the_limit_can_be_set_from_the_environment_and_falls_back_on_nonsense(
    monkeypatch, raw, expected
):
    monkeypatch.setenv("SP_STALE_RUN_MINUTES", raw)
    assert stale_minutes() == expected


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_a_finished_run_is_never_stale(status):
    assert assess(status, _ago(10_000), _ago(10_000), NOW).stale is False


def test_a_run_with_recent_activity_is_not_stale():
    verdict = assess("pass2_running", _ago(200), _ago(5), NOW)
    assert verdict.stale is False and verdict.reason == "active"


def test_a_run_silent_for_longer_than_the_limit_is_stale():
    verdict = assess("pass2_running", _ago(300), _ago(120), NOW)
    assert verdict.stale is True
    assert "no activity for 120 min" in verdict.reason and "limit 90" in verdict.reason


def test_the_limit_is_strict_at_the_boundary():
    assert assess("pass1_running", _ago(100), _ago(89), NOW).stale is False
    assert assess("pass1_running", _ago(100), _ago(90), NOW).stale is False  # exactly the limit
    assert assess("pass1_running", _ago(100), _ago(91), NOW).stale is True


def test_a_custom_limit_overrides_the_default():
    assert assess("pass1_running", _ago(10), _ago(5), NOW, stale_after=2).stale is True
    assert assess("pass1_running", _ago(10), _ago(5), NOW, stale_after=60).stale is False


def test_a_queued_run_gets_the_same_generous_limit_as_any_other():
    """Regression for a real finding: the orchestrator never sets pass1_running,
    so a LIVE run stays `queued` through prepare() and all of Pass 1 (a real
    killed run was still `queued` 25 seconds in). An early 'queued for 15
    minutes means it never started' rule would end a live, slow run."""
    slow_but_alive = assess("queued", _ago(30), _ago(30), NOW)
    assert slow_but_alive.stale is False
    long_dead = assess("queued", _ago(200), _ago(200), NOW)
    assert long_dead.stale is True and "limit 90" in long_dead.reason


def test_mixing_naive_and_timezone_aware_timestamps_does_not_raise():
    """The real 'offset-naive and offset-aware' bug this project already hit."""
    aware_now = NOW.replace(tzinfo=UTC)
    aware_activity = _ago(120).replace(tzinfo=UTC)
    assert assess("pass2_running", _ago(150), aware_activity, aware_now).stale is True
    assert assess("pass2_running", _ago(100), _ago(1), aware_now).stale is False


def test_a_run_with_no_timestamps_is_not_declared_stale():
    assert assess("pass1_running", None, None, NOW).stale is False


# --- against a database --------------------------------------------------------------


async def _engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine


async def _seed(
    engine,
    *,
    status,
    triggered_min,
    pass1_min=None,
    llm_min=None,
    output_min=None,
    ticker="AAPL",
):
    """A run with the given age and (optional) activity, in minutes before NOW."""
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        stock = (
            await session.execute(select(Stock).where(Stock.canonical_ticker == ticker))
        ).scalar_one_or_none()
        if stock is None:
            stock = Stock(
                stock_id=uuid4(),
                canonical_ticker=ticker,
                company_name=ticker,
                primary_exchange="NASDAQ",
                currency="USD",
            )
            session.add(stock)
            await session.commit()
        run = AnalysisRun(
            run_id=uuid4(),
            user_id=uuid4(),
            stock_id=stock.stock_id,
            account_type="trading",
            timeline="medium_term",
            triggered_by="manual",
            status=status,
            llm_config={},
            triggered_at=_ago(triggered_min),
            pass1_completed_at=_ago(pass1_min) if pass1_min is not None else None,
        )
        session.add(run)
        await session.commit()
        if llm_min is not None:
            call = LLMCall.from_call_log_entry(
                {"seq": 0, "call_site": "agent:fund", "model": "gpt-oss:20b", "attempt": 1},
                run_id=run.run_id,
                agent_pass="pass1",
            )
            call.created_at = _ago(llm_min)
            session.add(call)
        if output_min is not None:
            session.add(
                AgentOutput(
                    run_id=run.run_id,
                    agent_name="FUND",
                    agent_pass="pass1",
                    status="completed",
                    model_used="gpt-oss:20b",
                    prompt_version="v1",
                    created_at=_ago(output_min),
                )
            )
        await session.commit()
        return run.run_id


async def _get(engine, run_id) -> AnalysisRun:
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        return (
            await session.execute(select(AnalysisRun).where(AnalysisRun.run_id == run_id))
        ).scalar_one()


async def _error_rows(engine) -> list[ErrorRecord]:
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        return list((await session.execute(select(ErrorRecord))).scalars())


async def test_last_activity_is_the_newest_of_every_recorded_timestamp():
    engine = await _engine()
    run_id = await _seed(
        engine, status="pass2_running", triggered_min=200, pass1_min=100, llm_min=30, output_min=10
    )
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        snapshot = await load_snapshot(session, run_id)
    assert snapshot.last_activity_at == _ago(10)
    assert snapshot.triggered_at == _ago(200) and snapshot.ticker == "AAPL"


async def test_a_run_with_recent_activity_is_left_alone_however_old_it_is():
    engine = await _engine()
    run_id = await _seed(engine, status="pass2_running", triggered_min=500, llm_min=5)

    assert await end_run_if_stale(engine, run_id, now=NOW) is False

    assert (await _get(engine, run_id)).status == "pass2_running"
    assert await _error_rows(engine) == []


async def test_a_stale_run_is_ended_and_recorded():
    engine = await _engine()
    run_id = await _seed(engine, status="pass2_running", triggered_min=300, pass1_min=290)

    assert await end_run_if_stale(engine, run_id, now=NOW) is True

    run = await _get(engine, run_id)
    assert run.status == RunStatus.FAILED
    assert run.error_log[0]["stage"] == "orphaned"
    assert "no activity for 290 min" in run.error_log[0]["error"]
    rows = await _error_rows(engine)
    assert len(rows) == 1
    row = rows[0]
    assert (row.component, row.error_type, row.severity) == (
        "orchestrator",
        "run_orphaned",
        "medium",
    )
    assert row.dedup_subtype == "stale:pass2_running" and row.stock_ticker == "AAPL"
    assert row.run_id == run_id and row.context_json["last_status"] == "pass2_running"
    assert row.context_json["idle_minutes"] == 290.0


async def test_ending_a_stale_run_is_idempotent():
    engine = await _engine()
    run_id = await _seed(engine, status="pass1_running", triggered_min=300)

    assert await end_run_if_stale(engine, run_id, now=NOW) is True
    assert await end_run_if_stale(engine, run_id, now=NOW) is False

    assert len(await _error_rows(engine)) == 1


@pytest.mark.parametrize("status", ["completed", "failed"])
async def test_a_finished_run_is_never_touched(status):
    engine = await _engine()
    run_id = await _seed(engine, status=status, triggered_min=10_000)

    assert await end_run_if_stale(engine, run_id, now=NOW) is False

    assert (await _get(engine, run_id)).status == status
    assert await _error_rows(engine) == []


async def test_an_unknown_run_id_is_not_an_error():
    engine = await _engine()
    assert await end_run_if_stale(engine, uuid4(), now=NOW) is False


async def test_a_long_dead_queued_run_is_ended_but_one_still_in_its_first_phase_is_not():
    engine = await _engine()
    dead = await _seed(engine, status="queued", triggered_min=200, ticker="SHOP.TO")
    first_phase = await _seed(engine, status="queued", triggered_min=30, ticker="RY.TO")

    assert await end_run_if_stale(engine, dead, now=NOW) is True
    assert await end_run_if_stale(engine, first_phase, now=NOW) is False

    assert (await _get(engine, dead)).status == RunStatus.FAILED
    assert (await _get(engine, first_phase)).status == "queued"


async def test_a_custom_limit_is_honoured_by_end_run_if_stale():
    engine = await _engine()
    run_id = await _seed(engine, status="pass1_running", triggered_min=20, llm_min=10)

    assert await end_run_if_stale(engine, run_id, now=NOW) is False  # 10 min < 90
    assert await end_run_if_stale(engine, run_id, now=NOW, stale_after=5) is True


async def test_a_failure_while_ending_is_swallowed_not_raised():
    """A REAL failure -- the tables do not exist -- because a problem here must
    never break the request that asked."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    assert await end_run_if_stale(engine, uuid4(), now=NOW) is False


async def test_end_run_now_ends_a_running_run_but_never_a_finished_one():
    engine = await _engine()
    running = await _seed(engine, status="pass2_running", triggered_min=2, ticker="AAPL")
    done = await _seed(engine, status="completed", triggered_min=2, ticker="MSFT")

    assert await end_run_now(engine, running, stage="cancelled", message="stopped") is True
    assert await end_run_now(engine, done, stage="cancelled", message="stopped") is False

    ended = await _get(engine, running)
    assert ended.status == RunStatus.FAILED
    assert ended.error_log == [{"stage": "cancelled", "error": "stopped"}]
    assert (await _get(engine, done)).status == "completed"


async def test_the_listing_shows_every_unfinished_run_with_its_verdict():
    engine = await _engine()
    stale = await _seed(
        engine, status="pass2_running", triggered_min=400, pass1_min=390, ticker="AAPL"
    )
    active = await _seed(engine, status="pass1_running", triggered_min=3, llm_min=1, ticker="MSFT")
    await _seed(engine, status="completed", triggered_min=900, ticker="RY.TO")

    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        listing = await list_non_terminal_runs(session, NOW)

    by_id = {entry["run_id"]: entry for entry in listing}
    assert set(by_id) == {stale, active}
    assert by_id[stale]["stale"] is True and by_id[stale]["ticker"] == "AAPL"
    assert by_id[stale]["idle_minutes"] == 390.0 and by_id[stale]["age_minutes"] == 400.0
    assert by_id[active]["stale"] is False and by_id[active]["reason"] == "active"
