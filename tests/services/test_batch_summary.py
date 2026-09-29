"""Tests for services/batch_summary.py (86bc997wr): the report a quality-test
batch ends with. The point under test is that a batch which contains a failure,
or a run that completed on degraded data, can never read as a clean pass.
"""

from datetime import datetime, timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api.database import Base
from api.tables.analysis_runs import AnalysisRun
from api.tables.error_records import ErrorRecord
from api.tables.recommendations import Recommendation
from api.tables.stock import Stock
from services.batch_summary import (
    FAILED,
    NOT_STARTED,
    OK,
    RETRIES,
    TIMED_OUT,
    WITH_PROBLEMS,
    describe_run,
    format_batch_summary,
    parse_run_id,
)

RUN_ID = "7d62ef8d-3007-43ea-9cd7-300d65ebe5cc"
NOW = datetime(2026, 9, 29, 12, 0, 0)


# --- parse_run_id --------------------------------------------------------------------


def test_the_run_id_is_read_from_the_starting_line():
    line = f"Starting analysis: KO | account=tfsa | timeline=medium_term | run_id={RUN_ID}"
    assert parse_run_id(line) == UUID(RUN_ID)


@pytest.mark.parametrize(
    "line",
    [
        "",
        "An analysis is already running for KO: run_id=" + RUN_ID + " status=queued",
        "Starting analysis: KO | run_id=not-a-uuid",
        "No user profile found for user_id=" + RUN_ID + " -- create one first",
    ],
)
def test_a_line_that_is_not_the_starting_line_gives_no_run_id(line):
    """A blocked start prints a run_id too, but it belongs to the OTHER run: taking
    it would report someone else's run as this one's."""
    assert parse_run_id(line) is None


# --- format_batch_summary ------------------------------------------------------------


def _row(outcome, headline="h", **extra):
    return {"label": "KO | tfsa / medium_term", "outcome": outcome, "headline": headline, **extra}


def test_a_batch_of_clean_runs_says_so():
    text = format_batch_summary(
        [
            _row(OK, run_id=UUID(RUN_ID), seconds=300, outlook="bullish", confidence=72),
            _row(OK),
        ]
    )
    assert "TOTALS: 2 ok" in text
    assert "VERDICT: every run completed with no recorded problems" in text
    assert "bullish 72" in text and "5.0m" in text and RUN_ID[:8] in text


def test_one_run_completed_on_degraded_data_makes_the_whole_batch_attention():
    text = format_batch_summary([_row(OK), _row(WITH_PROBLEMS, "COMPLETED, but 1 error record(s)")])
    assert "TOTALS: 1 ok, 1 completed with problems" in text
    assert "VERDICT: ATTENTION" in text
    assert "COMPLETED, but 1 error record(s)" in text  # the reason is on the page


def test_every_kind_of_non_success_is_counted_and_explained():
    text = format_batch_summary(
        [
            _row(FAILED, "FAILED at data_pipeline: Company not found: 'PRMW.TO'"),
            _row(TIMED_OUT, "killed after 60 minutes"),
            _row(NOT_STARTED, "analyze.py never started a run"),
        ]
    )
    assert "1 failed, 1 timed out, 1 did not start" in text
    assert "Company not found: 'PRMW.TO'" in text
    assert "VERDICT: ATTENTION" in text


def test_retries_alone_are_called_out_and_are_not_a_failure_nor_a_clean_pass():
    text = format_batch_summary(
        [_row(OK), _row(RETRIES, "COMPLETED, 5 agent(s) needed retries and recovered: a, b")]
    )
    assert "TOTALS: 1 ok, 1 completed with retries" in text
    assert "VERDICT: no errors, but 1 run(s) needed retries" in text
    assert "investigate" in text
    assert "ATTENTION" not in text
    assert "recovered: a, b" in text  # which agents, on the page


def test_retries_next_to_a_failure_is_still_attention():
    text = format_batch_summary([_row(RETRIES), _row(FAILED, "FAILED at pass1: x")])
    assert "VERDICT: ATTENTION" in text


def test_an_empty_batch_is_not_reported_as_a_clean_pass():
    text = format_batch_summary([])
    assert "no runs" in text
    assert "VERDICT: ATTENTION" in text


def test_a_clean_run_row_has_no_headline_line():
    text = format_batch_summary([_row(OK, "COMPLETED with no recorded problems")])
    assert "COMPLETED with no recorded problems" not in text


# --- describe_run (real tables, in-memory) -------------------------------------------


async def _engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine


async def _seed_run(engine, status, *, error_log=None, recommendation=None, minutes=5):
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        stock = Stock(
            stock_id=uuid4(),
            canonical_ticker="KO",
            company_name="KO",
            primary_exchange="NYSE",
            currency="USD",
        )
        session.add(stock)
        await session.commit()
        run = AnalysisRun(
            run_id=uuid4(),
            user_id=uuid4(),
            stock_id=stock.stock_id,
            account_type="tfsa",
            timeline="medium_term",
            triggered_by="cli",
            status=status,
            llm_config={},
            triggered_at=NOW - timedelta(minutes=minutes),
            completed_at=NOW if status in ("completed", "failed") else None,
            error_log=error_log,
        )
        session.add(run)
        await session.commit()
        if recommendation:
            session.add(
                Recommendation(
                    run_id=run.run_id,
                    stock_id=stock.stock_id,
                    stock_outlook_direction=recommendation[0],
                    overall_confidence=recommendation[1],
                    account_recommendation={},
                    key_drivers=[],
                    key_risks=[],
                    bull_case_strength=50,
                    bear_case_strength=50,
                    synthesis_narrative="n",
                )
            )
            await session.commit()
        return run.run_id


async def test_describe_a_clean_completed_run_includes_the_outlook_and_duration():
    engine = await _engine()
    run_id = await _seed_run(engine, "completed", recommendation=("bullish", 72))
    async with async_sessionmaker(engine)() as session:
        row = await describe_run(session, run_id)
    assert row["outcome"] == OK
    assert (row["outlook"], row["confidence"], row["seconds"]) == ("bullish", 72, 300)


async def test_describe_a_failed_run_gives_the_reason_and_no_outlook():
    engine = await _engine()
    run_id = await _seed_run(
        engine, "failed", error_log=[{"stage": "pass1", "error": "Ollama unavailable"}]
    )
    async with async_sessionmaker(engine)() as session:
        row = await describe_run(session, run_id)
    assert row["outcome"] == "FAILED"
    assert "Ollama unavailable" in row["headline"]
    assert row["outlook"] is None


async def test_describe_a_run_that_completed_with_an_error_row_is_not_ok():
    engine = await _engine()
    run_id = await _seed_run(engine, "completed", recommendation=("neutral", 50))
    async with async_sessionmaker(engine)() as session:
        session.add(
            ErrorRecord(
                timestamp=NOW,
                first_seen=NOW,
                last_seen=NOW,
                severity="high",
                component="data_pipeline",
                error_type="sentiment_unscored",
                dedup_subtype="data:ollama:sentiment:sentiment_unscored",
                run_id=run_id,
                message="248 of 248 articles could not be scored",
                occurrence_count=1,
            )
        )
        await session.commit()
        row = await describe_run(session, run_id)
    assert row["outcome"] == WITH_PROBLEMS
    assert row["errors"] == 1


async def test_describe_an_unknown_run_is_reported_not_raised():
    engine = await _engine()
    async with async_sessionmaker(engine)() as session:
        row = await describe_run(session, uuid4())
    assert row["outcome"] == NOT_STARTED
