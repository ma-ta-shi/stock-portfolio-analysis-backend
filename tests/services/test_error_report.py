"""Tests for services/error_report.py (86bc997wr): the questions a developer asks
when something goes wrong, answered from a database seeded with the situations
the reports must tell apart. Hermetic: in-memory SQLite, an injected clock.
"""

from datetime import datetime, timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api.database import Base
from api.tables.agent_outputs import AgentOutput
from api.tables.analysis_runs import AnalysisRun
from api.tables.error_records import ErrorRecord
from api.tables.llm_calls import LLMCall
from api.tables.run_quality_summary import RunQualitySummary
from api.tables.stock import Stock
from services.error_report import (
    diagnose_run,
    format_diagnosis,
    format_prompt_comparison,
    format_recent_errors,
    format_run_summary,
    parse_since,
    prompt_comparison,
    recent_errors,
    resolve_run_id,
)
from services.run_liveness import end_stale_runs

NOW = datetime(2026, 9, 29, 12, 0, 0)  # naive UTC, like everything SQLite returns
WINDOW_START = NOW - timedelta(days=7)


def _ago(minutes: float = 0, *, days: float = 0) -> datetime:
    return NOW - timedelta(minutes=minutes, days=days)


async def _engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine


def _maker(engine):
    return async_sessionmaker(engine, expire_on_commit=False)


async def _run(
    engine,
    ticker,
    status,
    *,
    triggered_min,
    error_log=None,
    pass1_min=None,
    completed_min=None,
    llm_config=None,
):
    async with _maker(engine)() as session:
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
            llm_config=llm_config or {},
            triggered_at=_ago(triggered_min),
            pass1_completed_at=_ago(pass1_min) if pass1_min is not None else None,
            completed_at=_ago(completed_min) if completed_min is not None else None,
            error_log=error_log,
        )
        session.add(run)
        await session.commit()
        return run.run_id


async def _error(
    engine,
    run_id,
    *,
    minutes_ago,
    severity="high",
    component="agent",
    error_type="all_retries_exhausted",
    agent="FUND",
    fingerprint="FUND:narrative: too long (# chars, max #)",
    ticker="AAPL",
    message="narrative: too long (1913 chars, max 1080)",
    stack_trace=None,
    context=None,
    occurrence_count=1,
):
    when = _ago(minutes_ago)
    async with _maker(engine)() as session:
        session.add(
            ErrorRecord(
                timestamp=when,
                first_seen=when,
                last_seen=when,
                severity=severity,
                component=component,
                error_type=error_type,
                dedup_subtype=fingerprint,
                run_id=run_id,
                agent_name=agent,
                stock_ticker=ticker,
                message=message,
                stack_trace=stack_trace,
                context_json=context,
                occurrence_count=occurrence_count,
            )
        )
        await session.commit()


async def _agent(engine, run_id, name, status="completed", error_detail=None):
    async with _maker(engine)() as session:
        session.add(
            AgentOutput(
                run_id=run_id,
                agent_name=name,
                agent_pass="pass1",
                status=status,
                model_used="gpt-oss:20b",
                prompt_version="v1",
                error_detail=error_detail,
            )
        )
        await session.commit()


async def _call(
    engine,
    run_id,
    *,
    seq,
    attempt,
    site="agent:fund",
    validator_passed=None,
    validator_errors=None,
    parsed_ok=True,
    finish_reason="stop",
    parse_error=None,
    minutes_ago=5,
    auto_trimmed=False,
):
    entry = {
        "seq": seq,
        "attempt": attempt,
        "call_site": site,
        "model": "gpt-oss:20b",
        "finish_reason": finish_reason,
        "parsed_ok": parsed_ok,
        "parse_error": parse_error,
        "validator_passed": validator_passed,
        "validator_errors": validator_errors,
        "prompt_path": f"2026-09-29\\AAPL_abcd1234\\{seq}_{site.replace(':', '-')}.{attempt}.prompt.txt",
        "response_path": f"2026-09-29\\AAPL_abcd1234\\{seq}_{site.replace(':', '-')}.{attempt}.response.json",
        "total_duration_s": 8.0,
        "prompt_eval_count": 2000,
        "auto_trimmed": auto_trimmed,
    }
    async with _maker(engine)() as session:
        call = LLMCall.from_call_log_entry(entry, run_id=run_id, agent_pass="pass1")
        call.created_at = _ago(minutes_ago)
        session.add(call)
        await session.commit()


# --- parse_since ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("30m", NOW - timedelta(minutes=30)),
        ("24h", NOW - timedelta(hours=24)),
        ("7d", NOW - timedelta(days=7)),
        ("2w", NOW - timedelta(weeks=2)),
        ("2026-09-28", datetime(2026, 9, 28)),
        ("2026-09-28T10:30", datetime(2026, 9, 28, 10, 30)),
    ],
)
def test_since_accepts_relative_and_absolute_times(text, expected):
    assert parse_since(text, NOW) == expected


def test_since_rejects_nonsense_with_a_message_that_says_what_is_accepted():
    with pytest.raises(ValueError, match="24h"):
        parse_since("yesterday-ish", NOW)


# --- diagnose one run ------------------------------------------------------------------


async def test_diagnosing_an_unknown_run_returns_nothing():
    engine = await _engine()
    async with _maker(engine)() as session:
        assert await diagnose_run(session, uuid4(), now=NOW) is None


async def test_a_failed_run_is_explained_first_with_its_traceback_and_failed_agents():
    engine = await _engine()
    run_id = await _run(
        engine,
        "MSFT",
        "failed",
        triggered_min=300,
        error_log=[{"stage": "pass1", "error": "Ollama unavailable"}],
    )
    await _error(
        engine,
        run_id,
        minutes_ago=298,
        severity="critical",
        component="llm",
        error_type="ollama_connection_failed",
        agent=None,
        fingerprint="pass1",
        ticker="MSFT",
        message="Cannot reach Ollama at http://localhost:19999. Is it running?",
        stack_trace="Traceback (most recent call last):\n  File a.py, line 1, in f\nOllamaUnavailable: down",
        context={"stage": "pass1", "account_type": "trading"},
    )
    await _agent(engine, run_id, "RSRCH", "failed", "Cannot reach Ollama")

    async with _maker(engine)() as session:
        d = await diagnose_run(session, run_id, now=NOW)

    assert d["verdict"]["outcome"] == "FAILED"
    assert d["verdict"]["headline"] == "FAILED at pass1: Ollama unavailable"
    assert d["errors"][0]["fingerprint"] == "pass1"
    assert d["errors"][0]["traceback_tail"][-1] == "OllamaUnavailable: down"
    assert [a["agent_name"] for a in d["agents"]] == ["RSRCH"]
    text = format_diagnosis(d)
    assert text.index("RESULT: FAILED at pass1") < text.index("--- errors recorded")
    assert "[critical] llm/ollama_connection_failed" in text
    assert "OllamaUnavailable: down" in text
    assert "RSRCH (pass1) failed" in text


async def test_a_completed_run_still_shows_every_attempt_of_an_agent_that_was_rejected(
    tmp_path, monkeypatch
):
    """The dominant real failure: a validator rejects attempt 1 and attempt 2 is
    accepted. Nothing 'failed', so only the attempt history shows it, together
    with where the raw prompt and response were saved."""
    monkeypatch.setenv("SP_RUNS_DIR", str(tmp_path))
    engine = await _engine()
    run_id = await _run(engine, "AAPL", "completed", triggered_min=60, completed_min=55)
    await _agent(engine, run_id, "FUND")
    await _call(
        engine,
        run_id,
        seq=250,
        attempt=1,
        validator_passed=False,
        validator_errors=["caveats is empty -- must explain why"],
    )
    await _call(engine, run_id, seq=255, attempt=2, validator_passed=True)
    await _call(engine, run_id, seq=260, attempt=1, site="agent:tech", validator_passed=True)
    (tmp_path / "2026-09-29" / "AAPL_abcd1234").mkdir(parents=True)
    (tmp_path / "2026-09-29" / "AAPL_abcd1234" / "250_agent-fund.1.prompt.txt").write_text("x")

    async with _maker(engine)() as session:
        d = await diagnose_run(session, run_id, now=NOW)

    assert d["verdict"]["outcome"] == "COMPLETED WITH RETRIES"
    assert d["verdict"]["headline"] == ("COMPLETED, 1 agent(s) needed retries and recovered: fund")
    assert list(d["failing_calls"]) == ["agent:fund"]  # TECH had no trouble: not listed
    attempts = d["failing_calls"]["agent:fund"]
    assert [a["attempt"] for a in attempts] == [1, 2]
    assert attempts[0]["validator_passed"] is False
    assert attempts[0]["validator_errors"] == ["caveats is empty -- must explain why"]
    assert attempts[0]["prompt"]["exists"] is True  # the file is really there
    assert attempts[0]["response"]["exists"] is False  # and this one is not
    text = format_diagnosis(d)
    assert "attempt 1 (seq 250)" in text and "validator=REJECTED" in text
    assert "attempt 2 (seq 255)" in text and "validator=accepted" in text
    assert "- caveats is empty" in text
    assert "[found]" in text and "[MISSING]" in text


async def test_a_clean_completed_run_says_so():
    engine = await _engine()
    run_id = await _run(engine, "AAPL", "completed", triggered_min=60, completed_min=55)
    await _agent(engine, run_id, "FUND")
    await _call(engine, run_id, seq=1, attempt=1, validator_passed=True)

    async with _maker(engine)() as session:
        d = await diagnose_run(session, run_id, now=NOW)

    assert d["verdict"] == {
        "outcome": "OK",
        "headline": "COMPLETED with no recorded problems",
    }
    assert d["failing_calls"] == {}


async def test_precompute_call_failures_are_counted_not_listed_one_by_one():
    engine = await _engine()
    run_id = await _run(engine, "AAPL", "completed", triggered_min=60, completed_min=55)
    for i in range(6):
        await _call(
            engine,
            run_id,
            seq=i,
            attempt=1,
            site="precompute:sentiment",
            parsed_ok=(i % 2 == 0),
            parse_error=None if i % 2 == 0 else "bad json",
        )

    async with _maker(engine)() as session:
        d = await diagnose_run(session, run_id, now=NOW)

    assert d["failing_calls"] == {}
    assert d["other_failing_calls"] == [
        {"call_site": "precompute:sentiment", "failed": 3, "total": 6}
    ]


async def test_a_run_that_stopped_working_is_diagnosed_as_stale():
    engine = await _engine()
    run_id = await _run(engine, "AAPL", "pass2_running", triggered_min=7000, pass1_min=6990)

    async with _maker(engine)() as session:
        d = await diagnose_run(session, run_id, now=NOW)

    assert d["verdict"]["outcome"] == "STALE"
    assert "--fail-stale" in d["verdict"]["headline"]
    assert d["liveness"]["stale"] is True


async def test_a_run_that_is_working_is_diagnosed_as_running():
    engine = await _engine()
    run_id = await _run(engine, "AAPL", "pass1_running", triggered_min=3)

    async with _maker(engine)() as session:
        d = await diagnose_run(session, run_id, now=NOW)

    assert d["verdict"]["outcome"] == "RUNNING"


async def test_quality_counters_and_gates_are_reported_when_the_summary_exists():
    engine = await _engine()
    run_id = await _run(
        engine,
        "AAPL",
        "failed",
        triggered_min=60,
        error_log=[{"stage": "gate1", "error": "no data"}],
    )
    async with _maker(engine)() as session:
        session.add(
            RunQualitySummary(
                run_id=run_id,
                wall_clock_ms=1000,
                total_calls=4,
                retry_calls=1,
                validator_failures=2,
                gate1_passed=False,
                gate1_reason="no agent completed",
            )
        )
        await session.commit()

    async with _maker(engine)() as session:
        text = format_diagnosis(await diagnose_run(session, run_id, now=NOW))

    assert "validator failures 2" in text and "gate1: False (no agent completed)" in text


# --- the cross-run view -----------------------------------------------------------------


async def _seeded_week():
    """A week with: a recurring FUND validator failure, a hard Ollama failure, two
    legacy failed runs that predate the recorder, a stale run, a live run, a
    low-severity event, and a failure that first appeared long before the window."""
    engine = await _engine()
    ok = await _run(engine, "AAPL", "completed", triggered_min=600, completed_min=595)

    fund_a = await _run(
        engine,
        "AAPL",
        "failed",
        triggered_min=400,
        error_log=[{"stage": "cio_stage_a", "error": "x"}],
    )
    fund_b = await _run(
        engine,
        "MSFT",
        "failed",
        triggered_min=200,
        error_log=[{"stage": "cio_stage_a", "error": "x"}],
    )
    await _error(engine, fund_a, minutes_ago=399, ticker="AAPL")
    await _error(
        engine,
        fund_b,
        minutes_ago=199,
        ticker="MSFT",
        message="narrative: too long (1351 chars, max 1080)",
    )

    ollama = await _run(
        engine,
        "GOOGL",
        "failed",
        triggered_min=100,
        error_log=[{"stage": "pass1", "error": "Ollama unavailable"}],
    )
    await _error(
        engine,
        ollama,
        minutes_ago=98,
        severity="critical",
        component="llm",
        error_type="ollama_connection_failed",
        agent=None,
        fingerprint="pass1",
        ticker="GOOGL",
        message="Cannot reach Ollama",
    )

    # Recurs, but first appeared 30 days ago: recurring, NOT new.
    old_fp = dict(
        component="data_pipeline",
        error_type="prepare_exception",
        agent=None,
        fingerprint="ValueError@data/pipeline.py:_merge",
        severity="high",
    )
    ancient = await _run(
        engine,
        "RY.TO",
        "failed",
        triggered_min=60 * 24 * 30,
        error_log=[{"stage": "data_pipeline", "error": "old"}],
    )
    await _error(
        engine, ancient, minutes_ago=60 * 24 * 30, ticker="RY.TO", message="old failure", **old_fp
    )
    recent_dp = await _run(
        engine,
        "RY.TO",
        "failed",
        triggered_min=50,
        error_log=[{"stage": "data_pipeline", "error": "again"}],
    )
    await _error(engine, recent_dp, minutes_ago=49, ticker="RY.TO", message="again", **old_fp)

    low = await _run(engine, "AMZN", "completed", triggered_min=30, completed_min=25)
    await _error(
        engine,
        low,
        minutes_ago=29,
        severity="low",
        component="data_pipeline",
        error_type="provider_fallback",
        agent=None,
        fingerprint="fmp:get_quote:fallback",
        ticker="AMZN",
        message="fmp failed, yfinance used",
    )

    legacy = "User-Agent identity is not set"
    await _run(
        engine,
        "TD.TO",
        "failed",
        triggered_min=60 * 24 * 3,
        error_log=[{"stage": "data_pipeline", "error": legacy}],
    )
    await _run(
        engine,
        "BNS.TO",
        "failed",
        triggered_min=60 * 24 * 2,
        error_log=[{"stage": "data_pipeline", "error": legacy}],
    )

    stale = await _run(
        engine, "SHOP.TO", "pass2_running", triggered_min=60 * 24 * 5, pass1_min=60 * 24 * 5 - 4
    )
    live = await _run(engine, "META", "pass1_running", triggered_min=3)
    await _call(engine, live, seq=1, attempt=1, validator_passed=True, minutes_ago=1)

    # Validator rejections across runs, for the quality-signals section.
    for run_id, seq in ((ok, 10), (fund_a, 20), (fund_b, 30)):
        await _call(
            engine,
            run_id,
            seq=seq,
            attempt=1,
            validator_passed=False,
            validator_errors=[f"narrative: too long ({1000 + seq} chars, max 1080)"],
        )
        await _call(engine, run_id, seq=seq + 1, attempt=2, validator_passed=True)
    await _call(engine, ok, seq=40, attempt=1, site="agent:tech", validator_passed=True)
    return engine, {"stale": stale, "live": live, "fund_a": fund_a, "fund_b": fund_b}


async def test_health_says_attention_and_counts_what_needs_it():
    engine, ids = await _seeded_week()
    async with _maker(engine)() as session:
        d = await recent_errors(session, WINDOW_START, now=NOW)

    h = d["health"]
    assert h["verdict"] == "ATTENTION"
    assert h["failed"] == 6  # fund_a, fund_b, ollama, recent RY.TO, TD.TO, BNS.TO
    assert h["stale"] == 1 and h["running"] == 1
    assert d["runs_by_status"]["failed"] == 6 and d["runs_by_status"]["completed"] == 2


async def test_the_same_failure_across_runs_is_one_group_with_a_count():
    engine, ids = await _seeded_week()
    async with _maker(engine)() as session:
        d = await recent_errors(session, WINDOW_START, now=NOW)

    fund = next(g for g in d["groups"] if g["agent_name"] == "FUND")
    assert fund["count"] == 2 and fund["runs"] == 2
    assert fund["tickers"] == ["AAPL", "MSFT"]
    assert fund["fingerprint"] == "FUND:narrative: too long (# chars, max #)"
    assert fund["sample_run_id"] == ids["fund_b"]  # the most recent occurrence
    assert fund["new_in_window"] is True


async def test_a_failure_first_seen_long_ago_is_recurring_not_new():
    engine, _ = await _seeded_week()
    async with _maker(engine)() as session:
        d = await recent_errors(session, WINDOW_START, now=NOW)

    recurring = next(g for g in d["groups"] if g["component"] == "data_pipeline")
    assert recurring["count"] == 1  # only the recent one is inside the window
    assert recurring["first_seen"] == _ago(days=30)
    assert recurring["new_in_window"] is False
    assert "NEW" not in next(
        line for line in format_recent_errors(d).splitlines() if "prepare_exception" in line
    )


async def test_groups_are_ordered_most_severe_first():
    engine, _ = await _seeded_week()
    async with _maker(engine)() as session:
        d = await recent_errors(session, WINDOW_START, now=NOW)

    assert d["groups"][0]["severity"] == "critical"
    assert d["groups"][0]["error_type"] == "ollama_connection_failed"


async def test_two_agents_with_the_same_rule_stay_separate_groups():
    engine = await _engine()
    run_id = await _run(engine, "AAPL", "completed", triggered_min=30, completed_min=25)
    for agent in ("FUND", "TECH"):
        await _error(
            engine,
            run_id,
            minutes_ago=20,
            agent=agent,
            fingerprint=f"{agent}:narrative: too long (# chars, max #)",
        )

    async with _maker(engine)() as session:
        d = await recent_errors(session, WINDOW_START, now=NOW)

    assert sorted(g["agent_name"] for g in d["groups"]) == ["FUND", "TECH"]


async def test_low_severity_is_hidden_by_default_and_shown_with_all():
    engine, _ = await _seeded_week()
    async with _maker(engine)() as session:
        default = await recent_errors(session, WINDOW_START, now=NOW)
        everything = await recent_errors(session, WINDOW_START, now=NOW, include_low=True)

    assert all(g["severity"] != "low" for g in default["groups"])
    assert default["health"]["hidden_low_severity"] == 1
    assert any(g["error_type"] == "provider_fallback" for g in everything["groups"])
    assert "use --all" in format_recent_errors(default)


async def test_failed_runs_from_before_the_recorder_are_still_visible_and_grouped():
    engine, _ = await _seeded_week()
    async with _maker(engine)() as session:
        d = await recent_errors(session, WINDOW_START, now=NOW)

    assert len(d["legacy_groups"]) == 1
    legacy = d["legacy_groups"][0]
    assert legacy["fingerprint"] == "legacy:data_pipeline:User-Agent identity is not set"
    assert legacy["count"] == 2 and legacy["tickers"] == ["BNS.TO", "TD.TO"]


async def test_stuck_runs_are_listed_with_their_verdict_and_only_stale_ones_are_flagged():
    engine, ids = await _seeded_week()
    async with _maker(engine)() as session:
        d = await recent_errors(session, WINDOW_START, now=NOW)

    by_id = {s["run_id"]: s for s in d["stuck"]}
    assert by_id[ids["stale"]]["stale"] is True and by_id[ids["stale"]]["ticker"] == "SHOP.TO"
    assert by_id[ids["live"]]["stale"] is False
    text = format_recent_errors(d)
    assert "STALE: SHOP.TO pass2_running" in text and "running: META pass1_running" in text
    assert "--fail-stale" in text


async def test_quality_signals_aggregate_the_validators_rejections_across_runs():
    engine, _ = await _seeded_week()
    async with _maker(engine)() as session:
        d = await recent_errors(session, WINDOW_START, now=NOW)

    top = d["signals"][0]
    assert (top["kind"], top["agent"]) == ("validator", "fund")
    assert top["rule"] == "narrative: too long (# chars, max #)"
    assert top["count"] == 3 and top["runs"] == 3
    assert d["attempts_total"] == 8 and d["attempts_rejected"] == 3
    assert "3 of 8 agent attempts rejected by a validator" in format_recent_errors(d)


async def test_a_quiet_system_is_reported_healthy():
    engine = await _engine()
    await _run(engine, "AAPL", "completed", triggered_min=60, completed_min=55)

    async with _maker(engine)() as session:
        d = await recent_errors(session, WINDOW_START, now=NOW)

    assert d["health"]["verdict"] == "HEALTHY"
    text = format_recent_errors(d)
    assert "HEALTH: HEALTHY" in text and "(none)" in text


async def test_the_window_excludes_older_failures():
    engine, _ = await _seeded_week()
    async with _maker(engine)() as session:
        d = await recent_errors(session, NOW - timedelta(hours=1), now=NOW)

    assert d["health"]["failed"] == 1  # only the recent RY.TO failure (50 min ago)
    assert [g["error_type"] for g in d["groups"]] == ["prepare_exception"]


# --- ending stale runs ----------------------------------------------------------------


async def test_end_stale_runs_ends_only_the_stale_one_and_records_it():
    engine, ids = await _seeded_week()

    ended = await end_stale_runs(engine, now=NOW)

    assert [e["run_id"] for e in ended] == [ids["stale"]]
    async with _maker(engine)() as session:
        stale = (
            await session.execute(select(AnalysisRun).where(AnalysisRun.run_id == ids["stale"]))
        ).scalar_one()
        live = (
            await session.execute(select(AnalysisRun).where(AnalysisRun.run_id == ids["live"]))
        ).scalar_one()
        rows = (
            (
                await session.execute(
                    select(ErrorRecord).where(ErrorRecord.error_type == "run_orphaned")
                )
            )
            .scalars()
            .all()
        )
    assert stale.status == "failed" and live.status == "pass1_running"
    assert len(rows) == 1 and rows[0].stock_ticker == "SHOP.TO"
    assert await end_stale_runs(engine, now=NOW) == []  # idempotent


async def test_a_row_standing_for_many_events_counts_as_that_many():
    """250 unscored articles are ONE row with occurrence_count=250, not 250 rows;
    the report must count events, or the biggest problem would look like a one-off."""
    engine = await _engine()
    a = await _run(engine, "AAPL", "completed", triggered_min=60, completed_min=55)
    b = await _run(engine, "MSFT", "completed", triggered_min=30, completed_min=25)
    fp = "data:ollama:sentiment:sentiment_unscored"
    common = dict(
        severity="low",
        component="data_pipeline",
        error_type="sentiment_unscored",
        agent=None,
        fingerprint=fp,
        message="articles could not be scored",
    )
    await _error(engine, a, minutes_ago=59, ticker="AAPL", occurrence_count=250, **common)
    await _error(engine, b, minutes_ago=29, ticker="MSFT", occurrence_count=3, **common)

    async with _maker(engine)() as session:
        d = await recent_errors(session, WINDOW_START, now=NOW, include_low=True)

    group = d["groups"][0]
    assert group["count"] == 253 and group["runs"] == 2
    assert "x253 in 2 run(s)" in format_recent_errors(d)


async def test_the_error_report_survives_a_malformed_error_log_entry():
    engine = await _engine()
    await _run(engine, "AAPL", "failed", triggered_min=30, error_log=["a bare string", 7, None])

    async with _maker(engine)() as session:
        d = await recent_errors(session, WINDOW_START, now=NOW)

    assert d["legacy_groups"][0]["fingerprint"].startswith("legacy:unknown:")


async def test_a_credential_stored_upstream_is_not_printed_by_the_diagnosis():
    """agent_outputs.error_detail and llm_calls.validator_errors / parse_error are
    stored as reported; the report redacts on the way out."""
    engine = await _engine()
    run_id = await _run(
        engine, "AAPL", "failed", triggered_min=30, error_log=[{"stage": "pass1", "error": "x"}]
    )
    await _agent(engine, run_id, "FUND", "failed", "GET https://x.test/v3?apikey=S3CRET failed")
    await _call(
        engine,
        run_id,
        seq=1,
        attempt=1,
        validator_passed=False,
        validator_errors=["leaked token=S3CRET here"],
        parse_error="apikey: S3CRET",
    )

    async with _maker(engine)() as session:
        text = format_diagnosis(await diagnose_run(session, run_id, now=NOW))

    assert "S3CRET" not in text and "REDACTED" in text


# --- readability of real output ---------------------------------------------------------


async def test_traceback_marker_lines_are_dropped_from_the_tail():
    """Python 3.11+ underlines the failing expression with ^^^^ / ~~~~; that is
    noise once the source line is not next to it."""
    engine = await _engine()
    run_id = await _run(
        engine, "AAPL", "failed", triggered_min=30, error_log=[{"stage": "pass1", "error": "x"}]
    )
    await _error(
        engine,
        run_id,
        minutes_ago=20,
        stack_trace=(
            "Traceback (most recent call last):\n"
            '  File "base.py", line 276, in call_model\n'
            "    await _preflight()\n"
            "    ^^^^^^^^^^^^^^^^^^\n"
            "  ~~~~~~~~~~~~~~~~\n"
            "OllamaUnavailable: down"
        ),
    )

    async with _maker(engine)() as session:
        d = await diagnose_run(session, run_id, now=NOW)

    tail = d["errors"][0]["traceback_tail"]
    assert tail[-1] == "OllamaUnavailable: down"
    assert not any(set(line.strip()) <= set("^~") for line in tail)


async def test_agents_that_failed_for_the_same_reason_are_reported_on_one_line():
    """Ollama being down fails all five Pass 1 agents with the same message."""
    engine = await _engine()
    run_id = await _run(
        engine, "AAPL", "failed", triggered_min=30, error_log=[{"stage": "pass1", "error": "x"}]
    )
    for name in ("RSRCH", "FUND", "TECH"):
        await _agent(
            engine, run_id, name, "failed", "Cannot reach Ollama at http://localhost:19999"
        )

    async with _maker(engine)() as session:
        text = format_diagnosis(await diagnose_run(session, run_id, now=NOW))

    assert text.count("Cannot reach Ollama at http://localhost:19999") == 1
    assert "RSRCH (pass1), FUND (pass1), TECH (pass1) failed" in text
    assert "(3 of 3)" in text


async def test_times_are_shown_to_the_second_not_with_microseconds():
    engine, _ = await _seeded_week()
    async with _maker(engine)() as session:
        d = await recent_errors(session, WINDOW_START, now=NOW)

    text = format_recent_errors(d)
    assert "since 2026-09-22 12:00:00 UTC" in text
    assert "first seen 2026-09-29 " in text
    import re

    assert not re.search(r"\d{2}:\d{2}:\d{2}\.\d+", text)


# --- the end-of-run summary ----------------------------------------------------------


async def _summary_for(engine, run_id) -> str:
    async with _maker(engine)() as session:
        return format_run_summary(await diagnose_run(session, run_id, now=NOW))


async def test_a_clean_run_summary_says_so_and_points_to_diagnose_run():
    engine = await _engine()
    run_id = await _run(engine, "AAPL", "completed", triggered_min=60, completed_min=55)
    await _agent(engine, run_id, "FUND")
    await _call(engine, run_id, seq=1, attempt=1, validator_passed=True)

    text = await _summary_for(engine, run_id)

    assert "RESULT: COMPLETED with no recorded problems" in text
    assert "problems:" not in text
    assert f"python scripts/diagnose_run.py {run_id}" in text


async def test_a_run_that_completed_on_degraded_data_lists_what_degraded():
    """The case the summary exists for: status completed, outlook printed, but
    every article went unscored and one agent needed a retry."""
    engine = await _engine()
    run_id = await _run(engine, "KO", "completed", triggered_min=60, completed_min=55)
    await _error(
        engine,
        run_id,
        minutes_ago=58,
        severity="high",
        component="data_pipeline",
        error_type="sentiment_unscored",
        agent=None,
        fingerprint="data:ollama:sentiment:sentiment_unscored",
        message="248 of 248 articles could not be scored",
        occurrence_count=1,
    )
    await _error(
        engine,
        run_id,
        minutes_ago=58,
        severity="medium",
        component="data_pipeline",
        error_type="llm_request_failed",
        agent=None,
        fingerprint="data:ollama:filing_summary:llm_request_failed",
        message="timeout (Business)",
        occurrence_count=2,
    )
    await _agent(engine, run_id, "FUND")
    await _call(
        engine,
        run_id,
        seq=1,
        attempt=1,
        validator_passed=False,
        validator_errors=["caveats is empty -- must explain why"],
    )
    await _call(engine, run_id, seq=2, attempt=2, validator_passed=True)

    text = await _summary_for(engine, run_id)

    assert "COMPLETED, but 2 error record(s); 1 other agent(s) needed retries" in text
    # A data-layer row is labelled by its fingerprint, which names the provider
    # and operation (the message alone is only the exception text).
    assert (
        "[high] data:ollama:sentiment:sentiment_unscored: 248 of 248 articles could not be scored"
        in text
    )
    assert "[medium] data:ollama:filing_summary:llm_request_failed x2: timeout (Business)" in text
    assert (
        "fund: 1 of 2 attempt(s) rejected, then passed (caveats is empty -- must explain why)"
        in text
    )
    # Most severe first.
    assert text.index("[high]") < text.index("[medium]")


async def test_a_failed_run_summary_gives_the_reason_even_without_error_rows():
    engine = await _engine()
    run_id = await _run(
        engine,
        "AAPL",
        "failed",
        triggered_min=60,
        completed_min=59,
        error_log=[{"stage": "pass1", "error": "Ollama unavailable"}],
    )

    text = await _summary_for(engine, run_id)

    assert "RESULT: FAILED at pass1: Ollama unavailable" in text
    assert "[pass1] Ollama unavailable" in text


async def test_a_failed_agent_is_not_listed_twice():
    """A failed agent already has an error row; its rejected attempts must not add
    a second "rejected, then passed" line that would be untrue."""
    engine = await _engine()
    run_id = await _run(engine, "AAPL", "completed", triggered_min=60, completed_min=55)
    await _agent(engine, run_id, "FUND", status="failed", error_detail="narrative too long")
    await _error(engine, run_id, minutes_ago=57, agent="FUND")
    await _call(engine, run_id, seq=1, attempt=1, validator_passed=False)
    await _call(engine, run_id, seq=2, attempt=2, validator_passed=False)

    text = await _summary_for(engine, run_id)

    assert "agent=FUND" in text
    assert "accepted" not in text


async def test_low_severity_events_are_counted_not_listed_and_a_long_list_is_capped():
    engine = await _engine()
    run_id = await _run(engine, "AAPL", "completed", triggered_min=60, completed_min=55)
    await _error(engine, run_id, minutes_ago=50, severity="low", agent=None, fingerprint="x:low")
    for i in range(11):
        await _error(
            engine, run_id, minutes_ago=50, agent=f"A{i}", fingerprint=f"A{i}:rule", message=f"m{i}"
        )

    text = await _summary_for(engine, run_id)

    assert "(1 low-severity event(s) not shown)" in text
    assert text.count("[high] agent/all_retries_exhausted") == 8
    assert "... and 3 more" in text


async def test_a_multi_line_error_still_gives_a_one_line_headline():
    """Some provider errors span lines ('Company not found ... \n  Tip: ...'); the
    RESULT line and the batch table rely on the headline being one line."""
    engine = await _engine()
    run_id = await _run(
        engine,
        "KO",
        "failed",
        triggered_min=10,
        completed_min=9,
        error_log=[{"stage": "data_pipeline", "error": "Company not found: 'X'\n  Tip: search"}],
    )
    async with _maker(engine)() as session:
        d = await diagnose_run(session, run_id, now=NOW)
    assert d["verdict"]["headline"] == "FAILED at data_pipeline: Company not found: 'X' Tip: search"


async def test_retries_alone_are_their_own_outcome_named_but_not_a_problem():
    engine = await _engine()
    run_id = await _run(engine, "AAPL", "completed", triggered_min=60, completed_min=55)
    for name, site in (("FUND", "agent:fund"), ("TECH", "agent:tech")):
        await _agent(engine, run_id, name)
        await _call(engine, run_id, seq=1, attempt=1, site=site, validator_passed=False)
        await _call(engine, run_id, seq=2, attempt=2, site=site, validator_passed=True)

    async with _maker(engine)() as session:
        d = await diagnose_run(session, run_id, now=NOW)

    assert d["verdict"]["outcome"] == "COMPLETED WITH RETRIES"
    assert d["verdict"]["headline"].endswith("recovered: fund, tech")


async def test_a_failed_agent_is_not_also_counted_as_needing_retries():
    """FUND failed after two rejected attempts; TECH was rejected once and recovered.
    The verdict must say 1 failed and 1 OTHER agent retried, not count FUND twice."""
    engine = await _engine()
    run_id = await _run(engine, "AAPL", "completed", triggered_min=60, completed_min=55)
    await _agent(engine, run_id, "FUND", status="failed", error_detail="x")
    await _agent(engine, run_id, "TECH")
    await _call(engine, run_id, seq=1, attempt=1, validator_passed=False)
    await _call(engine, run_id, seq=2, attempt=2, validator_passed=False)
    await _call(engine, run_id, seq=3, attempt=1, site="agent:tech", validator_passed=False)
    await _call(engine, run_id, seq=4, attempt=2, site="agent:tech", validator_passed=True)

    async with _maker(engine)() as session:
        d = await diagnose_run(session, run_id, now=NOW)

    assert d["verdict"]["outcome"] == "COMPLETED WITH PROBLEMS"
    assert d["verdict"]["headline"] == (
        "COMPLETED, but 1 agent(s) failed (FUND); 1 other agent(s) needed retries"
    )


async def test_a_truncated_reply_is_reported_as_the_reason_when_no_validator_spoke():
    engine = await _engine()
    run_id = await _run(engine, "AAPL", "completed", triggered_min=60, completed_min=55)
    await _agent(engine, run_id, "FUND")
    await _call(engine, run_id, seq=1, attempt=1, finish_reason="length", parsed_ok=False)
    await _call(engine, run_id, seq=2, attempt=2, validator_passed=True)

    text = await _summary_for(engine, run_id)

    assert "fund: 1 of 2 attempt(s) rejected, then passed (finish_reason=length)" in text


async def test_a_hard_failure_row_keeps_its_component_label_in_the_summary():
    """Only data-layer rows are labelled by fingerprint; a hard failure keeps
    component/error_type and its agent."""
    engine = await _engine()
    run_id = await _run(engine, "AAPL", "completed", triggered_min=60, completed_min=55)
    await _agent(engine, run_id, "FUND", status="failed", error_detail="x")
    await _error(engine, run_id, minutes_ago=57, agent="FUND")

    text = await _summary_for(engine, run_id)

    assert "[high] agent/all_retries_exhausted agent=FUND" in text
    assert "data:" not in text


async def test_low_severity_events_alone_do_not_make_a_run_read_as_having_problems():
    """A provider that declined and a fallback covered it is hidden from the summary
    problem list, so it must not turn the verdict into 'problems' either."""
    engine = await _engine()
    run_id = await _run(engine, "RDDT", "completed", triggered_min=60, completed_min=55)
    await _agent(engine, run_id, "FUND")
    await _call(engine, run_id, seq=1, attempt=1, validator_passed=True)
    for path in ("quote", "dividends"):
        await _error(
            engine,
            run_id,
            minutes_ago=58,
            severity="low",
            component="data_pipeline",
            error_type="not_covered",
            agent=None,
            fingerprint=f"data:fmp:{path}:not_covered",
            message="HTTP 402",
        )

    async with _maker(engine)() as session:
        d = await diagnose_run(session, run_id, now=NOW)

    assert d["verdict"] == {
        "outcome": "OK",
        "headline": "COMPLETED with no recorded problems (2 low-severity event(s), see diagnose_run)",
    }
    text = await _summary_for(engine, run_id)
    assert "problems:" not in text
    assert "(2 low-severity event(s) not shown)" in text


async def test_low_events_beside_retries_still_read_as_retries_not_problems():
    engine = await _engine()
    run_id = await _run(engine, "RDDT", "completed", triggered_min=60, completed_min=55)
    await _agent(engine, run_id, "FUND")
    await _call(engine, run_id, seq=1, attempt=1, validator_passed=False)
    await _call(engine, run_id, seq=2, attempt=2, validator_passed=True)
    await _error(
        engine, run_id, minutes_ago=58, severity="low", agent=None, fingerprint="data:fmp:quote:x"
    )

    async with _maker(engine)() as session:
        d = await diagnose_run(session, run_id, now=NOW)

    assert d["verdict"]["outcome"] == "COMPLETED WITH RETRIES"
    assert d["verdict"]["headline"].endswith("(1 low-severity event(s), see diagnose_run)")


# --- which prompt text a run used (ledger BB-045) ---


async def test_the_diagnosis_says_which_prompt_state_the_run_used():
    engine = await _engine()
    run_id = await _run(
        engine,
        "AAPL",
        "completed",
        triggered_min=60,
        completed_min=55,
        llm_config={"model": "gpt-oss:20b", "prompts_hash": "abc123def456", "code_hash": "0f1e2d3c4b5a"},
    )
    async with _maker(engine)() as session:
        d = await diagnose_run(session, run_id, now=NOW)

    assert d["run"]["prompts_hash"] == "abc123def456"
    assert d["run"]["code_hash"] == "0f1e2d3c4b5a"
    assert "prompts: abc123def456 | code: 0f1e2d3c4b5a" in format_diagnosis(d)


async def test_a_run_from_before_hashes_were_recorded_says_so():
    engine = await _engine()
    run_id = await _run(engine, "AAPL", "completed", triggered_min=60, completed_min=55)
    async with _maker(engine)() as session:
        d = await diagnose_run(session, run_id, now=NOW)

    assert d["run"]["prompts_hash"] is None and d["run"]["code_hash"] is None
    assert "prompts: not recorded | code: not recorded" in format_diagnosis(d)


# --- the run summary says every rule, and when validation was overridden (BB-040) ------


async def test_the_summary_names_every_distinct_rule_not_just_the_first():
    """Both Macro rules fired; showing only the first hid the stale-data one."""
    engine = await _engine()
    run_id = await _run(engine, "AAPL", "completed", triggered_min=60, completed_min=55)
    await _agent(engine, run_id, "MACRO")
    await _call(
        engine,
        run_id,
        seq=1,
        attempt=1,
        site="agent:macro",
        validator_passed=False,
        validator_errors=[
            "risks: need <=3 items, got 4",
            "a real data-quality issue is flagged (stale data) while confidence is high",
        ],
    )
    await _call(engine, run_id, seq=2, attempt=2, site="agent:macro", validator_passed=True)

    text = await _summary_for(engine, run_id)

    assert "macro: 1 of 2 attempt(s) rejected, then passed" in text
    assert "risks: need <=3 items, got 4" in text
    assert "a real data-quality issue is flagged" in text


async def test_the_same_rule_with_different_numbers_is_listed_once_and_extras_are_counted():
    engine = await _engine()
    run_id = await _run(engine, "AAPL", "completed", triggered_min=60, completed_min=55)
    await _agent(engine, run_id, "SENT")
    attempts = [
        ["narrative: too long (2182 chars, max 1080)", "key_factors: need <=3 items, got 4"],
        ["narrative: too long (2173 chars, max 1080)", "summary: too long", "tone: bad"],
    ]
    for attempt, errors in enumerate(attempts, start=1):
        await _call(
            engine,
            run_id,
            seq=attempt,
            attempt=attempt,
            site="agent:sent",
            validator_passed=False,
            validator_errors=errors,
        )
    await _call(engine, run_id, seq=3, attempt=3, site="agent:sent", validator_passed=True)

    text = await _summary_for(engine, run_id)

    assert text.count("narrative: too long") == 1  # two attempts, one rule
    assert "+1 more" in text  # 4 distinct rules, 3 shown


async def test_an_agent_whose_every_attempt_failed_is_reported_as_used_despite_failing():
    """agent_completed() treats validation as a warning, so the last output is used. That is
    not a retry that passed, and it must not read like one."""
    engine = await _engine()
    run_id = await _run(engine, "ENB.TO", "completed", triggered_min=60, completed_min=55)
    await _agent(engine, run_id, "SENT")
    for attempt in (1, 2, 3):
        await _call(
            engine,
            run_id,
            seq=attempt,
            attempt=attempt,
            site="agent:sent",
            validator_passed=False,
            validator_errors=["narrative: too long (2182 chars, max 1080)"],
        )

    async with _maker(engine)() as session:
        d = await diagnose_run(session, run_id, now=NOW)
    text = format_run_summary(d)

    assert d["verdict"]["outcome"] == "COMPLETED WITH PROBLEMS"  # not "with retries"
    assert (
        "1 agent(s) used output that failed validation on every attempt (sent)"
        in d["verdict"]["headline"]
    )
    assert "sent: all 3 attempt(s) failed validation; the last output was used anyway" in text
    assert "then passed" not in text and "then accepted" not in text


async def test_recovered_and_exhausted_agents_are_counted_separately():
    engine = await _engine()
    run_id = await _run(engine, "AAPL", "completed", triggered_min=60, completed_min=55)
    for name in ("SENT", "TECH"):
        await _agent(engine, run_id, name)
    await _call(engine, run_id, seq=1, attempt=1, site="agent:sent", validator_passed=False)
    await _call(engine, run_id, seq=2, attempt=2, site="agent:sent", validator_passed=False)
    await _call(engine, run_id, seq=3, attempt=1, site="agent:tech", validator_passed=False)
    await _call(engine, run_id, seq=4, attempt=2, site="agent:tech", validator_passed=True)

    async with _maker(engine)() as session:
        d = await diagnose_run(session, run_id, now=NOW)

    assert d["verdict"]["headline"] == (
        "COMPLETED, but 1 agent(s) used output that failed validation on every attempt (sent); "
        "1 other agent(s) needed retries"
    )


async def test_a_data_event_row_shows_which_symbols_it_happened_for():
    engine = await _engine()
    run_id = await _run(engine, "MSFT", "completed", triggered_min=60, completed_min=55)
    await _agent(engine, run_id, "FUND")
    await _error(
        engine,
        run_id,
        minutes_ago=58,
        severity="medium",
        component="data_pipeline",
        error_type="empty_after_failure",
        agent=None,
        fingerprint="data:finnhub:get_news:empty_after_failure",
        message="no provider produced data and at least one raised",
        context={"symbols": ["WMT", "PEP"]},
    )

    text = await _summary_for(engine, run_id)

    assert "data:finnhub:get_news:empty_after_failure: no provider produced data" in text
    assert "[WMT, PEP]" in text


# --- run ids as the scripts accept them (BB-051) ----------------------------------------


async def test_a_full_id_a_dashed_prefix_and_a_short_prefix_all_resolve():
    engine = await _engine()
    run_id = await _run(engine, "AAPL", "completed", triggered_min=60, completed_min=55)
    async with _maker(engine)() as session:
        assert await resolve_run_id(session, str(run_id)) == run_id
        assert await resolve_run_id(session, run_id.hex) == run_id
        assert await resolve_run_id(session, run_id.hex[:8]) == run_id  # what the batch prints
        assert await resolve_run_id(session, str(run_id)[:13].upper()) == run_id  # dashes, caps


async def test_an_unknown_or_malformed_id_says_what_is_wrong():
    engine = await _engine()
    run_id = await _run(engine, "AAPL", "completed", triggered_min=60, completed_min=55)
    unused = "eeee" if not run_id.hex.startswith("eeee") else "dddd"
    async with _maker(engine)() as session:
        with pytest.raises(ValueError, match="no recent run"):
            await resolve_run_id(session, unused)
        with pytest.raises(ValueError, match="not a run id"):
            await resolve_run_id(session, "xyz")
        with pytest.raises(ValueError, match="not a run id"):
            await resolve_run_id(session, "ab")  # too short to be safe


async def test_a_prefix_shared_by_two_runs_is_refused_rather_than_guessed():
    engine = await _engine()
    async with _maker(engine)() as session:
        stock = Stock(
            stock_id=uuid4(),
            canonical_ticker="KO",
            company_name="KO",
            primary_exchange="NYSE",
            currency="USD",
        )
        session.add(stock)
        await session.commit()
        for tail in ("1", "2"):
            session.add(
                AnalysisRun(
                    run_id=UUID(f"abcd1234-0000-4000-8000-00000000000{tail}"),
                    user_id=uuid4(),
                    stock_id=stock.stock_id,
                    account_type="tfsa",
                    timeline="medium_term",
                    triggered_by="cli",
                    status="failed",
                    llm_config={},
                    triggered_at=_ago(10 + int(tail)),
                )
            )
        await session.commit()
        with pytest.raises(ValueError, match="matches 2 runs"):
            await resolve_run_id(session, "abcd1234")
        assert str(await resolve_run_id(session, "abcd1234-0000-4000-8000-000000000002")).endswith("2")


# --- quality by prompt state (BB-045) ---------------------------------------------------


async def _agent_run(engine, ticker, *, prompt_hash, code_hash="c0de00000001", minutes=60):
    """A completed run whose tech agent used a template with `prompt_hash`."""
    config = (
        {"prompt_hashes": {"technical_analyst/v1.txt": prompt_hash}, "code_hash": code_hash}
        if prompt_hash
        else {}
    )
    return await _run(
        engine,
        ticker,
        "completed",
        triggered_min=minutes,
        completed_min=minutes - 5,
        llm_config=config,
    )


async def test_rejection_rates_are_split_by_the_agents_own_prompt_hash():
    engine = await _engine()
    old = await _agent_run(engine, "AAPL", prompt_hash="aaaaaaaaaaaa", minutes=90)
    new = await _agent_run(engine, "MSFT", prompt_hash="bbbbbbbbbbbb", minutes=30)
    # old prompt: 2 of 3 attempts rejected; new prompt: 0 of 2
    await _call(engine, old, seq=1, attempt=1, site="agent:tech", validator_passed=False,
                validator_errors=["key_factors[0].sentiment: must be positive|negative|neutral"])
    await _call(engine, old, seq=2, attempt=2, site="agent:tech", validator_passed=False,
                validator_errors=["key_factors[1].sentiment: must be positive|negative|neutral"])
    await _call(engine, old, seq=3, attempt=3, site="agent:tech", validator_passed=True)
    await _call(engine, new, seq=1, attempt=1, site="agent:tech", validator_passed=True)
    await _call(engine, new, seq=2, attempt=1, site="agent:fund", validator_passed=True)

    async with _maker(engine)() as session:
        d = await prompt_comparison(session, NOW - timedelta(days=1))

    tech = {g["prompt_hash"]: g for g in d["groups"] if g["agent"] == "tech"}
    assert set(tech) == {"aaaaaaaaaaaa", "bbbbbbbbbbbb"}
    assert (tech["aaaaaaaaaaaa"]["attempts"], tech["aaaaaaaaaaaa"]["rejected"]) == (3, 2)
    assert round(tech["aaaaaaaaaaaa"]["rate"], 2) == 0.67
    assert (tech["bbbbbbbbbbbb"]["attempts"], tech["bbbbbbbbbbbb"]["rejected"]) == (1, 0)
    # the two sentiment errors are the same rule with a different index: counted together
    assert tech["aaaaaaaaaaaa"]["top_reasons"][0][1] == 2
    text = format_prompt_comparison(d)
    assert "aaaaaaaaaaaa" in text and "67%" in text and "technical_analyst/v1.txt" in text


async def test_runs_without_a_recorded_hash_fall_in_a_not_recorded_group_and_code_changes_are_flagged():
    engine = await _engine()
    unhashed = await _agent_run(engine, "AAPL", prompt_hash=None, minutes=90)
    one = await _agent_run(engine, "MSFT", prompt_hash="bbbbbbbbbbbb", code_hash="c0de00000001")
    two = await _agent_run(engine, "KO", prompt_hash="bbbbbbbbbbbb", code_hash="c0de00000002", minutes=20)
    for run_id in (unhashed, one, two):
        await _call(engine, run_id, seq=1, attempt=1, site="agent:tech", validator_passed=True)

    async with _maker(engine)() as session:
        d = await prompt_comparison(session, NOW - timedelta(days=1))
    text = format_prompt_comparison(d)

    hashes = {g["prompt_hash"]: g for g in d["groups"]}
    assert hashes["not recorded"]["runs"] == 1
    assert hashes["bbbbbbbbbbbb"]["runs"] == 2 and hashes["bbbbbbbbbbbb"]["code_states"] == 2
    assert "CODE CHANGED across 2 states" in text


async def test_the_comparison_ignores_the_window_before_since_and_non_agent_calls():
    engine = await _engine()
    run_id = await _agent_run(engine, "AAPL", prompt_hash="aaaaaaaaaaaa")
    await _call(engine, run_id, seq=1, attempt=1, site="agent:tech", validator_passed=True,
                minutes_ago=60 * 24 * 5)  # five days ago
    await _call(engine, run_id, seq=2, attempt=1, site="precompute:sentiment", validator_passed=True)

    async with _maker(engine)() as session:
        d = await prompt_comparison(session, NOW - timedelta(days=1))

    assert d["groups"] == []
    assert "no agent attempts in this window" in format_prompt_comparison(d)


def test_every_mapped_agent_template_is_a_real_prompt_file():
    """The by-prompt view joins call sites to template files by this table; a renamed or
    removed template would silently make its agent show as "not recorded" forever."""
    from agents.prompts import prompt_fingerprints
    from services.error_report import _AGENT_TEMPLATES

    hashed = set(prompt_fingerprints()["files"])
    missing = {site: path for site, path in _AGENT_TEMPLATES.items() if path not in hashed}
    assert missing == {}


async def test_a_call_the_trimmer_rescued_is_reported_as_a_quality_signal():
    """It never shows as a retry, so a run whose lists were cut looked clean (Fundamental: 43 of 61 first attempts)."""
    engine = await _engine()
    run_id = await _run(engine, "AAPL", "completed", triggered_min=60, completed_min=55)
    await _call(engine, run_id, seq=1, attempt=1, validator_passed=True, auto_trimmed=True)

    async with _maker(engine)() as session:
        d = await recent_errors(session, WINDOW_START, now=NOW)

    assert [(x["kind"], x["agent"]) for x in d["signals"]] == [("trimmed", "fund")]
    assert "trimmed   fund" in format_recent_errors(d)

