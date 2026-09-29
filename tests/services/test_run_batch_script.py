"""Tests for the subprocess driver in scripts/run_batch.py (86bc997wr): the part
that runs each analysis as its own process, reads its run_id from the output and
kills it on a timeout. Driven with tiny stand-in commands, never a real model.
"""

import importlib.util
import sys
from pathlib import Path
from uuid import UUID

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_batch.py"
RUN_ID = "7d62ef8d-3007-43ea-9cd7-300d65ebe5cc"


@pytest.fixture(scope="module")
def run_batch():
    # Running a script puts its own folder on sys.path (that is how it finds
    # _bootstrap); loading it by path from a test has to do the same.
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("run_batch_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


async def test_the_run_id_and_exit_code_come_from_the_child(run_batch, capsys):
    code = f"print('Starting analysis: KO | account=tfsa | run_id={RUN_ID}'); print('done'); raise SystemExit(1)"
    run_id, timed_out, exit_code = await run_batch._run_one(_py(code), timeout_s=30)
    assert (run_id, timed_out, exit_code) == (UUID(RUN_ID), False, 1)
    assert "done" in capsys.readouterr().out  # the child's output is echoed live


async def test_a_child_that_never_starts_a_run_gives_no_run_id(run_batch):
    code = "print('No user profile found for user_id=x -- create one first')"
    run_id, timed_out, exit_code = await run_batch._run_one(_py(code), timeout_s=30)
    assert (run_id, timed_out, exit_code) == (None, False, 0)


async def test_a_blocked_start_does_not_borrow_the_other_runs_id(run_batch):
    code = f"print('An analysis is already running for KO: run_id={RUN_ID} status=queued')"
    run_id, _, _ = await run_batch._run_one(_py(code), timeout_s=30)
    assert run_id is None


async def test_a_hung_child_is_killed_and_reported_with_its_run_id(run_batch):
    code = (
        f"import time; print('Starting analysis: KO | run_id={RUN_ID}', flush=True); time.sleep(60)"
    )
    run_id, timed_out, exit_code = await run_batch._run_one(_py(code), timeout_s=2)
    assert (run_id, timed_out, exit_code) == (UUID(RUN_ID), True, None)


async def test_a_run_the_batch_kills_is_ended_and_leaves_an_error_row(run_batch):
    from datetime import datetime
    from uuid import uuid4

    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from api.database import Base
    from api.tables.analysis_runs import AnalysisRun
    from api.tables.error_records import ErrorRecord
    from api.tables.stock import Stock

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        stock = Stock(
            stock_id=uuid4(),
            canonical_ticker="TD.TO",
            company_name="TD",
            primary_exchange="TSX",
            currency="CAD",
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
            status="queued",
            llm_config={},
            triggered_at=datetime(2026, 9, 29, 12, 0, 0),
        )
        session.add(run)
        await session.commit()
        run_id = run.run_id

    await run_batch._end_timed_out_run(engine, run_id, "TD.TO", 1)
    await run_batch._end_timed_out_run(engine, run_id, "TD.TO", 1)  # a second call is harmless

    async with maker() as session:
        ended = (
            await session.execute(select(AnalysisRun).where(AnalysisRun.run_id == run_id))
        ).scalar_one()
        rows = (await session.execute(select(ErrorRecord))).scalars().all()
    assert ended.status == "failed"
    assert ended.error_log[0]["stage"] == "timeout"
    assert len(rows) == 1  # written once, not once per call
    assert (rows[0].error_type, rows[0].stock_ticker, rows[0].dedup_subtype) == (
        "run_timed_out",
        "TD.TO",
        "timeout",
    )
