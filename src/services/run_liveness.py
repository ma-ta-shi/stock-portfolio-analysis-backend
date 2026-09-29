"""Stuck-run detection and ending (86bc997wr).

A run whose process died (kill -9, the console closed, power loss, a server
restart that never got to clean up) stays non-terminal forever. That has two
costs: a client polling it sees "running" indefinitely, and it blocks every new
run for the same ticker, because only one non-terminal run per stock may exist.
Two real ones sat for 5 and 6 days.

The rule here is deliberately generous, because the one thing it must never do
is end a run that is still working. A non-terminal run is STALE when nothing
has happened for `SP_STALE_RUN_MINUTES` (default 90). "Something happened" is
read from timestamps that already exist: when the run was triggered, when each
pass finished, and the newest llm_calls / agent_outputs row.

Why 90 is safe. The longest a live run can be silent is its first phase
(prepare() then Pass 1), and the orchestrator never moves a run out of `queued`
during it (it never sets pass1_running), so a live run in its first phase looks
exactly like one that never started. prepare()'s precompute llm_calls rows are
committed when it returns, which usually restarts the clock, but that is not
guaranteed (no LLM calls, no rows), so the derivation ignores it. Worst case,
if EVERY call hangs until its own timeout: sentiment scoring (250 articles / 10
concurrent x 60s = 25 min) + filing summaries (~5 min) + Pass 1 (3 retries x
300s = 15 min) is about 45 min. Real runs take 5 to 10. So 90 is twice the
worst case. The cost of that margin is only that a run whose process died
lingers up to 90 minutes instead of forever (two real ones sat for 5 and 6
days); ending a live run would cost far more. There is deliberately no shorter
special case for `queued`.

Assumptions to re-check before relying on this elsewhere:
- Every timestamp is UTC. True on SQLite (CURRENT_TIMESTAMP is UTC). On
  PostgreSQL, `func.now()` defaults are server-local time and PostgreSQL's now()
  is the transaction start, so the database timezone must be UTC (or these
  comparisons must move to database-side time) before deploying there: a server
  behind UTC would make a brand-new run look hours old.
- Runs are created and started together. A batch scheduler that pre-creates
  many `queued` runs and works through them one at a time (see
  docs/technical/for-later/batch-analysis-scheduler.md) would have a run wait
  behind others, silent, longer than the limit: it would need to create runs
  when it starts them, or this rule would need the owning-process check below.

Ending a run is a single conditional UPDATE that only fires if the run is still
in the status that was assessed, so it can never overwrite a run that finished
in the meantime, and calling it twice is harmless.

The residual risk, stated plainly: a process that is alive but hung for longer
than the limit (every call stuck to its timeout, or the event loop blocked) and
then resumed could find its run already ended and overwrite the status on its
next stage commit. If a false positive ever appears, the upgrade is to record
the owning process on the run (pid plus create time) and test it here; this
module is the only place that would change.
"""

import os
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from api.tables.agent_outputs import AgentOutput
from api.tables.analysis_runs import AnalysisRun, RunStatus
from api.tables.llm_calls import LLMCall
from api.tables.stock import Stock
from services.error_recorder import build_error_note, safe_text, write_error_notes

logger = structlog.get_logger(__name__)

DEFAULT_STALE_MINUTES = 90
TERMINAL_STATUSES = (RunStatus.COMPLETED.value, RunStatus.FAILED.value)


def stale_minutes() -> int:
    """SP_STALE_RUN_MINUTES, falling back to the default on anything unusable."""
    raw = os.environ.get("SP_STALE_RUN_MINUTES")
    try:
        value = int(raw) if raw else DEFAULT_STALE_MINUTES
    except ValueError:
        value = DEFAULT_STALE_MINUTES
    return value if value > 0 else DEFAULT_STALE_MINUTES


def _naive_utc(value: datetime | None) -> datetime | None:
    """Every timestamp here is UTC. SQLite hands them back naive, but a value
    set in Python (datetime.now(UTC)) can still be timezone-aware in the same
    session; comparing the two raises TypeError (the SENT agent's real
    'offset-naive and offset-aware' bug), so normalize both sides."""
    if value is None:
        return None
    if value.tzinfo is not None:
        return value.astimezone(UTC).replace(tzinfo=None)
    return value


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


@dataclass(frozen=True)
class Assessment:
    stale: bool
    reason: str
    idle_minutes: float


def assess(
    status: str,
    triggered_at: datetime | None,
    last_activity_at: datetime | None,
    now: datetime | None = None,
    *,
    stale_after: int | None = None,
) -> Assessment:
    """Pure: is this run stale? The clock and the limit are injectable."""
    if status in TERMINAL_STATUSES:
        return Assessment(False, "finished", 0.0)
    current = _naive_utc(now) if now is not None else _now()
    limit = stale_after if stale_after is not None else stale_minutes()
    triggered = _naive_utc(triggered_at)
    last = _naive_utc(last_activity_at) or triggered
    if last is None:
        return Assessment(False, "no timestamps to judge by", 0.0)
    idle = (current - last).total_seconds() / 60
    if idle > limit:
        return Assessment(True, f"no activity for {idle:.0f} min (limit {limit})", idle)
    return Assessment(False, "active", idle)


@dataclass(frozen=True)
class RunSnapshot:
    run_id: object
    status: str
    triggered_at: datetime | None
    last_activity_at: datetime | None
    ticker: str | None
    user_id: object
    account_type: str | None
    timeline: str | None


async def load_snapshot(session, run_id) -> RunSnapshot | None:
    """The run plus its last-activity timestamp, from data that already exists."""
    row = (
        await session.execute(
            select(
                AnalysisRun.status,
                AnalysisRun.triggered_at,
                AnalysisRun.pass1_completed_at,
                AnalysisRun.pass2_completed_at,
                AnalysisRun.user_id,
                AnalysisRun.account_type,
                AnalysisRun.timeline,
                Stock.canonical_ticker,
            )
            .join(Stock, Stock.stock_id == AnalysisRun.stock_id, isouter=True)
            .where(AnalysisRun.run_id == run_id)
        )
    ).one_or_none()
    if row is None:
        return None
    newest_call = (
        await session.execute(select(func.max(LLMCall.created_at)).where(LLMCall.run_id == run_id))
    ).scalar_one()
    newest_output = (
        await session.execute(
            select(func.max(AgentOutput.created_at)).where(AgentOutput.run_id == run_id)
        )
    ).scalar_one()
    seen = [
        _naive_utc(t)
        for t in (
            row.triggered_at,
            row.pass1_completed_at,
            row.pass2_completed_at,
            newest_call,
            newest_output,
        )
        if t is not None
    ]
    return RunSnapshot(
        run_id=run_id,
        status=row.status,
        triggered_at=_naive_utc(row.triggered_at),
        last_activity_at=max(seen) if seen else None,
        ticker=row.canonical_ticker,
        user_id=row.user_id,
        account_type=row.account_type,
        timeline=row.timeline,
    )


async def end_run_now(bind, run_id, *, stage: str, message: str) -> bool:
    """End a run that is definitely over (e.g. its own task was cancelled).

    A conditional UPDATE in its own session: it only fires while the run is
    non-terminal, so it can never overwrite a finished run, and it works even
    when the run's own session is mid-rollback. Returns True if it ended one.
    Never raises for an ordinary failure.
    """
    try:
        factory = async_sessionmaker(bind, expire_on_commit=False)
        async with factory() as session:
            result = await session.execute(
                update(AnalysisRun)
                .where(
                    AnalysisRun.run_id == run_id,
                    AnalysisRun.status.not_in(list(TERMINAL_STATUSES)),
                )
                .values(
                    status=RunStatus.FAILED.value,
                    error_log=[{"stage": stage, "error": safe_text(message, 500)}],
                )
            )
            await session.commit()
            return result.rowcount == 1
    except Exception:
        logger.warning("end_run_failed", run_id=str(run_id), stage=stage, exc_info=True)
        return False


async def end_run_if_stale(
    bind, run_id, *, now: datetime | None = None, stale_after: int | None = None
) -> bool:
    """End the run if (and only if) it is stale. Returns True if it was ended.

    Cheap and idempotent, so it is safe to call from a request path: the
    conditional UPDATE matches the exact status that was assessed, so a run that
    moved on in the meantime is left alone, and a second call finds it terminal.
    Records a `run_orphaned` row in error_records. Never raises for an ordinary
    failure (a problem here must not break the request that asked).
    """
    try:
        factory = async_sessionmaker(bind, expire_on_commit=False)
        async with factory() as session:
            snapshot = await load_snapshot(session, run_id)
            if snapshot is None:
                return False
            verdict = assess(
                snapshot.status,
                snapshot.triggered_at,
                snapshot.last_activity_at,
                now,
                stale_after=stale_after,
            )
            if not verdict.stale:
                return False
            message = f"Run ended as stale: {verdict.reason}. Last status was {snapshot.status}."
            result = await session.execute(
                update(AnalysisRun)
                .where(AnalysisRun.run_id == run_id, AnalysisRun.status == snapshot.status)
                .values(
                    status=RunStatus.FAILED.value,
                    error_log=[{"stage": "orphaned", "error": message}],
                )
            )
            await session.commit()
            ended = result.rowcount == 1
        if ended:
            await write_error_notes(
                bind,
                [
                    build_error_note(
                        "orchestrator",
                        "run_orphaned",
                        "medium",
                        message,
                        run_id=run_id,
                        user_id=snapshot.user_id,
                        stock_ticker=snapshot.ticker,
                        dedup_subtype=f"stale:{snapshot.status}",
                        context={
                            "last_status": snapshot.status,
                            "triggered_at": str(snapshot.triggered_at),
                            "last_activity_at": str(snapshot.last_activity_at),
                            "idle_minutes": round(verdict.idle_minutes, 1),
                            "stale_after_minutes": stale_after
                            if stale_after is not None
                            else stale_minutes(),
                            "account_type": snapshot.account_type,
                            "timeline": snapshot.timeline,
                            "ended_by": "stale-run rule",
                        },
                    )
                ],
            )
            logger.warning("stale_run_ended", run_id=str(run_id), reason=verdict.reason)
        return ended
    except Exception:
        logger.warning("end_stale_run_failed", run_id=str(run_id), exc_info=True)
        return False


async def end_stale_runs(
    bind, *, now: datetime | None = None, stale_after: int | None = None
) -> list[dict]:
    """End every stale run (the `recent_errors.py --fail-stale` action). Returns
    the listing entries of the runs that were actually ended. Each one goes
    through end_run_if_stale, so the same conditional-UPDATE guarantees apply."""
    factory = async_sessionmaker(bind, expire_on_commit=False)
    async with factory() as session:
        listing = await list_non_terminal_runs(session, now, stale_after=stale_after)
    ended = []
    for entry in listing:
        if entry["stale"] and await end_run_if_stale(
            bind, entry["run_id"], now=now, stale_after=stale_after
        ):
            ended.append(entry)
    return ended


async def list_non_terminal_runs(
    session, now: datetime | None = None, *, stale_after: int | None = None
) -> list[dict]:
    """Every run that has not finished, with its age, last activity and verdict
    (for the reports; does not change anything)."""
    run_ids = (
        (
            await session.execute(
                select(AnalysisRun.run_id).where(AnalysisRun.status.not_in(list(TERMINAL_STATUSES)))
            )
        )
        .scalars()
        .all()
    )
    current = _naive_utc(now) if now is not None else _now()
    out = []
    for run_id in run_ids:
        snapshot = await load_snapshot(session, run_id)
        if snapshot is None:
            continue
        verdict = assess(
            snapshot.status,
            snapshot.triggered_at,
            snapshot.last_activity_at,
            current,
            stale_after=stale_after,
        )
        out.append(
            {
                "run_id": run_id,
                "ticker": snapshot.ticker,
                "status": snapshot.status,
                "triggered_at": snapshot.triggered_at,
                "last_activity_at": snapshot.last_activity_at,
                "age_minutes": (current - snapshot.triggered_at).total_seconds() / 60
                if snapshot.triggered_at
                else None,
                "idle_minutes": verdict.idle_minutes,
                "stale": verdict.stale,
                "reason": verdict.reason,
            }
        )
    return out
