"""Analysis API routes (86bbuhjup): POST /api/analysis + status/results,
matching docs/api-conventions.md's "Analysis Workflow" section.

WebSocket emission (`agent_complete`/`pass_complete`) is explicitly OUT of
scope for this port -- no WebSocket code exists anywhere in this repo today
despite CLAUDE.md's pipeline description referencing one; polling the status
endpoint (this file's own GET .../status) is sufficient to prove the
pipeline runs end-to-end, which was this session's stated goal. Real-time
push is a UX layer on top of a working orchestrator, not a precondition for
one -- flagged, not silently dropped.
"""
import re

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID, uuid4

from agents.base import MODEL
from api.database import AsyncSessionLocal
from api.schemas.analysis import (
    AgentOutputSummary,
    AnalysisCreate,
    AnalysisCreateResponse,
    AnalysisFailure,
    AnalysisProgress,
    AnalysisResultResponse,
    AnalysisStatusResponse,
)
from api.tables.agent_outputs import AgentOutput
from api.tables.analysis_runs import AnalysisRun, RunStatus
from api.tables.recommendations import Recommendation
from api.tables.stock import Stock
from data.providers.router import Router
from services.error_recorder import safe_text
from services.orchestrator import AnalysisOrchestrator
from services.run_liveness import TERMINAL_STATUSES, end_run_if_stale

logger = structlog.get_logger(__name__)

router = APIRouter()

# RunStatus (fine-grained, DB-internal: queued/pass1_running/pass1_complete/
# pass2_running/pass2_complete/synthesis_running/completed/failed) -> the
# coarser external vocabulary docs/api-conventions.md documents for this
# endpoint ("queued"|"running"|"complete"|"failed"|"cancelled"|"timeout").
# cancelled/timeout are real, documented values this orchestrator just never
# produces today (no cancellation or timeout-specific handling exists) --
# not wrong to omit, this mapping just never emits them.
_STATUS_TO_EXTERNAL = {
    RunStatus.QUEUED: ("queued", None),
    RunStatus.PASS1_RUNNING: ("running", 1),
    RunStatus.PASS1_COMPLETE: ("running", 1),
    RunStatus.PASS2_RUNNING: ("running", 2),
    RunStatus.PASS2_COMPLETE: ("running", 2),
    RunStatus.SYNTHESIS_RUNNING: ("running", 2),
    RunStatus.COMPLETED: ("complete", None),
    RunStatus.FAILED: ("failed", None),
}


def _external_status(raw_status: str) -> tuple[str, int | None]:
    """The status column has no DB-level enum/CHECK constraint (plain
    String(30) -- see RunStatus's own docstring), and this orchestrator is
    the only writer today, but a stale row from a schema this enum no
    longer covers, or any other unexpected value, must degrade to a clear
    result rather than an unhandled ValueError from RunStatus(raw_status)
    surfacing as a raw, undiagnosable 500 to an API caller.
    """
    try:
        return _STATUS_TO_EXTERNAL[RunStatus(raw_status)]
    except ValueError:
        logger.error("unrecognized_run_status", status=raw_status)
        return "failed", None


async def get_async_db():
    async with AsyncSessionLocal() as db:
        yield db


def _not_found(code: str, message: str, **details) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={"error": {"code": code, "message": message, "details": details}},
    )


async def resolve_or_create_stock(ticker: str, db: AsyncSession) -> Stock:
    """orchestration-engine.md's own documented step 2 ("Resolve ticker ->
    canonical stock_id") -- DataPipeline.prepare() takes a stock_id, not a
    ticker string, so this is a real, necessary prerequisite, not an
    implementation detail that falls out for free.

    No longer module-private (86bc90p0a): `scripts/analyze.py` is a second,
    real caller outside this module, which made the old leading-underscore
    name actively misleading rather than just unused convention.
    """
    existing = (
        await db.execute(select(Stock).where(Stock.canonical_ticker == ticker))
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    async with Router(ticker=ticker) as provider_router:
        info = await provider_router.get_company_info(ticker)
    if not info:
        raise _not_found("STOCK_NOT_FOUND", f"No company info found for ticker {ticker}", ticker=ticker)

    # NormalizedCompanyInfo.asset_type gate (86bbpk6uf part 2): equity-only
    # agents must never score a fund/index/preferred -- see
    # data/providers/base.py's own docstring for why this field exists.
    asset_type = info.get("asset_type")
    if asset_type in ("etf", "other"):
        raise HTTPException(
            status_code=422,
            detail={
                "error": {
                    "code": "UNSUPPORTED_ASSET_TYPE",
                    "message": f"{ticker} is asset_type={asset_type!r}; this pipeline covers individual equities only",
                    "details": {"ticker": ticker, "asset_type": asset_type},
                }
            },
        )

    stock = Stock(
        stock_id=uuid4(),
        canonical_ticker=ticker,
        company_name=info.get("name") or ticker,
        primary_exchange=info.get("primary_exchange") or "",
        currency=info.get("currency") or "",
        sector=info.get("sector"),
        industry=info.get("industry"),
    )
    # Real TOCTOU race, confirmed live (2026-09-23): two concurrent requests
    # for the same brand-new ticker can both pass the "existing" SELECT
    # above before either commits -- stocks.canonical_ticker's own real
    # UNIQUE constraint (stock.py) stops a duplicate ROW from ever
    # persisting, but without this catch the LOSING request's own commit
    # raised an unhandled IntegrityError straight into an unhandled 500,
    # reproduced directly via two genuinely concurrent httpx requests
    # (tests/api/test_analysis_routes.py). SAVEPOINT isolation (not a bare
    # db.add()+commit()) matches the same pattern this session already
    # established for _run_shadow_cio's own failed-commit recovery --
    # its own docstring explains why a plain try/db.rollback() around a
    # non-nested commit is the wrong shape here.
    try:
        async with db.begin_nested():
            db.add(stock)
            await db.flush()
    except IntegrityError:
        await db.rollback()
        return (
            await db.execute(select(Stock).where(Stock.canonical_ticker == ticker))
        ).scalar_one()
    await db.commit()
    await db.refresh(stock)
    return stock


# Local filesystem paths in an exception message (a FileNotFoundError, say) are
# not something an API caller should see.
# A single drive letter NOT preceded by another word character: without that,
# the `s:` at the end of `https:` looks like a drive and swallows the whole URL.
_WINDOWS_PATH = re.compile(r"(?<!\w)[A-Za-z]:[\\/][^\s'\"<>|]+")
# Not preceded by a word character, dot, colon, dash or slash: that keeps the
# path part of a URL (https://host/v3/x) from being mistaken for a file path.
_UNIX_PATH = re.compile(r"(?<![\w.:/-])/(?:[\w.@+-]+/)+[\w.@+-]*")


def _failure_from(run: AnalysisRun, external_status: str) -> AnalysisFailure | None:
    """The sanitized reason a failed run stopped, from analysis_runs.error_log.

    None unless the run failed. A stage name and a short single-line message:
    credentials redacted (safe_text), local paths removed, whitespace collapsed
    (so a multi-line message can't smuggle a traceback), capped at 300 chars.
    """
    if external_status != "failed":
        return None
    entries = [e for e in (run.error_log or []) if isinstance(e, dict)]
    entry = entries[0] if entries else {}
    stage = " ".join(str(entry.get("stage") or "unknown").split())[:50]
    message = " ".join(safe_text(entry.get("error") or "no reason recorded", 1000).split())
    message = _UNIX_PATH.sub("[path]", _WINDOWS_PATH.sub("[path]", message))[:300]
    return AnalysisFailure(stage=stage, message=message)


async def _end_if_stale(db: AsyncSession, run: AnalysisRun) -> None:
    """A client polling a run whose process died would otherwise see "running"
    forever. If the run is stale, end it (idempotent conditional UPDATE, see
    services/run_liveness.py) and reload it, so the response says "failed"."""
    if run.status not in TERMINAL_STATUSES and await end_run_if_stale(db.bind, run.run_id):
        await db.refresh(run)


async def _fail_run_if_still_active(run_id: UUID, exc: BaseException) -> None:
    """Safety net (86bc997wr): an exception that escaped the orchestrator
    without one of its own handled branches marking the run FAILED (a bug in a
    pass, an unguarded recommendation/prediction write) leaves the run stuck in
    a non-terminal status forever -- the real database already holds two such
    orphans. Marks it FAILED.

    A conditional UPDATE, not an ORM write: it only touches a run that is still
    non-terminal, so it can never overwrite a `completed` run or clobber the
    error_log a handled failure already wrote. Runs in its own fresh session,
    because the orchestrator's may be mid-rollback. Status only -- the error
    row for this same exception is already written by AnalysisOrchestrator.run()
    (its catch-all), so recording it again here would double-count it.

    Never raises: a failure here is logged, not allowed to mask the original.
    """
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(
                update(AnalysisRun)
                .where(
                    AnalysisRun.run_id == run_id,
                    AnalysisRun.status.not_in([RunStatus.COMPLETED.value, RunStatus.FAILED.value]),
                )
                .values(
                    status=RunStatus.FAILED.value,
                    error_log=[{
                        "stage": "unhandled",
                        "error": f"{type(exc).__name__}: {safe_text(exc)}",
                    }],
                )
            )
            await db.commit()
    except Exception:
        logger.exception("background_run_status_repair_failed", run_id=str(run_id))


async def _run_analysis_background(run_id: UUID) -> None:
    """Own, independent AsyncSession -- deliberately NOT the request-scoped
    session FastAPI's `Depends(get_async_db)` yields. BackgroundTasks run
    after the response is sent, and the ordering between a yield-dependency's
    teardown and background task execution is not something to rely on --
    opening a fresh session here sidesteps that ambiguity entirely rather
    than risking a closed session mid-run.
    """
    async with AsyncSessionLocal() as db:
        run = (
            await db.execute(select(AnalysisRun).where(AnalysisRun.run_id == run_id))
        ).scalar_one()
        try:
            await AnalysisOrchestrator().run(run, db)
        except Exception as exc:
            # AnalysisOrchestrator.run() already sets run.status=FAILED and
            # commits on every caught failure path before re-raising -- this
            # catch exists so the exception is actually logged (FastAPI/
            # Starlette otherwise swallows a background task's exception
            # silently, a real observability gap for a run that's already the
            # ONE thing running unattended), and so a run that an UNHANDLED
            # exception left non-terminal still ends FAILED.
            logger.exception("background_analysis_failed", run_id=str(run_id))
            await _fail_run_if_still_active(run_id, exc)


@router.post("/", response_model=AnalysisCreateResponse, status_code=202)
async def create_analysis(
    body: AnalysisCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_db),
):
    stock = await resolve_or_create_stock(body.ticker, db)

    in_progress = (
        await db.execute(
            select(AnalysisRun).where(
                AnalysisRun.stock_id == stock.stock_id,
                AnalysisRun.status.notin_([RunStatus.COMPLETED, RunStatus.FAILED]),
            )
        )
    ).scalars().first()
    # A run whose process died stays non-terminal forever and would block this
    # ticker for good. If the blocker is provably stale (no activity for far
    # longer than any real run can be silent -- see services/run_liveness.py),
    # end it and carry on; a run that is actually working still gets the 409.
    if in_progress is not None and await end_run_if_stale(db.bind, in_progress.run_id):
        in_progress = None
    if in_progress is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "error": {
                    "code": "ANALYSIS_IN_PROGRESS",
                    "message": f"An analysis is already running for {body.ticker}",
                    "details": {"ticker": body.ticker, "analysis_id": str(in_progress.run_id)},
                }
            },
        )

    run = AnalysisRun(
        run_id=uuid4(),
        user_id=body.user_id,
        stock_id=stock.stock_id,
        account_type=body.account_type,
        timeline=body.timeline,
        triggered_by="manual",
        status=RunStatus.QUEUED,
        llm_config={"model": MODEL},
    )
    # Real TOCTOU race, confirmed live (2026-09-23): the in_progress SELECT
    # above and this insert aren't serialized against a second concurrent
    # request doing the same check -- two genuinely concurrent requests for
    # the same brand-new ticker both passed that check and both landed a
    # row before this fix. analysis_runs's own partial unique index
    # (api/tables/analysis_runs.py) now makes a second concurrent insert
    # raise IntegrityError instead of silently succeeding; this turns that
    # into the same clean 409 the sequential-request path already returns
    # above, rather than an unhandled 500 for whoever lost the race.
    try:
        async with db.begin_nested():
            db.add(run)
            await db.flush()
    except IntegrityError:
        await db.rollback()
        # Deliberately does NOT re-query for the winning run's analysis_id
        # the way the sequential-request 409 above does -- found live
        # (2026-09-23) that issuing another db.execute() here, immediately
        # after this rollback, raised a second, unrelated MissingGreenlet
        # error under genuine concurrent connection contention (reproduced
        # via two real concurrent httpx requests against a real on-disk
        # SQLite file, not just the sequential-request case). A client that
        # hits this rare raced path gets a 409 with no analysis_id; they can
        # still find the winning run via GET .../status for the ticker.
        raise HTTPException(
            status_code=409,
            detail={
                "error": {
                    "code": "ANALYSIS_IN_PROGRESS",
                    "message": f"An analysis is already running for {body.ticker}",
                    "details": {"ticker": body.ticker},
                }
            },
        ) from None
    await db.commit()
    await db.refresh(run)

    background_tasks.add_task(_run_analysis_background, run.run_id)
    return AnalysisCreateResponse(analysis_id=run.run_id, status=RunStatus.QUEUED.value)


@router.get("/{analysis_id}/status", response_model=AnalysisStatusResponse)
async def get_analysis_status(analysis_id: UUID, db: AsyncSession = Depends(get_async_db)):
    run = (
        await db.execute(select(AnalysisRun).where(AnalysisRun.run_id == analysis_id))
    ).scalar_one_or_none()
    if run is None:
        raise _not_found("ANALYSIS_NOT_FOUND", f"No analysis found for id {analysis_id}", analysis_id=str(analysis_id))

    await _end_if_stale(db, run)
    external_status, pass_num = _external_status(run.status)
    progress = None
    if pass_num is not None:
        agents_complete = (
            await db.execute(
                select(func.count()).select_from(AgentOutput).where(
                    AgentOutput.run_id == run.run_id,
                    AgentOutput.agent_pass == f"pass{pass_num}",
                    AgentOutput.status == "completed",
                )
            )
        ).scalar_one()
        progress = AnalysisProgress(pass_=pass_num, agents_complete=agents_complete)

    return AnalysisStatusResponse(
        status=external_status, progress=progress, failure=_failure_from(run, external_status)
    )


@router.get("/{analysis_id}", response_model=AnalysisResultResponse)
async def get_analysis_result(analysis_id: UUID, db: AsyncSession = Depends(get_async_db)):
    run = (
        await db.execute(select(AnalysisRun).where(AnalysisRun.run_id == analysis_id))
    ).scalar_one_or_none()
    if run is None:
        raise _not_found("ANALYSIS_NOT_FOUND", f"No analysis found for id {analysis_id}", analysis_id=str(analysis_id))

    await _end_if_stale(db, run)
    external_status, _pass_num = _external_status(run.status)
    recommendation = (
        await db.execute(select(Recommendation).where(Recommendation.run_id == run.run_id))
    ).scalar_one_or_none()
    agent_outputs = (
        await db.execute(select(AgentOutput).where(AgentOutput.run_id == run.run_id))
    ).scalars().all()

    result = AnalysisResultResponse(
        analysis_id=run.run_id,
        status=external_status,
        disagreement_score=run.disagreement_score,
        disagreement_class=run.disagreement_class,
        agent_outputs=[AgentOutputSummary.model_validate(row) for row in agent_outputs],
        failure=_failure_from(run, external_status),
    )
    if recommendation is not None:
        result.stock_outlook_direction = recommendation.stock_outlook_direction
        result.overall_confidence = recommendation.overall_confidence
        result.account_recommendation = recommendation.account_recommendation
        result.expected_return_tier = recommendation.expected_return_tier
        result.key_drivers = recommendation.key_drivers
        result.synthesis_narrative = recommendation.synthesis_narrative
        result.bull_case_strength = recommendation.bull_case_strength
        result.bear_case_strength = recommendation.bear_case_strength
    return result
