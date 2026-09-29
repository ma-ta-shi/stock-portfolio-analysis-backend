"""Launches a real analysis run for a ticker (86bc90p0a). CLAUDE.md documents
`python -m stock_picker.cli analyze SHOP.TO --account tfsa --timeline
medium_term`; confirmed no `stock_picker` package, `cli.py`, or build system
exists anywhere in this repo (no `[project.scripts]` in pyproject.toml either)
-- that command is aspirational, not real, same finding `run_trace.py` already
recorded for its own sibling `run-trace` command. This ships as its own
standalone script instead, for the same reason `run_trace.py` did.

Runs the exact same `AnalysisOrchestrator` the API uses -- real Ollama calls,
real DB writes, no mocking. Reuses `resolve_or_create_stock()` and the
`AnalysisRun` construction pattern from `api/routes/analysis.py` rather than
reimplementing them.

Uses AsyncSessionLocal, not the sync SessionLocal `api.database` also exposes
-- CLAUDE.md states "Always use AsyncSession -- never sync Session"
unconditionally. Imports `api.database` directly, never `api.main` -- that
module calls `Base.metadata.create_all(bind=engine)` as an unconditional
import-time side effect (a real, confirmed hazard: it would silently create a
table in the live app.db outside Alembic's tracking), which this script must
not trigger.

Calls `load_dotenv()` itself (found live: without it, a real run failed with
edgartools' "User-Agent identity is not set", since SP_EDGAR_IDENTITY is real
in .env but nothing loads .env outside api/main.py's own import-time call --
which this script deliberately never imports, per the paragraph above). This
script is a genuine second entry point that hits real provider APIs
(resolve_or_create_stock's Router call, DataPipeline.prepare() inside the
orchestrator), unlike run_trace.py, which only reads already-written DB rows
and never needed this.

Also reconfigures stdout/stderr to UTF-8, for the same reason and by the same
precedent as load_dotenv() above: api/main.py already does this (86bb7j0kh --
structlog logs non-cp1252 text on a plain Windows console, which otherwise
raises UnicodeEncodeError out of the very fallback-catching code path meant to
degrade gracefully), and this script is a second real entry point missing
that same baseline setup -- found live via a real MSFT run whose pipeline
failure was exactly this bug.

Usage: python scripts/analyze.py <ticker> --account {tfsa,rrsp,trading}
       --timeline {short_term,medium_term,long_term} --user-id <uuid>
"""

import argparse
import asyncio
import sys
from pathlib import Path
from uuid import UUID, uuid4

from dotenv import load_dotenv

# Same fix as api/main.py:30-31 (86bb7j0kh), same reason: must run before
# anything else logs.
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

_BACKEND_ROOT = Path(__file__).resolve().parents[1]

# Explicit path, not a bare load_dotenv() -- that searches from the current
# working directory, not the script's own location, so invoking this from
# anywhere other than backend/ would silently skip .env with no error
# (caught during review: sys.path.insert two lines below already anchors off
# __file__ for exactly this reason; load_dotenv() was inconsistent with it).
load_dotenv(_BACKEND_ROOT / ".env")

sys.path.insert(0, str(_BACKEND_ROOT / "src"))

from agents.base import MODEL  # noqa: E402
from api.crud.user_profile import get_user_profile  # noqa: E402
from api.database import AsyncSessionLocal  # noqa: E402

# Mapper-reachability imports -- same chain run_trace.py already documents:
# AnalysisRun -> AgentOutput/Recommendation/Stock, Recommendation ->
# Prediction -> PredictionCheckpoint. UserProfile has no relationship() to
# resolve (bare FK column on AnalysisRun.user_id), so it needs no further
# reachability imports of its own.
from api.tables.agent_outputs import AgentOutput  # noqa: E402, F401
from api.tables.analysis_runs import AnalysisRun, RunStatus  # noqa: E402
from api.tables.prediction_checkpoints import PredictionCheckpoint  # noqa: E402, F401
from api.tables.predictions import Prediction  # noqa: E402, F401
from api.tables.recommendations import Recommendation  # noqa: E402
from api.tables.stock import Stock  # noqa: E402, F401
from api.tables.user_profile import UserProfile  # noqa: E402, F401
from fastapi import HTTPException  # noqa: E402
from services.orchestrator import AnalysisOrchestrator  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.exc import IntegrityError  # noqa: E402

# Imported here, not duplicated -- see its own updated docstring for why it's
# no longer module-private to api/routes/analysis.py.
from api.routes.analysis import resolve_or_create_stock  # noqa: E402


def _print_in_progress(ticker: str, run_id: UUID | None = None, status: str | None = None) -> None:
    if run_id is None:
        # Matches analysis.py's own create_analysis() race branch, which
        # deliberately omits analysis_id here too -- its own comment explains
        # why re-querying immediately after this rollback is unsafe
        # (a second, unrelated MissingGreenlet error, found live). The
        # caller's own run_id is meaningless to print: that insert never
        # landed, so it isn't a real, lookupable id.
        print(
            f"An analysis is already running for {ticker} (race detected) -- check with GET .../status or query analysis_runs directly"
        )
    else:
        print(f"An analysis is already running for {ticker}: run_id={run_id} status={status}")


async def _analyze(ticker: str, account_type: str, timeline: str, user_id: UUID) -> int:
    async with AsyncSessionLocal() as db:
        profile = await get_user_profile(db, user_id)
        if profile is None:
            print(f"No user profile found for user_id={user_id} -- create one first")
            return 1

        try:
            stock = await resolve_or_create_stock(ticker, db)
        except HTTPException as e:
            print(f"Error: {e.detail['error']['message']}")
            return 1

        in_progress = (
            (
                await db.execute(
                    select(AnalysisRun).where(
                        AnalysisRun.stock_id == stock.stock_id,
                        AnalysisRun.status.notin_([RunStatus.COMPLETED, RunStatus.FAILED]),
                    )
                )
            )
            .scalars()
            .first()
        )
        if in_progress is not None:
            _print_in_progress(ticker, in_progress.run_id, in_progress.status)
            return 1

        run = AnalysisRun(
            run_id=uuid4(),
            user_id=user_id,
            stock_id=stock.stock_id,
            account_type=account_type,
            timeline=timeline,
            triggered_by="cli",
            status=RunStatus.QUEUED,
            llm_config={"model": MODEL},
        )
        try:
            async with db.begin_nested():
                db.add(run)
                await db.flush()
        except IntegrityError:
            await db.rollback()
            # Same rare TOCTOU race analysis.py's own create_analysis()
            # guards against -- vanishingly unlikely for one interactive CLI
            # invocation, but a scripted caller could still hit it.
            _print_in_progress(ticker)
            return 1
        await db.commit()
        await db.refresh(run)

        print(
            f"Starting analysis: {ticker} | account={account_type} | timeline={timeline} | run_id={run.run_id}"
        )

        try:
            await AnalysisOrchestrator().run(run, db)
        except Exception as e:
            # orchestrator.run() already sets run.status=FAILED and commits
            # before re-raising on this path (a Gate 1/Gate 2 failure instead
            # returns normally with status already FAILED -- handled below,
            # not here). This catch exists only so the CLI prints something
            # useful instead of a raw traceback.
            print(f"Pipeline failed: {e} (run_id={run.run_id}, status={run.status})")
            return 1

        recommendation = (
            await db.execute(select(Recommendation).where(Recommendation.run_id == run.run_id))
        ).scalar_one_or_none()

        print(f"\n=== {ticker} | run_id={run.run_id} ===")
        print(f"status: {run.status}")
        print(f"disagreement: {run.disagreement_score} ({run.disagreement_class})")
        if recommendation is None:
            # Expected, not an error, whenever status == "failed" -- a Gate
            # 1/Gate 2 failure means the CIO never ran, so no Recommendation
            # was ever created (RunStatus's own docstring).
            print("no recommendation -- pipeline failed before synthesis")
        else:
            print(
                f"outlook: {recommendation.stock_outlook_direction} | confidence: {recommendation.overall_confidence}"
            )
            print(f"narrative: {recommendation.synthesis_narrative}")
        print(f"\nFull trace: python scripts/run_trace.py {run.run_id}")
        return 0 if run.status == RunStatus.COMPLETED else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker")
    parser.add_argument("--account", required=True, choices=["tfsa", "rrsp", "trading"])
    parser.add_argument(
        "--timeline", required=True, choices=["short_term", "medium_term", "long_term"]
    )
    parser.add_argument("--user-id", required=True, type=UUID)
    args = parser.parse_args()
    sys.exit(asyncio.run(_analyze(args.ticker, args.account, args.timeline, args.user_id)))


if __name__ == "__main__":
    main()
