"""Prints one run's quality summary plus its full llm_calls trace (86bbwachy
Phase 5's own §5.8 "the surface" ask -- a real CLI entry point, not a
UI). CLAUDE.md documents `python -m stock_picker.cli run-trace {run_id}`;
confirmed directly that no `stock_picker` package, `cli.py`, or build
system exists anywhere in this repo (no `[project.scripts]` in
pyproject.toml either) -- that command is aspirational, not real. This
ships as its own standalone script instead, matching the one existing
precedent for a script in this repo (split_prompt_docs.py) rather than
inventing an installable-package layout for one read-only reporting
command.

Uses AsyncSessionLocal, not the sync SessionLocal that api.database also
exposes -- CLAUDE.md states "Always use AsyncSession -- never sync
Session" unconditionally, not as a rule this script's own convenience
gets to relax. Imports api.database directly, never api.main -- api/main.py
calls Base.metadata.create_all(bind=engine) as an unconditional
import-time side effect (a real, confirmed hazard: importing api.main
during this ticket's own implementation silently created a table in the
live app.db outside Alembic's tracking), which a read-only reporting
script run against the real app.db must not trigger.

Usage: python scripts/run_trace.py <run_id>
"""

import argparse
import asyncio
import sys
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.database import AsyncSessionLocal  # noqa: E402
# Mapper-reachability imports (AgentOutput/Prediction/PredictionCheckpoint) --
# the exact, verified relationship() graph reachable from AnalysisRun/
# RunQualitySummary, not the full 9-table set main.py/alembic/conftest.py
# import: AnalysisRun -> AgentOutput/Recommendation/RunQualitySummary/Stock,
# Recommendation -> Prediction, Prediction -> PredictionCheckpoint. Confirmed
# by grepping every relationship() in that whole chain -- ShadowPrediction is
# never referenced by any of them (AnalysisRun has no relationship to it),
# so it isn't needed here despite appearing in every other file's own
# reachability block.
from api.tables.agent_outputs import AgentOutput  # noqa: E402, F401
from api.tables.analysis_runs import AnalysisRun  # noqa: E402
from api.tables.llm_calls import LLMCall  # noqa: E402
from api.tables.prediction_checkpoints import PredictionCheckpoint  # noqa: E402, F401
from api.tables.predictions import Prediction  # noqa: E402, F401
from api.tables.recommendations import Recommendation  # noqa: E402, F401
from api.tables.run_quality_summary import RunQualitySummary  # noqa: E402
from api.tables.stock import Stock  # noqa: E402
from sqlalchemy import select  # noqa: E402


def _fmt(v, na="N/A"):
    # Plain ASCII, not an em dash -- a Windows console without UTF-8 output
    # configured renders non-ASCII as mojibake (confirmed live: tests/conftest.py's
    # own sys.stdout.reconfigure() exists for exactly this class of problem,
    # but that fix is test-session-only, not something this standalone
    # script can rely on for whatever console it's actually run from).
    return na if v is None else v


async def _run_trace(run_id: UUID) -> int:
    async with AsyncSessionLocal() as session:
        run = (
            await session.execute(select(AnalysisRun).where(AnalysisRun.run_id == run_id))
        ).scalar_one_or_none()
        if run is None:
            print(f"No analysis_runs row for run_id={run_id}")
            return 1

        stock = (
            await session.execute(select(Stock).where(Stock.stock_id == run.stock_id))
        ).scalar_one_or_none()
        ticker = stock.canonical_ticker if stock else "?"

        print(f"=== {ticker} | {run.account_type} / {run.timeline} | run_id={run_id} ===")
        print(f"status: {run.status} | triggered: {run.triggered_at} | completed: {_fmt(run.completed_at)}")

        summary = (
            await session.execute(select(RunQualitySummary).where(RunQualitySummary.run_id == run_id))
        ).scalar_one_or_none()
        if summary is None:
            print("\n(no run_quality_summary row yet -- the run may still be in progress)")
        else:
            print("\n--- quality summary ---")
            print(f"wall clock: {summary.wall_clock_ms}ms | total LLM time: {summary.total_llm_ms}ms "
                  f"| slowest call: {_fmt(summary.slowest_call_ms)}ms")
            print(f"calls: {summary.total_calls} ({summary.retry_calls} retries) | "
                  f"tokens: {summary.total_prompt_tokens} prompt / {summary.total_completion_tokens} completion "
                  f"| thinking chars: {summary.total_thinking_chars}")
            print(f"truncated: {summary.truncated_calls} | empty_content: {summary.empty_content_calls} "
                  f"| validator failures: {summary.validator_failures}")
            print(f"gate1: {_fmt(summary.gate1_passed)} ({_fmt(summary.gate1_reason, '')}) | "
                  f"gate2: {_fmt(summary.gate2_passed)} ({_fmt(summary.gate2_reason, '')})")
            print(f"outlook: {_fmt(summary.stock_outlook)} | confidence: {_fmt(summary.overall_confidence)}")
            if summary.agents_with_empty_key_factors:
                print(f"empty key_factors: {summary.agents_with_empty_key_factors}")
            if summary.agents_with_empty_risks:
                print(f"empty risks: {summary.agents_with_empty_risks}")
            if summary.agents_with_empty_narrative:
                print(f"empty narrative: {summary.agents_with_empty_narrative}")
            if summary.human_quality_rating or summary.human_quality_note:
                print(f"human rating: {_fmt(summary.human_quality_rating)} -- {_fmt(summary.human_quality_note, '')}")

        calls = (
            await session.execute(
                select(LLMCall).where(LLMCall.run_id == run_id).order_by(LLMCall.seq)
            )
        ).scalars().all()
        print(f"\n--- llm_calls ({len(calls)}) ---")
        for c in calls:
            print(
                f"[{_fmt(c.seq)}] {c.call_site} attempt={c.attempt} finish_reason={_fmt(c.finish_reason)} "
                f"validator_passed={_fmt(c.validator_passed)}"
            )
            if c.prompt_path:
                print(f"      prompt:   {c.prompt_path}")
            if c.response_path:
                print(f"      response: {c.response_path}")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", type=UUID)
    args = parser.parse_args()
    sys.exit(asyncio.run(_run_trace(args.run_id)))


if __name__ == "__main__":
    main()
