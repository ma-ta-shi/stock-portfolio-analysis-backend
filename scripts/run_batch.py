"""Runs several analyses one after another and reports everything that went wrong
(86bc997wr). For quality testing: run the batch, read the summary at the end.

    python scripts/run_batch.py KO AAPL SHOP.TO --account tfsa --timeline medium_term --user-id <uuid>
    python scripts/run_batch.py MSFT --repeat 3 ...          # the same ticker 3 times

Each analysis is a separate `scripts/analyze.py` process, one at a time (Ollama
does not run calls concurrently, so parallel runs would only slow each other).
A crash in one run cannot stop the batch. Each run prints its own RUN SUMMARY as
it finishes. At the end this prints one row per run (outcome, time, outlook), the
totals, and the error groups recorded during the batch (new versus recurring),
so nothing that went wrong is only visible if you go looking.

A run that outlives --run-timeout-minutes is killed and ended as failed. Exit
code: 0 if every run completed with no errors (retries that recovered are reported
but are not failures), else 1.

Imports api.database directly, never api.main (which runs create_all against the
real app.db at import time).
"""

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime
from uuid import UUID

import _bootstrap  # UTF-8 console, sys.path, table imports; must come first
from api.database import AsyncSessionLocal
from services.batch_summary import (
    NOT_STARTED,
    OK,
    RETRIES,
    TIMED_OUT,
    describe_run,
    format_batch_summary,
    parse_run_id,
)
from services.error_recorder import build_error_note, write_error_notes
from services.error_report import format_recent_errors, recent_errors
from services.run_liveness import end_run_now

_BACKEND_ROOT = _bootstrap.BACKEND_ROOT


async def _end_timed_out_run(bind, run_id: UUID, ticker: str, minutes: int) -> None:
    """The batch killed this run, so its own process never got to say why. End it
    as failed AND leave an error row (as the stale-run path does), or the report
    would file it under 'failed before the recorder existed'."""
    message = f"batch killed the run after {minutes} minutes"
    if await end_run_now(bind, run_id, stage="timeout", message=message):
        await write_error_notes(
            bind,
            [
                build_error_note(
                    "orchestrator",
                    "run_timed_out",
                    "high",
                    message,
                    run_id=run_id,
                    stock_ticker=ticker,
                    dedup_subtype="timeout",
                    context={"stage": "timeout", "timeout_minutes": minutes},
                )
            ],
        )


async def _run_one(cmd: list[str], timeout_s: float) -> tuple[UUID | None, bool, int | None]:
    """Run one analyze.py, echoing its output live. Returns (run_id, timed_out,
    exit code). run_id comes from the child's 'Starting analysis:' line."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(_BACKEND_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    run_id: UUID | None = None

    async def pump() -> None:
        nonlocal run_id
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            print(line, flush=True)
            if run_id is None:
                run_id = parse_run_id(line)

    try:
        await asyncio.wait_for(pump(), timeout=timeout_s)
        await proc.wait()
        return run_id, False, proc.returncode
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return run_id, True, None


async def _main(args: argparse.Namespace) -> int:
    started_at = datetime.now(UTC).replace(tzinfo=None)
    plan = [(t, i + 1) for i in range(args.repeat) for t in args.tickers]
    rows: list[dict] = []
    for n, (ticker, rep) in enumerate(plan, 1):
        label = f"{ticker} | {args.account} / {args.timeline}" + (
            f" | repeat {rep}" if args.repeat > 1 else ""
        )
        print(f"\n##### [{n}/{len(plan)}] {label} #####", flush=True)
        cmd = [
            sys.executable,
            str(_BACKEND_ROOT / "scripts" / "analyze.py"),
            ticker,
            "--account",
            args.account,
            "--timeline",
            args.timeline,
            "--user-id",
            str(args.user_id),
        ]
        run_id, timed_out, code = await _run_one(cmd, args.run_timeout_minutes * 60)
        async with AsyncSessionLocal() as session:
            if run_id is not None and timed_out:
                await _end_timed_out_run(session.bind, run_id, ticker, args.run_timeout_minutes)
            row = (
                await describe_run(session, run_id)
                if run_id is not None
                else {
                    "outcome": NOT_STARTED,
                    "headline": "analyze.py never started a run (see its output above)",
                }
            )
        if timed_out:
            row["outcome"] = TIMED_OUT
            row["headline"] = f"killed after {args.run_timeout_minutes} minutes"
        row["label"] = label
        rows.append(row)

    print("\n" + format_batch_summary(rows))
    async with AsyncSessionLocal() as session:
        report = await recent_errors(session, started_at, include_low=False)
    print("\n" + format_recent_errors(report))
    return 0 if rows and all(r["outcome"] in (OK, RETRIES) for r in rows) else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--account", required=True, choices=["tfsa", "rrsp", "trading"])
    parser.add_argument(
        "--timeline", required=True, choices=["short_term", "medium_term", "long_term"]
    )
    parser.add_argument("--user-id", required=True, type=UUID)
    parser.add_argument("--repeat", type=int, default=1, help="run the whole list this many times")
    parser.add_argument(
        "--run-timeout-minutes",
        type=int,
        default=60,
        help="kill and fail a single run after this long (default 60)",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    sys.exit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
