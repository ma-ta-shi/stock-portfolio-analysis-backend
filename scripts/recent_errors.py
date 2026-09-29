"""What has gone wrong lately, across runs (86bc997wr).

    python scripts/recent_errors.py                 # the last 7 days
    python scripts/recent_errors.py --since 24h     # 30m, 24h, 7d, 2w, or 2026-09-28
    python scripts/recent_errors.py --all           # include low-severity events
    python scripts/recent_errors.py --fail-stale    # end runs whose process died

Prints a health verdict, then the failures grouped by fingerprint (how many
times, in how many runs, first and last seen, which tickers, and whether this
is the first time it has EVER happened), failed runs that predate the recorder,
runs that never finished, and the validator rules that reject the models'
answers most often.

"Did I fix it?": run it with --since set to when you shipped the fix and look
at the group's last-seen time.

--fail-stale ends every run that has shown no activity for SP_STALE_RUN_MINUTES
(default 90: twice the ~45 minutes a run could be silent if every call hung to
its timeout; real runs take 5 to 10). It never touches a run that is still
working. Run from backend/ (paths are relative to it).

Imports api.database directly, never api.main: api/main.py runs
Base.metadata.create_all against the real app.db at import time.
"""

import argparse
import asyncio
import sys

import _bootstrap  # noqa: F401  (UTF-8 console, sys.path, table imports; must come first)
from api.database import AsyncSessionLocal
from services.error_report import (
    format_recent_errors,
    parse_since,
    recent_errors,
)
from services.run_liveness import end_stale_runs


async def _main(since_text: str, include_low: bool, fail_stale: bool) -> int:
    try:
        since = parse_since(since_text)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 2
    async with AsyncSessionLocal() as session:
        if fail_stale:
            ended = await end_stale_runs(session.bind)
            if ended:
                for entry in ended:
                    print(
                        f"Ended stale run: {entry['ticker']} {entry['status']} "
                        f"run_id={entry['run_id']} ({entry['reason']})"
                    )
            else:
                print("No stale runs to end.")
            print()
        report = await recent_errors(session, since, include_low=include_low)
    print(format_recent_errors(report))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--since", default="7d", help="window: 30m, 24h, 7d, 2w, or a date (default 7d)"
    )
    parser.add_argument("--all", action="store_true", help="include low-severity events")
    parser.add_argument("--fail-stale", action="store_true", help="end runs that have gone quiet")
    args = parser.parse_args()
    sys.exit(asyncio.run(_main(args.since, args.all, args.fail_stale)))


if __name__ == "__main__":
    main()
