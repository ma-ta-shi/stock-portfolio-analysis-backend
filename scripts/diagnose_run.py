"""Explains one run: what happened, why, and where the evidence is (86bc997wr).

    python scripts/diagnose_run.py <run_id>

Prints, in this order: the verdict (one line), the errors recorded for the run
(with the last lines of the traceback), any agent that failed or needed
attention, every model call that was rejected or retried (all attempts in
order, with the validator's own error strings) with the saved prompt/response
file paths and whether they exist, and the run's quality counters.

Everything comes from data the pipeline already stores; this script only reads.
Run it from backend/ (the database and var/runs paths are relative to it). The
run_id is printed by scripts/analyze.py, and `scripts/recent_errors.py` lists
recent failures with their run_ids.

Imports api.database directly, never api.main: api/main.py runs
Base.metadata.create_all against the real app.db at import time.
"""

import argparse
import asyncio
import sys
from uuid import UUID

import _bootstrap  # noqa: F401  (UTF-8 console, sys.path, table imports; must come first)
from api.database import AsyncSessionLocal
from services.error_report import diagnose_run, format_diagnosis


async def _main(run_id: UUID) -> int:
    async with AsyncSessionLocal() as session:
        diagnosis = await diagnose_run(session, run_id)
    if diagnosis is None:
        print(f"No analysis_runs row for run_id={run_id}")
        return 1
    print(format_diagnosis(diagnosis))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("run_id", type=UUID)
    args = parser.parse_args()
    sys.exit(asyncio.run(_main(args.run_id)))


if __name__ == "__main__":
    main()
