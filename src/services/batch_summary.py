"""The end-of-batch report for scripts/run_batch.py (86bc997wr).

A quality-test batch is many runs; the question afterwards is not "did the last
one finish" but "across all of them, what went wrong, and where". This turns
each run's diagnosis into one row, totals the outcomes, and leaves the
per-fingerprint detail to recent_errors (the script prints that too, windowed
to the batch).

The logic lives here rather than in the script so it can be tested without
running a model.
"""

import re
from uuid import UUID

from sqlalchemy import select

from api.tables.recommendations import Recommendation
from services.error_report import diagnose_run

_RUN_ID_LINE = re.compile(r"Starting analysis:.*\brun_id=([0-9a-fA-F-]{36})")

# Outcomes, in the order the totals print. The first four come from diagnose_run's
# verdict; the rest are things only the batch can know (the run never began, or
# the batch had to kill it).
OK = "OK"
RETRIES = "COMPLETED WITH RETRIES"
WITH_PROBLEMS = "COMPLETED WITH PROBLEMS"
FAILED = "FAILED"
STALE = "STALE"
NOT_STARTED = "DID NOT START"
TIMED_OUT = "TIMED OUT"
_ORDER = [OK, RETRIES, WITH_PROBLEMS, FAILED, STALE, TIMED_OUT, NOT_STARTED]


def parse_run_id(line: str) -> UUID | None:
    """The run_id analyze.py prints on its 'Starting analysis:' line, else None."""
    match = _RUN_ID_LINE.search(line)
    if match is None:
        return None
    try:
        return UUID(match.group(1))
    except ValueError:
        return None


async def describe_run(session, run_id: UUID) -> dict:
    """One row of the batch table for a run that exists: outcome, the verdict
    line, the problem counts and (when the CIO finished) the outlook."""
    d = await diagnose_run(session, run_id)
    if d is None:
        return {"run_id": run_id, "outcome": NOT_STARTED, "headline": "no run row found"}
    recommendation = (
        await session.execute(select(Recommendation).where(Recommendation.run_id == run_id))
    ).scalar_one_or_none()
    run = d["run"]
    seconds = None
    if run["completed_at"] is not None and run["triggered_at"] is not None:
        seconds = (run["completed_at"] - run["triggered_at"]).total_seconds()
    quality = d["quality"] or {}
    return {
        "run_id": run_id,
        "ticker": run["ticker"],
        "outcome": d["verdict"]["outcome"],
        "headline": d["verdict"]["headline"],
        "prompts": run["prompts_hash"],
        "code": run["code_hash"],
        "errors": sum(1 for e in d["errors"] if e["severity"] != "low"),
        "retries": quality.get("retry_calls"),
        "seconds": seconds,
        "outlook": None if recommendation is None else recommendation.stock_outlook_direction,
        "confidence": None if recommendation is None else recommendation.overall_confidence,
    }


def _state_line(label: str, noun: str, key: str, rows: list[dict]) -> str | None:
    """One line saying whether every run in the batch used the same `key` state (a prompt
    hash or a code hash), or that the batch mixed states, or that none were recorded."""
    states = sorted({r[key] for r in rows if r.get(key)})
    unrecorded = sum(1 for r in rows if r.get("run_id") and not r.get(key))
    if len(states) == 1 and not unrecorded:
        return f"{label}: every run used {noun} state {states[0]}"
    if not states and unrecorded:
        return f"{label}: not recorded for these runs (they predate it)"
    if states:
        mixed = ", ".join(states) + (f" and {unrecorded} with none recorded" if unrecorded else "")
        return f"{label}: this batch mixes {noun} states ({mixed}): compare runs only within one state"
    return None


def format_batch_summary(rows: list[dict]) -> str:
    """rows: one dict per attempted run with `label` (ticker/account/timeline text),
    `outcome`, `headline`, and optionally run_id, errors, retries, seconds, outlook,
    confidence. Returns the table plus the totals."""
    out = [f"=== BATCH SUMMARY: {len(rows)} run(s) ==="]
    for i, r in enumerate(rows, 1):
        seconds = r.get("seconds")
        took = "" if seconds is None else f" {seconds / 60:.1f}m"
        outlook = ""
        if r.get("outlook"):
            outlook = f" | {r['outlook']} {r.get('confidence')}"
        rid = f" | {str(r['run_id'])[:8]}" if r.get("run_id") else ""
        out.append(f"{i:>2}. {r['label']}{rid} | {r['outcome']}{took}{outlook}")
        if r["outcome"] != OK:
            out.append(f"      {r['headline']}")
    counts = {name: sum(1 for r in rows if r["outcome"] == name) for name in _ORDER}
    other = len(rows) - sum(counts.values())
    parts = [f"{n} {name.lower()}" for name, n in counts.items() if n]
    if other:
        parts.append(f"{other} other")
    out.append("")
    out.append("TOTALS: " + (", ".join(parts) if parts else "no runs"))
    # Which prompt text and which code each run used. Both are edited in place, so a batch
    # that spans an edit mixes two states and must not be compared as one.
    for label, noun, key in (("PROMPTS", "prompt", "prompts"), ("CODE", "code", "code")):
        line = _state_line(label, noun, key, rows)
        if line:
            out.append(line)
    if not rows:
        verdict = "VERDICT: ATTENTION - nothing ran"
    elif counts[OK] == len(rows):
        verdict = "VERDICT: every run completed with no recorded problems"
    elif counts[OK] + counts[RETRIES] == len(rows):
        verdict = (
            "VERDICT: no errors, but "
            f"{counts[RETRIES]} run(s) needed retries. Not failures, but investigate: the "
            "rules that rejected answers are listed below (prompt and validator fixes)"
        )
    else:
        verdict = "VERDICT: ATTENTION - see the runs above and the error groups below"
    out.append(verdict)
    return "\n".join(out)
