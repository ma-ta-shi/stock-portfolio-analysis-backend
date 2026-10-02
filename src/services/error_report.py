"""The read side of error tracking (86bc997wr): answers the questions a developer
actually asks when something goes wrong.

    Is the system healthy right now?            recent_errors(...)  header
    What failed in the last batch?              recent_errors(...)  groups
    Is this failure new or recurring?           recent_errors(...)  count / first / last / "new"
    Why did agent X fail on ticker Y?           diagnose_run(...)
    Did I fix it?                               recent_errors(since=<the fix>)

Everything here READS existing data (analysis_runs, error_records,
agent_outputs, llm_calls, run_quality_summary); the only writer is
end_stale_runs(), which lives in run_liveness. The functions take a session and
an injectable clock, so they are unit-tested against a seeded in-memory
database, and the scripts (scripts/diagnose_run.py, scripts/recent_errors.py)
only parse arguments and print what the format_* functions return.

Text output is plain ASCII: it is read on a Windows console.
"""

import os
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID

from sqlalchemy import func, select

from api.tables.agent_outputs import AgentOutput
from api.tables.analysis_runs import AnalysisRun
from api.tables.error_records import ErrorRecord
from api.tables.llm_calls import LLMCall
from api.tables.run_quality_summary import RunQualitySummary
from api.tables.stock import Stock
from services.error_recorder import rule_slug, safe_text
from services.run_liveness import (
    TERMINAL_STATUSES,
    assess,
    list_non_terminal_runs,
    load_snapshot,
)
from services.run_liveness import _naive_utc, _now  # one definition of "now" and of UTC-naive

_SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}
_NORMAL_FINISH = (None, "stop")


def parse_since(text: str, now: datetime | None = None) -> datetime:
    """'30m', '24h', '7d', '2w', an ISO date ('2026-09-28') or datetime
    ('2026-09-28T10:30'), as a naive UTC datetime. Raises ValueError otherwise."""
    current = _naive_utc(now) if now is not None else _now()
    text = text.strip()
    match = re.fullmatch(r"(\d+)\s*([mhdw])", text, flags=re.IGNORECASE)
    if match:
        unit = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}[match.group(2).lower()]
        return current - timedelta(**{unit: int(match.group(1))})
    try:
        return _naive_utc(datetime.fromisoformat(text))
    except ValueError:
        raise ValueError(
            f"cannot read {text!r} as a time: use 30m, 24h, 7d, 2w, or a date like 2026-09-28"
        ) from None


def runs_dir() -> Path:
    """Where agents/capture.py saves raw prompts and responses (SP_RUNS_DIR)."""
    return Path(os.environ.get("SP_RUNS_DIR", "var/runs"))


def resolve_evidence(relative_path: str | None) -> dict | None:
    """A saved prompt/response path, as stored in llm_calls, resolved against the
    runs directory, with whether the file is actually there."""
    if not relative_path:
        return None
    resolved = runs_dir() / str(relative_path).replace("\\", "/")
    return {"path": str(resolved), "exists": resolved.exists()}


def _traceback_tail(stack_trace: str | None, lines: int = 6) -> list[str]:
    if not stack_trace:
        return []
    # Python 3.11+ underlines the failing expression with a line of ^^^^ / ~~~~;
    # useful in a terminal next to its source line, noise here.
    kept = [
        line.rstrip()
        for line in stack_trace.splitlines()
        if line.strip() and not re.fullmatch(r"[\s^~]+", line)
    ]
    return kept[-lines:]


def _ts(value: datetime | None) -> str:
    """A timestamp to the second (the stored values carry microseconds)."""
    return "N/A" if value is None else value.strftime("%Y-%m-%d %H:%M:%S")


async def resolve_run_id(session, text: str) -> UUID:
    """A run id as the scripts accept it: the full id (with or without dashes) or a
    UNIQUE prefix of at least 4 hex characters, which is what the batch table prints
    (8 characters). Raises ValueError, with a message that says what to do, for
    anything else. Prefixes are matched in Python over the newest 5000 runs, so it is
    portable across SQLite (hex ids) and PostgreSQL (uuid type)."""
    cleaned = text.strip().lower().replace("-", "")
    if not re.fullmatch(r"[0-9a-f]{4,32}", cleaned):
        raise ValueError(
            f"{text!r} is not a run id: give the full id, or at least 4 hex characters of it"
        )
    if len(cleaned) == 32:
        return UUID(cleaned)
    ids = (
        (
            await session.execute(
                select(AnalysisRun.run_id).order_by(AnalysisRun.triggered_at.desc()).limit(5000)
            )
        )
        .scalars()
        .all()
    )
    matches = [run_id for run_id in ids if run_id.hex.startswith(cleaned)]
    if not matches:
        raise ValueError(f"no recent run has an id starting with {cleaned!r}")
    if len(matches) > 1:
        options = ", ".join(str(m)[:12] for m in matches[:5])
        raise ValueError(f"{cleaned!r} matches {len(matches)} runs ({options}...): use more characters")
    return matches[0]


# --- diagnose one run -------------------------------------------------------------


async def diagnose_run(session, run_id, *, now: datetime | None = None) -> dict | None:
    """Everything known about why one run went the way it did, in one place.
    None if there is no such run."""
    row = (
        await session.execute(
            select(AnalysisRun, Stock.canonical_ticker)
            .join(Stock, Stock.stock_id == AnalysisRun.stock_id, isouter=True)
            .where(AnalysisRun.run_id == run_id)
        )
    ).one_or_none()
    if row is None:
        return None
    run, ticker = row

    error_rows = (
        (
            await session.execute(
                select(ErrorRecord)
                .where(ErrorRecord.run_id == run_id)
                .order_by(ErrorRecord.timestamp)
            )
        )
        .scalars()
        .all()
    )
    agent_rows = (
        (
            await session.execute(
                select(AgentOutput)
                .where(AgentOutput.run_id == run_id)
                .order_by(AgentOutput.created_at)
            )
        )
        .scalars()
        .all()
    )
    call_rows = (
        (
            await session.execute(
                select(LLMCall).where(LLMCall.run_id == run_id).order_by(LLMCall.seq)
            )
        )
        .scalars()
        .all()
    )
    summary = (
        await session.execute(select(RunQualitySummary).where(RunQualitySummary.run_id == run_id))
    ).scalar_one_or_none()

    # Calls: an agent that had ANY trouble gets every attempt listed in order, so
    # "attempt 1 rejected, attempt 2 accepted" is visible; other call sites (the
    # hundreds of precompute sentiment calls) are only counted.
    by_site: dict[str, list[LLMCall]] = {}
    for call in call_rows:
        by_site.setdefault(call.call_site, []).append(call)
    failing_calls: dict[str, list[dict]] = {}
    other_failing: list[dict] = []
    for site, calls in by_site.items():
        troubled = [c for c in calls if _call_had_trouble(c)]
        if not troubled:
            continue
        if site.startswith("agent:"):
            failing_calls[site] = [_call_dict(c) for c in calls]
        else:
            other_failing.append({"call_site": site, "failed": len(troubled), "total": len(calls)})

    liveness = None
    if run.status not in TERMINAL_STATUSES:
        snapshot = await load_snapshot(session, run_id)
        if snapshot is not None:
            verdict_now = assess(
                snapshot.status, snapshot.triggered_at, snapshot.last_activity_at, now
            )
            liveness = {
                "stale": verdict_now.stale,
                "reason": verdict_now.reason,
                "idle_minutes": verdict_now.idle_minutes,
                "last_activity_at": snapshot.last_activity_at,
            }

    errors = [
        {
            "timestamp": e.timestamp,
            "severity": e.severity,
            "component": e.component,
            "error_type": e.error_type,
            "agent_name": e.agent_name,
            "fingerprint": e.dedup_subtype,
            "count": e.occurrence_count or 1,
            "message": e.message,
            "traceback_tail": _traceback_tail(e.stack_trace),
            "context": e.context_json,
        }
        for e in error_rows
    ]
    agents = [
        {
            "agent_name": a.agent_name,
            "agent_pass": a.agent_pass,
            "status": a.status,
            # agent_outputs.error_detail is stored as the agent reported it;
            # redact on the way out too, so a report never prints a credential.
            "error_detail": None if a.error_detail is None else safe_text(a.error_detail, 4000),
            "analysis_confidence": a.analysis_confidence,
            "data_quality_assessment": a.data_quality_assessment,
        }
        for a in agent_rows
    ]
    return {
        "run": {
            "run_id": run.run_id,
            "ticker": ticker,
            "status": run.status,
            "account_type": run.account_type,
            "timeline": run.timeline,
            "triggered_at": run.triggered_at,
            "completed_at": run.completed_at,
            "error_log": run.error_log or [],
            # Which prompt text this run used (ledger BB-045); None for runs made
            # before it was recorded.
            "prompts_hash": (run.llm_config or {}).get("prompts_hash"),
            "code_hash": (run.llm_config or {}).get("code_hash"),
        },
        "verdict": _verdict(run, errors, agents, failing_calls, liveness),
        "errors": errors,
        "agents": agents,
        "failing_calls": failing_calls,
        "other_failing_calls": other_failing,
        "quality": None
        if summary is None
        else {
            "wall_clock_ms": summary.wall_clock_ms,
            "total_calls": summary.total_calls,
            "retry_calls": summary.retry_calls,
            "truncated_calls": summary.truncated_calls,
            "empty_content_calls": summary.empty_content_calls,
            "validator_failures": summary.validator_failures,
            "gate1": (summary.gate1_passed, summary.gate1_reason),
            "gate2": (summary.gate2_passed, summary.gate2_reason),
        },
        "liveness": liveness,
    }


def _call_had_trouble(call: LLMCall) -> bool:
    return (
        call.validator_passed is False
        or call.parsed_ok is False
        or call.finish_reason not in _NORMAL_FINISH
    )


def _call_dict(call: LLMCall) -> dict:
    return {
        "seq": call.seq,
        "attempt": call.attempt,
        "finish_reason": call.finish_reason,
        "parsed_ok": call.parsed_ok,
        "validator_passed": call.validator_passed,
        # Stored unredacted upstream (llm_calls); redacted here on the way out.
        "validator_errors": [safe_text(e, 1000) for e in (call.validator_errors or [])],
        "parse_error": None if call.parse_error is None else safe_text(call.parse_error, 1000),
        "latency_ms": call.latency_ms,
        "prompt_tokens": call.prompt_tokens,
        "prompt": resolve_evidence(call.prompt_path),
        "response": resolve_evidence(call.response_path),
    }


def _last_attempt_failed(calls: list[dict]) -> bool:
    """True when the agent's FINAL attempt still failed validation or parsing."""
    last = calls[-1]
    return last["validator_passed"] is False or last["parsed_ok"] is False


def _retried_agents(agents: list[dict], failing_calls: dict) -> tuple[list[str], list[str]]:
    """(recovered, exhausted): agents whose calls were rejected or malformed at least
    once and then either PASSED on a later attempt (recovered), or failed on every
    attempt and had their last output used anyway (exhausted; see agent_completed():
    validation is a warning, not a gate, settled 2026-09-02). A failed agent is
    excluded from both (it already has its own error row).
    agent_outputs names are upper case (FUND); call sites are lower case (agent:fund)."""
    failed = {a["agent_name"].lower() for a in agents if a["status"] == "failed"}
    recovered: list[str] = []
    exhausted: list[str] = []
    for site in sorted(failing_calls):
        name = site.removeprefix("agent:")
        if name.lower() in failed:
            continue
        (exhausted if _last_attempt_failed(failing_calls[site]) else recovered).append(name)
    return recovered, exhausted


def _verdict(run, errors, agents, failing_calls, liveness) -> dict:
    """One line that says what happened, first."""
    if run.status == "failed":
        entry = (run.error_log or [{}])[0]
        stage = entry.get("stage") or (errors[0]["component"] if errors else "unknown")
        message = entry.get("error") or (errors[0]["message"] if errors else "no reason recorded")
        message = " ".join(
            str(message).split()
        )  # the headline is one line (some errors span several)
        return {"outcome": "FAILED", "headline": f"FAILED at {stage}: {message}"}
    if run.status == "completed":
        failed_agents = [a["agent_name"] for a in agents if a["status"] == "failed"]
        retried, exhausted = _retried_agents(agents, failing_calls)
        # Low-severity events (a provider that declined and a fallback covered it, a
        # link that failed and the chain moved on) are hidden everywhere else, so
        # they must not make the verdict read "problems" either; they are counted.
        low = sum(1 for e in errors if e["severity"] == "low")
        errors = [e for e in errors if e["severity"] != "low"]
        low_note = f" ({low} low-severity event(s), see diagnose_run)" if low else ""
        if failed_agents or errors or exhausted:
            bits = []
            if failed_agents:
                bits.append(f"{len(failed_agents)} agent(s) failed ({', '.join(failed_agents)})")
            if errors:
                bits.append(f"{len(errors)} error record(s)")
            if exhausted:
                # Not a recovered retry: every attempt failed and the last output was used.
                bits.append(
                    f"{len(exhausted)} agent(s) used output that failed validation on every "
                    f"attempt ({', '.join(exhausted)})"
                )
            if retried:
                bits.append(f"{len(retried)} other agent(s) needed retries")
            return {
                "outcome": "COMPLETED WITH PROBLEMS",
                "headline": "COMPLETED, but " + "; ".join(bits),
            }
        if retried:
            # Not a failure (every one of these passed on a later attempt), but the main
            # source of prompt and validator fixes: named, never folded into OK.
            return {
                "outcome": "COMPLETED WITH RETRIES",
                "headline": f"COMPLETED, {len(retried)} agent(s) needed retries and recovered: "
                + ", ".join(retried)
                + low_note,
            }
        return {"outcome": "OK", "headline": "COMPLETED with no recorded problems" + low_note}
    if liveness and liveness["stale"]:
        return {
            "outcome": "STALE",
            "headline": f"STALE ({run.status}): {liveness['reason']}. "
            "Run recent_errors.py --fail-stale to end it.",
        }
    return {"outcome": "RUNNING", "headline": f"RUNNING ({run.status})"}


def _fmt(value, na="N/A"):
    return na if value is None else value


def format_diagnosis(d: dict) -> str:
    run = d["run"]
    out = [
        f"=== {_fmt(run['ticker'], '?')} | {run['account_type']} / {run['timeline']} | "
        f"run_id={run['run_id']} ===",
        f"RESULT: {d['verdict']['headline']}",
        f"status: {run['status']} | triggered: {_ts(run['triggered_at'])} | "
        f"completed: {_ts(run['completed_at'])}",
        f"prompts: {_fmt(run.get('prompts_hash'), 'not recorded')} | "
        f"code: {_fmt(run.get('code_hash'), 'not recorded')}",
    ]
    if d["liveness"]:
        lv = d["liveness"]
        out.append(
            f"last activity: {_ts(lv['last_activity_at'])} "
            f"({lv['idle_minutes']:.0f} min ago) - {lv['reason']}"
        )

    out.append(f"\n--- errors recorded ({len(d['errors'])}) ---")
    if not d["errors"] and run["error_log"]:
        out.append("(no error_records rows; this run's error_log:)")
        for entry in run["error_log"]:
            out.append(f"  [{_fmt(entry.get('stage'))}] {_fmt(entry.get('error'))}")
    elif not d["errors"]:
        out.append("(none)")
    for e in d["errors"]:
        who = f"{e['component']}/{e['error_type']}" + (
            f" agent={e['agent_name']}" if e["agent_name"] else ""
        )
        out.append(f"[{e['severity']}] {who}  ({_ts(e['timestamp'])})")
        out.append(f"  fingerprint: {_fmt(e['fingerprint'])}")
        out.append(f"  {e['message']}")
        if e["traceback_tail"]:
            out.append("  traceback (last lines):")
            out.extend(f"    {line}" for line in e["traceback_tail"])
        if e["context"]:
            shown = {k: v for k, v in e["context"].items() if k not in ("calls", "errors")}
            out.append(f"  context: {shown}")

    troubled_agents = [a for a in d["agents"] if a["status"] != "completed" or a["error_detail"]]
    out.append(f"\n--- agents needing attention ({len(troubled_agents)} of {len(d['agents'])}) ---")
    if not troubled_agents:
        out.append("(none)")
    # Agents that failed for the same reason (Ollama down takes out all five
    # Pass 1 agents at once) are one line, not five copies of the same message.
    by_reason: dict[tuple, list[str]] = {}
    for a in troubled_agents:
        by_reason.setdefault((a["status"], a["error_detail"]), []).append(
            f"{a['agent_name']} ({a['agent_pass']})"
        )
    for (status, detail), names in by_reason.items():
        out.append(f"{', '.join(names)} {status}")
        if detail:
            out.append(f"  {detail}")

    out.append(f"\n--- agent calls with trouble ({len(d['failing_calls'])}) ---")
    if not d["failing_calls"]:
        out.append("(none)")
    for site, attempts in d["failing_calls"].items():
        out.append(site)
        for c in attempts:
            verdict = {False: "REJECTED", True: "accepted", None: "n/a"}[c["validator_passed"]]
            out.append(
                f"  attempt {_fmt(c['attempt'])} (seq {_fmt(c['seq'])}) finish={_fmt(c['finish_reason'])} "
                f"parsed={_fmt(c['parsed_ok'])} validator={verdict}"
            )
            for err in c["validator_errors"]:
                out.append(f"      - {err}")
            if c["parse_error"]:
                out.append(f"      parse error: {c['parse_error']}")
            for label in ("prompt", "response"):
                ev = c[label]
                if ev:
                    out.append(
                        f"      {label}: {ev['path']} [{'found' if ev['exists'] else 'MISSING'}]"
                    )
    for other in d["other_failing_calls"]:
        out.append(f"{other['call_site']}: {other['failed']} of {other['total']} calls had trouble")

    q = d["quality"]
    out.append("\n--- run quality ---")
    if q is None:
        out.append("(no run_quality_summary row)")
    else:
        out.append(
            f"wall clock {q['wall_clock_ms']}ms | calls {q['total_calls']} ({q['retry_calls']} retries) | "
            f"validator failures {q['validator_failures']} | truncated {q['truncated_calls']} | "
            f"empty content {q['empty_content_calls']}"
        )
        out.append(
            f"gate1: {_fmt(q['gate1'][0])} ({_fmt(q['gate1'][1], '')}) | "
            f"gate2: {_fmt(q['gate2'][0])} ({_fmt(q['gate2'][1], '')})"
        )
    return "\n".join(out)


# --- recent errors, tracked across runs -----------------------------------------------


def _why_rejected(calls: list[dict], *, limit: int = 3, width: int = 70) -> str:
    """Every DISTINCT reason any attempt of this call was not accepted, in order:
    each validator error (the same rule with different numbers counts once), then a
    parse error, then an abnormal finish (a truncated reply). Showing only the first
    error hid that a second rule (for example the Macro stale-data rule) fired in
    every rejection. At most `limit` are shown, then a "+N more"."""
    reasons: dict[str, str] = {}
    for c in calls:
        candidates = list(c["validator_errors"] or [])
        if c["parse_error"]:
            candidates.append(f"unparseable: {c['parse_error']}")
        if c["finish_reason"] not in _NORMAL_FINISH:
            candidates.append(f"finish_reason={c['finish_reason']}")
        for text in candidates:
            reasons.setdefault(rule_slug(text), _short(text, width))
    if not reasons:
        return "no reason recorded"
    shown = list(reasons.values())[:limit]
    extra = len(reasons) - len(shown)
    return "; ".join(shown) + (f"; +{extra} more" if extra else "")


def _symbols_note(context) -> str:
    """The symbols a data event happened for, when the hook knew them (a peer's news
    call and the stock's own call are otherwise indistinguishable)."""
    symbols = (context or {}).get("symbols") if isinstance(context, dict) else None
    if not symbols:
        return ""
    shown = ", ".join(symbols[:5])
    return f" [{shown}{', ...' if len(symbols) > 5 else ''}]"


_SUMMARY_MAX_LINES = 8


def format_run_summary(d: dict) -> str:
    """The closing block printed after every run: the verdict, then every
    problem in one line each. A run can complete on degraded data (an agent that
    failed, articles that could not be scored, a provider that was skipped) and
    still print a clean-looking outlook; this is what says so. Low-severity
    events are only counted here (the full list is in diagnose_run)."""
    run = d["run"]
    out = [
        f"=== RUN SUMMARY: {_fmt(run['ticker'], '?')} | {run['account_type']} / "
        f"{run['timeline']} | run_id={run['run_id']} ===",
        f"RESULT: {d['verdict']['headline']}",
    ]
    problems: list[str] = []
    low = 0
    for e in sorted(d["errors"], key=lambda e: _SEVERITY_RANK.get(e["severity"], 9)):
        if e["severity"] == "low":
            low += 1
            continue
        fingerprint = e["fingerprint"] or ""
        if fingerprint.startswith("data:"):
            # A data-layer event: the fingerprint IS the useful label
            # (data:fred:get_macro_data:fetch_failed). The message alone is just
            # the exception text and would not say which provider broke.
            who = fingerprint
        else:
            who = f"{e['component']}/{e['error_type']}" + (
                f" agent={e['agent_name']}" if e["agent_name"] else ""
            )
        times = f" x{e['count']}" if e["count"] > 1 else ""
        problems.append(
            f"[{e['severity']}] {who}{times}: {_short(e['message'], 110)}"
            f"{_symbols_note(e.get('context'))}"
        )
    if not d["errors"]:
        for entry in d["run"]["error_log"]:
            problems.append(f"[{_fmt(entry.get('stage'))}] {_short(entry.get('error'), 110)}")
    recovered, exhausted = _retried_agents(d["agents"], d["failing_calls"])
    # Most serious first: an agent whose every attempt failed validation and whose last
    # output was used anyway (validation is a warning, not a gate), then recovered retries.
    for name in exhausted:
        calls = d["failing_calls"][f"agent:{name}"]
        problems.append(
            f"{name}: all {len(calls)} attempt(s) failed validation; the last output was "
            f"used anyway ({_why_rejected(calls)})"
        )
    for name in recovered:
        calls = d["failing_calls"][f"agent:{name}"]
        rejected = [
            c
            for c in calls
            if c["validator_passed"] is False
            or c["parsed_ok"] is False
            or c["finish_reason"] not in _NORMAL_FINISH
        ]
        problems.append(
            f"{name}: {len(rejected)} of {len(calls)} attempt(s) rejected, then passed"
            f" ({_why_rejected(calls)})"
        )
    if problems:
        out.append("problems:")
        out.extend(f"  {line}" for line in problems[:_SUMMARY_MAX_LINES])
        if len(problems) > _SUMMARY_MAX_LINES:
            out.append(f"  ... and {len(problems) - _SUMMARY_MAX_LINES} more")
    if low:
        out.append(f"({low} low-severity event(s) not shown)")
    out.append(f"details: python scripts/diagnose_run.py {run['run_id']}")
    return "\n".join(out)


async def recent_errors(
    session,
    since: datetime,
    *,
    now: datetime | None = None,
    include_low: bool = False,
    stale_after: int | None = None,
    top: int = 10,
) -> dict:
    """The cross-run view: health, recurring failures, stuck runs, and the
    quality signals from the model calls themselves."""
    since = _naive_utc(since)
    current = _naive_utc(now) if now is not None else _now()

    # -- runs in the window, by status
    status_counts = dict(
        (
            await session.execute(
                select(AnalysisRun.status, func.count())
                .where(AnalysisRun.triggered_at >= since)
                .group_by(AnalysisRun.status)
            )
        ).all()
    )

    # -- error_records grouped by fingerprint
    # Columns, not whole ORM rows: this report never needs the stack traces or
    # context blobs (up to ~20KB a row), only the fields it groups and prints.
    rows = (
        await session.execute(
            select(
                ErrorRecord.timestamp,
                ErrorRecord.severity,
                ErrorRecord.component,
                ErrorRecord.error_type,
                ErrorRecord.agent_name,
                ErrorRecord.dedup_subtype,
                ErrorRecord.run_id,
                ErrorRecord.stock_ticker,
                ErrorRecord.message,
                ErrorRecord.occurrence_count,
            ).where(ErrorRecord.timestamp >= since)
        )
    ).all()
    first_ever = {
        (component, error_type, agent or "", fp or ""): first
        for component, error_type, agent, fp, first in (
            await session.execute(
                select(
                    ErrorRecord.component,
                    ErrorRecord.error_type,
                    ErrorRecord.agent_name,
                    ErrorRecord.dedup_subtype,
                    func.min(ErrorRecord.timestamp),
                ).group_by(
                    ErrorRecord.component,
                    ErrorRecord.error_type,
                    ErrorRecord.agent_name,
                    ErrorRecord.dedup_subtype,
                )
            )
        ).all()
    }
    grouped: dict[tuple, dict] = {}
    hidden_low = 0
    for r in rows:
        if r.severity == "low" and not include_low:
            hidden_low += 1
            continue
        key = (r.component, r.error_type, r.agent_name or "", r.dedup_subtype or "")
        g = grouped.setdefault(
            key,
            {
                "component": r.component,
                "error_type": r.error_type,
                "agent_name": r.agent_name,
                "fingerprint": r.dedup_subtype,
                "severity": r.severity,
                "count": 0,
                "first_seen_in_window": r.timestamp,
                "last_seen": r.timestamp,
                "tickers": set(),
                "runs": set(),
                "sample_run_id": r.run_id,
                "sample_message": r.message,
            },
        )
        # A row can stand for many identical events within its run (250 unscored
        # articles are one row with a count of 250), so count events, not rows.
        g["count"] += r.occurrence_count or 1
        g["tickers"].add(r.stock_ticker or "?")
        g["runs"].add(r.run_id)
        if _SEVERITY_RANK.get(r.severity, 9) < _SEVERITY_RANK.get(g["severity"], 9):
            g["severity"] = r.severity
        if r.timestamp < g["first_seen_in_window"]:
            g["first_seen_in_window"] = r.timestamp
        if r.timestamp >= g["last_seen"]:
            g["last_seen"] = r.timestamp
            g["sample_run_id"] = r.run_id
            g["sample_message"] = r.message
    groups = []
    for key, g in grouped.items():
        ever = _naive_utc(first_ever.get(key))
        groups.append(
            {
                **g,
                "tickers": sorted(g["tickers"]),
                "runs": len(g["runs"]),
                "first_seen": ever,
                # New = the very first time this failure has EVER been recorded
                # falls inside the window, not merely that it appeared in it.
                "new_in_window": ever is not None and ever >= since,
            }
        )
    groups.sort(
        key=lambda g: (
            _SEVERITY_RANK.get(g["severity"], 9),
            -g["count"],
            -g["last_seen"].timestamp(),
        )
    )

    # -- failed runs that predate the recorder (or never got a row): read from error_log
    failed_runs = (
        await session.execute(
            select(AnalysisRun, Stock.canonical_ticker)
            .join(Stock, Stock.stock_id == AnalysisRun.stock_id, isouter=True)
            .where(AnalysisRun.status == "failed", AnalysisRun.triggered_at >= since)
        )
    ).all()
    # Only the failed runs in the window are asked about, not every run_id the
    # table has ever held.
    with_rows = set(
        (
            await session.execute(
                select(ErrorRecord.run_id).where(
                    ErrorRecord.run_id.in_([run.run_id for run, _ in failed_runs])
                )
            )
        )
        .scalars()
        .all()
    )
    legacy: dict[str, dict] = {}
    for run, ticker in failed_runs:
        if run.run_id in with_rows:
            continue
        entries = [e for e in (run.error_log or []) if isinstance(e, dict)] or [
            {"stage": "unknown", "error": "no reason recorded"}
        ]
        for entry in entries:
            key = f"legacy:{entry.get('stage') or 'unknown'}:{rule_slug(entry.get('error') or '', 50)}"
            g = legacy.setdefault(
                key,
                {
                    "fingerprint": key,
                    "count": 0,
                    "tickers": set(),
                    "last_seen": run.triggered_at,
                    "sample_run_id": run.run_id,
                    "sample_message": entry.get("error"),
                },
            )
            g["count"] += 1
            g["tickers"].add(ticker or "?")
            if run.triggered_at >= g["last_seen"]:
                g["last_seen"] = run.triggered_at
                g["sample_run_id"] = run.run_id
    legacy_groups = sorted(
        ({**g, "tickers": sorted(g["tickers"])} for g in legacy.values()),
        key=lambda g: -g["count"],
    )

    # -- stuck runs
    stuck = await list_non_terminal_runs(session, current, stale_after=stale_after)

    # -- quality signals: what the model calls themselves show
    agent_calls = (
        await session.execute(
            select(
                LLMCall.run_id,
                LLMCall.call_site,
                LLMCall.validator_errors,
                LLMCall.validator_passed,
                LLMCall.parsed_ok,
                LLMCall.parse_error,
                LLMCall.finish_reason,
            ).where(LLMCall.created_at >= since, LLMCall.call_site.like("agent:%"))
        )
    ).all()
    signal_count: Counter = Counter()
    signal_runs: dict[tuple, set] = {}
    signal_sample: dict[tuple, object] = {}
    for call in agent_calls:
        agent = call.call_site.split(":", 1)[1]
        found = []
        for err in call.validator_errors or []:
            found.append(("validator", agent, rule_slug(err, 70)))
        if call.parsed_ok is False:
            found.append(("parse", agent, rule_slug(call.parse_error or "unparseable", 70)))
        if call.finish_reason not in _NORMAL_FINISH:
            found.append(("finish", agent, str(call.finish_reason)))
        for signal in found:
            signal_count[signal] += 1
            signal_runs.setdefault(signal, set()).add(call.run_id)
            signal_sample[signal] = call.run_id
    signals = [
        {
            "kind": kind,
            "agent": agent,
            "rule": rule,
            "count": n,
            "runs": len(signal_runs[(kind, agent, rule)]),
            "sample_run_id": signal_sample[(kind, agent, rule)],
        }
        for (kind, agent, rule), n in signal_count.most_common(top)
    ]
    attempts_total = len(agent_calls)
    attempts_rejected = sum(1 for c in agent_calls if c.validator_passed is False)

    failed = status_counts.get("failed", 0)
    stale_count = sum(1 for s in stuck if s["stale"])
    needs_attention = bool(failed or stale_count or groups)
    return {
        "since": since,
        "now": current,
        "runs_by_status": status_counts,
        "runs_total": sum(status_counts.values()),
        "health": {
            "verdict": "ATTENTION" if needs_attention else "HEALTHY",
            "failed": failed,
            "stale": stale_count,
            "running": len(stuck) - stale_count,
            "error_groups": len(groups),
            "hidden_low_severity": hidden_low,
            "top": None if not groups else groups[0],
        },
        "groups": groups,
        "legacy_groups": legacy_groups,
        "stuck": stuck,
        "signals": signals,
        "attempts_total": attempts_total,
        "attempts_rejected": attempts_rejected,
    }


def format_recent_errors(d: dict) -> str:
    h = d["health"]
    statuses = ", ".join(f"{n} {s}" for s, n in sorted(d["runs_by_status"].items())) or "no runs"
    out = [
        f"=== error report | since {_ts(d['since'])} UTC | {d['runs_total']} run(s): {statuses} ===",
        f"HEALTH: {h['verdict']} - {h['failed']} failed run(s), {h['stale']} stale, "
        f"{h['running']} running, {h['error_groups']} distinct error group(s)"
        + (
            f" (+{h['hidden_low_severity']} low-severity hidden; use --all)"
            if h["hidden_low_severity"]
            else ""
        ),
    ]
    if h["top"]:
        t = h["top"]
        out.append(f"most frequent: {_group_label(t)} x{t['count']}")

    out.append(f"\n--- error groups ({len(d['groups'])}) ---")
    if not d["groups"]:
        out.append("(none)")
    for g in d["groups"]:
        flag = "  NEW" if g["new_in_window"] else ""
        out.append(
            f"[{g['severity']}] {_group_label(g)}  x{g['count']} in {g['runs']} run(s){flag}"
        )
        out.append(
            f"    first seen {_ts(g['first_seen'])} | last seen {_ts(g['last_seen'])} | "
            f"tickers {', '.join(g['tickers'])}"
        )
        out.append(f"    e.g. run {g['sample_run_id']}: {_short(g['sample_message'])}")

    if d["legacy_groups"]:
        out.append(
            f"\n--- failed runs with no error record (before the recorder) ({len(d['legacy_groups'])}) ---"
        )
        for g in d["legacy_groups"]:
            out.append(f"{g['fingerprint']}  x{g['count']} | tickers {', '.join(g['tickers'])}")
            out.append(
                f"    last {_ts(g['last_seen'])} | e.g. run {g['sample_run_id']}: "
                f"{_short(g['sample_message'])}"
            )

    out.append(f"\n--- runs not finished ({len(d['stuck'])}) ---")
    if not d["stuck"]:
        out.append("(none)")
    for s in d["stuck"]:
        mark = "STALE" if s["stale"] else "running"
        out.append(
            f"{mark}: {_fmt(s['ticker'], '?')} {s['status']} | age {_fmt_minutes(s['age_minutes'])} | "
            f"idle {_fmt_minutes(s['idle_minutes'])} | {s['reason']} | run {s['run_id']}"
        )
    if h["stale"]:
        out.append("(stale runs can be ended with: recent_errors.py --fail-stale)")

    rate = ""
    if d["attempts_total"]:
        rate = f"{d['attempts_rejected']} of {d['attempts_total']} agent attempts rejected by a validator"
    out.append(f"\n--- quality signals: {rate or 'no agent calls in the window'} ---")
    for s in d["signals"]:
        out.append(
            f"{s['kind']:9} {s['agent']:14} {s['rule']}  x{s['count']} in {s['runs']} run(s)"
        )
    return "\n".join(out)


def _group_label(g: dict) -> str:
    who = f"{g['component']}/{g['error_type']}"
    if g.get("agent_name"):
        who += f" agent={g['agent_name']}"
    return f"{who} [{_fmt(g['fingerprint'], '-')}]"


def _short(text: object, limit: int = 140) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _fmt_minutes(value: float | None) -> str:
    if value is None:
        return "N/A"
    if value < 90:
        return f"{value:.0f}m"
    hours = value / 60
    return f"{hours:.1f}h" if hours < 48 else f"{hours / 24:.1f}d"


# --- quality by prompt state (ledger BB-045) -----------------------------------------------

# The template file each agent call site reads. `analysis_runs.llm_config["prompt_hashes"]` is
# keyed by template file, and `llm_calls.call_site` names the agent, so this joins the two.
_AGENT_TEMPLATES = {
    "agent:rsrch": "stock_researcher/v1.txt",
    "agent:fund": "fundamental_analyst/v1.txt",
    "agent:tech": "technical_analyst/v1.txt",
    "agent:sent": "sentiment_analyst/v1.txt",
    "agent:macro": "macro_economist/v1.txt",
    "agent:bull": "bull_advocate/v1.txt",
    "agent:bear": "bear_advocate/v1.txt",
    "agent:risk_stage_a": "risk_advisor/v1_stage_a.txt",
    "agent:tax": "tax_strategist/v1.txt",
    "agent:cio_stage_a": "cio/v1_stage_a.txt",
    "agent:cio_stage_b": "cio/v1_stage_b.txt",
    "agent:shadow_cio": "shadow_cio/v1.txt",
}
_NOT_RECORDED = "not recorded"


async def prompt_comparison(session, since: datetime) -> dict:
    """How often each agent's attempts were rejected, split by the hash of THAT agent's own
    template, so an edit to one prompt shows as a before and after (ledger BB-045). An
    attempt counts as rejected on a validator failure, a parse failure or a truncated reply,
    the same test the diagnosis uses. Runs from before hashes were recorded fall in one
    "not recorded" group per agent. `code_states` is how many different code hashes the runs
    in a group span: a prompt comparison is only fair when it is 1, because a code change
    alters what the agents are told as well."""
    since = _naive_utc(since)
    rows = (
        await session.execute(
            select(
                LLMCall.run_id,
                LLMCall.call_site,
                LLMCall.validator_passed,
                LLMCall.parsed_ok,
                LLMCall.finish_reason,
                LLMCall.validator_errors,
                LLMCall.created_at,
                AnalysisRun.llm_config,
            )
            .join(AnalysisRun, AnalysisRun.run_id == LLMCall.run_id)
            .where(LLMCall.call_site.like("agent:%"), LLMCall.created_at >= since)
            .order_by(LLMCall.created_at)
        )
    ).all()
    groups: dict[tuple[str, str], dict] = {}
    for run_id, site, passed, parsed, finish, errors, created, config in rows:
        config = config or {}
        template = _AGENT_TEMPLATES.get(site)
        digest = (config.get("prompt_hashes") or {}).get(template) if template else None
        group = groups.setdefault(
            (site, digest or _NOT_RECORDED),
            {"runs": set(), "attempts": 0, "rejected": 0, "reasons": Counter(), "code": set()},
        )
        group["runs"].add(run_id)
        group["attempts"] += 1
        group["last"] = created
        if config.get("code_hash"):
            group["code"].add(config["code_hash"])
        if passed is False or parsed is False or finish not in _NORMAL_FINISH:
            group["rejected"] += 1
            if errors:
                reason = rule_slug(errors[0])
            elif parsed is False:
                reason = "unparseable output"
            else:
                reason = f"finish_reason={finish}"
            group["reasons"][reason] += 1
    out = []
    for (site, digest), g in groups.items():
        out.append(
            {
                "agent": site.removeprefix("agent:"),
                "site": site,
                "template": _AGENT_TEMPLATES.get(site),
                "prompt_hash": digest,
                "runs": len(g["runs"]),
                "attempts": g["attempts"],
                "rejected": g["rejected"],
                "rate": g["rejected"] / g["attempts"] if g["attempts"] else 0.0,
                "top_reasons": g["reasons"].most_common(2),
                "code_states": len(g["code"]),
                "last": g["last"],
            }
        )
    out.sort(key=lambda r: (r["agent"], -(r["last"].timestamp() if r["last"] else 0)))
    return {"since": since, "groups": out}


def format_prompt_comparison(d: dict) -> str:
    lines = [
        f"=== quality by prompt state: agent attempts since {_ts(d['since'])} UTC ===",
        "rejected = validator failure, parse failure or truncated reply, per attempt. Compare two "
        "rows of the same agent only when their code states are 1 (a code change alters what the "
        "agent is told too).",
    ]
    if not d["groups"]:
        lines.append("(no agent attempts in this window)")
        return "\n".join(lines)
    current = None
    for g in d["groups"]:
        if g["agent"] != current:
            current = g["agent"]
            lines.append(f"\n{g['agent']}  ({g['template'] or 'no template known'})")
        code_note = "" if g["code_states"] <= 1 else f" | CODE CHANGED across {g['code_states']} states"
        lines.append(
            f"  {g['prompt_hash']:<14} {g['runs']:>3} run(s) {g['attempts']:>4} attempt(s) "
            f"{g['rejected']:>3} rejected ({g['rate']:.0%})  last {_ts(g['last'])}{code_note}"
        )
        for reason, count in g["top_reasons"]:
            lines.append(f"      x{count}  {_short(reason, 100)}")
    return "\n".join(lines)
