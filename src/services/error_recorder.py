"""Persist run failures to error_records (86bc997wr).

Two pure, orchestrator-agnostic pieces:

- build_error_note(): turns a failure into a plain dict shaped like an
  ErrorRecord row. Synchronous, does no I/O, touches no ORM object, and never
  raises -- so it is safe to call from anywhere in a run, including inside
  asyncio.gather coroutines and on a session that is poisoned by a failed flush
  (see orchestrator.py's comments on why reading ORM attributes there is not).
- write_error_notes(): writes a batch of notes in ONE short, independent
  session and transaction, and swallows every failure. A problem recording an
  error must never change the outcome of the run it describes.

The orchestrator queues notes while a run executes and flushes them once at the
end of run() (see AnalysisOrchestrator._note_error / _flush_error_notes), which
is why nothing here needs to care about the run's own session state.

Named error_recorder, not capture: agents/capture.py already exists and is a
different thing (a per-call artifact recorder).
"""

import json
import re
import traceback
from datetime import UTC, datetime
from pathlib import Path

import structlog
from sqlalchemy.ext.asyncio import async_sessionmaker

from api.tables.error_records import ErrorRecord

logger = structlog.get_logger()

# Long enough for a real traceback, short enough that one runaway message
# can't bloat the table. Messages keep their head (the cause is usually first);
# tracebacks keep their tail (the final frames and the error line are last).
_MAX_TEXT = 8000

# ErrorRecord's String(N) column lengths. SQLite ignores them, PostgreSQL (the
# documented prod database) raises on overflow, so clamp here rather than let a
# long exception class name or dedup key fail the whole batch there.
_COLUMN_LIMITS = {
    "severity": 10,
    "component": 50,
    "error_type": 50,
    "dedup_subtype": 100,
    "agent_name": 30,
    "stock_ticker": 20,
}

# Provider errors can carry request URLs, headers and dict reprs (open ticket
# 86bbq7dmj: FMP's aiohttp exceptions leak the apikey into tracebacks). Anything
# stored would otherwise persist those credentials into the database.
#
# A credential NAME is anything ending in one of these words, so `apikey`,
# `x-api-key`, `FMP_API_KEY`, `refresh_token`, `client_secret` all count. It must
# END there: `prompt_tokens` and `max_tokens` are token COUNTS, not credentials,
# and redacting them would wipe exactly the evidence an error row exists to keep.
_CRED_NAME = (
    r"(?:[A-Za-z0-9]+[_-])*"
    r"(?:api[_-]?key|apikey|access[_-]?token|refresh[_-]?token|auth[_-]?token|"
    r"client[_-]?secret|token|secret|password|passwd|authorization)"
)
# name=value, name: value, "name": "value", 'name': 'value' (URL query, header
# line, JSON, dict repr). Keeps the name and the separator, hides the value.
_SECRET_ASSIGN = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?P<q>[\"']?)(?P<name>" + _CRED_NAME + r")(?P=q)"
    r"(?P<sep>\s*[=:]\s*)(?P<vq>[\"']?)(?P<val>[^&\s\"',}\]\)]+)"
)
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")
_BASIC = re.compile(r"(?i)\bbasic\s+[A-Za-z0-9+/=]{16,}")
_USERINFO = re.compile(r"(://[^/\s:@]+:)[^@\s/]+@")
_AUTH_HEADER_LINE = re.compile(
    r"(?im)^(\s*(?:authorization|proxy-authorization|x-api-key|api-key|cookie|set-cookie)\s*:\s*).+$"
)
_ENCODED_ASSIGN = re.compile(r"(?i)((?:api[_-]?key|apikey|token|secret|password)%3D)[^&\s]+")
_KEY_PARAM = re.compile(r"(?i)([?&]key=)[^&\s\"']+")


def _redact(text: str) -> str:
    text = _BEARER.sub("Bearer [REDACTED]", text)
    text = _BASIC.sub("Basic [REDACTED]", text)
    text = _USERINFO.sub(r"\1[REDACTED]@", text)
    text = _AUTH_HEADER_LINE.sub(r"\1[REDACTED]", text)
    text = _ENCODED_ASSIGN.sub(r"\1[REDACTED]", text)
    text = _KEY_PARAM.sub(r"\1[REDACTED]", text)
    return _SECRET_ASSIGN.sub(
        lambda m: f"{m['q']}{m['name']}{m['q']}{m['sep']}{m['vq']}[REDACTED]", text
    )


# For a context dict, a secret can also sit under a telling key rather than in
# `name=value` text. Same rule as _CRED_NAME: the key must END in the word.
_SECRET_KEY = re.compile(
    r"(?i)(?:^|[_-])(?:api[_-]?key|apikey|token|secret|password|passwd|authorization|credentials?)$"
)


def _scrub(value):
    """Redact a context structure in place of its parts, recursively: blank the
    value of any key that names a credential, and redact every string value.

    Done on the structure BEFORE serialization, never on the serialized JSON: a
    regex over JSON text can consume the backslash before an escaped quote and
    leave invalid JSON, which used to throw the whole context away.
    """
    if isinstance(value, dict):
        return {
            k: "[REDACTED]" if _SECRET_KEY.search(str(k)) else _scrub(v) for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    if isinstance(value, str):
        return _redact(value)
    return value


def _json_default(value: object) -> str:
    """json.dumps fallback that can't fail: one unprintable value must not cost
    the whole context. Redacted, since it stringifies arbitrary objects."""
    try:
        return _redact(str(value))
    except Exception:
        return f"<unprintable {type(value).__name__}>"


def describe_exception(exc: BaseException) -> str:
    """`ExcClass: message`, safe to call from inside an except/finally: an
    exception whose __str__ itself raises must not replace the original error."""
    try:
        return f"{type(exc).__name__}: {exc}"
    except Exception:
        return type(exc).__name__


def _clean(value: object) -> str:
    """str() that can't fail, redacted, and safe to hand to any DB driver.

    Round-tripping through utf-8 with errors="replace" also neutralizes lone
    surrogates and characters a legacy Windows code page can't encode (a real
    `charmap` failure is in this project's own run history), so recording an
    error never fails on the content of the error itself.
    """
    try:
        text = str(value)
    except Exception:
        text = f"<unprintable {type(value).__name__}>"
    text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    return _redact(text)


def _head(text: str, limit: int = _MAX_TEXT) -> str:
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _tail(text: str, limit: int = _MAX_TEXT) -> str:
    return text if len(text) <= limit else "[truncated]..." + text[-limit:]


def _clamp(field: str, value: str | None) -> str | None:
    return None if value is None else value[: _COLUMN_LIMITS[field]]


_MAX_CONTEXT_CHARS = 12000


def _context_json(context: dict | None) -> dict | None:
    if not context:
        return None
    try:
        # Redact the structure (credential-named keys blanked, every string
        # value redacted), stringify anything not natively JSON-serializable,
        # then serialize. Never redact the serialized text (see _scrub).
        dumped = json.dumps(_scrub(context), default=_json_default)
        if len(dumped) > _MAX_CONTEXT_CHARS:
            # Keep the row writable and say what was dropped, rather than
            # store an unbounded blob.
            return {"truncated": True, "dropped_keys": sorted(context)}
        return json.loads(dumped)
    except Exception:
        return None


# --- fingerprints and call diagnostics ---------------------------------------
#
# A fingerprint is what makes "the same failure" recognizable across runs. It
# is stored in ErrorRecord.dedup_subtype, and reports group on
# (component, error_type, agent_name, dedup_subtype).

# The directory holding this project's own code (backend/src), taken from where
# this very module lives, so it is right on every machine. A frame counts as
# "ours" only if its file is under it. (An earlier version searched the path for
# "/src/", which also matched any parent directory named src, e.g. a checkout at
# C:/src/..., and made fingerprints differ from machine to machine.) A module
# constant so tests can point it at the test directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Digits and quoted values change from run to run ("1913 chars", a ticker); the
# rule that failed does not.
_VARIABLE_PARTS = re.compile("\\d+(?:\\.\\d+)?|'[^']*'|\"[^\"]*\"")


def rule_slug(text: object, limit: int = 80) -> str:
    """A stable identity for a validation failure: the message with numbers and
    quoted values replaced by '#'.

    "narrative: too long (1913 chars, max 1080)" and "... (1351 chars, max
    1080)" are the same failure and slug to the same string. Quoted values are
    the run-varying part in this project's validators (`'bullish' must be ...`);
    field names appear unquoted (`key_factors[2].sentiment`), so they survive.
    """
    return " ".join(_VARIABLE_PARTS.sub("#", _clean(text)).split())[:limit]


def _relative_to_project(filename: str) -> str | None:
    """`data/pipeline.py` for a file under the project root, else None."""
    normalized = str(filename).replace("\\", "/")
    if "site-packages" in normalized:
        return None
    root = str(_PROJECT_ROOT).replace("\\", "/").rstrip("/") + "/"
    # Windows paths are case-insensitive.
    if normalized.lower().startswith(root.lower()):
        return normalized[len(root) :]
    return None


def build_fingerprint(exc: BaseException, stage: str | None = None) -> str:
    """`ExcClass@path/in/src.py:function` -- the exception's class plus the
    innermost frame that is this project's own code, plus the HTTP status when
    the exception carries one (a provider's 402 and 429 are different problems).

    The function name, not the line number (line numbers churn with every edit),
    and never the message (it holds tickers, dates and ids, so identical bugs
    would look different). Falls back to `ExcClass@stage` when no project frame
    is in the traceback (an exception that was built but never raised, say).
    """
    name = type(exc).__name__
    try:
        status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
        suffix = f"#{status}" if isinstance(status, int) and not isinstance(status, bool) else ""
    except Exception:
        suffix = ""
    try:
        for frame in reversed(traceback.extract_tb(exc.__traceback__)):
            relative = _relative_to_project(frame.filename)
            if relative is not None:
                return f"{name}@{relative}:{frame.name}{suffix}"
    except Exception:
        pass
    return f"{name}@{stage or 'unknown'}{suffix}"


def summarize_call_log(call_log, last: int = 3) -> dict:
    """The evidence for a failed agent, small enough to live on the error row.

    Takes the runner's last few call_log entries (one per attempt) and keeps what
    it takes to diagnose without a join: which attempt failed and how (finish
    reason, parse error, the validator's own error strings), and where the raw
    prompt and response were saved. Never the prompt or response text itself, and
    every string is redacted and capped. Returns {} when there is nothing to say.

    The paths are exactly what llm_calls stores: relative to SP_RUNS_DIR.
    """
    try:
        entries = list(call_log or [])[-last:]
        if not entries:
            return {}
        calls = []
        for entry in entries:
            validator_errors = [
                _head(_clean(e), 200) for e in list(entry.get("validator_errors") or [])[:5]
            ]
            parse_error = entry.get("parse_error")
            summary = {
                "seq": entry.get("seq"),
                "attempt": entry.get("attempt"),
                "call_site": entry.get("call_site"),
                "finish_reason": entry.get("finish_reason"),
                "parsed_ok": entry.get("parsed_ok"),
                "validator_passed": entry.get("validator_passed"),
                "validator_errors": validator_errors or None,
                "parse_error": _head(_clean(parse_error), 300) if parse_error else None,
                "prompt_path": entry.get("prompt_path"),
                "response_path": entry.get("response_path"),
                "prompt_tokens": entry.get("prompt_eval_count"),
                "completion_tokens": entry.get("eval_count"),
                "latency_s": entry.get("total_duration_s"),
            }
            calls.append({k: v for k, v in summary.items() if v is not None})
        result: dict = {"calls": calls, "call_site": calls[-1].get("call_site")}
        prompt_path = next(
            (c["prompt_path"] for c in reversed(calls) if c.get("prompt_path")), None
        )
        if prompt_path:
            result["artifact_dir"] = str(prompt_path).replace("\\", "/").rsplit("/", 1)[0]
        return {k: v for k, v in result.items() if v is not None}
    except Exception:
        return {}


def safe_text(value: object, limit: int = 500) -> str:
    """A short, redacted, always-encodable rendering of `value` -- for callers
    outside this module that persist error text somewhere else (e.g. an
    analysis_runs.error_log entry) and must not store a credential doing it."""
    return _head(_clean(value), limit)


def build_error_note(
    component: str,
    error_type: str,
    severity: str,
    message: object,
    *,
    run_id=None,
    user_id=None,
    stock_ticker: str | None = None,
    agent_name: str | None = None,
    dedup_subtype: str | None = None,
    exc: BaseException | None = None,
    context: dict | None = None,
    occurrence_count: int = 1,
) -> dict:
    """A plain dict whose keys are ErrorRecord column names. Never raises.

    `occurrence_count` is how many times this same event happened within the run
    (a degradation event that fired for 250 articles is one row with a count of
    250, not 250 rows).

    `exc` supplies the stack trace. traceback.format_exception works on an
    exception that was already caught and stored (its __traceback__ survives),
    which is what lets the orchestrator note failures collected out of an
    asyncio.gather instead of inside an except block.
    """
    if dedup_subtype is None and exc is not None:
        # No explicit key given: fingerprint the failure by where it was raised.
        dedup_subtype = build_fingerprint(exc, (context or {}).get("stage"))
    stack_trace = None
    if exc is not None:
        try:
            stack_trace = _tail(
                _clean(
                    "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                ).rstrip()
            )
        except Exception:
            stack_trace = None

    # Naive UTC, matching what ErrorRecord's own func.now() default produces on
    # SQLite. Stamped when the failure happened, not when the batch is flushed.
    now = datetime.now(UTC).replace(tzinfo=None)
    return {
        "timestamp": now,
        "first_seen": now,
        "last_seen": now,
        "severity": _clamp("severity", _clean(severity)),
        "component": _clamp("component", _clean(component)),
        "error_type": _clamp("error_type", _clean(error_type)),
        "dedup_subtype": _clamp(
            "dedup_subtype", None if dedup_subtype is None else _clean(dedup_subtype)
        ),
        "run_id": run_id,
        "user_id": user_id,
        "agent_name": _clamp("agent_name", agent_name),
        "stock_ticker": _clamp("stock_ticker", stock_ticker),
        "message": _head(_clean(message)),
        "stack_trace": stack_trace,
        "context_json": _context_json(context),
        "occurrence_count": _count(occurrence_count),
    }


def _count(value: object) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


async def write_error_notes(bind, notes: list[dict]) -> bool:
    """Write every note in one transaction, in an independent session.

    `bind` is the caller's AsyncEngine (AnalysisOrchestrator holds
    `db.bind`). A session built from it is separate from the run's own
    session, so a poisoned or half-rolled-back run session cannot affect it --
    and it works unchanged against the in-memory engines the tests use, where
    a fresh AsyncSessionLocal() would hit the real app.db instead.

    Returns True if the batch was written, False if it was empty or the write
    failed. Never raises for an ordinary failure; asyncio.CancelledError is a
    BaseException and deliberately still propagates.
    """
    if not notes:
        return False
    try:
        session_factory = async_sessionmaker(bind, expire_on_commit=False)
        async with session_factory() as session:
            session.add_all(ErrorRecord(**note) for note in notes)
            await session.commit()
        return True
    except Exception as exc:
        logger.warning(
            "error_record_write_failed",
            run_id=str(notes[0].get("run_id")),
            note_count=len(notes),
            error=_head(_clean(exc), 500),
        )
        return False
