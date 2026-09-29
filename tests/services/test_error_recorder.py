"""Tests for services/error_recorder.py (86bc997wr): building error notes and
writing them to error_records without ever raising. Hermetic -- in-memory
SQLite only, no network, no Ollama.
"""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api.database import Base
from api.tables.error_records import ErrorRecord
from services import error_recorder
from services.error_recorder import (
    _relative_to_project,
    build_error_note,
    build_fingerprint,
    describe_exception,
    rule_slug,
    summarize_call_log,
    write_error_notes,
)
from services.error_report import parse_since

# The test directory stands in for "this project's own code" where a test needs
# a fingerprint from a frame it raised itself.
TESTS_ROOT = Path(__file__).resolve().parents[1]


async def _engine(create_tables: bool = True):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    if create_tables:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    return engine


async def _rows(engine) -> list[ErrorRecord]:
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        return list((await session.execute(select(ErrorRecord))).scalars())


def _raised(exc: Exception) -> Exception:
    """An exception that was actually raised, so it carries a real traceback --
    the same shape the orchestrator collects out of asyncio.gather."""

    def _inner_frame_for_test():
        raise exc

    try:
        _inner_frame_for_test()
    except Exception as caught:
        return caught


# --- build_error_note --------------------------------------------------------


def test_note_has_every_errorrecord_column_it_is_meant_to_set():
    run_id, user_id = uuid4(), uuid4()
    note = build_error_note(
        "agent",
        "all_retries_exhausted",
        "high",
        "cio_stage_a failed",
        run_id=run_id,
        user_id=user_id,
        stock_ticker="AAPL",
        agent_name="cio_stage_a",
        dedup_subtype="stage_a",
        context={"stage": "cio_stage_a"},
    )
    assert note["component"] == "agent"
    assert note["error_type"] == "all_retries_exhausted"
    assert note["severity"] == "high"
    assert note["run_id"] == run_id and note["user_id"] == user_id
    assert note["stock_ticker"] == "AAPL" and note["agent_name"] == "cio_stage_a"
    assert note["dedup_subtype"] == "stage_a"
    assert note["context_json"] == {"stage": "cio_stage_a"}
    assert note["stack_trace"] is None
    # Stamped when the failure happened, naive UTC like the column's own default.
    assert note["timestamp"] == note["first_seen"] == note["last_seen"]
    assert note["timestamp"].tzinfo is None
    assert abs((datetime.now(UTC).replace(tzinfo=None) - note["timestamp"]).total_seconds()) < 5
    # Every key must be a real ErrorRecord column, or ErrorRecord(**note) breaks.
    assert set(note) <= {c.name for c in ErrorRecord.__table__.columns}


def test_note_takes_its_stack_trace_from_an_already_caught_exception():
    exc = _raised(ValueError("boom"))
    note = build_error_note("data_pipeline", "prepare_exception", "high", str(exc), exc=exc)
    assert "ValueError: boom" in note["stack_trace"]
    assert "_inner_frame_for_test" in note["stack_trace"]


def test_note_redacts_api_keys_in_message_traceback_and_context():
    url = "https://example.test/v3/profile/AAPL?apikey=SECRET123&x=1"
    exc = _raised(RuntimeError(f"GET {url} failed; header Authorization: Bearer abc.DEF-123"))
    note = build_error_note(
        "data_pipeline",
        "prepare_exception",
        "high",
        str(exc),
        exc=exc,
        context={
            "url": url,
            "nested": {"api_token": "HUNTER2"},
            "items": [{"password": "HUNTER2"}],
        },
    )
    blob = f"{note['message']} {note['stack_trace']} {note['context_json']}"
    for secret in ("SECRET123", "abc.DEF-123", "HUNTER2"):
        assert secret not in blob
    assert "apikey=[REDACTED]" in note["message"]
    assert "x=1" in note["message"]  # only the secret value is removed


def test_note_truncates_message_from_the_end_and_traceback_from_the_start():
    exc = _raised(ValueError("x" * 20000))
    note = build_error_note("orchestrator", "pipeline_exception", "critical", "y" * 20000, exc=exc)
    assert note["message"].startswith("y" * 100) and note["message"].endswith("...[truncated]")
    assert len(note["message"]) <= 8000 + len("...[truncated]")
    # A traceback keeps its tail: the error line is the most useful part.
    assert note["stack_trace"].startswith("[truncated]...")
    assert note["stack_trace"].endswith("x")
    assert len(note["stack_trace"]) <= 8000 + len("[truncated]...")


def test_note_clamps_short_columns_so_postgres_would_not_reject_them():
    note = build_error_note(
        "c" * 80,
        "t" * 80,
        "high",
        "m",
        agent_name="a" * 40,
        stock_ticker="T" * 40,
        dedup_subtype="d" * 200,
    )
    assert len(note["component"]) == 50 and len(note["error_type"]) == 50
    assert len(note["agent_name"]) == 30 and len(note["stock_ticker"]) == 20
    assert len(note["dedup_subtype"]) == 100


def test_note_never_raises_on_hostile_input():
    class Unprintable:
        def __str__(self):
            raise RuntimeError("no str for you")

    note = build_error_note(
        "agent",
        "agent_exception",
        "medium",
        Unprintable(),
        context={"obj": Unprintable(), "when": datetime(2026, 9, 29)},
    )
    assert "unprintable" in note["message"]
    assert note["context_json"] is not None  # non-serializable values are stringified


def test_note_survives_characters_a_legacy_code_page_cannot_encode():
    """A real `charmap` failure is in this project's own run history."""
    message = "bad \ud800 lone surrogate, non-breaking‑hyphen, café, → arrow"
    note = build_error_note("data_pipeline", "prepare_exception", "high", message)
    note["message"].encode("utf-8")  # must be encodable, or a DB driver would choke
    assert "‑" in note["message"] and "café" in note["message"]


# --- write_error_notes -------------------------------------------------------


async def test_write_persists_every_note_in_one_batch():
    engine = await _engine()
    run_id = uuid4()
    notes = [
        build_error_note("agent", "all_retries_exhausted", "high", f"failure {i}", run_id=run_id)
        for i in range(3)
    ]
    assert await write_error_notes(engine, notes) is True
    rows = await _rows(engine)
    assert sorted(r.message for r in rows) == ["failure 0", "failure 1", "failure 2"]
    assert all(r.run_id == run_id and r.status == "open" and r.occurrence_count == 1 for r in rows)


async def test_write_of_nothing_is_a_no_op():
    engine = await _engine()
    assert await write_error_notes(engine, []) is False
    assert await _rows(engine) == []


async def test_a_failing_write_is_swallowed_and_logged_not_raised():
    """A REAL failure -- the table does not exist -- not a mocked exception."""
    engine = await _engine(create_tables=False)
    note = build_error_note("agent", "agent_exception", "medium", "x", run_id=uuid4())
    with structlog.testing.capture_logs() as logs:
        assert await write_error_notes(engine, [note]) is False
    failed = [log for log in logs if log["event"] == "error_record_write_failed"]
    assert len(failed) == 1 and failed[0]["note_count"] == 1


async def test_cancellation_is_not_swallowed(monkeypatch):
    def _cancelled(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(error_recorder, "async_sessionmaker", _cancelled)
    engine = await _engine()
    with pytest.raises(asyncio.CancelledError):
        await write_error_notes(engine, [build_error_note("agent", "x", "low", "m")])


# --- fingerprints (86bc997wr) ---------------------------------------------------


def test_rule_slug_is_the_same_for_the_same_failure_with_different_numbers():
    a = rule_slug("narrative: too long (1913 chars, max 1080 = ~180 words)")
    b = rule_slug("narrative: too long (1351 chars, max 1080 = ~180 words)")
    assert a == b == "narrative: too long (# chars, max # = ~# words)"


def test_rule_slug_ignores_quoted_values_but_keeps_the_rule():
    slug = rule_slug(
        "key_factors[2].sentiment: 'bullish' must be positive|negative|neutral", limit=100
    )
    assert slug == "key_factors[#].sentiment: # must be positive|negative|neutral"


def test_rule_slug_keeps_different_rules_apart_and_is_bounded():
    assert rule_slug("narrative: too long (1 chars)") != rule_slug("caveats is empty")
    assert len(rule_slug("x" * 500)) == 80


def test_fingerprint_names_the_class_and_the_innermost_project_frame(monkeypatch):
    # The test directory stands in for "this project's own code".
    monkeypatch.setattr(error_recorder, "_PROJECT_ROOT", TESTS_ROOT)
    exc = _raised(ValueError("ticker AAPL on 2026-09-29 went wrong"))
    fingerprint = build_fingerprint(exc, "stage")
    assert fingerprint == "ValueError@services/test_error_recorder.py:_inner_frame_for_test"
    # The message holds tickers and dates, so identical bugs would look different.
    assert "AAPL" not in fingerprint and "2026" not in fingerprint


def test_fingerprint_is_stable_across_different_messages(monkeypatch):
    monkeypatch.setattr(error_recorder, "_PROJECT_ROOT", TESTS_ROOT)
    a = build_fingerprint(_raised(KeyError("AAPL")), "s")
    b = build_fingerprint(_raised(KeyError("MSFT")), "s")
    assert a == b


def test_fingerprint_falls_back_to_the_stage_without_a_project_frame():
    # Built but never raised: no traceback at all.
    assert build_fingerprint(ValueError("x"), "data_pipeline") == "ValueError@data_pipeline"
    assert build_fingerprint(ValueError("x")) == "ValueError@unknown"


def test_a_note_without_an_explicit_key_gets_the_fingerprint_and_an_explicit_key_wins(monkeypatch):
    monkeypatch.setattr(error_recorder, "_PROJECT_ROOT", TESTS_ROOT)
    exc = _raised(ValueError("boom"))
    auto = build_error_note(
        "agent", "agent_exception", "high", "m", exc=exc, context={"stage": "s"}
    )
    assert (
        auto["dedup_subtype"] == "ValueError@services/test_error_recorder.py:_inner_frame_for_test"
    )
    explicit = build_error_note("llm", "x", "high", "m", exc=exc, dedup_subtype="pass1")
    assert explicit["dedup_subtype"] == "pass1"


# --- call diagnostics embedded in a failed agent's note --------------------------


def _entry(seq, attempt, **extra):
    return {
        "seq": seq,
        "attempt": attempt,
        "call_site": "agent:fund",
        "model": "gpt-oss:20b",
        "finish_reason": "stop",
        "parsed_ok": True,
        "validator_passed": False,
        "validator_errors": ["narrative: too long (1913 chars, max 1080)"],
        "prompt_path": f"2026-09-29\\AAPL_abcd1234\\{seq}_agent-fund.{attempt}.prompt.txt",
        "response_path": f"2026-09-29\\AAPL_abcd1234\\{seq}_agent-fund.{attempt}.response.json",
        "prompt_eval_count": 2143,
        "eval_count": 380,
        "total_duration_s": 8.27,
        **extra,
    }


def test_call_summary_keeps_the_last_attempts_with_their_evidence_pointers():
    log = [_entry(i, i) for i in range(1, 6)]  # five attempts; only the last 3 matter
    summary = summarize_call_log(log)
    assert [c["seq"] for c in summary["calls"]] == [3, 4, 5]
    last = summary["calls"][-1]
    assert last["validator_errors"] == ["narrative: too long (1913 chars, max 1080)"]
    assert last["finish_reason"] == "stop" and last["validator_passed"] is False
    assert last["prompt_tokens"] == 2143 and last["completion_tokens"] == 380
    assert last["prompt_path"].endswith("5_agent-fund.5.prompt.txt")
    assert summary["call_site"] == "agent:fund"
    # The folder holding the raw prompt/response files, relative to SP_RUNS_DIR.
    assert summary["artifact_dir"] == "2026-09-29/AAPL_abcd1234"


def test_call_summary_carries_a_parse_error_and_redacts_credentials():
    log = [
        _entry(
            1,
            1,
            parsed_ok=False,
            validator_passed=None,
            validator_errors=None,
            parse_error="Expecting ',' delimiter: line 1 column 5431 (char 5430)",
        )
    ]
    log.append(
        _entry(2, 2, validator_errors=["upstream https://example.test/x?apikey=SECRET123 failed"])
    )
    summary = summarize_call_log(log)
    assert summary["calls"][0]["parse_error"].startswith("Expecting ',' delimiter")
    assert "validator_errors" not in summary["calls"][0]  # nothing to say: left out, not None
    blob = str(summary)
    assert "SECRET123" not in blob and "apikey=[REDACTED]" in blob


def test_call_summary_is_bounded_per_string_and_per_list():
    log = [_entry(1, 1, validator_errors=["e" * 900] * 12, parse_error="p" * 900)]
    call = summarize_call_log(log)["calls"][0]
    assert len(call["validator_errors"]) == 5
    assert all(len(e) <= 200 + len("...[truncated]") for e in call["validator_errors"])
    assert len(call["parse_error"]) <= 300 + len("...[truncated]")


@pytest.mark.parametrize("log", [None, [], [{}], [{"validator_errors": 7}], ["not a dict"]])
def test_call_summary_never_raises_and_says_nothing_when_there_is_nothing(log):
    result = summarize_call_log(log)
    assert isinstance(result, dict)


def test_an_oversized_context_is_replaced_with_a_marker_not_stored_unbounded():
    note = build_error_note("agent", "x", "high", "m", context={"stage": "s", "blob": "z" * 50000})
    assert note["context_json"] == {"truncated": True, "dropped_keys": ["blob", "stage"]}


# --- review findings: redaction must hide credentials and NOT the evidence ---------


def test_token_counts_are_not_mistaken_for_credentials():
    """A real bug: a bare `token` substring rule replaced prompt_tokens and
    completion_tokens with [REDACTED] in every failed agent's row -- the exact
    evidence the row exists to keep. Went through build_error_note this time."""
    call_log = [
        {
            "seq": 1,
            "attempt": 1,
            "call_site": "agent:fund",
            "model": "m",
            "validator_passed": False,
            "validator_errors": ["narrative: too long"],
            "prompt_eval_count": 2143,
            "eval_count": 380,
        }
    ]
    note = build_error_note(
        "agent",
        "all_retries_exhausted",
        "high",
        "m",
        context={"stage": "pass1", **summarize_call_log(call_log)},
    )
    call = note["context_json"]["calls"][0]
    assert call["prompt_tokens"] == 2143 and call["completion_tokens"] == 380
    assert "REDACTED" not in str(note["context_json"])


def test_keys_that_are_token_counts_survive_but_real_credential_keys_do_not():
    context = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "max_tokens": 30,
        "tokenizer": "x",
        "token": "AAA",
        "api_token": "BBB",
        "refresh_token": "CCC",
        "client_secret": "DDD",
        "password": "EEE",
        "authorization": "FFF",
        "apikey": "GGG",
        "x-api-key": "HHH",
        "nested": {"FMP_API_KEY": "III", "max_tokens": 5},
    }
    stored = build_error_note("agent", "x", "high", "m", context=context)["context_json"]
    assert (stored["prompt_tokens"], stored["completion_tokens"], stored["max_tokens"]) == (
        10,
        20,
        30,
    )
    assert stored["tokenizer"] == "x" and stored["nested"]["max_tokens"] == 5
    for secret in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III"):
        assert secret not in str(stored)


@pytest.mark.parametrize(
    "text, secret",
    [
        ("GET /v3/profile?apikey=S3CRET&x=1", "S3CRET"),
        ("GET /v3/profile?symbol=AAPL&key=S3CRET", "S3CRET"),
        ("params={'apikey': 'S3CRET', 'symbol': 'AAPL'}", "S3CRET"),
        ('{"apikey": "S3CRET", "symbol": "AAPL"}', "S3CRET"),
        ("apikey = S3CRET", "S3CRET"),
        ("apikey:S3CRET", "S3CRET"),
        ("refresh_token=S3CRET&x=1", "S3CRET"),
        ("client_secret=S3CRET", "S3CRET"),
        ("FMP_API_KEY=S3CRET", "S3CRET"),
        ("x_api_key=S3CRET", "S3CRET"),
        ("auth_token=S3CRET", "S3CRET"),
        ("headers: {'X-Api-Key': 'S3CRET'}", "S3CRET"),
        ("Authorization: Bearer S3CRETVALUE123", "S3CRETVALUE123"),
        ("Authorization: Basic dXNlcjpzM2NyZXQtY3JlZGVudGlhbHM=", "dXNlcjpzM2NyZXQ"),
        ("Authorization: Token S3CRETVALUE123", "S3CRETVALUE123"),
        ("https://user:S3CRET@host.example/path", "S3CRET"),
        ("GET /x?apikey%3DS3CRET&y=1", "S3CRET"),
        ("Cookie: session=S3CRETSESSIONVALUE", "S3CRETSESSIONVALUE"),
    ],
)
def test_every_common_credential_shape_is_redacted_in_a_message(text, secret):
    note = build_error_note("data_pipeline", "prepare_exception", "high", text)
    assert secret not in note["message"], note["message"]
    assert "REDACTED" in note["message"]


def test_redaction_leaves_ordinary_text_and_token_counts_alone():
    text = (
        "prompt_tokens=2143 max_tokens: 4096 basic assumptions hold; symbol=AAPL page=2 tokenizer=x"
    )
    assert build_error_note("agent", "x", "low", text)["message"] == text


def test_a_secret_with_an_escaped_quote_no_longer_costs_the_whole_context():
    """Redacting serialized JSON could eat the backslash before an escaped quote,
    leave invalid JSON, and the whole context became None."""
    context = {"stage": "s", "detail": 'upstream said: token=abc\\"def and then stopped'}
    stored = build_error_note("agent", "x", "high", "m", context=context)["context_json"]
    assert stored is not None and stored["stage"] == "s"
    assert "abc" not in str(stored) and "REDACTED" in stored["detail"]


def test_a_secret_inside_a_stringified_object_in_the_context_is_redacted():
    class Holder:
        def __str__(self):
            return "conn(apikey=S3CRET)"

    stored = build_error_note("agent", "x", "high", "m", context={"obj": Holder()})["context_json"]
    assert "S3CRET" not in str(stored)


# --- review findings: fingerprints ---------------------------------------------------


def test_a_real_exception_from_project_code_is_fingerprinted_with_the_real_root():
    """No monkeypatching: parse_since really raises inside backend/src."""
    try:
        parse_since("not a time")
    except ValueError as exc:
        assert build_fingerprint(exc, "s") == "ValueError@services/error_report.py:parse_since"


def test_only_files_under_the_project_root_count_and_parent_src_directories_do_not(monkeypatch):
    monkeypatch.setattr(error_recorder, "_PROJECT_ROOT", Path("C:/proj/backend/src"))
    assert _relative_to_project("C:\\proj\\backend\\src\\data\\pipeline.py") == "data/pipeline.py"
    assert _relative_to_project("c:\\PROJ\\Backend\\SRC\\data\\p.py") == "data/p.py"  # Windows
    # A checkout that merely lives under some directory called src is not ours.
    assert _relative_to_project("C:\\src\\other\\backend\\tests\\test_x.py") is None
    assert _relative_to_project("/usr/src/python/lib/json/decoder.py") is None
    assert (
        _relative_to_project("C:\\proj\\backend\\.venv\\Lib\\site-packages\\aiohttp\\c.py") is None
    )
    assert _relative_to_project("C:\\proj\\backend\\srcx\\data\\p.py") is None  # prefix, not parent


def test_an_http_status_on_the_exception_separates_otherwise_identical_failures(monkeypatch):
    monkeypatch.setattr(error_recorder, "_PROJECT_ROOT", TESTS_ROOT)

    class HttpError(Exception):
        def __init__(self, status):
            super().__init__(f"HTTP {status}")
            self.status = status

    a = build_fingerprint(_raised(HttpError(402)), "s")
    b = build_fingerprint(_raised(HttpError(429)), "s")
    assert a.endswith("#402") and b.endswith("#429") and a != b
    assert build_fingerprint(HttpError(500), "stage").endswith("@stage#500")


# --- review findings: recording must never replace the real error -------------------


def test_describing_an_exception_whose_str_raises_does_not_raise():
    class Hostile(Exception):
        def __str__(self):
            raise RuntimeError("no str for you")

    assert describe_exception(Hostile()) == "Hostile"
    assert describe_exception(ValueError("boom")) == "ValueError: boom"
