"""Tests for LLMCall.from_call_log_entry (86bbwachy Phase 3 consolidation) --
the shared call_log-entry-to-row mapping services/orchestrator.py's own
_llm_call_rows and data/pipeline.py's own _precompute_llm_call_rows both
call into, instead of each carrying its own near-identical ~20-line field
mapping. Direct, isolated coverage of the mapping itself, independent of
either caller's own slicing/consumed-index concerns.
"""
# Mapper-reachability for constructing a real LLMCall() (needed by every
# test below) is handled once, globally, by tests/conftest.py -- see its
# own comment for why.
from api.tables.llm_calls import LLMCall


def _entry(**overrides) -> dict:
    defaults = dict(
        seq=0,
        call_site="agent:fund",
        model="gpt-oss:20b",
        attempt=1,
        context_tag="analysis",
        options_json={"num_ctx": 32768},
        prompt_path="2026-09-24/AAPL_aaaaaaaa/0_agent-fund.1.prompt.txt",
        context_path=None,
        response_path="2026-09-24/AAPL_aaaaaaaa/0_agent-fund.1.response.json",
        prompt_eval_count=40,
        eval_count=8,
        thinking_chars=0,
        total_duration_s=5.0,
        finish_reason="stop",
        parsed_ok=True,
        parse_error=None,
        validator_passed=True,
        validator_errors=[],
        auto_trimmed=False,
    )
    return {**defaults, **overrides}


def test_maps_every_field():
    row = LLMCall.from_call_log_entry(_entry(), run_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", agent_pass="pass1")

    assert row.run_id == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    assert row.context_tag == "analysis"
    assert row.seq == 0
    assert row.call_site == "agent:fund"
    assert row.agent_pass == "pass1"
    assert row.attempt == 1
    assert row.model == "gpt-oss:20b"
    assert row.options_json == {"num_ctx": 32768}
    assert row.prompt_path == "2026-09-24/AAPL_aaaaaaaa/0_agent-fund.1.prompt.txt"
    assert row.context_path is None
    assert row.prompt_tokens == 40
    assert row.completion_tokens == 8
    assert row.thinking_chars == 0
    assert row.latency_ms == 5000
    assert row.finish_reason == "stop"
    assert row.parsed_ok is True
    assert row.parse_error is None
    assert row.validator_passed is True
    assert row.validator_errors == []
    assert row.auto_trimmed is False
    # agent_output_id is never set here -- both real callers backfill it
    # themselves afterward (orchestrator.py's _agent_output_row) or leave
    # it unset entirely for calls with no AgentOutput to link to
    # (pipeline.py's precompute rows).
    assert row.agent_output_id is None


def test_call_site_is_a_hard_requirement():
    """Real producers (agents/base.py's call_model/_call_generate,
    agents/capture.py's record_call) always set this -- a KeyError here on
    a malformed entry is correct, not something to paper over with a
    fallback that would silently misattribute the row (see this
    classmethod's own docstring)."""
    entry = _entry()
    del entry["call_site"]
    try:
        LLMCall.from_call_log_entry(entry, run_id="x", agent_pass="pass1")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_model_is_a_hard_requirement():
    entry = _entry()
    del entry["model"]
    try:
        LLMCall.from_call_log_entry(entry, run_id="x", agent_pass="pass1")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_missing_optional_fields_default_sensibly():
    minimal = {"call_site": "agent:tech", "model": "gpt-oss:20b"}
    row = LLMCall.from_call_log_entry(minimal, run_id="x", agent_pass="pass1")

    assert row.context_tag == "analysis"
    assert row.attempt == 1
    assert row.options_json == {}
    assert row.auto_trimmed is False
    assert row.seq is None
    assert row.latency_ms is None
    assert row.prompt_tokens is None


def test_zero_second_duration_is_not_treated_as_missing():
    row = LLMCall.from_call_log_entry(_entry(total_duration_s=0.0), run_id="x", agent_pass="pass1")
    assert row.latency_ms == 0
