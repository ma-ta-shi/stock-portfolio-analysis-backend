import asyncio
import json

import aiohttp
import pytest

import agents.base as base_module
from agents.base import MAX_RETRIES, MODEL, BaseRunner, OllamaUnavailable, _preflight


# ---------- Fakes standing in for aiohttp's response/session objects ----------


class FakeResponse:
    def __init__(
        self,
        status: int = 200,
        json_data=None,
        raise_timeout: bool = False,
        raise_client_error: bool = False,
        raise_outer_decode_error: bool = False,
    ) -> None:
        self.status = status
        self._json_data = json_data
        self._raise_timeout = raise_timeout
        self._raise_client_error = raise_client_error
        self._raise_outer_decode_error = raise_outer_decode_error

    async def __aenter__(self) -> "FakeResponse":
        if self._raise_timeout:
            raise asyncio.TimeoutError()
        if self._raise_client_error:
            raise aiohttp.ClientError("connection reset")
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise aiohttp.ClientError(f"HTTP {self.status}")

    async def json(self):
        if self._raise_outer_decode_error:
            # Real aiohttp behavior on a truncated/malformed body: response.json()
            # calls json.loads() on the raw text, which raises json.JSONDecodeError
            # directly, not wrapped in an aiohttp-specific exception.
            json.loads("{not valid json")
        return self._json_data


class FakeSession:
    """Wired with either a single FakeResponse (returned every call) or a list
    (popped in order) -- lists are what the retry tests need."""

    def __init__(self, responses) -> None:
        self._responses = list(responses) if isinstance(responses, list) else [responses]
        self.calls: list[dict] = []
        self.closed = False

    def post(self, url: str, json: dict, **kwargs) -> FakeResponse:
        self.calls.append({"url": url, "json": json})
        return self._next()

    def get(self, url: str, **kwargs) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self._next()

    def _next(self) -> FakeResponse:
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]

    async def close(self) -> None:
        self.closed = True


def _ollama_chat_response(content: dict, thinking: str = "", done_reason: str = "stop") -> dict:
    return {
        "message": {"content": json.dumps(content), "thinking": thinking},
        "total_duration": 1_000_000_000,  # 1s in nanoseconds
        "prompt_eval_count": 100,
        "eval_count": 50,
        "done_reason": done_reason,
    }


def _ollama_generate_response(
    content: dict,
    thinking: str = "",
    context: list[int] | None = None,
    prompt_eval_cached_count: int = 0,
    done_reason: str = "stop",
) -> dict:
    """86bc2d414: /api/generate's response shape, confirmed live during that
    ticket's planning -- `response` (text) and `thinking` are separate
    top-level fields (unlike call_model's nested `message.content`/
    `message.thinking`), plus `context`, the raw token-ID continuation array."""
    return {
        "response": json.dumps(content),
        "thinking": thinking,
        "context": context if context is not None else [1, 2, 3],
        "total_duration": 1_000_000_000,
        "prompt_eval_count": 100,
        "prompt_eval_cached_count": prompt_eval_cached_count,
        "eval_count": 50,
        "done_reason": done_reason,
    }


def _tags_response(model_names: list[str]) -> dict:
    return {"models": [{"name": name} for name in model_names]}


def _pass_validator(result: dict) -> tuple[bool, list[str]]:
    return True, []


@pytest.fixture(autouse=True)
def _skip_preflight_by_default(monkeypatch):
    """Most tests below exercise call_model/call_with_validation, not preflight
    itself -- default the module-level cache to "already checked" so those
    tests never attempt a real network call. The preflight tests explicitly
    reset this back to False."""
    monkeypatch.setattr(base_module, "_preflight_checked", True)


# ---------- _preflight ----------


class TestPreflight:
    @pytest.mark.asyncio
    async def test_raises_ollama_unavailable_when_unreachable(self, monkeypatch):
        monkeypatch.setattr(base_module, "_preflight_checked", False)
        session = FakeSession(FakeResponse(raise_client_error=True))
        with pytest.raises(OllamaUnavailable, match="Cannot reach Ollama"):
            await _preflight(session=session)

    @pytest.mark.asyncio
    async def test_raises_ollama_unavailable_when_timed_out(self, monkeypatch):
        monkeypatch.setattr(base_module, "_preflight_checked", False)
        session = FakeSession(FakeResponse(raise_timeout=True))
        with pytest.raises(OllamaUnavailable, match="Cannot reach Ollama"):
            await _preflight(session=session)

    @pytest.mark.asyncio
    async def test_raises_ollama_unavailable_when_model_not_pulled(self, monkeypatch):
        monkeypatch.setattr(base_module, "_preflight_checked", False)
        session = FakeSession(FakeResponse(200, _tags_response(["some-other-model"])))
        with pytest.raises(OllamaUnavailable, match="is not pulled"):
            await _preflight(session=session)
        assert base_module._preflight_checked is False, "a failure must not be cached"

    @pytest.mark.asyncio
    async def test_succeeds_and_caches(self, monkeypatch):
        monkeypatch.setattr(base_module, "_preflight_checked", False)
        session = FakeSession(FakeResponse(200, _tags_response([MODEL])))
        await _preflight(session=session)
        assert base_module._preflight_checked is True

    @pytest.mark.asyncio
    async def test_runs_only_once_across_calls(self, monkeypatch):
        monkeypatch.setattr(base_module, "_preflight_checked", False)
        session = FakeSession(FakeResponse(200, _tags_response([MODEL])))
        await _preflight(session=session)
        await _preflight(session=session)
        await _preflight(session=session)
        assert len(session.calls) == 1


# ---------- call_model ----------


class TestCallModel:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        runner = BaseRunner()
        runner.session = FakeSession(FakeResponse(200, _ollama_chat_response({"answer": "ok"})))
        result = await runner.call_model("system prompt", "user message")
        assert result == {"answer": "ok"}

    @pytest.mark.asyncio
    async def test_sends_expected_payload_shape(self):
        runner = BaseRunner()
        session = FakeSession(FakeResponse(200, _ollama_chat_response({"a": 1})))
        runner.session = session
        await runner.call_model("sys", "usr", max_tokens=500, temperature=0.5)

        call = session.calls[0]
        assert call["json"]["model"] == MODEL
        assert call["json"]["stream"] is False
        # These two are the exact regressions CLAUDE.md's LLM Routing section warns
        # about -- think must be the string "low" (never a bool), and num_ctx must
        # always be present explicitly.
        assert call["json"]["think"] == "low"
        assert call["json"]["options"]["num_ctx"] == base_module.NUM_CTX
        assert call["json"]["options"]["num_predict"] == 500
        assert call["json"]["options"]["temperature"] == 0.5
        assert call["json"]["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ]

    @pytest.mark.asyncio
    async def test_raises_json_decode_error_on_empty_content(self):
        runner = BaseRunner()
        runner.session = FakeSession(
            FakeResponse(200, {"message": {"content": "", "thinking": "lots of reasoning"}})
        )
        with pytest.raises(json.JSONDecodeError):
            await runner.call_model("sys", "usr")

    @pytest.mark.asyncio
    async def test_raises_json_decode_error_on_malformed_outer_response_body(self):
        """A 200 status with a genuinely truncated/malformed body makes
        response.json() raise json.JSONDecodeError directly -- not wrapped in
        aiohttp.ClientError -- so it must surface the same way as
        _parse_json_response's own JSONDecodeError, not slip through
        uncaught."""
        runner = BaseRunner()
        runner.session = FakeSession(FakeResponse(200, raise_outer_decode_error=True))
        with pytest.raises(json.JSONDecodeError):
            await runner.call_model("sys", "usr")

    @pytest.mark.asyncio
    async def test_raises_client_error_on_bad_status(self):
        runner = BaseRunner()
        runner.session = FakeSession(FakeResponse(500))
        with pytest.raises(aiohttp.ClientError):
            await runner.call_model("sys", "usr")

    @pytest.mark.asyncio
    async def test_records_call_log_entry(self):
        runner = BaseRunner()
        runner.session = FakeSession(
            FakeResponse(200, _ollama_chat_response({"a": 1}, thinking="x"))
        )
        await runner.call_model("sys", "usr")
        assert len(runner.call_log) == 1
        entry = runner.call_log[0]
        assert entry["total_duration_s"] == 1.0
        assert entry["thinking_chars"] == 1
        # current_agent defaults to None -- subclasses (not built by this
        # ticket) are expected to set it before calling in; until then the
        # class name is the fallback label.
        assert entry["agent"] == "BaseRunner"

    @pytest.mark.asyncio
    async def test_call_log_uses_current_agent_when_set(self):
        runner = BaseRunner()
        runner.current_agent = "TECH"
        runner.session = FakeSession(FakeResponse(200, _ollama_chat_response({"a": 1})))
        await runner.call_model("sys", "usr")
        assert runner.call_log[0]["agent"] == "TECH"

    @pytest.mark.asyncio
    async def test_capture_disabled_without_ticker(self):
        """86bbwachy Phase 2: ticker is the gate. Every pre-existing test in
        this class never sets it, and must keep passing with no artifact
        writes and no crash -- this pins that behavior explicitly rather
        than relying on it being implicit."""
        runner = BaseRunner()
        runner.session = FakeSession(FakeResponse(200, _ollama_chat_response({"a": 1})))
        await runner.call_model("sys", "usr")
        entry = runner.call_log[0]
        assert entry["prompt_path"] is None
        assert entry["context_path"] is None
        assert entry["response_path"] is None
        assert entry["seq"] is None  # no seq_counter set either

    @pytest.mark.asyncio
    async def test_capture_writes_real_artifacts_when_ticker_is_set(self, tmp_path, monkeypatch):
        import agents.capture as capture_module

        monkeypatch.setattr(capture_module, "RUNS_DIR", str(tmp_path))
        runner = BaseRunner()
        runner.ticker = "AAPL"
        runner.run_id = "11111111-2222-3333-4444-555555555555"
        runner.current_agent = "FUND"
        runner.session = FakeSession(
            FakeResponse(200, _ollama_chat_response({"a": 1}, thinking="reasoning"))
        )
        await runner.call_model("system text", "user text")

        entry = runner.call_log[0]
        assert entry["call_site"] == "agent:fund"
        assert entry["context_tag"] == "analysis"
        assert entry["model"] == MODEL
        assert entry["options_json"]["num_ctx"] > 0
        assert entry["finish_reason"] == "stop"
        assert entry["parsed_ok"] is True

        prompt_path = tmp_path / entry["prompt_path"]
        response_path = tmp_path / entry["response_path"]
        assert prompt_path.exists()
        assert "system text" in prompt_path.read_text()
        assert "user text" in prompt_path.read_text()
        assert json.loads(response_path.read_text())["message"]["content"] == json.dumps({"a": 1})
        assert entry["context_path"] is None  # no structured context for /api/chat

    @pytest.mark.asyncio
    async def test_capture_records_parse_failure(self, tmp_path, monkeypatch):
        import agents.capture as capture_module

        monkeypatch.setattr(capture_module, "RUNS_DIR", str(tmp_path))
        runner = BaseRunner()
        runner.ticker = "AAPL"
        runner.session = FakeSession(FakeResponse(200, _ollama_chat_response({}, thinking="only reasoning")))
        # Empty content -- real failure mode already covered by
        # test_raises_json_decode_error_on_empty_content; this test is about
        # what capture records, not the exception itself.
        runner.session._responses[0]._json_data["message"]["content"] = ""
        with pytest.raises(json.JSONDecodeError):
            await runner.call_model("sys", "usr")
        entry = runner.call_log[0]
        assert entry["parsed_ok"] is False
        assert entry["finish_reason"] == "empty_content"
        assert entry["parse_error"]

    @pytest.mark.asyncio
    async def test_capture_seq_increments_across_calls_via_shared_counter(self, tmp_path, monkeypatch):
        """A shared itertools.count() (matching orchestrator.py's own
        real, run-wide counter) must keep incrementing across separate
        runner instances, not reset per runner -- the whole point of seq
        being run-scoped, not runner-scoped (spec §5.2)."""
        import itertools

        import agents.capture as capture_module

        monkeypatch.setattr(capture_module, "RUNS_DIR", str(tmp_path))
        shared_counter = itertools.count()

        runner_a = BaseRunner()
        runner_a.ticker = "AAPL"
        runner_a.seq_counter = shared_counter
        runner_a.session = FakeSession(FakeResponse(200, _ollama_chat_response({"a": 1})))
        await runner_a.call_model("sys", "usr")

        runner_b = BaseRunner()
        runner_b.ticker = "AAPL"
        runner_b.seq_counter = shared_counter
        runner_b.session = FakeSession(FakeResponse(200, _ollama_chat_response({"b": 2})))
        await runner_b.call_model("sys", "usr")

        assert runner_a.call_log[0]["seq"] == 0
        assert runner_b.call_log[0]["seq"] == 1

    @pytest.mark.asyncio
    async def test_uses_injected_session_without_creating_a_new_one(self):
        session = FakeSession(FakeResponse(200, _ollama_chat_response({"a": 1})))
        runner = BaseRunner(session=session)
        await runner.call_model("sys", "usr")
        assert runner.session is session


# ---------- session lifecycle ----------


class TestSessionLifecycle:
    @pytest.mark.asyncio
    async def test_context_manager_creates_and_closes_an_owned_session(self):
        async with BaseRunner() as runner:
            assert isinstance(runner.session, aiohttp.ClientSession)
            session = runner.session
        assert session.closed

    @pytest.mark.asyncio
    async def test_context_manager_leaves_an_injected_session_open(self):
        injected = FakeSession(FakeResponse(200, {}))
        async with BaseRunner(session=injected) as runner:
            assert runner.session is injected
        assert injected.closed is False


# ---------- call_with_validation ----------


class TestCallWithValidation:
    @pytest.mark.asyncio
    async def test_passes_on_first_try(self):
        runner = BaseRunner()
        runner.session = FakeSession(FakeResponse(200, _ollama_chat_response({"ok": True})))
        result, errors = await runner.call_with_validation("sys", "usr", _pass_validator)
        assert result == {"ok": True}
        assert errors == []

    @pytest.mark.asyncio
    async def test_retries_on_validation_failure_and_shows_previous_response(self):
        session = FakeSession(
            [
                FakeResponse(200, _ollama_chat_response({"narrative": "short"})),
                FakeResponse(
                    200, _ollama_chat_response({"narrative": "a much longer narrative now"})
                ),
            ]
        )
        runner = BaseRunner()
        runner.session = session

        def validator(result):
            if result["narrative"] == "short":
                return False, ["narrative: too short (5 chars, min 10)"]
            return True, []

        result, errors = await runner.call_with_validation("sys", "usr", validator)

        assert errors == []
        assert result["narrative"] == "a much longer narrative now"
        second_user_msg = session.calls[1]["json"]["messages"][1]["content"]
        assert "YOUR PREVIOUS RESPONSE" in second_user_msg
        assert '"short"' in second_user_msg

    @pytest.mark.asyncio
    async def test_auto_trims_a_small_overshoot_without_a_retry(self):
        long_summary = "word " * 90  # 90 words, over an 80-word max
        session = FakeSession(
            FakeResponse(200, _ollama_chat_response({"assessment_summary": long_summary.strip()}))
        )
        runner = BaseRunner()
        runner.session = session

        def validator(result):
            words = result["assessment_summary"].split()
            if len(words) > 80:
                return False, [f"assessment_summary: too long ({len(words)} words, max 80)"]
            return True, []

        result, errors = await runner.call_with_validation("sys", "usr", validator)
        assert errors == []
        assert len(result["assessment_summary"].split()) <= 80
        assert len(session.calls) == 1, "auto-trim must not spend a retry"

    @pytest.mark.asyncio
    async def test_retries_on_json_parse_error(self):
        session = FakeSession(
            [
                FakeResponse(200, {"message": {"content": "not valid json", "thinking": ""}}),
                FakeResponse(200, _ollama_chat_response({"ok": True})),
            ]
        )
        runner = BaseRunner()
        runner.session = session
        result, errors = await runner.call_with_validation("sys", "usr", _pass_validator)
        assert result == {"ok": True}
        assert errors == []

    @pytest.mark.asyncio
    async def test_retries_on_client_error_and_timeout(self):
        session = FakeSession(
            [
                FakeResponse(raise_client_error=True),
                FakeResponse(raise_timeout=True),
                FakeResponse(200, _ollama_chat_response({"ok": True})),
            ]
        )
        runner = BaseRunner()
        runner.session = session
        result, errors = await runner.call_with_validation("sys", "usr", _pass_validator)
        assert result == {"ok": True}
        assert errors == []

    @pytest.mark.asyncio
    async def test_exhausts_max_retries_and_returns_last_errors(self):
        def always_fails(result):
            return False, ["field: always wrong"]

        session = FakeSession(FakeResponse(200, _ollama_chat_response({"a": 1})))
        runner = BaseRunner()
        runner.session = session
        result, errors = await runner.call_with_validation("sys", "usr", always_fails)
        assert errors == ["field: always wrong"]
        assert len(session.calls) == MAX_RETRIES

    @pytest.mark.asyncio
    async def test_does_not_catch_ollama_unavailable(self, monkeypatch):
        """Deliberate, not a gap -- see the docstring on call_with_validation.
        Ollama-not-running / model-not-pulled are non-transient config
        problems; retrying can't fix either, and catching this here would
        flatten its actionable message into a generic error string. This
        test exists so that decision can't be silently reverted later.

        Mocks _preflight() itself rather than relying on call_model's
        internal (session-less) call to it failing some other way -- an
        earlier version of this test injected a failing FakeSession into
        `runner.session`, which call_model never actually passes to
        _preflight(), so it silently hit the real local Ollama instead of
        the fake and asserted the wrong exception entirely.
        """

        async def _always_unavailable(session=None):
            raise OllamaUnavailable("Ollama isn't running")

        monkeypatch.setattr(base_module, "_preflight", _always_unavailable)
        runner = BaseRunner()
        runner.session = FakeSession(FakeResponse(200, _ollama_chat_response({"ok": True})))
        with pytest.raises(OllamaUnavailable):
            await runner.call_with_validation("sys", "usr", _pass_validator)


# ---------- soft-error acceptance (86bbuhjup) ----------


class TestSoftErrorAcceptance:
    """SOFT_ERROR_PREFIXES/split_soft_errors -- CIO §122.5 / Pass 2 Rule 12:
    citation-breadth and narrative-length leniency should apply retry pressure
    on earlier attempts but not cost the agent its whole output once attempts
    run out. Confirmed live during 86bbuhjup that neither this file nor the
    harness's own simulation/runners/base.py actually wired this in before now
    -- SOFT_ERROR_PREFIXES was imported and combined but never referenced in
    _retry_loop, in either place."""

    @pytest.mark.asyncio
    async def test_accepts_on_final_attempt_when_only_soft_errors_remain(self):
        session = FakeSession(
            [FakeResponse(200, _ollama_chat_response({"a": i})) for i in range(MAX_RETRIES)]
        )
        runner = BaseRunner()
        runner.session = session

        def validator(result):
            return False, ["synthesis_narrative: too short (1200 chars, min 1500)"]

        result, errors = await runner.call_with_validation("sys", "usr", validator)
        assert errors == []
        assert len(session.calls) == MAX_RETRIES
        assert runner.call_log[-1]["passed"] is True
        assert runner.call_log[-1]["soft_errors"] == [
            "synthesis_narrative: too short (1200 chars, min 1500)"
        ]

    @pytest.mark.asyncio
    async def test_does_not_accept_early_when_only_soft_errors_present(self):
        """Only the FINAL attempt gets soft-error leniency -- an earlier attempt
        must still see the error and get a real chance to fix it, not be let
        off early."""
        session = FakeSession(
            [
                FakeResponse(200, _ollama_chat_response({"narrative": "short"})),
                FakeResponse(200, _ollama_chat_response({"narrative": "long enough now"})),
            ]
        )
        runner = BaseRunner()
        runner.session = session
        seen = {"count": 0}

        def validator(result):
            seen["count"] += 1
            if seen["count"] == 1:
                return False, ["synthesis_narrative: too short (1200 chars, min 1500)"]
            return True, []

        result, errors = await runner.call_with_validation("sys", "usr", validator)
        assert errors == []
        assert len(session.calls) == 2, "the first attempt must retry, not get soft-accepted"

    @pytest.mark.asyncio
    async def test_hard_error_on_final_attempt_still_fails(self):
        """A hard error mixed in with a soft one on the final attempt is still
        a real failure -- leniency only applies when NOTHING but soft errors
        remain."""
        session = FakeSession(FakeResponse(200, _ollama_chat_response({"a": 1})))
        runner = BaseRunner()
        runner.session = session

        def validator(result):
            return False, [
                "synthesis_narrative: too short (1200 chars, min 1500)",
                "stock_outlook: must be one of bullish|bearish, got 'sideways'",
            ]

        result, errors = await runner.call_with_validation("sys", "usr", validator)
        assert len(session.calls) == MAX_RETRIES
        assert "stock_outlook: must be one of bullish|bearish, got 'sideways'" in errors


# ---------- _call_generate (86bc2d414) ----------


class TestCallGenerate:
    @pytest.mark.asyncio
    async def test_capture_writes_artifacts_for_a_continuation_call(self, tmp_path, monkeypatch):
        """86bbwachy Phase 2, the /api/generate path -- a continuation
        successor turn (system=None, context passed in) only has the
        incremental prompt text to capture, not the full effective
        context. Confirms that's what actually gets written, not a crash
        or a silently-empty file."""
        import agents.capture as capture_module

        monkeypatch.setattr(capture_module, "RUNS_DIR", str(tmp_path))
        runner = BaseRunner()
        runner.ticker = "SHOP.TO"
        runner.run_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        runner.current_agent = "risk_stage_b"
        runner.session = FakeSession(
            FakeResponse(200, _ollama_generate_response({"b": 2}, context=[9, 9, 9]))
        )
        await runner._call_generate("continuation increment", context=[1, 2, 3])

        entry = runner.call_log[0]
        assert entry["call_site"] == "agent:risk_stage_b"
        assert entry["parsed_ok"] is True
        prompt_path = tmp_path / entry["prompt_path"]
        assert prompt_path.read_text() == "continuation increment"
        assert entry["context_path"] is None

    @pytest.mark.asyncio
    async def test_happy_path_returns_result_and_context(self):
        runner = BaseRunner()
        runner.session = FakeSession(
            FakeResponse(200, _ollama_generate_response({"answer": "ok"}, context=[1, 2, 3]))
        )
        result, context = await runner._call_generate("prompt text", system="sys")
        assert result == {"answer": "ok"}
        assert context == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_sends_expected_payload_shape_when_starting(self):
        runner = BaseRunner()
        session = FakeSession(FakeResponse(200, _ollama_generate_response({"a": 1})))
        runner.session = session
        await runner._call_generate(
            "usr prompt", system="sys prompt", max_tokens=500, temperature=0.5
        )

        call = session.calls[0]
        assert call["url"] == f"{base_module.OLLAMA_HOST}/api/generate"
        assert call["json"]["model"] == MODEL
        assert call["json"]["prompt"] == "usr prompt"
        assert call["json"]["system"] == "sys prompt"
        assert "context" not in call["json"]
        assert call["json"]["stream"] is False
        assert call["json"]["think"] == "low"
        assert "format" not in call["json"], (
            "format:'json' corrupts gpt-oss:20b on /api/generate -- see "
            "docs/technical/ollama-generate-format-json-finding.md"
        )
        assert call["json"]["options"]["num_ctx"] == base_module.NUM_CTX
        assert call["json"]["options"]["num_predict"] == 500
        assert call["json"]["options"]["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_sends_expected_payload_shape_when_continuing(self):
        runner = BaseRunner()
        session = FakeSession(FakeResponse(200, _ollama_generate_response({"a": 1})))
        runner.session = session
        await runner._call_generate("continuation prompt", context=[9, 9, 9])

        call = session.calls[0]
        assert call["json"]["prompt"] == "continuation prompt"
        assert call["json"]["context"] == [9, 9, 9]
        assert "system" not in call["json"], (
            "confirmed live during 86bc2d414 planning: a continuation call needs "
            "no system field -- Stage A's role/output are already available via "
            "context alone"
        )

    @pytest.mark.asyncio
    async def test_raises_json_decode_error_on_empty_response(self):
        runner = BaseRunner()
        runner.session = FakeSession(
            FakeResponse(200, {"response": "", "thinking": "lots of reasoning", "context": []})
        )
        with pytest.raises(json.JSONDecodeError):
            await runner._call_generate("prompt")

    @pytest.mark.asyncio
    async def test_raises_client_error_on_bad_status(self):
        runner = BaseRunner()
        runner.session = FakeSession(FakeResponse(500))
        with pytest.raises(aiohttp.ClientError):
            await runner._call_generate("prompt")

    @pytest.mark.asyncio
    async def test_raises_json_decode_error_on_malformed_outer_response_body(self):
        """Parity with TestCallModel's equivalent test -- a 200 status with a
        genuinely truncated/malformed body makes response.json() raise
        json.JSONDecodeError directly, not wrapped in aiohttp.ClientError, the
        same way it does for call_model's /api/chat path."""
        runner = BaseRunner()
        runner.session = FakeSession(FakeResponse(200, raise_outer_decode_error=True))
        with pytest.raises(json.JSONDecodeError):
            await runner._call_generate("prompt")

    @pytest.mark.asyncio
    async def test_records_call_log_entry(self):
        runner = BaseRunner()
        runner.session = FakeSession(
            FakeResponse(200, _ollama_generate_response({"a": 1}, thinking="x"))
        )
        await runner._call_generate("prompt")
        assert len(runner.call_log) == 1
        entry = runner.call_log[0]
        assert entry["total_duration_s"] == 1.0
        assert entry["thinking_chars"] == 1
        assert entry["agent"] == "BaseRunner"

    @pytest.mark.asyncio
    async def test_call_log_uses_current_agent_when_set(self):
        runner = BaseRunner()
        runner.current_agent = "RISK"
        runner.session = FakeSession(FakeResponse(200, _ollama_generate_response({"a": 1})))
        await runner._call_generate("prompt")
        assert runner.call_log[0]["agent"] == "RISK"

    @pytest.mark.asyncio
    async def test_thinking_is_captured_separately_from_response(self):
        runner = BaseRunner()
        runner.session = FakeSession(
            FakeResponse(200, _ollama_generate_response({"a": 1}, thinking="some reasoning"))
        )
        result, _ = await runner._call_generate("prompt")
        assert result == {"a": 1}
        assert runner.last_thinking == "some reasoning"

    @pytest.mark.asyncio
    async def test_captures_prompt_eval_cached_count(self):
        """The one deterministic, wording-independent signal that a
        continuation call actually reused Stage A's context -- see the
        docstring on _call_generate's timing capture."""
        runner = BaseRunner()
        runner.session = FakeSession(
            FakeResponse(
                200, _ollama_generate_response({"a": 1}, context=[9, 9], prompt_eval_cached_count=42)
            )
        )
        await runner._call_generate("prompt", context=[1, 2, 3])
        assert runner.last_timing["prompt_eval_cached_count"] == 42


# ---------- call_with_validation_start (86bc2d414) ----------


class TestCallWithValidationStart:
    @pytest.mark.asyncio
    async def test_passes_on_first_try_and_returns_context(self):
        runner = BaseRunner()
        runner.session = FakeSession(
            FakeResponse(200, _ollama_generate_response({"ok": True}, context=[1, 2, 3]))
        )
        result, errors, context = await runner.call_with_validation_start(
            "sys", "usr", _pass_validator
        )
        assert result == {"ok": True}
        assert errors == []
        assert context == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_retries_and_shows_previous_response(self):
        session = FakeSession(
            [
                FakeResponse(200, _ollama_generate_response({"narrative": "short"})),
                FakeResponse(
                    200,
                    _ollama_generate_response(
                        {"narrative": "a much longer narrative now"}, context=[9, 9]
                    ),
                ),
            ]
        )
        runner = BaseRunner()
        runner.session = session

        def validator(result):
            if result["narrative"] == "short":
                return False, ["narrative: too short (5 chars, min 10)"]
            return True, []

        result, errors, context = await runner.call_with_validation_start(
            "sys", "usr", validator
        )
        assert errors == []
        assert result["narrative"] == "a much longer narrative now"
        assert context == [9, 9]
        second_prompt = session.calls[1]["json"]["prompt"]
        assert "YOUR PREVIOUS RESPONSE" in second_prompt
        assert '"short"' in second_prompt

    @pytest.mark.asyncio
    async def test_retries_on_json_parse_error(self):
        session = FakeSession(
            [
                FakeResponse(200, {"response": "not valid json", "thinking": "", "context": []}),
                FakeResponse(200, _ollama_generate_response({"ok": True}, context=[7])),
            ]
        )
        runner = BaseRunner()
        runner.session = session
        result, errors, context = await runner.call_with_validation_start(
            "sys", "usr", _pass_validator
        )
        assert result == {"ok": True}
        assert errors == []
        assert context == [7]

    @pytest.mark.asyncio
    async def test_retries_on_client_error_and_timeout(self):
        session = FakeSession(
            [
                FakeResponse(raise_client_error=True),
                FakeResponse(raise_timeout=True),
                FakeResponse(200, _ollama_generate_response({"ok": True}, context=[7])),
            ]
        )
        runner = BaseRunner()
        runner.session = session
        result, errors, context = await runner.call_with_validation_start(
            "sys", "usr", _pass_validator
        )
        assert result == {"ok": True}
        assert errors == []
        assert context == [7]

    @pytest.mark.asyncio
    async def test_exhaustion_returns_none_context_not_last_attempts(self):
        def always_fails(result):
            return False, ["field: always wrong"]

        session = FakeSession(
            FakeResponse(200, _ollama_generate_response({"a": 1}, context=[1, 1, 1]))
        )
        runner = BaseRunner()
        runner.session = session
        result, errors, context = await runner.call_with_validation_start(
            "sys", "usr", always_fails
        )
        assert errors == ["field: always wrong"]
        assert context is None, (
            "a caller has no legitimate reason to continue from a Stage A that "
            "never validated -- must not hand back a never-confirmed context"
        )
        assert len(session.calls) == MAX_RETRIES

    @pytest.mark.asyncio
    async def test_auto_trims_without_a_retry(self):
        long_summary = "word " * 90  # 90 words, over an 80-word max
        session = FakeSession(
            FakeResponse(
                200,
                _ollama_generate_response(
                    {"assessment_summary": long_summary.strip()}, context=[5]
                ),
            )
        )
        runner = BaseRunner()
        runner.session = session

        def validator(result):
            words = result["assessment_summary"].split()
            if len(words) > 80:
                return False, [f"assessment_summary: too long ({len(words)} words, max 80)"]
            return True, []

        result, errors, context = await runner.call_with_validation_start(
            "sys", "usr", validator
        )
        assert errors == []
        assert context == [5]
        assert len(session.calls) == 1, "auto-trim must not spend a retry"

    @pytest.mark.asyncio
    async def test_does_not_catch_ollama_unavailable(self, monkeypatch):
        async def _always_unavailable(session=None):
            raise OllamaUnavailable("Ollama isn't running")

        monkeypatch.setattr(base_module, "_preflight", _always_unavailable)
        runner = BaseRunner()
        runner.session = FakeSession(FakeResponse(200, _ollama_generate_response({"ok": True})))
        with pytest.raises(OllamaUnavailable):
            await runner.call_with_validation_start("sys", "usr", _pass_validator)


# ---------- call_with_validation_continue (86bc2d414) ----------


class TestCallWithValidationContinue:
    @pytest.mark.asyncio
    async def test_passes_on_first_try(self):
        runner = BaseRunner()
        runner.session = FakeSession(FakeResponse(200, _ollama_generate_response({"ok": True})))
        result, errors = await runner.call_with_validation_continue(
            "prompt", [1, 2, 3], _pass_validator
        )
        assert result == {"ok": True}
        assert errors == []

    @pytest.mark.asyncio
    async def test_every_retry_reuses_the_same_original_context(self):
        """The core correctness property of a continuation retry loop: a
        failed attempt's own new context must never replace the original --
        that would silently drift the chain away from Stage A's real state
        with each failed attempt."""
        session = FakeSession(
            [
                FakeResponse(200, _ollama_generate_response({"narrative": "short"})),
                FakeResponse(200, _ollama_generate_response({"narrative": "a much longer narrative now"})),
            ]
        )
        runner = BaseRunner()
        runner.session = session

        def validator(result):
            if result["narrative"] == "short":
                return False, ["narrative: too short (5 chars, min 10)"]
            return True, []

        original_context = [42, 43, 44]
        result, errors = await runner.call_with_validation_continue(
            "prompt", original_context, validator
        )
        assert errors == []
        assert session.calls[0]["json"]["context"] == original_context
        assert session.calls[1]["json"]["context"] == original_context

    @pytest.mark.asyncio
    async def test_omits_system_field(self):
        runner = BaseRunner()
        session = FakeSession(FakeResponse(200, _ollama_generate_response({"a": 1})))
        runner.session = session
        await runner.call_with_validation_continue("prompt", [1, 2, 3], _pass_validator)
        assert "system" not in session.calls[0]["json"]

    @pytest.mark.asyncio
    async def test_retries_on_json_parse_error(self):
        session = FakeSession(
            [
                FakeResponse(200, {"response": "not valid json", "thinking": "", "context": []}),
                FakeResponse(200, _ollama_generate_response({"ok": True})),
            ]
        )
        runner = BaseRunner()
        runner.session = session
        result, errors = await runner.call_with_validation_continue(
            "prompt", [1, 2, 3], _pass_validator
        )
        assert result == {"ok": True}
        assert errors == []

    @pytest.mark.asyncio
    async def test_retries_on_client_error_and_timeout(self):
        session = FakeSession(
            [
                FakeResponse(raise_client_error=True),
                FakeResponse(raise_timeout=True),
                FakeResponse(200, _ollama_generate_response({"ok": True})),
            ]
        )
        runner = BaseRunner()
        runner.session = session
        result, errors = await runner.call_with_validation_continue(
            "prompt", [1, 2, 3], _pass_validator
        )
        assert result == {"ok": True}
        assert errors == []

    @pytest.mark.asyncio
    async def test_exhausts_max_retries_and_returns_last_errors(self):
        def always_fails(result):
            return False, ["field: always wrong"]

        session = FakeSession(FakeResponse(200, _ollama_generate_response({"a": 1})))
        runner = BaseRunner()
        runner.session = session
        result, errors = await runner.call_with_validation_continue(
            "prompt", [1, 2, 3], always_fails
        )
        assert errors == ["field: always wrong"]
        assert len(session.calls) == MAX_RETRIES

    @pytest.mark.asyncio
    async def test_does_not_catch_ollama_unavailable(self, monkeypatch):
        async def _always_unavailable(session=None):
            raise OllamaUnavailable("Ollama isn't running")

        monkeypatch.setattr(base_module, "_preflight", _always_unavailable)
        runner = BaseRunner()
        runner.session = FakeSession(FakeResponse(200, _ollama_generate_response({"ok": True})))
        with pytest.raises(OllamaUnavailable):
            await runner.call_with_validation_continue("prompt", [1, 2, 3], _pass_validator)


# ---------- _parse_json_response (86bc2d414 trailing-content tolerance) ----------


class TestParseJsonResponse:
    def test_parses_clean_json(self):
        assert base_module._parse_json_response('{"a": 1}') == {"a": 1}

    def test_strips_markdown_fences(self):
        text = '```json\n{"a": 1}\n```'
        assert base_module._parse_json_response(text) == {"a": 1}

    def test_tolerates_trailing_stray_brace(self):
        """The trailing-brace failure mode from
        docs/technical/two-turn-execution-mechanism.md: a continuation
        successor turn with a flatter schema than its predecessor's has a
        reproducible tendency to append one stray extra `}`."""
        assert base_module._parse_json_response('{"a": 1}}') == {"a": 1}

    def test_tolerates_trailing_garbage_text(self):
        text = '{"a": 1} some trailing note the model appended'
        assert base_module._parse_json_response(text) == {"a": 1}

    def test_still_raises_on_genuinely_malformed_json(self):
        with pytest.raises(json.JSONDecodeError):
            base_module._parse_json_response("not valid json at all")


# ---------- _auto_trim ----------


class TestAutoTrim:
    def test_trims_to_last_sentence_boundary_when_present(self):
        # Lengths are derived, not hand-counted, so the fixture can't drift out
        # of sync with the assertions: `prefix` ends in ". " past the trim
        # window's halfway point, so the trimmer should cut there rather than
        # hard-truncating mid-filler.
        prefix = "A" * 23 + ". "
        text = prefix + "xxx"
        limit = 26
        errors = [f"narrative: too long ({len(text)} chars, max {limit})"]
        result = {"narrative": text}
        resolved = BaseRunner._auto_trim(result, errors)
        assert resolved is True
        assert result["narrative"] == prefix.rstrip()

    def test_hard_truncates_to_limit_when_no_sentence_boundary(self):
        result = {"narrative": "x" * 45}
        errors = ["narrative: too long (45 chars, max 40)"]
        resolved = BaseRunner._auto_trim(result, errors)
        assert resolved is True
        assert result["narrative"] == "x" * 40

    def test_does_not_trim_a_gross_overshoot(self):
        result = {"narrative": "x" * 100}
        errors = ["narrative: too long (100 chars, max 40)"]
        resolved = BaseRunner._auto_trim(result, errors)
        assert resolved is False
        assert result["narrative"] == "x" * 100

    def test_truncates_array_by_importance_not_position(self):
        result = {
            "key_factors": [
                {"factor": "a", "importance": "low"},
                {"factor": "b", "importance": "high"},
                {"factor": "c", "importance": "medium"},
            ]
        }
        errors = ["key_factors: need <=2 items, got 3"]
        resolved = BaseRunner._auto_trim(result, errors)
        assert resolved is True
        kept = [f["factor"] for f in result["key_factors"]]
        assert kept == ["b", "c"]  # the low-importance entry is dropped, order preserved

    def test_extracts_an_unambiguous_percentage_band(self):
        result = {"position_sizing_recommendation": "3-5% (adopted from Risk Advisor)"}
        errors = ["position_sizing_recommendation: must be a percentage band"]
        resolved = BaseRunner._auto_trim(result, errors)
        assert resolved is True
        assert result["position_sizing_recommendation"] == "3-5%"

    def test_returns_false_when_nothing_matches(self):
        result = {"a": 1}
        resolved = BaseRunner._auto_trim(result, ["unrelated: some other error"])
        assert resolved is False


# ---------- timing_summary ----------


class TestTimingSummary:
    def test_empty_call_log(self):
        runner = BaseRunner()
        assert runner.timing_summary() == {"calls": 0}

    @pytest.mark.asyncio
    async def test_aggregates_across_calls(self):
        session = FakeSession(
            [
                FakeResponse(200, _ollama_chat_response({"narrative": "short"})),
                FakeResponse(200, _ollama_chat_response({"narrative": "long enough now"})),
            ]
        )
        runner = BaseRunner()
        runner.session = session

        def validator(result):
            if result["narrative"] == "short":
                return False, ["narrative: too short (5 chars, min 10)"]
            return True, []

        await runner.call_with_validation("sys", "usr", validator)
        summary = runner.timing_summary()
        assert summary["calls"] == 2
        assert summary["retry_calls"] == 1
        assert summary["total_llm_s"] == 2.0
