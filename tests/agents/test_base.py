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


def _ollama_chat_response(content: dict, thinking: str = "") -> dict:
    return {
        "message": {"content": json.dumps(content), "thinking": thinking},
        "total_duration": 1_000_000_000,  # 1s in nanoseconds
        "prompt_eval_count": 100,
        "eval_count": 50,
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
