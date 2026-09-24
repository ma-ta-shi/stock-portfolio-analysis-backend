import itertools

import agents.capture as capture_module
from agents.capture import CaptureContext, artifact_dir, map_finish_reason, record_call


class TestMapFinishReason:
    def test_passes_through_a_real_done_reason(self):
        assert map_finish_reason("stop") == "stop"

    def test_empty_content_overrides_any_done_reason(self):
        assert map_finish_reason("stop", empty_content=True) == "empty_content"

    def test_none_done_reason_stays_none_without_empty_content(self):
        assert map_finish_reason(None) is None


class TestArtifactDir:
    def test_uses_run_id_prefix_when_given(self, tmp_path, monkeypatch):
        monkeypatch.setattr(capture_module, "RUNS_DIR", str(tmp_path))
        d = artifact_dir("11111111-2222-3333-4444-555555555555", "AAPL")
        assert d.name == "AAPL_11111111"

    def test_falls_back_to_standalone_with_no_run_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(capture_module, "RUNS_DIR", str(tmp_path))
        d = artifact_dir(None, "AAPL")
        assert d.name == "AAPL_standalone"


class TestRecordCall:
    """86bbwachy Phase 3 -- record_call is the precompute counterpart of
    agents/base.py's own inline capture block inside call_model/
    _call_generate. These tests exercise it directly, the same way
    test_base.py's own TestCallModel exercises that inline block."""

    def _capture(self, tmp_path) -> CaptureContext:
        return CaptureContext(
            run_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            ticker="AAPL",
            seq_counter=itertools.count(),
        )

    def test_writes_real_artifacts_and_appends_a_call_log_entry(self, tmp_path, monkeypatch):
        monkeypatch.setattr(capture_module, "RUNS_DIR", str(tmp_path))
        capture = self._capture(tmp_path)

        record_call(
            capture,
            call_site="precompute:sentiment",
            model="gpt-oss:20b",
            options={"num_ctx": 8192},
            prompt_text="Classify the sentiment of this headline.",
            response_body={
                "total_duration": 5_000_000_000,
                "prompt_eval_count": 40,
                "eval_count": 8,
                "done_reason": "stop",
                "message": {"content": '{"sentiment": "positive"}'},
            },
            thinking_chars=0,
            empty_content=False,
            parsed_ok=True,
        )

        assert len(capture.call_log) == 1
        entry = capture.call_log[0]
        assert entry["call_site"] == "precompute:sentiment"
        assert entry["seq"] == 0
        assert entry["context_tag"] == "analysis"
        assert entry["model"] == "gpt-oss:20b"
        assert entry["options_json"] == {"num_ctx": 8192}
        assert entry["attempt"] == 1
        assert entry["total_duration_s"] == 5.0
        assert entry["prompt_eval_count"] == 40
        assert entry["eval_count"] == 8
        assert entry["thinking_chars"] == 0
        assert entry["finish_reason"] == "stop"
        assert entry["parsed_ok"] is True
        assert entry["parse_error"] is None
        # Same Windows/NTFS colon regression this ticket's own fix covers
        # for agents/base.py's capture -- call_site has a colon, the
        # written path must not.
        assert ":" not in entry["prompt_path"]

        prompt_path = tmp_path / entry["prompt_path"]
        assert prompt_path.read_text() == "Classify the sentiment of this headline."
        assert entry["context_path"] is None  # no structured template input for precompute

    def test_empty_content_maps_finish_reason_regardless_of_done_reason(self, tmp_path, monkeypatch):
        monkeypatch.setattr(capture_module, "RUNS_DIR", str(tmp_path))
        capture = self._capture(tmp_path)

        record_call(
            capture,
            call_site="precompute:filing_mda",
            model="gpt-oss:20b",
            options={"num_ctx": 32768, "num_predict": 2000},
            prompt_text="Summarize.",
            response_body={"total_duration": 1_000_000_000, "done_reason": "stop", "response": ""},
            thinking_chars=12,
            empty_content=True,
            parsed_ok=False,
            parse_error="malformed response",
        )

        entry = capture.call_log[0]
        assert entry["finish_reason"] == "empty_content"
        assert entry["parsed_ok"] is False
        assert entry["parse_error"] == "malformed response"

    def test_missing_total_duration_leaves_latency_fields_none_not_zero(self, tmp_path, monkeypatch):
        """response_body with no total_duration at all (should not happen
        for a real Ollama response, but a malformed/truncated body is
        exactly the case capture exists to record) must not silently
        report 0s -- matches the `is not None` fix already applied to
        _llm_call_rows for the identical falsy-zero risk."""
        monkeypatch.setattr(capture_module, "RUNS_DIR", str(tmp_path))
        capture = self._capture(tmp_path)

        record_call(
            capture,
            call_site="precompute:sentiment",
            model="gpt-oss:20b",
            options={},
            prompt_text="x",
            response_body={},
            thinking_chars=0,
            empty_content=True,
            parsed_ok=False,
            parse_error="malformed response",
        )

        entry = capture.call_log[0]
        assert entry["total_duration_s"] is None
        assert entry["prompt_eval_count"] is None
        assert entry["eval_count"] is None

    def test_non_dict_response_body_does_not_raise(self, tmp_path, monkeypatch):
        """Real regression, found on review: response.json() succeeding
        only guarantees valid JSON, not a dict -- a malformed/unexpected
        Ollama response (top-level null, a bare string, a list) is exactly
        the case this whole module exists to record, not one it can
        assume away. Both sentiment.py's and filing_summarizer.py's own
        callers already got the identical fix for their own local
        extraction; this is the one place inside record_call itself doing
        response_body.get(...) directly."""
        monkeypatch.setattr(capture_module, "RUNS_DIR", str(tmp_path))
        capture = self._capture(tmp_path)

        record_call(
            capture,
            call_site="precompute:sentiment",
            model="gpt-oss:20b",
            options={},
            prompt_text="x",
            response_body=None,
            thinking_chars=0,
            empty_content=True,
            parsed_ok=False,
            parse_error="malformed response",
        )

        entry = capture.call_log[0]
        assert entry["total_duration_s"] is None
        assert entry["prompt_eval_count"] is None
        assert entry["eval_count"] is None
        assert entry["finish_reason"] == "empty_content"
        # the raw (weird) response is still written to disk for real
        # debugging, not silently swallowed by the same guard
        response_path = tmp_path / entry["response_path"]
        assert response_path.read_text() == "null"

    def test_seq_increments_across_calls_via_the_shared_counter(self, tmp_path, monkeypatch):
        monkeypatch.setattr(capture_module, "RUNS_DIR", str(tmp_path))
        capture = self._capture(tmp_path)

        for _ in range(3):
            record_call(
                capture,
                call_site="precompute:sentiment",
                model="gpt-oss:20b",
                options={},
                prompt_text="x",
                response_body={"done_reason": "stop"},
                thinking_chars=0,
                empty_content=False,
                parsed_ok=True,
            )

        assert [entry["seq"] for entry in capture.call_log] == [0, 1, 2]
