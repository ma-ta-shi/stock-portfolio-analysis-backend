import asyncio
import importlib
import itertools
import json

import pytest

import agents.capture as capture_module
import data.precompute.sentiment as sentiment_module
from agents.capture import CaptureContext
from data.precompute.sentiment import _score_article, summarize_news


def test_ollama_url_reads_ollama_host_env_var(monkeypatch):
    """Real bug, confirmed live 2026-09-23: this module used to hardcode
    localhost:11434 instead of reading OLLAMA_HOST like agents/base.py
    already does -- proven live via a test pointing OLLAMA_HOST at an
    unreachable port to simulate an Ollama outage, where this module kept
    silently hitting the real Ollama instance underneath it. A bare
    "the module is internally consistent with its own constant" check
    (agents/test_base.py's own convention for this same class of
    module-level constant) would pass trivially even against the original
    hardcoded bug -- reload with a real env var override is what actually
    proves the value is configurable, not just self-consistent."""
    monkeypatch.setenv("OLLAMA_HOST", "http://example-ollama-host:9999")
    try:
        importlib.reload(sentiment_module)
        assert sentiment_module._OLLAMA_URL == "http://example-ollama-host:9999/api/chat"
    finally:
        importlib.reload(sentiment_module)  # restore the real default for later tests


class FakeResponse:
    def __init__(
        self,
        status: int,
        json_data=None,
        raise_timeout: bool = False,
        raise_outer_decode_error: bool = False,
    ) -> None:
        self.status = status
        self._json_data = json_data
        self._raise_timeout = raise_timeout
        self._raise_outer_decode_error = raise_outer_decode_error

    async def __aenter__(self) -> "FakeResponse":
        if self._raise_timeout:
            raise asyncio.TimeoutError()
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def json(self):
        if self._raise_outer_decode_error:
            # Real aiohttp behavior on a truncated/malformed body: response.json()
            # calls json.loads() on the raw text, which raises json.JSONDecodeError
            # directly - not wrapped in an aiohttp-specific exception. A mock that
            # just returns a pre-built dict (as this class otherwise does) can never
            # reproduce this failure mode, so it needs its own explicit path.
            json.loads("{not valid json")
        return self._json_data


class FakeSession:
    """Returns the same FakeResponse for every call. No sequential/list-of-
    responses mode (unlike test_finnhub.py's FakeSession) - this module has
    no retry logic to exercise (explicit no-fallback design), so nothing
    here ever needs more than one canned response per session."""

    def __init__(self, response: FakeResponse) -> None:
        self._response = response
        self.calls: list[dict] = []

    def post(self, url: str, json: dict, timeout=None) -> FakeResponse:
        self.calls.append({"url": url, "json": json})
        return self._response


def _ollama_response(sentiment: str) -> dict:
    return {"message": {"content": json.dumps({"sentiment": sentiment})}}


def _article(id_="N1", **overrides) -> dict:
    article = {
        "id": id_,
        "date": "2026-08-20",
        "headline": "Company reports record profits",
        "source": "Reuters",
        "quality_tier": "primary",
        "text": "Full article body.",
        "url": "https://example.com/a1",
    }
    article.update(overrides)
    return article


# ---------- _score_article ----------


@pytest.mark.asyncio
async def test_score_article_success():
    session = FakeSession(FakeResponse(200, _ollama_response("positive")))
    result = await _score_article(session, "headline", "text")
    assert result == "positive"


@pytest.mark.asyncio
async def test_score_article_sends_expected_payload_shape():
    session = FakeSession(FakeResponse(200, _ollama_response("neutral")))
    await _score_article(session, "Company X reports earnings", "Full body text")
    call = session.calls[0]
    assert call["json"]["model"] == "gpt-oss:20b"
    assert call["json"]["think"] == "low"
    assert call["json"]["stream"] is False
    assert "format" in call["json"]
    assert "Company X reports earnings" in call["json"]["messages"][0]["content"]
    # Ollama silently truncates to its own small default context window if this
    # isn't set explicitly, and a truncated call doesn't error - it makes gpt-oss
    # fabricate a confident, wrong answer instead (docs/technical/ollama-num-ctx-finding.md).
    assert call["json"]["options"]["num_ctx"] == 8192


@pytest.mark.asyncio
async def test_score_article_none_on_non_200_status():
    session = FakeSession(FakeResponse(500, None))
    result = await _score_article(session, "headline", "text")
    assert result is None


@pytest.mark.asyncio
async def test_score_article_none_on_timeout():
    session = FakeSession(FakeResponse(200, None, raise_timeout=True))
    result = await _score_article(session, "headline", "text")
    assert result is None


@pytest.mark.asyncio
async def test_score_article_none_on_malformed_inner_content_json():
    """The LLM's "content" field is itself a JSON-encoded string per the
    requested format - this is malformed at that inner layer."""
    session = FakeSession(FakeResponse(200, {"message": {"content": "not valid json"}}))
    result = await _score_article(session, "headline", "text")
    assert result is None


@pytest.mark.asyncio
async def test_score_article_none_on_malformed_outer_response_body():
    """Regression: a 200 status with a genuinely truncated/malformed
    outer response body makes response.json() raise json.JSONDecodeError
    directly (not wrapped in aiohttp.ClientError) - this must degrade to
    None like every other failure mode, not propagate and crash the batch."""
    session = FakeSession(FakeResponse(200, None, raise_outer_decode_error=True))
    result = await _score_article(session, "headline", "text")
    assert result is None


@pytest.mark.asyncio
async def test_score_article_none_on_missing_sentiment_key():
    session = FakeSession(FakeResponse(200, {"message": {"content": json.dumps({})}}))
    result = await _score_article(session, "headline", "text")
    assert result is None


@pytest.mark.asyncio
async def test_score_article_none_on_unexpected_label():
    """Belt-and-suspenders: even though Ollama's format param is schema-
    constrained, don't trust it blindly."""
    session = FakeSession(FakeResponse(200, _ollama_response("very positive")))
    result = await _score_article(session, "headline", "text")
    assert result is None


@pytest.mark.asyncio
async def test_score_article_none_on_missing_message_shape():
    session = FakeSession(FakeResponse(200, {"unexpected": "shape"}))
    result = await _score_article(session, "headline", "text")
    assert result is None


@pytest.mark.asyncio
async def test_score_article_none_on_top_level_json_null_body():
    """Regression: a 200 status whose body is the bare JSON value `null`
    (not `{}`) is still valid JSON -- response.json() succeeds and returns
    Python None. A rewrite done for capture's sake (86bbwachy Phase 3)
    moved message/content extraction outside the try/except that used to
    catch the TypeError None["message"] raises, which would have let this
    crash the whole summarize_news() batch instead of degrading to a
    single None score. Caught on review, not by this test failing first --
    written after the fix to lock in the real failure mode found."""
    session = FakeSession(FakeResponse(200, None))
    result = await _score_article(session, "headline", "text")
    assert result is None


# ---------- capture (86bbwachy Phase 3) ----------


def _capture(**overrides) -> CaptureContext:
    defaults = dict(
        run_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", ticker="AAPL", seq_counter=itertools.count()
    )
    return CaptureContext(**{**defaults, **overrides})


@pytest.mark.asyncio
async def test_score_article_capture_none_by_default_writes_nothing(tmp_path, monkeypatch):
    """No capture kwarg at all -- every existing caller/test above this
    point in the file keeps working exactly as before, no new files."""
    monkeypatch.setattr(capture_module, "RUNS_DIR", str(tmp_path))
    session = FakeSession(FakeResponse(200, _ollama_response("positive")))
    result = await _score_article(session, "headline", "text")
    assert result == "positive"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_score_article_captures_a_real_success(tmp_path, monkeypatch):
    monkeypatch.setattr(capture_module, "RUNS_DIR", str(tmp_path))
    session = FakeSession(FakeResponse(200, _ollama_response("positive")))
    capture = _capture()

    result = await _score_article(session, "headline", "text", capture=capture)

    assert result == "positive"
    assert len(capture.call_log) == 1
    entry = capture.call_log[0]
    assert entry["call_site"] == "precompute:sentiment"
    assert entry["parsed_ok"] is True
    assert entry["parse_error"] is None
    assert ":" not in entry["prompt_path"]
    assert (tmp_path / entry["prompt_path"]).exists()
    assert (tmp_path / entry["response_path"]).exists()


@pytest.mark.asyncio
async def test_score_article_captures_a_malformed_response_as_parsed_ok_false(tmp_path, monkeypatch):
    """A capture row still gets written for a failure that got past the
    HTTP layer -- the same "one row per round trip, success or failure"
    contract agents/base.py's own capture already follows."""
    monkeypatch.setattr(capture_module, "RUNS_DIR", str(tmp_path))
    session = FakeSession(FakeResponse(200, {"message": {"content": "not valid json"}}))
    capture = _capture()

    result = await _score_article(session, "headline", "text", capture=capture)

    assert result is None
    assert len(capture.call_log) == 1
    entry = capture.call_log[0]
    assert entry["parsed_ok"] is False
    assert entry["parse_error"] is not None


@pytest.mark.asyncio
async def test_score_article_captures_no_http_response_at_all(tmp_path, monkeypatch):
    """A pure connection-level failure (timeout, no response body ever
    received) must NOT produce a capture row -- there is nothing real to
    write as a response artifact, same boundary agents/base.py's own
    call_model draws (it never reaches its own capture point either when
    raise_for_status() raises before a body exists)."""
    monkeypatch.setattr(capture_module, "RUNS_DIR", str(tmp_path))
    session = FakeSession(FakeResponse(200, None, raise_timeout=True))
    capture = _capture()

    result = await _score_article(session, "headline", "text", capture=capture)

    assert result is None
    assert capture.call_log == []


@pytest.mark.asyncio
async def test_score_article_captures_a_top_level_json_null_body_without_raising(tmp_path, monkeypatch):
    """Same real regression as test_score_article_none_on_top_level_json_null_body
    above, but through the capture path specifically -- record_call's own
    response_body.get(...) calls needed the identical isinstance guard,
    one layer deeper (agents/capture.py's own fix)."""
    monkeypatch.setattr(capture_module, "RUNS_DIR", str(tmp_path))
    session = FakeSession(FakeResponse(200, None))
    capture = _capture()

    result = await _score_article(session, "headline", "text", capture=capture)

    assert result is None
    assert len(capture.call_log) == 1
    entry = capture.call_log[0]
    assert entry["parsed_ok"] is False
    assert entry["total_duration_s"] is None
    assert entry["prompt_eval_count"] is None


class _FakeClientSessionCM:
    """summarize_news's own `async with aiohttp.ClientSession() as session:`
    needs something that supports the async-context-manager protocol --
    FakeSession itself deliberately doesn't (every other test in this file
    swaps out _score_article entirely, so the real aiohttp.ClientSession()
    is constructed for real and never actually used). This wraps a
    FakeSession so this one integration test can fake the whole chain."""

    def __init__(self, session: "FakeSession") -> None:
        self._session = session

    async def __aenter__(self) -> "FakeSession":
        return self._session

    async def __aexit__(self, *exc) -> bool:
        return False


@pytest.mark.asyncio
async def test_summarize_news_threads_capture_through_to_every_article(tmp_path, monkeypatch):
    """Integration-ish: exercises the real _score_article (not a
    monkeypatched fake, unlike this file's other summarize_news tests) so
    the actual threading from summarize_news -> _score_bounded ->
    _score_article is what's under test, not just each layer in
    isolation."""
    monkeypatch.setattr(capture_module, "RUNS_DIR", str(tmp_path))
    fake_session = FakeSession(FakeResponse(200, _ollama_response("neutral")))
    monkeypatch.setattr(
        sentiment_module.aiohttp, "ClientSession", lambda: _FakeClientSessionCM(fake_session)
    )
    capture = _capture()
    articles = [_article(id_="N1"), _article(id_="N2")]

    result = await summarize_news(articles, capture=capture)

    assert [a["sentiment"] for a in result["articles"]] == ["neutral", "neutral"]
    assert len(capture.call_log) == 2
    assert {entry["seq"] for entry in capture.call_log} == {0, 1}


# ---------- summarize_news ----------


@pytest.mark.asyncio
async def test_summarize_news_empty_input_returns_none_source():
    result = await summarize_news([])
    assert result == {"articles": [], "sentiment_source": None}


@pytest.mark.asyncio
async def test_summarize_news_scores_each_article(monkeypatch):
    responses = iter(["positive", "negative"])

    async def fake_score(session, headline, text, **kwargs):
        return next(responses)

    monkeypatch.setattr("data.precompute.sentiment._score_article", fake_score)

    articles = [_article(id_="N1"), _article(id_="N2")]
    result = await summarize_news(articles)

    assert result["sentiment_source"] == "local_llm"
    assert [a["sentiment"] for a in result["articles"]] == ["positive", "negative"]


@pytest.mark.asyncio
async def test_summarize_news_preserves_order(monkeypatch):
    async def fake_score(session, headline, text, **kwargs):
        return "neutral"

    monkeypatch.setattr("data.precompute.sentiment._score_article", fake_score)

    articles = [_article(id_="N1"), _article(id_="N2"), _article(id_="N3")]
    result = await summarize_news(articles)

    assert [a["id"] for a in result["articles"]] == ["N1", "N2", "N3"]


@pytest.mark.asyncio
async def test_summarize_news_final_shape_drops_text_and_url(monkeypatch):
    async def fake_score(session, headline, text, **kwargs):
        return "positive"

    monkeypatch.setattr("data.precompute.sentiment._score_article", fake_score)

    result = await summarize_news([_article()])
    article = result["articles"][0]
    assert set(article.keys()) == {"id", "date", "headline", "source", "quality_tier", "sentiment"}


@pytest.mark.asyncio
async def test_summarize_news_source_is_local_llm_even_if_every_score_fails(monkeypatch):
    """sentiment_source reflects the methodology attempted, not the
    success rate - it's set whenever there were articles to score, even
    if every individual call failed."""

    async def fake_score(session, headline, text, **kwargs):
        return None

    monkeypatch.setattr("data.precompute.sentiment._score_article", fake_score)

    result = await summarize_news([_article()])
    assert result["sentiment_source"] == "local_llm"
    assert result["articles"][0]["sentiment"] is None


@pytest.mark.asyncio
async def test_summarize_news_does_not_call_assign_news_ids(monkeypatch):
    """Regression: an earlier draft of this design called
    news_id_assignment.assign_news_ids() internally, which would silently
    produce locally-scoped IDs instead of the globally-assigned ones every
    other Pass-1 agent sees for the same article."""
    called = []
    monkeypatch.setattr(
        "data.precompute.news_id_assignment.assign_news_ids",
        lambda articles: called.append(articles) or articles,
    )

    async def fake_score(session, headline, text, **kwargs):
        return "neutral"

    monkeypatch.setattr("data.precompute.sentiment._score_article", fake_score)

    await summarize_news([_article()])
    assert called == []
