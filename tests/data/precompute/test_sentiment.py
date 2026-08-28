import asyncio
import json

import pytest

from data.precompute.sentiment import _score_article, summarize_news


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


# ---------- summarize_news ----------


@pytest.mark.asyncio
async def test_summarize_news_empty_input_returns_none_source():
    result = await summarize_news([])
    assert result == {"articles": [], "sentiment_source": None}


@pytest.mark.asyncio
async def test_summarize_news_scores_each_article(monkeypatch):
    responses = iter(["positive", "negative"])

    async def fake_score(session, headline, text):
        return next(responses)

    monkeypatch.setattr("data.precompute.sentiment._score_article", fake_score)

    articles = [_article(id_="N1"), _article(id_="N2")]
    result = await summarize_news(articles)

    assert result["sentiment_source"] == "local_llm"
    assert [a["sentiment"] for a in result["articles"]] == ["positive", "negative"]


@pytest.mark.asyncio
async def test_summarize_news_preserves_order(monkeypatch):
    async def fake_score(session, headline, text):
        return "neutral"

    monkeypatch.setattr("data.precompute.sentiment._score_article", fake_score)

    articles = [_article(id_="N1"), _article(id_="N2"), _article(id_="N3")]
    result = await summarize_news(articles)

    assert [a["id"] for a in result["articles"]] == ["N1", "N2", "N3"]


@pytest.mark.asyncio
async def test_summarize_news_final_shape_drops_text_and_url(monkeypatch):
    async def fake_score(session, headline, text):
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

    async def fake_score(session, headline, text):
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

    async def fake_score(session, headline, text):
        return "neutral"

    monkeypatch.setattr("data.precompute.sentiment._score_article", fake_score)

    await summarize_news([_article()])
    assert called == []
