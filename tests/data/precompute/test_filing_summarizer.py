import asyncio
import json

import pytest

from data.precompute.filing_summarizer import (
    _MAX_INPUT_CHARS,
    _TOKEN_BUDGET,
    _truncate_to_budget,
    summarize_filing_section,
)
from data.schemas.common import FilingDigest


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
            json.loads("{not valid json")
        return self._json_data


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self._response = response
        self.calls: list[dict] = []

    def post(self, url: str, json: dict, timeout=None) -> FakeResponse:
        self.calls.append({"url": url, "json": json})
        return self._response


def _generate_response(text: str, eval_count: int) -> dict:
    return {"response": text, "eval_count": eval_count, "done_reason": "stop"}


# ---------- _truncate_to_budget ----------


def test_truncate_to_budget_passthrough_when_short():
    text = "One short sentence."
    assert _truncate_to_budget(text) == text


def test_truncate_to_budget_cuts_on_sentence_boundary():
    # Two sentences, the second pushing well past the ~1000-char budget.
    first = "The bank reported record net income of five billion dollars this quarter."
    second = " " + ("Additional narrative detail about every operating segment. " * 40)
    result = _truncate_to_budget(first + second)
    assert result.endswith(".")
    assert len(result) <= _TOKEN_BUDGET * 4
    assert result.startswith("The bank reported record net income")
    # cut landed on a real sentence end, not mid-word
    assert not result.endswith("Additional")


def test_truncate_to_budget_hard_cut_when_no_sentence_boundary():
    text = "word " * 500  # 2500 chars, no sentence punctuation
    result = _truncate_to_budget(text)
    assert len(result) <= _TOKEN_BUDGET * 4
    assert result == text[: _TOKEN_BUDGET * 4].rstrip()


# ---------- summarize_filing_section: payload shape ----------


@pytest.mark.asyncio
async def test_sends_expected_payload_shape():
    session = FakeSession(FakeResponse(200, _generate_response("A digest.", 40)))
    await summarize_filing_section(session, "Some filing section text.", "MDA")

    body = session.calls[0]["json"]
    assert body["model"] == "gpt-oss:20b"
    assert body["think"] == "low"
    assert body["stream"] is False
    # num_ctx must be explicit - Ollama silently truncates otherwise and gpt-oss
    # then fabricates (docs/technical/ollama-num-ctx-finding.md).
    assert body["options"]["num_ctx"] == 32768
    assert body["options"]["num_predict"] == 2000


@pytest.mark.asyncio
async def test_prompt_focus_varies_by_section():
    session = FakeSession(FakeResponse(200, _generate_response("A digest.", 40)))
    await summarize_filing_section(session, "text", "Business")
    business_prompt = session.calls[0]["json"]["prompt"]

    session = FakeSession(FakeResponse(200, _generate_response("A digest.", 40)))
    await summarize_filing_section(session, "text", "MDA")
    mda_prompt = session.calls[0]["json"]["prompt"]

    assert "reportable segments" in business_prompt
    assert "Do NOT include revenue, earnings, dividend" in business_prompt
    assert "financial results and outlook" in mda_prompt
    assert "reportable segments" not in mda_prompt


@pytest.mark.asyncio
async def test_input_truncated_to_max_input_chars():
    long_text = "x" * (_MAX_INPUT_CHARS + 5_000)
    session = FakeSession(FakeResponse(200, _generate_response("A digest.", 40)))
    await summarize_filing_section(session, long_text, "MDA")

    prompt = session.calls[0]["json"]["prompt"]
    # the section text is truncated to _MAX_INPUT_CHARS before it reaches the prompt
    assert ("x" * _MAX_INPUT_CHARS) in prompt
    assert ("x" * (_MAX_INPUT_CHARS + 1)) not in prompt


# ---------- summarize_filing_section: happy path + truncation ----------


@pytest.mark.asyncio
async def test_within_budget_response_returned_verbatim():
    session = FakeSession(FakeResponse(200, _generate_response("  A grounded digest.  ", 180)))
    result = await summarize_filing_section(session, "text", "MDA")

    assert isinstance(result, FilingDigest)
    assert result.section == "MDA"
    assert result.content == "A grounded digest."
    assert result.token_count == 180


@pytest.mark.asyncio
async def test_over_budget_response_truncated_and_token_count_rederived():
    sentences = "This is a grounded sentence with real figures like $5 billion. " * 20
    session = FakeSession(FakeResponse(200, _generate_response(sentences, 420)))
    result = await summarize_filing_section(session, "text", "MDA")

    assert result is not None
    assert result.content.endswith(".")
    assert len(result.content) <= _TOKEN_BUDGET * 4
    assert result.content  # non-empty
    # re-derived from the truncated length, not the stale 420 eval_count
    assert result.token_count == max(1, len(result.content) // 4)
    assert result.token_count != 420


# ---------- summarize_filing_section: failure modes ----------


@pytest.mark.asyncio
async def test_empty_section_text_returns_none_without_calling_ollama():
    session = FakeSession(FakeResponse(200, _generate_response("unused", 10)))
    assert await summarize_filing_section(session, "   ", "MDA") is None
    assert session.calls == []


@pytest.mark.asyncio
async def test_non_200_status_returns_none():
    session = FakeSession(FakeResponse(500, None))
    assert await summarize_filing_section(session, "text", "MDA") is None


@pytest.mark.asyncio
async def test_timeout_returns_none():
    session = FakeSession(FakeResponse(200, None, raise_timeout=True))
    assert await summarize_filing_section(session, "text", "MDA") is None


@pytest.mark.asyncio
async def test_malformed_outer_body_returns_none():
    session = FakeSession(FakeResponse(200, None, raise_outer_decode_error=True))
    assert await summarize_filing_section(session, "text", "MDA") is None


@pytest.mark.asyncio
async def test_missing_response_key_returns_none():
    session = FakeSession(FakeResponse(200, {"eval_count": 40, "done_reason": "stop"}))
    assert await summarize_filing_section(session, "text", "MDA") is None


@pytest.mark.asyncio
async def test_missing_eval_count_returns_none():
    session = FakeSession(FakeResponse(200, {"response": "a digest", "done_reason": "stop"}))
    assert await summarize_filing_section(session, "text", "MDA") is None


@pytest.mark.asyncio
async def test_blank_model_output_returns_none():
    session = FakeSession(FakeResponse(200, _generate_response("   ", 3)))
    assert await summarize_filing_section(session, "text", "MDA") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output",
    [
        "I'm sorry, but the excerpt you provided doesn't contain the substantive sections.",
        "The provided text only contains table-of-contents listings and boilerplate.",
        "I cannot summarize this because the input does not appear to include results.",
        "If you can share the detailed narrative sections, I'll write the summary.",
    ],
)
async def test_refusal_output_returns_none(output):
    session = FakeSession(FakeResponse(200, _generate_response(output, 60)))
    assert await summarize_filing_section(session, "some filing text", "Business") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output",
    [
        "Total revenue was $17,? billion, up from the prior year.",
        "The bank reported net income of $5,? million this quarter.",
        "Revenue reached $XX million with strong growth across segments.",
    ],
)
async def test_placeholder_figures_output_returns_none(output):
    session = FakeSession(FakeResponse(200, _generate_response(output, 60)))
    assert await summarize_filing_section(session, "some filing text", "MDA") is None


@pytest.mark.asyncio
async def test_stray_markdown_bold_is_stripped():
    out = "Teck's key assets include the **Highland Valley Copper Mine** and **Red Dog Mine**."
    session = FakeSession(FakeResponse(200, _generate_response(out, 40)))
    result = await summarize_filing_section(session, "some filing text", "Business")
    assert result is not None
    assert "**" not in result.content
    assert "Highland Valley Copper Mine" in result.content


@pytest.mark.asyncio
async def test_grounded_digest_with_real_figures_is_kept():
    good = (
        "Suncor reported adjusted funds from operations of $5.329 billion ($4.52 per share), "
        "up from $2.689 billion a year earlier, and returned $1.756 billion to shareholders."
    )
    session = FakeSession(FakeResponse(200, _generate_response(good, 60)))
    result = await summarize_filing_section(session, "some filing text", "MDA")
    assert result is not None
    assert result.content == good
