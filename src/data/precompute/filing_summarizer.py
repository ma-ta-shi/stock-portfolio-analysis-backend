"""Filing-section digest generation for research_sources.py (ClickUp
86bbu1j02, feeding 86ban0x1u).

Takes one filing section's raw text - a 10-K Item 1 / Item 7 for US filers,
or a Canadian cross-listed filer's quarterly 6-K earnings exhibit / annual
40-F AIF via providers/ca_crosslisting.py - and returns a <=250-token
FilingDigest from a single local LLM call.

Mirrors precompute/sentiment.py: aiohttp to a local Ollama endpoint, no
retry and no model fallback (2026-08-07 no-fallback-chains decision), any
failure returns None so the digest is simply omitted rather than faked.

`num_ctx` is set explicitly on every call. Without it Ollama silently
truncates input past a small default window - no error, no log - and
gpt-oss then fabricates a confident wrong answer from pretraining instead
of failing. See docs/technical/ollama-num-ctx-finding.md.

gpt-oss does not reliably self-limit output length: a "150 words"
instruction returned ~450 tokens in live checks, "120 words" ~270. So the
prompt aims low and this module always trims an over-budget draft to the
last full sentence within the token budget - that trim is a normal path,
not a rare safety net.

No caching here: summarize_filing_section is the single-section primitive
(like sentiment.py's _score_article). Batching and the
(cik, accession, section_id) digest cache are research_sources.py's and
86baq4zg7's job.
"""

import asyncio
import json
import re
from typing import Literal

import aiohttp
import structlog

from data.schemas.common import FilingDigest

logger = structlog.get_logger(__name__)

# The model sometimes answers that the excerpt is all boilerplate / table of
# contents and asks for the real section, or emits placeholder figures
# ("$17,? billion") when it has no numbers to work from. Either way the "digest"
# is worthless and worse than a missing one - return None so the caller omits it.
_NON_ANSWER_RE = re.compile(
    r"\bi(?:'m| am) sorry\b|\bi (?:can(?:not|'t)|am unable|couldn't)\b"
    r"|\b(?:does|do)(?: not|n't) (?:contain|include|appear|describe|provide|discuss|mention)\b"
    r"|\bno (?:substantive|meaningful|actual|real) "
    r"|\b(?:the (?:excerpt|provided text|input|filing text)|this excerpt) (?:you |)"
    r"(?:provided |)(?:doesn't|does not|only|is|appears|consists|seems)"
    r"|\bif you can (?:share|provide)\b|\bplease (?:share|provide) \b"
    r"|\btable[- ]of[- ]contents (?:listings?|entries)\b",
    re.I,
)
_PLACEHOLDER_RE = re.compile(r"[$€£]\s?[\d,]*[?xX—]{1,}\b|\b\d[?]\s|\b\d+,[?]")

_OLLAMA_URL = "http://localhost:11434/api/generate"  # /generate: one prompt, no chat turns
_MODEL = "gpt-oss:20b"
_THINK = "low"  # gpt-oss reasoning-effort is a string ("low"/"medium"/"high"), not the
# boolean qwen used - a bool would silently be the wrong shape here.
_MAX_INPUT_CHARS = 30_000  # ~7.5k tokens. A 20k-char prefix already covered results +
# outlook + full guidance in live checks (NTR's quarterly MD&A); 30k adds margin. More
# than this is input a 250-token digest doesn't need and just adds latency. The caller
# (ca_crosslisting._find_mda_source) hands over text that already starts at real content.
_NUM_CTX = 32_768  # comfortably covers _MAX_INPUT_CHARS + thinking + _NUM_PREDICT, and
# sits on the flat part of the latency-vs-num_ctx curve (ollama-num-ctx-finding.md).
_NUM_PREDICT = 2_000  # generous on purpose: a slightly-over-budget draft should finish
# so it can be trimmed cleanly, not get cut mid-sentence by a tight default.
_TOKEN_BUDGET = 250
_TIMEOUT = aiohttp.ClientTimeout(total=90)  # ~7-15s observed warm for a 20-30k-char call

# The MD&A prompt asks for verbatim figures; the Business prompt deliberately
# does NOT - an AIF business description carries few current financials, and asking
# for numbers made the model invent them (GFL: a fabricated "$2,045 million in
# revenue" that appears nowhere in the source). Financials for the Business side
# come from the fundamentals precompute, not this digest.
_MDA_PROMPT = (
    "Summarize the following filing text about the company's quarterly financial "
    "results and outlook in about 150 words, one paragraph. Preserve verbatim: "
    "specific numbers, percentages, product names, and direct management statements. "
    "Do not paraphrase figures, and do not state any figure that is not in the text. "
    "Omit boilerplate (legal disclaimers, forward-looking-statement safe harbors, "
    "tables of contents). If the input has formatting noise from document conversion, "
    "ignore it.\n\n"
)
_BUSINESS_PROMPT = (
    "Summarize what this company does, from the following filing text, in about 150 "
    "words, one paragraph: its business, principal operations, reportable segments, "
    "key products or properties, and geographic markets. Preserve verbatim: segment "
    "names, product names, facility/mine/property names, and place names. Do NOT "
    "include revenue, earnings, dividend, or other financial figures - if the text "
    "does not describe the business itself, say so briefly. Ignore formatting noise "
    "from document conversion.\n\n"
)


def _prompt(section_text: str, section: Literal["Business", "MDA"]) -> str:
    lead = _BUSINESS_PROMPT if section == "Business" else _MDA_PROMPT
    return f"{lead}{section_text}"


def _truncate_to_budget(text: str) -> str:
    """Cut `text` to the last sentence-ending punctuation at or before
    _TOKEN_BUDGET * 4 characters (~250 tokens in gpt-oss's tokenizer). Falls
    back to a hard character cut if no sentence boundary is in range."""
    limit = _TOKEN_BUDGET * 4
    if len(text) <= limit:
        return text
    window = text[:limit]
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if cut == -1:
        cut = window.rfind(".")
    if cut == -1:
        return window.rstrip()
    return window[: cut + 1].rstrip()


async def summarize_filing_section(
    session: aiohttp.ClientSession,
    section_text: str,
    section: Literal["Business", "MDA"],
) -> FilingDigest | None:
    """One filing section -> a <=250-token FilingDigest, or None on any
    failure (empty input, HTTP error, timeout, malformed response body). No
    retry, no fallback - a genuine failure surfaces as a missing digest,
    which research_sources.py omits."""
    if not section_text or not section_text.strip():
        return None

    payload = {
        "model": _MODEL,
        "prompt": _prompt(section_text[:_MAX_INPUT_CHARS], section),
        "stream": False,
        "think": _THINK,
        "options": {"num_ctx": _NUM_CTX, "num_predict": _NUM_PREDICT},
    }
    try:
        async with session.post(_OLLAMA_URL, json=payload, timeout=_TIMEOUT) as response:
            if response.status != 200:
                logger.warning("filing_summary_http_error", status=response.status, section=section)
                return None
            data = await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
        # json.JSONDecodeError is reachable, not just defensive: a 200 with a
        # truncated/malformed body makes response.json() raise it directly (a plain
        # ValueError subclass), not wrapped in aiohttp.ClientError - same reasoning
        # as sentiment.py's _score_article.
        logger.warning("filing_summary_request_failed", error=str(exc), section=section)
        return None

    try:
        digest = data["response"].strip()
        eval_count = int(data["eval_count"])
    except (KeyError, TypeError, ValueError, AttributeError):
        logger.warning("filing_summary_malformed_response", section=section)
        return None

    if not digest:
        logger.warning("filing_summary_empty_digest", section=section)
        return None

    if _NON_ANSWER_RE.search(digest) or _PLACEHOLDER_RE.search(digest):
        # the model refused (excerpt was all boilerplate) or fabricated placeholder
        # figures - a wrong digest poisons the evidence base, so omit it
        logger.warning("filing_summary_non_answer", section=section)
        return None

    digest = digest.replace("**", "")  # strip stray markdown bold the model sometimes adds

    if eval_count > _TOKEN_BUDGET:
        digest = _truncate_to_budget(digest)
        token_count = max(1, len(digest) // 4)
    else:
        token_count = eval_count

    return FilingDigest(section=section, content=digest, token_count=token_count)
