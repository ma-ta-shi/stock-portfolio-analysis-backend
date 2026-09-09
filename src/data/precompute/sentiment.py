"""Per-article sentiment scoring for the Sentiment Analyst agent (Pass 1),
ClickUp 86ban0wf4.

Makes an LLM call - the ticket describes this as "one of the two documented
LLM exceptions" in the precompute layer (that framing is the ticket's own
wording, not CLAUDE.md's - CLAUDE.md has no "exception" language about
precompute LLM calls at all; the second exception isn't confirmed here
either). No caching, no DB access - pure aside from the LLM call itself.

Consumed by whatever prompt-building glue reads this into the real
Sentiment Analyst payload ("Agent Prompts/Current Prompts/Sentiment Analyst
Agent Prompt.md"), confirmed by direct read, not the ticket's own draft:
the payload's news line is `N{num} | {date} | {headline} | source: {source}
| tier: {tier} | sentiment: {sentiment_score_or_none}` - no per-article
summary field anywhere, and no confidence value is rendered either.

Both US and Finnhub-sourced and (once wired up) CA articles are scored
identically now: confirmed live that Finnhub's /company-news has no
sentiment field on the free tier (same gap as Canadian, not a US-only
one), so this module is the only sentiment source for either market.

Caller contract: takes news_id_assignment.assign_news_ids()'s output
directly - does NOT call assign_news_ids() itself. ID assignment happens
once per run, globally, over the widest window any Pass-1 agent uses; if
this module assigned its own IDs on its own narrower window, it would
silently produce different numbering than every other agent sees for the
same article, breaking the "same article, same N{num} everywhere"
guarantee. The caller fetches the widest window once, calls
assign_news_ids() once, and passes each agent (including this one) its own
window-filtered slice of that single global result.
"""

import asyncio
import json

import aiohttp
import structlog

logger = structlog.get_logger(__name__)

_OLLAMA_URL = "http://localhost:11434/api/chat"
_MODEL = "gpt-oss:20b"  # 2026-08-26: switched from qwen3.6:35b-a3b, which doesn't fit in
# this machine's VRAM (61%/39% GPU/CPU split per `ollama ps`, a live-measured 209s per
# call) - see docs/decision-log.md. gpt-oss:20b runs 100% GPU, confirmed live: 5.4s total
# for a cold-loaded call (4.9s of that is one-time model load), 58ms eval for 7 tokens.
_THINK = "low"  # gpt-oss's reasoning-effort parameter is a string ("low"/"medium"/"high"),
# not the boolean qwen used - confirmed live, a bool would silently be the wrong shape here.
_TIMEOUT = aiohttp.ClientTimeout(total=60)  # generous relative to the ~5s real cold-start
# cost confirmed live for this model - kept wide rather than tight, since a full pipeline
# run's first call still pays a one-time model-load cost.
_MAX_CONCURRENT = 10  # matches the ticket's own original design parameter - a single
# local Ollama instance still serializes GPU-bound inference across concurrent requests,
# even though each individual call is now fast enough that queuing isn't the bottleneck
# it was with the model this replaced.
_NUM_CTX = 8192  # Ollama silently truncates context to its own small default if this
# isn't set explicitly - confirmed live (docs/technical/ollama-num-ctx-finding.md) that
# a truncated call doesn't error, it makes gpt-oss fabricate a confident, wrong answer
# instead of reading the (missing, truncated-away) source text. This call's real input
# (headline + article["text"], which is news_id_assignment.py's article.get("summary",
# ""), a short snippet, never a full article body) is nowhere near this budget - 8192
# is headroom, not a tight fit, and stays on the flat part of the latency-vs-num_ctx
# curve (confirmed live: cost is flat from 2048 through 32768, only jumps at the
# model's full 131072).
_VALID_LABELS = ("positive", "negative", "neutral")

_FORMAT_SCHEMA = {
    "type": "object",
    "properties": {"sentiment": {"type": "string", "enum": list(_VALID_LABELS)}},
    "required": ["sentiment"],
}


def _prompt(headline: str, text: str) -> str:
    body = f"\n\n{text}" if text else ""
    return (
        "Classify the sentiment of this news headline as positive, negative, or neutral."
        f"\n\nHeadline: {headline}{body}"
    )


async def _score_article(session: aiohttp.ClientSession, headline: str, text: str) -> str | None:
    """Returns "positive"|"negative"|"neutral", or None on any failure - no
    fallback (2026-08-07 no-fallback-chains decision): a genuine scoring
    failure surfaces as missing data, not a silent retry on another model."""
    payload = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": _prompt(headline, text)}],
        "stream": False,
        "think": _THINK,
        "format": _FORMAT_SCHEMA,
        "options": {"num_ctx": _NUM_CTX},
    }
    try:
        async with session.post(_OLLAMA_URL, json=payload, timeout=_TIMEOUT) as response:
            if response.status != 200:
                logger.warning("sentiment_score_http_error", status=response.status)
                return None
            data = await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
        # json.JSONDecodeError is real and reachable here, not just defensive: a 200
        # status with a genuinely truncated/malformed body (partial write, mid-response
        # crash) makes response.json() raise json.JSONDecodeError directly - it's a
        # plain ValueError subclass, not wrapped in aiohttp.ClientError, so it doesn't
        # get caught without being listed explicitly.
        logger.warning("sentiment_score_request_failed", error=str(exc))
        return None

    try:
        content = data["message"]["content"]
        parsed = json.loads(content)
        sentiment = parsed["sentiment"]
    except (KeyError, TypeError, ValueError):
        logger.warning("sentiment_score_malformed_response")
        return None

    # Belt-and-suspenders: Ollama's `format` param is genuinely schema-constrained
    # (verified live), not just a prompt hint - but still validate rather than
    # trust it blindly, matching the "missing over wrong" posture used throughout.
    if sentiment not in _VALID_LABELS:
        logger.warning("sentiment_score_unexpected_label", label=sentiment)
        return None
    return sentiment


async def summarize_news(id_assigned_articles: list[dict]) -> dict:
    """Scores each already-ID-assigned article's sentiment via a local LLM
    call. Returns {"articles": [...], "sentiment_source": ...} - the
    caller distributes these two keys across DataBundle.news_with_sentiment
    and DataBundle.sentiment_source respectively."""
    if not id_assigned_articles:
        return {"articles": [], "sentiment_source": None}

    semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

    async def _score_bounded(session: aiohttp.ClientSession, article: dict) -> str | None:
        async with semaphore:
            return await _score_article(session, article["headline"], article["text"])

    async with aiohttp.ClientSession() as session:
        sentiments = await asyncio.gather(
            *(_score_bounded(session, article) for article in id_assigned_articles)
        )

    articles = [
        {
            "id": article["id"],
            "date": article["date"],
            "headline": article["headline"],
            "source": article["source"],
            "quality_tier": article["quality_tier"],
            "sentiment": sentiment,
        }
        for article, sentiment in zip(id_assigned_articles, sentiments)
    ]
    return {"articles": articles, "sentiment_source": "local_llm"}
