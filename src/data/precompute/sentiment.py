"""Sentiment scoring of news headlines for the Sentiment Analyst agent (Pass 1),
ClickUp 86ban0wf4. Scored in batches of ~30 headlines per model call (BB-023; it was one call
per article); `_score_article` remains as the single-item path for anything a batch missed.

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

US (Finnhub-sourced) and CA (openbb-tmx-sourced, wired 86bbqh23f) articles
are scored identically: confirmed live that Finnhub's /company-news has no
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
import os

import aiohttp
import structlog

# agents/capture.py, not agents/base.py -- the OLLAMA_HOST/_MODEL constants
# just below are still deliberately duplicated rather than imported (see
# their own comment). capture.py is different: 86bbwachy's own plan always
# scoped it as the one shared, DB-agnostic primitive both agents/base.py's
# retry-wrapped calls AND precompute's one-shot calls (this module,
# filing_summarizer.py) route through -- a real, intended dependency, not
# an accidental crossing of the boundary the comment below is about.
from agents.capture import CaptureContext, record_call
from data.degradation import LLM_REQUEST_FAILED, SENTIMENT_UNSCORED
from data.precompute.headline_rules import rule_label
from data.degradation import report as report_degradation

logger = structlog.get_logger(__name__)

# Real bug, confirmed live 2026-09-23 (found while live-testing the
# OllamaUnavailable abort path): this used to hardcode localhost:11434
# instead of reading OLLAMA_HOST from the environment the way
# agents/base.py already does. In any deployment where Ollama isn't on
# localhost (a remote GPU box, a different port, a container), this module
# would silently keep hitting the wrong address while the rest of the
# pipeline correctly used the configured host -- proven live: a test
# pointing OLLAMA_HOST at an unreachable port to simulate an Ollama outage
# for agents/base.py's own callers left this module still hitting the
# real, working Ollama instance underneath it, undetected. Same env var
# name/default as agents/base.py's own OLLAMA_HOST, duplicated rather than
# imported -- data/precompute/ has no existing dependency on agents/base.py
# anywhere in this codebase, and importing across that boundary just for
# one constant would invert it for no real benefit.
_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_OLLAMA_URL = f"{_OLLAMA_HOST}/api/chat"
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
_VALID_LABELS = ("positive", "negative", "neutral", "unrelated")
# Measured 2026-10-09 against 63 headlines labelled by hand without sight of the model's labels (ledger BB-111): the old prompt (no
# company named, no label definitions, sampling at the model's default temperature of 1) matched 54 to 60% (3 repeats); naming the
# company, defining the four labels and scoring at temperature 0 matched 71%, and 78% against 40 fresh headlines (old prompt 57 to 62%).
# The errors that remained were borderline items (a routine product launch, a refinancing), not reversals of direction.
_COMPANY_FALLBACK = "the company these items are about"
_RUBRIC = """say how it affects the stock of {company}. Choose exactly one label:
- positive: favourable for {company}'s business or share price on the facts reported, such as results or guidance ahead of expectations, a contract or customer win, an investment or expansion, a new partnership, a rating or price-target upgrade or raise, a buyback or dividend increase.
- negative: unfavourable for {company} on the facts reported, such as results or guidance below expectations, a production cut, a rating or price-target downgrade or cut, a strike, a lawsuit or fine, a lost customer, a warning.
- neutral: no clear effect on {company}: a scheduled event (earnings date, webcast, transcript), a routine daily price recap ("closed at $X, up 1%"), a hold or reiterated rating, commentary, a comparison or a stock list with no new facts, or news that is truly mixed (one measure ahead, another behind).
- unrelated: the item is not about {company}; its name appears only in passing (a story from a forum or a customer, another company's news, a rival's or a partner's story, or a general list).
Judge the facts, not the mood of words like "falls", "soars" or "struggles". Judge each item on its own."""
_TEMPERATURE = 0  # the model's default (1) made labels move between runs of the same input

# Headlines scored per model call. One call per article was ~250 calls (~16 minutes of model
# time, about 65% of ALL model time in a run) for a busy ticker; the scored sample is now
# ~180 articles in ~6 calls. Measured on KO's 242 saved per-article labels (2026-10-01): the
# per-article scorer agrees with ITSELF on 75% of a re-scored sample. Batches of 30 returning a
# plain labels array agree with the saved labels on 71% and take 24 s of model time for all
# 242 (the same batch returning numbered {n, sentiment} objects: 70%, 52 s; saved label mix
# 34/14/52 positive/negative/neutral vs 29/17/54 for the array and 36/20/44 for the objects).
# Larger batches and a one-letter string were tested and rejected: batches of 60 agree on only
# 62%, and the letter string never says "positive" (44%).
_BATCH_SIZE = 30
_BATCH_TIMEOUT = aiohttp.ClientTimeout(total=180)
_BATCH_TEXT_CHARS = 300  # the summary snippet is a hint, not the article

_FORMAT_SCHEMA = {
    "type": "object",
    "properties": {"sentiment": {"type": "string", "enum": list(_VALID_LABELS)}},
    "required": ["sentiment"],
}


def _batch_prompt(items: list[dict], company: str | None = None) -> str:
    lines = []
    for number, item in enumerate(items, start=1):
        text = (item.get("text") or "")[:_BATCH_TEXT_CHARS]
        lines.append(f"{number}. {item['headline']}" + (f" | {text}" if text else ""))
    return (
        "For each news item below, "
        + _RUBRIC.format(company=company or _COMPANY_FALLBACK)
        + f" Return exactly {len(items)} labels, in the order of the items.\n\n"
        + "\n".join(lines)
    )


def _batch_format_schema(count: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {"type": "string", "enum": list(_VALID_LABELS)},
            }
        },
        "required": ["labels"],
    }


def _prompt(headline: str, text: str, company: str | None = None) -> str:
    body = f"\n\n{text}" if text else ""
    return (
        "For the news item below, " + _RUBRIC.format(company=company or _COMPANY_FALLBACK)
        + f"\n\nHeadline: {headline}{body}"
    )


async def _score_article(
    session: aiohttp.ClientSession,
    headline: str,
    text: str,
    *,
    company: str | None = None,
    capture: CaptureContext | None = None,
) -> str | None:
    """Returns "positive"|"negative"|"neutral", or None on any failure - no
    fallback (2026-08-07 no-fallback-chains decision): a genuine scoring
    failure surfaces as missing data, not a silent retry on another model.

    `capture` (86bbwachy Phase 3) is None by default -- every existing
    caller/test keeps writing zero artifact files, same as
    agents/base.py's own "gated on ticker being set" precedent. Only
    summarize_news's own real caller (DataPipeline.prepare(), when it has
    a real run_id from the orchestrator) passes one."""
    prompt_text = _prompt(headline, text, company)
    payload = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": False,
        "think": _THINK,
        "format": _FORMAT_SCHEMA,
        "options": {"num_ctx": _NUM_CTX, "temperature": _TEMPERATURE},
    }
    try:
        async with session.post(_OLLAMA_URL, json=payload, timeout=_TIMEOUT) as response:
            if response.status != 200:
                logger.warning("sentiment_score_http_error", status=response.status)
                report_degradation(
                    "ollama", "sentiment_score", LLM_REQUEST_FAILED, f"HTTP {response.status}"
                )
                return None
            data = await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
        # json.JSONDecodeError is real and reachable here, not just defensive: a 200
        # status with a genuinely truncated/malformed body (partial write, mid-response
        # crash) makes response.json() raise json.JSONDecodeError directly - it's a
        # plain ValueError subclass, not wrapped in aiohttp.ClientError, so it doesn't
        # get caught without being listed explicitly. No capture on this path -- there
        # is no response body at all to write as an artifact (matches agents/base.py's
        # own call_model, which never reaches its capture point on this class of
        # failure either).
        logger.warning("sentiment_score_request_failed", error=str(exc))
        report_degradation("ollama", "sentiment_score", LLM_REQUEST_FAILED, str(exc), exc=exc)
        return None

    # message/content pulled out here, before the parse try/except, so both
    # the capture call below and the parse logic can use them regardless of
    # which one fails -- restructured from the original flat try/except
    # (which extracted `content` inside the try) for that reason, not a
    # behavior change: a missing "message"/"content" key still ends up as
    # the same sentiment_score_malformed_response warning via json.loads("")
    # raising JSONDecodeError (a ValueError subclass), still caught below.
    #
    # isinstance guards, not bare .get() -- real regression caught on
    # review: response.json() succeeding only means `data` is valid JSON,
    # not that it's a dict (a top-level JSON null/string/list/number is
    # just as valid). The ORIGINAL code's data["message"]["content"] was
    # inside a try/except that already caught the TypeError a None/non-dict
    # `data` raises; moving this out for capture's sake would have lost
    # that protection and let one weird Ollama response crash the entire
    # summarize_news() batch instead of degrading to a single None score.
    safe_data = data if isinstance(data, dict) else {}
    message = safe_data.get("message")
    message = message if isinstance(message, dict) else {}
    content = message.get("content", "")
    content = content if isinstance(content, str) else ""
    thinking = message.get("thinking")
    thinking = thinking if isinstance(thinking, str) else ""

    sentiment: str | None = None
    parse_error: str | None = None
    try:
        parsed = json.loads(content)
        sentiment = parsed["sentiment"]
    except (KeyError, TypeError, ValueError):
        logger.warning("sentiment_score_malformed_response")
        parse_error = "malformed response"
    else:
        # Belt-and-suspenders: Ollama's `format` param is genuinely schema-constrained
        # (verified live), not just a prompt hint - but still validate rather than
        # trust it blindly, matching the "missing over wrong" posture used throughout.
        if sentiment not in _VALID_LABELS:
            logger.warning("sentiment_score_unexpected_label", label=sentiment)
            parse_error = f"unexpected label: {sentiment}"
            sentiment = None

    if capture is not None:
        record_call(
            capture,
            call_site="precompute:sentiment",
            model=_MODEL,
            options=payload["options"],
            prompt_text=prompt_text,
            response_body=data,
            thinking_chars=len(thinking),
            empty_content=not content.strip(),
            parsed_ok=sentiment is not None,
            parse_error=parse_error,
        )

    return sentiment


async def _score_batch(
    session: aiohttp.ClientSession,
    items: list[dict],
    *,
    company: str | None = None,
    capture: CaptureContext | None = None,
) -> list[str | None]:
    """One model call for up to `_BATCH_SIZE` headlines. Returns one label (or None) per
    item, in order; None for every item on any request or parse failure, and for any item
    the model skipped or labelled with something that is not a valid label. Same
    no-fallback posture as `_score_article`: a failure is missing data, never a guess."""
    prompt_text = _batch_prompt(items, company)
    payload = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": False,
        "think": _THINK,
        "format": _batch_format_schema(len(items)),
        "options": {"num_ctx": _NUM_CTX, "temperature": _TEMPERATURE},
    }
    try:
        async with session.post(_OLLAMA_URL, json=payload, timeout=_BATCH_TIMEOUT) as response:
            if response.status != 200:
                logger.warning("sentiment_batch_http_error", status=response.status)
                report_degradation(
                    "ollama", "sentiment_score", LLM_REQUEST_FAILED, f"HTTP {response.status}"
                )
                return [None] * len(items)
            data = await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
        logger.warning("sentiment_batch_request_failed", error=str(exc))
        report_degradation("ollama", "sentiment_score", LLM_REQUEST_FAILED, str(exc), exc=exc)
        return [None] * len(items)

    safe_data = data if isinstance(data, dict) else {}
    message = safe_data.get("message")
    message = message if isinstance(message, dict) else {}
    content = message.get("content", "")
    content = content if isinstance(content, str) else ""
    thinking = message.get("thinking")
    thinking = thinking if isinstance(thinking, str) else ""

    labels: list[str | None] = [None] * len(items)
    parse_error: str | None = None
    try:
        # Position is the alignment: the schema pins the array to exactly one label per item.
        for index, label in enumerate(json.loads(content)["labels"][: len(items)]):
            if label in _VALID_LABELS:
                labels[index] = label
    except (KeyError, TypeError, ValueError):
        logger.warning("sentiment_batch_malformed_response")
        parse_error = "malformed response"

    if capture is not None:
        record_call(
            capture,
            call_site="precompute:sentiment",
            model=_MODEL,
            options=payload["options"],
            prompt_text=prompt_text,
            response_body=data,
            thinking_chars=len(thinking),
            empty_content=not content.strip(),
            parsed_ok=any(label is not None for label in labels),
            parse_error=parse_error,
        )
    return labels


async def _score_chunk(
    session: aiohttp.ClientSession,
    chunk: list[dict],
    *,
    company: str | None,
    capture: CaptureContext | None,
) -> list[str | None]:
    """A batch, asked again once if it produced nothing usable; whatever is still
    unlabelled after that is scored one article at a time (a rare path: at most the
    chunk's size in extra calls, none when the batch was fine)."""
    labels = await _score_batch(session, chunk, company=company, capture=capture)
    if not any(label is not None for label in labels):
        labels = await _score_batch(session, chunk, company=company, capture=capture)
    missing = [i for i, label in enumerate(labels) if label is None]
    if missing:
        singles = await asyncio.gather(
            *(
                _score_article(session, chunk[i]["headline"], chunk[i]["text"], company=company, capture=capture)
                for i in missing
            )
        )
        for index, label in zip(missing, singles, strict=True):
            labels[index] = label
    return labels


async def summarize_news(
    id_assigned_articles: list[dict],
    *,
    company: str | None = None,
    ticker: str | None = None,
    capture: CaptureContext | None = None,
) -> dict:
    """Labels each already-ID-assigned article positive, negative, neutral or unrelated (not about
    `company`). Headlines whose wording fixes the label are labelled by pattern
    (headline_rules.py, which needs `company`/`ticker` to know the item names the company); the
    rest go to a local LLM, ~30 articles per call at temperature 0 (the caller passes the bounded
    SAMPLE to score, not everything fetched). Returns {"articles": [...], "sentiment_source": ...} - the
    caller distributes these two keys across DataBundle.news_with_sentiment
    and DataBundle.sentiment_source respectively.

    `capture`, when given, is threaded straight through to every
    _score_article call unchanged (86bbwachy Phase 3) -- one shared
    CaptureContext, not a new one per article, since seq/call_log need to
    be shared across every article this run scores."""
    if not id_assigned_articles:
        return {"articles": [], "sentiment_source": None}

    # Headlines whose label their wording fixes (price recaps, webcasts, rating changes) are labelled by pattern and
    # never reach the model (data/precompute/headline_rules.py).
    rule_labels = [rule_label(a["headline"], a.get("text"), company, ticker) for a in id_assigned_articles]
    to_model = [a for a, label in zip(id_assigned_articles, rule_labels, strict=True) if label is None]
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT)
    chunks = [to_model[start : start + _BATCH_SIZE] for start in range(0, len(to_model), _BATCH_SIZE)]

    async def _score_bounded(session: aiohttp.ClientSession, chunk: list[dict]):
        async with semaphore:
            return await _score_chunk(session, chunk, company=company, capture=capture)

    async with aiohttp.ClientSession() as session:
        chunk_labels = await asyncio.gather(*(_score_bounded(session, chunk) for chunk in chunks))
    from_model = iter(label for labels in chunk_labels for label in labels)
    sentiments = [label if label is not None else next(from_model) for label in rule_labels]

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
    # sentiment_source is "local_llm" even when NO article was scored, so it reads
    # as success. Say how many were left unscored, in one event with the count
    # (the return value is unchanged).
    unscored = sum(1 for sentiment in sentiments if sentiment is None)
    if unscored:
        report_degradation(
            "ollama",
            "sentiment",
            SENTIMENT_UNSCORED,
            f"{unscored} of {len(sentiments)} articles could not be scored",
            context={"unscored": unscored, "total": len(sentiments)},
        )
    return {"articles": articles, "sentiment_source": "local_llm"}
