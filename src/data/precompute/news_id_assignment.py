"""news_id_assignment.py — assigns stable, globally-unique N{num} citation
IDs to news articles for one analysis run (ClickUp 86ban0wft).

Every Pass 1 agent that references company news reads from this single ID
namespace, assigned once per run over the widest window any agent uses
(currently the Stock Researcher's 180-day long-term window, per
docs/technical/data-pipeline.md's precompute/news_id_assignment.py
section) — not per-agent, so "Researcher's N3" and "Sentiment's N3" are
always the same article. An agent with a narrower window (e.g. Sentiment's
7/14/30-day window) just gets whichever subset of IDs fall inside its
window; downstream consumers check set membership, not contiguity, so a
non-contiguous subset like {N12, N13, N15} is expected and valid.

Called from DataPipeline.prepare() right after the news fetch, before any
agent-specific window filtering. Output shape matches the news_items entry
in data-pipeline.md §4 (ResearchSourcesBundle): list[dict] with id, date,
headline, source, quality_tier - plus text (ClickUp 86ban0wf4), a
passthrough of the raw article's body/summary needed by sentiment.py's LLM
scoring call, not part of the original data-pipeline.md spec but additive
and safe (no existing field renamed or removed).
"""

from datetime import UTC, datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# quality_tier source classification — no mapping is specified anywhere in
# the docs (data-pipeline.md only names the three-tier enum itself), so
# this is a first-pass design, not sourced from any doc, same disclosure
# pattern as fundamentals.py's health_rating thresholds. Case-insensitive
# substring match against the raw `source` string providers return, both
# unnormalized: Finnhub gives a publisher name ("Reuters", "Yahoo"),
# openbb-tmx gives a wire name suffixed " via QuoteMedia" ("Canada
# Newswire via QuoteMedia", "PR Newswire via QuoteMedia"). Default is
# "low": an unrecognized source gets the most conservative tier rather
# than being assumed reliable.
#
# "canada newswire" sits alongside "pr newswire" / "business wire" /
# "globenewswire": it is the same class of primary-source wire and is
# openbb-tmx's dominant Canadian source (86bbqh23f — confirmed live it was
# ~80% of RY.TO's feed and would otherwise all tier "low" while the same
# feed's PR-Newswire articles tiered "primary").
#
# No "sec" entry: a plausible-looking wire-service keyword that was cut on
# review — Finnhub's company-news `source` field is a publisher name
# ("Reuters", "Yahoo", "Motley Fool"), never literally "SEC" (that's an
# edgartools/regulatory-filing concept, not a Finnhub news-wire value), and
# a bare 3-character substring risks matching an unrelated source name that
# happens to contain "sec".
_PRIMARY_SOURCE_KEYWORDS = (
    "pr newswire",
    "business wire",
    "globenewswire",
    "canada newswire",
    "reuters",
)
_SECONDARY_SOURCE_KEYWORDS = (
    "bloomberg",
    "cnbc",
    "marketwatch",
    "barron",
    "yahoo",
    "forbes",
    "wall street journal",
    "financial times",
    "associated press",
    "ap news",
)


def classify_quality_tier(source: str) -> str:
    normalized = (source or "").lower()
    if any(keyword in normalized for keyword in _PRIMARY_SOURCE_KEYWORDS):
        return "primary"
    if any(keyword in normalized for keyword in _SECONDARY_SOURCE_KEYWORDS):
        return "secondary"
    return "low"


def _parse_published_at(value: Any) -> datetime | None:
    """Both wired providers return published_at as a string formatted
    "%Y-%m-%d %H:%M:%S" (finnhub.py and, since 86bbqh23f, openbb_tmx.py) —
    a real datetime is also accepted defensively, in case a future provider
    returns one directly. Returns None for anything else (missing,
    malformed).

    A datetime with tzinfo is converted to UTC then stripped to naive
    (not a bare `.replace(tzinfo=None)`, which would keep the wrong
    wall-clock numbers and silently mis-sort). The string path parses
    naive as-is.

    KNOWN, out of scope here (86bbr4azz): the two providers' strings are
    not the same reference — Finnhub emits box-local time, openbb-tmx
    emits America/New_York wall clock. Sorting a CA-only or US-only list
    is fine; a merged CA+US list would mis-sort by the offset. 86bbr4azz,
    which builds that merge, must reconcile the two."""
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(UTC)
        return value.replace(tzinfo=None)
    if isinstance(value, str):
        try:
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    return None


def assign_news_ids(articles: list[dict]) -> list[dict]:
    """Assign stable N{num} IDs, oldest first, over the widest window any
    Pass 1 agent uses.

    Articles missing or carrying an unparseable published_at are dropped
    rather than failing the whole run — one bad upstream record shouldn't
    block ID assignment for every other article, the same graceful-
    degradation principle applied throughout precompute/.
    """
    dated_articles = []
    dropped_count = 0
    for article in articles:
        published_at = _parse_published_at(article.get("published_at"))
        if published_at is None:
            dropped_count += 1
            continue
        dated_articles.append((published_at, article))

    if dropped_count:
        # A handful of drops from real upstream noise is expected and fine
        # to stay silent about; a provider-wide format change would drop
        # everything, and that failure mode must show up in logs, not just
        # manifest as "empty news_items" three layers downstream.
        logger.warning(
            "news_id_assignment_dropped_articles",
            dropped_count=dropped_count,
            total_count=len(articles),
        )

    dated_articles.sort(key=lambda pair: pair[0])

    return [
        {
            "id": f"N{i}",
            "date": published_at,
            "headline": article.get("headline", ""),
            "source": article.get("source", ""),
            "quality_tier": classify_quality_tier(article.get("source", "")),
            # text: passthrough for sentiment.py's LLM scoring call, which needs more
            # than a bare headline to score well. Added here (not re-derived
            # downstream) because this function sorts and drops unparseable-date
            # articles - its output is neither the same length nor order as the
            # input, so a caller can't zip back to the original raw article by
            # position afterward. Both providers emit `summary` in the house shape
            # (86bbqh23f wired openbb-tmx to it), but CA `summary` is near-always
            # "" - openbb-tmx/QuoteMedia articles are headline-only - so CA
            # scoring runs on the headline alone; sentiment.py treats `text` as
            # optional. No "url" field - considered, cut: nothing reads it.
            "text": article.get("summary", ""),
        }
        for i, (published_at, article) in enumerate(dated_articles, start=1)
    ]
