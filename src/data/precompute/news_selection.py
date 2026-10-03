"""news_selection.py: which articles each agent gets, so a busy ticker cannot flood a prompt.

A busy ticker has ~100 articles a day (MSFT, measured 2026-09-30); a prompt cannot carry a
month of that and a local model should not be asked to score all of it (one call per
article was ~65% of all model time in a run). So the data layer picks, deterministically
and without a model call, a small time-balanced set per day:

- a SCORED SAMPLE (about SCORED_PER_DAY a day) whose sentiment labels give the tone and its
  weekly trend, large enough that the estimate is meaningful, and
- a SHOWN subset (about SHOWN_PER_DAY a day) that the Sentiment agent actually reads and
  cites. Statistical power therefore does not cost prompt size.

A low-volume ticker, whose days hold fewer articles than the per-day size, loses nothing.

Ranking inside a day: items that name the company or its ticker first (measured 2026-10-01:
only 29-34% of MSFT's Finnhub items, 56-62% of AAPL's and 39-43% of KO's do; the rest are
market round-ups that merely list the ticker), then higher quality tier, then rotating
across sources so one publisher (87% of MSFT's articles are Yahoo) does not fill every slot, then newest first;
near-duplicate headlines are dropped. First-pass heuristic, not a relevance judgement.

The Sentiment sample (scored and shown) is then limited to those company-naming items, as long as the window holds
at least MIN_NAMED_TO_FILTER of them. Measured 2026-10-03 over 10 live tickers: the old "a preference, never a
filter" rule filled quiet days with items that never name the company, and the thinner the coverage the worse it got
(shown headlines not naming the company: MSFT 2%, AAPL 2%, KO 10%, RDDT 17%, SHOP.TO 11%, TD.TO 17%, CNQ.TO 40%,
ENB.TO 45%, RY.TO 57%, BAM.TO 62%): a funds notice, an Ashland sawmill, personal-finance clickbait, all scored into
the tone and listed to the agent. Fewer, on-topic headlines are the honest picture of a thinly covered name. Below the
minimum (a name the matcher cannot find) the old behaviour stays, so nothing is ever emptied. The Researcher's list is
unchanged.
"""

import re
from collections import defaultdict
from collections.abc import Sequence
from datetime import date, datetime

SCORED_PER_DAY = 6
# The Sentiment sample is limited to items naming the company when at least this many do across the window.
MIN_NAMED_TO_FILTER = 5
SHOWN_PER_DAY = 2
# The Stock Researcher reads headlines, not scores, and drops "low" tier ones (its own renderer
# already did): this many across the window, best-ranked first within each day.
RESEARCHER_PER_DAY = 2

_TIER_RANK = {"primary": 0, "secondary": 1, "low": 2}


def _day_of(article: dict) -> date:
    value = article["date"]
    return value.date() if isinstance(value, datetime) else value


def _headline_key(headline: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (headline or "").lower()).strip()


def _name_pattern(names: Sequence[str]) -> re.Pattern | None:
    """Matches any name or ticker as a whole word, any case. A Canadian symbol is matched
    without its exchange suffix ("SHOP.TO" is written "SHOP" in a headline)."""
    words = []
    for name in names:
        bare = re.sub(r"\.(TO|V)$", "", (name or "").strip(), flags=re.IGNORECASE)
        if len(bare) >= 2:
            words.append(re.escape(bare))
    if not words:
        return None
    return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(words) + r")(?![A-Za-z0-9])", re.IGNORECASE)


def _tier_then_source(group: list[dict]) -> list[dict]:
    """Higher tier first; inside a tier, round-robin across sources (each source's items
    newest first) so one publisher does not take every slot."""
    ordered: list[dict] = []
    for tier in sorted(
        {a.get("quality_tier", "low") for a in group}, key=lambda t: _TIER_RANK.get(t, 9)
    ):
        by_source: dict[str, list[dict]] = defaultdict(list)
        for article in group:
            if article.get("quality_tier", "low") == tier:
                by_source[article.get("source", "")].append(article)
        queues = list(by_source.values())  # insertion order = newest source first
        while any(queues):
            for queue in queues:
                if queue:
                    ordered.append(queue.pop(0))
    return ordered


def _rank_within_day(articles: list[dict], pattern: re.Pattern | None) -> tuple[list[dict], int]:
    """Best first: items naming the company, then by tier, then round-robin across sources;
    duplicates (same normalised headline) dropped. Also returns how many of the leading items name the company."""
    seen: set[str] = set()
    unique = []
    for article in sorted(articles, key=lambda a: a["date"], reverse=True):
        key = _headline_key(article.get("headline", ""))
        if key and key in seen:
            continue
        seen.add(key)
        unique.append(article)

    if pattern is None:
        return _tier_then_source(unique), 0
    on_topic = [a for a in unique if pattern.search(f"{a.get('headline', '')} {a.get('text', '')}")]
    on_topic_ids = {id(a) for a in on_topic}
    rest = [a for a in unique if id(a) not in on_topic_ids]
    return _tier_then_source(on_topic) + _tier_then_source(rest), len(on_topic)


def select_news(
    articles: list[dict],
    *,
    scored_per_day: int = SCORED_PER_DAY,
    shown_per_day: int = SHOWN_PER_DAY,
    researcher_per_day: int = RESEARCHER_PER_DAY,
    names: Sequence[str] = (),
) -> dict:
    """Takes the dated, fetched articles (dicts with at least `date`, `headline`, `source`,
    `quality_tier`). Returns the SAME dicts, never copies, as three lists:

    - "scored":     the sample whose sentiment gets scored (superset of "shown")
    - "shown":      the subset the Sentiment agent lists
    - "researcher": the Stock Researcher's headlines ("low" tier excluded)
    - "fetched":    how many articles went in (after de-duplication)

    `names`: the company's name variants and ticker; items naming one rank first within their
    day, and the scored and shown samples keep only those when the window holds at least
    MIN_NAMED_TO_FILTER of them (see the module note); otherwise a day with no such item still
    gets its best ones. The researcher list is never filtered.

    Every list is ordered oldest first, like the ids that are assigned over them."""
    shown_per_day = min(shown_per_day, scored_per_day)  # `shown` is always part of `scored`
    pattern = _name_pattern(names)
    by_day: dict[date, list[dict]] = defaultdict(list)
    for article in articles:
        by_day[_day_of(article)].append(article)

    scored: list[dict] = []
    shown: list[dict] = []
    researcher: list[dict] = []
    fetched = 0
    days = [(day, *_rank_within_day(by_day[day], pattern)) for day in sorted(by_day)]
    only_named = pattern is not None and sum(named for _, _, named in days) >= MIN_NAMED_TO_FILTER
    for day, ranked, named in days:
        fetched += len(ranked)
        pool = ranked[:named] if only_named else ranked
        scored.extend(pool[:scored_per_day])
        shown.extend(pool[:shown_per_day])
        researcher.extend(
            [a for a in ranked if a.get("quality_tier") != "low"][:researcher_per_day]
        )

    def oldest_first(items: list[dict]) -> list[dict]:
        return sorted(items, key=lambda a: a["date"])

    return {
        "scored": oldest_first(scored),
        "shown": oldest_first(shown),
        "researcher": oldest_first(researcher),
        "fetched": fetched,
    }
