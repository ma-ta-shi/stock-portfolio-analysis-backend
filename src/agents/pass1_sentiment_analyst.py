"""Pass 1 — Sentiment Analyst runner (v1.2).

Production port of `simulation/runners/pass1_sentiment_analyst.py` (86bbuhjup).
NOT a clean port -- field-by-field notes, verified against
`data/precompute/sentiment.py` and `data/pipeline.py`'s DataBundle assembly:

- `news_sentiment_score` (harness: a 0-1 aggregate float) and
  `news_dominant_themes` (harness: a precomputed list): no precompute source
  for either. `precompute/sentiment.py::summarize_news()` only produces a
  PER-ARTICLE label (positive/negative/neutral, via a real LLM classification
  call) -- no aggregate score, no theme extraction. Individual N{num}-cited
  articles with their sentiment labels are rendered instead, matching this
  agent's own prompt (rule 4: "Group news into themes... anchored to one
  representative news_id") -- theme/aggregate synthesis is the LLM's job, the
  same "give raw data, let the LLM interpret" pattern as every other agent's
  interpretive_fields.
- Peer sentiment was retired (2026-10-03): it was `[]` in 48 of 48 real runs, so the payload block, the coverage
  flag, the prompt text and the output field are gone.
- Analyst: the monthly rating distribution (Finnhub for US, yfinance for CA) plus dated rating changes and
  price-target moves (yfinance `upgrades_downgrades`, both markets), summarised in code by
  `data/precompute/sentiment_signals.py` as upgrade, downgrade, initiation and target-move counts over 90 days.
- `insider_buys_90d`/`insider_sells_90d`: `insider_activity` on DataBundle is
  `{"transactions": [...]}`, a raw `NormalizedInsiderTransaction` list (date,
  insider_name, is_issuer, transaction_type, shares, value) -- not
  pre-aggregated. Counted here the same way
  `research_sources.py::_compute_insider_direction` derives its own signal
  (excludes `is_issuer=True` rows -- that's buyback activity, not personal
  insider direction -- and non-directional `exercise`/`gift`/`other` types).
- Short interest: the trend and the three-value read are computed in code (`sentiment_signals.py`) and shown with
  the as-of date; the model no longer produces `short_interest_interpretation` (it failed the validator in 27 of
  48 first attempts) and the Pass 2 view carries the code's value.
- `canadian_sentiment_inferred`: the real prompt's own caveat text ("Canadian
  articles are scored from headlines only") is CA-market-specific, not a
  proxy for the harness's old `canadian_data_limited` flag -- this reads
  `bundle.canadian_data_flags is not None` directly (None for US, populated
  for CA, per DataBundle's own None-for-US enforcement), the real CA/US
  signal.
- "BASE RELIABILITY SCORE"/reliability cap=70 (harness fixture text): omitted,
  same D6-retirement reasoning as every other Pass 1 runner in this port.

86bbummwp Tier 1a: `data_coverage_line` is now built from the same
`field_presence` map `build_user_message()` already computes for
`input_field_coverage`, via the shared `render_data_coverage_line()` helper --
was hardcoded to "Data coverage: standard." on every run before this.

86bbummwp 1d: the retry loop's validator now also enforces the mandatory
Canadian sentiment-inference caveat the user message already tells the model
about via `{canadian_sentiment_inferred}` (rule 7) -- previously requested in
the prompt but never mechanically checked, and previously unwired anywhere
with a phrase ("Finnhub") that didn't match any real prompt at all.

86bbummwp Tier 2: D6's mechanical flags, now real, stored fields on
`agent_outputs` -- the hardest of the 5 agents for this ticket item, since
neither `stale_data` nor `anomalies` had any existing precedent to build on.
- `stale_data`: no ready age field exists for either series (unlike
  Technical/Macro's own `*_age_days` fields) -- computed directly here from
  the most recent article's own `date` and short interest's own `as_of_date`
  (a real, already-fetched field, confirmed on `NormalizedShortInterest` --
  `data/providers/base.py` -- never rendered into the prompt today), each
  against `data_vintage`.
- `anomalies`: genuinely new divergence logic, no existing precedent
  (`contrarian_signals` is LLM-produced, a different concept -- see the field
  notes above). Flags when strongly positive aggregate news sentiment
  co-occurs with net insider selling (reusing `_insider_counts()` below) or
  elevated short interest -- real signals already computed for other
  purposes, combined here for the first time.
"""
from datetime import date, datetime, timedelta
from functools import partial

from agents.base import BaseRunner
from agents.prompts import fill, load_template
from agents.utils import (
    RenderedField,
    compute_data_quality_assessment,
    render_data_coverage_line,
    render_data_warnings,
    to_data_coverage,
)
from agents.validators.common import validate_confidence_requires_caveat_when_flagged
from agents.validators.pass1 import validate_canadian_caveat, validate_sentiment_analyst, validate_theme_news_ids
from data.precompute.insider import summarize_insider_activity
from data.precompute.sentiment_signals import (
    ELEVATED_SHORT_INTEREST_PCT,
    analyst_changes_for_bundle,
    summarize_short_interest,
)
from data.schemas.data_bundle import DataBundle

_INSIDER_WINDOW_DAYS = 90
_STALE_NEWS_DAYS = 14
_STALE_SHORT_INTEREST_DAYS = 45
_POSITIVE_SENTIMENT_RATIO = 0.7


def _validate_with_caveats(
    output: dict,
    canadian_sentiment_inferred: bool,
    material_absent: list[str],
    anomalies: list[str],
    stale_data: list[str],
    news_ids: set[str] | None = None,
) -> tuple[bool, list[str]]:
    """Composing validator (86bbummwp 1d, extended by the follow-on
    confidence/data-quality coupling rule) -- merges the base schema check
    with the mandatory Canadian sentiment-inference caveat, the same
    closure-composition pattern as pass1_technical_analyst.py's own
    `_validate_with_caveats`. The real prompt already tells the model this
    caveat is mandatory when the flag is true (rule 7) -- this only makes an
    already-visible instruction mechanically enforced, not a new one. The new
    confidence/data-quality rule is the backstop for a real, live case (Macro
    Economist claiming `analysis_confidence: "high"` while its own
    stale_data flag showed real staleness) -- see
    `validate_confidence_requires_caveat_when_flagged`'s own docstring."""
    passed, errors = validate_sentiment_analyst(output)
    ca_passed, ca_errors = validate_canadian_caveat(output, canadian_sentiment_inferred)
    cq_passed, cq_errors = validate_confidence_requires_caveat_when_flagged(
        output,
        is_high=output.get("analysis_confidence") == "high",
        material_absent=material_absent,
        anomalies=anomalies,
        stale_data=stale_data,
    )
    id_passed, id_errors = validate_theme_news_ids(output, news_ids) if news_ids is not None else (True, [])
    return passed and ca_passed and cq_passed and id_passed, errors + ca_errors + cq_errors + id_errors


_COVERAGE_GAP_SENTENCES = {
    "news_block": "no news articles available",
    "analyst_activity": "no analyst rating data available",
    "analyst_consensus": "no analyst consensus data available",
    "short_interest": "no short interest data available",
}


def _data_coverage_line(field_presence: dict[str, bool]) -> str:
    return render_data_coverage_line(field_presence, _COVERAGE_GAP_SENTENCES)


def _fmt(v):
    return "N/A" if v is None else str(v)


def _target(v) -> str:
    """The average price target to cents: Yahoo's 578.82245 was being copied into evidence as written."""
    return "N/A" if v is None else f"{v:.2f}"


def _insider_counts(bundle: DataBundle) -> tuple[int, int]:
    """Qualifying insider purchases and sales in the window; the row filter lives in summarize_insider_activity."""
    summary = summarize_insider_activity(bundle.insider_activity.get("transactions", []), None)
    return summary["buy_count"], summary["sell_count"]


def _stale_data(bundle: DataBundle) -> list[str]:
    """D6's `stale_data` flag (86bbummwp Tier 2) -- see module docstring for
    why this computes age directly rather than reading an existing field.

    Compares at `.date()` granularity, not full datetime subtraction --
    caught live (2026-09-25, a real SENT run on RDDT): `finnhub.py` builds
    article dates via bare `datetime.fromtimestamp()`/`datetime.now()` (both
    naive), while `bundle.data_vintage` is `datetime.now(UTC)` (aware) --
    subtracting them raised `TypeError: can't subtract offset-naive and
    offset-aware datetimes` and failed the whole agent. `.date()` sidesteps
    the mismatch entirely (a `date` has no tzinfo to begin with), the same
    technique `_insider_counts()` above already uses for its own date
    comparison -- day-level precision is all `stale_data` needs anyway."""
    stale = []
    articles = bundle.news_with_sentiment or []
    if articles:
        most_recent = max(a["date"] for a in articles)
        most_recent_date = most_recent.date() if isinstance(most_recent, datetime) else most_recent
        if (bundle.data_vintage.date() - most_recent_date).days > _STALE_NEWS_DAYS:
            stale.append("news")
    si = bundle.short_interest
    as_of = si.get("as_of_date") if si else None
    if as_of and (bundle.data_vintage.date() - date.fromisoformat(as_of)).days > _STALE_SHORT_INTEREST_DAYS:
        stale.append("short_interest")
    return stale


def _anomalies(bundle: DataBundle) -> list[str]:
    """D6's `anomalies` flag (86bbummwp Tier 2) -- see module docstring for
    why this is genuinely new logic, not adapted from an existing check."""
    articles = bundle.news_with_sentiment or []
    scored = [a for a in articles if a.get("sentiment") in ("positive", "negative")]
    if not scored:
        return []
    positive_ratio = sum(1 for a in scored if a["sentiment"] == "positive") / len(scored)
    if positive_ratio < _POSITIVE_SENTIMENT_RATIO:
        return []

    flags = []
    buys_90d, sells_90d = _insider_counts(bundle)
    if sells_90d > 0 and sells_90d > buys_90d:
        flags.append(
            f"news sentiment is {positive_ratio:.0%} positive but insiders are net sellers "
            f"over the past {_INSIDER_WINDOW_DAYS} days ({sells_90d} sales vs {buys_90d} purchases)"
        )
    si_pct = (bundle.short_interest or {}).get("short_interest_pct")
    if si_pct is not None and si_pct >= ELEVATED_SHORT_INTEREST_PCT:
        flags.append(
            f"news sentiment is {positive_ratio:.0%} positive despite elevated short interest "
            f"({si_pct}% of float)"
        )
    return flags


def _article_day(article: dict) -> date:
    value = article["date"]
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


_TONES = ("positive", "negative", "neutral")


def _tally(articles: list[dict]) -> dict[str, int]:
    counts = dict.fromkeys(_TONES, 0)
    for a in articles:
        if a.get("sentiment") in counts:
            counts[a["sentiment"]] += 1
    return counts


def _tone_counts(articles: list[dict]) -> str:
    counts = _tally(articles)
    return ", ".join(f"{counts[k]} {k}" for k in _TONES)


def _tone_shares(articles: list[dict]) -> str:
    """Each tone as a share of the WHOLE scored sample, worked out here: a KO run wrote "72% positive" for 48 positive
    of 109 scored (that is 48 of the 67 that were not neutral; the sample is 44% positive, 17% negative)."""
    counts = _tally(articles)
    total = sum(counts.values()) or 1
    return ", ".join(f"{round(100 * counts[k] / total)}% {k}" for k in _TONES)


def _coverage_lines(articles: list[dict], shown: list[dict], coverage: dict) -> list[str]:
    """What the prompt promises ("article_count, distribution counts") and a sentiment trend
    the agent can read across the whole window, not only from the headlines listed. The
    listed headlines are a small slice of a scored SAMPLE of what was fetched (BB-023);
    saying so keeps the sample from being mistaken for everything that was published."""
    lines = [
        f"  Coverage: {coverage['fetched']} articles fetched over the last {coverage['window_days']} days; "
        f"{len(articles)} sampled evenly across the days and scored; {len(shown)} listed below.",
        f"  Tone of the scored sample: {_tone_counts(articles)} ({_tone_shares(articles)}).",
    ]
    weeks: dict[date, list[dict]] = {}
    for a in articles:
        day = _article_day(a)
        weeks.setdefault(day - timedelta(days=day.weekday()), []).append(a)
    for week_start in sorted(weeks):
        week = weeks[week_start]
        lines.append(f"  Week of {week_start.isoformat()}: {_tone_counts(week)} (n={len(week)})")
    lines.append(
        "  Weekly counts come from a sample, so they can show a large shift in tone, not a small one."
    )
    return lines


def _news_block(bundle: DataBundle) -> RenderedField:
    articles = bundle.news_with_sentiment or []
    if not articles:
        return RenderedField(text="  (no news articles available)", present=False)
    # `shown` marks the headlines the agent lists; items without the key (older callers) are all shown.
    shown = [a for a in articles if a.get("shown", True)]
    lines = []
    coverage = getattr(bundle, "news_coverage", None)
    if coverage:
        lines.extend(_coverage_lines(articles, shown, coverage))
    for a in shown:
        lines.append(
            f"  {a['id']} {_article_day(a).strftime('%m-%d')}: {a['headline']} ({a['source']}, {a['quality_tier']}, "
            f"sentiment={a.get('sentiment') or 'unscored'})"
        )
    return RenderedField(text="\n".join(lines), present=True)


def _analyst_activity_block(bundle: DataBundle) -> RenderedField:
    trends = bundle.analyst_recommendation_trends
    changes = getattr(bundle, "analyst_rating_changes", None)
    if not trends and not changes:
        return RenderedField(text="  N/A -- no rating distribution or rating changes returned.", present=False)
    lines, seen = [], set()
    for row in (trends or [])[:2]:  # most recent 2 periods; Yahoo's 0m and -1m are often identical, so skip a repeat
        counts = tuple(row.get(k) for k in ("strong_buy", "buy", "hold", "sell", "strong_sell"))
        if counts in seen:
            continue
        seen.add(counts)
        lines.append(
            f"  {row.get('period', '?')}: strong_buy={_fmt(row.get('strong_buy'))} "
            f"buy={_fmt(row.get('buy'))} hold={_fmt(row.get('hold'))} "
            f"sell={_fmt(row.get('sell'))} strong_sell={_fmt(row.get('strong_sell'))}"
        )
    lines.append(f"  Rating changes: {analyst_changes_for_bundle(bundle)['text']}")
    return RenderedField(text="\n".join(lines), present=True)


def _short_interest_block(bundle: DataBundle) -> RenderedField:
    si = bundle.short_interest
    if not si:
        return RenderedField(text="  N/A -- no short interest data available.", present=False)
    return RenderedField(text=f"  {summarize_short_interest(si)['text']}", present=True)


def build_user_message(bundle: DataBundle) -> tuple[str, dict[str, bool]]:
    """Returns (rendered user message, field presence map) -- the second
    element is 86bbwachy Phase 4's own new addition. insider_activity
    (buys_90d/sells_90d) has no entry: both are real counts, always
    renderable as a number (0 is a legitimate real answer, not "N/A" --
    there's no genuine absent case the way there is for the other
    blocks).
    """
    ctx = bundle.context
    company_info = bundle.company_info
    flags = bundle.canadian_data_flags

    canadian_flag = ""
    if flags is not None:
        canadian_flag = "\nCanadian stock: article sentiment is scored from headlines only (no article body)."

    insider_activity = summarize_insider_activity(
        bundle.insider_activity.get("transactions", []), bundle.price_info.get("market_cap"),
        bundle.insider_activity.get("value_currency"), bundle.price_info.get("currency"),
    )["text"]
    news = _news_block(bundle)
    analyst_activity = _analyst_activity_block(bundle)
    short_interest = _short_interest_block(bundle)
    consensus_rating = bundle.analyst_consensus.get("consensus_rating")

    text = f"""{bundle.stock.ticker} ({company_info.get('name')}) | {company_info.get('sector')} | {bundle.stock.exchange} | {bundle.stock.currency}
Timeline: {ctx.timeline} | Account: {ctx.account_type} | As of: {bundle.data_vintage.isoformat()}{canadian_flag}

NEWS SENTIMENT (NEWS):
{news.text}

ANALYST ACTIVITY (ANALYST):
{analyst_activity.text}
  Consensus: {_fmt(consensus_rating)} | Avg target: {_target(bundle.analyst_consensus.get('target_mean'))} {bundle.stock.currency} | Analysts: {_fmt(bundle.analyst_consensus.get('num_analysts'))}

INSIDER ACTIVITY (INSIDER / SIGNALS):
  Insider activity (90d): {insider_activity}

SHORT INTEREST (SHORT):
{short_interest.text}"""

    field_presence = {
        "news_block": news.present,
        "analyst_activity": analyst_activity.present,
        "analyst_consensus": consensus_rating is not None,
        "short_interest": short_interest.present,
    }
    return text, field_presence


class SentimentAnalystRunner(BaseRunner):
    async def run(self, bundle: DataBundle) -> tuple[dict, list[str]]:
        self.current_agent = "SENT"
        user_msg, field_presence = build_user_message(bundle)
        # 86bbwachy Phase 4 -- set before the LLM call is attempted, so a
        # failed call still records whether its own input was already
        # incomplete.
        self.last_field_coverage = field_presence
        # 86bbummwp Tier 2 -- D6's mechanical flags, set here for the same
        # reason as last_field_coverage above.
        self.last_data_coverage = to_data_coverage(field_presence, _COVERAGE_GAP_SENTENCES)
        self.last_stale_data = _stale_data(bundle)
        self.last_anomalies = _anomalies(bundle)
        material_absent = list(self.last_data_coverage["absent"])
        # 86bbummwp Tier 3 -- D6 section 3's per-agent mechanical data_quality_assessment,
        # set here for the same reason as last_field_coverage above.
        self.last_data_quality_assessment = compute_data_quality_assessment(
            self.last_stale_data, self.last_anomalies, material_absent
        )
        system_prompt = fill(
            load_template("sentiment_analyst"),
            {
                "canonical_ticker": bundle.stock.ticker,
                "company_name": bundle.company_info.get("name"),
                "sector": bundle.company_info.get("sector"),
                "data_coverage_line": _data_coverage_line(field_presence),
                "data_warnings": render_data_warnings(self.last_anomalies, self.last_stale_data),
                "memory_brief": "",
                "accuracy_brief": "",
                "canadian_sentiment_inferred": str(bundle.canadian_data_flags is not None).lower(),
                # {num} is citation notation (N1, N2, ...), not a per-call
                # placeholder -- deliberately absent from this dict so fill()
                # leaves it untouched.
            },
        )
        return await self.call_with_validation(
            system_prompt,
            user_msg,
            partial(
                _validate_with_caveats,
                canadian_sentiment_inferred=bundle.canadian_data_flags is not None,
                material_absent=material_absent,
                anomalies=self.last_anomalies,
                stale_data=self.last_stale_data,
                news_ids={a["id"] for a in (bundle.news_with_sentiment or []) if a.get("shown", True)},
            ),
            max_tokens=3500,
            temperature=0.3,
        )
