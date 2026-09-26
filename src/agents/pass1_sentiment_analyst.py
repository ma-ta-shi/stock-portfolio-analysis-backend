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
- `peer_sentiment` is hardcoded `[]` in `data/pipeline.py`'s DataBundle
  assembly today -- a real, currently-permanent gap (not this port's to fix),
  renders as honestly unavailable.
- `analyst_upgrades_30d`/`analyst_downgrades_30d` (harness: derived counts):
  no precompute module derives an upgrade/downgrade count. The real
  `analyst_recommendation_trends` (US-only, `None` for CA -- confirmed via
  `data/providers/finnhub.py`) is a list of PER-PERIOD buy/hold/sell/
  strong_buy/strong_sell snapshots, not a movement count -- deriving
  "upgrades" from two periods' distributions would be a real, non-trivial
  formula this runner doesn't have a verified source for. The two most
  recent periods' raw distributions are rendered instead, honest about being
  raw counts, not a movement judgment.
- `insider_buys_90d`/`insider_sells_90d`: `insider_activity` on DataBundle is
  `{"transactions": [...]}`, a raw `NormalizedInsiderTransaction` list (date,
  insider_name, is_issuer, transaction_type, shares, value) -- not
  pre-aggregated. Counted here the same way
  `research_sources.py::_compute_insider_direction` derives its own signal
  (excludes `is_issuer=True` rows -- that's buyback activity, not personal
  insider direction -- and non-directional `exercise`/`gift`/`other` types).
- Short interest 30-day trend: `NormalizedShortInterest`'s own docstring
  states "the 30-day trend payload line comes from shares_short vs
  shares_short_prior_month" -- rendered as the two raw counts (a factual
  input line), not a pre-judged trend label; `short_interest_interpretation`
  (trend/interpretation) stays the LLM's own output field, per
  `agents/validators/pass1.py`'s schema for it.
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
`peer_sentiment` is always `False` (a permanent, currently-hardcoded gap, see
`build_user_message()`'s own docstring) and is deliberately still included as
a real, always-mentioned gap sentence, not excluded as a "known" absence.

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
from datetime import UTC, date, datetime, timedelta
from functools import partial

from agents.base import BaseRunner
from agents.prompts import fill, load_template
from agents.utils import (
    RenderedField,
    render_data_coverage_line,
    render_data_warnings,
    to_data_coverage,
)
from agents.validators.common import validate_confidence_requires_caveat_when_flagged
from agents.validators.pass1 import validate_canadian_caveat, validate_sentiment_analyst
from data.schemas.data_bundle import DataBundle

_INSIDER_WINDOW_DAYS = 90
_STALE_NEWS_DAYS = 14
_STALE_SHORT_INTEREST_DAYS = 45
_POSITIVE_SENTIMENT_RATIO = 0.7
_ELEVATED_SHORT_INTEREST_PCT = 10.0


def _validate_with_caveats(
    output: dict,
    canadian_sentiment_inferred: bool,
    material_absent: list[str],
    anomalies: list[str],
    stale_data: list[str],
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
    return passed and ca_passed and cq_passed, errors + ca_errors + cq_errors


_COVERAGE_GAP_SENTENCES = {
    "news_block": "no news articles available",
    "analyst_activity": "no analyst upgrade/downgrade data available",
    "analyst_consensus": "no analyst consensus data available",
    "short_interest": "no short interest data available",
    "peer_sentiment": "peer sentiment comparison is not yet available",
}


def _data_coverage_line(field_presence: dict[str, bool]) -> str:
    return render_data_coverage_line(field_presence, _COVERAGE_GAP_SENTENCES)


def _fmt(v):
    return "N/A" if v is None else str(v)


def _insider_counts(bundle: DataBundle) -> tuple[int, int]:
    transactions = bundle.insider_activity.get("transactions", [])
    cutoff = (datetime.now(UTC) - timedelta(days=_INSIDER_WINDOW_DAYS)).date().isoformat()
    qualifying = [
        t for t in transactions
        if t.get("date") and t["date"] >= cutoff
        and not t.get("is_issuer")
        and t.get("transaction_type") in ("purchase", "sale")
    ]
    buys = sum(1 for t in qualifying if t["transaction_type"] == "purchase")
    sells = sum(1 for t in qualifying if t["transaction_type"] == "sale")
    return buys, sells


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
    if si_pct is not None and si_pct >= _ELEVATED_SHORT_INTEREST_PCT:
        flags.append(
            f"news sentiment is {positive_ratio:.0%} positive despite elevated short interest "
            f"({si_pct}% of float)"
        )
    return flags


def _news_block(bundle: DataBundle) -> RenderedField:
    articles = bundle.news_with_sentiment or []
    if not articles:
        return RenderedField(text="  (no news articles available)", present=False)
    lines = []
    for a in articles:
        lines.append(
            f"  {a['id']}: {a['headline']} ({a['source']}, {a['quality_tier']}, "
            f"sentiment={a.get('sentiment') or 'unscored'})"
        )
    return RenderedField(text="\n".join(lines), present=True)


def _analyst_activity_block(bundle: DataBundle) -> RenderedField:
    # Permanently None for CA tickers (Finnhub is US-only, confirmed via
    # data/providers/finnhub.py -- see module docstring), not a per-run
    # fetch failure for those; genuinely absent for a US ticker only when
    # the fetch itself returned nothing.
    trends = bundle.analyst_recommendation_trends
    if not trends:
        return RenderedField(
            text="  N/A — not available for Canadian stocks, or no data returned.", present=False
        )
    lines = []
    for row in trends[:2]:  # most recent 2 periods
        lines.append(
            f"  {row.get('period', '?')}: strong_buy={_fmt(row.get('strong_buy'))} "
            f"buy={_fmt(row.get('buy'))} hold={_fmt(row.get('hold'))} "
            f"sell={_fmt(row.get('sell'))} strong_sell={_fmt(row.get('strong_sell'))}"
        )
    return RenderedField(text="\n".join(lines), present=True)


def _short_interest_block(bundle: DataBundle) -> RenderedField:
    si = bundle.short_interest
    if not si:
        return RenderedField(text="  N/A — no short interest data available.", present=False)
    text = (
        f"  Short interest % of float: {_fmt(si.get('short_interest_pct'))}\n"
        f"  Days to cover: {_fmt(si.get('days_to_cover'))}\n"
        f"  Shares short: {_fmt(si.get('shares_short'))} "
        f"(30d prior: {_fmt(si.get('shares_short_prior_month'))})"
    )
    return RenderedField(text=text, present=True)


def build_user_message(bundle: DataBundle) -> tuple[str, dict[str, bool]]:
    """Returns (rendered user message, field presence map) -- the second
    element is 86bbwachy Phase 4's own new addition. insider_activity
    (buys_90d/sells_90d) has no entry: both are real counts, always
    renderable as a number (0 is a legitimate real answer, not "N/A" --
    there's no genuine absent case the way there is for the other
    blocks). peer_sentiment always reports present=False -- a permanent,
    currently-hardcoded gap (data/pipeline.py's own DataBundle assembly,
    see module docstring), not a per-run signal, same D3-style permanent
    gap as Stock Researcher's transcript_excerpts.
    """
    ctx = bundle.context
    company_info = bundle.company_info
    flags = bundle.canadian_data_flags

    canadian_flag = ""
    if flags is not None:
        canadian_flag = "\nCANADIAN DATA LIMITED: true — Finnhub analyst upgrade/downgrade data unavailable, article sentiment scored from headlines only."

    buys_90d, sells_90d = _insider_counts(bundle)
    news = _news_block(bundle)
    analyst_activity = _analyst_activity_block(bundle)
    short_interest = _short_interest_block(bundle)
    consensus_rating = bundle.analyst_consensus.get("consensus_rating")

    text = f"""{bundle.stock.ticker} ({company_info.get('name')}) | {company_info.get('sector')} | {bundle.stock.exchange} | {bundle.stock.currency}
Timeline: {ctx.timeline} | Account: {ctx.account_type} | As of: {bundle.data_vintage.isoformat()}{canadian_flag}
social_sentiment: unknown (always — orchestrator-set, do not override)

NEWS SENTIMENT (NEWS):
{news.text}

ANALYST ACTIVITY (ANALYST):
{analyst_activity.text}
  Consensus: {_fmt(consensus_rating)} | Avg target: {_fmt(bundle.analyst_consensus.get('target_mean'))} {bundle.stock.currency}
  Analyst count: {_fmt(bundle.analyst_consensus.get('num_analysts'))}

INSIDER ACTIVITY (INSIDER / SIGNALS):
  Insider buys (90d): {buys_90d}
  Insider sells (90d): {sells_90d}

SHORT INTEREST (SHORT):
{short_interest.text}

PEER SENTIMENT:
  N/A — not currently available (peer sentiment comparison is not yet built into the data pipeline)."""

    field_presence = {
        "news_block": news.present,
        "analyst_activity": analyst_activity.present,
        "analyst_consensus": consensus_rating is not None,
        "short_interest": short_interest.present,
        "peer_sentiment": False,
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
                material_absent=self.last_data_coverage["absent"],
                anomalies=self.last_anomalies,
                stale_data=self.last_stale_data,
            ),
            max_tokens=3500,
            temperature=0.3,
        )
