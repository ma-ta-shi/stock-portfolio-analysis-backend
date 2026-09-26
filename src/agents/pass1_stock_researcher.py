"""Pass 1 — Stock Researcher runner (v1.3).

Production port of `simulation/runners/pass1_stock_researcher.py` (86bbuhjup).
NOT a clean port: the harness's `build_user_message(fixture)` reads a flat
fixture dict; this reads a real `DataBundle` instead, and several fields don't
map 1:1 -- see the field-by-field notes below (each is a real, verified finding
against `data/schemas/research_sources_bundle.py` and
`data/precompute/research_sources.py`, not a guess):

- `competitive_position`/`key_risks` (harness fixture fields): no equivalent
  anywhere in ResearchSourcesBundle, and none is fabricated here. The real
  prompt's own step 2 instruction already tells the LLM to derive competitive
  position "from the PEER blocks" -- the harness fixture's pre-written
  paragraph was authorial scaffolding, not a genuine input requirement.
- `earnings_transcript_highlights`: `transcript_excerpts` is permanently `[]`
  today (D3, precompute/research_sources.py's own documented, deliberate
  scope decision) -- honestly renders as unavailable, not a bug in this port.
- `management_signals`: the harness fixture used free-text narrative; the real
  `ManagementSignals` model gives derived enums (`buyback_activity`,
  `dividend_activity`, `insider_net_direction_90d`) instead -- rendered from
  those directly, not reconstructed as prose.
- `peer_1_token`/`peer_2_token`: the harness fills these with the peer's REAL
  ticker (`peers[0].get("ticker")`) -- but research_sources.py's own docstring
  ("A peer's own ticker is not separately templated anywhere in the live
  prompt (only {peer_1_token}, the literal citation string 'PEER_1')") is
  explicit that this must be the literal citation token, not a real ticker.
  Filling the real ticker would leak an anonymized peer's identity straight
  into the system prompt. Implemented per the precompute module's documented
  contract here, not copied from the harness.
- "BASE RELIABILITY SCORE FOR THIS AGENT: X/100" (harness fixture block):
  omitted. Reliability scores were fully retired project-wide (decision D6,
  docs/technical/pass1-confidence-model.md) -- the harness text predates that
  decision and was never cleaned up there; carrying it into production would
  reintroduce a retired concept.
- `data_coverage_line`: the harness hardcodes "standard." (its fixtures always
  have full data). Production builds this from `missing_sources_list` (a real
  field precompute/research_sources.py already produces for exactly this
  purpose) plus the permanent transcript-excerpts gap -- a real signal, not a
  literal.
- `price['ytd_return_pct']`: no precompute module anywhere computes this
  (confirmed: not in NormalizedQuote, not in fundamentals.py's price_info).
  Genuine gap, surfaced by omission rather than fabricated.
- `beta`: not in `DataBundle.price_info` (a plain NormalizedQuote passthrough
  -- current_price/market_cap/currency/high_52w/low_52w only), but IS present
  in `DataBundle.risk_metrics` (confirmed live during the DB persistence
  audit). Documented there as "Risk Advisor's" field by primary-consumer
  grouping, but DataBundle is assembled once for the whole run before any
  agent runs -- reading it here is not a layering violation, same as
  price_info/dividend_info already being read by both Fundamental and
  Stock Researcher in the harness fixture.
- `eligible_canadian_dividend`: no discrete field anywhere -- Canadian
  dividend-eligibility is baked into `tax_metrics`'s own rendered string
  (Tax Strategist's field), not exposed separately. Omitted, not fabricated.

86bbummwp 1d: the retry loop's validator now also enforces the mandatory
filing-depth caveat the user message already tells the model about via
`{has_filing_digest}` (rule 5) -- previously requested in the prompt but
never mechanically checked. This is a market-agnostic requirement, not a
Canadian one, despite living in `agents/validators/pass1.py` next to
`validate_canadian_caveat` -- see that function's own docstring for why RSRCH
was split out of it entirely rather than folded in under a renamed flag.

86bbummwp Tier 2: D6's mechanical flags, now real, stored fields on
`agent_outputs`.
- `data_coverage`: NOT built via the shared `to_data_coverage()` helper (see
  `agents/utils.py`) -- that helper mirrors `render_data_coverage_line()`'s
  own `(field_presence: dict[str, bool], gap_sentences)` shape, which doesn't
  fit here: this agent's own `_data_coverage_line()` reads
  `missing_sources_list` (a `list[str]`), not a presence dict, for the exact
  same reason it was never migrated to `render_data_coverage_line()` in Tier
  1. `_data_coverage()` below mirrors `_data_coverage_line()`'s own logic
  instead, so both representations of the same underlying signal stay in
  sync by construction, not by convention.
- `anomalies`: one new cross-check, `buyback_activity`/`insider_net_direction_90d`
  computed independently from the same insider-transaction fetch
  (`precompute/research_sources.py`) and never compared until now. Real,
  correct, cheap logic -- but `_compute_buyback_activity()`'s own docstring
  documents that `"active"` is structurally rare for both markets (Form 4
  doesn't capture large-scale US buybacks; the CA aggregate fallback
  hardcodes `is_issuer=False`), so don't expect this to fire often in
  practice.
- `stale_data`: `latest_filing_age_days`/`latest_news_age_days` are real,
  already-computed fields (`ResearchSourcesBundle`), same "age of latest
  item, not fetch recency" shape as Macro Economist's own age fields -- a
  filing filed 80 days ago is completely normal (companies report roughly
  quarterly), so the threshold has to survive a normal filing cycle, not
  flag most companies as stale most of the time. `latest_transcript_age_days`
  is permanently `None` today (D3, transcripts don't exist yet) -- not
  flagged here, that's a `data_coverage` gap (already tracked via
  `missing_sources_list`), not a staleness signal; nothing to measure an age
  from in the first place.
"""
from functools import partial

from agents.base import BaseRunner
from agents.prompts import fill, load_template
from agents.utils import RenderedField, compute_data_quality_assessment, render_data_warnings
from agents.validators.common import validate_confidence_requires_caveat_when_flagged
from agents.validators.pass1 import validate_filing_depth_caveat, validate_stock_researcher
from data.schemas.data_bundle import DataBundle

_COVERAGE_SOURCES = ("filing_digests", "peer_blocks", "news_items")
_STALE_FILING_DAYS = 120  # ~1 quarter (90d) plus a buffer for real-world reporting lag
_STALE_NEWS_DAYS = 30     # no fresh news at all in a month is a real gap worth flagging


def _validate_with_caveats(
    output: dict,
    has_filing_digest: bool,
    material_absent: list[str],
    anomalies: list[str],
    stale_data: list[str],
) -> tuple[bool, list[str]]:
    """Composing validator (86bbummwp 1d, extended by the follow-on
    confidence/data-quality coupling rule) -- merges the base schema check
    with the mandatory filing-depth caveat, the same closure-composition
    pattern as pass1_technical_analyst.py's own `_validate_with_caveats`
    (`call_with_validation()`'s validator callable takes exactly one
    positional arg, so every extra param here is closed over via `partial()`
    at the call site instead). The real prompt already tells the model this
    caveat is mandatory when the flag is false (rule 5) -- this only makes
    an already-visible instruction mechanically enforced, not a new one.

    `material_absent` is the caller's own `self.last_data_coverage["absent"]`
    with `"transcript_excerpts"` already excluded -- that item is PERMANENTLY
    absent (D3, transcripts don't exist in this system at all yet), so it
    would otherwise trip the new confidence/data-quality rule on every single
    run regardless of real data quality. See
    `validate_confidence_requires_caveat_when_flagged`'s own docstring.

    Note: `validate_stock_researcher`'s own schema already requires
    `caveats` to have >=1 item on every call, unconditionally -- so the new
    rule's own "caveats is empty" branch can never actually be the deciding
    factor here (any output that reaches it already satisfies that base
    requirement, or already failed for that unrelated reason). Composed
    anyway for uniformity across all 7 call sites, and in case that base
    bound is ever loosened -- not dead code removed, just currently
    non-load-bearing for this one agent specifically."""
    passed, errors = validate_stock_researcher(output)
    fd_passed, fd_errors = validate_filing_depth_caveat(output, has_filing_digest)
    cq_passed, cq_errors = validate_confidence_requires_caveat_when_flagged(
        output,
        is_high=output.get("analysis_confidence") == "high",
        material_absent=material_absent,
        anomalies=anomalies,
        stale_data=stale_data,
    )
    return passed and fd_passed and cq_passed, errors + fd_errors + cq_errors


def _data_coverage_line(bundle: DataBundle) -> str:
    gaps = []
    missing = set(bundle.research_sources.missing_sources_list)
    if "filing_digests" in missing:
        gaps.append("no filing digest available for this name")
    if "peer_blocks" in missing:
        gaps.append("no peer comparables available")
    if "news_items" in missing:
        gaps.append("no recent news available")
    # Permanent, not conditional on this run -- D3, see module docstring.
    gaps.append("earnings transcript excerpts are not available (not yet built into the data pipeline)")
    return "standard." if not gaps else "; ".join(gaps) + "."


def _data_coverage(bundle: DataBundle) -> dict:
    """D6's structured data_coverage flag (86bbummwp Tier 2) -- mirrors
    `_data_coverage_line()`'s own logic exactly (same 3 conditional sources,
    same permanent transcript-excerpts gap) so the two representations can't
    drift apart."""
    missing = set(bundle.research_sources.missing_sources_list)
    present = [s for s in _COVERAGE_SOURCES if s not in missing]
    absent = [s for s in _COVERAGE_SOURCES if s in missing]
    absent.append("transcript_excerpts")  # permanent, D3 -- see module docstring
    return {"present": present, "absent": absent}


def _anomalies(bundle: DataBundle) -> list[str]:
    """D6's `anomalies` flag (86bbummwp Tier 2) -- one cross-check between two
    management-signal fields computed independently from the same
    insider-transaction fetch (`precompute/research_sources.py`) and never
    compared before now. See module docstring for the real-world caveat on
    how often `buyback_activity == "active"` actually occurs."""
    ms = bundle.research_sources.management_signals
    if ms.buyback_activity == "active" and ms.insider_net_direction_90d == "selling":
        return ["buyback program active while insiders are net sellers over the past 90 days"]
    return []


def _stale_data(bundle: DataBundle) -> list[str]:
    """D6's `stale_data` flag (86bbummwp Tier 2) -- see module docstring for
    why `latest_transcript_age_days` is never checked here."""
    rs = bundle.research_sources
    stale = []
    if rs.latest_filing_age_days is not None and rs.latest_filing_age_days > _STALE_FILING_DAYS:
        stale.append("filings")
    if rs.latest_news_age_days is not None and rs.latest_news_age_days > _STALE_NEWS_DAYS:
        stale.append("news")
    return stale


def _business_description(bundle: DataBundle) -> RenderedField:
    business = next(
        (d for d in bundle.research_sources.filing_digests if d.section == "Business"), None
    )
    if business:
        return RenderedField(text=business.content, present=True)
    return RenderedField(text="N/A — no filing digest available for this name.", present=False)


def _filing_highlights(bundle: DataBundle) -> RenderedField:
    digests = bundle.research_sources.filing_digests
    if not digests:
        return RenderedField(text="N/A", present=False)
    text = "\n".join(f"FILING:{d.section}: {d.content}" for d in digests)
    return RenderedField(text=text, present=True)


def _earnings_transcript(bundle: DataBundle) -> RenderedField:
    # Permanently absent today, not a per-run data problem (D3, see module
    # docstring: transcript_excerpts is a deliberate, permanent [] until a
    # future ticket builds real transcript sourcing) -- present will always
    # be False here for every real run, not a bug in this field.
    excerpts = bundle.research_sources.transcript_excerpts
    if not excerpts:
        return RenderedField(
            text="N/A — earnings transcript excerpts are not available (not yet built into the data pipeline).",
            present=False,
        )
    text = "\n".join(f"[{t.quarter} {t.type}] {t.content}" for t in excerpts)
    return RenderedField(text=text, present=True)


def _recent_developments(bundle: DataBundle) -> RenderedField:
    # v1.3 drops "low" quality_tier headlines before the LLM sees them
    # (precompute/research_sources.py's own documented behavior) -- this is
    # the "still-unbuilt code" that module's docstring names as responsible
    # for that filter.
    items = [i for i in bundle.research_sources.news_items if i.quality_tier != "low"]
    if not items:
        return RenderedField(text="  (none available)", present=False)
    text = "\n".join(
        f"  {item.id}: {item.headline} ({item.source}, {item.date.date().isoformat()})"
        for item in items
    )
    return RenderedField(text=text, present=True)


def _management_signals(bundle: DataBundle) -> str:
    ms = bundle.research_sources.management_signals
    lines = [
        f"  Insider activity (90d): {ms.insider_net_direction_90d or 'unknown'}",
        f"  Buyback activity: {ms.buyback_activity}",
        f"  Dividend activity: {ms.dividend_activity}",
    ]
    if ms.c_suite_changes_12mo is not None:
        lines.append(f"  C-suite changes (12mo): {ms.c_suite_changes_12mo}")
    return "\n".join(lines)


def _peers_block(bundle: DataBundle) -> RenderedField:
    peers = bundle.research_sources.peer_blocks
    if not peers:
        return RenderedField(text="PEER COMPARABLES: N/A — no peer data available for this name.", present=False)
    lines = [f"  {p.peer_id}: {p.content}" for p in peers]
    return RenderedField(text="PEER COMPARABLES:\n" + "\n".join(lines), present=True)


def _beta_line(bundle: DataBundle) -> RenderedField:
    # Split out of _price_context on its own -- beta is a real, sometimes-
    # missing field (module docstring's own note), unlike current_price/
    # market_cap/52w range, which are plain quote passthroughs virtually
    # always present for any real, already-resolved ticker. Tracking
    # presence at the whole-block level would conflate an always-there
    # field with a genuinely sometimes-absent one.
    beta = bundle.risk_metrics.get("beta")
    text = f"  Beta: {beta if beta is not None else 'N/A'}"
    return RenderedField(text=text, present=beta is not None)


def _price_context(bundle: DataBundle) -> str:
    p = bundle.price_info
    currency = p.get("currency")
    lines = [
        f"  Current: {p.get('current_price')} {currency} | 52w High: {p.get('high_52w')} | 52w Low: {p.get('low_52w')}",
        f"  Market Cap: {p.get('market_cap', 'N/A')}",
        _beta_line(bundle).text,
    ]
    return "\n".join(lines)


def _dividend_context(bundle: DataBundle) -> RenderedField:
    d = bundle.dividend_info
    yield_pct = d.get("dividend_yield")
    payout_pct = d.get("payout_ratio")
    yield_str = f"{yield_pct * 100:.2f}%" if yield_pct is not None else "N/A"
    payout_str = f"{payout_pct * 100:.2f}%" if payout_pct is not None else "N/A"
    text = f"  Yield: {yield_str} | Payout ratio: {payout_str}"
    # present if either component is real -- the two are independent
    # signals (a name could have a real yield but no payout ratio data, or
    # vice versa), so "both missing" is the honest absent case, not "both
    # required."
    return RenderedField(text=text, present=yield_pct is not None or payout_pct is not None)


def build_user_message(bundle: DataBundle) -> tuple[str, dict[str, bool]]:
    """Returns (rendered user message, field presence map) -- the second
    element is 86bbwachy Phase 4's own new addition, collected from each
    RenderedField-returning helper's own `.present` flag as it's rendered,
    not re-derived separately afterward. management_signals/price_context
    (beyond beta, split out into its own entry) are deliberately NOT in the
    map -- neither ever renders a genuine "N/A" for the whole block (see
    each helper's own code), so there's no meaningful present/absent
    distinction to report for them."""
    ctx = bundle.context
    company_info = bundle.company_info

    # CanadianDataFlags is the real, mechanical-flags-era replacement for the
    # harness fixture's ad-hoc `canadian_data_limited` boolean (None for US
    # stocks, per DataBundle's own None-for-US enforcement -- see
    # data/schemas/canadian_data_flags.py). CanadianDataFlags.sedar_filing_available
    # and research_sources.has_filing_digest are populated from the same
    # underlying fetch (not independent signals) but are genuinely different
    # in scope: this one is CA-only (canadian_data_flags is None for US), the
    # other is market-agnostic -- see research_sources_bundle.py's own
    # docstring for the 2026-09-23 rename that split these two apart in name.
    canadian_flag = ""
    flags = bundle.canadian_data_flags
    if flags is not None and not flags.sedar_filing_available:
        canadian_flag = "\nCANADIAN DATA LIMITED: true — Finnhub sentiment unavailable, social_sentiment=unknown."

    business = _business_description(bundle)
    recent_developments = _recent_developments(bundle)
    filing_highlights = _filing_highlights(bundle)
    earnings_transcript = _earnings_transcript(bundle)
    peers = _peers_block(bundle)
    beta = _beta_line(bundle)
    dividend = _dividend_context(bundle)

    text = f"""COMPANY_X (TICKER_X) | {company_info.get('sector')} | {bundle.stock.exchange} | {bundle.stock.currency}
Timeline: {ctx.timeline} | Account: {ctx.account_type} | As of: {bundle.data_vintage.isoformat()}{canadian_flag}

DATA COVERAGE: {_data_coverage_line(bundle)}

BUSINESS DESCRIPTION:
{business.text}

RECENT DEVELOPMENTS (NEWS):
{recent_developments.text}

MANAGEMENT SIGNALS:
{_management_signals(bundle)}

FILING HIGHLIGHTS:
{filing_highlights.text}

EARNINGS TRANSCRIPT:
{earnings_transcript.text}

{peers.text}

PRICE CONTEXT:
{_price_context(bundle)}

DIVIDEND / INCOME:
{dividend.text}"""

    field_presence = {
        "business_description": business.present,
        "recent_developments": recent_developments.present,
        "filing_highlights": filing_highlights.present,
        "earnings_transcript": earnings_transcript.present,
        "peers_block": peers.present,
        "beta": beta.present,
        "dividend_context": dividend.present,
    }
    return text, field_presence


class StockResearcherRunner(BaseRunner):
    async def run(self, bundle: DataBundle) -> tuple[dict, list[str]]:
        self.current_agent = "RSRCH"
        ctx = bundle.context
        peers = bundle.research_sources.peer_blocks
        user_msg, field_presence = build_user_message(bundle)
        # 86bbwachy Phase 4 -- set before the LLM call is attempted, so a
        # failed call still records whether its own input was already
        # incomplete (see _agent_output_row's own comment on why that's
        # read regardless of completion status).
        self.last_field_coverage = field_presence
        # 86bbummwp Tier 2 -- D6's mechanical flags, set here for the same
        # reason as last_field_coverage above.
        self.last_data_coverage = _data_coverage(bundle)
        self.last_anomalies = _anomalies(bundle)
        self.last_stale_data = _stale_data(bundle)
        # transcript_excerpts is permanently absent (D3) -- excluded here, not in
        # _data_coverage() itself, so the stored/badge-facing fact stays untouched
        # while the confidence/data-quality rule and the new data_quality_assessment
        # rollup below don't fire/downgrade on it every run.
        material_absent = [a for a in self.last_data_coverage["absent"] if a != "transcript_excerpts"]
        # 86bbummwp Tier 3 -- D6 section 3's per-agent mechanical data_quality_assessment,
        # set here for the same reason as last_field_coverage above.
        self.last_data_quality_assessment = compute_data_quality_assessment(
            self.last_stale_data, self.last_anomalies, material_absent
        )
        system_prompt = fill(
            load_template("stock_researcher"),
            {
                # COMPANY_X / TICKER_X, N{num}, and FILING:{section} are the
                # entity-anonymization and citation-notation conventions this
                # agent's own docstring documents -- deliberately absent here
                # so fill() leaves them exactly as the model needs to see
                # them, not substituted with the real name.
                "sector": bundle.company_info.get("sector"),
                "timeline": ctx.timeline,
                "timeline_instruction": f"Timeline: {ctx.timeline}.",
                "data_coverage_line": _data_coverage_line(bundle),
                "data_warnings": render_data_warnings(self.last_anomalies, self.last_stale_data),
                "memory_brief": "",
                "sector_moat_hint": "",
                "has_filing_digest": str(bundle.research_sources.has_filing_digest).lower(),
                # Literal citation tokens, not real peer tickers -- see module
                # docstring for why this differs from the harness.
                "peer_1_token": peers[0].peer_id if len(peers) > 0 else "a peer company",
                "peer_2_token": peers[1].peer_id if len(peers) > 1 else "another peer",
            },
        )
        return await self.call_with_validation(
            system_prompt,
            user_msg,
            partial(
                _validate_with_caveats,
                has_filing_digest=bundle.research_sources.has_filing_digest,
                material_absent=material_absent,
                anomalies=self.last_anomalies,
                stale_data=self.last_stale_data,
            ),
            max_tokens=4000,
            temperature=0.3,
        )
