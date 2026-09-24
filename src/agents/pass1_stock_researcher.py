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
"""
from agents.base import BaseRunner
from agents.prompts import fill, load_template
from agents.validators.pass1 import validate_stock_researcher
from data.schemas.data_bundle import DataBundle


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


def _business_description(bundle: DataBundle) -> str:
    business = next(
        (d for d in bundle.research_sources.filing_digests if d.section == "Business"), None
    )
    return business.content if business else "N/A — no filing digest available for this name."


def _filing_highlights(bundle: DataBundle) -> str:
    digests = bundle.research_sources.filing_digests
    if not digests:
        return "N/A"
    return "\n".join(f"FILING:{d.section}: {d.content}" for d in digests)


def _earnings_transcript(bundle: DataBundle) -> str:
    excerpts = bundle.research_sources.transcript_excerpts
    if not excerpts:
        return "N/A — earnings transcript excerpts are not available (not yet built into the data pipeline)."
    return "\n".join(f"[{t.quarter} {t.type}] {t.content}" for t in excerpts)


def _recent_developments(bundle: DataBundle) -> str:
    # v1.3 drops "low" quality_tier headlines before the LLM sees them
    # (precompute/research_sources.py's own documented behavior) -- this is
    # the "still-unbuilt code" that module's docstring names as responsible
    # for that filter.
    items = [i for i in bundle.research_sources.news_items if i.quality_tier != "low"]
    if not items:
        return "  (none available)"
    return "\n".join(
        f"  {item.id}: {item.headline} ({item.source}, {item.date.date().isoformat()})"
        for item in items
    )


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


def _peers_block(bundle: DataBundle) -> str:
    peers = bundle.research_sources.peer_blocks
    if not peers:
        return "PEER COMPARABLES: N/A — no peer data available for this name."
    lines = [f"  {p.peer_id}: {p.content}" for p in peers]
    return "PEER COMPARABLES:\n" + "\n".join(lines)


def _price_context(bundle: DataBundle) -> str:
    p = bundle.price_info
    beta = bundle.risk_metrics.get("beta")
    currency = p.get("currency")
    lines = [
        f"  Current: {p.get('current_price')} {currency} | 52w High: {p.get('high_52w')} | 52w Low: {p.get('low_52w')}",
        f"  Market Cap: {p.get('market_cap', 'N/A')}",
        f"  Beta: {beta if beta is not None else 'N/A'}",
    ]
    return "\n".join(lines)


def _dividend_context(bundle: DataBundle) -> str:
    d = bundle.dividend_info
    yield_pct = d.get("dividend_yield")
    payout_pct = d.get("payout_ratio")
    yield_str = f"{yield_pct * 100:.2f}%" if yield_pct is not None else "N/A"
    payout_str = f"{payout_pct * 100:.2f}%" if payout_pct is not None else "N/A"
    return f"  Yield: {yield_str} | Payout ratio: {payout_str}"


def build_user_message(bundle: DataBundle) -> str:
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

    return f"""COMPANY_X (TICKER_X) | {company_info.get('sector')} | {bundle.stock.exchange} | {bundle.stock.currency}
Timeline: {ctx.timeline} | Account: {ctx.account_type} | As of: {bundle.data_vintage.isoformat()}{canadian_flag}

DATA COVERAGE: {_data_coverage_line(bundle)}

BUSINESS DESCRIPTION:
{_business_description(bundle)}

RECENT DEVELOPMENTS (NEWS):
{_recent_developments(bundle)}

MANAGEMENT SIGNALS:
{_management_signals(bundle)}

FILING HIGHLIGHTS:
{_filing_highlights(bundle)}

EARNINGS TRANSCRIPT:
{_earnings_transcript(bundle)}

{_peers_block(bundle)}

PRICE CONTEXT:
{_price_context(bundle)}

DIVIDEND / INCOME:
{_dividend_context(bundle)}"""


class StockResearcherRunner(BaseRunner):
    async def run(self, bundle: DataBundle) -> tuple[dict, list[str]]:
        self.current_agent = "RSRCH"
        ctx = bundle.context
        peers = bundle.research_sources.peer_blocks
        user_msg = build_user_message(bundle)
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
                "data_warnings": "",
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
            validate_stock_researcher,
            max_tokens=4000,
            temperature=0.3,
        )
