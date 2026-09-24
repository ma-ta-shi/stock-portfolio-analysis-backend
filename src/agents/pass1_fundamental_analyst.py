"""Pass 1 — Fundamental Analyst runner (v5.0).

Production port of `simulation/runners/pass1_fundamental_analyst.py` (86bbuhjup).
NOT a clean port -- field-by-field notes, each verified against
`data/precompute/fundamentals.py` directly, not assumed from the harness fixture:

- All ratio/margin/growth fields on `DataBundle` are raw fractions (e.g.
  `0.142`), not the harness fixture's already-`_pct`-suffixed values (`14.2`) --
  every percentage below is rendered as `value * 100`, not passed through.
- `sector_pe_median` (harness name) is `peer_metrics["sector_medians"]
  ["sector_median_pe"]` for real -- `compute_peer_comparison()`'s own docstring
  confirms the `sector_median_*` naming is deliberate, matching the live
  prompt's own placeholder names (a closer match to the prompt than the
  harness fixture's `sector_pe_median` was).
- `guidance_vs_consensus` (harness fixture: a pre-computed input verdict) is
  actually one of Fundamental's own OUTPUT `interpretive_fields`
  (`agents/validators/pass1.py::IF_SPEC_FUNDAMENTAL`) -- an LLM judgment, not
  precomputed data. No precompute module produces a verdict for this. Instead,
  the raw `earnings_surprises` history (`growth_metrics["earnings_surprises"]`,
  actual-vs-estimate EPS/revenue) is rendered so the LLM can form that
  judgment itself, the same "give raw data, let the LLM interpret" pattern
  already used for `health_rating` (also an interpretive_fields output, per
  `compute_balance_sheet_metrics`'s own docstring: "No health_rating --
  resolved as an LLM interpretive_fields value, not this module's job").
- `currency_mismatch` (`compute_all()`'s own return, explicitly documented as
  needed by "the payload builder... to flag or suppress the affected
  multiples") is computed in `precompute/fundamentals.py` but never stored on
  `DataBundle` -- confirmed by reading `data/pipeline.py`'s DataBundle
  construction, which only pulls 6 of `compute_all()`'s 9 return keys.
  Genuine, real gap in the data layer, not something this runner can recover
  (the underlying `fin.currency` isn't a DataBundle field either) -- surfaced
  here, not silently worked around or fabricated.
- `eligible_canadian_dividend`, `analyst_ratings`/`avg_price_target` naming:
  same conventions established in pass1_stock_researcher.py -- the former has
  no discrete field anywhere (omitted, not fabricated); the latter reads
  `bundle.analyst_consensus` (`consensus_rating`/`num_analysts`/`buy_count`/
  `hold_count`/`sell_count`/`target_mean` -- confirmed real field names via
  `NormalizedAnalystRatings`), a Sentiment-primary-consumer field per
  DataBundle's own per-agent grouping comment, cross-consumed here the same
  way Stock Researcher already reads price_info/dividend_info.
- "BASE RELIABILITY SCORE" block: omitted, same D6-retirement reasoning as
  pass1_stock_researcher.py.
- `missing_fields`/`currency_mismatch` (86bbwachy Phase 4): now persisted onto
  DataBundle (`data/schemas/data_bundle.py` + `data/pipeline.py`'s bundle
  assembly) -- the module docstring note above about them being discarded is
  now stale, kept only as history of why this was the one agent needing a
  schema change for input_field_coverage. `_data_coverage_line()` itself is
  UNCHANGED -- deliberately not wired to the new fields, per this ticket's own
  scope split: that prompt line is D6's `data_coverage` concept, explicitly
  deferred as its own unit; `missing_fields` here feeds only the new, separate
  `input_field_coverage` signal build_user_message() returns.
"""
from agents.base import BaseRunner
from agents.prompts import fill, load_template
from agents.utils import RenderedField
from agents.validators.pass1 import validate_fundamental_analyst
from data.schemas.data_bundle import DataBundle

# Maps this prompt's own rendered field name to its "bucket.key" entry in
# compute_all()'s missing_fields list (data/precompute/fundamentals.py) --
# only the valuation/growth/profitability/balance_sheet buckets are scanned
# there (confirmed by reading compute_all() directly); dividend_info/
# peer_metrics/analyst_consensus are never in missing_fields at all, tracked
# separately below instead.
_MISSING_FIELDS_MAP = {
    "pe_ratio": "valuation_metrics.pe_ratio",
    "forward_pe": "valuation_metrics.forward_pe",
    "peg_ratio": "valuation_metrics.peg_ratio",
    "revenue_growth_yoy": "growth_metrics.revenue_growth_yoy",
    "revenue_growth_3yr_cagr": "growth_metrics.revenue_growth_3yr_cagr",
    "eps_growth_yoy": "growth_metrics.eps_growth_yoy",
    "gross_margin": "profitability_metrics.gross_margin",
    "operating_margin": "profitability_metrics.operating_margin",
    "net_margin": "profitability_metrics.net_margin",
    "roe": "profitability_metrics.roe",
    "fcf_to_net_income": "profitability_metrics.fcf_to_net_income",
    "debt_to_equity": "balance_sheet_metrics.debt_to_equity",
    "current_ratio": "balance_sheet_metrics.current_ratio",
    "interest_coverage": "balance_sheet_metrics.interest_coverage",
    "cash_position": "balance_sheet_metrics.cash_position",
}


def _fmt(v, suffix: str = "", pct: bool = False):
    if v is None:
        return "N/A"
    if pct:
        return f"{v * 100:.2f}{suffix}"
    return f"{v}{suffix}"


def _data_coverage_line() -> str:
    # D6's own data_coverage concept, deliberately deferred as a whole unit
    # (86bbwachy Phase 4) -- stays the harness's literal default, not wired
    # to the now-real missing_fields/currency_mismatch fields below.
    return "standard."


def _missing_fields_presence(bundle: DataBundle) -> dict[str, bool]:
    missing = set(bundle.missing_fields)
    return {name: full_key not in missing for name, full_key in _MISSING_FIELDS_MAP.items()}


def _peers_text(bundle: DataBundle) -> RenderedField:
    records = bundle.peer_metrics.get("peer_records", [])
    if not records:
        return RenderedField(text="", present=False)
    lines = []
    for i, r in enumerate(records, 1):
        metrics = ", ".join(f"{k}={v}" for k, v in r.items() if v is not None and k != "ticker")
        lines.append(f"  PEER_{i} ({r.get('ticker', '?')}): {metrics}")
    return RenderedField(text="PEER DATA:\n" + "\n".join(lines), present=True)


def _earnings_surprises_text(bundle: DataBundle) -> RenderedField:
    surprises = bundle.growth_metrics.get("earnings_surprises") or []
    if not surprises:
        return RenderedField(text="N/A — no earnings surprise history available.", present=False)
    lines = []
    for s in surprises[:2]:  # newest first, per get_earnings_surprises' own contract
        eps_actual = s.get("eps_actual")
        eps_est = s.get("eps_estimated")
        surprise_pct = s.get("eps_surprise_pct")
        surprise_str = f"{surprise_pct:.2f}%" if surprise_pct is not None else "N/A"
        lines.append(
            f"  {s.get('period_end', '?')}: EPS actual {eps_actual if eps_actual is not None else 'N/A'} "
            f"vs estimate {eps_est if eps_est is not None else 'N/A'} ({surprise_str} surprise)"
        )
    return RenderedField(text="\n".join(lines), present=True)


def build_user_message(bundle: DataBundle) -> tuple[str, dict[str, bool]]:
    ctx = bundle.context
    company_info = bundle.company_info
    val = bundle.valuation_metrics
    growth = bundle.growth_metrics
    prof = bundle.profitability_metrics
    bal = bundle.balance_sheet_metrics
    div = bundle.dividend_info
    price = bundle.price_info
    analyst = bundle.analyst_consensus
    sector_medians = bundle.peer_metrics.get("sector_medians", {})
    earnings_surprises = _earnings_surprises_text(bundle)
    peers = _peers_text(bundle)

    text = f"""{bundle.stock.ticker} ({company_info.get('name')}) | {company_info.get('sector')} | {bundle.stock.exchange} | {bundle.stock.currency}
Timeline: {ctx.timeline} | Account: {ctx.account_type} | As of: {bundle.data_vintage.isoformat()}

VALUATION (VAL):
  P/E: {_fmt(val.get('pe_ratio'))} vs Sector median: {_fmt(sector_medians.get('sector_median_pe'))}
  Forward P/E: {_fmt(val.get('forward_pe'))} | PEG: {_fmt(val.get('peg_ratio'))}
  Current price: {price.get('current_price')} {bundle.stock.currency} | Market cap: {_fmt(price.get('market_cap'))}
  52w range: {price.get('low_52w')} - {price.get('high_52w')}

GROWTH (GROWTH):
  Revenue growth YoY: {_fmt(growth.get('revenue_growth_yoy'), '%', pct=True)} | 3yr CAGR: {_fmt(growth.get('revenue_growth_3yr_cagr'), '%', pct=True)}
  EPS growth YoY: {_fmt(growth.get('eps_growth_yoy'), '%', pct=True)}

PROFITABILITY (PROF):
  Gross margin: {_fmt(prof.get('gross_margin'), '%', pct=True)} | Operating margin: {_fmt(prof.get('operating_margin'), '%', pct=True)} | Net margin: {_fmt(prof.get('net_margin'), '%', pct=True)}
  ROE: {_fmt(prof.get('roe'), '%', pct=True)}
  FCF to net income: {_fmt(prof.get('fcf_to_net_income'))}

BALANCE SHEET (BAL):
  D/E: {_fmt(bal.get('debt_to_equity'))} | Current ratio: {_fmt(bal.get('current_ratio'))}
  Interest coverage: {_fmt(bal.get('interest_coverage'))} | Cash: {_fmt(bal.get('cash_position'))}

DIVIDEND (DIV):
  Yield: {_fmt(div.get('dividend_yield'), '%', pct=True)} | Payout: {_fmt(div.get('payout_ratio'), '%', pct=True)}
  5yr dividend growth: {_fmt(div.get('dividend_growth_5yr'), '%', pct=True)} | Regularity: {div.get('dividend_regularity', 'N/A')}

ANALYST CONSENSUS (ANALYST):
  Coverage: {_fmt(analyst.get('num_analysts'))} analysts | Buy: {_fmt(analyst.get('buy_count'))} | Hold: {_fmt(analyst.get('hold_count'))} | Sell: {_fmt(analyst.get('sell_count'))}
  Consensus: {_fmt(analyst.get('consensus_rating'))} | Avg target: {_fmt(analyst.get('target_mean'))} {bundle.stock.currency}

EARNINGS SURPRISE HISTORY (for your own guidance-vs-consensus judgment):
{earnings_surprises.text}

{peers.text}"""

    field_presence = _missing_fields_presence(bundle)
    field_presence["earnings_surprises"] = earnings_surprises.present
    field_presence["peers_block"] = peers.present
    return text, field_presence


class FundamentalAnalystRunner(BaseRunner):
    async def run(self, bundle: DataBundle) -> tuple[dict, list[str]]:
        self.current_agent = "FUND"
        ctx = bundle.context
        user_msg, field_presence = build_user_message(bundle)
        # 86bbwachy Phase 4 -- set before the LLM call is attempted, so a
        # failed call still records whether its own input was already
        # incomplete.
        self.last_field_coverage = field_presence
        system_prompt = fill(
            load_template("fundamental_analyst"),
            {
                "canonical_ticker": bundle.stock.ticker,
                "company_name": bundle.company_info.get("name"),
                "sector": bundle.company_info.get("sector"),
                "timeline": ctx.timeline,
                "timeline_instruction": f"Timeline: {ctx.timeline}.",
                "data_coverage_line": _data_coverage_line(),
                "data_warnings": "",
                "memory_brief": "",
                "sector_specific_valuation_instruction": "",
            },
        )
        return await self.call_with_validation(
            system_prompt,
            user_msg,
            validate_fundamental_analyst,
            max_tokens=4000,
            temperature=0.3,
        )
