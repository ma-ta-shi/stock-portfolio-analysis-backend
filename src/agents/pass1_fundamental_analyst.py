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
  assembly). `_data_coverage_line()` (86bbummwp Tier 1a) now reads a coarse
  subset of the same `missing_fields`-derived presence map -- only
  `peers_block`/`earnings_surprises`, not all 15 per-ratio keys, which stay
  granular-only in `input_field_coverage` and would be too noisy for a single
  prose line.

86bbummwp Tier 2: D6's mechanical flags, now real, stored fields on
`agent_outputs`.
- `data_coverage`: built via the shared `to_data_coverage()` helper from the
  same `field_presence`/`_COVERAGE_GAP_SENTENCES` pair already used for the
  prose line above -- no new object needed.
- `anomalies`: extends `_valid_peer_value()`'s already-tuned plausibility
  ranges (`precompute/fundamentals.py::_PEER_METRIC_VALID_RANGE`) to the
  PRIMARY stock's own values, not just peers. Deliberately a separate,
  additive check in THIS file, not a change to `_valid_peer_value()` or its
  call sites in `fundamentals.py` -- the primary stock's own out-of-range
  number still renders exactly as before (a real, if implausible, fact the
  LLM should see), this only adds a second, mechanical flag alongside it.
  Only 6 of the range dict's 8 metrics have a real primary-stock field
  anywhere on `DataBundle` -- `pb_ratio`/`ev_ebitda` are peer-only fields
  with no subject-stock equivalent (confirmed directly against
  `valuation_metrics`'s real keys), so those two are skipped here, not
  fabricated.
- `stale_data`: the one item needing an actual precompute-layer change, not
  just orchestrator wiring -- `latest_financials_period_end` (real, already
  fetched on every quarter, never forwarded past `compute_all()` before this)
  is now a real field on `DataBundle`
  (`fundamentals.py::compute_all()` + `data/pipeline.py`'s bundle assembly +
  `data/schemas/data_bundle.py`), giving Fundamental Analyst its first real
  freshness signal anywhere. Threshold is D6's own suggested ~180 days
  (`docs/technical/pass1-confidence-model.md`).
"""
from datetime import date
from functools import partial

from agents.base import BaseRunner
from agents.prompts import fill, load_template
from agents.utils import RenderedField, render_data_coverage_line, render_data_warnings, to_data_coverage
from agents.validators.common import validate_confidence_requires_caveat_when_flagged
from agents.validators.pass1 import validate_fundamental_analyst
from data.precompute.fundamentals import _PEER_METRIC_VALID_RANGE
from data.schemas.data_bundle import DataBundle

# metric name -> (DataBundle attribute, dict key) for the 6 of
# _PEER_METRIC_VALID_RANGE's 8 metrics that have a real primary-stock field.
_PRIMARY_PLAUSIBILITY_FIELDS = {
    "pe_ratio": ("valuation_metrics", "pe_ratio"),
    "debt_to_equity": ("balance_sheet_metrics", "debt_to_equity"),
    "revenue_growth_yoy": ("growth_metrics", "revenue_growth_yoy"),
    "gross_margin": ("profitability_metrics", "gross_margin"),
    "operating_margin": ("profitability_metrics", "operating_margin"),
    "roe": ("profitability_metrics", "roe"),
}
_STALE_FINANCIALS_DAYS = 180  # D6's own suggested threshold

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


# Coarse-only subset of the full field_presence map for the DATA COVERAGE
# line (86bbummwp Tier 1a) -- the other 15 per-ratio keys in
# _missing_fields_presence() are real signal for input_field_coverage but
# too granular for a single prose sentence. No pre-filtering needed here:
# render_data_coverage_line() itself already ignores any field_presence key
# not in gap_sentences (found during critical review -- an earlier version of
# this function pre-filtered field_presence down to just these two keys
# first, which was both redundant with that check and a latent KeyError risk
# if either key were ever absent).
_COVERAGE_GAP_SENTENCES = {
    "peers_block": "no peer data available",
    "earnings_surprises": "no earnings surprise history available",
}


def _data_coverage_line(field_presence: dict[str, bool]) -> str:
    return render_data_coverage_line(field_presence, _COVERAGE_GAP_SENTENCES)


def _missing_fields_presence(bundle: DataBundle) -> dict[str, bool]:
    missing = set(bundle.missing_fields)
    return {name: full_key not in missing for name, full_key in _MISSING_FIELDS_MAP.items()}


def _anomalies(bundle: DataBundle) -> list[str]:
    """D6's `anomalies` flag (86bbummwp Tier 2) -- see module docstring for
    why this is a separate, additive check rather than a change to
    `_valid_peer_value()` itself."""
    flags = []
    for metric, (bucket_name, field) in _PRIMARY_PLAUSIBILITY_FIELDS.items():
        value = getattr(bundle, bucket_name).get(field)
        if value is None:
            continue
        low, high = _PEER_METRIC_VALID_RANGE[metric]
        if not (low < value < high):
            flags.append(f"{field}={value} is outside the plausible range ({low}, {high})")
    return flags


def _stale_data(bundle: DataBundle) -> list[str]:
    """D6's `stale_data` flag (86bbummwp Tier 2) -- see module docstring for
    why this required a precompute-layer change."""
    period_end = bundle.latest_financials_period_end
    if period_end is None:
        return []
    age_days = (bundle.data_vintage.date() - date.fromisoformat(period_end)).days
    return ["financials"] if age_days > _STALE_FINANCIALS_DAYS else []


def _validate_with_caveats(
    output: dict,
    material_absent: list[str],
    anomalies: list[str],
    stale_data: list[str],
) -> tuple[bool, list[str]]:
    """Composing validator (86bbummwp follow-on) -- this agent had no
    caveat-specific validator to compose with `validate_fundamental_analyst`
    before now (unlike Technical/Sentiment/Stock Researcher), so this wrapper
    is new here, not extended, but follows the identical pattern: the base
    schema check plus the new confidence/data-quality coupling rule, the
    backstop for a real, live case (Macro Economist claiming
    `analysis_confidence: "high"` while its own stale_data flag showed real
    staleness) -- see `validate_confidence_requires_caveat_when_flagged`'s
    own docstring."""
    passed, errors = validate_fundamental_analyst(output)
    cq_passed, cq_errors = validate_confidence_requires_caveat_when_flagged(
        output,
        is_high=output.get("analysis_confidence") == "high",
        material_absent=material_absent,
        anomalies=anomalies,
        stale_data=stale_data,
    )
    return passed and cq_passed, errors + cq_errors


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
        # 86bbummwp Tier 2 -- D6's mechanical flags, set here for the same
        # reason as last_field_coverage above.
        self.last_data_coverage = to_data_coverage(field_presence, _COVERAGE_GAP_SENTENCES)
        self.last_anomalies = _anomalies(bundle)
        self.last_stale_data = _stale_data(bundle)
        system_prompt = fill(
            load_template("fundamental_analyst"),
            {
                "canonical_ticker": bundle.stock.ticker,
                "company_name": bundle.company_info.get("name"),
                "sector": bundle.company_info.get("sector"),
                "timeline": ctx.timeline,
                "timeline_instruction": f"Timeline: {ctx.timeline}.",
                "data_coverage_line": _data_coverage_line(field_presence),
                "data_warnings": render_data_warnings(self.last_anomalies, self.last_stale_data),
                "memory_brief": "",
                "sector_specific_valuation_instruction": "",
            },
        )
        return await self.call_with_validation(
            system_prompt,
            user_msg,
            partial(
                _validate_with_caveats,
                material_absent=self.last_data_coverage["absent"],
                anomalies=self.last_anomalies,
                stale_data=self.last_stale_data,
            ),
            max_tokens=4000,
            temperature=0.3,
        )
