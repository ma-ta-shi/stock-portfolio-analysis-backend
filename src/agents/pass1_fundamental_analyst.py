"""Pass 1 — Fundamental Analyst runner (v5.0).

Production port of `simulation/runners/pass1_fundamental_analyst.py` (86bbuhjup).
NOT a clean port -- field-by-field notes, each verified against
`data/precompute/fundamentals.py` directly, not assumed from the harness fixture:

- All ratio/margin/growth fields on `DataBundle` are raw fractions (e.g.
  `0.142`), not the harness fixture's already-`_pct`-suffixed values (`14.2`) --
  every percentage below is rendered as `value * 100`, not passed through.
- The peer comparison is the industry P/E benchmark (`peer_metrics["industry_benchmark"]`, built by
  `data/industry_benchmark.py`): the median and middle half of the trailing P/E over the stock's Yahoo industry. The
  per-peer table this payload used to carry (5 companies' margins, growth, ROE and leverage) was removed on
  2026-10-03: the advocates cited it for valuation only (40 of 42 citations), and its selection was the weakest part.
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
  `industry_benchmark`/`earnings_surprises`, not all 15 per-ratio keys, which stay
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
from agents.utils import (
    RenderedField,
    as_of_date,
    compute_data_quality_assessment,
    currency_note,
    render_data_coverage_line,
    render_data_warnings,
    to_data_coverage,
)
from agents.validators.common import validate_confidence_requires_caveat_when_flagged
from agents.validators.pass1 import validate_fundamental_analyst
from data.industry_benchmark import is_canadian, pe_vs_industry
from data.precompute.fundamentals import _PEER_METRIC_VALID_RANGE, earnings_surprise_pattern
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
    "industry_benchmark": "no industry P/E benchmark available",
    "earnings_surprises": "no earnings surprise history available",
}


def _data_coverage_line(field_presence: dict[str, bool], limits: str = "") -> str:
    line = render_data_coverage_line(field_presence, _COVERAGE_GAP_SENTENCES)
    if not limits:
        return line
    return f"{limits}." if line == "standard." else f"{line.rstrip('.')}; {limits}."


def _missing_fields_presence(bundle: DataBundle) -> dict[str, bool]:
    missing = set(bundle.missing_fields)
    return {name: full_key not in missing for name, full_key in _MISSING_FIELDS_MAP.items()}


def _anomalies(bundle: DataBundle) -> list[str]:
    """D6's `anomalies` flag (86bbummwp Tier 2) -- see module docstring for
    why this is a separate, additive check rather than a change to
    `_valid_peer_value()` itself."""
    flags = []
    hidden = set((bundle.metric_profile or {}).get("hidden", []))
    for metric, (bucket_name, field) in _PRIMARY_PLAUSIBILITY_FIELDS.items():
        if field in hidden:
            continue
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


def _industry_pe_text(bundle: DataBundle) -> RenderedField:
    """The industry P/E benchmark with its basis, and where this stock's P/E sits in it (computed in code)."""
    na = bundle.not_applicable or {}
    if "pe_ratio" in na.get("fields", []):
        return RenderedField(text=f"Industry P/E: not applicable ({na.get('reason')}: earnings multiples mislead for this kind of company).", present=True)
    bench = bundle.peer_metrics.get("industry_benchmark")
    if not bench:
        return RenderedField(text="Industry P/E: N/A — no industry benchmark available (too few comparable companies, or the data was unavailable).", present=False)
    pct, position = pe_vs_industry(bundle.valuation_metrics.get("pe_ratio"), bench)
    basis = f"{bench['companies']} companies, {bench['industry']}, {bench['market']}, Yahoo trailing P/E"
    if bench["market"] == "US" and is_canadian(bundle.stock.ticker):
        basis += "; the TSX industry has too few companies"
    if position == "not_meaningful":
        where = " | This P/E is not_meaningful against the industry (earnings are near zero)"
    else:
        where = f" | This P/E is {position} ({pct:+.0f}% against the median)" if position else ""
    text = f"Industry P/E: median {bench['median_pe']:.1f}, middle half {bench['p25_pe']:.1f} to {bench['p75_pe']:.1f} ({basis}){where}"
    return RenderedField(text=text, present=True)


def _r2(v) -> str:
    return "N/A" if v is None else f"{v:.2f}"


def _pct(v) -> str:
    return "N/A" if v is None else f"{v * 100:.1f}%"


def _money(v, currency: str | None = None) -> str:
    """373310658259 -> 373.31B USD; 12907000000 -> 12.91B."""
    if v is None:
        return "N/A"
    for limit, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(v) >= limit:
            return f"{v / limit:.2f}{suffix}" + (f" {currency}" if currency else "")
    return f"{v:.0f}" + (f" {currency}" if currency else "")


def _earnings_surprises_text(bundle: DataBundle) -> RenderedField:
    surprises = bundle.growth_metrics.get("earnings_surprises") or []
    if not surprises:
        return RenderedField(text="N/A — no earnings surprise history available.", present=False)
    lines = []
    for s in surprises[:4]:  # newest first for both markets (compute_growth_metrics sorts them)
        eps_actual, eps_est, surprise_pct = s.get("eps_actual"), s.get("eps_estimated"), s.get("eps_surprise_pct")
        surprise_str = f"{surprise_pct:+.1f}%" if surprise_pct is not None else "N/A"
        lines.append(f"  {s.get('period_end', '?')}: EPS actual {_r2(eps_actual)} vs estimate {_r2(eps_est)} ({surprise_str})")
    return RenderedField(text="\n".join(lines), present=True)


def _profile(bundle: DataBundle) -> dict:
    return bundle.metric_profile or {"group": "standard", "hidden": [], "lens": "", "limits": ""}


def build_user_message(bundle: DataBundle) -> tuple[str, dict[str, bool]]:
    """The payload follows the design doc's format (ratios to 2 dp, percentages to 1 dp, magnitudes abbreviated) and is
    built from the stock's metric profile (data/precompute/fundamentals.py::metric_group): a metric the profile hides is
    left out entirely, not marked "N/A" or "not applicable" (the agent wrote "operating margin not applicable" up as a
    risk for TD.TO), and a metric that should be there but is empty still reads N/A."""
    ctx = bundle.context
    company_info = bundle.company_info
    val, growth, prof = bundle.valuation_metrics, bundle.growth_metrics, bundle.profitability_metrics
    bal, div, price, analyst = bundle.balance_sheet_metrics, bundle.dividend_info, bundle.price_info, bundle.analyst_consensus
    profile = _profile(bundle)
    group, hidden = profile["group"], set(profile["hidden"])
    currency = bundle.stock.currency
    earnings_surprises = _earnings_surprises_text(bundle)
    industry_pe = _industry_pe_text(bundle)
    # REIT earnings multiples that came out empty (existing handling, not tuned): "not applicable", not a data gap.
    na = bundle.not_applicable or {}
    na_fields = set(na.get("fields", []))
    na_label = f"not applicable ({na.get('reason')})"

    def shown(label: str, key: str, value, fmt) -> str | None:
        """'Label: value' unless the profile hides the metric; an empty REIT-style metric reads 'not applicable'."""
        if key in hidden:
            return None
        if value is None and key in na_fields:
            return f"{label}: {na_label}"
        return f"{label}: {fmt(value)}"

    def line(*parts: str | None) -> str | None:
        kept = [p for p in parts if p]
        return "  " + " | ".join(kept) if kept else None

    header = (
        f"{bundle.stock.ticker} ({company_info.get('name')}) | {company_info.get('sector')} | "
        f"{company_info.get('industry') or 'industry unknown'} | {bundle.stock.exchange} | {currency}\n"
        f"Timeline: {ctx.timeline} | Account: {ctx.account_type} | As of: {as_of_date(bundle)}"
        f"{currency_note(bundle.currency_mismatch)}"
    )
    eps_growth = growth.get("eps_growth_yoy")
    eps_note = " (over 100%: base-year distortion likely, not used for PEG)" if eps_growth is not None and eps_growth > 1.0 else ""

    val_lines = [
        # Forward P/E and PEG are derived and often undefined (no forward EPS; growth over 100% or near zero), so they are
        # left out when empty instead of reading "PEG: N/A", which the agent turned into a "PEG not available" caveat on
        # every TD.TO run.
        line(shown("P/E", "pe_ratio", val.get("pe_ratio"), _r2),
             shown("Forward P/E", "forward_pe", val.get("forward_pe"), _r2) if val.get("forward_pe") is not None else None,
             shown("PEG", "peg_ratio", val.get("peg_ratio"), _r2) if val.get("peg_ratio") is not None else None,
             shown("P/B", "pb_ratio", val.get("pb_ratio"), _r2),
             shown("P/S", "ps_ratio", val.get("ps_ratio"), _r2), shown("EV/EBITDA", "ev_ebitda", val.get("ev_ebitda"), _r2)),
        None if "pe_ratio" in hidden else f"  {industry_pe.text}",
        f"  Price: {_r2(price.get('current_price'))} {currency} | Market cap: {_money(price.get('market_cap'), currency)} | "
        f"52w range: {_r2(price.get('low_52w'))} - {_r2(price.get('high_52w'))}",
    ]
    growth_lines = [
        line(f"Revenue growth YoY: {_pct(growth.get('revenue_growth_yoy'))} ({growth.get('revenue_growth_yoy_basis') or 'period unknown'})",
             f"Last fiscal year: {_pct(growth.get('revenue_growth_annual'))}", f"3yr CAGR: {_pct(growth.get('revenue_growth_3yr_cagr'))}"),
        f"  EPS growth (last fiscal year): {_pct(eps_growth)}{eps_note}",
    ]
    trend = prof.get("margin_trend")
    prof_lines = [
        line(shown("Gross margin", "gross_margin", prof.get("gross_margin"), _pct),
             shown("Operating margin", "operating_margin", prof.get("operating_margin"), _pct),
             shown("Net margin", "net_margin", prof.get("net_margin"), _pct),
             f"Net margin trend (YoY): {trend or 'N/A'}"),
        line(shown("ROE", "roe", prof.get("roe"), _pct), shown("ROA", "roa", prof.get("roa"), _pct) if "roa" in prof else None),
        line(shown("FCF to net income", "fcf_to_net_income", prof.get("fcf_to_net_income"), _r2)),
    ]
    bal_lines = [
        line(shown("D/E (financial debt, excluding leases)", "debt_to_equity", bal.get("debt_to_equity"), _r2),
             shown("Current ratio", "current_ratio", bal.get("current_ratio"), _r2),
             shown("Interest coverage", "interest_coverage", bal.get("interest_coverage"), _r2)),
        line(f"Equity to assets: {_pct(bal.get('equity_to_assets'))}" if "equity_to_assets" in bal else None,
             f"Cash: {_money(bal.get('cash_position'), currency)}" if group != "financials" else None),
    ]
    if group == "pre_profit":
        fcf, runway = bal.get("free_cash_flow"), bal.get("cash_runway_quarters")
        burn = (f"Free cash flow (TTM): {_money(fcf, currency)}" if fcf is not None else "Free cash flow (TTM): N/A")
        bal_lines.append(line(burn, f"Cash runway: {runway:.1f} quarters at this burn" if runway is not None else
                              ("self-funding (free cash flow is positive)" if fcf is not None and fcf >= 0 else None)))
    if div.get("dividend_regularity") == "none":
        div_lines = ["  No dividend paid."]
    else:
        div_lines = [
            line(shown("Yield", "dividend_yield", div.get("dividend_yield"), _pct), shown("Payout", "payout_ratio", div.get("payout_ratio"), _pct)),
            f"  5yr dividend growth: {_pct(div.get('dividend_growth_5yr'))} | Regularity: {div.get('dividend_regularity', 'N/A')}",
        ]

    def block(title: str, lines: list[str | None]) -> str:
        return title + "\n" + "\n".join(x for x in lines if x)

    sections = [header]
    if profile.get("lens"):
        sections.append(f"LENS: {profile['lens']}")
    sections += [
        block("VALUATION (VAL):", val_lines),
        block("GROWTH (GROWTH):", growth_lines),
        block("PROFITABILITY (PROF):", prof_lines),
        block("BALANCE SHEET (BAL):", bal_lines) if any(bal_lines) else "",
        block("DIVIDEND (DIV):", div_lines),
        block("ANALYST CONSENSUS (ANALYST):", [
            f"  Coverage: {_fmt(analyst.get('num_analysts'))} analysts | Buy: {_fmt(analyst.get('buy_count'))} | "
            f"Hold: {_fmt(analyst.get('hold_count'))} | Sell: {_fmt(analyst.get('sell_count'))}",
            f"  Consensus: {_fmt(analyst.get('consensus_rating'))} | Avg target: {_r2(analyst.get('target_mean'))} {currency}"]),
        block("EARNINGS SURPRISE HISTORY (newest first):", [earnings_surprises.text]),
    ]
    text = "\n\n".join(x for x in sections if x)

    field_presence = _missing_fields_presence(bundle)
    field_presence["earnings_surprises"] = earnings_surprises.present
    field_presence["industry_benchmark"] = industry_pe.present
    return text, field_presence


def merge_decided_fields(output: dict, bundle: DataBundle) -> dict:
    """Fields code decides, merged after the model returns (the pattern Technical and Macro use): the earnings surprise
    pattern as `guidance_vs_consensus`. Stored output keeps the same keys, so the Pass 2 view reads the same place."""
    if not isinstance(output, dict):
        return output
    itf = output.get("interpretive_fields")
    itf = dict(itf) if isinstance(itf, dict) else {}
    itf["guidance_vs_consensus"] = earnings_surprise_pattern(bundle.growth_metrics.get("earnings_surprises"))
    return {**output, "interpretive_fields": itf}


class FundamentalAnalystRunner(BaseRunner):
    GROUND_MODE = "once"  # a figure in the narrative, summary or caveats not in this agent's data fails its first attempt only (agents/grounding.py)

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
        # 86bbummwp Tier 3 -- D6 section 3's per-agent mechanical data_quality_assessment,
        # set here for the same reason as last_field_coverage above. No permanent-absence
        # exclusion needed here (unlike RSRCH/SENT) -- confirmed no hardcoded True/False
        # literal exists anywhere in this agent's own field_presence construction.
        self.last_data_quality_assessment = compute_data_quality_assessment(
            self.last_stale_data, self.last_anomalies, self.last_data_coverage["absent"]
        )
        system_prompt = fill(
            load_template("fundamental_analyst"),
            {
                "canonical_ticker": bundle.stock.ticker,
                "company_name": bundle.company_info.get("name"),
                "sector": bundle.company_info.get("sector"),
                "timeline": ctx.timeline,
                "data_coverage_line": _data_coverage_line(field_presence, _profile(bundle).get("limits", "")),
                "data_warnings": render_data_warnings(self.last_anomalies, self.last_stale_data),
            },
        )
        result, errors = await self.call_with_validation(
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
        return merge_decided_fields(result, bundle), errors
