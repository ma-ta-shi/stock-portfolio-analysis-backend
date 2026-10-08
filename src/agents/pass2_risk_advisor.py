"""Pass 2 — Risk Advisor runner (v1.1).

Production port of `simulation/runners/pass2_risk_advisor.py` (86bbuhjup).
NOT a clean port -- field-by-field notes, verified against
`data/precompute/risk_metrics.py::compute_all()`:

- `beta`/`max_drawdown_1yr_pct` (harness: `price_data`/`orchestrator_precomputed`)
  both live on the real, purpose-built `bundle.risk_metrics` dict -- confirmed
  real fields: `beta`, `annualized_vol_pct`, `max_drawdown_1yr_pct`,
  `recovery_1yr_days`, `max_drawdown_3yr_pct`, `recovery_3yr_days`,
  `adv_millions`, `adv_currency`. A genuine superset of the harness fixture
  (recovery-day counts, 3yr drawdown, and a real annualized-vol figure the
  harness never had at all) -- rendered in addition to the harness's original
  fields, not just a name-mapped subset.
- `LIQ` (avg dollar volume): `risk_metrics.py::_adv_millions()`'s own
  docstring confirms it matches `technicals.py`'s `avg_dollar_volume_20`
  exactly (same rolling window), just expressed in millions -- Risk Advisor's
  own purpose-built field is used here rather than reaching into Technical
  Analyst's `liquidity_flags`.
- `YTD` (harness: `price['ytd_return_pct']`): same genuine gap as every other
  Pass 1/2 runner in this port -- no precompute module anywhere computes a
  YTD return. Omitted, not fabricated.
- `CORR`/`CONC` (portfolio correlation / concentration risk): not rendered. No
  portfolio system exists, and the Risk Advisor is account-neutral and per stock;
  portfolio fit belongs to the Portfolio Optimizer.

Uses groundedness_score (NOT confidence).
NO recommendation field.
beta and max_drawdown_1yr are orchestrator-injected (NOT LLM-produced).
downside_scenarios: 2-4 items, no two within ±2% estimated_impact_pct.
key_factors: sentiment must be negative|neutral only.

86bbummwp follow-on: Stage A's own `_validate_with_caveats()` now composes
`validate_risk_advisor_stage_a` with a confidence/data-quality coupling rule
-- if `groundedness_score` is high (see `_validate_with_caveats`'s own
docstring for the threshold) while this agent's own `field_presence` shows a
real gap and `caveats` is empty, validation fails and forces a retry. Real
evidence this matters: `groundedness_score` was found to be exactly 92 on
every real row queried, regardless of ticker or data completeness -- see
`agents/validators/common.py::validate_confidence_requires_caveat_when_flagged`
for the shared mechanism (also used by all 5 Pass 1 agents and Tax
Strategist).
"""
from functools import partial

from agents.base import BaseRunner
from agents.compression import build_pass2_user_message, extract_confidence_levels, extract_data_quality_levels
from agents.prompts import fill, load_template
from agents.utils import build_pass1_reliability_warnings, researcher_thesis_archetype
from agents.validators.common import (
    GROUNDEDNESS_HIGH_THRESHOLD,
    validate_confidence_requires_caveat_when_flagged,
)
from agents.validators.pass2 import validate_risk_advisor_stage_a
from data.schemas.data_bundle import DataBundle


def _fmt(v, suffix: str = ""):
    return "N/A" if v is None else f"{v}{suffix}"


def _precomputed_risk_metrics(bundle: DataBundle) -> tuple[str, dict[str, bool]]:
    """(rendered text, per-field presence map). One combined text block, not
    per-field helpers like Stock Researcher's -- confirmed by direct
    reading: risk_metrics.py's own compute_all() has no single aggregate
    'why' field, each metric independently degrades to None on its own
    insufficient-history threshold (module docstring: 1yr drawdown needs
    ~1yr of bars at 80% coverage, 3yr needs ~3yr, ADV needs 20 days). No
    RenderedField wrapper here (unlike every other in-scope agent's
    per-field helpers) -- a single coarse text+bool pair would answer only
    "was everything here real," and input_field_coverage needs the
    per-field map returned below instead, so the presence side is a dict
    from the start rather than a bool this function would otherwise never
    use. 52w High/Low (VOL's first half) come from price_position, a plain
    quote passthrough virtually always present for a resolved ticker --
    not tracked separately, unlike annualized_vol (VOL's second half),
    which shares BETA's own 60-bar minimum and genuinely can be absent.
    CORR/CONC used to be rendered here as placeholders ("unknown" and a generic
    literal); there is no portfolio context, so they are gone from this block.

    The benchmark the beta is measured against is shown by name (it was computed
    but never rendered). It is context, not a material field, so it is not part
    of the presence map and cannot trip the data-gap caveat rule.
    """
    rm = bundle.risk_metrics
    pp = bundle.price_position
    benchmark = f" vs {rm['benchmark']}" if rm.get("benchmark") else ""
    # R-squared: how much of the stock's movement the benchmark explains (see risk_metrics._beta_and_r2).
    r2 = f", R-squared {rm['beta_r2']}" if rm.get("beta_r2") is not None else ""
    text = (
        f"BETA: beta = {_fmt(rm.get('beta'))}{benchmark} (1-year daily returns{r2})\n"
        f"VOL:  52w High={_fmt(pp.get('high_52w'))}, 52w Low={_fmt(pp.get('low_52w'))}, "
        f"Annualized vol={_fmt(rm.get('annualized_vol_pct'), '%')}\n"
        f"DD:   Max drawdown (1yr) = {_fmt(rm.get('max_drawdown_1yr_pct'), '%')} "
        f"(recovery: {_fmt(rm.get('recovery_1yr_days'), ' business days')})\n"
        f"      Max drawdown (3yr) = {_fmt(rm.get('max_drawdown_3yr_pct'), '%')} "
        f"(recovery: {_fmt(rm.get('recovery_3yr_days'), ' business days')})\n"
        f"LIQ:  Avg dollar volume (20d) = {_fmt(rm.get('adv_millions'), 'M')} {rm.get('adv_currency', '')}"
    )
    field_presence = {
        "beta": rm.get("beta") is not None,
        "annualized_vol": rm.get("annualized_vol_pct") is not None,
        "max_drawdown_1yr": rm.get("max_drawdown_1yr_pct") is not None,
        "max_drawdown_3yr": rm.get("max_drawdown_3yr_pct") is not None,
        "avg_dollar_volume": rm.get("adv_millions") is not None,
    }
    return text, field_presence


def _validate_with_caveats(output: dict, material_absent: list[str]) -> tuple[bool, list[str]]:
    """Composing validator (86bbummwp follow-on) -- Risk Advisor Stage A
    previously called `validate_risk_advisor_stage_a` bare, no composition at
    all. Adds the same confidence/data-quality coupling rule as the 5 Pass 1
    agents, adapted for `groundedness_score` (numeric 0-100, not an enum) --
    `is_high` is `groundedness_score >= GROUNDEDNESS_HIGH_THRESHOLD` (shared
    with Tax Strategist -- see that constant's own comment in
    `validators/common.py` for why 85 is a starting guess, not a verified
    number). `material_absent` comes from this agent's own `field_presence`
    (`beta`/`annualized_vol`/
    `max_drawdown_1yr`/`max_drawdown_3yr`/`avg_dollar_volume`) -- all 5 are
    real per-run signals (insufficient price history for this specific
    ticker/run), unlike Stock Researcher's `transcript_excerpts` or Tax
    Strategist's `wht`, so no exclusion is needed here. Real evidence this
    rule matters: `groundedness_score` was found to be exactly 92 on all 13
    real rows queried, regardless of ticker or real data completeness -- see
    `validate_confidence_requires_caveat_when_flagged`'s own docstring for
    the general mechanism."""
    passed, errors = validate_risk_advisor_stage_a(output)
    gs = output.get("groundedness_score")
    cq_passed, cq_errors = validate_confidence_requires_caveat_when_flagged(
        output,
        is_high=isinstance(gs, (int, float)) and gs >= GROUNDEDNESS_HIGH_THRESHOLD,
        material_absent=material_absent,
    )
    return passed and cq_passed, errors + cq_errors


def get_system_prompt(bundle: DataBundle, compressed_pass1: dict) -> str:
    ctx = bundle.context
    confidence_levels = extract_confidence_levels(compressed_pass1)
    # 86bbummwp Tier 3 -- see pass2_bull_advocate.py's own comment on this
    # same addition for why both signals are shown, never collapsed.
    quality_levels = extract_data_quality_levels(compressed_pass1)
    risk_metrics_text, _ = _precomputed_risk_metrics(bundle)
    return fill(
        load_template("risk_advisor", stage="a"),
        {
            "ticker": bundle.stock.ticker,
            "company_name": bundle.company_info.get("name"),
            "sector": bundle.company_info.get("sector"),
            "timeline": ctx.timeline,
            "timeline_instruction": f"Timeline: {ctx.timeline}.",
            "precomputed_risk_metrics": risk_metrics_text,
            "researcher_thesis_archetype": researcher_thesis_archetype(compressed_pass1),
            "pass1_reliability_warnings": build_pass1_reliability_warnings(
                confidence_levels, agent_quality=quality_levels
            ) or "(none)",
            "accuracy_brief": "",
            "winning_patterns_brief": "",
            "memory_brief": "",
        },
    )


def build_user_message(
    bundle: DataBundle,
    compressed_pass1: dict,
) -> tuple[str, dict[str, bool]]:
    """Stage A's user message is the Pass 1 summaries only, and is account-neutral.

    The risk metrics and the reliability warnings are in the system prompt
    (get_system_prompt) and used to be appended here as well, so every call paid
    for them twice. The account used to appear here too (a header line, an
    `ACCOUNT | TIMELINE` line and a code-generated STOP-LOSS NOTE) although stage A
    is account-neutral: one read per ticker and timeline, shared by every account.
    (Position sizing and the Risk Advisor's account-specific stage B were removed
    2026-10-01: Risk has no loss budget or portfolio to size against.)
    The second return value is the presence map for the risk metrics."""
    base = build_pass2_user_message(bundle, compressed_pass1, account_neutral=True)
    _, field_presence = _precomputed_risk_metrics(bundle)
    return base, field_presence


class RiskAdvisorRunner(BaseRunner):
    GROUND_MODE = "log"  # figures not in this agent's input are recorded in validator_errors, never a retry (agents/grounding.py)

    async def run(
        self,
        bundle: DataBundle,
        compressed_pass1: dict,
    ) -> tuple[dict, list[str]]:
        # call_site (86bbwachy Phase 2) is derived from current_agent; the agent id
        # is still "risk_stage_a" so historical llm_calls rows keep the same name.
        self.current_agent = "risk_stage_a"
        user_msg, field_presence = build_user_message(bundle, compressed_pass1)
        # 86bbwachy Phase 4 -- set before the LLM call is attempted, so a
        # failed call still records whether its own input was already
        # incomplete.
        self.last_field_coverage = field_presence
        system_prompt = get_system_prompt(bundle, compressed_pass1)
        material_absent = [k for k, v in field_presence.items() if not v]
        result, errors, _context = await self.call_with_validation_start(
            system_prompt,
            user_msg,
            partial(_validate_with_caveats, material_absent=material_absent),
            max_tokens=5000,
            temperature=0.3,
        )
        # Orchestrator injects beta and max_drawdown_1yr
        if result and "risk_profile" in result:
            rm = bundle.risk_metrics
            result["risk_profile"]["beta"] = rm.get("beta")
            result["risk_profile"]["max_drawdown_1yr"] = rm.get("max_drawdown_1yr_pct")
        if result and "recommendation" in result:
            del result["recommendation"]
        return result, errors
