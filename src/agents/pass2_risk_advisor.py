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
- `CORR`/`CONC` (portfolio correlation / concentration risk): still honest
  placeholders ("unknown (not provided for this analysis)" / a generic
  literal) -- no portfolio system exists yet, confirmed by this runner's own
  `run_stage_b()` docstring below, which already documents the same gap for
  `portfolio_context`. Not something this port can close.

Uses groundedness_score (NOT confidence).
NO recommendation field.
beta and max_drawdown_1yr are orchestrator-injected (NOT LLM-produced).
downside_scenarios: 2-4 items, no two within ±2% estimated_impact_pct.
stop_loss_suggestion: null for TFSA/RRSP medium/long-term; numeric for Trading short-term.
position_size_recommendation format: "X-Y%".
key_factors: sentiment must be negative|neutral only.
"""
from agents.base import BaseRunner
from agents.compression import build_pass2_user_message, extract_confidence_levels
from agents.prompts import fill, load_template
from agents.utils import build_pass1_reliability_warnings, researcher_thesis_archetype
from agents.validators.pass2 import validate_risk_advisor_stage_a, validate_risk_advisor_stage_b
from data.schemas.data_bundle import DataBundle


def _fmt(v, suffix: str = ""):
    return "N/A" if v is None else f"{v}{suffix}"


def _precomputed_risk_metrics(bundle: DataBundle) -> str:
    rm = bundle.risk_metrics
    pp = bundle.price_position
    return (
        f"BETA: beta = {_fmt(rm.get('beta'))} (will be injected into output by orchestrator)\n"
        f"VOL:  52w High={_fmt(pp.get('high_52w'))}, 52w Low={_fmt(pp.get('low_52w'))}, "
        f"Annualized vol={_fmt(rm.get('annualized_vol_pct'), '%')}\n"
        f"DD:   Max drawdown (1yr) = {_fmt(rm.get('max_drawdown_1yr_pct'), '%')} "
        f"(recovery: {_fmt(rm.get('recovery_1yr_days'), ' days')}) (will be injected by orchestrator)\n"
        f"      Max drawdown (3yr) = {_fmt(rm.get('max_drawdown_3yr_pct'), '%')} "
        f"(recovery: {_fmt(rm.get('recovery_3yr_days'), ' days')})\n"
        f"LIQ:  Avg dollar volume (20d) = {_fmt(rm.get('adv_millions'), 'M')} {rm.get('adv_currency', '')}\n"
        f"CORR: Portfolio correlation = unknown (not provided for this analysis)\n"
        f"CONC: Standard concentration risk assessment based on position sizing"
    )


def build_user_message(
    bundle: DataBundle,
    compressed_pass1: dict,
    account_type: str | None = None,
) -> str:
    base = build_pass2_user_message(bundle, compressed_pass1, account_type)
    confidence_levels = extract_confidence_levels(compressed_pass1)
    warnings = build_pass1_reliability_warnings(confidence_levels)
    if warnings:
        base += f"\n\nRELIABILITY WARNINGS: {warnings}"

    ctx = bundle.context
    acct = account_type or ctx.account_type

    base += f"""

RISK METRIC TOKENS (ORCHESTRATOR PRE-COMPUTED):
{_precomputed_risk_metrics(bundle)}

ACCOUNT: {acct.upper()} | TIMELINE: {ctx.timeline}
STOP-LOSS NOTE: {"null is appropriate (TFSA/RRSP medium/long-term)" if acct in ("tfsa", "rrsp") and ctx.timeline in ("medium_term", "long_term") else "numeric stop-loss may be appropriate"}"""

    return base


class RiskAdvisorRunner(BaseRunner):
    def __init__(self, session=None):
        super().__init__(session=session)
        # Set for real by run() (context=None on total Stage A exhaustion, never a
        # failed attempt's own context -- see BaseRunner._retry_loop's docstring).
        self.last_stage_a_context: list[int] | None = None

    async def run(
        self,
        bundle: DataBundle,
        compressed_pass1: dict,
        account_type: str | None = None,
    ) -> tuple[dict, list[str]]:
        # Stage-specific, not just "risk" -- call_site (86bbwachy Phase 2) is
        # derived from current_agent, so llm_calls can trace Stage A and
        # Stage B as the two distinct round trips they really are, even
        # though AgentOutput still merges them into one row (orchestrator.py's
        # own _run_risk) -- llm_calls is the finer-grained trace that row
        # doesn't need to be.
        self.current_agent = "risk_stage_a"
        acct = account_type or bundle.context.account_type
        user_msg = build_user_message(bundle, compressed_pass1, acct)
        ctx = bundle.context
        confidence_levels = extract_confidence_levels(compressed_pass1)
        system_prompt = fill(
            load_template("risk_advisor", stage="a"),
            {
                "ticker": bundle.stock.ticker,
                "company_name": bundle.company_info.get("name"),
                "sector": bundle.company_info.get("sector"),
                "timeline": ctx.timeline,
                "timeline_instruction": f"Timeline: {ctx.timeline}.",
                "precomputed_risk_metrics": _precomputed_risk_metrics(bundle),
                "researcher_thesis_archetype": researcher_thesis_archetype(compressed_pass1),
                "pass1_reliability_warnings": build_pass1_reliability_warnings(confidence_levels) or "(none)",
                "accuracy_brief": "",
                "winning_patterns_brief": "",
                "memory_brief": "",
            },
        )
        result, errors, context = await self.call_with_validation_start(
            system_prompt,
            user_msg,
            validate_risk_advisor_stage_a,
            max_tokens=5000,
            temperature=0.3,
        )
        # context is None on total exhaustion (never a failed attempt's own context --
        # see BaseRunner._retry_loop's docstring); run_stage_b() must not be called
        # when this is None.
        self.last_stage_a_context = context
        # Orchestrator injects beta and max_drawdown_1yr
        if result and "risk_profile" in result:
            rm = bundle.risk_metrics
            result["risk_profile"]["beta"] = rm.get("beta")
            result["risk_profile"]["max_drawdown_1yr"] = rm.get("max_drawdown_1yr_pct")
        if result and "recommendation" in result:
            del result["recommendation"]
        return result, errors

    async def run_stage_b(
        self,
        bundle: DataBundle,
        context: list[int],
        account_type: str | None = None,
    ) -> tuple[dict, list[str]]:
        """Account-specific sizing/stop-loss overlay, continuing from Stage
        A's own context (see run()'s last_stage_a_context). `context` must
        be a real, success-only context -- callers get this from
        self.last_stage_a_context after confirming run()'s own errors == [].

        No portfolio system exists yet, so portfolio_context is genuinely
        "(none provided)" for every real scenario today -- not a stand-in
        for missing wiring.
        """
        self.current_agent = "risk_stage_b"  # see run()'s own comment on why
        ctx = bundle.context
        acct = account_type or ctx.account_type
        stage_b_prompt = fill(
            load_template("risk_advisor", stage="b"),
            {
                "account_type": acct,
                "timeline": ctx.timeline,
                "account_instruction": f"Account: {acct.upper()}.",
                "portfolio_context": "(none provided)",
                "precomputed_portfolio_fit_metrics": "(none -- no portfolio_context provided)",
            },
        )
        return await self.call_with_validation_continue(
            stage_b_prompt,
            context,
            validate_risk_advisor_stage_b,
            max_tokens=1500,
            temperature=0.3,
        )
