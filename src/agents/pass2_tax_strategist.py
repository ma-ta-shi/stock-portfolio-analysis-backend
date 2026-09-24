"""Pass 2 — Tax Strategist runner (v1.5).

Production port of `simulation/runners/pass2_tax_strategist.py` (86bbuhjup).
NOT a mechanical port -- and a genuine SIMPLIFICATION over the harness, not
just a translation:

- The harness manually reconstructs the whole "TAX-RELEVANT DATA" block
  (DIVID/LIST/DOM/WHT/ELIG/ROOM/CGAIN/LOSS lines) field-by-field from
  `fixture["orchestrator_precomputed"]` + `fixture["fundamental_data"]`.
  Production doesn't need to: `DataBundle.tax_metrics` IS that exact block,
  already fully rendered by `precompute/tax_metrics.py::build_tax_metrics_field()`
  at `DataPipeline.prepare()` time (confirmed by reading that function's real
  output -- DIVID/ELIG/LIST/DOM/WHT/effective-yield/tax-drag/CGAIN/ROOM/LOSS/
  US_SITUS/TAX_RULE_SNAPSHOT lines, a superset of what the harness rebuilds
  by hand). This runner reads it directly rather than re-deriving it.
- Real, load-bearing wrinkle this simplification surfaces: `bundle.tax_metrics`
  is fixed at DataBundle-construction time for ONE account_type
  (`bundle.context.account_type`) -- `DataPipeline.prepare()` only ever builds
  it for the context it was given. If this runner is called with an
  `account_type` override that DIFFERS from `bundle.context.account_type`
  (the same override capability the harness's own `run()` signature has),
  `bundle.tax_metrics` would silently describe the WRONG account. Handled by
  calling `build_tax_metrics_field()` fresh for that specific account_type in
  the override case -- it's a plain, callable function (not tied to pipeline
  internals) that accepts a real DataBundle directly.
- `account_state` (ROOM remaining, superficial-loss window, YTD realized
  gains/losses) stays `None` -- not a DataBundle field, and no real service
  produces one yet. Matches `build_precomputed_tax_metrics()`'s own
  documented normal case ("account_state is genuinely, usually None in
  practice today because no real service exists to produce one") -- not a
  gap this runner introduces.
- `{tax_rules_reference}` (the full reference document text for the system
  prompt) now reads via `precompute/tax_metrics.py::load_tax_rules_reference()`
  directly, rather than duplicating a second, independently-maintained
  file-path-and-read implementation the way the harness's own `_load_tax_rules()`
  did -- one real source for "how to find and read the tax rules file."

Uses groundedness_score (NOT confidence).
NO recommendation or archetype field.
Passthroughs (±0.1 tolerance): dividend_yield_pct, withholding_tax_rate_pct, effective_after_tax_yield_pct.
account_fit_score: excellent|good|fair|poor — "poor" mandates non-null cross_account_recommendation.
"""
from agents.base import BaseRunner
from agents.compression import build_pass2_user_message, extract_confidence_levels
from agents.prompts import fill, load_template
from agents.utils import build_pass1_reliability_warnings, researcher_thesis_archetype
from agents.validators.pass2 import validate_tax_strategist
from data.precompute.tax_metrics import build_tax_metrics_field, load_tax_rules_reference
from data.schemas.data_bundle import DataBundle


def _tax_metrics_block(bundle: DataBundle, account_type: str) -> str:
    if account_type == bundle.context.account_type and bundle.tax_metrics is not None:
        return bundle.tax_metrics
    # Override case: bundle.tax_metrics was built for a different account_type
    # than this call needs -- rebuild fresh for the one actually requested.
    return build_tax_metrics_field(bundle.stock.ticker, account_type, bundle)


def get_system_prompt(bundle: DataBundle, compressed_pass1: dict, account_type: str) -> str:
    # fill(), not .format() -- the real prompt's Output Schema block embeds
    # literal JSON, which .format() reads as placeholders and raises KeyError.
    ctx = bundle.context
    confidence_levels = extract_confidence_levels(compressed_pass1)
    tax_rules_reference, _ = load_tax_rules_reference()

    template = load_template("tax_strategist")
    return fill(
        template,
        {
            "ticker": bundle.stock.ticker,
            "company_name": bundle.company_info.get("name"),
            "sector": bundle.company_info.get("sector"),
            "timeline": ctx.timeline,
            "timeline_instruction": f"Timeline: {ctx.timeline}.",
            "account_type": account_type,
            "account_instruction": f"Account: {account_type.upper()}.",
            "tax_rules_reference": tax_rules_reference,
            "precomputed_tax_metrics": _tax_metrics_block(bundle, account_type),
            "user_tax_context": "",
            "memory_brief": "",
            "accuracy_brief": "",
            "winning_patterns_brief": "",
            "pass1_reliability_warnings": build_pass1_reliability_warnings(confidence_levels) or "(none)",
            "researcher_thesis_archetype": researcher_thesis_archetype(compressed_pass1),
        },
    )


def build_user_message(
    bundle: DataBundle,
    compressed_pass1: dict,
    account_type: str,
) -> str:
    base = build_pass2_user_message(bundle, compressed_pass1, account_type)
    confidence_levels = extract_confidence_levels(compressed_pass1)
    warnings = build_pass1_reliability_warnings(confidence_levels)
    if warnings:
        base += f"\n\nRELIABILITY WARNINGS: {warnings}"

    base += f"""

TAX-RELEVANT DATA (PRE-COMPUTED BY ORCHESTRATOR):
{_tax_metrics_block(bundle, account_type)}

ANALYSIS TARGET ACCOUNT: {account_type.upper()}"""

    return base


class TaxStrategistRunner(BaseRunner):
    async def run(
        self,
        bundle: DataBundle,
        compressed_pass1: dict,
        account_type: str | None = None,
    ) -> tuple[dict, list[str]]:
        self.current_agent = "tax"
        acct = account_type or bundle.context.account_type
        system_prompt = get_system_prompt(bundle, compressed_pass1, acct)
        user_msg = build_user_message(bundle, compressed_pass1, acct)
        result, errors = await self.call_with_validation(
            system_prompt,
            user_msg,
            validate_tax_strategist,
            max_tokens=4000,
            temperature=0.3,
        )
        # Strip any recommendation field (defense in depth)
        if result and "recommendation" in result:
            del result["recommendation"]
        return result, errors
