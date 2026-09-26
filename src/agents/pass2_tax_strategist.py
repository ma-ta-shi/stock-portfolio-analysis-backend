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

86bbummwp follow-on: `_validate_with_caveats()` now composes
`validate_tax_strategist` with two real fixes at once, both landing on the
same call site. (1) `account_type` is finally supplied to
`validate_tax_strategist` -- it had taken this param since 2026-09-01 to
enable "Rule 14" (loss-harvesting must be null for TFSA/RRSP), but the real
call site never passed it, so Rule 14 had never actually fired in
production. (2) The same confidence/data-quality coupling rule as Risk
Advisor and all 5 Pass 1 agents -- see
`agents/validators/common.py::validate_confidence_requires_caveat_when_flagged`
-- gated on `groundedness_score` (real evidence: 90/92 on real rows where
`divid` was genuinely absent). `wht` is excluded from the material-gap check
at the call site: `WHT_GRID` has zero entries for `us_reit`/
`limited_partnership`/`adr` classifications, so it's permanently absent for
those, never a real per-run signal.
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
from agents.validators.pass2 import validate_tax_strategist
from data.precompute.tax_metrics import build_tax_metrics_field, load_tax_rules_reference
from data.schemas.data_bundle import DataBundle


def _tax_metrics_block(bundle: DataBundle, account_type: str) -> str:
    if account_type == bundle.context.account_type and bundle.tax_metrics is not None:
        return bundle.tax_metrics
    # Override case: bundle.tax_metrics was built for a different account_type
    # than this call needs -- rebuild fresh for the one actually requested.
    return build_tax_metrics_field(bundle.stock.ticker, account_type, bundle)


def _tax_metrics_field_presence(tax_metrics_text: str) -> dict[str, bool]:
    """Reads presence directly off the already-rendered tax_metrics block
    (see build_precomputed_tax_metrics in data/precompute/tax_metrics.py)
    -- unlike every other in-scope agent, there is no structured/dict form
    of this data available at the agent layer, only the text
    DataPipeline.prepare() hands it, so presence is read off known marker
    substrings rather than raw field values (86bbwachy Phase 4's own plan
    documents this as this agent's distinct shape, not an oversight).

    WHT: collapses "not modelled for this classification/account" (a
    WHT_GRID miss) and "no data" into one False -- input_field_coverage
    only needs "was this call's WHT line a real, usable rate," not why it
    wasn't.

    DIVID: documented limitation, not fixed here -- compute_trailing_dividend()
    returns (None, 0) for BOTH "genuinely no dividend history" and "history
    exists but nothing fell in the trailing 365-day window" (e.g. a stock
    that suspended its dividend last year) -- both render as the same
    "DIVID: no dividend history" line and both read False here, the same
    conflation this plan's own table documents, not new.

    The ETF early-return path (build_precomputed_tax_metrics's own
    structure=="etf" branch) renders neither a DIVID nor a WHT line at all
    -- both correctly read False there too."""
    return {
        "divid": "DIVID:" in tax_metrics_text and "DIVID: no dividend history" not in tax_metrics_text,
        "wht": "WHT (this account," in tax_metrics_text
        and "NOT MODELLED per REF withholding grid" not in tax_metrics_text,
    }


def _validate_with_caveats(
    output: dict, material_absent: list[str], account_type: str
) -> tuple[bool, list[str]]:
    """Composing validator (86bbummwp follow-on) -- Tax Strategist previously
    called `validate_tax_strategist` bare, never supplying `account_type`
    even though that function's own signature has taken it (for Rule 14,
    loss-harvesting must be null for TFSA/RRSP) since 2026-09-01 -- so Rule
    14 has never actually fired in production. Fixed here as the same call
    site is already being restructured for the new confidence/data-quality
    rule.

    `material_absent` must already exclude `"wht"` (see the caller) -- WHT_GRID
    (`data/precompute/tax_metrics.py`) has zero entries for `us_reit`/
    `limited_partnership`/`adr` dividend classifications, so `wht` is
    PERMANENTLY absent for those, never a real per-run signal; passing it
    through unfiltered would fail validation on every single run for those
    classifications. `divid` is kept -- a real, if imperfect, per-stock
    signal (real evidence: `groundedness_score=90`/`92` on two real rows
    where `divid` was genuinely absent). See
    `validate_confidence_requires_caveat_when_flagged`'s own docstring for
    the general mechanism, and `GROUNDEDNESS_HIGH_THRESHOLD`'s own comment
    (shared with Risk Advisor, `validators/common.py`) for why 85 is a
    starting guess, not a verified number."""
    passed, errors = validate_tax_strategist(output, account_type=account_type)
    gs = output.get("groundedness_score")
    cq_passed, cq_errors = validate_confidence_requires_caveat_when_flagged(
        output,
        is_high=isinstance(gs, (int, float)) and gs >= GROUNDEDNESS_HIGH_THRESHOLD,
        material_absent=material_absent,
    )
    return passed and cq_passed, errors + cq_errors


def get_system_prompt(bundle: DataBundle, compressed_pass1: dict, account_type: str) -> str:
    # fill(), not .format() -- the real prompt's Output Schema block embeds
    # literal JSON, which .format() reads as placeholders and raises KeyError.
    ctx = bundle.context
    confidence_levels = extract_confidence_levels(compressed_pass1)
    # 86bbummwp Tier 3 -- see pass2_bull_advocate.py's own comment on this
    # same addition for why both signals are shown, never collapsed.
    quality_levels = extract_data_quality_levels(compressed_pass1)
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
            "pass1_reliability_warnings": build_pass1_reliability_warnings(
                confidence_levels, agent_quality=quality_levels
            ) or "(none)",
            "researcher_thesis_archetype": researcher_thesis_archetype(compressed_pass1),
        },
    )


def build_user_message(
    bundle: DataBundle,
    compressed_pass1: dict,
    account_type: str,
) -> tuple[str, dict[str, bool]]:
    base = build_pass2_user_message(bundle, compressed_pass1, account_type)
    confidence_levels = extract_confidence_levels(compressed_pass1)
    quality_levels = extract_data_quality_levels(compressed_pass1)
    warnings = build_pass1_reliability_warnings(confidence_levels, agent_quality=quality_levels)
    if warnings:
        base += f"\n\nRELIABILITY WARNINGS: {warnings}"

    tax_metrics_text = _tax_metrics_block(bundle, account_type)
    base += f"""

TAX-RELEVANT DATA (PRE-COMPUTED BY ORCHESTRATOR):
{tax_metrics_text}

ANALYSIS TARGET ACCOUNT: {account_type.upper()}"""

    return base, _tax_metrics_field_presence(tax_metrics_text)


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
        user_msg, field_presence = build_user_message(bundle, compressed_pass1, acct)
        # 86bbwachy Phase 4 -- set before the LLM call is attempted, so a
        # failed call still records whether its own input was already
        # incomplete.
        self.last_field_coverage = field_presence
        # wht is permanently absent for us_reit/limited_partnership/adr classifications
        # (WHT_GRID has zero entries for any of them) -- excluded here, not in
        # _tax_metrics_field_presence() itself, so input_field_coverage's own stored
        # fact stays untouched while the new confidence/data-quality rule doesn't fire
        # on it every run for those classifications.
        material_absent = [k for k, v in field_presence.items() if not v and k != "wht"]
        result, errors = await self.call_with_validation(
            system_prompt,
            user_msg,
            partial(_validate_with_caveats, material_absent=material_absent, account_type=acct),
            max_tokens=4000,
            temperature=0.3,
        )
        # Strip any recommendation field (defense in depth)
        if result and "recommendation" in result:
            del result["recommendation"]
        return result, errors
