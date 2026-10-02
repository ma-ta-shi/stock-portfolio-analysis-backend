"""Pass 2 — Tax Strategist runner (v1.5).

Production port of `simulation/runners/pass2_tax_strategist.py` (86bbuhjup).
NOT a mechanical port -- and a genuine SIMPLIFICATION over the harness, not
just a translation:

- The harness manually reconstructs the whole "TAX-RELEVANT DATA" block
  (DIVID/LIST/DOM/WHT/ELIG/ROOM/CGAIN/LOSS lines) field-by-field from
  `fixture["orchestrator_precomputed"]` + `fixture["fundamental_data"]`.
  Production doesn't need to: `_tax_metrics_block()` calls
  `precompute/tax_metrics.py::build_tax_metrics_field()` directly, fresh, every
  call (confirmed by reading that function's real output -- DIVID/ELIG/LIST/
  DOM/WHT/effective-yield/tax-drag/CGAIN/MARG/ROOM/LOSS/US_SITUS/
  TAX_RULE_SNAPSHOT lines, a superset of what the harness rebuilds by hand).
- `account_state` (ROOM remaining, superficial-loss window, YTD realized
  gains/losses) and `user_tax_profile` (province/income/marginal-rate override
  -- drives MARG) are real, per-user inputs threaded in from the orchestrator
  (86bc8efvb), fetched via `UserProfile`/`get_user_profile()`. `DataBundle`
  itself carries neither -- it's frozen (immutable) and both depend on
  per-user data that can't be known at `DataPipeline.prepare()` time, which is
  why there used to be a `bundle.tax_metrics` field built up front and a fast
  path reusing it: that field was removed (86bc8efvb) once it became clear it
  could never stay correct once real per-user data entered the picture, and
  this runner always rebuilds the block fresh instead. Either can genuinely be
  `None` (no `UserProfile` row, or one with fields unset) -- the precompute
  layer already handles that as real, intentional absence, not a gap this
  runner introduces.
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
from agents.validators.pass2 import validate_tax_against_expectation, validate_tax_strategist
from data.precompute.tax_metrics import (
    AccountStateInput,
    UserTaxProfileInput,
    TaxExpectation,
    build_tax_metrics_field,
    compute_tax_expectation,
    load_tax_rules_reference,
)
from data.schemas.data_bundle import DataBundle


# The Pass 1 views the Tax Strategist reads. TECH, SENT and MACRO were ~1.3k tokens of its
# prompt but none of its rules reads them, its own rule 2 calls SENT and TECH orthogonal, and
# measured over 41 real outputs they were cited in 3, 6 and 2 (FUND 19, RSRCH 15). A 43-run A/B
# on the same scenarios (full input vs these two) showed no DETECTABLE loss (first-attempt pass 47% vs
# 45%, account-fit agreement with the anchors 71% vs 68%, passthrough 98% vs 99%); at that sample size
# a difference of a few points is noise, and the oracle does not measure everything TECH/SENT/MACRO
# might add. Revisit if Tax output quality questions come up.
TAX_PASS1_AGENTS = ("RSRCH", "FUND")


def _only_tax_agents(levels: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in levels.items() if k in TAX_PASS1_AGENTS}


def _tax_metrics_block(
    bundle: DataBundle,
    account_type: str,
    account_state: AccountStateInput | None = None,
    user_tax_profile: UserTaxProfileInput | None = None,
) -> str:
    """Always builds fresh (86bc8efvb) -- there is no cached bundle.tax_metrics
    to fall back to any more (that field was removed: it could never be correct
    once account_state/user_tax_profile carry real per-user data DataBundle
    doesn't hold, so the old "reuse the cached string when account_type matches"
    fast path would have silently kept returning a string built before either
    was known). Re-reading the tax rules reference file on every call is an
    accepted cost, per load_tax_rules_reference()'s own documented stance
    ("re-reading one small local file per call is cheap")."""
    return build_tax_metrics_field(
        bundle.stock.ticker,
        account_type,
        bundle,
        account_state=account_state,
        user_tax_profile=user_tax_profile,
    )


def _tax_metrics_field_presence(tax_metrics_text: str, account_type: str) -> dict[str, bool]:
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
    -- both correctly read False there too.

    MARG (86bc8efvb): a simple substring check -- MARG is user-profile-gated,
    same class as ROOM, and follows ROOM's own convention of omitting the line
    entirely when genuinely absent rather than rendering an "unknown"
    placeholder, so presence is just "is the line there at all," no WHT-style
    second condition needed.

    ROOM (86bc8efvb): unlike MARG, ROOM DOES need a second condition -- a
    *stale* room figure still renders (with a "(stale, ...)" suffix, see
    build_precomputed_tax_metrics), so a plain substring check would read
    `True` even when the underlying number can't actually be trusted. Staleness
    is deliberately made to read as absent here, not just annotated in the
    text, so it flows into the same material_absent/confidence-caveat gate
    below as genuine absence -- the point being that whether the agent notices
    the caveat is not left to chance.

    ROOM (Tax Strategist Wave 2): the line now lists BOTH registered rooms
    (`ROOM: TFSA remaining $X | RRSP remaining $Y`, the analysed account first),
    so presence is read off the analysed account's own item -- another account's
    stale or missing figure must not flip it. For a Trading analysis there is no
    room of its own, so `room` stays False exactly as before (the runner already
    excludes it from material_absent for trading)."""
    return {
        "divid": "DIVID:" in tax_metrics_text and "DIVID: no dividend history" not in tax_metrics_text,
        "wht": "WHT (this account," in tax_metrics_text
        and "NOT MODELLED per REF withholding grid" not in tax_metrics_text,
        "marg": "MARG:" in tax_metrics_text,
        "room": _room_present(tax_metrics_text, account_type),
    }


def _room_present(tax_metrics_text: str, account_type: str) -> bool:
    if account_type == "trading":
        return False
    wanted = f"{account_type.upper()} remaining"
    for line in tax_metrics_text.splitlines():
        if line.startswith("ROOM:"):
            for item in line[len("ROOM:") :].split(" | "):
                if item.strip().startswith(wanted):
                    return "(stale" not in item
    return False


def _validate_with_caveats(
    output: dict,
    material_absent: list[str],
    account_type: str,
    expectation: TaxExpectation | None = None,
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
    if expectation is not None:
        # The code decides the numbers and the verdict (fit, efficiency, cross-account move); the model
        # must copy them. Part of the same retry loop as every other check.
        expectation_errors = validate_tax_against_expectation(output, expectation)
        passed = passed and not expectation_errors
        errors = errors + expectation_errors
    gs = output.get("groundedness_score")
    cq_passed, cq_errors = validate_confidence_requires_caveat_when_flagged(
        output,
        is_high=isinstance(gs, (int, float)) and gs >= GROUNDEDNESS_HIGH_THRESHOLD,
        material_absent=material_absent,
    )
    return passed and cq_passed, errors + cq_errors


def get_system_prompt(
    bundle: DataBundle,
    compressed_pass1: dict,
    account_type: str,
    account_state: AccountStateInput | None = None,
    user_tax_profile: UserTaxProfileInput | None = None,
    tax_metrics_text: str | None = None,
) -> str:
    # `tax_metrics_text`: the runner builds the block once and hands it to both this and
    # build_user_message (the presence flags are read off the same text); omitted, it is
    # built here, as before.
    # fill(), not .format() -- the real prompt's Output Schema block embeds
    # literal JSON, which .format() reads as placeholders and raises KeyError.
    ctx = bundle.context
    confidence_levels = _only_tax_agents(extract_confidence_levels(compressed_pass1))
    # 86bbummwp Tier 3 -- see pass2_bull_advocate.py's own comment on this
    # same addition for why both signals are shown, never collapsed.
    quality_levels = _only_tax_agents(extract_data_quality_levels(compressed_pass1))
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
            "precomputed_tax_metrics": (
                tax_metrics_text
                if tax_metrics_text is not None
                else _tax_metrics_block(bundle, account_type, account_state, user_tax_profile)
            ),
            # 86bc8efvb deliberately leaves this "" -- Rule 13 references
            # user_tax_context.expected_retirement_marginal_rate_pct, a field
            # that was never added to UserProfile (retirement-timeline fields
            # were explicitly cut during that ticket's design). Populating this
            # with province/income prose would just be a second, redundant
            # channel for what MARG (the token, above) already carries
            # correctly. Rule 13's wording is a separate, real prompt-revision
            # issue, not fixed here.
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
    account_state: AccountStateInput | None = None,
    user_tax_profile: UserTaxProfileInput | None = None,
    tax_metrics_text: str | None = None,
) -> tuple[str, dict[str, bool]]:
    """The user message is the Pass 1 summaries only. The pre-computed tax block, the Pass 1
    reliability warnings and the analysed account are in the system prompt (get_system_prompt)
    and used to be appended here as well, so every call paid for each of them twice (~1k
    tokens). The second return value is the presence flags read off the block."""
    base = build_pass2_user_message(bundle, compressed_pass1, account_type, TAX_PASS1_AGENTS)
    if tax_metrics_text is None:
        tax_metrics_text = _tax_metrics_block(bundle, account_type, account_state, user_tax_profile)
    return base, _tax_metrics_field_presence(tax_metrics_text, account_type)


class TaxStrategistRunner(BaseRunner):
    async def run(
        self,
        bundle: DataBundle,
        compressed_pass1: dict,
        account_type: str | None = None,
        account_state: AccountStateInput | None = None,
        user_tax_profile: UserTaxProfileInput | None = None,
    ) -> tuple[dict, list[str]]:
        self.current_agent = "tax"
        acct = account_type or bundle.context.account_type
        tax_metrics_text = _tax_metrics_block(bundle, acct, account_state, user_tax_profile)
        system_prompt = get_system_prompt(
            bundle, compressed_pass1, acct, account_state, user_tax_profile, tax_metrics_text
        )
        user_msg, field_presence = build_user_message(
            bundle, compressed_pass1, acct, account_state, user_tax_profile, tax_metrics_text
        )
        # 86bbwachy Phase 4 -- set before the LLM call is attempted, so a
        # failed call still records whether its own input was already
        # incomplete.
        self.last_field_coverage = field_presence
        # wht is permanently absent for us_reit/limited_partnership/adr classifications
        # (WHT_GRID has zero entries for any of them) -- excluded here, not in
        # _tax_metrics_field_presence() itself, so input_field_coverage's own stored
        # fact stays untouched while the new confidence/data-quality rule doesn't fire
        # on it every run for those classifications.
        # room (86bc8efvb): ROOM never renders for a Trading account at all
        # (build_precomputed_tax_metrics's own ROOM branch only handles
        # tfsa/rrsp) -- that's architecturally correct absence, not missing
        # data, the same class of exclusion as wht's above. Excluded here, same
        # reasoning, same place.
        excluded_when_absent = {"wht"} | ({"room"} if acct == "trading" else set())
        material_absent = [
            k for k, v in field_presence.items() if not v and k not in excluded_when_absent
        ]
        expectation = compute_tax_expectation(
            bundle.stock.ticker, acct, bundle, user_tax_profile
        )
        result, errors = await self.call_with_validation(
            system_prompt,
            user_msg,
            partial(
                _validate_with_caveats,
                material_absent=material_absent,
                account_type=acct,
                expectation=expectation,
            ),
            max_tokens=4000,
            temperature=0.3,
        )
        # Strip any recommendation field (defense in depth)
        if result and "recommendation" in result:
            del result["recommendation"]
        return result, errors
