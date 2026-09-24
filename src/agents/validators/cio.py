"""CIO validator — validates the two-stage CIO synthesis output.

Split into `validate_cio_stage_a` / `validate_cio_stage_b` (ClickUp 86bbuhk82) to match the real
two-stage prompt split from 86bbdutn6. The old single `validate_cio()` validated a hybrid schema
that never matched either real stage: it required Stage-B-only fields
(`position_sizing_recommendation`, `synthesis_narrative`, `risk_profile_summary`, `tax_summary`)
on Stage A output -- which is the only thing any runner invokes today -- and it nested
`risk_reward_consumed_as`/`tail_risk_cross_check` under a `risk_profile_summary` object that
Stage A's real schema doesn't have at all (those two fields are top-level in Stage A; Stage B
reuses the `risk_profile_summary` name for a completely different pair of fields,
`position_size_source`/`sizing_justification`). Confirmed via a full field-name extraction of
`backend/prompts/cio/v1_stage_a.txt` and `v1_stage_b.txt`, not just their enum-shaped fields.
"""
import re

from agents.utils import PASS1_AGENT_IDS, char_count
from agents.validators.common import _sweep_declared_enums

SOFT_ERROR_PREFIXES = (
    "what_would_change_my_mind: must cite a Pass 1 agent ID",
    "synthesis_narrative: too short",
)


def split_soft_errors(errors: list[str]) -> tuple[list[str], list[str]]:
    """(hard, soft). Soft = citation-breadth rules that must not cost the agent."""
    soft = [e for e in errors if e.startswith(SOFT_ERROR_PREFIXES)]
    return [e for e in errors if not e.startswith(SOFT_ERROR_PREFIXES)], soft


VALID_OUTLOOKS = {"bullish", "somewhat_bullish", "neutral", "somewhat_bearish", "bearish"}
# Aligned to the CIO and Shadow CIO prompts (both agree; the runner and this
# validator did not). The tier is a RELATIVE-performance bucket -- the prompt is
# explicit that it is "not a price target" -- so "outperform" carries the intended
# semantics and "high_upside" (absolute) did not. It is scored against actual
# relative performance, so the wrong vocabulary corrupts calibration, not just
# validation. See prompt-revision-protocol.md 128.3 / 129.1.
VALID_RETURN_TIERS = {"strong_outperform", "outperform", "market_perform",
                      "underperform", "strong_underperform"}
# Ordered worst-to-best so Stage B's "only worse or unchanged" rule (prompt rule 3) can be
# checked as a simple index comparison.
_RETURN_TIER_ORDER = ["strong_underperform", "underperform", "market_perform",
                      "outperform", "strong_outperform"]
VALID_POSITION_SIZES = {"overweight", "marketweight", "underweight", "avoid"}
# PASS2_AGENT_IDS was defined here and never referenced anywhere in the codebase -- dead,
# deleted rather than carried forward into the stage split. PASS1_AGENT_IDS now imported from
# agents.utils (the one canonical definition) rather than redefined locally.


TAIL_RISK_IMPACT_THRESHOLD_PCT = -20.0
# Not the audit's untested -15% figure. Derived from the real worst-case estimated_impact_pct
# distribution across 26 captured Risk Advisor calls (sorted: -15, -18x6, -18.5x4, -25x4, -28,
# -30x3, -35, -40, -45, -75, -80x3) -- -15% barely discriminates (all 26 real worst-cases are
# already -15% or worse, so a -15% threshold puts nearly everything on one side) and sits
# exactly on a recurring real value. -20% sits in an actual gap in the real distribution
# (nothing observed between -18.5 and -25) and splits the historical set into two real-sized
# groups. See the 86bbuhk82 plan for the full record -- still not a rigorously "correct" number,
# just evidence-grounded rather than copied from an audit test that never exercised either
# candidate boundary.


def validate_cio_stage_a(
    output: dict,
    disagreement_category: str = "consensus",
    tail_inputs_present: bool | None = None,
    risk_reward_source: str | None = None,
    bear_tail_risk_level: str | None = None,
    risk_worst_estimated_impact_pct: float | None = None,
) -> tuple[bool, list[str]]:
    """Validate CIO Stage A (account-neutral directional read) output.

    Args:
        output: The CIO Stage A output dict.
        disagreement_category: consensus|mild_dissent|split_decision|high_conflict
        tail_inputs_present: whether Bear's tail_risk_assessment and Risk's downside_scenarios
            were both present in the payload -- supplied by the caller because it depends on the
            payload, not on the CIO's own output (same shape as Tax's account_type parameter).
        risk_reward_source: Risk Advisor Stage A's own risk_reward_ratio value, for the
            cross-check that CIO's risk_reward_consumed_as must match it exactly (prompt rule 11).
        bear_tail_risk_level: Bear's own `tail_risk_level` (elevated|moderate|negligible), and
        risk_worst_estimated_impact_pct: the worst (most negative) `estimated_impact_pct` among
            Risk Advisor's `downside_scenarios` -- together these drive the numeric tail-risk
            cross-check (86bbuhk82 item 2). Both come from the payload, not the CIO's own
            output, same reasoning as `tail_inputs_present`.
    """
    errors: list[str] = []

    for field in (
        "stock_outlook", "confidence", "expected_return_tier", "thesis_summary",
        "what_would_change_my_mind", "bull_case_assessment", "bear_case_assessment",
        "risk_reward_consumed_as", "tail_risk_cross_check",
        "key_decision_factors", "parallel_predictions",
    ):
        if field not in output or output[field] is None:
            errors.append(f"Missing required CIO Stage A field: {field}")

    outlook = output.get("stock_outlook", "")
    if outlook not in VALID_OUTLOOKS:
        errors.append(f"stock_outlook: must be one of {VALID_OUTLOOKS}, got '{outlook}'")

    conf = output.get("confidence")
    if not isinstance(conf, (int, float)) or not (0 <= conf <= 100):
        errors.append(f"confidence: must be 0-100, got {conf}")
    conf = int(conf) if isinstance(conf, (int, float)) else 0

    if disagreement_category == "split_decision" and conf > 65:
        errors.append(
            f"CIO confidence={conf} but disagreement_category=split_decision requires confidence<=65"
        )
    if disagreement_category == "high_conflict" and outlook in {"bullish", "bearish"}:
        errors.append(
            f"CIO stock_outlook='{outlook}' but disagreement_category=high_conflict requires neutral/somewhat range"
        )

    if output.get("expected_return_tier") not in VALID_RETURN_TIERS:
        errors.append(f"expected_return_tier: must be one of {VALID_RETURN_TIERS}")

    wm = output.get("what_would_change_my_mind", "")
    if not wm.strip():
        errors.append("what_would_change_my_mind: must be non-empty")
    has_pass1_ref = any(aid in wm for aid in PASS1_AGENT_IDS)
    # The old check also accepted any of ten generic terms ("price", "growth", ...),
    # which appear in essentially every equity narrative -- so the rule passed always and
    # verified nothing, while its error message claimed "specific Pass 1 agent findings"
    # (§158.3). Now requires an actual agent ID. Listed in SOFT_ERROR_PREFIXES so it
    # applies retry pressure without costing the agent on the final attempt (§122.5).
    if not has_pass1_ref:
        errors.append(
            "what_would_change_my_mind: must cite a Pass 1 agent ID "
            "(RSRCH/FUND/TECH/SENT/MACRO), not merely a generic financial term"
        )

    # bull_case_assessment / bear_case_assessment: the real schema requires a SUBSTANTIVE
    # engagement (Stage 3 decision text: "a substantive engagement with the advocate's
    # strongest_argument (>=40 chars -- agree, refute, or qualify, not a restatement)") --
    # never enforced before this rewrite. verdict/weight_applied's own enum values are
    # covered by the shared DECLARED_ENUMS sweep (see validators/common.py), not here.
    for side in ("bull_case_assessment", "bear_case_assessment"):
        assessment = output.get(side)
        if not isinstance(assessment, dict):
            if side in output:
                errors.append(f"{side}: must be an object, got {type(assessment).__name__}")
            continue
        engagement = str(assessment.get("engagement") or "")
        if len(engagement.strip()) < 40:
            errors.append(
                f"{side}.engagement: must be a substantive engagement (>=40 chars), "
                f"got {len(engagement.strip())}"
            )

    # risk_reward_consumed_as / tail_risk_cross_check are TOP-LEVEL Stage A fields (confirmed
    # via the real v1_stage_a.txt schema) -- not nested under a risk_profile_summary object,
    # which Stage A's schema doesn't have at all (risk_profile_summary is a Stage B field with
    # unrelated content, see validate_cio_stage_b).
    rrc = output.get("risk_reward_consumed_as")
    if rrc is not None and rrc not in {"favorable", "neutral", "unfavorable"}:
        errors.append(f"risk_reward_consumed_as: must be one of favorable|neutral|unfavorable, got {rrc!r}")
    if risk_reward_source is not None and rrc is not None and rrc != risk_reward_source:
        # The matching rule the CIO prompt (rule 11 in the prior single-schema prompt; the
        # Stage A prompt's INPUT SEMANTICS table states the same requirement) mandates this
        # match -- previously implemented but pointed at the wrong (nested) path.
        errors.append(
            f"risk_reward_consumed_as: must match the Risk Advisor's risk_reward_ratio "
            f"{risk_reward_source!r}, got {rrc!r}"
        )

    trc = output.get("tail_risk_cross_check")
    valid_trc = {"aligned", "bear_elevated_risk_advisor_mild",
                 "bear_mild_risk_advisor_elevated", "insufficient_data"}
    if trc is not None and trc not in valid_trc:
        errors.append(f"tail_risk_cross_check: must be one of {valid_trc}, got {trc!r}")
    elif tail_inputs_present is False and trc not in (None, "insufficient_data"):
        # H-b: tail_risk_cross_check must not assert agreement it could not verify.
        # Observed live: the CIO emitted `aligned` when NEITHER Bear's tail_risk_assessment
        # nor Risk's downside_scenarios was in the payload (§162.1).
        errors.append(
            f"tail_risk_cross_check: must be 'insufficient_data' when Bear's "
            f"tail_risk_assessment and Risk's downside_scenarios are absent from the "
            f"payload -- got {trc!r}, which asserts a comparison that could not be made"
        )

    # Numeric cross-check (item 2): the CIO's classification must match what the mapping
    # actually derives from Bear's level and Risk's worst number, not just be a syntactically
    # valid enum value -- E82's own finding was that the CIO gets this right when told the
    # explicit mapping, and gets it wrong (defers to whichever side "argued its case") when
    # not. Only checked when both real inputs are supplied by the caller.
    if (
        trc is not None
        and bear_tail_risk_level is not None
        and risk_worst_estimated_impact_pct is not None
    ):
        R = risk_worst_estimated_impact_pct
        if bear_tail_risk_level == "elevated" and R > TAIL_RISK_IMPACT_THRESHOLD_PCT:
            expected_trc = "bear_elevated_risk_advisor_mild"
        elif bear_tail_risk_level == "negligible" and R < TAIL_RISK_IMPACT_THRESHOLD_PCT:
            expected_trc = "bear_mild_risk_advisor_elevated"
        else:
            expected_trc = "aligned"
        if trc != expected_trc:
            errors.append(
                f"tail_risk_cross_check: given Bear tail_risk_level={bear_tail_risk_level!r} "
                f"and Risk Advisor's worst downside scenario {R}% (threshold "
                f"{TAIL_RISK_IMPACT_THRESHOLD_PCT}%), expected {expected_trc!r}, got {trc!r}"
            )

    kdf = output.get("key_decision_factors", [])
    if not isinstance(kdf, list):
        errors.append("key_decision_factors: must be a list")
    else:
        if len(kdf) < 3:
            errors.append(f"key_decision_factors: need >=3 items, got {len(kdf)}")
        if len(kdf) > 6:
            errors.append(f"key_decision_factors: need <=6 items, got {len(kdf)}")
        for i, kf in enumerate(kdf):
            if isinstance(kf, dict):
                if kf.get("direction") not in {"bullish", "bearish", "neutral"}:
                    errors.append(
                        f"key_decision_factors[{i}].direction: must be bullish|bearish|neutral")
                # Real schema declares 4 values (bull|bear|risk|pass1) -- "tax" was in the old
                # validator's set but Tax Strategist has no input into Stage A at all (the
                # Stage A prompt's own INPUT SEMANTICS table lists only Bull/Bear/Risk).
                if kf.get("source") not in {"bull", "bear", "risk", "pass1"}:
                    errors.append(
                        f"key_decision_factors[{i}].source: must be bull|bear|risk|pass1")
                if not str(kf.get("evidence") or "").strip():
                    errors.append(f"key_decision_factors[{i}].evidence: must be non-empty")

    pp = output.get("parallel_predictions", {})
    if not isinstance(pp, dict):
        errors.append("parallel_predictions: must be a dict")
    else:
        for horizon in ("1_month", "3_month"):
            if horizon not in pp:
                errors.append(f"parallel_predictions.{horizon}: missing")
            else:
                pred = pp[horizon]
                if isinstance(pred, dict):
                    if pred.get("direction") not in VALID_OUTLOOKS:
                        errors.append(
                            f"parallel_predictions.{horizon}.direction: must be one of {VALID_OUTLOOKS}")
                    pc = pred.get("confidence")
                    if not isinstance(pc, (int, float)) or not (0 <= pc <= 100):
                        errors.append(f"parallel_predictions.{horizon}.confidence: must be 0-100")

    # verdict/weight_applied (bull_case_assessment, bear_case_assessment) covered here.
    _sweep_declared_enums(output, errors)
    return len(errors) == 0, errors


def validate_cio_stage_b(
    output: dict,
    stage_a_expected_return_tier: str | None = None,
    tax_efficiency_source: str | None = None,
) -> tuple[bool, list[str]]:
    """Validate CIO Stage B (account-specific overlay + narrative) output.

    Args:
        output: The CIO Stage B output dict.
        stage_a_expected_return_tier: Stage A's own tier for this stock/timeline, for the
            "Stage B can only move to a worse tier, or stay the same, never better" rule
            (Stage B prompt rule 3) -- never checked before this rewrite since the old code
            treated CIO as one call and had no Stage A value to compare against.
        tax_efficiency_source: the Tax Strategist's tax_efficiency_for_account value, for the
            cross-check that tax_summary.tax_efficiency_consumed_as must match it exactly.
    """
    errors: list[str] = []

    for field in (
        "position_sizing_recommendation", "expected_return_tier", "synthesis_narrative",
        "tax_summary", "risk_profile_summary",
    ):
        if field not in output or output[field] is None:
            errors.append(f"Missing required CIO Stage B field: {field}")

    psr = str(output.get("position_sizing_recommendation") or "")
    if not re.match(r"^\s*\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?\s*%\s*$", psr):
        errors.append(
            f"position_sizing_recommendation: must be a percentage band \"X-Y%\" "
            f"(the Risk Advisor emits e.g. '3-5%'), got {psr!r}")

    tier = output.get("expected_return_tier")
    if tier not in VALID_RETURN_TIERS:
        errors.append(f"expected_return_tier: must be one of {VALID_RETURN_TIERS}")
    elif stage_a_expected_return_tier in _RETURN_TIER_ORDER:
        # Rule 3: Stage B may only move the tier to a WORSE bucket than Stage A's, or leave it
        # unchanged -- tax drag is a cost, never a reason to improve on the general read.
        if _RETURN_TIER_ORDER.index(tier) > _RETURN_TIER_ORDER.index(stage_a_expected_return_tier):
            errors.append(
                f"expected_return_tier: Stage B moved from Stage A's "
                f"{stage_a_expected_return_tier!r} to a BETTER tier {tier!r} -- tax drag can "
                f"only hold the tier or make it worse, never improve it"
            )

    # 250-400 words per the Stage B prompt's own hard constraint == 1500-2400 chars at the
    # ~6 chars/word heuristic used elsewhere in this codebase -- same bound the old single-call
    # validator had, just relocated to the stage that actually produces this field.
    narrative = output.get("synthesis_narrative", "")
    nc = char_count(narrative)
    if nc < 1500:
        errors.append(f"synthesis_narrative: too short ({nc} chars, min 1500)")
    if nc > 2400:
        errors.append(f"synthesis_narrative: too long ({nc} chars, max 2400)")

    tax_summary = output.get("tax_summary")
    if not isinstance(tax_summary, dict):
        if "tax_summary" in output:
            errors.append(f"tax_summary: must be an object, got {type(tax_summary).__name__}")
    else:
        valid_efficiency = {"favorable", "neutral", "unfavorable"}
        eff = tax_summary.get("tax_efficiency_consumed_as")
        if eff is not None and eff not in valid_efficiency:
            errors.append(f"tax_summary.tax_efficiency_consumed_as: must be one of {valid_efficiency}, got {eff!r}")
        if tax_efficiency_source is not None and eff is not None and eff != tax_efficiency_source:
            errors.append(
                f"tax_summary.tax_efficiency_consumed_as: must match the Tax Strategist's "
                f"tax_efficiency_for_account {tax_efficiency_source!r}, got {eff!r}"
            )
        valid_impact = {"strengthens", "neutral", "weakens"}
        impact = tax_summary.get("tax_impact_on_recommendation")
        if impact is not None and impact not in valid_impact:
            errors.append(f"tax_summary.tax_impact_on_recommendation: must be one of {valid_impact}, got {impact!r}")

    risk_profile_summary = output.get("risk_profile_summary")
    if not isinstance(risk_profile_summary, dict):
        if "risk_profile_summary" in output:
            errors.append(
                f"risk_profile_summary: must be an object, got {type(risk_profile_summary).__name__}"
            )
    else:
        # Stage B's own risk_profile_summary shares a name with a Stage A concept but is
        # unrelated content -- position sizing provenance, not the tail-risk cross-check
        # (that lives at the top level in Stage A, see validate_cio_stage_a).
        valid_source = {"risk_advisor_adopted", "risk_advisor_adjusted", "default_conservative"}
        source = risk_profile_summary.get("position_size_source")
        if source is not None and source not in valid_source:
            errors.append(f"risk_profile_summary.position_size_source: must be one of {valid_source}, got {source!r}")

    return len(errors) == 0, errors


def validate_cio_handles_missing_fundamental(output: dict) -> tuple[bool, list[str]]:
    """Validate Scenario 7: CIO does not treat missing Fundamental data as if it existed.

    When FUND pass2_view is empty (stub/insufficient), the CIO should acknowledge the
    missing fundamental data in synthesis_narrative and not make valuation claims.

    `synthesis_narrative` is produced in Stage B, not Stage A -- this check operates on
    whichever output dict the caller passes (typically a merged view of both stages).
    """
    errors: list[str] = []

    def _text(v) -> str:
        """tax_summary is an OBJECT in the prompt schema (a string only in the runner's
        paraphrase). Flatten it rather than assuming either shape -- this function
        scans for acknowledgment phrases, so nested values matter as much as prose."""
        if isinstance(v, dict):
            return " ".join(str(x) for x in v.values())
        return str(v or "")

    narrative = _text(output.get("synthesis_narrative")).lower()
    tax_summary = _text(output.get("tax_summary")).lower()
    kdf_text = " ".join(
        kf.get("factor", "") for kf in output.get("key_decision_factors", []) if isinstance(kf, dict)
    ).lower()

    combined = narrative + " " + tax_summary + " " + kdf_text

    # The CIO should acknowledge missing fundamental data
    acknowledgment_phrases = [
        "fundamental", "financial data", "no financial", "unavailable", "insufficient",
        "pre-revenue", "missing", "without fundamental", "no valuation", "clinical",
        "pipeline", "binary", "trial",
    ]
    has_acknowledgment = any(phrase in combined for phrase in acknowledgment_phrases)
    if not has_acknowledgment:
        errors.append(
            "CIO (Scenario 7): synthesis_narrative should acknowledge missing Fundamental Analyst data "
            "for pre-revenue biotech — do not assume financial data existed"
        )

    # The CIO should NOT make confident P/E or revenue-based valuation claims
    inappropriate_claims = ["p/e", "pe ratio", "price to earnings", "revenue growth", "margin expansion"]
    for claim in inappropriate_claims:
        if claim in narrative:
            errors.append(
                f"CIO (Scenario 7): narrative contains '{claim}' but FUND data was insufficient — "
                "CIO must not fabricate fundamental data"
            )

    return len(errors) == 0, errors
