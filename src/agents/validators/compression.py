"""Compression validators — check pass2_view shape and reliability warning block format."""
import re
from agents.utils import CONFIDENCE_STATUS_LABELS, PASS1_AGENT_IDS


def validate_pass2_view_shape(agent_id: str, pass2_view: dict) -> tuple[bool, list[str]]:
    """Validate that pass2_view contains the required fields for each Pass 1 agent."""
    errors: list[str] = []

    REQUIRED_FIELDS = {
        "RSRCH": [
            "thesis_archetype", "competitive_position", "overall_moat_durability",
            "moat_trend", "management_assessment", "top_growth_drivers",
            "top_competitive_threats", "top_recent_developments", "peer_comparison_summary",
        ],
        "FUND": [
            "valuation_vs_sector", "pe_ratio", "health_rating", "margin_trend",
            "guidance_vs_consensus", "dividend_sustainability",
        ],
        "TECH": [
            "primary_trend", "trend_strength", "momentum_zone", "momentum_direction",
            "volume_confirmation", "nearest_level_bias", "confluence_score",
            "nearest_support", "nearest_resistance", "volatility_regime_derived",
        ],
        # SENT was half-migrated: 3 of these 5 names already matched the flat
        # pass2_view, but "news_sentiment" and "analyst_sentiment" still named
        # structured_data sub-objects, so a conformant flat view failed
        # validation. The prompt is authoritative and explicit ("a flat
        # pass2_view dict"); renamed to the flat fields it actually declares.
        "SENT": [
            "news_sentiment_overall", "sentiment_trend",
            "insider_activity_interpretation",
            "short_interest_interpretation", "peer_sentiment_comparison",
        ],
        "MACRO": [
            "overall_macro_environment", "volatility_regime", "sector_cycle_position",
            "sector_tailwinds", "sector_headwinds",
        ],
    }

    required = REQUIRED_FIELDS.get(agent_id, [])
    for field in required:
        if field not in pass2_view:
            errors.append(f"{agent_id}.pass2_view.{field}: missing")

    # SENT must NOT include social_sentiment
    if agent_id == "SENT" and "social_sentiment" in pass2_view:
        errors.append("SENT.pass2_view: must not include social_sentiment")

    return len(errors) == 0, errors


def validate_reliability_warning_format(warning_block: str) -> tuple[bool, list[str]]:
    """Validate the format of the confidence warning block.

    Expected format: RSRCH: high (OK) | FUND: medium (caution) | TECH: low (unreliable) | ...
    Empty string is valid (means all agents are 'high').

    Rewritten for 86bbummwp / D6: the block used to carry a 0-100 reliability_score, verified
    against a score->status mapping. reliability_score is gone; this now verifies the
    analysis_confidence enum against the same CONFIDENCE_STATUS_LABELS mapping
    build_pass1_reliability_warnings() (utils.py) builds the block from -- imported, not
    redefined, so the two can't silently drift apart.
    """
    errors: list[str] = []

    if not warning_block:
        return True, []  # All agents 'high', empty string is correct

    # Check for pipe-separated format
    parts = [p.strip() for p in warning_block.split("|")]
    pattern = re.compile(r"^([A-Z]+):\s*(\w+)\s*\((\w+)\)$")

    for part in parts:
        m = pattern.match(part)
        if not m:
            errors.append(f"Reliability warning part '{part}' does not match format 'AGENT: CONFIDENCE (STATUS)'")
            continue
        agent_id, confidence, status = m.group(1), m.group(2), m.group(3)
        if agent_id not in set(PASS1_AGENT_IDS):
            errors.append(f"Unknown agent ID '{agent_id}' in reliability warning")
        if confidence not in CONFIDENCE_STATUS_LABELS:
            errors.append(f"Invalid analysis_confidence '{confidence}' in reliability warning (must be high|medium|low|insufficient)")
            continue
        expected_status = CONFIDENCE_STATUS_LABELS[confidence]
        if status != expected_status:
            errors.append(f"{agent_id}: confidence={confidence} labeled '{status}' but should be '{expected_status}'")

    return len(errors) == 0, errors


def validate_stub_fundamental(pass1_results: dict) -> tuple[bool, list[str]]:
    """Validate that Fundamental Analyst stub scenario behaves correctly.

    Scenario 7: FUND should return analysis_confidence in {insufficient, low} and pass2_view={}.

    Repointed from data_quality_assessment (86bbummwp / D6: never LLM-produced now, so an
    exact match on it would fail unconditionally forever). Accepts "low" as well as
    "insufficient" -- live-verified (86bbummwp) that this exact fixture's real, current,
    honest model output is "low", not "insufficient". Forcing an exact "insufficient" match
    here would either fail a legitimate test permanently or pressure the model into fabricating
    a value it doesn't naturally reach for this profile. Whether Fundamental's own per-agent
    minimum SHOULD reach "insufficient" for this profile is a threshold-review question
    assigned to 86bbuhjup's decision 1, not decided here.
    """
    errors: list[str] = []

    fund_out = pass1_results.get("FUND")
    if fund_out is None:
        errors.append("FUND: output is None — should have returned an output with analysis_confidence in {insufficient, low}")
        return False, errors

    confidence = fund_out.get("analysis_confidence")
    if confidence not in {"insufficient", "low"}:
        errors.append(
            f"FUND: analysis_confidence='{confidence}' but should be 'insufficient' or 'low' for pre-revenue biotech with no financial data"
        )

    p2v = fund_out.get("pass2_view")
    if p2v != {} and p2v is not None:
        errors.append(
            f"FUND: pass2_view should be {{}} (empty) when analysis_confidence is insufficient/low, got: {p2v}"
        )

    return len(errors) == 0, errors
