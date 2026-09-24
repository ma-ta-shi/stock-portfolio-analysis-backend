"""Shadow CIO validator tests. Nothing enforced any shadow rule before 2026-09-01.

The two rules that are OPPOSITE to the primary CIO's get explicit tests, because the
likeliest way to break this validator is to "align" it with the CIO's.
"""
from agents.validators.shadow_cio import (
    check_cold_start_exemption,
    check_numeric_groundedness,
    validate_shadow_cio,
)


def _valid_shadow() -> dict:
    return {
        "stock_outlook": "somewhat_bearish",
        "confidence": 58,
        "expected_return_tier": "underperform",
        "thesis_summary": (
            "Bear-weighted read for a medium-term TFSA holding: the withholding drag and a "
            "45% P/B premium outweigh steady revenue growth, leaving limited relative upside."
        ),
        "bear_weighting_application_note": (
            "The override changed the read: a neutrally-weighted view would have landed neutral; "
            "weighting the valuation premium moved it one notch bearish."
        ),
        "what_would_change_my_mind": "A P/B re-rating toward peers, or two quarters of margin expansion.",
        "parallel_predictions": {
            "1_month": {"direction": "neutral", "confidence": 50},
            "3_month": {"direction": "somewhat_bearish", "confidence": 55},
        },
    }


class TestShadowCIO:
    def test_valid_output_passes(self):
        passed, errors = validate_shadow_cio(_valid_shadow())
        assert passed, errors

    def test_missing_bear_weighting_note_fails(self):
        out = _valid_shadow()
        del out["bear_weighting_application_note"]
        passed, errors = validate_shadow_cio(out)
        assert not passed and any("bear_weighting" in e for e in errors)

    def test_bear_weighting_note_too_short_fails(self):
        """It must state whether the override changed the read, not just that it ran."""
        out = _valid_shadow()
        out["bear_weighting_application_note"] = "Applied."
        passed, errors = validate_shadow_cio(out)
        assert not passed and any("bear_weighting" in e for e in errors)

    def test_recommendation_field_forbidden(self):
        out = _valid_shadow()
        out["recommendation"] = "sell"
        passed, errors = validate_shadow_cio(out)
        assert not passed and any("recommendation" in e for e in errors)

    def test_thesis_summary_length_bounds(self):
        out = _valid_shadow()
        out["thesis_summary"] = "Too short."
        passed, errors = validate_shadow_cio(out)
        assert not passed and any("thesis_summary" in e for e in errors)

    def test_return_tier_uses_the_prompt_vocabulary(self):
        """`high_upside` is the runner's old vocabulary; both prompts use `outperform`."""
        out = _valid_shadow()
        out["expected_return_tier"] = "moderate_downside"
        passed, errors = validate_shadow_cio(out)
        assert not passed and any("expected_return_tier" in e for e in errors)

    def test_split_decision_caps_confidence(self):
        out = _valid_shadow()
        out["confidence"] = 80
        passed, errors = validate_shadow_cio(out, disagreement_category="split_decision")
        assert not passed and any("<=65" in e for e in errors)

    def test_high_conflict_forces_neutral_range(self):
        out = _valid_shadow()
        out["stock_outlook"] = "bearish"
        passed, errors = validate_shadow_cio(out, disagreement_category="high_conflict")
        assert not passed and any("high_conflict" in e for e in errors)

    # --- the two rules that are OPPOSITE to the primary CIO -----------------------
    def test_low_groundedness_caps_confidence_at_45(self):
        """A rule the primary CIO does not have."""
        out = _valid_shadow()
        out["confidence"] = 60
        passed, errors = validate_shadow_cio(out, low_groundedness=True)
        assert not passed and any("<=45" in e for e in errors)

    def test_cold_start_cap_does_NOT_apply_to_shadow(self):
        """The shadow is explicitly EXEMPT. Confidence >65 under an active cold-start
        cap is CORRECT, and must never be an error — the likeliest way to break this
        validator is to copy the primary's rule onto it."""
        out = _valid_shadow()
        out["confidence"] = 78
        passed, errors = validate_shadow_cio(out, disagreement_category="consensus")
        assert passed, f"shadow must not be cold-start capped: {errors}"
        assert check_cold_start_exemption(out, cold_start_cap_active=True), \
            "exemption should be reported informationally"

    def test_parallel_predictions_shape(self):
        out = _valid_shadow()
        out["parallel_predictions"]["1_month"]["direction"] = "up"
        passed, errors = validate_shadow_cio(out)
        assert not passed and any("1_month.direction" in e for e in errors)


# --- numeric groundedness (86bbt1kct) -- informational, never a hard failure ----

def _payload() -> tuple[dict, dict]:
    compressed_pass1 = {"FUND": {"pass2_view": {"pe_ratio": 22.4}}}
    pass2_outputs = {
        "bull": {"confidence": 70},
        "bear": {"confidence": 55, "structured_data": {"tail_risk_assessment": {"evidence": "293.0752"}}},
        "risk": {"risk_profile": {"beta": 1.35, "max_drawdown_1yr": -28.6}},
    }
    return compressed_pass1, pass2_outputs


class TestNumericGroundedness:
    def test_grounded_number_produces_no_warning(self):
        compressed_pass1, pass2_outputs = _payload()
        out = _valid_shadow()
        out["thesis_summary"] += " Beta of 1.35 supports this."
        warnings = check_numeric_groundedness(out, compressed_pass1, pass2_outputs)
        assert warnings == []

    def test_fabricated_number_is_flagged(self):
        compressed_pass1, pass2_outputs = _payload()
        out = _valid_shadow()
        out["thesis_summary"] += " Revenue grew 847.3% last quarter."
        warnings = check_numeric_groundedness(out, compressed_pass1, pass2_outputs)
        assert warnings and "847.3" in warnings[0]

    def test_rounded_number_within_tolerance_is_grounded(self):
        """The E47 lesson: a correctly-rounded 293.08 against a payload's
        293.0752 must NOT be flagged as fabricated."""
        compressed_pass1, pass2_outputs = _payload()
        out = _valid_shadow()
        out["thesis_summary"] += " Cited at 293.08 in the tail-risk evidence."
        warnings = check_numeric_groundedness(out, compressed_pass1, pass2_outputs)
        assert warnings == []

    def test_confidence_scale_ints_are_not_flagged(self):
        """Small ints (e.g. confidence=58, already required by the schema)
        are ubiquitous and prove nothing about groundedness -- must not be
        treated as fabricated just because 58 doesn't literally appear in
        the input payload."""
        compressed_pass1, pass2_outputs = _payload()
        out = _valid_shadow()
        assert out["confidence"] == 58
        warnings = check_numeric_groundedness(out, compressed_pass1, pass2_outputs)
        assert warnings == []

    def test_never_fails_validation_even_when_ungrounded(self):
        """Informational only -- validate_shadow_cio's own pass/fail must be
        completely unaffected by numeric groundedness, mirroring
        check_cold_start_exemption's existing pattern."""
        compressed_pass1, pass2_outputs = _payload()
        out = _valid_shadow()
        out["thesis_summary"] += " A fabricated 999.9% figure."
        passed, errors = validate_shadow_cio(out)
        warnings = check_numeric_groundedness(out, compressed_pass1, pass2_outputs)
        assert passed, errors
        assert warnings  # the fabrication IS detected, just not as a validation failure
