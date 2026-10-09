"""The Technical prompt after the 2026-10-04 deep dive: every limit stated, no fields code now decides, no re-stated payload."""

from pathlib import Path

PROMPT = (Path(__file__).resolve().parents[2] / "prompts" / "technical_analyst" / "v1.txt").read_text(encoding="utf-8")


def test_every_limit_the_validator_enforces_is_in_the_prompt():
    """25 of 51 real runs retried; most wrote more list items than the cap because only the validator knew it."""
    assert "risks 1-3" in PROMPT and "at most 80 words" in PROMPT
    assert "600-1,800 characters" in PROMPT


def test_key_factors_are_not_asked_for_and_the_narrative_carries_the_evidence():
    """56 first-attempt errors came from key_factors[].sentiment; the list is gone (nothing read it), the narrative cites the values."""
    assert "key_factors" not in PROMPT
    assert "The narrative carries the evidence" in PROMPT


def test_the_model_is_not_asked_for_fields_code_decides():
    for field in ("momentum_zone", "momentum_divergence", "momentum_direction", "suggested_invalidation_level",
                  "nearest_level_bias"):
        assert field not in PROMPT


def test_the_precomputed_values_are_not_restated_in_the_system_prompt():
    """The system prompt used to carry weekly_trend, rs_leadership, rsi_zone_adjusted, nearest levels: all in the payload."""
    for placeholder in ("{weekly_trend}", "{rs_leadership}", "{rsi_zone_adjusted}", "{nearest_support}",
                        "{nearest_resistance}", "{volatility_regime_derived}", "{weekly_rsi_zone}", "{timeline_instruction}"):
        assert placeholder not in PROMPT


def test_the_mandatory_caveat_rules_survive():
    assert "{earnings_proximity_days}" in PROMPT and "earnings-proximity caveat" in PROMPT
    assert "thin-volume caveat" in PROMPT


def test_it_does_not_invite_caveats_that_say_a_condition_is_absent():
    """13 of 49 replay caveats were negations ("so no thin-volume caveat"); caveats reach Pass 2 verbatim. Two other
    wordings were tried and measured: quoting the payload warning line made the model write that line as a caveat for a
    name 17 days from earnings, and pointing at "the payload warning line" brought back routine earnings mentions."""
    assert "never write a caveat that says a condition is absent" in PROMPT
    assert "EARNINGS IN" not in PROMPT


def test_it_stays_lean():
    """5.8k characters before the rewrite, including a 900-character narrative field line."""
    assert len(PROMPT) < 4900
