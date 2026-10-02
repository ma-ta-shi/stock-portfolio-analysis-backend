"""The Risk Advisor prompts must state what their validators enforce.

Measured 2026-10-01 (Risk Advisor Wave 2) on 81 real stage A attempts: only 20% passed first try, and
the causes were all prompt-versus-validator: the narrative was asked for in words ("250-400") while the
validator counts characters (the model writes 6.8 a word, so 400 words would be rejected as too long
and its median of 218 words sat on the floor), `key_factors` sentiment was `negative|neutral` with no
reason given so mitigants came out "positive", and the rule that no two scenario impacts may be within
2 points was stated only in a rule far from the field. Each test here pins one such pair.
"""
import re

import pytest

from agents.prompts import load_template
from agents.validators.pass2 import validate_risk_advisor_stage_a

STAGE_A = load_template("risk_advisor", stage="a")


def _narrative_errors(n_chars: int) -> list[str]:
    out = {"narrative": "x" * n_chars}
    return [e for e in validate_risk_advisor_stage_a(out)[1] if e.startswith("narrative: too")]


@pytest.mark.parametrize(
    "stated",
    [
        "1,000-2,400 characters",  # the hard-constraints line
        "key_factors: 2-4 items",
        "// 2-4 scenarios",
        "under 1,000 or over 2,400 is rejected",
    ],
)
def test_stage_a_bounds_are_stated(stated):
    assert stated in STAGE_A


def test_the_narrative_the_prompt_asks_for_sits_inside_the_validators_bounds():
    """Prompt aim 1,800-2,200; validator rejects under 1,000 and over 2,400."""
    assert "1,800-2,200 characters" in STAGE_A
    assert _narrative_errors(1_000) == [] and _narrative_errors(999) != []
    assert _narrative_errors(2_400) == [] and _narrative_errors(2_401) != []


def test_the_prompt_no_longer_counts_the_narrative_in_words():
    assert "250-400 words" not in STAGE_A
    assert "stop at 400 words" not in STAGE_A


def test_the_sentiment_rule_explains_why_and_says_what_to_do_with_a_mitigant():
    assert re.search(r'"sentiment": "negative\|neutral"', STAGE_A)
    assert "a mitigating factor is" in STAGE_A and 'never "positive"' in STAGE_A


def test_the_distinct_impact_rule_is_on_the_impact_field():
    assert re.search(r"estimated_impact_pct\": <number[^>]*at least 2 points", STAGE_A)


@pytest.mark.parametrize("enum", ["very_high|high|moderate|low", "high|medium|low", "favorable|neutral|unfavorable"])
def test_every_declared_enum_is_in_the_stage_a_schema(enum):
    assert enum in STAGE_A


def test_the_timeline_enum_matches_the_validators():
    for value in ("within_4_weeks", "1_to_3_months", "3_to_12_months", "1_to_3_years", "3_plus_years"):
        assert value in STAGE_A


def test_the_volatility_scale_is_stated_on_the_field():
    assert re.search(r"volatility_assessment.*low under 18%.*moderate 18-30%.*high 30-45%.*very_high over 45%", STAGE_A)


def test_every_token_the_validator_accepts_from_the_metrics_block_is_offered():
    """LIQ is in the block and accepted by the validator; the prompt used to offer BETA VOL DD only."""
    assert "BETA VOL DD LIQ" in STAGE_A


def test_the_prompt_does_not_ask_for_comparisons_no_source_supplies():
    assert "sector median" not in STAGE_A and "{sector} norms" not in STAGE_A


def test_the_thesis_summary_token_requirement_is_in_its_schema_line():
    assert re.search(r'"thesis_summary": "[^"]*name at least one of BETA, VOL, DD in capitals', STAGE_A)


def test_the_prompt_does_not_mention_a_stop_loss():
    """Removed from the pipeline 2026-10-01: nothing read it and the value was only the technical
    analyst's nearest support echoed back."""
    assert not re.search(r"stop.?loss", STAGE_A, re.I)


def test_the_prompt_explains_what_beta_is_and_what_a_low_r_squared_means():
    assert "WHAT BETA IS" in STAGE_A
    assert "NOT how volatile it is" in STAGE_A
    assert "below 0.10" in STAGE_A and "data_sanity_flags" in STAGE_A


def test_the_beta_example_does_not_name_a_specific_index():
    """The old example read "vs S&P/TSX" and the model copied it onto US stocks (about 15% of outputs
    that named a benchmark named the wrong one)."""
    assert "S&P/TSX" not in STAGE_A
    assert "the benchmark named on the BETA line" in STAGE_A


def test_the_drawdown_repeat_rule_is_in_the_rule_and_in_the_schema_line():
    """Prose alone ("the worst scenario must be at least as severe as the drawdown") moved 0 of 3 volatile
    stocks in replays; making one scenario the explicit repeat, stated in the schema line, moved 4 of 4. The
    rule is scoped to a 1-year drawdown of 15% or more: on small drawdowns it collided with the model's own
    -10 scenario and broke first-try validity."""
    assert "ONE SCENARIO IS THE DRAWDOWN REPEAT" in STAGE_A
    assert "is 15% or more" in STAGE_A
    schema_line = next(ln for ln in STAGE_A.splitlines() if "estimated_impact_pct" in ln and "<number" in ln)
    assert "15% or more" in schema_line and "1-year DD" in schema_line


def test_the_prompt_stops_the_drawdown_repeat_from_inventing_a_period():
    """Found reading real outputs against their inputs: after the repeat rule, 3 of 21 outputs said the drawdown
    'mirrors the 2023-24 decline' (0 of 25 before). The inputs give no dates."""
    assert "cause not identified in the inputs" in STAGE_A
    assert "name no calendar period" in STAGE_A


def test_the_prompt_ties_dividend_scenarios_to_fund_and_beta_wording_to_the_r_squared():
    """KO and MSFT got a -15 to -20% dividend-cut scenario with a trigger reading 'dividend_sustainability
    adequate'; all three KO outputs called a beta with R-squared 0.02 an 'inverse' relationship."""
    assert "When FUND rates the dividend adequate or strong, write no dividend scenario" in STAGE_A
    assert "the sign of such a beta means nothing" in STAGE_A


def test_scenario_spacing_is_stated_for_every_scenario():
    assert "at least 3 points away from every other's" in STAGE_A
