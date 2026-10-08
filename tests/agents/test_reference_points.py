"""The upside and downside numbers Bull and Bear used to invent are written by code (agents/reference_points.py)."""
import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

from agents.pass2_bear_advocate import BearAdvocateRunner
from agents.pass2_bull_advocate import BullAdvocateRunner
from agents.prompts import load_template
from agents.reference_points import asymmetry_reference


def _bundle(**overrides) -> SimpleNamespace:
    defaults = dict(
        stock=SimpleNamespace(ticker="TD.TO", currency="CAD", exchange="TSX"),
        company_info={"name": "Toronto-Dominion Bank", "sector": "Finance"},
        context=SimpleNamespace(account_type="tfsa", timeline="medium_term"),
        data_vintage=datetime(2026, 10, 7, tzinfo=UTC),
        analyst_consensus={"target_mean": 183.4},
        analyst_rating_changes=[],
        price_info={"current_price": 162.68},
        support_resistance={"nearest_support": 167.2973, "pct_to_support": 0.46, "nearest_resistance": 173.7717, "pct_to_resistance": -4.4},
        risk_metrics={"max_drawdown_1yr_pct": 8.5, "annualized_vol_pct": 16.5},
    )
    return SimpleNamespace(**{**defaults, **overrides})


def test_every_figure_is_quoted_from_the_source_that_computed_it():
    text = asymmetry_reference(_bundle())
    assert "consensus target 183.4 (+12.7% against the price, SENT)" in text
    assert "nearest resistance 173.77 (4.4% above, TECH, from the last close)" in text
    assert "nearest support 167.3 (0.46% below, TECH, from the last close)" in text
    assert "1-year maximum drawdown 8.5% (DD)" in text and "12-month move of 16.5% either way (VOL)" in text


def test_technicals_sign_convention_never_reaches_the_text():
    """Technical stores (price - level) / price, so a resistance above the price is negative. The first version printed "-1.35% above"."""
    for resistance in (-1.35, 1.35):
        text = asymmetry_reference(_bundle(support_resistance={"nearest_resistance": 164.62, "pct_to_resistance": resistance,
                                                                "nearest_support": 159.23, "pct_to_support": -1.97}))
        assert "nearest resistance 164.62 (1.35% above, TECH, from the last close)" in text and "nearest support 159.23 (1.97% below, TECH, from the last close)" in text
        assert "-1.35" not in text and "-1.97" not in text


def test_the_support_distance_is_technicals_own_not_recomputed_from_the_quote():
    """The quote (162.68) is below the support (167.30); a percent recomputed from it would put support above the price. The percent
    TECH computed from its own last close is quoted as it is."""
    text = asymmetry_reference(_bundle())
    assert "0.46% below" in text and "2.8%" not in text


def test_missing_inputs_are_left_out_and_nothing_is_invented():
    text = asymmetry_reference(_bundle(analyst_consensus={}, support_resistance={}))
    assert "Upside reference" not in text and "consensus" not in text
    assert text.startswith("Downside reference: 1-year maximum drawdown 8.5% (DD)")


def test_a_missing_input_leaves_the_figure_out_and_never_raises():
    """The line is written after the model has answered; an exception here would throw away a good answer."""
    bundle = _bundle()
    del bundle.data_vintage
    text = asymmetry_reference(bundle)
    assert "consensus target" not in text and "1-year maximum drawdown 8.5% (DD)" in text


def test_no_data_says_so():
    text = asymmetry_reference(_bundle(analyst_consensus={}, support_resistance={}, risk_metrics={}))
    assert text == "Upside and downside reference points are not available in the data."


def _run(runner_cls, model_output):
    runner = runner_cls()

    async def _fake_call(system_prompt, user_msg, validator, **_kw):
        return model_output, []

    compressed = {"RSRCH": {"assessment_summary": "s", "analysis_confidence": "high", "caveats": [],
                            "pass2_view": {"thesis_archetype": "quality_compounder"}, "narrative": "n"}}
    with patch.object(runner, "call_with_validation", side_effect=_fake_call):
        return asyncio.run(runner.run(_bundle(), compressed))[0]


def test_bull_and_bear_asymmetry_is_the_code_written_line_even_if_the_model_wrote_one():
    expected = asymmetry_reference(_bundle())
    for cls in (BullAdvocateRunner, BearAdvocateRunner):
        result = _run(cls, {"structured_data": {"asymmetry_assessment": "a 5-7% decline"}})
        assert result["structured_data"]["asymmetry_assessment"] == expected
        # a missing structured_data is left alone (the validator rejects that output; nothing to fill)
        assert "structured_data" not in _run(cls, {})


def test_neither_advocate_prompt_asks_the_model_for_the_asymmetry_numbers():
    assert "asymmetry_assessment" not in load_template("bull_advocate")
    assert "asymmetry_assessment" not in load_template("bear_advocate")


def test_risk_prompt_no_longer_points_at_a_section_that_does_not_exist():
    prompt = load_template("risk_advisor", stage="a")
    assert "SEVERITY" not in prompt and "scaled to THIS stock using the VOL and DD lines" in prompt
