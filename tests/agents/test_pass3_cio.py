"""Tests for agents/pass3_cio.py (86bbuhjup). Close to a clean port -- see
module docstring. Focus: the real seam (bundle.context/stock/company_info
reads) and a regression check that the byte-for-byte-ported summary
functions still work against real-shaped Pass 1/2 output dicts."""
from datetime import UTC, datetime
from types import SimpleNamespace

from agents.pass3_cio import (
    _build_tax_strategist_summary,
    build_advocate_summary,
    build_advocates_block,
    build_pass1_summaries,
    build_risk_advisor_stage_a_summary,
    build_user_message,
)


def _bundle(**overrides) -> SimpleNamespace:
    defaults = dict(
        stock=SimpleNamespace(ticker="SHOP.TO", currency="CAD", exchange="TSX"),
        company_info={"name": "Shopify Inc", "sector": "Technology"},
        context=SimpleNamespace(account_type="tfsa", timeline="medium_term"),
        data_vintage=datetime(2026, 9, 23, tzinfo=UTC),
    )
    return SimpleNamespace(**{**defaults, **overrides})


def test_build_user_message_renders_real_context_fields():
    msg = build_user_message(_bundle(), {}, {}, 20, "consensus")
    assert "SYNTHESIS REQUEST: SHOP.TO (Shopify Inc) | Technology" in msg
    assert "Timeline: medium_term | Account: tfsa" in msg
    assert "Disagreement Score: 20/100 — Category: consensus" in msg


def test_disagreement_cap_warning_for_split_decision():
    msg = build_user_message(_bundle(), {}, {}, 45, "split_decision")
    assert "DISAGREEMENT CAP: category=split_decision" in msg
    assert "confidence must be ≤65" in msg


def test_disagreement_cap_warning_for_high_conflict():
    msg = build_user_message(_bundle(), {}, {}, 70, "high_conflict")
    assert "stock_outlook must be neutral/somewhat range" in msg


def test_pass1_summaries_includes_confidence_finding_and_quality():
    """86bbummwp Tier 3 extended this line with a mechanical quality={enum}
    alongside the model's own confidence={enum} -- deliberately not
    collapsed into one value, see build_pass1_reliability_warnings()'s own
    docstring for why. This replaces the old byte-for-byte-ported assertion,
    which is no longer accurate now that the line's shape has deliberately
    changed."""
    compressed = {
        "RSRCH": {"analysis_confidence": "high", "data_quality_assessment": "medium",
                   "assessment_summary": "Durable moat.",
                   "pass2_view": {"thesis_archetype": "secular_grower"}},
        "FUND": None,
    }
    summary = build_pass1_summaries(compressed)
    assert "RSRCH: confidence=high, quality=medium | finding: Durable moat." in summary
    assert "FUND: NOT AVAILABLE" in summary


def test_pass1_summaries_quality_defaults_to_low_when_missing():
    compressed = {
        "RSRCH": {"analysis_confidence": "high", "assessment_summary": "Durable moat.",
                   "pass2_view": {"thesis_archetype": "secular_grower"}},
    }
    summary = build_pass1_summaries(compressed)
    assert "quality=low" in summary


def test_pass1_summaries_insufficient_data_branch_also_shows_quality():
    """quality is a mechanical, pre-LLM-call fact -- shown even when the
    agent's own output was too thin to build a pass2_view, matching
    extract_data_quality_levels()'s own deliberate non-gating on pass2_view."""
    compressed = {
        "RSRCH": {"analysis_confidence": "insufficient", "data_quality_assessment": "high",
                   "pass2_view": {}},
    }
    summary = build_pass1_summaries(compressed)
    assert "INSUFFICIENT DATA (confidence=insufficient, quality=high)" in summary


def test_advocate_summary_still_works_byte_for_byte_ported():
    bull_output = {
        "recommendation": "bullish", "confidence": 72,
        "thesis_summary": "Strong growth thesis.",
        "strongest_argument": "Margin expansion.", "weakest_point": "Valuation.",
    }
    summary = build_advocate_summary(bull_output, "BULL")
    assert "BULL CASE ADVOCATE:" in summary
    assert "confidence: 72/100" in summary


def test_risk_advisor_stage_a_summary_still_works_byte_for_byte_ported():
    risk_output = {
        "groundedness_score": 80,
        "risk_profile": {"risk_reward_ratio": 2.1, "beta": 1.8, "max_drawdown_1yr": -28.4},
        "thesis_summary": "Moderate risk.", "strongest_signal": "BETA: 1.8",
    }
    summary = build_risk_advisor_stage_a_summary(risk_output)
    assert "risk_reward_ratio: 2.1" in summary
    assert "beta: 1.8 | max_drawdown_1yr: -28.4%" in summary


# ---------- _build_tax_strategist_summary (86bc8eg3j) ----------
# Field shapes below match what 86bc8efvb's real live verification (real RY.TO
# data, real Ollama gpt-oss:20b call) actually observed for a Trading-account
# run -- not invented from the schema alone.


def test_tax_strategist_summary_includes_capital_gains_and_optimization_actions():
    """The real gap this ticket closes: both fields were computed by every real
    Tax Strategist run but never reached the CIO's own Stage B input."""
    tax_result = {
        "groundedness_score": 60,
        "thesis_summary": "Trading account holding.",
        "strongest_signal": "MARG: 31.48%",
        "tax_profile": {
            "tax_efficiency_for_account": "favorable",
            "account_fit_score": "excellent",
            "dividend_classification": "canadian_eligible",
            "wht_interpretation": "No withholding tax applies in a trading account.",
            "capital_gains_treatment_summary": (
                "Capital gains are taxed at a 50% inclusion rate in a trading account; "
                "Canadian dividends retain the dividend tax credit."
            ),
            "tax_optimization_actions": [
                {
                    "action": "Avoid repurchasing within 30 days of a loss to preserve it.",
                    "applies_to": "trading",
                    "estimated_benefit_pct": 2.3,
                    "justification": (
                        "Superficial-loss rule denial would eliminate the tax benefit "
                        "(REF/Taxable)."
                    ),
                },
            ],
        },
    }
    summary = _build_tax_strategist_summary(tax_result)
    assert (
        "capital_gains_treatment_summary: Capital gains are taxed at a 50% "
        "inclusion rate in a trading account" in summary
    )
    assert (
        "tax_optimization_actions: Avoid repurchasing within 30 days of a loss to "
        "preserve it. (benefit=2.3%/yr, justification=Superficial-loss rule denial "
        "would eliminate the tax benefit (REF/Taxable).)" in summary
    )


def test_tax_strategist_summary_tax_optimization_actions_empty_when_absent():
    """Matches key_tax_risks's own existing `or []` degrade pattern -- no crash,
    just an empty rendered section."""
    tax_result = {"tax_profile": {"capital_gains_treatment_summary": "Tax-free in TFSA."}}
    summary = _build_tax_strategist_summary(tax_result)
    assert summary.endswith("tax_optimization_actions: ")


def test_tax_strategist_summary_capital_gains_summary_defaults_to_na_when_missing():
    """Rendering fallback, distinct from the new validator check on the same
    field (agents/validators/pass2.py) -- a caller building a summary from an
    already-invalid output still gets a safe 'N/A', not a crash."""
    tax_result = {"tax_profile": {}}
    summary = _build_tax_strategist_summary(tax_result)
    assert "capital_gains_treatment_summary: N/A" in summary


def test_build_user_message_has_zero_tax_strategist_input():
    """Stage A must be account-neutral by design -- confirmed directly in the
    Stage A runtime prompt ("the SAME read... regardless of which account
    eventually holds the position") and the Stage A validator's own comment
    ("tax" removed from key_decision_factors[].source for exactly this reason).
    An earlier version of build_user_message() leaked Tax Strategist's own
    account-specific facts here anyway; this asserts none of them can appear
    even when tax data IS available (not just when it's absent -- the
    'renders_real_context_fields' test above never passes a tax key at all, so
    it wouldn't catch this regression). Checks every field that was actually in
    the leaked block, not just a couple of them -- a partial reintroduction
    (e.g. someone adds back just account_fit_score under a new header) should
    fail this test exactly as loudly as reintroducing the whole block would."""
    pass2_outputs = {
        "tax": {
            "groundedness_score": 75,
            "thesis_summary": "Trading account holding.",
            "tax_profile": {
                "account_fit_score": "good",
                "tax_efficiency_for_account": "neutral",
                "effective_after_tax_yield_pct": 2.3,
                "capital_gains_treatment_summary": "Taxed at a 50% inclusion rate.",
                "cross_account_recommendation": {
                    "better_account": "tfsa",
                    "reasoning": "Shelters future gains.",
                    "drag_delta_pct": 0.0,
                },
                "tax_optimization_actions": [
                    {
                        "action": "Move to a TFSA to shelter future gains.",
                        "estimated_benefit_pct": 0.0,
                        "justification": "Tax-free growth in a TFSA (REF/TFSA).",
                    },
                ],
            },
        },
    }
    msg = build_user_message(_bundle(), {}, pass2_outputs, 20, "consensus")
    assert "TAX STRATEGIST" not in msg
    assert "account_fit_score" not in msg
    assert "tax_efficiency_for_account" not in msg
    assert "effective_after_tax_yield_pct" not in msg
    assert "capital_gains_treatment_summary" not in msg
    assert "cross_account_recommendation" not in msg
    assert "tax_optimization_actions" not in msg

# ---------- Tax Strategist fields that used to stop short of the CIO (Tax Strategist Wave 2) ----------


def _tax_result_with_detail() -> dict:
    return {
        "groundedness_score": 80,
        "thesis_summary": "Trading holding.",
        "strongest_signal": "WHT: 15% withheld, recoverable as a foreign tax credit",
        "caveats": ["ROOM not on file for RRSP.", "WHT not modelled for the REIT sleeve."],
        "narrative": "WHT: 15% is withheld on the US dividend in Trading and is recoverable as a credit.",
        "tax_profile": {
            "tax_efficiency_for_account": "neutral",
            "account_fit_score": "fair",
            "tfsa_contribution_room_impact": "A sell here frees $4,200 of TFSA room on Jan 1.",
            "rrsp_contribution_room_impact": None,
            "loss_harvesting_opportunity": {
                "available": True,
                "estimated_tax_savings_pct": 1.2,
                "superficial_loss_window_safe": True,
                "dual_listed_check_applied": "not_applicable",
                "remediation_suggestion": "wait 31 days",
                "reasoning": "LOSS: window open, YTD gains to offset.",
            },
            "tax_optimization_actions": [],
        },
    }


def test_tax_strategist_summary_passes_narrative_caveats_room_impact_and_loss_harvesting():
    summary = _build_tax_strategist_summary(_tax_result_with_detail())
    assert "caveats: ROOM not on file for RRSP.; WHT not modelled for the REIT sleeve." in summary
    assert "tfsa_contribution_room_impact: A sell here frees $4,200 of TFSA room on Jan 1." in summary
    assert "loss_harvesting_opportunity: available=True, estimated_tax_savings=1.2%" in summary
    assert "remediation=wait 31 days; LOSS: window open, YTD gains to offset." in summary
    assert (
        "tax_narrative (supporting context; the fields above are authoritative): "
        "WHT: 15% is withheld on the US dividend in Trading and is recoverable as a credit."
    ) in summary


def test_tax_strategist_summary_omits_a_null_room_impact_and_an_inapplicable_loss_harvest():
    result = _tax_result_with_detail()
    result["tax_profile"]["loss_harvesting_opportunity"] = None
    result["tax_profile"]["tfsa_contribution_room_impact"] = "null"
    result["caveats"] = []
    summary = _build_tax_strategist_summary(result)
    assert "rrsp_contribution_room_impact" not in summary
    assert "tfsa_contribution_room_impact" not in summary
    assert "loss_harvesting_opportunity" not in summary
    assert "caveats:" not in summary


def test_tax_strategist_summary_keeps_the_structured_fields_before_the_narrative():
    summary = _build_tax_strategist_summary(_tax_result_with_detail())
    assert summary.index("account_fit_score: fair") < summary.index("tax_optimization_actions:")
    assert summary.index("tax_optimization_actions:") < summary.index("tax_narrative")

# ---------- Risk Advisor stage A summary: what the CIO now receives (Risk Advisor Wave 2) ----------


def _risk_output() -> dict:
    return {
        "groundedness_score": 92,
        "thesis_summary": "Moderate risk.",
        "strongest_signal": "FUND: payout ratio 126%",
        "caveats": ["FUND data is stale."],
        "risk_profile": {
            "risk_reward_ratio": "unfavorable",
            "volatility_assessment": "low",
            "beta": 0.1,
            "max_drawdown_1yr": 17.9,
            "data_sanity_flags": ["BETA 0.1 with VOL 17% is a weak benchmark link"],
            "downside_scenarios": [
                {"scenario": "Dividend cut", "estimated_impact_pct": -25, "probability": "medium",
                 "timeline": "3_to_12_months", "trigger": "FUND payout ratio"},
                {"scenario": "Margin squeeze", "estimated_impact_pct": -10, "probability": "low",
                 "timeline": "1_to_3_years", "trigger": "MACRO rates"},
            ],
        },
    }


def test_risk_advisor_summary_carries_each_scenarios_probability_and_timeline():
    summary = build_risk_advisor_stage_a_summary(_risk_output())
    assert "Dividend cut (-25%, medium probability, 3_to_12_months)" in summary
    assert "Margin squeeze (-10%, low probability, 1_to_3_years)" in summary


def test_risk_advisor_summary_passes_the_volatility_assessment_flags_and_caveats():
    summary = build_risk_advisor_stage_a_summary(_risk_output())
    assert "volatility_assessment: low" in summary
    assert "data_sanity_flags: BETA 0.1 with VOL 17% is a weak benchmark link" in summary
    assert "caveats: FUND data is stale." in summary


def test_risk_advisor_summary_omits_flags_and_caveats_on_a_clean_run():
    out = _risk_output()
    out["caveats"] = []
    out["risk_profile"]["data_sanity_flags"] = []
    summary = build_risk_advisor_stage_a_summary(out)
    assert "data_sanity_flags" not in summary and "caveats" not in summary


def test_risk_advisor_summary_ends_with_the_narrative_as_supporting_detail():
    out = _risk_output()
    out["narrative"] = "Liquidity is strong at 10.4 B USD a day (LIQ). The key signal is the drawdown (DD)."
    summary = build_risk_advisor_stage_a_summary(out)
    last = summary.splitlines()[-1]
    assert last.startswith("  narrative (supporting detail; the structured fields above are authoritative): ")
    assert "Liquidity is strong at 10.4 B USD a day (LIQ)" in last


def test_risk_advisor_summary_omits_the_narrative_line_when_there_is_none():
    assert "narrative" not in build_risk_advisor_stage_a_summary(_risk_output())
    for blank in ("", "   "):
        assert "narrative" not in build_risk_advisor_stage_a_summary({**_risk_output(), "narrative": blank})


def test_the_stage_a_user_message_carries_the_risk_narrative_through_the_real_builder():
    """The path the CIO actually takes: build_user_message with the orchestrator's pass2_outputs keys."""
    risk = {**_risk_output(), "narrative": "Liquidity is strong at 10.4 B USD a day (LIQ)."}
    msg = build_user_message(_bundle(), {}, {"risk": risk}, 20, "consensus")
    assert "RISK ADVISOR:" in msg and "Liquidity is strong at 10.4 B USD a day (LIQ)." in msg
    assert "TAX STRATEGIST" not in msg


def test_the_shadow_cio_gets_the_risk_narrative_too():
    """Shadow CIO and CIO stage A share build_risk_advisor_stage_a_summary, so one change reaches both."""
    from agents import pass3_shadow_cio

    out = {**_risk_output(), "narrative": "Liquidity is strong (LIQ)."}
    assert "narrative (supporting detail" in pass3_shadow_cio.build_risk_advisor_stage_a_summary(out)


# ---------- CIO stage B: no Risk Advisor input, no sizing (removed 2026-10-01) ----------


async def test_stage_b_prompt_is_filled_without_a_risk_advisor_input_or_sizing():
    import re

    from agents.pass3_cio import CIORunner

    sent = {}

    async def fake_call(self, system_prompt, user_msg, validator, **kw):
        sent["system"] = system_prompt
        return {"expected_return_tier": "market_perform"}, []

    runner = CIORunner()
    runner.call_with_validation = fake_call.__get__(runner)
    stage_a = {"stock_outlook": "neutral", "confidence": 60, "expected_return_tier": "market_perform"}
    await runner.run_stage_b(_bundle(), stage_a, {"tax_profile": {"tax_efficiency_for_account": "neutral"}}, "tfsa")

    system = sent["system"]
    assert not re.findall(r"\{[a-z_]+\}", system), "unfilled template placeholder"
    assert "RISK ADVISOR" not in system
    assert "position_sizing_recommendation" not in system.split("Produce valid JSON")[1]
    assert "risk_profile_summary" not in system


# ---------- advocate summary: the design's fields the first port dropped ----------


def _bull_with_detail() -> dict:
    return {
        "recommendation": "bullish", "confidence": 68, "thesis_summary": "Steady compounder.",
        "strongest_argument": "FCF conversion near 1.0 (FUND).", "weakest_point": "Distribution spend (RSRCH).",
        "caveats": ["RSRCH confidence is medium."],
        "narrative": "Coca-Cola converts almost all net income into free cash flow (FUND).",
        "structured_data": {
            "core_arguments": [
                {"argument": "Cash conversion", "strength": "primary", "evidence": "FUND: fcf_to_net_income 0.9987"},
                {"argument": "Brand moat", "strength": "secondary", "evidence": "RSRCH: moat_trend strengthening"},
            ],
            "asymmetry_assessment": "Modest upside, limited downside.",
        },
    }


def test_advocate_summary_shows_each_core_argument_with_its_strength_and_evidence():
    summary = build_advocate_summary(_bull_with_detail(), "BULL")
    assert "- [primary] Cash conversion | evidence: FUND: fcf_to_net_income 0.9987" in summary
    assert "- [secondary] Brand moat | evidence: RSRCH: moat_trend strengthening" in summary


def test_advocate_summary_carries_the_asymmetry_caveats_and_labelled_narrative_last():
    summary = build_advocate_summary(_bull_with_detail(), "BULL")
    assert "asymmetry_assessment: Modest upside, limited downside." in summary
    assert "caveats: RSRCH confidence is medium." in summary
    assert "narrative (supporting detail; the structured fields above are authoritative): Coca-Cola converts" in summary
    assert summary.rindex("narrative (supporting detail") > summary.index("core_arguments:")


def test_advocate_summary_omits_what_the_advocate_did_not_produce():
    summary = build_advocate_summary({"recommendation": "bearish", "confidence": 30, "thesis_summary": "t"}, "BEAR")
    assert "narrative" not in summary and "caveats" not in summary and "asymmetry" not in summary


def test_the_shadow_cio_gets_the_same_advocate_detail():
    from agents import pass3_shadow_cio

    assert "narrative (supporting detail" in pass3_shadow_cio.build_advocates_block(_bull_with_detail(), _bear_output())


def _bear_output() -> dict:
    return {
        "recommendation": "bearish", "confidence": 35, "thesis_summary": "Premium valuation.",
        "strongest_argument": "P/E well above peers (FUND).", "weakest_point": "Cash conversion is strong (FUND).",
        "structured_data": {
            "core_arguments": [{"argument": "Valuation", "strength": "primary", "evidence": "FUND: pe_ratio 24"}],
            "downside_triggers": [{"trigger": "Volume decline"}],
        },
    }


def test_advocates_block_shows_both_sides_under_each_field_with_the_leading_side_alternating():
    block = build_advocates_block(_bull_with_detail(), _bear_output())
    lines = block.splitlines()
    # Field order is Bull's, then Bear-only fields; each shared field lists both advocates.
    first = lines.index("recommendation:")
    assert lines[first + 1].startswith("  BULL: bullish")
    assert lines[first + 2].startswith("  BEAR: bearish")
    second = lines.index("thesis_summary:")
    assert lines[second + 1].startswith("  BEAR: ") and lines[second + 2].startswith("  BULL: ")
    third = lines.index("strongest_argument:")
    assert lines[third + 1].startswith("  BULL: ")
    # Multi-line core_arguments keep their items under the right side.
    core = lines.index("core_arguments:")
    assert any("Cash conversion" in line for line in lines[core:core + 6])
    # Side-only fields appear once, under their own side.
    trig = lines.index("downside_triggers:")
    assert lines[trig + 1].startswith("  BEAR: ") and not any("BULL" in line for line in lines[trig + 2:trig + 3])


def test_advocates_block_carries_every_value_the_one_after_the_other_rendering_does():
    bull, bear = _bull_with_detail(), _bear_output()
    block = build_advocates_block(bull, bear)
    for text in (build_advocate_summary(bull, "BULL") + build_advocate_summary(bear, "BEAR")).splitlines():
        value = text.split(": ", 1)[-1].strip()
        if len(value) > 12 and "CASE ADVOCATE" not in value:
            assert value in block, value


def test_advocates_block_falls_back_when_an_advocate_is_missing():
    block = build_advocates_block(_bull_with_detail(), None)
    assert "BULL CASE ADVOCATE:" in block and "BEAR CASE ADVOCATE: NOT AVAILABLE" in block


def test_stage_a_prompt_and_message_carry_no_tail_risk_cross_check():
    from agents.prompts import load_template

    assert "TAIL-RISK" not in load_template("cio", stage="a")
    assert "TAIL-RISK" not in build_user_message(_bundle(), {}, {}, 20, "consensus")


def test_stage_a_schema_asks_for_the_reasoning_before_the_verdict():
    """The verdict used to be the first field written, so the assessments and factors could only justify it
    (PLUG: bear case 'accepted, high weight', risk 'unfavorable', four of five factors bearish, yet somewhat
    bullish). Replays with the reasoning first moved the two weak names to neutral and bearish."""
    from agents.prompts import load_template

    schema = load_template("cio", stage="a").split("## OUTPUT FORMAT")[1]
    for reasoning in ("bull_case_assessment", "bear_case_assessment", "risk_reward_consumed_as", "key_decision_factors"):
        assert schema.index(f'"{reasoning}"') < schema.index('"stock_outlook"')
    assert schema.index('"stock_outlook"') < schema.index('"confidence"') < schema.index('"expected_return_tier"')


def _with_own_case_fields(out: dict, side: str) -> dict:
    out = {**out, "thesis_risks": [{"risk": f"{side} invalidator", "severity": "high", "likelihood": "medium", "evidence": "FUND: payout_ratio 0.61"}]}
    out["structured_data"] = {
        **out.get("structured_data", {}),
        "timeline_focused_argument": f"{side} medium-term case.",
        "market_misreads": [{"misread": f"{side} misread", "why_this_persists": "neutral sentiment", "supporting_pass1_agents": ["SENT"], "evidence": "SENT: neutral"}],
        "valuation_argument": {"claim": f"{side} valuation claim", "supporting_pass1_agents": ["FUND"], "evidence": "FUND: pe_ratio 26.04"},
    }
    return out


def test_the_cio_sees_each_sides_timeline_case_market_misreads_and_own_invalidators():
    """These Pass 2 fields were produced for the CIO and never rendered (the reference pseudocode dropped them)."""
    block = build_advocates_block(_with_own_case_fields(_bull_with_detail(), "Bull"), _with_own_case_fields(_bear_output(), "Bear"))
    for text in ("Bull medium-term case.", "Bear medium-term case.", "Bull misread (why it persists: neutral sentiment)",
                 "Bear invalidator (severity high, likelihood medium) | evidence: FUND: payout_ratio 0.61",
                 "Bull valuation claim | evidence: FUND: pe_ratio 26.04", "Bear valuation claim | evidence: FUND: pe_ratio 26.04"):
        assert text in block, text
    assert "thesis_risks (what would invalidate this side's case):" in block


def test_stage_a_prompt_tells_the_cio_how_to_use_the_advocates_own_case_fields():
    from agents.prompts import load_template

    prompt = load_template("cio", stage="a")
    for name in ("timeline_focused_argument", "market_misreads", "thesis_risks"):
        assert name in prompt
    assert "conditions drawn from the thesis_risks of the side you back" in prompt


def test_stage_a_message_ends_with_the_ask():
    assert build_user_message(_bundle(), {}, {}, 20, "consensus").rstrip().endswith("then the verdict.")


def test_stage_b_gets_the_tax_fields_written_for_the_account_lens():
    tax = _tax_result_with_detail()
    tax["tax_profile"] = {**tax["tax_profile"], "account_axis_summary": "TFSA shelters all gains.",
                          "context_aware_strongest_argument": "Tax-free gains dominate.", "listing_summary": "TSX listing, no withholding."}
    summary = _build_tax_strategist_summary(tax)
    for text in ("account_axis_summary: TFSA shelters all gains.", "context_aware_strongest_argument: Tax-free gains dominate.",
                 "listing_summary: TSX listing, no withholding."):
        assert text in summary
    assert "account_axis_summary" not in _build_tax_strategist_summary(_tax_result_with_detail())


def test_stage_b_prompt_covers_an_unavailable_tax_strategist_and_tax_illustrations():
    from agents.prompts import load_template

    prompt = load_template("cio", stage="b")
    assert "NOT AVAILABLE" in prompt
    assert "is not a forecast for this stock" in prompt


def test_stage_b_prompt_keeps_the_room_point_to_a_tfsa_and_bars_hold_buy_sell():
    """PG in an RRSP came back with TFSA logic ("any loss would permanently reduce future room"); PG in a taxable account
    ended "the recommendation to hold the stock"."""
    from agents.prompts import load_template

    prompt = load_template("cio", stage="b")
    assert "that point is specific to a TFSA" in prompt
    assert "do not tell the investor to hold, buy or sell" in prompt
