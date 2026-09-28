"""Tests for agents/pass3_cio.py (86bbuhjup). Close to a clean port -- see
module docstring. Focus: the real seam (bundle.context/stock/company_info
reads) and a regression check that the byte-for-byte-ported summary
functions still work against real-shaped Pass 1/2 output dicts."""
from datetime import UTC, datetime
from types import SimpleNamespace

from agents.pass3_cio import (
    _build_tax_strategist_summary,
    build_advocate_summary,
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
