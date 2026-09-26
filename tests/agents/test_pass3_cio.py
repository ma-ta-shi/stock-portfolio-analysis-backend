"""Tests for agents/pass3_cio.py (86bbuhjup). Close to a clean port -- see
module docstring. Focus: the real seam (bundle.context/stock/company_info
reads) and a regression check that the byte-for-byte-ported summary
functions still work against real-shaped Pass 1/2 output dicts."""
from datetime import UTC, datetime
from types import SimpleNamespace

from agents.pass3_cio import (
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
