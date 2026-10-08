"""pytest tests for Pass 2 validators — Bull, Bear, Tax Strategist, Risk Advisor."""
import pytest
from agents.validators.pass2 import (
    validate_bull_advocate,
    validate_bear_advocate,
    validate_tax_strategist,
    validate_risk_advisor_stage_a,
    validate_tax_against_expectation,
    validate_tax_passthroughs,
    _evidence_carries_value,
)
from agents.validators.compression import (
    validate_reliability_warning_format,
    validate_stub_fundamental,
)
from agents.utils import build_pass1_reliability_warnings, compute_outlook_distance


# ─── Bull Advocate Tests ──────────────────────────────────────────────────────

def _valid_bull_output() -> dict:
    return {
        "recommendation": "bullish",
        "confidence": 72,
        "thesis_summary": "GlobalTech offers a compelling medium-term bull case for a TFSA account driven by accelerating AI copilot adoption and margin expansion. FCF yield of 4.8% provides a margin of safety.",
        "strongest_argument": "FUND: FCF yield 4.8% vs sector 3.2% with revenue growing 14.2% — a rare combination of quality and growth at a reasonable price.",
        "weakest_point": "TECH: Stock is trading in a sideways range below 52w highs; if the $178 support breaks, technical selling could pressure the stock before fundamentals reassert.",
        "weakest_point_type": "negative_catalyst",
        "caveats": ["RSRCH analysis_confidence=high but DataStream integration risk not yet quantifiable."],
        "thesis_risks": [
            {"risk": "SAP AI acceleration could erode moat", "severity": "high", "likelihood": "medium", "evidence": "RSRCH: SAP announced AI copilot roadmap"},
        ],
        "narrative": "GlobalTech Industries presents a compelling medium-term bull case anchored in three reinforcing themes. First, the AI copilot adoption trajectory exceeds management targets (RSRCH: 2,400 customers in 90 days) — a leading indicator that this feature cycle is genuine, not hype. FUND data reinforces the quality of this growth: FCF yield of 4.8% versus the sector average of 3.2% demonstrates that the business is self-funding its AI investment without diluting shareholders. Second, the operational leverage story is beginning to play out. FUND shows operating margin at 24.8% with CFO guidance of 120bps expansion in H2 2026 — this is a concrete, near-term catalyst with specific management accountability. For a TFSA medium-term holder, this margin expansion represents tax-free compounding. Third, the technical setup, while not strong, provides a reasonable entry. TECH identifies $178.50 as a support level that has held on multiple tests, suggesting institutional buying at this level. The confluence score of 1/3 is modest but not a red flag for a medium-term fundamental thesis. The bear case rests primarily on valuation (28.5x PE vs 26x sector) and competitive risk from SAP. MACRO headwinds from elevated rates are real but well-discounted by the 8.4% YTD pullback. SENT indicators show net positive insider activity with CEO conviction buy at $183 outweighing VP-level sales. The risk-reward remains skewed to the upside for a patient TFSA investor with a 12-18 month horizon, particularly given the tax-free nature of any capital appreciation realized from the margin expansion catalyst.",
        "structured_data": {
            "core_arguments": [
                {"argument": "AI copilot is inflecting — 2,400 activations in 90 days signals genuine product-market fit", "supporting_pass1_agents": ["RSRCH", "FUND"], "evidence": "RSRCH: AI copilot adoption above internal targets; FUND: FCF yield insulates from R&D spend", "strength": "primary"},
                {"argument": "Operating leverage inflection in H2 2026 is a concrete near-term catalyst", "supporting_pass1_agents": ["FUND"], "evidence": "FUND: CFO guided 120bps margin expansion in H2 2026", "strength": "secondary"},
            ],
            "catalysts": [
                {"catalyst": "H2 2026 margin expansion confirmation in Q2 earnings", "timing": "3_to_12_months", "probability": "high", "supporting_pass1_agents": ["FUND"], "evidence": "FUND: CFO guidance explicit"},
                {"catalyst": "AI copilot attach rate milestone — 5,000 activations would trigger analyst upgrades", "timing": "1_to_3_months", "probability": "medium", "supporting_pass1_agents": ["RSRCH", "SENT"], "evidence": "RSRCH: 2,400 in 90 days; SENT: 3 analyst upgrades in 30 days"},
            ],
            "valuation_argument": {
                "claim": "At 28.5x vs 26x sector, the premium is justified by 14.2% revenue growth and 4.8% FCF yield — 120bps above sector average.",
                "supporting_pass1_agents": ["FUND"],
                "evidence": "FUND: FCF yield 4.8% vs sector 3.2%; P/E 28.5x vs sector 26x",
            },
            "market_misreads": [
                {"misread": "Market is underweighting the AI copilot attach rate as a durable revenue stream vs. a feature", "why_this_persists": "AI monetization remains unproven across most software companies.", "supporting_pass1_agents": ["RSRCH", "SENT"], "evidence": "RSRCH: 2,400 activations well ahead of plan; SENT: analyst coverage still not pricing AI attach revenue"},
            ],
            "thesis_archetype_alignment": {
                "researcher_archetype": "quality_compounder",
                "bull_archetype": "quality_compounder",
                "alignment_with_researcher": "agrees",
                "disagreement_reason": "",
            },
            "context_aware_strongest_argument": "For a TFSA medium-term investor, the FCF yield and tax-free margin expansion compounding is the dominant argument — capital gains from operational leverage are untaxed in TFSA.",
            "asymmetry_assessment": "If the AI copilot cycle plays out, the stock re-rates to $210+ (+12%); if SAP disrupts the moat, downside is to the $160 book value range (-15%).",
        },
    }


class TestBullAdvocate:
    def test_valid_output_passes(self):
        out = _valid_bull_output()
        passed, errors = validate_bull_advocate(out)
        assert passed, f"Valid bull output should pass: {errors}"

    def test_recommendation_must_be_bullish(self):
        out = _valid_bull_output()
        out["recommendation"] = "neutral"
        passed, errors = validate_bull_advocate(out)
        assert not passed
        assert any("recommendation" in e for e in errors)

    def test_confidence_out_of_range_fails(self):
        out = _valid_bull_output()
        out["confidence"] = 105
        passed, errors = validate_bull_advocate(out)
        assert not passed

    def test_no_clear_invalidator_at_low_confidence_fails(self):
        out = _valid_bull_output()
        out["confidence"] = 60
        out["weakest_point_type"] = "no_clear_invalidator"
        passed, errors = validate_bull_advocate(out)
        assert not passed
        assert any("no_clear_invalidator" in e for e in errors)

    def test_no_clear_invalidator_at_high_confidence_allowed(self):
        out = _valid_bull_output()
        out["confidence"] = 75
        out["weakest_point_type"] = "no_clear_invalidator"
        passed, errors = validate_bull_advocate(out)
        assert passed, f"no_clear_invalidator should be allowed at confidence=75: {errors}"

    def test_high_confidence_with_no_clear_invalidator_fails(self):
        out = _valid_bull_output()
        out["confidence"] = 82
        out["weakest_point_type"] = "no_clear_invalidator"
        passed, errors = validate_bull_advocate(out)
        assert not passed
        assert any("80" in e or "confidence" in e for e in errors)

    def test_narrative_too_short_fails(self):
        out = _valid_bull_output()
        out["narrative"] = "Short bull case."
        passed, errors = validate_bull_advocate(out)
        assert not passed
        assert any("narrative" in e and "short" in e for e in errors)

    def test_narrative_missing_3_agents_fails(self):
        out = _valid_bull_output()
        out["narrative"] = "A RSRCH FUND-based narrative that only mentions two agents. " + "X" * 1500
        passed, errors = validate_bull_advocate(out)
        assert not passed
        assert any("3" in e and ("agent" in e or "Pass 1" in e) for e in errors)

    def test_no_primary_core_argument_fails(self):
        out = _valid_bull_output()
        for ca in out["structured_data"]["core_arguments"]:
            ca["strength"] = "secondary"  # no primary
        passed, errors = validate_bull_advocate(out)
        assert not passed
        assert any("primary" in e for e in errors)

    def test_invalid_archetype_fails(self):
        out = _valid_bull_output()
        out["structured_data"]["thesis_archetype_alignment"]["bull_archetype"] = "growth_compounder"
        passed, errors = validate_bull_advocate(out)
        assert not passed

    def test_invalid_alignment_with_researcher_fails(self):
        out = _valid_bull_output()
        out["structured_data"]["thesis_archetype_alignment"]["alignment_with_researcher"] = "somewhat"
        passed, errors = validate_bull_advocate(out)
        assert not passed
        assert any("alignment_with_researcher" in e for e in errors)

    def test_valid_alignment_with_researcher_disagrees_passes(self):
        out = _valid_bull_output()
        out["structured_data"]["thesis_archetype_alignment"]["alignment_with_researcher"] = "disagrees"
        out["structured_data"]["thesis_archetype_alignment"]["disagreement_reason"] = (
            "Researcher rates this a cyclical_recovery, but the durable AI copilot attach "
            "rate supports a quality_compounder read instead."
        )
        # researcher_archetype must actually match the narrative above (cyclical_recovery,
        # not quality_compounder) -- the base fixture's own bull_archetype is
        # quality_compounder, so leaving researcher_archetype at its base value
        # (quality_compounder, same as bull_archetype) would make this fixture assert the
        # exact live-observed contradiction TestArchetypeAlignmentCrossCheck below exists
        # to catch: same archetype label, but alignment_with_researcher="disagrees".
        out["structured_data"]["thesis_archetype_alignment"]["researcher_archetype"] = "cyclical_recovery"
        passed, errors = validate_bull_advocate(out)
        assert passed, errors

    def test_historical_analogy_is_gone_and_a_stray_one_is_ignored(self):
        out = _valid_bull_output()
        assert "historical_analogy" not in out["structured_data"]
        assert validate_bull_advocate(out)[0]
        out["structured_data"]["historical_analogy"] = {"company": "SomeRealCo", "analogy_fit": "weak"}
        assert validate_bull_advocate(out)[0]


# ─── Bear Advocate Tests ──────────────────────────────────────────────────────

def _valid_bear_output() -> dict:
    return {
        "recommendation": "bearish",
        "confidence": 55,
        "thesis_summary": "GlobalTech's premium valuation at 28.5x PE is not supported by the competitive dynamics. SAP's AI acceleration and the DataStream integration burden create meaningful downside risk in this medium-term TFSA context.",
        "strongest_argument": "RSRCH: SAP's AI roadmap directly attacks GlobalTech's switching cost moat — the primary bull thesis anchor. If moat erodes, the valuation premium collapses.",
        "weakest_point": "FUND: FCF yield of 4.8% provides genuine cushion that limits downside; the bear case requires moat erosion to play out faster than the market expects.",
        "caveats": [],
        "thesis_risks": [
            {"risk": "AI copilot success would be a fundamental thesis blow to bear case", "severity": "high", "likelihood": "medium", "evidence": "thesis assumption: bull case is plausible if attach rates sustain"},
        ],
        "narrative": "GlobalTech Industries faces a convergence of headwinds that the current premium valuation does not adequately price. The core of the bear thesis is moat erosion. RSRCH data reveals that SAP has announced a directly competing AI copilot product targeting the same mid-market ERP customer base. GlobalTech's 28% market share is built on high switching costs — if SAP's AI product achieves parity, those switching costs decline structurally. FUND metrics show the stock trades at 28.5x PE versus the sector's 26x median. This premium is historically justified by FCF generation and growth, but FUND's revenue growth at 14.2% is already starting to decelerate on a multi-quarter basis (QoQ growth of 3.8% annualizes to ~16% — decelerating from 14.2% YoY). TECH signals are not supportive: the stock is in a sideways-to-down trend with RSI at 51 — not oversold, not attractively washed out. SENT data shows mixed insider activity: VP-level sells in April alongside the CEO buy. MACRO headwinds from elevated 4.5% rates (MACRO) disproportionately compress growth multiples. The bear scenario is straightforward: if 2 of 3 catalysts fail (AI copilot growth stalls, SAP wins first renewal, DataStream integration disappoints), multiple contracts to 24-25x on slower growth, implying a stock price of $158-165 — a 12-16% decline from current levels. For a TFSA investor, this permanent capital loss scenario is particularly damaging as it destroys contribution room irreversibly. The DataStream integration D/E ratio of 0.42 (FUND) also constrains the balance sheet's ability to absorb execution misses.",
        "structured_data": {
            "core_arguments": [
                {"argument": "SAP AI roadmap threatens core switching cost moat — if SAP achieves parity, the PE premium collapses", "supporting_pass1_agents": ["RSRCH", "FUND"], "evidence": "RSRCH: SAP competing AI product; FUND: PE premium depends on moat durability", "strength": "primary"},
            ],
            "downside_triggers": [
                {"trigger": "First GlobalTech-to-SAP ERP migration reported by a named customer", "timing": "3_to_12_months", "probability": "low", "supporting_pass1_agents": ["RSRCH"], "evidence": "RSRCH: SAP AI roadmap"},
                {"trigger": "Q2 2026 guidance miss showing DataStream integration costs above plan", "timing": "1_to_3_months", "probability": "medium", "supporting_pass1_agents": ["FUND"], "evidence": "FUND: D/E 0.42 post-acquisition"},
            ],
            "valuation_argument": {
                "claim": "At 28.5x PE with decelerating growth risk, any catalyst miss could trigger multiple compression to 24-25x, implying $158-165 downside",
                "supporting_pass1_agents": ["FUND"],
                "evidence": "FUND: PE 28.5x vs sector 26x; revenue growth shows signs of deceleration",
            },
            "market_misreads": [
                {"misread": "Market overweights AI copilot NTM revenue without pricing execution risk", "why_this_persists": "AI narrative creates sentiment premium that delays multiple compression.", "supporting_pass1_agents": ["SENT", "RSRCH"], "evidence": "SENT: 3 upgrades on AI narrative; RSRCH: attach rates impressive but base is small at 2,400 customers"},
            ],
            "tail_risk_assessment": {
                "tail_risk_level": "moderate",
                "scenario": "DataStream integration fails AND SAP wins 3+ customers within 6 months, triggering multiple compression on top of margin pressure.",
                "triggering_event": "DataStream integration cost overrun disclosed alongside a named SAP customer win.",
                "supporting_pass1_agents": ["FUND", "RSRCH"],
                "evidence": "FUND: D/E 0.42 post-acquisition; RSRCH: SAP competing AI product",
            },
            # Real schema per backend/prompts/bear_advocate/v1.txt: researcher_archetype,
            # bear_archetype, agrees_with_researcher (bool), disagreement_note. The
            # fixture previously here had "bear_archetype_rationale" -- a field that
            # doesn't exist in the real prompt schema at all, and was missing
            # researcher_archetype/agrees_with_researcher/disagreement_note entirely.
            # Found only because nothing validated this object before 86bbt1k1p's
            # follow-up fix (86bbuhjup finding 5) -- corrected to match the real schema.
            "thesis_archetype_alignment": {
                "researcher_archetype": "value_trap_candidate",
                "bear_archetype": "value_trap_candidate",
                "agrees_with_researcher": True,
                "disagreement_note": "",
            },
            "context_aware_strongest_argument": "For a TFSA medium-term investor, a 15-22% capital loss would permanently destroy TFSA room — the downside risk is asymmetrically painful in this account type.",
            "asymmetry_assessment": "If the bear case plays out, downside is 15-22% (to $145-160); if wrong, upside is limited to 10-15% at 30x PE.",
        },
    }


class TestBearAdvocate:
    def test_valid_output_passes(self):
        out = _valid_bear_output()
        passed, errors = validate_bear_advocate(out)
        assert passed, f"Valid bear output should pass: {errors}"

    def test_recommendation_must_be_bearish(self):
        out = _valid_bear_output()
        out["recommendation"] = "neutral"
        passed, errors = validate_bear_advocate(out)
        assert not passed

class TestBearArchetypeAlignment:
    """86bbuhjup finding 5: this whole object was completely unvalidated before this fix
    -- Bull's equivalent fields were checked, Bear's were not, confirmed via grep (zero
    matches). Mirrors TestBullAdvocate's existing archetype/alignment tests field-for-
    field, adjusted for Bear's real schema (agrees_with_researcher is a bool, not a
    string enum like Bull's alignment_with_researcher)."""

    def test_invalid_bear_archetype_fails(self):
        out = _valid_bear_output()
        out["structured_data"]["thesis_archetype_alignment"]["bear_archetype"] = "growth_compounder"
        passed, errors = validate_bear_advocate(out)
        assert not passed
        assert any("bear_archetype" in e for e in errors)

    def test_non_bool_agrees_with_researcher_fails(self):
        out = _valid_bear_output()
        out["structured_data"]["thesis_archetype_alignment"]["agrees_with_researcher"] = "true"
        passed, errors = validate_bear_advocate(out)
        assert not passed
        assert any("agrees_with_researcher" in e and "boolean" in e for e in errors)

    def test_disagrees_with_empty_note_fails(self):
        out = _valid_bear_output()
        out["structured_data"]["thesis_archetype_alignment"]["agrees_with_researcher"] = False
        out["structured_data"]["thesis_archetype_alignment"]["disagreement_note"] = ""
        passed, errors = validate_bear_advocate(out)
        assert not passed
        assert any("disagreement_note" in e for e in errors)

    def test_disagrees_with_real_note_passes(self):
        out = _valid_bear_output()
        out["structured_data"]["thesis_archetype_alignment"] = {
            "researcher_archetype": "cyclical_recovery",
            "bear_archetype": "value_trap_candidate",
            "agrees_with_researcher": False,
            "disagreement_note": "Researcher rates this a cyclical_recovery, but the "
            "structural moat erosion supports a value_trap_candidate read instead.",
        }
        passed, errors = validate_bear_advocate(out)
        assert passed, errors


class TestArchetypeAlignmentCrossCheck:
    """86bbt1k1p follow-up (86bbuhjup findings 5+7): alignment_with_researcher /
    agrees_with_researcher must actually reflect whether the advocate's own archetype
    matches the REAL researcher archetype -- live-observed contradiction on a real
    scenario_03 run (bull_archetype == researcher_archetype, both
    "value_trap_candidate", but alignment_with_researcher == "disagrees"). Checked
    against real_researcher_archetype, the caller-supplied ground-truth value, not the
    advocate's own echoed copy of it."""

    def test_bull_same_archetype_but_disagrees_fails(self):
        """The exact live-observed bug, reproduced as a regression test."""
        out = _valid_bull_output()
        out["structured_data"]["thesis_archetype_alignment"] = {
            "researcher_archetype": "value_trap_candidate",
            "bull_archetype": "value_trap_candidate",
            "alignment_with_researcher": "disagrees",
            "disagreement_reason": "The bullish case hinges on a turnaround catalyst.",
        }
        passed, errors = validate_bull_advocate(out, real_researcher_archetype="value_trap_candidate")
        assert not passed
        assert any("alignment_with_researcher" in e and "matches" in e for e in errors)

    def test_bull_different_archetype_but_agrees_fails(self):
        out = _valid_bull_output()
        out["structured_data"]["thesis_archetype_alignment"] = {
            "researcher_archetype": "cyclical_recovery",
            "bull_archetype": "quality_compounder",
            "alignment_with_researcher": "agrees",
            "disagreement_reason": "",
        }
        passed, errors = validate_bull_advocate(out, real_researcher_archetype="cyclical_recovery")
        assert not passed
        assert any("alignment_with_researcher" in e and "differs" in e for e in errors)

    def test_bull_consistent_agrees_passes(self):
        out = _valid_bull_output()  # base fixture: bull_archetype == researcher_archetype == "quality_compounder", alignment="agrees"
        passed, errors = validate_bull_advocate(out, real_researcher_archetype="quality_compounder")
        assert passed, errors

    def test_bull_cross_check_skipped_when_real_value_not_supplied(self):
        """Backward-compat: the exact contradiction from test_bull_same_archetype_but_
        disagrees_fails must NOT fail when real_researcher_archetype isn't passed --
        every pre-existing caller (including every test written before this fix)."""
        out = _valid_bull_output()
        out["structured_data"]["thesis_archetype_alignment"] = {
            "researcher_archetype": "value_trap_candidate",
            "bull_archetype": "value_trap_candidate",
            "alignment_with_researcher": "disagrees",
            "disagreement_reason": "The bullish case hinges on a turnaround catalyst.",
        }
        passed, errors = validate_bull_advocate(out)
        assert passed, errors

    def test_bear_same_archetype_but_disagrees_fails(self):
        out = _valid_bear_output()
        out["structured_data"]["thesis_archetype_alignment"] = {
            "researcher_archetype": "value_trap_candidate",
            "bear_archetype": "value_trap_candidate",
            "agrees_with_researcher": False,
            "disagreement_note": "Some rationale.",
        }
        passed, errors = validate_bear_advocate(out, real_researcher_archetype="value_trap_candidate")
        assert not passed
        assert any("agrees_with_researcher" in e and "matches" in e for e in errors)

    def test_bear_different_archetype_but_agrees_fails(self):
        out = _valid_bear_output()
        out["structured_data"]["thesis_archetype_alignment"] = {
            "researcher_archetype": "secular_grower",
            "bear_archetype": "value_trap_candidate",
            "agrees_with_researcher": True,
            "disagreement_note": "",
        }
        passed, errors = validate_bear_advocate(out, real_researcher_archetype="secular_grower")
        assert not passed
        assert any("agrees_with_researcher" in e and "differs" in e for e in errors)

    def test_bear_consistent_disagrees_passes(self):
        out = _valid_bear_output()
        out["structured_data"]["thesis_archetype_alignment"] = {
            "researcher_archetype": "secular_grower",
            "bear_archetype": "value_trap_candidate",
            "agrees_with_researcher": False,
            "disagreement_note": "Structural moat erosion supports a value trap read instead.",
        }
        passed, errors = validate_bear_advocate(out, real_researcher_archetype="secular_grower")
        assert passed, errors

    def test_narrative_too_short_fails(self):
        out = _valid_bear_output()
        out["narrative"] = "Short bear case."
        passed, errors = validate_bear_advocate(out)
        assert not passed


class TestBearTailRiskAssessment:
    """Rule 7 (86bbuhk82 item 2): content requirements are CONDITIONAL on tail_risk_level,
    not unconditional -- confirmed live (a real 'negligible' call correctly returned
    scenario/triggering_event/evidence all empty; requiring them non-empty unconditionally
    would have rejected the documented, expected case)."""

    def test_negligible_with_empty_content_passes(self):
        out = _valid_bear_output()
        out["structured_data"]["tail_risk_assessment"] = {
            "tail_risk_level": "negligible", "scenario": "", "triggering_event": "",
            "supporting_pass1_agents": [], "evidence": "",
        }
        passed, errors = validate_bear_advocate(out)
        assert passed, f"negligible with empty content is the documented, expected case: {errors}"

    def test_elevated_with_empty_content_fails(self):
        out = _valid_bear_output()
        out["structured_data"]["tail_risk_assessment"] = {
            "tail_risk_level": "elevated", "scenario": "", "triggering_event": "",
            "supporting_pass1_agents": [], "evidence": "",
        }
        passed, errors = validate_bear_advocate(out)
        assert not passed
        assert any("scenario" in e for e in errors)
        assert any("triggering_event" in e for e in errors)
        assert any("evidence" in e for e in errors)
        assert any("supporting_pass1_agents" in e for e in errors)

    def test_moderate_with_empty_content_fails(self):
        out = _valid_bear_output()
        out["structured_data"]["tail_risk_assessment"] = {
            "tail_risk_level": "moderate", "scenario": "", "triggering_event": "",
            "supporting_pass1_agents": [], "evidence": "",
        }
        passed, errors = validate_bear_advocate(out)
        assert not passed

    def test_elevated_with_content_passes(self):
        out = _valid_bear_output()
        out["structured_data"]["tail_risk_assessment"] = {
            "tail_risk_level": "elevated",
            "scenario": "A sustained oil price collapse below $60/bbl for 12+ months.",
            "triggering_event": "WTI closing below $60 for 3 consecutive months.",
            "supporting_pass1_agents": ["MACRO"],
            "evidence": "MACRO: commodity_context",
        }
        passed, errors = validate_bear_advocate(out)
        assert passed, errors

    def test_missing_tail_risk_level_fails(self):
        out = _valid_bear_output()
        out["structured_data"]["tail_risk_assessment"] = {
            "scenario": "", "triggering_event": "", "supporting_pass1_agents": [], "evidence": "",
        }
        passed, errors = validate_bear_advocate(out)
        assert not passed
        assert any("tail_risk_level" in e for e in errors)

    def test_invalid_tail_risk_level_value_fails(self):
        """The enum-value check itself is the shared sweep (tail_risk_level is in
        DECLARED_ENUMS), not this conditional-content check -- verifying they compose."""
        out = _valid_bear_output()
        out["structured_data"]["tail_risk_assessment"]["tail_risk_level"] = "catastrophic"
        passed, errors = validate_bear_advocate(out)
        assert not passed
        assert any("tail_risk_level" in e and "catastrophic" in e for e in errors)


# ─── Tax Strategist Tests ─────────────────────────────────────────────────────

def _valid_tax_output() -> dict:
    return {
        "groundedness_score": 82,
        "thesis_summary": "Holding GLBL in a TFSA incurs a 15% US withholding tax on the 1.8% dividend that is not recoverable under the Canada-US tax treaty. The effective yield drops to 1.53%. For a pure capital appreciation thesis in TFSA, the tax drag is minimal.",
        "strongest_signal": "WHT: 15% US withholding tax on dividends is NOT recoverable in a TFSA — this costs 0.27% annually on the 1.8% yield.",
        "caveats": ["Tax rules verified against CRA guidance as of March 2026."],
        "narrative": "GlobalTech Industries (GLBL) is a US-domiciled company (DOM) trading on NASDAQ (LIST). The dividend yield of 1.8% (DIVID) triggers US withholding tax of 15% (WHT) when held in a TFSA. Unlike an RRSP, a TFSA is not recognized as a retirement account under the Canada-US tax treaty (Article XVIII), so the withholding tax exemption does not apply. This results in an effective after-tax yield of 1.53% (ELIG: not eligible for Canadian dividend tax credit as a US-domiciled stock). The 0.27% annual tax drag from withholding is modest and should not be the primary factor in the investment decision for a capital-appreciation-oriented thesis. For a medium-term TFSA investor, the dominant tax advantage is the capital gains exemption (CGAIN) — any price appreciation from $187 to the analyst target of $205 would be entirely tax-free in a TFSA, compared to a 50% inclusion rate in a taxable account. The ROOM consideration (ROOM) is relevant: holding a high-growth stock in TFSA maximizes the tax-free compounding advantage. If an investor expects 10-15% total return from GlobalTech, holding it in TFSA rather than trading account saves approximately $500-750 per $5,000 position in capital gains taxes annually at a 40% marginal rate. However, if dividend income is the primary objective (e.g., for retirement income), the RRSP is superior due to the WHT exemption, which would save the 0.27% annual drag and preserve the full 1.8% yield. Summary recommendation: TFSA is a good fit for capital-appreciation investors; RRSP is marginally better for income-focused investors due to WHT exemption on the dividend. FUND reports a payout ratio of 22%, so the dividend is a small component of total return and the withholding drag stays secondary; MACRO notes no pending treaty change that would alter the Article XVIII treatment over the holding period (REF/TFSA: contribution room is restored on January 1 of the year following a sell, so an exit is not permanently costly).",
        "tax_profile": {
            "account_fit_score": "good",
            "tax_efficiency_for_account": "favorable",
            "dividend_yield_pct": 1.8,
            "withholding_tax_rate_pct": 15.0,
            "effective_after_tax_yield_pct": 1.53,
            "is_eligible_canadian_dividend": False,
            # 86bc8eg3j: the real schema field is capital_gains_treatment_summary
            # (prompts/tax_strategist/v1.txt Step 5/Rule 8), not capital_gains_treatment
            # -- this fixture's own field name predates that and was never checked by
            # anything, so the mismatch went unnoticed until a real validator check
            # (added here) started requiring the real field name.
            "capital_gains_treatment": "Tax-free in TFSA — all capital gains are completely exempt from Canadian tax.",
            "capital_gains_treatment_summary": "Tax-free in TFSA — all capital gains are completely exempt from Canadian tax.",
            "loss_risk_assessment": "Capital losses in TFSA permanently reduce available TFSA room and cannot be used to offset gains elsewhere.",
            "cross_account_recommendation": None,
            "recommended_account": "current_account_is_optimal",
            # Rule 12 requires 1-4. This fixture carried none until 2026-09-01 -- the
            # canonical "valid" example violated three of the prompt's own rules
            # (no Pass 1 ID in the narrative, no REF, no key_tax_risks) because
            # nothing had ever enforced them.
            "key_tax_risks": [
                {
                    "risk": "A future US estate-tax filing obligation arises if US-situs holdings exceed USD 60,000.",
                    "severity": "low",
                    "evidence": "DOM: US-domiciled ordinary shares are US-situs property",
                },
                {
                    "risk": "The 15% withholding is permanently lost in a TFSA and cannot be reclaimed as a foreign tax credit.",
                    "severity": "medium",
                    "evidence": "WHT: 15% non-recoverable in TFSA per REF/TFSA",
                },
            ],
            "tax_optimization_actions": [],
        },
    }


class TestTaxStrategist:
    def test_valid_output_passes(self):
        out = _valid_tax_output()
        passed, errors = validate_tax_strategist(out)
        assert passed, f"Valid tax output should pass: {errors}"

    def test_no_recommendation_required(self):
        out = _valid_tax_output()
        out["recommendation"] = "buy"  # must NOT have this
        passed, errors = validate_tax_strategist(out)
        assert not passed
        assert any("recommendation" in e for e in errors)

    def test_confidence_instead_of_groundedness_fails(self):
        out = _valid_tax_output()
        del out["groundedness_score"]
        out["confidence"] = 75
        passed, errors = validate_tax_strategist(out)
        assert not passed
        assert any("groundedness_score" in e or "confidence" in e for e in errors)

    def test_a_poor_fit_without_a_move_is_not_rejected_by_the_structural_check(self):
        """Whether a move is required is decided by the code (validate_tax_against_expectation):
        a poor fit with no account that has room left legitimately has none."""
        out = _valid_tax_output()
        out["tax_profile"]["account_fit_score"] = "poor"
        out["tax_profile"]["cross_account_recommendation"] = None
        passed, errors = validate_tax_strategist(out)
        assert not any("cross_account_recommendation" in e for e in errors)

    def test_poor_account_fit_with_recommendation_passes(self):
        out = _valid_tax_output()
        out["tax_profile"]["account_fit_score"] = "poor"
        out["tax_profile"]["cross_account_recommendation"] = "Consider moving to RRSP to recover US dividend withholding tax."
        passed, errors = validate_tax_strategist(out)
        assert passed, f"poor account_fit with recommendation should pass: {errors}"

    def test_invalid_account_fit_score_fails(self):
        out = _valid_tax_output()
        out["tax_profile"]["account_fit_score"] = "bad"  # invalid
        passed, errors = validate_tax_strategist(out)
        assert not passed

    def test_invalid_tax_efficiency_fails(self):
        out = _valid_tax_output()
        out["tax_profile"]["tax_efficiency_for_account"] = "positive"  # invalid
        passed, errors = validate_tax_strategist(out)
        assert not passed

    def test_invalid_dividend_classification_fails(self):
        """Declared in the real schema, unenforced until 86bbuhk82 -- surfaced by the
        permanent enum-inventory test, not a live failure."""
        out = _valid_tax_output()
        out["tax_profile"]["dividend_classification"] = "domestic"  # invalid
        passed, errors = validate_tax_strategist(out)
        assert not passed
        assert any("dividend_classification" in e for e in errors)

    def test_missing_capital_gains_treatment_summary_fails(self):
        """86bc8eg3j: requested by the prompt's Output Schema (Step 5/Rule 8) but
        never enforced until now -- added once this field started flowing into
        the CIO's own Stage B input, to hold it to the same reliability bar as
        tax_optimization_actions."""
        out = _valid_tax_output()
        del out["tax_profile"]["capital_gains_treatment_summary"]
        passed, errors = validate_tax_strategist(out)
        assert not passed
        assert any("capital_gains_treatment_summary" in e for e in errors)

    def test_empty_capital_gains_treatment_summary_fails(self):
        out = _valid_tax_output()
        out["tax_profile"]["capital_gains_treatment_summary"] = "   "
        passed, errors = validate_tax_strategist(out)
        assert not passed
        assert any("capital_gains_treatment_summary" in e for e in errors)

    def test_valid_dividend_classification_passes(self):
        out = _valid_tax_output()
        out["tax_profile"]["dividend_classification"] = "us"
        passed, errors = validate_tax_strategist(out)
        assert passed, errors

    def test_valid_better_account_passes(self):
        out = _valid_tax_output()
        out["tax_profile"]["account_fit_score"] = "poor"
        out["tax_profile"]["cross_account_recommendation"] = {
            "better_account": "rrsp",
            "reasoning": "RRSP treaty exemption recovers the full withholding tax.",
            "drag_delta_pct": -0.27,
        }
        passed, errors = validate_tax_strategist(out)
        assert passed, errors

    def test_invalid_applies_to_fails(self):
        out = _valid_tax_output()
        out["tax_profile"]["tax_optimization_actions"] = [
            {"action": "Move the position.", "applies_to": "savings_account",
             "justification": "WHT: exempt in RRSP under REF treaty Art. XVIII."}
        ]
        passed, errors = validate_tax_strategist(out)
        assert not passed
        assert any("applies_to" in e for e in errors)

    def test_valid_applies_to_passes(self):
        out = _valid_tax_output()
        out["tax_profile"]["tax_optimization_actions"] = [
            {"action": "Move the position to an RRSP.", "applies_to": "rrsp",
             "justification": "WHT: the 15% withholding is exempt in an RRSP under REF treaty Art. XVIII."}
        ]
        passed, errors = validate_tax_strategist(out)
        assert passed, errors

    def test_passthrough_tolerance(self):
        out = _valid_tax_output()
        fixture = {
            "fundamental_data": {"dividend_yield_pct": 1.8, "eligible_canadian_dividend": False},
            "orchestrator_precomputed": {
                "WHT_rate_pct": 15.0,
                "effective_after_tax_yield_tfsa_pct": 1.53,
            },
        }
        passed, errors = validate_tax_passthroughs(out, fixture, "tfsa")
        assert passed, f"Passthroughs within tolerance should pass: {errors}"

    def test_passthrough_out_of_tolerance_fails(self):
        out = _valid_tax_output()
        out["tax_profile"]["dividend_yield_pct"] = 3.5  # wrong value
        fixture = {
            "fundamental_data": {"dividend_yield_pct": 1.8, "eligible_canadian_dividend": False},
            "orchestrator_precomputed": {
                "WHT_rate_pct": 15.0,
                "effective_after_tax_yield_tfsa_pct": 1.53,
            },
        }
        passed, errors = validate_tax_passthroughs(out, fixture, "tfsa")
        assert not passed
        assert any("dividend_yield_pct" in e for e in errors)


# ─── Risk Advisor Tests (ClickUp 86bbuhk82; stage B removed 2026-10-01) ───────────────
# `risk_profile` holds volatility_assessment/beta_interpretation/
# max_drawdown_interpretation/downside_scenarios/risk_reward_ratio/data_sanity_flags.

def _valid_risk_stage_a_output() -> dict:
    return {
        "groundedness_score": 78,
        "thesis_summary": "GlobalTech presents a moderate risk profile. Beta of 1.18 amplifies market moves. The DataStream acquisition increases leverage but FCF is strong enough to manage it. TFSA medium-term investors face asymmetric loss risk from permanent room destruction.",
        "strongest_signal": "FUND: D/E of 0.42 post-DataStream acquisition is the highest in 5 years — rising leverage in a high-rate environment constrains financial flexibility.",
        "caveats": [],
        "narrative": "GlobalTech Industries presents a moderate-to-acceptable risk profile for a TFSA medium-term investment. The primary risk is the post-DataStream acquisition leverage (FUND: D/E 0.42), which is not alarming in absolute terms but represents a significant increase from the historical 0.15 baseline. In a 4.5% rate environment (MACRO), higher leverage amplifies interest expense sensitivity, and a further rate increase would compound the refinancing risk on any floating-rate debt taken on for the acquisition. The beta of 1.18 (BETA: injected) means GlobalTech will experience roughly 18% more volatility than the broader market on both the upside and downside — manageable for a medium-term holder but meaningful for a TFSA account, where realized losses permanently reduce sheltered contribution room rather than merely deferring a tax bill. The 52-week high-to-low range (VOL: $215.80 to $142.30) demonstrates real downside exposure of 34% from the prior peak, though the stock is currently trading at $187.42 — well above the 52-week low, suggesting the worst of the drawdown has likely already occurred absent a new catalyst. Two distinct downside scenarios exist: (1) SAP competitive disruption causing multiple compression as switching costs erode, and (2) DataStream integration failure driving a delay to the promised margin expansion. These two scenarios are structurally distinct and not meaningfully correlated with each other, which provides a fair stress test of the risk framework rather than two versions of the same underlying event. Overall, GlobalTech represents an appropriate risk profile for a TFSA medium-term position sized moderately rather than aggressively.",
        "risk_profile": {
            "volatility_assessment": "moderate",
            "beta_interpretation": "A beta of 1.18 means GlobalTech moves about 18% more than the broader market on average.",
            "max_drawdown_interpretation": "The 34% peak-to-trough decline (VOL: $215.80 to $142.30) took roughly two quarters to recover from, per DD.",
            "downside_scenarios": [
                {"scenario": "SAP competitive disruption triggers customer churn and multiple compression", "probability": "medium", "estimated_impact_pct": -18.0, "trigger": "RSRCH: SAP AI roadmap is a direct competitive threat", "timeline": "3_to_12_months"},
                {"scenario": "DataStream integration costs exceed guidance, delaying margin expansion 2+ quarters", "probability": "medium", "estimated_impact_pct": -10.0, "trigger": "FUND: D/E 0.42 post-acquisition, integration risk noted in filing", "timeline": "1_to_3_months"},
            ],
            "risk_reward_ratio": "neutral",
            "data_sanity_flags": [],
        },
    }


class TestRiskAdvisorStageA:
    def test_valid_output_passes(self):
        out = _valid_risk_stage_a_output()
        passed, errors = validate_risk_advisor_stage_a(out)
        assert passed, f"Valid risk Stage A output should pass: {errors}"

    def test_no_recommendation_field(self):
        out = _valid_risk_stage_a_output()
        out["recommendation"] = "underweight"
        passed, errors = validate_risk_advisor_stage_a(out)
        assert not passed
        assert any("recommendation" in e for e in errors)

    def test_too_few_downside_scenarios_fails(self):
        out = _valid_risk_stage_a_output()
        out["risk_profile"]["downside_scenarios"] = [out["risk_profile"]["downside_scenarios"][0]]
        passed, errors = validate_risk_advisor_stage_a(out)
        assert not passed
        assert any("downside_scenarios" in e for e in errors)

    def test_scenario_impacts_close_together_are_accepted(self):
        """The 2-point spacing rule was dropped 2026-10-07: across 465 saved scenario pairs it never caught a duplicated cause (wording
        overlap of the rejected pairs 0.04, the same as every other pair) and forced the model to spread what it judged equal."""
        out = _valid_risk_stage_a_output()
        out["risk_profile"]["downside_scenarios"][0]["estimated_impact_pct"] = -18.0
        out["risk_profile"]["downside_scenarios"][1]["estimated_impact_pct"] = -17.5
        passed, errors = validate_risk_advisor_stage_a(out)
        assert not any("close" in e for e in errors)

    def test_invalid_risk_reward_ratio_fails(self):
        out = _valid_risk_stage_a_output()
        out["risk_profile"]["risk_reward_ratio"] = "good"  # must be favorable|neutral|unfavorable
        passed, errors = validate_risk_advisor_stage_a(out)
        assert not passed

    def test_invalid_volatility_assessment_fails(self):
        """Never checked before this rewrite -- the field existed in the real schema
        but had no validator coverage at all."""
        out = _valid_risk_stage_a_output()
        out["risk_profile"]["volatility_assessment"] = "extreme"
        passed, errors = validate_risk_advisor_stage_a(out)
        assert not passed
        assert any("volatility_assessment" in e for e in errors)

    def test_invalid_scenario_probability_fails(self):
        """Also never checked before -- only the count and impact-pct spacing were."""
        out = _valid_risk_stage_a_output()
        out["risk_profile"]["downside_scenarios"][0]["probability"] = "certain"
        passed, errors = validate_risk_advisor_stage_a(out)
        assert not passed
        assert any("probability" in e for e in errors)

    def test_invalid_scenario_timeline_fails(self):
        out = _valid_risk_stage_a_output()
        out["risk_profile"]["downside_scenarios"][0]["timeline"] = "next_quarter"
        passed, errors = validate_risk_advisor_stage_a(out)
        assert not passed
        assert any("timeline" in e for e in errors)


# ---------- validate_tax_against_expectation: the code decides, the model copies ----------


def _expectation(account="trading", classification="us", yield_pct=2.43):
    from data.precompute.tax_metrics import compute_account_verdict, resolve_withholding

    return {
        "yield_pct": yield_pct,
        "withholding": resolve_withholding(classification, account),
        "verdict": compute_account_verdict(account, classification, yield_pct, 29.65, 6.39),
    }


def _matching_trading_output() -> dict:
    """What the model should write for KO (2.43% US dividend) in a Trading account."""
    out = _valid_tax_output()
    out["tax_profile"].update({
        "dividend_yield_pct": 2.4, "withholding_tax_rate_pct": 15.0, "annual_tax_drag_pct": 0.72,
        "effective_after_tax_yield_pct": 1.71, "account_fit_score": "fair", "tax_efficiency_for_account": "neutral",
    })
    return out


class TestTaxAgainstExpectation:
    def test_an_output_that_copies_the_verdict_passes(self):
        assert validate_tax_against_expectation(_matching_trading_output(), _expectation()) == []

    def test_a_wrong_fit_or_efficiency_is_rejected(self):
        out = _matching_trading_output()
        out["tax_profile"]["account_fit_score"] = "poor"
        out["tax_profile"]["tax_efficiency_for_account"] = "unfavorable"
        errors = validate_tax_against_expectation(out, _expectation())
        assert any("account_fit_score" in e and "'fair'" in e for e in errors)
        assert any("tax_efficiency_for_account" in e and "'neutral'" in e for e in errors)

    def test_the_passthroughs_are_checked_against_the_block(self):
        out = _matching_trading_output()
        out["tax_profile"].update({"dividend_yield_pct": 4.0, "withholding_tax_rate_pct": 0.0,
                                   "annual_tax_drag_pct": 0.4, "effective_after_tax_yield_pct": 2.1})
        errors = validate_tax_against_expectation(out, _expectation())
        for field in ("dividend_yield_pct", "withholding_tax_rate_pct", "annual_tax_drag_pct", "effective_after_tax_yield_pct"):
            assert any(field in e for e in errors), field

    def test_not_computed_defaults_are_enforced(self):
        from data.precompute.tax_metrics import compute_account_verdict

        expectation = {"yield_pct": 2.0, "withholding": None,
                       "verdict": compute_account_verdict("tfsa", "us_reit", 2.0, 29.65, 6.39)}
        out = _valid_tax_output()
        out["tax_profile"].update({"account_fit_score": "excellent", "tax_efficiency_for_account": "favorable"})
        errors = validate_tax_against_expectation(out, expectation)
        assert any("'fair'" in e for e in errors) and any("'neutral'" in e for e in errors)

    def test_no_dividend_expects_zero_or_null_figures(self):
        expectation = _expectation("trading", "canadian_eligible", None)
        out = _valid_tax_output()
        out["tax_profile"].update({"dividend_yield_pct": None, "withholding_tax_rate_pct": 0.0,
                                   "annual_tax_drag_pct": 0.0, "effective_after_tax_yield_pct": 0.0,
                                   "account_fit_score": "good", "tax_efficiency_for_account": "favorable"})
        assert validate_tax_against_expectation(out, expectation) == []
        out["tax_profile"]["annual_tax_drag_pct"] = 1.2
        assert validate_tax_against_expectation(out, expectation)

    def test_an_etf_has_no_verdict_so_nothing_is_checked(self):
        assert validate_tax_against_expectation(_valid_tax_output(), {"verdict": None}) == []


# ─── Compression / Reliability Warning Tests ──────────────────────────────────

class TestReliabilityWarnings:
    def test_all_high_produces_empty_string(self):
        levels = {"RSRCH": "high", "FUND": "high", "TECH": "high", "SENT": "high", "MACRO": "high"}
        result = build_pass1_reliability_warnings(levels)
        assert result == "", f"All 'high' should produce empty string, got: '{result}'"

    def test_mixed_confidence_produces_warning(self):
        levels = {"RSRCH": "high", "FUND": "medium", "TECH": "low", "SENT": "insufficient", "MACRO": "high"}
        result = build_pass1_reliability_warnings(levels)
        assert "caution" in result
        assert "unreliable" in result
        assert "skipped" in result
        assert "OK" in result

    def test_warning_format_valid(self):
        warning = "RSRCH: high (OK) | FUND: medium (caution) | TECH: low (unreliable) | SENT: insufficient (skipped) | MACRO: high (OK)"
        passed, errors = validate_reliability_warning_format(warning)
        assert passed, f"Valid warning format should pass: {errors}"

    def test_empty_string_is_valid(self):
        passed, errors = validate_reliability_warning_format("")
        assert passed

    def test_wrong_status_for_confidence_fails(self):
        warning = "RSRCH: medium (OK)"  # medium confidence labeled OK but should be caution
        passed, errors = validate_reliability_warning_format(warning)
        assert not passed


class TestStubFundamental:
    def test_stub_passes_when_insufficient(self):
        pass1 = {"FUND": {"analysis_confidence": "insufficient", "pass2_view": {}}}
        passed, errors = validate_stub_fundamental(pass1)
        assert passed, f"Correct stub should pass: {errors}"

    def test_stub_passes_when_low(self):
        """Live-verified (86bbummwp): the real, current model output for this exact fixture
        profile is 'low', not 'insufficient' -- accepted deliberately, not a broken test."""
        pass1 = {"FUND": {"analysis_confidence": "low", "pass2_view": {}}}
        passed, errors = validate_stub_fundamental(pass1)
        assert passed, f"'low' + empty pass2_view should pass: {errors}"

    def test_stub_fails_when_confidence_not_insufficient_or_low(self):
        pass1 = {"FUND": {"analysis_confidence": "medium", "pass2_view": {"health_rating": "stressed"}}}
        passed, errors = validate_stub_fundamental(pass1)
        assert not passed
        assert any("insufficient" in e for e in errors)

    def test_stub_fails_when_pass2_view_nonempty(self):
        pass1 = {"FUND": {"analysis_confidence": "insufficient", "pass2_view": {"pe_ratio": 28.5}}}
        passed, errors = validate_stub_fundamental(pass1)
        assert not passed

    def test_stub_fails_when_fund_none(self):
        pass1 = {"FUND": None}
        passed, errors = validate_stub_fundamental(pass1)
        assert not passed


# ─── Disagreement Score Tests -- moved to test_utils.py (86bbuhkr1 rebuild) so this
# logic has one canonical test location instead of two independently-maintained
# copies (this codebase has already had a field-copy silently drift across 4 copies
# once -- researcher_thesis_archetype, per utils.py's own docstring). See
# test_utils.py::TestDisagreementScore.


# ─── Outlook Distance Tests (86bbt1kct) ────────────────────────────────────────

class TestOutlookDistance:
    OUTLOOKS = ["bullish", "somewhat_bullish", "neutral", "somewhat_bearish", "bearish"]

    def test_identical_outlook_is_zero_distance_not_high_divergence(self):
        for o in self.OUTLOOKS:
            distance, high_divergence = compute_outlook_distance(o, o)
            assert distance == 0
            assert high_divergence is False

    def test_full_flip_bullish_to_bearish_is_distance_4_high_divergence(self):
        distance, high_divergence = compute_outlook_distance("bullish", "bearish")
        assert distance == 4
        assert high_divergence is True

    def test_one_notch_apart_is_not_high_divergence(self):
        distance, high_divergence = compute_outlook_distance("bullish", "somewhat_bullish")
        assert distance == 1
        assert high_divergence is False

    def test_exactly_two_notches_is_not_high_divergence(self):
        # high_divergence is a STRICT >2, per docs/agents/shadow_cio.md's own
        # ">2 positions" wording -- exactly 2 must not count.
        distance, high_divergence = compute_outlook_distance("bullish", "neutral")
        assert distance == 2
        assert high_divergence is False

    def test_three_notches_is_high_divergence(self):
        distance, high_divergence = compute_outlook_distance("bullish", "somewhat_bearish")
        assert distance == 3
        assert high_divergence is True

    def test_all_25_pairs_are_symmetric_and_bounded(self):
        # Exhaustive over the full 5x5 outlook grid: distance is symmetric
        # (order of primary/shadow shouldn't matter) and never exceeds 4.
        for a in self.OUTLOOKS:
            for b in self.OUTLOOKS:
                d_ab, hd_ab = compute_outlook_distance(a, b)
                d_ba, hd_ba = compute_outlook_distance(b, a)
                assert d_ab == d_ba
                assert hd_ab == hd_ba
                assert 0 <= d_ab <= 4
                assert hd_ab == (d_ab > 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------- evidence must carry a VALUE, not just field names ----------
# Added 2026-08-31. Bull emitted "FUND: revenue_growth_yoy, roe" -- two field names,
# no values -- and the old check passed it because the string contained "FUND".
# Every case below is a string an advocate actually produced in a live run.


@pytest.mark.parametrize("text", [
    "FUND: revenue_growth_yoy, roe",      # measured Bull failure, pre-fix
    "RSRCH: top_growth_drivers",          # measured Bull failure, pre-fix
    "TECH: primary_trend",
    "FUND",
])
def test_bare_field_names_are_rejected(text):
    assert not _evidence_carries_value(text)


@pytest.mark.parametrize("text", [
    "FUND: revenue_growth_yoy 15.7%",                  # numeric
    "FUND: debt_to_equity 4.08 vs peer median 3.81",   # comparison
    "FUND: dividend_sustainability strong",            # enum value, NO digit
    "RSRCH: overall_moat_durability none",             # enum value, NO digit
    "RSRCH: revenue mix not disclosed",                # qualitative absence claim
])
def test_evidence_quoting_a_value_is_accepted(text):
    """A digit is not required: an enum value counts, which is why the metric
    'contains a digit' understated compliance at 78% when it was actually 100%."""
    assert _evidence_carries_value(text)


def test_retry_message_states_the_exact_shortfall():
    """_auto_trim can salvage an overshoot but nothing can expand a short field, so
    the only mechanical help is telling the model exactly how much it needs."""
    from agents.base import BaseRunner
    msg = BaseRunner._build_retry_message(
        BaseRunner.__new__(BaseRunner), "u", {"narrative": "x"},
        ["narrative: too short (696 chars, min 720 = ~120 words)"], 1)
    assert "24 characters short" in msg
    assert "at least 720" in msg


# ---------- Risk Advisor: metric tokens are a valid citation ----------
# The prompt authorises them twice (output schema + its own validation rules), and
# strongest_signal is "the dominant downside scenario OR SANITY FLAG" -- a sanity
# flag about BETA/VOL/DD consistency has no Pass 1 agent to cite. The old check
# rejected the prompt's own worked examples and caused a terminal failure live,
# which drops the field from the CIO payload entirely.

@pytest.mark.parametrize("signal", [
    "BETA: 0.79 - below-market systematic risk",      # the prompt's example format
    "VOL: 15.4% annualized, DD 3yr: -14.5%",
    "BETA+DD: low beta with shallow drawdowns",        # documented multi-source form
    "LIQ vs implied float looks inconsistent",         # CoT step 6 sanity flag
    "FUND: payout ratio 112%",                         # Pass 1 ID still accepted
])
def test_risk_advisor_accepts_metric_token_citations(signal):
    from agents.validators.pass2 import validate_risk_advisor_stage_a
    _, errors = validate_risk_advisor_stage_a({"strongest_signal": signal})
    assert not [e for e in errors if "strongest_signal" in e]


@pytest.mark.parametrize("signal", [
    "The stock looks risky overall",                   # no citation at all
    "Added volatile names to the book",                # DD in ADDED, VOL in volatile
])
def test_risk_advisor_still_rejects_uncited_signals(signal):
    """Word-boundary matched -- substring hits must not count as citations."""
    from agents.validators.pass2 import validate_risk_advisor_stage_a
    _, errors = validate_risk_advisor_stage_a({"strongest_signal": signal})
    assert [e for e in errors if "strongest_signal" in e]


class TestTaxStrategistCitationEnforcement:
    """Rules 7, 9, 10, 12, 13, 14 -- declared by the prompt, unenforced until 2026-09-01.

    Every check below is written to fail against the PREVIOUS validator, which
    inspected field presence, two array counts, narrative length and the tax_profile
    enums only. A validator that checks little still returns PASS, which is why the
    gap survived: six live runs "passed" while nothing verified a single citation.
    """

    def _err(self, out, account_type=None) -> list[str]:
        passed, errors = validate_tax_strategist(out, account_type=account_type)
        assert not passed, "expected this mutation to be rejected"
        return errors

    # --- rule 7: strongest_signal -------------------------------------------------
    def test_uncited_strongest_signal_fails(self):
        out = _valid_tax_output()
        out["strongest_signal"] = "The tax treatment here is broadly favourable for this investor."
        assert any("strongest_signal" in e for e in self._err(out))

    def test_strongest_signal_accepts_a_pass1_id(self):
        out = _valid_tax_output()
        out["strongest_signal"] = "FUND: a 22% payout ratio keeps the withholding drag secondary."
        passed, errors = validate_tax_strategist(out)
        assert passed, errors

    def test_narrative_without_ref_fails(self):
        out = _valid_tax_output()
        out["narrative"] = out["narrative"].replace("REF/TFSA", "guidance")
        assert any("REF" in e for e in self._err(out))

    def test_narrative_needs_three_distinct_metric_tokens(self):
        out = _valid_tax_output()
        for tok in ("DOM", "LIST", "DIVID", "WHT", "ELIG", "CGAIN", "ROOM"):
            out["narrative"] = out["narrative"].replace(tok, "x")
        out["narrative"] += " REF and FUND and MACRO remain cited here."
        assert any("tax-metric tokens" in e for e in self._err(out))

    def test_metric_tokens_are_word_boundary_matched(self):
        """LISTED must not satisfy LIST, nor LOSSES satisfy LOSS.

        Without \b the three-token requirement is trivially met by ordinary prose,
        which would make the check look enforced while enforcing nothing.
        """
        out = _valid_tax_output()
        for tok in ("DOM", "LIST", "DIVID", "WHT", "ELIG", "CGAIN", "ROOM"):
            out["narrative"] = out["narrative"].replace(tok, "x")
        out["narrative"] += " LISTED on an exchange, LOSSES carried forward, ROOMS available. REF FUND MACRO."
        assert any("tax-metric tokens" in e for e in self._err(out))

    # --- rule 12: key_tax_risks ---------------------------------------------------
    def test_missing_key_tax_risks_fails(self):
        out = _valid_tax_output()
        del out["tax_profile"]["key_tax_risks"]
        assert any("key_tax_risks" in e for e in self._err(out))

    def test_key_tax_risk_bad_severity_fails(self):
        out = _valid_tax_output()
        out["tax_profile"]["key_tax_risks"][0]["severity"] = "critical"
        assert any("severity" in e for e in self._err(out))

    def test_key_tax_risk_uncited_evidence_fails(self):
        out = _valid_tax_output()
        out["tax_profile"]["key_tax_risks"][0]["evidence"] = "this is generally considered risky"
        assert any("key_tax_risks[0].evidence" in e for e in self._err(out))

    # --- rule 13: tax_optimization_actions ----------------------------------------
    def test_action_without_citation_fails(self):
        out = _valid_tax_output()
        out["tax_profile"]["tax_optimization_actions"] = [
            {"action": "Move the position to an RRSP.", "justification": "It is usually better there."}
        ]
        assert any("justification" in e for e in self._err(out))

    def test_action_with_citation_passes(self):
        out = _valid_tax_output()
        out["tax_profile"]["tax_optimization_actions"] = [
            {"action": "Move the position to an RRSP.",
             "justification": "WHT: the 15% withholding is exempt in an RRSP under REF treaty Art. XVIII."}
        ]
        passed, errors = validate_tax_strategist(out)
        assert passed, errors

    # --- rule 14: loss harvesting is account-dependent ----------------------------
    def test_loss_harvesting_in_registered_account_fails(self):
        """Not bookkeeping: a capital loss is not claimable in a TFSA or RRSP, so
        suggesting harvesting there is wrong advice, not merely unhelpful."""
        for acct in ("tfsa", "rrsp"):
            out = _valid_tax_output()
            out["tax_profile"]["loss_harvesting_opportunity"] = {
                "superficial_loss_window_safe": True, "estimated_benefit_pct": 1.2,
            }
            assert any("loss_harvesting_opportunity" in e for e in self._err(out, account_type=acct))

    def test_loss_harvesting_allowed_in_trading_account(self):
        out = _valid_tax_output()
        out["tax_profile"]["loss_harvesting_opportunity"] = {
            "superficial_loss_window_safe": True, "estimated_benefit_pct": 1.2,
        }
        passed, errors = validate_tax_strategist(out, account_type="trading")
        assert passed, errors

    def test_rule_14_is_skipped_when_account_type_is_not_supplied(self):
        """Backward compatibility: existing callers pass one argument."""
        out = _valid_tax_output()
        out["tax_profile"]["loss_harvesting_opportunity"] = {"superficial_loss_window_safe": True}
        passed, errors = validate_tax_strategist(out)
        assert passed, errors

    def test_soft_errors_are_only_narrative_citation_breadth(self):
        """Anything that makes the output unusable or unsafe must stay hard."""
        from agents.validators.pass2 import split_soft_errors
        hard, soft = split_soft_errors([
            "narrative: needs >=3 distinct tax-metric tokens, got ['WHT']",
            "narrative: needs >=1 REF citation",
            "tax_profile.account_fit_score: must be one of ...",
            "tax_profile.loss_harvesting_opportunity: must be null in a TFSA -- ...",
            "narrative: too short (1135 chars, min 1200)",
        ])
        assert len(soft) == 2
        assert len(hard) == 3
        assert any("loss_harvesting" in e for e in hard), "safety rule must stay hard"
        assert any("too short" in e for e in hard), "length is not a citation-breadth rule"

class TestTaxStrategistMargToken:
    """MARG is advertised by the prompt (rule 1) and rendered in the pre-computed block, so the
    validator must accept it as a citation token like the other metric tokens."""

    def test_strongest_signal_may_cite_marg(self):
        out = _valid_tax_output()
        out["strongest_signal"] = "MARG: a 29.65% marginal rate makes the RRSP deduction worth more here."
        passed, errors = validate_tax_strategist(out)
        assert passed, errors

    def test_marg_counts_toward_the_three_token_narrative_floor(self):
        out = _valid_tax_output()
        for tok in ("DOM", "LIST", "ELIG", "CGAIN", "ROOM"):
            out["narrative"] = out["narrative"].replace(tok, "x")
        # DIVID and WHT remain; without MARG that is two tokens and the floor fails.
        assert any("tax-metric tokens" in e for e in validate_tax_strategist(out)[1])
        out["narrative"] += " MARG: the 29.65% marginal rate sets the value of the deduction."
        passed, errors = validate_tax_strategist(out)
        assert passed, errors

    def test_marg_is_word_boundary_matched(self):
        out = _valid_tax_output()
        for tok in ("DOM", "LIST", "ELIG", "CGAIN", "ROOM"):
            out["narrative"] = out["narrative"].replace(tok, "x")
        out["narrative"] += " The MARGINAL rate is high."
        assert any("tax-metric tokens" in e for e in validate_tax_strategist(out)[1])


def test_core_argument_evidence_must_quote_a_value_not_only_field_names():
    """Bull emitted "FUND: revenue_growth_yoy, roe" with the figures in the payload. This was checked on key_factors until those were
    cut from Pass 2 (2026-10-07); core_arguments carries the evidence now, for both advocates."""
    from agents.validators.pass2 import validate_bear_advocate, validate_bull_advocate
    for validate in (validate_bull_advocate, validate_bear_advocate):
        out = {"structured_data": {"core_arguments": [{"argument": "a", "strength": "primary", "evidence": "FUND: debt_to_equity, roe"}]}}
        _, errors = validate(out)
        assert any("core_arguments[0].evidence" in e and "without values" in e for e in errors)
        out["structured_data"]["core_arguments"][0]["evidence"] = "FUND: roe 15.7%"
        _, errors = validate(out)
        assert not any("core_arguments[0].evidence" in e for e in errors)


def test_the_pass2_agents_no_longer_ask_for_key_factors():
    from agents.validators.pass2 import validate_bear_advocate, validate_bull_advocate
    for validate in (validate_bull_advocate, validate_bear_advocate):
        _, errors = validate({})
        assert not any("key_factors" in e for e in errors)
