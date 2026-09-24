"""pytest tests for the CIO validator, split Stage A / Stage B (ClickUp 86bbuhk82).

Rewritten from a single `_valid_cio_output()` fixture testing one monolithic
`validate_cio()` -- that schema never matched either real stage (it required
Stage-B-only fields on Stage A output, and `bull_case_assessment`/
`bear_case_assessment` as bare strings, when the real schema has always
declared them as objects with `verdict`/`engagement`/`weight_applied`/
`weight_reason`). Fixtures below are built directly from
`backend/prompts/cio/v1_stage_a.txt` / `v1_stage_b.txt`.
"""
import pytest

from agents.validators.cio import (
    split_soft_errors,
    validate_cio_handles_missing_fundamental,
    validate_cio_stage_a,
    validate_cio_stage_b,
)


def _valid_stage_a_output() -> dict:
    return {
        "stock_outlook": "somewhat_bullish",
        "confidence": 62,
        "expected_return_tier": "outperform",
        "thesis_summary": "GlobalTech offers moderate upside driven by AI copilot adoption and operating leverage, balanced against elevated valuation and competitive headwinds from SAP.",
        "what_would_change_my_mind": "If RSRCH data showed a SAP customer win specifically from GlobalTech's installed base, or if FUND reported Q2 margin below 23% (below guidance), I would move to bearish.",
        "bull_case_assessment": {
            "verdict": "accepted",
            "engagement": "The bull's AI copilot adoption argument holds weight given RSRCH's confirmed 2,400 activations and FUND's stable 4.8% FCF yield.",
            "weight_applied": "high",
            "weight_reason": "High data quality from RSRCH and FUND, and the argument aligns with the company's financials.",
        },
        "bear_case_assessment": {
            "verdict": "partially_accepted",
            "engagement": "The bear's SAP competitive concern is credible, but RSRCH shows no customer losses yet and NRR remains stable at 128%.",
            "weight_applied": "medium",
            "weight_reason": "Moderate confidence -- the risk is real but not yet materialized in the data.",
        },
        "risk_reward_consumed_as": "favorable",
        "tail_risk_cross_check": "aligned",
        "tail_risk_note": "",
        "key_decision_factors": [
            {"factor": "AI copilot adoption rate", "direction": "bullish", "source": "pass1", "evidence": "FUND: revenue +15.7% YoY"},
            {"factor": "SAP competitive threat", "direction": "bearish", "source": "bear", "evidence": "BEAR: P/B 45% above peers"},
            {"factor": "Operating leverage inflection H2 2026", "direction": "bullish", "source": "pass1", "evidence": "FUND: revenue +15.7% YoY"},
            {"factor": "Elevated rates compressing multiples", "direction": "bearish", "source": "bear", "evidence": "BEAR: P/B 45% above peers"},
        ],
        "accuracy_influence_note": "All Pass 1 agents had reliability >=70 (all OK). Analysis confidence is high. No reliability adjustments applied.",
        "dissent_note": "Bull and Bear had a disagreement score of 52 (split_decision). Bull emphasizes growth quality; Bear emphasizes valuation risk. CIO resolved toward somewhat_bullish given the lack of current evidence for moat erosion.",
        "parallel_predictions": {
            "1_month": {"direction": "neutral", "confidence": 45},
            "3_month": {"direction": "bullish", "confidence": 58},
        },
    }


_STAGE_B_NARRATIVE = (
    "After weighing the comprehensive Pass 1 research and Pass 2 advocacy positions, the CIO concludes with a somewhat_bullish outlook at 62/100 confidence. "
    "The bull case (Bull confidence 72/100) rests on three verifiable thesis anchors: AI copilot adoption exceeding internal targets (RSRCH), FCF yield of 4.8% providing a margin of safety (FUND), and management credibility via insider buying (RSRCH). "
    "The bear case (Bear confidence 55/100) is primarily a valuation and competitive risk argument: at 28.5x PE versus the sector's 26x median, the stock prices in a successful AI cycle -- any disappointment reprices sharply downward. "
    "The SAP AI roadmap (RSRCH) is the most significant bear argument that the CIO takes seriously, but the current evidence (no customer losses yet, NRR stable at 128%) does not yet validate the threat as acute. "
    "TECH indicators show a sideways-to-up trend on the weekly with RSI at 51 -- not a strong setup but not deteriorating either. "
    "SENT reflects net positive sentiment with insider buying outweighing the VP-level sells. "
    "MACRO headwinds from 4.5% rates are real and cap the multiple upside. "
    "The Risk Advisor's risk_reward_ratio of 'neutral' (RISK) and Tax Strategist's 'good' account_fit_score for TFSA (TAX) support a marketweight position. "
    "The disagreement score of 52 (split_decision) means this is a genuinely dual-signal stock -- the confidence cap of 65 is appropriate. "
    "Key uncertainty: AI copilot attach rate sustainability beyond the initial 2,400 activations. Further monitoring required."
)


def _valid_stage_b_output() -> dict:
    return {
        "position_sizing_recommendation": "3-5%",
        "expected_return_tier": "outperform",  # unchanged from Stage A's tier -- allowed
        "synthesis_narrative": _STAGE_B_NARRATIVE,
        "tax_summary": {
            "tax_efficiency_consumed_as": "favorable",
            "tax_impact_on_recommendation": "neutral",
            "cross_account_note": "No cross-account move indicated.",
        },
        "risk_profile_summary": {
            "position_size_source": "risk_advisor_adopted",
            "sizing_justification": "Adopted the Risk Advisor's 3-5% band unchanged.",
        },
    }


class TestCIOStageA:
    def test_valid_output_passes(self):
        out = _valid_stage_a_output()
        passed, errors = validate_cio_stage_a(out, "split_decision")
        assert passed, f"Valid CIO Stage A output should pass: {errors}"

    def test_invalid_stock_outlook_fails(self):
        out = _valid_stage_a_output()
        out["stock_outlook"] = "buy"  # invalid
        passed, errors = validate_cio_stage_a(out, "split_decision")
        assert not passed
        assert any("stock_outlook" in e for e in errors)

    def test_confidence_out_of_range_fails(self):
        out = _valid_stage_a_output()
        out["confidence"] = 110
        passed, errors = validate_cio_stage_a(out, "split_decision")
        assert not passed

    def test_split_decision_confidence_cap(self):
        out = _valid_stage_a_output()
        out["confidence"] = 70  # over 65 cap for split_decision
        passed, errors = validate_cio_stage_a(out, "split_decision")
        assert not passed
        assert any("split_decision" in e or "65" in e for e in errors)

    def test_split_decision_within_cap_passes(self):
        out = _valid_stage_a_output()
        out["confidence"] = 65  # exactly at cap
        passed, errors = validate_cio_stage_a(out, "split_decision")
        assert passed, f"confidence=65 should pass for split_decision: {errors}"

    def test_high_conflict_full_bullish_fails(self):
        out = _valid_stage_a_output()
        out["stock_outlook"] = "bullish"  # full bullish not allowed in high_conflict
        passed, errors = validate_cio_stage_a(out, "high_conflict")
        assert not passed
        assert any("high_conflict" in e for e in errors)

    def test_high_conflict_somewhat_bullish_passes(self):
        out = _valid_stage_a_output()
        out["stock_outlook"] = "somewhat_bullish"
        out["confidence"] = 55
        passed, errors = validate_cio_stage_a(out, "high_conflict")
        assert passed, f"somewhat_bullish should pass for high_conflict: {errors}"

    def test_high_conflict_neutral_passes(self):
        out = _valid_stage_a_output()
        out["stock_outlook"] = "neutral"
        out["confidence"] = 50
        passed, errors = validate_cio_stage_a(out, "high_conflict")
        assert passed, f"neutral should pass for high_conflict: {errors}"

    def test_invalid_expected_return_tier_fails(self):
        out = _valid_stage_a_output()
        out["expected_return_tier"] = "buy"  # invalid
        passed, errors = validate_cio_stage_a(out, "split_decision")
        assert not passed

    def test_too_few_key_decision_factors_fails(self):
        out = _valid_stage_a_output()
        out["key_decision_factors"] = out["key_decision_factors"][:2]  # only 2
        passed, errors = validate_cio_stage_a(out, "split_decision")
        assert not passed
        assert any("key_decision_factors" in e for e in errors)

    def test_too_many_key_decision_factors_fails(self):
        out = _valid_stage_a_output()
        extra = {"factor": "Extra factor", "direction": "neutral", "source": "risk", "evidence": "RISK: beta 0.9 vs benchmark"}
        out["key_decision_factors"] = out["key_decision_factors"] + [extra, extra, extra]  # 7 total
        passed, errors = validate_cio_stage_a(out, "split_decision")
        assert not passed

    def test_invalid_kdf_direction_fails(self):
        out = _valid_stage_a_output()
        # "positive" is the RUNNER's old vocabulary; the prompt specifies
        # bullish|bearish|neutral. This also guards against reverting the validator.
        out["key_decision_factors"][0]["direction"] = "positive"  # invalid per the prompt
        passed, errors = validate_cio_stage_a(out, "split_decision")
        assert not passed

    def test_invalid_kdf_source_fails(self):
        """"tax" was in the old validator's source enum, but Tax Strategist has no input
        into Stage A at all -- confirmed via the real Stage A schema (bull|bear|risk|pass1,
        4 values, no "tax")."""
        out = _valid_stage_a_output()
        out["key_decision_factors"][0]["source"] = "tax"
        passed, errors = validate_cio_stage_a(out, "split_decision")
        assert not passed
        assert any("source" in e for e in errors)

    def test_missing_parallel_predictions_fails(self):
        out = _valid_stage_a_output()
        del out["parallel_predictions"]["3_month"]
        passed, errors = validate_cio_stage_a(out, "split_decision")
        assert not passed

    def test_invalid_prediction_direction_fails(self):
        out = _valid_stage_a_output()
        out["parallel_predictions"]["1_month"]["direction"] = "flat"  # invalid
        passed, errors = validate_cio_stage_a(out, "split_decision")
        assert not passed

    def test_consensus_category_no_confidence_cap(self):
        out = _valid_stage_a_output()
        out["confidence"] = 85  # allowed for consensus category
        out["stock_outlook"] = "bullish"  # also allowed
        passed, errors = validate_cio_stage_a(out, "consensus")
        assert passed, f"High confidence should be allowed for consensus: {errors}"

    def test_engagement_too_short_fails(self):
        """Stage 3 decision text requires a SUBSTANTIVE engagement (>=40 chars) --
        never enforced before this rewrite."""
        out = _valid_stage_a_output()
        out["bull_case_assessment"]["engagement"] = "Agreed."
        passed, errors = validate_cio_stage_a(out, "split_decision")
        assert not passed
        assert any("engagement" in e for e in errors)

    def test_verdict_and_weight_applied_enforced_via_sweep(self):
        out = _valid_stage_a_output()
        out["bull_case_assessment"]["verdict"] = "sort_of"
        out["bear_case_assessment"]["weight_applied"] = "super_high"
        passed, errors = validate_cio_stage_a(out, "split_decision")
        assert not passed
        assert any("verdict" in e for e in errors)
        assert any("weight_applied" in e for e in errors)


class TestCIOStageB:
    def test_valid_output_passes(self):
        out = _valid_stage_b_output()
        passed, errors = validate_cio_stage_b(out)
        assert passed, f"Valid CIO Stage B output should pass: {errors}"

    def test_position_sizing_accepts_a_percentage_band(self):
        """Verified at BOTH ends: Risk emits '3-5%' live, its prompt mandates "X-Y%",
        the CIO prompt says "X-Y%", and the DB column is String(50)."""
        out = _valid_stage_b_output()
        for good in ("3-5%", "1-2%", "5-8%", "0.5-1.5%"):
            out["position_sizing_recommendation"] = good
            passed, errors = validate_cio_stage_b(out)
            assert passed, f"{good!r} should be valid: {errors}"

    def test_position_sizing_rejects_the_old_enum(self):
        out = _valid_stage_b_output()
        out["position_sizing_recommendation"] = "marketweight"
        passed, errors = validate_cio_stage_b(out)
        assert not passed and any("percentage band" in e for e in errors)

    def test_narrative_too_short_fails(self):
        out = _valid_stage_b_output()
        out["synthesis_narrative"] = "Short CIO summary."
        passed, errors = validate_cio_stage_b(out)
        assert not passed
        assert any("synthesis_narrative" in e for e in errors)

    def test_tier_may_stay_the_same(self):
        out = _valid_stage_b_output()
        passed, errors = validate_cio_stage_b(out, stage_a_expected_return_tier="outperform")
        assert passed, errors

    def test_tier_may_move_worse(self):
        out = _valid_stage_b_output()
        out["expected_return_tier"] = "market_perform"
        passed, errors = validate_cio_stage_b(out, stage_a_expected_return_tier="outperform")
        assert passed, errors

    def test_tier_may_not_move_better(self):
        """Rule 3: tax drag is a cost that can only hold the tier or make it worse --
        never a reason to improve on the general read. Never checked before this rewrite,
        since the old single-call validator had no Stage A value to compare against."""
        out = _valid_stage_b_output()
        out["expected_return_tier"] = "strong_outperform"
        passed, errors = validate_cio_stage_b(out, stage_a_expected_return_tier="outperform")
        assert not passed
        assert any("BETTER tier" in e for e in errors)

    def test_tax_efficiency_must_match_its_source(self):
        """Prompt rule 11 and the orchestration spec both say "Mismatch -> retry"."""
        out = _valid_stage_b_output()
        out["tax_summary"]["tax_efficiency_consumed_as"] = "favorable"
        passed, errors = validate_cio_stage_b(out, tax_efficiency_source="unfavorable")
        assert not passed and any("must match" in e for e in errors)

        out2 = _valid_stage_b_output()
        out2["tax_summary"]["tax_efficiency_consumed_as"] = "favorable"
        passed, errors = validate_cio_stage_b(out2, tax_efficiency_source="favorable")
        assert passed, errors

    def test_previously_unchecked_enums_are_now_validated(self):
        out = _valid_stage_b_output()
        out["risk_profile_summary"]["position_size_source"] = "made_it_up"
        passed, errors = validate_cio_stage_b(out)
        assert not passed
        assert any("position_size_source" in e for e in errors)

        out2 = _valid_stage_b_output()
        out2["tax_summary"]["tax_impact_on_recommendation"] = "helps_a_lot"
        passed, errors = validate_cio_stage_b(out2)
        assert not passed
        assert any("tax_impact_on_recommendation" in e for e in errors)


class TestCIORiskRewardCrossCheck:
    """Stage A's own cross-check -- risk_reward_consumed_as is a TOP-LEVEL field in the
    real Stage A schema, not nested under risk_profile_summary (which Stage A doesn't
    have at all; the old validator's nesting never matched either real stage)."""

    def test_must_match_risk_advisors_ratio(self):
        out = _valid_stage_a_output()
        out["risk_reward_consumed_as"] = "favorable"
        passed, errors = validate_cio_stage_a(out, "split_decision", risk_reward_source="unfavorable")
        assert not passed and any("must match" in e for e in errors)

    def test_matching_ratio_passes(self):
        out = _valid_stage_a_output()
        out["risk_reward_consumed_as"] = "favorable"
        passed, errors = validate_cio_stage_a(out, "split_decision", risk_reward_source="favorable")
        assert passed, errors


class TestCIOTailRiskNumericCrossCheck:
    """The -20% mapping (86bbuhk82 item 2) -- confirmed live end-to-end (a real CIO call got
    the direction backwards on attempt 1 and self-corrected on retry using this exact check's
    error message), but that live run is non-deterministic and doesn't run in CI. These are
    the deterministic unit tests for the same four branches."""

    def test_elevated_with_mild_risk_numbers_expects_mild_classification(self):
        out = _valid_stage_a_output()
        out["tail_risk_cross_check"] = "aligned"  # wrong on purpose
        passed, errors = validate_cio_stage_a(
            out, "split_decision",
            bear_tail_risk_level="elevated", risk_worst_estimated_impact_pct=-10.0,
        )
        assert not passed
        assert any("bear_elevated_risk_advisor_mild" in e for e in errors)

        out["tail_risk_cross_check"] = "bear_elevated_risk_advisor_mild"
        passed, errors = validate_cio_stage_a(
            out, "split_decision",
            bear_tail_risk_level="elevated", risk_worst_estimated_impact_pct=-10.0,
        )
        assert passed, errors

    def test_negligible_with_severe_risk_numbers_expects_elevated_classification(self):
        """The exact scenario a real live run hit: Bear said negligible, Risk Advisor's own
        worst downside scenario was -22% (worse than -20%) -- the CIO got it backwards on
        attempt 1 (said bear_elevated_risk_advisor_mild) and this check caught it."""
        out = _valid_stage_a_output()
        out["tail_risk_cross_check"] = "bear_elevated_risk_advisor_mild"  # the real live mistake
        passed, errors = validate_cio_stage_a(
            out, "split_decision",
            bear_tail_risk_level="negligible", risk_worst_estimated_impact_pct=-22.0,
        )
        assert not passed
        assert any("bear_mild_risk_advisor_elevated" in e for e in errors)

        out["tail_risk_cross_check"] = "bear_mild_risk_advisor_elevated"
        passed, errors = validate_cio_stage_a(
            out, "split_decision",
            bear_tail_risk_level="negligible", risk_worst_estimated_impact_pct=-22.0,
        )
        assert passed, errors

    def test_moderate_always_expects_aligned_regardless_of_risk_numbers(self):
        """Inherited verbatim from the audit's own E82 mapping rule -- only elevated/negligible
        trigger the numeric comparison; moderate always resolves to aligned by design, even
        against an extreme risk number. Not a gap introduced here."""
        out = _valid_stage_a_output()
        out["tail_risk_cross_check"] = "aligned"
        passed, errors = validate_cio_stage_a(
            out, "split_decision",
            bear_tail_risk_level="moderate", risk_worst_estimated_impact_pct=-80.0,
        )
        assert passed, errors

    def test_exact_threshold_boundary_resolves_to_aligned(self):
        """R exactly at -20% doesn't satisfy either strict inequality (>/<), so it falls to
        the default -- deliberate, not an off-by-one: -20% was chosen to sit in a real gap in
        the historical data specifically so an exact-boundary hit is rare in practice."""
        out = _valid_stage_a_output()
        out["tail_risk_cross_check"] = "aligned"
        passed, errors = validate_cio_stage_a(
            out, "split_decision",
            bear_tail_risk_level="negligible", risk_worst_estimated_impact_pct=-20.0,
        )
        assert passed, errors

    def test_mapping_not_checked_when_real_inputs_not_supplied(self):
        """Backward compatible: callers that don't have Bear/Risk's real values (or tests
        exercising other behavior) get no numeric cross-check at all, not a crash."""
        out = _valid_stage_a_output()
        out["tail_risk_cross_check"] = "bear_elevated_risk_advisor_mild"  # would fail the mapping if checked
        passed, errors = validate_cio_stage_a(out, "split_decision")
        assert passed, errors


class TestCIOMissingFundamental:
    def _valid_no_fund(self) -> dict:
        """validate_cio_handles_missing_fundamental reads synthesis_narrative, tax_summary,
        and key_decision_factors -- a merged view spanning both stages, matching how the
        orchestrator will actually assemble the two real calls' output for this check."""
        out = {**_valid_stage_a_output(), **_valid_stage_b_output()}
        out["synthesis_narrative"] = (
            "BioxRx Therapeutics is a pre-revenue clinical-stage biotech. "
            "The Fundamental Analyst was unable to produce a meaningful analysis due to insufficient financial data — "
            "there are no revenue, PE, margin, or FCF metrics available for a pre-revenue company. "
            "The CIO synthesis therefore relies primarily on RSRCH (pipeline assessment), "
            "TECH (technical setup), SENT (insider buying signal), and MACRO (biotech funding environment). "
            "The investment thesis is purely pipeline-driven: BXR-401 CARDINAL Phase 3 data expected Q4 2026 is a binary event. "
            "Insider buying by CMO and CEO at $7.80-8.10 is a positive signal (SENT). "
            "TECH confirms the stock is in a strong downtrend (oversold RSI 28.8) which may represent a capitulation setup ahead of the data readout. "
            "MACRO notes that higher rates increase cost of capital pressure for clinical-stage biotechs (unfavorable). "
            "Without fundamental valuation data, position sizing must be conservative — speculative sizing of 1-2% maximum in TFSA given the binary clinical outcome risk and permanent room loss from any capital loss."
        )
        out["stock_outlook"] = "neutral"
        out["confidence"] = 45
        out["thesis_summary"] = "Pre-revenue biotech with binary Phase 3 outcome; no fundamental data available. Pipeline quality and insider conviction support a speculative TFSA position."
        return out

    def test_missing_fund_acknowledged_passes(self):
        out = self._valid_no_fund()
        passed, errors = validate_cio_handles_missing_fundamental(out)
        assert passed, f"CIO acknowledging missing FUND data should pass: {errors}"

    def test_fabricating_pe_fails(self):
        out = self._valid_no_fund()
        out["synthesis_narrative"] += " The P/E ratio of 18x is attractive relative to peers."
        passed, errors = validate_cio_handles_missing_fundamental(out)
        assert not passed
        assert any("p/e" in e.lower() for e in errors)

    def test_no_acknowledgment_fails(self):
        out = {**_valid_stage_a_output(), **_valid_stage_b_output()}
        # Generic CIO output without acknowledging missing FUND
        out["synthesis_narrative"] = (
            "The stock looks good based on various technical and sentiment indicators. "
            "The bull case is strong and the bear case is weak. We recommend buying. " * 20
        )
        passed, errors = validate_cio_handles_missing_fundamental(out)
        assert not passed


class TestCIOTailRiskAndCitation:
    """H-b / H-c, added 2026-09-01.

    Both guard rules the CIO prompt states and the validator did not enforce:
    `tail_risk_cross_check` was emitted as `aligned` on absent inputs (§162.1), and the
    wwcmm check accepted the word "price" as "specific Pass 1 agent findings" (§158.3).

    `tail_risk_cross_check` is a TOP-LEVEL Stage A field (confirmed via the real
    v1_stage_a.txt schema), not nested under risk_profile_summary as the pre-split
    validator assumed.
    """

    def test_tail_risk_must_be_insufficient_when_inputs_absent(self):
        out = _valid_stage_a_output()
        out["tail_risk_cross_check"] = "aligned"
        passed, errors = validate_cio_stage_a(out, "split_decision", tail_inputs_present=False)
        assert not passed
        assert any("insufficient_data" in e for e in errors)

    def test_tail_risk_insufficient_data_accepted_when_inputs_absent(self):
        out = _valid_stage_a_output()
        out["tail_risk_cross_check"] = "insufficient_data"
        passed, errors = validate_cio_stage_a(out, "split_decision", tail_inputs_present=False)
        assert passed, errors

    def test_tail_risk_comparison_allowed_when_inputs_present(self):
        out = _valid_stage_a_output()
        out["tail_risk_cross_check"] = "aligned"
        passed, errors = validate_cio_stage_a(out, "split_decision", tail_inputs_present=True)
        assert passed, errors

    def test_wwcmm_generic_term_no_longer_passes(self):
        """"price" and "growth" used to satisfy a rule demanding Pass 1 agent findings."""
        out = _valid_stage_a_output()
        out["what_would_change_my_mind"] = "A sustained fall in the share price or slower growth."
        passed, errors = validate_cio_stage_a(out, "split_decision")
        assert not passed
        assert any("Pass 1 agent ID" in e for e in errors)

    def test_wwcmm_agent_id_passes(self):
        out = _valid_stage_a_output()
        out["what_would_change_my_mind"] = "FUND: two quarters of revenue growth below 10% would flip this."
        passed, errors = validate_cio_stage_a(out, "split_decision")
        assert passed, errors

    def test_wwcmm_failure_is_SOFT(self):
        """It must apply retry pressure without costing the agent on a final attempt."""
        hard, soft = split_soft_errors([
            "what_would_change_my_mind: must cite a Pass 1 agent ID (RSRCH/FUND/TECH/SENT/MACRO), not merely a generic financial term",
            "stock_outlook: must be one of ...",
        ])
        assert len(soft) == 1 and len(hard) == 1

    def test_narrative_undershoot_is_SOFT(self):
        """The prompt states BOTH "250-400 words" and "~1,500-2,400 chars"; 250 words is
        ~1,375 chars, so the two disagree and the floor was unwinnable (terminal failures
        at 1499/1452/1359). Kept, not lowered -- but soft, so it cannot cost the agent."""
        hard, soft = split_soft_errors([
            "synthesis_narrative: too short (1452 chars, min 1500)",
            "stock_outlook: must be one of ...",
        ])
        assert len(soft) == 1 and len(hard) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
