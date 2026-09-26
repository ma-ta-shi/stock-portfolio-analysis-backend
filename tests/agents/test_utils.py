"""Tests for agents/utils.py -- production port of simulation/utils.py (86bbuhjup).

Consolidates the harness's own test coverage for this logic (previously split across
simulation/tests/test_pass2_validators.py's TestDisagreementScore/TestOutlookDistance
and simulation/tests/test_gates.py, since production keeps all of it in one module
matching the harness's own utils.py structure). gate1_check's tests specifically lock
in the 2026-09-16 completion-only contract, not the stale pre-decision count threshold
the harness's own copy still had until this same port caught and fixed it.
"""
import json

import pytest

from agents.utils import (
    CONFIDENCE_STATUS_LABELS,
    PASS1_AGENT_IDS,
    RenderedField,
    agent_completed,
    build_pass1_reliability_warnings,
    char_count,
    compute_disagreement_score,
    compute_outlook_distance,
    extract_numeric_tokens,
    gate1_check,
    gate2_check,
    parse_json_response,
    render_data_coverage_line,
    render_data_warnings,
    researcher_thesis_archetype,
    to_data_coverage,
    truncate_to_tokens,
    word_count,
)


class TestRenderedField:
    """86bbwachy Phase 4 -- the shared text+presence bundle every in-scope
    agent's own field-rendering helpers return."""

    def test_bundles_text_and_present_together(self):
        field = RenderedField(text="Yield: 5.23%", present=True)
        assert field.text == "Yield: 5.23%"
        assert field.present is True

    def test_absent_field_still_carries_its_rendered_na_text(self):
        """present=False doesn't mean text is empty -- the "N/A ..." wording
        still needs to reach the prompt; present is the separate,
        machine-readable fact alongside it."""
        field = RenderedField(text="N/A", present=False)
        assert field.text == "N/A"
        assert field.present is False


def _completed():
    return {"pass2_view": {"thesis_archetype": "secular_grower"}, "analysis_confidence": "high"}


class TestDisagreementScore:
    def test_opposite_equal_confidence_gives_split_decision(self):
        score, label = compute_disagreement_score(70, 70)
        assert label in ("split_decision", "mild_dissent"), f"Got {label} ({score})"

    def test_asymmetric_confidence_can_give_high_conflict(self):
        score, label = compute_disagreement_score(90, 30)
        assert label in ("split_decision", "high_conflict"), f"Got {label} ({score})"

    def test_zero_confidence_gives_consensus(self):
        score, label = compute_disagreement_score(0, 0)
        assert label == "consensus"

    def test_high_both_gives_split(self):
        score, label = compute_disagreement_score(75, 70)
        assert score >= 40, f"Expected split_decision range, got score={score}"


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

    def test_exactly_two_notches_is_not_high_divergence(self):
        # high_divergence is a STRICT >2 -- exactly 2 must not count.
        distance, high_divergence = compute_outlook_distance("bullish", "neutral")
        assert distance == 2
        assert high_divergence is False

    def test_three_notches_is_high_divergence(self):
        distance, high_divergence = compute_outlook_distance("bullish", "somewhat_bearish")
        assert distance == 3
        assert high_divergence is True

    def test_all_25_pairs_are_symmetric_and_bounded(self):
        for a in self.OUTLOOKS:
            for b in self.OUTLOOKS:
                d_ab, hd_ab = compute_outlook_distance(a, b)
                d_ba, hd_ba = compute_outlook_distance(b, a)
                assert d_ab == d_ba
                assert 0 <= d_ab <= 4
                assert hd_ab == (d_ab > 2)


class TestAgentCompleted:
    def test_none_not_completed(self):
        assert agent_completed(None) is False

    def test_empty_dict_not_completed(self):
        assert agent_completed({}) is False

    def test_non_dict_not_completed(self):
        assert agent_completed("not a dict") is False

    def test_all_none_values_not_completed(self):
        assert agent_completed({"analysis_confidence": None, "narrative": None}) is False

    def test_real_content_is_completed(self):
        assert agent_completed({"analysis_confidence": "high", "narrative": "text"}) is True

    def test_validation_failed_but_present_still_counts_completed(self):
        """SETTLED 2026-09-02 (E77): validation status is a warning, not a gate."""
        assert agent_completed({"malformed_field": "garbage but present"}) is True


class TestGate1CompletionOnly:
    """SETTLED 2026-09-16: no count threshold, no RSRCH/FUND requirement -- fails
    only when every single Pass 1 agent produced nothing real. Regression coverage
    for the staleness this exact port caught in the harness's own copy."""

    def test_all_empty_fails(self):
        pass1 = {"RSRCH": {}, "FUND": None, "TECH": {}, "SENT": None, "MACRO": {}}
        passed, reason = gate1_check(pass1)
        assert not passed
        assert "No Pass 1 agent" in reason

    def test_single_agent_completed_passes_no_count_threshold(self):
        pass1 = {"RSRCH": _completed(), "FUND": None, "TECH": {}, "SENT": None, "MACRO": {}}
        passed, reason = gate1_check(pass1)
        assert passed, reason

    def test_single_completed_agent_need_not_be_rsrch_or_fund(self):
        pass1 = {"RSRCH": None, "FUND": None, "TECH": _completed(), "SENT": None, "MACRO": None}
        passed, reason = gate1_check(pass1)
        assert passed, reason

    def test_all_completed_passes(self):
        pass1 = {aid: _completed() for aid in PASS1_AGENT_IDS}
        passed, reason = gate1_check(pass1)
        assert passed, reason


class TestGate2BothAdvocatesMandatory:
    def test_both_present_passes(self):
        passed, reason = gate2_check({"bull": _completed(), "bear": _completed()})
        assert passed, reason

    def test_bull_missing_fails(self):
        passed, reason = gate2_check({"bull": None, "bear": _completed()})
        assert not passed
        assert "Bull" in reason

    def test_bear_missing_fails(self):
        passed, reason = gate2_check({"bull": _completed(), "bear": None})
        assert not passed
        assert "Bear" in reason

    def test_tax_and_risk_absence_does_not_affect_gate2(self):
        passed, reason = gate2_check(
            {"bull": _completed(), "bear": _completed(), "tax": None, "risk": None}
        )
        assert passed, reason


class TestReliabilityWarnings:
    def test_all_high_returns_empty_string(self):
        confidence = {aid: "high" for aid in PASS1_AGENT_IDS}
        assert build_pass1_reliability_warnings(confidence) == ""

    def test_mixed_confidence_formats_all_agents(self):
        confidence = {"RSRCH": "high", "FUND": "medium", "TECH": "low", "SENT": "insufficient", "MACRO": "high"}
        result = build_pass1_reliability_warnings(confidence)
        assert "FUND: medium (caution)" in result
        assert "TECH: low (unreliable)" in result
        assert "SENT: insufficient (skipped)" in result

    def test_missing_agent_defaults_to_insufficient(self):
        result = build_pass1_reliability_warnings({})
        for aid in PASS1_AGENT_IDS:
            assert f"{aid}: insufficient (skipped)" in result

    def test_status_labels_cover_all_confidence_levels(self):
        assert set(CONFIDENCE_STATUS_LABELS) == {"high", "medium", "low", "insufficient"}


class TestRenderDataCoverageLine:
    """86bbummwp Tier 1a -- shared DATA COVERAGE line renderer for Fundamental,
    Technical, Sentiment, and Macro Economist (Stock Researcher keeps its own
    list-based implementation, not migrated here -- different input shape)."""

    GAP_SENTENCES = {
        "news_block": "no news articles available",
        "short_interest": "no short interest data available",
    }

    def test_all_present_returns_standard(self):
        result = render_data_coverage_line(
            {"news_block": True, "short_interest": True}, self.GAP_SENTENCES
        )
        assert result == "standard."

    def test_one_gap_renders_its_sentence(self):
        result = render_data_coverage_line(
            {"news_block": False, "short_interest": True}, self.GAP_SENTENCES
        )
        assert result == "no news articles available."

    def test_multiple_gaps_joined_with_semicolons(self):
        result = render_data_coverage_line(
            {"news_block": False, "short_interest": False}, self.GAP_SENTENCES
        )
        assert result == "no news articles available; no short interest data available."

    def test_key_absent_from_field_presence_is_not_applicable_not_a_gap(self):
        """The Macro Economist nuance: a key genuinely absent from
        field_presence (e.g. `statcan` for a US stock) means "not
        applicable," never a mentioned gap -- distinct from present-but-False."""
        result = render_data_coverage_line({"news_block": True}, self.GAP_SENTENCES)
        assert result == "standard."

    def test_key_present_but_false_is_a_real_gap_even_if_others_are_absent(self):
        result = render_data_coverage_line({"short_interest": False}, self.GAP_SENTENCES)
        assert result == "no short interest data available."

    def test_field_presence_key_with_no_matching_gap_sentence_is_ignored(self):
        """A field_presence entry not in gap_sentences at all (e.g. a caller
        passing extra keys never meant for this line) is silently skipped,
        not a KeyError."""
        result = render_data_coverage_line(
            {"news_block": True, "some_other_field": False}, self.GAP_SENTENCES
        )
        assert result == "standard."


class TestToDataCoverage:
    """86bbummwp Tier 2 -- D6's structured data_coverage flag, same
    two-argument shape and filtering as render_data_coverage_line() above so
    every agent can pass the exact same field_presence/gap_sentences pair to
    both."""

    GAP_SENTENCES = {
        "news_block": "no news articles available",
        "short_interest": "no short interest data available",
    }

    def test_all_present(self):
        result = to_data_coverage({"news_block": True, "short_interest": True}, self.GAP_SENTENCES)
        assert result == {"present": ["news_block", "short_interest"], "absent": []}

    def test_all_absent(self):
        result = to_data_coverage({"news_block": False, "short_interest": False}, self.GAP_SENTENCES)
        assert result == {"present": [], "absent": ["news_block", "short_interest"]}

    def test_mixed(self):
        result = to_data_coverage({"news_block": True, "short_interest": False}, self.GAP_SENTENCES)
        assert result == {"present": ["news_block"], "absent": ["short_interest"]}

    def test_key_absent_from_field_presence_is_excluded_entirely(self):
        """Same "not applicable" semantics as render_data_coverage_line() --
        a key genuinely absent from field_presence (e.g. Macro's statcan for
        a US stock) must not appear in either list."""
        result = to_data_coverage({"news_block": True}, self.GAP_SENTENCES)
        assert result == {"present": ["news_block"], "absent": []}

    def test_field_presence_key_with_no_matching_gap_sentence_is_ignored(self):
        result = to_data_coverage(
            {"news_block": True, "some_other_field": False}, self.GAP_SENTENCES
        )
        assert result == {"present": ["news_block"], "absent": []}

    def test_empty_field_presence_returns_empty_lists(self):
        assert to_data_coverage({}, self.GAP_SENTENCES) == {"present": [], "absent": []}


class TestRenderDataWarnings:
    """86bbummwp follow-on -- generalizes Technical Analyst's own former
    _data_warnings_line() (anomalies only) to also cover stale_data, found
    missing during the investigation that led to this fix."""

    def test_empty_when_both_empty(self):
        assert render_data_warnings([], []) == ""

    def test_anomalies_only_joined_with_semicolons(self):
        result = render_data_warnings(
            ["US yield curve is inverted", "VIX is in a high-volatility regime (35.0)"], []
        )
        assert result == "US yield curve is inverted; VIX is in a high-volatility regime (35.0)"

    def test_stale_data_only_rendered_as_one_segment(self):
        result = render_data_warnings([], ["rate", "cpi"])
        assert result == "stale data: rate, cpi"

    def test_both_combined_anomalies_first(self):
        result = render_data_warnings(["US yield curve is inverted"], ["rate", "cpi"])
        assert result == "US yield curve is inverted; stale data: rate, cpi"

    def test_single_stale_series_no_trailing_comma(self):
        assert render_data_warnings([], ["price"]) == "stale data: price"


class TestResearcherThesisArchetype:
    def test_reads_from_pass2_view(self):
        compressed = {"RSRCH": {"pass2_view": {"thesis_archetype": "quality_compounder"}}}
        assert researcher_thesis_archetype(compressed) == "quality_compounder"

    def test_missing_rsrch_returns_unclassified(self):
        assert researcher_thesis_archetype({}) == "unclassified"

    def test_rsrch_without_pass2_view_returns_unclassified(self):
        assert researcher_thesis_archetype({"RSRCH": {"analysis_confidence": "insufficient"}}) == "unclassified"


class TestParseJsonResponse:
    def test_plain_json(self):
        assert parse_json_response('{"a": 1}') == {"a": 1}

    def test_strips_markdown_fences(self):
        assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}

    def test_tolerates_trailing_content_after_valid_json(self):
        """Defense-in-depth against the trailing-brace failure mode -- a
        continuation successor turn with a flatter schema than its
        predecessor's has a reproducible tendency to append one stray `}`."""
        assert parse_json_response('{"a": 1}}') == {"a": 1}

    def test_raises_on_genuinely_malformed_json(self):
        with pytest.raises(json.JSONDecodeError):
            parse_json_response("not json at all {")


class TestSmallHelpers:
    def test_truncate_to_tokens_under_limit_unchanged(self):
        assert truncate_to_tokens("short", 100) == "short"

    def test_truncate_to_tokens_over_limit_truncates(self):
        text = "x" * 500
        result = truncate_to_tokens(text, 10)  # ~40 chars
        assert result.endswith("...")
        assert len(result) < len(text)

    def test_word_count(self):
        assert word_count("one two three") == 3
        assert word_count("") == 0

    def test_char_count(self):
        assert char_count("abc") == 3
        assert char_count("") == 0

    def test_extract_numeric_tokens(self):
        tokens = extract_numeric_tokens("Revenue grew 14.2% to $185M, a 2.4x increase")
        assert "14.2%" in tokens
        assert "$185" in tokens
        assert "2.4x" in tokens
