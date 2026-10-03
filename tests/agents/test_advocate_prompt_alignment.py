"""The Bull and Bear prompts must state what their validators enforce.

Measured 2026-10-02 on the saved runs: Bull passed validation on the first attempt 14 of 30 times and Bear
19 of 30. The top cause was the narrative rule (name at least 3 of RSRCH/FUND/TECH/SENT/MACRO): when it
failed the model wrote the prose names ("macro", "fundamentals") instead of the capital IDs, because the
rule lived only in the constraints block and the narrative was asked for in words while the validator
counts characters. The rest were literals and vocabularies the schema never explained. Each test pins one
such pair so the prompt and validator cannot drift apart again."""
import re

import pytest

from agents.prompts import load_template
from agents.utils import PASS1_AGENT_IDS
from agents.validators.common import THESIS_ARCHETYPES
from agents.validators.pass2 import validate_bear_advocate, validate_bull_advocate

BULL = load_template("bull_advocate")
BEAR = load_template("bear_advocate")
BOTH = pytest.mark.parametrize("prompt", [BULL, BEAR], ids=["bull", "bear"])


def _narrative_errors(validator, n_chars: int, ids: str = "RSRCH FUND TECH") -> list[str]:
    text = (ids + " ") + "x" * (n_chars - len(ids) - 1)
    return [e for e in validator({"narrative": text})[1] if e.startswith("narrative: too")]


@pytest.mark.parametrize("validator", [validate_bull_advocate, validate_bear_advocate])
def test_the_validator_bounds_are_what_the_prompt_says(validator):
    assert _narrative_errors(validator, 999) and not _narrative_errors(validator, 1000)
    assert not _narrative_errors(validator, 2800) and _narrative_errors(validator, 2801)


@BOTH
def test_the_narrative_ask_sits_inside_the_validators_bounds_and_is_in_characters(prompt):
    assert "1,500-2,400 characters" in prompt
    assert "275-440 words" not in prompt


@BOTH
def test_the_narrative_requirement_is_on_the_schema_line_and_names_every_id(prompt):
    line = next(ln for ln in prompt.splitlines() if ln.strip().startswith('"narrative":'))
    for agent_id in PASS1_AGENT_IDS:
        assert agent_id in line
    assert "at least 3 different IDs" in line and "five short paragraphs" in line


@BOTH
def test_the_constraints_block_states_the_id_rule_and_the_summary_bounds(prompt):
    assert "at least 3 DISTINCT Pass 1 agent IDs in capitals" in prompt
    assert "thesis_summary: 80-500 characters" in prompt


@BOTH
def test_the_narrative_is_not_called_the_document_the_cio_reads_and_the_harness_is_not_mentioned(prompt):
    assert "the document the cio reads" not in prompt.lower()
    assert "harness" not in prompt.lower()
    assert "Shown to the user and also given to the CIO" in prompt


def test_bull_lists_the_archetype_vocabulary_the_validator_accepts():
    line = next(ln for ln in BULL.splitlines() if '"bull_archetype"' in ln)
    assert set(re.findall(r"[a-z_]+", line.split(":", 1)[1])) >= THESIS_ARCHETYPES


def test_bear_lists_the_archetype_vocabulary_the_validator_accepts():
    line = next(ln for ln in BEAR.splitlines() if '"bear_archetype"' in ln)
    assert set(re.findall(r"[a-z_]+", line.split(":", 1)[1])) >= THESIS_ARCHETYPES


def test_bull_key_factor_sentiment_comes_with_its_reason():
    line = next(ln for ln in BULL.splitlines() if ln.strip().startswith('"key_factors": ['))
    assert 'sentiment is always "positive"' in line and "thesis_risks" in line


def test_bear_agrees_flag_says_when_it_is_true():
    line = next(ln for ln in BEAR.splitlines() if '"agrees_with_researcher"' in ln)
    assert "true only when bear_archetype equals researcher_archetype" in line and "disagreement_note" in line


def test_bull_agrees_flag_says_when_it_is_true():
    line = next(ln for ln in BULL.splitlines() if '"alignment_with_researcher"' in ln)
    assert "agrees only when bull_archetype equals researcher_archetype" in line


def test_bull_has_no_historical_analogy_and_states_no_outside_facts():
    """Removed 2026-10-02: `none_found` in 76 of 76 saved outputs, nothing read it, and it needs facts the
    Pass 1 payload does not contain."""
    assert "analogy" not in BULL.lower()
    assert "NO OUTSIDE FACTS" in BULL


def test_bear_treats_a_missing_figure_as_a_data_gap_not_an_argument():
    assert "A figure the payload does not provide is a data gap for `caveats`, not an argument" in BEAR


@BOTH
def test_no_stale_field_name_in_the_examples(prompt):
    assert "management_signals" not in prompt


@BOTH
def test_the_evidence_must_back_the_claim_rule_is_stated(prompt):
    """Read against real inputs (2026-10-02) the facts were grounded but the reasoning slipped: a core argument
    whose own evidence cut the other way, a margin level written as expansion, a few shares of insider
    activity as a core argument. No validator can see these, so the prompt states the rule."""
    assert "EVIDENCE MUST BACK THE DIRECTION OF THE CLAIM" in prompt
    assert "margin_trend" in prompt and "operating_margin" in prompt
    assert "weakest_point" in prompt.split("EVIDENCE MUST BACK THE DIRECTION OF THE CLAIM")[1][:600]


@BOTH
def test_the_benchmark_is_named_as_an_industry_median_with_its_range_and_count(prompt):
    """MSFT's 'sector median' P/E was FTNT's 61.8 (three peers had a P/E); both advocates argued a 54% discount from it.
    The benchmark is now the whole industry's median with its middle half, and the prompt says what it is."""
    assert "industry_pe_median" in prompt and "industry_pe_count" in prompt and "industry_pe_range" in prompt
    assert "peer_pe_median" not in prompt


@BOTH
def test_the_pe_position_is_copied_from_the_code_computed_field_and_in_range_is_in_line(prompt):
    """An industry median is unstable where the industry is heterogeneous (MSFT's moved between 23.5 and 85.3 with the
    universe), so only a P/E outside the industry's middle half counts as a premium or a discount."""
    assert "pe_vs_industry_median_pct" in prompt and "pe_vs_industry" in prompt
    assert "within_range" in prompt and "above_range" in prompt and "below_range" in prompt
    assert "pe_vs_peer_median_pct" not in prompt


@BOTH
def test_insider_activity_is_a_core_argument_only_when_heavy(prompt):
    assert "insider_materiality" in prompt and "insider_activity_90d" in prompt
