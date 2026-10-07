"""The Macro prompt after the 2026-10-06 deep dive: every limit stated once, the narrative target above the floor."""

from pathlib import Path

PROMPT = (Path(__file__).resolve().parents[2] / "prompts" / "macro_economist" / "v1.txt").read_text(encoding="utf-8")


def test_every_limit_the_validator_enforces_is_in_the_prompt():
    """23 of 57 real first attempts failed; most wrote more list items than the cap because only the validator knew it."""
    assert "risks 1-3" in PROMPT and "assessment_summary 80 words or fewer" in PROMPT
    assert "key_factors" not in PROMPT  # removed 2026-10-07 (ledger BB-107)


def test_the_narrative_target_sits_above_the_480_floor():
    """The model wrote about 500 characters when told 560 (5 of 16 replay calls under 480); told 640 it passed 15 of 16."""
    assert "640-720 characters" in PROMPT
    assert "HARD FLOOR" not in PROMPT  # the repeated 900-character warning is gone


def test_the_statcan_block_has_a_token_and_a_rule():
    assert "STATCAN" in PROMPT and "STATISTICS CANADA block" in PROMPT


def test_it_says_what_the_growth_label_means():
    assert "YoY growth vs last quarter" in PROMPT
