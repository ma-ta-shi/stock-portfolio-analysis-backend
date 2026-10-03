"""The Sentiment prompt after the 2026-10-03 deep dive: lean, no dead inputs, every limit stated once."""

from pathlib import Path

PROMPT = (Path(__file__).resolve().parents[2] / "prompts" / "sentiment_analyst" / "v1.txt").read_text(encoding="utf-8")


def test_it_no_longer_mentions_peers_or_social_sentiment():
    for dead in ("PEER", "peer_sentiment", "social_sentiment", "SOCIAL"):
        assert dead not in PROMPT


def test_every_limit_the_validator_enforces_is_in_the_prompt():
    """42 of 48 first attempts wrote 4 or 5 key_factors because only the validator knew the cap was 3."""
    assert "key_factors 1-3 items" in PROMPT and "risks 0-2" in PROMPT and "at most 80 words" in PROMPT
    assert "600-1080 characters" in PROMPT


def test_the_model_is_not_asked_for_fields_code_now_decides():
    assert "short_interest_interpretation" not in PROMPT and "analyst_sentiment" not in PROMPT


def test_the_canadian_caveat_phrase_survives_exactly():
    assert "Canadian articles are scored from headlines only" in PROMPT
    assert "{canadian_sentiment_inferred}" in PROMPT


def test_it_stays_lean():
    """About 8.5k characters before; the narrative field alone used to be 900 of them."""
    assert len(PROMPT) < 5200


def test_a_theme_is_never_anchored_to_a_source_label():
    """KO twice anchored a theme to ANALYST (2 of 24 replays)."""
    assert "never ANALYST, INSIDER or SHORT" in PROMPT
