"""The Sentiment prompt after the 2026-10-03 deep dive: lean, no dead inputs, every limit stated once."""

from pathlib import Path

PROMPT = (Path(__file__).resolve().parents[2] / "prompts" / "sentiment_analyst" / "v1.txt").read_text(encoding="utf-8")


def test_it_no_longer_mentions_peers_or_social_sentiment():
    for dead in ("PEER", "peer_sentiment", "social_sentiment", "SOCIAL"):
        assert dead not in PROMPT


def test_every_limit_the_validator_enforces_is_in_the_prompt():
    """Most first attempts wrote more list items than the cap because only the validator knew it."""
    assert "risks 0-2" in PROMPT and "at most 80 words" in PROMPT
    assert "450-1,500 characters" in PROMPT
    assert "key_factors" not in PROMPT  # removed 2026-10-07 (ledger BB-107)


def test_the_model_is_not_asked_for_fields_code_now_decides():
    assert "short_interest_interpretation" not in PROMPT and "analyst_sentiment" not in PROMPT


def test_the_canadian_caveat_phrase_survives_exactly():
    assert "Canadian articles are scored from headlines only" in PROMPT
    assert "{canadian_sentiment_inferred}" in PROMPT


def test_it_stays_lean():
    """About 8.5k characters before; the narrative field alone used to be 900 of them."""
    assert len(PROMPT) < 5200


