import pytest
from pydantic import ValidationError

from data.schemas.canadian_data_flags import CanadianDataFlags


def _flags(**overrides) -> CanadianDataFlags:
    defaults = dict(
        sentiment_source="finnhub",
        has_transcript=True,
        transcript_source="finnhub",
        analyst_count=5,
        news_article_count=12,
        news_sources=["openbb_tmx", "globenewswire"],
        sedar_filing_available=False,
        statcan_available=True,
    )
    return CanadianDataFlags(**{**defaults, **overrides})


def test_valid_construction():
    flags = _flags()
    assert flags.sentiment_source == "finnhub"
    assert flags.news_sources == ["openbb_tmx", "globenewswire"]


@pytest.mark.parametrize("sentiment_source", ["finnhub", "local_llm", "keyword"])
def test_accepts_all_real_sentiment_sources(sentiment_source):
    _flags(sentiment_source=sentiment_source)


def test_rejects_invalid_sentiment_source():
    with pytest.raises(ValidationError):
        _flags(sentiment_source="gpt4")


def test_transcript_source_accepts_finnhub():
    flags = _flags(transcript_source="finnhub")
    assert flags.transcript_source == "finnhub"


def test_transcript_source_accepts_none():
    """has_transcript=False pairs with transcript_source=None — no
    transcript, no source to name."""
    flags = _flags(has_transcript=False, transcript_source=None)
    assert flags.transcript_source is None


def test_transcript_source_rejects_values_other_than_finnhub_or_none():
    with pytest.raises(ValidationError):
        _flags(transcript_source="local_llm")


def test_rejects_has_transcript_true_with_no_source():
    """Real gap caught on review: has_transcript=True with
    transcript_source=None is self-contradictory and must not construct
    silently — this is exactly the kind of bad upstream data this model
    exists to catch."""
    with pytest.raises(ValidationError):
        _flags(has_transcript=True, transcript_source=None)


def test_rejects_has_transcript_false_with_a_source():
    with pytest.raises(ValidationError):
        _flags(has_transcript=False, transcript_source="finnhub")


def test_rejects_negative_analyst_count():
    with pytest.raises(ValidationError):
        _flags(analyst_count=-1)


def test_rejects_negative_news_article_count():
    with pytest.raises(ValidationError):
        _flags(news_article_count=-1)


def test_accepts_zero_counts():
    """Zero is a real, valid value (no analysts, no articles) — only
    negative counts are invalid."""
    flags = _flags(analyst_count=0, news_article_count=0)
    assert flags.analyst_count == 0


def test_is_frozen():
    flags = _flags()
    with pytest.raises(ValidationError):
        flags.analyst_count = 10


def test_forbids_extra_fields():
    """Catches a precompute module silently adding/renaming a field
    before it reaches an agent prompt — e.g. the scattered
    has_finnhub_sentiment/canadian_data_limited booleans this model
    replaces must not sneak back in as extras."""
    with pytest.raises(ValidationError):
        _flags(has_finnhub_sentiment=True)


def test_requires_all_fields_no_implicit_defaults():
    """No field should silently default — every dimension must be
    explicitly populated by whatever builds this (DataPipeline.prepare())."""
    with pytest.raises(ValidationError):
        CanadianDataFlags(sentiment_source="finnhub")
