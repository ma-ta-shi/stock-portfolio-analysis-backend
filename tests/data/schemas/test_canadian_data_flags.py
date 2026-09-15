import json

import pytest
from pydantic import ValidationError

from data.schemas.canadian_data_flags import CanadianDataFlags


def _flags(**overrides) -> CanadianDataFlags:
    defaults = dict(
        analyst_count=5,
        news_article_count=12,
        news_sources=["openbb_tmx", "globenewswire"],
        sedar_filing_available=False,
        statcan_available=True,
    )
    return CanadianDataFlags(**{**defaults, **overrides})


def test_valid_construction():
    flags = _flags()
    assert flags.news_sources == ["openbb_tmx", "globenewswire"]


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
    explicitly populated by whatever builds this (DataPipeline.prepare()).
    Uses a real remaining field (analyst_count) — sentiment_source, used
    here before 86bawptx4 removed it, would now fail for the wrong reason
    (an unrecognized field, not a missing-required-field case) and this
    test would silently stop testing what it claims to."""
    with pytest.raises(ValidationError):
        CanadianDataFlags(analyst_count=5)


# ---------- model_dump() round-trip (ClickUp 86bawp88h) ----------


def test_round_trips_through_model_dump():
    flags = _flags()
    assert CanadianDataFlags.model_validate(flags.model_dump()) == flags


def test_json_mode_dump_is_json_serializable_and_round_trips():
    flags = _flags()
    dumped = flags.model_dump(mode="json")
    json.dumps(dumped)
    assert CanadianDataFlags.model_validate(dumped) == flags
