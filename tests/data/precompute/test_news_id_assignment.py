from datetime import datetime, timedelta, timezone

import structlog

from data.precompute.news_id_assignment import assign_news_ids, classify_quality_tier


def _article(
    headline="Headline", source="Reuters", published_at="2026-08-01 09:00:00", **overrides
):
    article = {
        "headline": headline,
        "source": source,
        "published_at": published_at,
        "url": "https://example.com",
    }
    article.update(overrides)
    return article


# ---------- classify_quality_tier ----------


def test_classify_quality_tier_primary_source():
    assert classify_quality_tier("Reuters") == "primary"


def test_classify_quality_tier_is_case_insensitive():
    assert classify_quality_tier("REUTERS") == "primary"
    assert classify_quality_tier("business wire") == "primary"


def test_classify_quality_tier_matches_substring_not_exact():
    """Finnhub's source strings are unnormalized (e.g. "GlobeNewswire Inc.")
    — substring matching, not exact equality, is the whole point."""
    assert classify_quality_tier("GlobeNewswire Inc.") == "primary"


def test_classify_quality_tier_secondary_source():
    assert classify_quality_tier("CNBC") == "secondary"
    assert classify_quality_tier("Yahoo Finance") == "secondary"


def test_classify_quality_tier_unknown_source_defaults_to_low():
    assert classify_quality_tier("SomeRandomBlog") == "low"


def test_classify_quality_tier_does_not_false_positive_on_sec_substring():
    """Regression guard for a real gap caught on review: "sec" was
    originally a primary-tier keyword with no evidence it ever appears as
    a real Finnhub source value, and it's a substring-collision risk
    against any source name containing "sec" as a word fragment (removed).
    A source like "Insurance Journal" or "Second Street Research" must not
    get bumped to primary just for containing that substring."""
    assert classify_quality_tier("Second Street Research") == "low"


def test_classify_quality_tier_empty_source_defaults_to_low():
    assert classify_quality_tier("") == "low"


def test_classify_quality_tier_none_source_defaults_to_low():
    assert classify_quality_tier(None) == "low"


# ---------- assign_news_ids ----------


def test_assigns_stable_n_prefixed_ids():
    articles = [_article(), _article()]
    result = assign_news_ids(articles)
    assert [item["id"] for item in result] == ["N1", "N2"]


def test_assigns_ids_oldest_first():
    articles = [
        _article(headline="Newest", published_at="2026-08-03 09:00:00"),
        _article(headline="Oldest", published_at="2026-08-01 09:00:00"),
        _article(headline="Middle", published_at="2026-08-02 09:00:00"),
    ]
    result = assign_news_ids(articles)
    assert [item["headline"] for item in result] == ["Oldest", "Middle", "Newest"]
    assert [item["id"] for item in result] == ["N1", "N2", "N3"]


def test_date_field_is_a_real_datetime():
    result = assign_news_ids([_article(published_at="2026-08-01 09:00:00")])
    assert result[0]["date"] == datetime(2026, 8, 1, 9, 0, 0)


def test_output_carries_headline_source_and_quality_tier():
    result = assign_news_ids([_article(headline="Bank raises rates", source="Reuters")])
    assert result[0]["headline"] == "Bank raises rates"
    assert result[0]["source"] == "Reuters"
    assert result[0]["quality_tier"] == "primary"


def test_output_carries_text_for_sentiment_scoring():
    """ClickUp 86ban0wf4: sentiment.py needs the article body text as LLM
    scoring input, passed through here since this function's sort+drop
    breaks positional correlation back to the raw article list. No "url"
    field - considered and cut, nothing anywhere reads it."""
    result = assign_news_ids([_article(summary="Full article body here.")])
    assert result[0]["text"] == "Full article body here."
    assert "url" not in result[0]


def test_missing_summary_defaults_text_to_empty_string_not_raise():
    result = assign_news_ids([_article()])
    assert result[0]["text"] == ""


def test_empty_article_list_returns_empty_list():
    assert assign_news_ids([]) == []


def test_drops_article_with_missing_published_at():
    articles = [_article(headline="Keep me"), {"headline": "Drop me", "source": "Reuters"}]
    result = assign_news_ids(articles)
    assert [item["headline"] for item in result] == ["Keep me"]


def test_drops_article_with_malformed_published_at():
    articles = [
        _article(headline="Keep me"),
        _article(headline="Drop me", published_at="not-a-date"),
    ]
    result = assign_news_ids(articles)
    assert [item["headline"] for item in result] == ["Keep me"]


def test_missing_headline_and_source_default_to_empty_string_not_raise():
    result = assign_news_ids([{"published_at": "2026-08-01 09:00:00"}])
    assert result[0]["headline"] == ""
    assert result[0]["source"] == ""
    assert result[0]["quality_tier"] == "low"


def test_accepts_a_real_datetime_for_published_at_not_just_a_string():
    """Defensive: a future/fixed CA provider could return a real datetime
    directly instead of Finnhub's string format."""
    result = assign_news_ids([_article(published_at=datetime(2026, 8, 1, 9, 0, 0))])
    assert result[0]["date"] == datetime(2026, 8, 1, 9, 0, 0)


def test_tz_aware_datetime_is_normalized_to_naive():
    """openbb-tmx's news results carry tz-aware timestamps (per NewsItem's
    own contract docstring); Finnhub's parsed strings are always naive.
    Real gap this catches: sorting a mix of the two raises TypeError unless
    both are normalized to the same (naive) shape first."""
    result = assign_news_ids(
        [_article(published_at=datetime(2026, 8, 1, 9, 0, 0, tzinfo=timezone.utc))]
    )
    assert result[0]["date"] == datetime(2026, 8, 1, 9, 0, 0)
    assert result[0]["date"].tzinfo is None


def test_tz_aware_datetime_is_converted_to_utc_not_just_stripped():
    """Real bug caught on review: a bare `.replace(tzinfo=None)` keeps the
    original wall-clock numbers, silently misplacing a non-UTC tz-aware
    article by its offset instead of raising. A 09:00 in UTC-5 is 14:00
    UTC — the stored naive value must reflect that conversion, not the
    original 09:00."""
    utc_minus_5 = timezone(timedelta(hours=-5))
    result = assign_news_ids(
        [_article(published_at=datetime(2026, 8, 1, 9, 0, 0, tzinfo=utc_minus_5))]
    )
    assert result[0]["date"] == datetime(2026, 8, 1, 14, 0, 0)


def test_sorts_cleanly_when_tz_aware_and_naive_datetimes_are_mixed():
    """The scenario the normalization above exists for: a US (naive,
    string-parsed) and a CA (tz-aware, once that provider is fixed)
    article in the same run must not crash the sort."""
    articles = [
        _article(headline="US article", published_at="2026-08-02 09:00:00"),
        _article(
            headline="CA article", published_at=datetime(2026, 8, 1, 9, 0, 0, tzinfo=timezone.utc)
        ),
    ]
    result = assign_news_ids(articles)
    assert [item["headline"] for item in result] == ["CA article", "US article"]


def test_ids_are_globally_assigned_not_windowed():
    """This module assigns IDs over whatever article list it's given — it
    has no concept of a per-agent window. Window filtering is the caller's
    job (research_sources.py/sentiment.py slicing the widest-window result
    down further); assign_news_ids itself must not filter anything out
    beyond dropping genuinely unusable records."""
    articles = [_article(published_at=f"2026-08-{day:02d} 09:00:00") for day in range(1, 11)]
    result = assign_news_ids(articles)
    assert len(result) == 10
    assert result[-1]["id"] == "N10"


# ---------- dropped-article logging (real gap caught on review — a
# provider-wide date-format change would previously drop every article
# with no trace in the logs) ----------


def test_logs_a_warning_when_articles_are_dropped():
    with structlog.testing.capture_logs() as logs:
        assign_news_ids([_article(headline="Drop me", published_at="not-a-date")])
    assert any(log["event"] == "news_id_assignment_dropped_articles" for log in logs)
    warning = next(log for log in logs if log["event"] == "news_id_assignment_dropped_articles")
    assert warning["dropped_count"] == 1
    assert warning["total_count"] == 1


def test_does_not_log_when_no_articles_are_dropped():
    with structlog.testing.capture_logs() as logs:
        assign_news_ids([_article()])
    assert logs == []
