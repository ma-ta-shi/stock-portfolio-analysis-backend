"""Tests for agents/pass1_sentiment_analyst.py (86bbuhjup). No harness
equivalent -- see test_pass1_stock_researcher.py's docstring for why."""
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from agents.pass1_sentiment_analyst import (
    _anomalies,
    _data_coverage_line,
    _stale_data,
    _validate_with_caveats,
    build_user_message,
)


def _bundle(**overrides) -> SimpleNamespace:
    defaults = dict(
        stock=SimpleNamespace(ticker="AAPL", currency="USD", exchange="NASDAQ"),
        company_info={"name": "Apple Inc.", "sector": "Technology"},
        context=SimpleNamespace(account_type="tfsa", timeline="medium_term"),
        data_vintage=datetime(2026, 9, 23, tzinfo=UTC),
        canadian_data_flags=None,
        news_with_sentiment=[
            {"id": "N1", "headline": "Apple announces new product", "source": "Reuters",
             "quality_tier": "primary", "sentiment": "positive",
             "date": datetime(2026, 9, 20, tzinfo=UTC)},
        ],
        analyst_consensus={"consensus_rating": "buy", "num_analysts": 40, "target_mean": 210.0,
                            "buy_count": 30, "hold_count": 8, "sell_count": 2},
        analyst_recommendation_trends=None,
        insider_activity={"transactions": []},
        short_interest={"short_interest_pct": 1.2, "days_to_cover": 1.8,
                         "shares_short": 50_000_000, "shares_short_prior_month": 55_000_000},
        peer_sentiment=[],
    )
    return SimpleNamespace(**{**defaults, **overrides})


def test_renders_real_header_fields():
    msg, _ = build_user_message(_bundle())
    assert "AAPL (Apple Inc.) | Technology | NASDAQ | USD" in msg
    assert "Timeline: medium_term | Account: tfsa" in msg


def test_renders_news_with_real_citation_id_and_sentiment_label():
    msg, _ = build_user_message(_bundle())
    assert "N1: Apple announces new product" in msg
    assert "sentiment=positive" in msg


def test_no_news_renders_honestly():
    bundle = _bundle(news_with_sentiment=[])
    msg, _ = build_user_message(bundle)
    assert "(no news articles available)" in msg


def test_no_aggregate_score_or_themes_fabricated():
    """news_sentiment_score/news_dominant_themes have no precompute source
    -- must never appear as fabricated fields."""
    msg, _ = build_user_message(_bundle())
    assert "news_sentiment_score" not in msg
    assert "news_dominant_themes" not in msg


def test_insider_counts_exclude_issuer_and_non_directional_rows():
    cutoff = datetime.now(UTC).date()
    recent = (cutoff - timedelta(days=10)).isoformat()
    bundle = _bundle(insider_activity={"transactions": [
        {"date": recent, "is_issuer": False, "transaction_type": "purchase", "shares": 100, "value": 1000},
        {"date": recent, "is_issuer": False, "transaction_type": "purchase", "shares": 50, "value": 500},
        {"date": recent, "is_issuer": False, "transaction_type": "sale", "shares": 20, "value": 200},
        {"date": recent, "is_issuer": True, "transaction_type": "buyback", "shares": 9999, "value": 99999},
        {"date": recent, "is_issuer": False, "transaction_type": "exercise", "shares": 10, "value": 100},
    ]})
    msg, _ = build_user_message(bundle)
    assert "Insider buys (90d): 2" in msg
    assert "Insider sells (90d): 1" in msg


def test_insider_counts_exclude_stale_transactions():
    stale = (datetime.now(UTC).date() - timedelta(days=200)).isoformat()
    bundle = _bundle(insider_activity={"transactions": [
        {"date": stale, "is_issuer": False, "transaction_type": "purchase", "shares": 100, "value": 1000},
    ]})
    msg, _ = build_user_message(bundle)
    assert "Insider buys (90d): 0" in msg


def test_short_interest_renders_real_fields():
    msg, _ = build_user_message(_bundle())
    assert "Short interest % of float: 1.2" in msg
    assert "Days to cover: 1.8" in msg
    assert "Shares short: 50000000 (30d prior: 55000000)" in msg


def test_no_short_interest_renders_honestly():
    bundle = _bundle(short_interest=None)
    msg, _ = build_user_message(bundle)
    assert "N/A — no short interest data available." in msg


def test_analyst_recommendation_trends_none_renders_honestly_for_ca():
    msg, _ = build_user_message(_bundle())  # default None
    assert "N/A — not available for Canadian stocks" in msg


def test_analyst_recommendation_trends_renders_raw_distributions_not_upgrade_counts():
    """No formula derives upgrade/downgrade counts from period snapshots --
    the raw distributions render instead, not a fabricated movement count."""
    bundle = _bundle(analyst_recommendation_trends=[
        {"period": "2026-09-01", "strong_buy": 10, "buy": 15, "hold": 5, "sell": 1, "strong_sell": 0},
        {"period": "2026-08-01", "strong_buy": 8, "buy": 16, "hold": 6, "sell": 1, "strong_sell": 0},
    ])
    msg, _ = build_user_message(bundle)
    assert "2026-09-01: strong_buy=10 buy=15 hold=5 sell=1 strong_sell=0" in msg
    assert "analyst_upgrades_30d" not in msg


def test_peer_sentiment_renders_honestly_not_fabricated():
    """peer_sentiment is hardcoded [] in the data pipeline today -- a real
    gap, must never render a fabricated comparison."""
    msg, _ = build_user_message(_bundle())
    assert "PEER SENTIMENT:" in msg
    assert "not currently available" in msg


def test_canadian_flag_absent_for_us_stock():
    msg, _ = build_user_message(_bundle())  # canadian_data_flags=None
    assert "CANADIAN DATA LIMITED" not in msg


def test_canadian_flag_present_for_ca_stock():
    bundle = _bundle(canadian_data_flags=SimpleNamespace(
        analyst_count=0, news_article_count=5, news_sources=["openbb_tmx"],
        sedar_filing_available=True, statcan_available=True,
    ))
    msg, _ = build_user_message(bundle)
    assert "CANADIAN DATA LIMITED: true" in msg


# ---------- field_presence (86bbwachy Phase 4) ----------


def test_field_presence_default_bundle():
    """Default fixture: real news, real short interest, real consensus, but
    analyst_recommendation_trends=None (the CA/no-data default) and
    peer_sentiment permanently absent."""
    _, presence = build_user_message(_bundle())
    assert presence == {
        "news_block": True,
        "analyst_activity": False,
        "analyst_consensus": True,
        "short_interest": True,
        "peer_sentiment": False,
    }


def test_field_presence_news_block_false_with_no_articles():
    bundle = _bundle(news_with_sentiment=[])
    _, presence = build_user_message(bundle)
    assert presence["news_block"] is False


def test_field_presence_analyst_activity_true_with_real_trends():
    bundle = _bundle(analyst_recommendation_trends=[
        {"period": "2026-09-01", "strong_buy": 10, "buy": 15, "hold": 5, "sell": 1, "strong_sell": 0},
    ])
    _, presence = build_user_message(bundle)
    assert presence["analyst_activity"] is True


def test_field_presence_analyst_consensus_false_without_a_rating():
    bundle = _bundle(analyst_consensus={"consensus_rating": None, "num_analysts": None, "target_mean": None})
    _, presence = build_user_message(bundle)
    assert presence["analyst_consensus"] is False


def test_field_presence_short_interest_false_when_unavailable():
    bundle = _bundle(short_interest=None)
    _, presence = build_user_message(bundle)
    assert presence["short_interest"] is False


def test_field_presence_peer_sentiment_always_false():
    """Permanent gap -- always False regardless of bundle content, since
    nothing in the pipeline populates it yet (see build_user_message's own
    docstring)."""
    _, presence = build_user_message(_bundle())
    assert presence["peer_sentiment"] is False


def test_field_presence_has_no_entry_for_insider_activity():
    """buys_90d/sells_90d are always real counts (0 is a legitimate
    answer, not N/A) -- no meaningful presence/absence to report."""
    _, presence = build_user_message(_bundle())
    assert "insider_activity" not in presence


# ---------- _data_coverage_line (86bbummwp Tier 1a) ----------


def test_data_coverage_line_flags_default_fixtures_real_gaps():
    """Default fixture has no analyst_activity (CA/no-data default) and the
    permanent peer_sentiment gap -- both must be mentioned, not a bare
    'standard.'"""
    _, presence = build_user_message(_bundle())
    line = _data_coverage_line(presence)
    assert "no analyst upgrade/downgrade data available" in line
    assert "peer sentiment comparison is not yet available" in line
    assert "no news articles available" not in line  # default fixture has real news


def test_data_coverage_line_never_standard_due_to_permanent_peer_sentiment_gap():
    """peer_sentiment is a permanent gap, so this line is never a bare
    'standard.' for any real run -- confirmed even with every other field
    present."""
    bundle = _bundle(analyst_recommendation_trends=[
        {"period": "2026-09-01", "strong_buy": 10, "buy": 15, "hold": 5, "sell": 1, "strong_sell": 0},
    ])
    _, presence = build_user_message(bundle)
    line = _data_coverage_line(presence)
    assert "no analyst upgrade/downgrade data available" not in line
    assert "peer sentiment comparison is not yet available" in line


# ---------- _validate_with_caveats (86bbummwp 1d) ----------


def _valid_sentiment_analyst_output(**overrides) -> dict:
    base = {
        "assessment_summary": "Positive news flow with modest insider buying and stable analyst coverage.",
        "analysis_confidence": "high",
        "caveats": ["Coverage limited to public news and disclosed insider filings."],
        "key_factors": [
            {"factor": "Positive news flow", "importance": "high", "sentiment": "positive", "evidence": "N1: product launch"},
        ],
        "risks": [],
        "narrative": (
            "News flow for COMPANY_X has been broadly positive over the review window, anchored by "
            "a well-received product launch (N1) and steady analyst coverage. Insider activity shows "
            "modest net buying, a mildly constructive signal given the absence of any offsetting "
            "negative catalysts during the same period. Short interest remains stable relative to "
            "the prior month, showing no meaningful build in bearish positioning among short sellers. "
            "Analyst consensus remains a buy rating with a steady average price target, and no "
            "recent upgrades or downgrades have been recorded that would suggest a shift in the "
            "professional community's view of the name. Overall sentiment skews cautiously positive, "
            "with no material contrarian signals currently evident in the available data, though "
            "continued monitoring of the news cycle over the coming weeks is still warranted."
        ),
        "structured_data": {
            "news_sentiment": {"overall": "positive", "dominant_themes": ["product launch"], "sentiment_trend": "stable"},
            "analyst_sentiment": {"consensus_direction": "bullish", "recent_changes": "no changes", "avg_price_target": 20.0},
            "insider_activity_interpretation": "Modest net insider buying, a mildly positive signal.",
            "short_interest_interpretation": {"trend": "stable", "interpretation": "normal"},
            "social_sentiment": "unknown",
            "narrative_momentum": "stable",
            "positioning_assessment": "neutral",
            "analyst_consensus": "buy",
            "peer_sentiment_comparison": "Not currently available.",
        },
        "contrarian_signals": [],
        "pass2_view": {},
    }
    base.update(overrides)
    return base


def _validate(output, canadian_sentiment_inferred, material_absent=None, anomalies=None, stale_data=None):
    return _validate_with_caveats(
        output,
        canadian_sentiment_inferred=canadian_sentiment_inferred,
        material_absent=material_absent or [],
        anomalies=anomalies or [],
        stale_data=stale_data or [],
    )


def test_validate_with_caveats_passes_when_not_inferred_and_schema_valid():
    passed, errors = _validate(_valid_sentiment_analyst_output(), canadian_sentiment_inferred=False)
    assert passed, errors


def test_validate_with_caveats_fails_on_base_schema_error_regardless_of_inference():
    out = _valid_sentiment_analyst_output(structured_data={
        **_valid_sentiment_analyst_output()["structured_data"], "social_sentiment": "bearish",
    })
    passed, errors = _validate(out, canadian_sentiment_inferred=False)
    assert not passed
    assert any("social_sentiment" in e for e in errors)


def test_validate_with_caveats_flags_missing_canadian_caveat():
    out = _valid_sentiment_analyst_output()  # no Canadian inference mention
    passed, errors = _validate(out, canadian_sentiment_inferred=True)
    assert not passed
    assert any("scored from headlines only" in e for e in errors)


def test_validate_with_caveats_passes_with_real_current_phrase_when_inferred():
    out = _valid_sentiment_analyst_output(caveats=[
        "Article sentiment is scored by a local LLM on both markets, not supplied by a data "
        "provider, and is not validated against a ground-truth dataset. Canadian articles are "
        "scored from headlines only — the Canadian news feed returns no article body."
    ])
    passed, errors = _validate(out, canadian_sentiment_inferred=True)
    assert passed, errors


def test_validate_with_caveats_not_checked_when_not_inferred():
    out = _valid_sentiment_analyst_output()  # no Canadian inference mention, but not inferred
    passed, errors = _validate(out, canadian_sentiment_inferred=False)
    assert passed, errors


# ---------- confidence/data-quality coupling rule (86bbummwp follow-on) ----------


def test_validate_with_caveats_flags_high_confidence_with_anomaly_and_no_caveat():
    out = _valid_sentiment_analyst_output(caveats=[])
    passed, errors = _validate(out, canadian_sentiment_inferred=False, anomalies=["real divergence"])
    assert not passed
    assert any("caveats is empty" in e for e in errors)


def test_validate_with_caveats_passes_high_confidence_with_anomaly_when_caveat_present():
    out = _valid_sentiment_analyst_output(caveats=["News sentiment positive despite net insider selling."])
    passed, errors = _validate(out, canadian_sentiment_inferred=False, anomalies=["real divergence"])
    assert passed, errors


# ---------- _stale_data (86bbummwp Tier 2) ----------


def test_stale_data_empty_for_default_fixture():
    """Default fixture: news is 3 days old (well within threshold), and
    short_interest has no as_of_date key at all (not applicable, not stale)."""
    assert _stale_data(_bundle()) == []


def test_stale_data_flags_old_news():
    bundle = _bundle(news_with_sentiment=[
        {"id": "N1", "headline": "Old story", "source": "Reuters", "quality_tier": "primary",
         "sentiment": "neutral", "date": datetime(2026, 8, 1, tzinfo=UTC)},
    ])
    assert _stale_data(bundle) == ["news"]


def test_stale_data_handles_naive_article_date():
    """Regression test for a real bug caught live (2026-09-25, a real SENT
    run on RDDT): finnhub.py builds article dates via bare
    datetime.fromtimestamp()/datetime.now() (naive), while data_vintage is
    datetime.now(UTC) (aware) -- subtracting them raised "can't subtract
    offset-naive and offset-aware datetimes" and failed the whole agent.
    Must not raise, naive or aware."""
    bundle = _bundle(news_with_sentiment=[
        {"id": "N1", "headline": "Old story", "source": "Reuters", "quality_tier": "primary",
         "sentiment": "neutral", "date": datetime(2026, 8, 1)},  # naive, no tzinfo
    ])
    assert _stale_data(bundle) == ["news"]


def test_stale_data_uses_most_recent_article_not_oldest():
    bundle = _bundle(news_with_sentiment=[
        {"id": "N1", "headline": "Old", "source": "Reuters", "quality_tier": "primary",
         "sentiment": "neutral", "date": datetime(2026, 1, 1, tzinfo=UTC)},
        {"id": "N2", "headline": "Recent", "source": "Reuters", "quality_tier": "primary",
         "sentiment": "neutral", "date": datetime(2026, 9, 22, tzinfo=UTC)},
    ])
    assert _stale_data(bundle) == []  # the recent one is what matters


def test_stale_data_no_news_is_not_flagged():
    """No articles at all is a coverage gap (data_coverage's job), not a
    staleness signal -- nothing to compute an age from."""
    assert _stale_data(_bundle(news_with_sentiment=[])) == []


def test_stale_data_flags_old_short_interest():
    bundle = _bundle(short_interest={
        "short_interest_pct": 1.2, "days_to_cover": 1.8, "shares_short": 50_000_000,
        "shares_short_prior_month": 55_000_000, "as_of_date": "2026-06-01",
    })
    assert "short_interest" in _stale_data(bundle)


def test_stale_data_recent_short_interest_not_flagged():
    bundle = _bundle(short_interest={
        "short_interest_pct": 1.2, "days_to_cover": 1.8, "shares_short": 50_000_000,
        "shares_short_prior_month": 55_000_000, "as_of_date": "2026-09-10",
    })
    assert "short_interest" not in _stale_data(bundle)


def test_stale_data_no_short_interest_is_not_flagged():
    assert "short_interest" not in _stale_data(_bundle(short_interest=None))


# ---------- _anomalies (86bbummwp Tier 2) ----------


def test_anomalies_empty_for_default_fixture():
    """Default: 1 positive article, no insider transactions, low short
    interest -- no divergence to flag."""
    assert _anomalies(_bundle()) == []


def test_anomalies_empty_when_sentiment_not_strongly_positive():
    bundle = _bundle(news_with_sentiment=[
        {"id": "N1", "headline": "Mixed", "source": "Reuters", "quality_tier": "primary",
         "sentiment": "negative", "date": datetime(2026, 9, 20, tzinfo=UTC)},
    ])
    assert _anomalies(bundle) == []


def test_anomalies_flags_positive_sentiment_with_net_insider_selling():
    recent = (datetime.now(UTC).date() - timedelta(days=10)).isoformat()
    bundle = _bundle(
        insider_activity={"transactions": [
            {"date": recent, "is_issuer": False, "transaction_type": "sale", "shares": 100, "value": 1000},
            {"date": recent, "is_issuer": False, "transaction_type": "sale", "shares": 100, "value": 1000},
        ]},
    )
    result = _anomalies(bundle)
    assert any("net sellers" in f for f in result)


def test_anomalies_flags_positive_sentiment_with_elevated_short_interest():
    bundle = _bundle(short_interest={
        "short_interest_pct": 15.0, "days_to_cover": 4.0, "shares_short": 90_000_000,
        "shares_short_prior_month": 80_000_000,
    })
    result = _anomalies(bundle)
    assert any("elevated short interest" in f for f in result)


def test_anomalies_no_scored_articles_returns_empty():
    """Only unscored articles (sentiment=None) -- nothing to compute a ratio
    from, must not crash."""
    bundle = _bundle(news_with_sentiment=[
        {"id": "N1", "headline": "Unscored", "source": "Reuters", "quality_tier": "primary",
         "sentiment": None, "date": datetime(2026, 9, 20, tzinfo=UTC)},
    ])
    assert _anomalies(bundle) == []
