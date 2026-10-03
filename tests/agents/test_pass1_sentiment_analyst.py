"""Tests for agents/pass1_sentiment_analyst.py (86bbuhjup). No harness
equivalent -- see test_pass1_stock_researcher.py's docstring for why."""
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from agents.pass1_sentiment_analyst import (
    _anomalies,
    _data_coverage_line,
    _insider_counts,
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
        price_info={"market_cap": 3.0e12, "current_price": 200.0},
        analyst_rating_changes=None,
        short_interest={"short_interest_pct": 1.2, "days_to_cover": 1.8,
                         "shares_short": 50_000_000, "shares_short_prior_month": 55_000_000},
    )
    return SimpleNamespace(**{**defaults, **overrides})


def test_renders_real_header_fields():
    msg, _ = build_user_message(_bundle())
    assert "AAPL (Apple Inc.) | Technology | NASDAQ | USD" in msg
    assert "Timeline: medium_term | Account: tfsa" in msg


def test_renders_news_with_real_citation_id_and_sentiment_label():
    msg, _ = build_user_message(_bundle())
    assert "N1 09-20: Apple announces new product" in msg
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
    assert "Insider activity (90d): " in msg
    assert _insider_counts(bundle) == (2, 1)


def test_insider_counts_exclude_stale_transactions():
    stale = (datetime.now(UTC).date() - timedelta(days=200)).isoformat()
    bundle = _bundle(insider_activity={"transactions": [
        {"date": stale, "is_issuer": False, "transaction_type": "purchase", "shares": 100, "value": 1000},
    ]})
    msg, _ = build_user_message(bundle)
    assert _insider_counts(bundle) == (0, 0)
    assert "no qualifying insider purchases or sales" in msg


def test_short_interest_renders_real_fields():
    msg, _ = build_user_message(_bundle())
    assert "1.2% of float, 1.8 days to cover" in msg
    assert "shares short 50,000,000 against 55,000,000 a month earlier (-9.1%)" in msg


def test_no_short_interest_renders_honestly():
    bundle = _bundle(short_interest=None)
    msg, _ = build_user_message(bundle)
    assert "N/A -- no short interest data available." in msg


def test_no_analyst_distribution_and_no_changes_renders_honestly():
    msg, _ = build_user_message(_bundle())  # both default None
    assert "N/A -- no rating distribution or rating changes returned." in msg


def test_rating_changes_and_the_target_against_the_price_are_rendered_for_a_canadian_name():
    """TD.TO 2026-10-03: yfinance has the distribution and the dated changes; the old payload said 'not available
    for Canadian stocks'."""
    bundle = _bundle(
        analyst_recommendation_trends=[{"period": "0m", "strong_buy": 5, "buy": 4, "hold": 3, "sell": 0, "strong_sell": 2}],
        analyst_rating_changes=[
            {"date": "2026-09-20", "action": "up", "price_target_action": "Raises"},
            {"date": "2026-09-10", "action": "main", "price_target_action": "Lowers"},
        ],
    )
    msg, _ = build_user_message(bundle)
    assert "0m: strong_buy=5 buy=4 hold=3 sell=0 strong_sell=2" in msg
    assert "Rating changes: 1 upgrade, 0 downgrades, 0 new initiations in the last 90d; price targets 1 raised, 1 cut" in msg
    assert "average target +5.0% against the price" in msg


def test_a_quiet_name_says_when_the_last_change_was():
    bundle = _bundle(analyst_rating_changes=[{"date": "2026-06-01", "action": "main", "price_target_action": "Raises"}])
    msg, _ = build_user_message(bundle)
    assert "Rating changes: none in the last 90d (latest 2026-06-01)" in msg


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


def test_no_peer_or_social_text_is_in_the_payload():
    """Peer sentiment was '[]' in 48 of 48 real runs and social sentiment is always unknown: neither is shown."""
    msg, _ = build_user_message(_bundle())
    assert "PEER" not in msg and "social_sentiment" not in msg


def test_canadian_flag_absent_for_us_stock():
    msg, _ = build_user_message(_bundle())  # canadian_data_flags=None
    assert "Canadian stock" not in msg


def test_canadian_flag_present_for_ca_stock():
    bundle = _bundle(canadian_data_flags=SimpleNamespace(
        analyst_count=0, news_article_count=5, news_sources=["openbb_tmx"],
        sedar_filing_available=True, statcan_available=True,
    ))
    msg, _ = build_user_message(bundle)
    assert "Canadian stock: article sentiment is scored from headlines only" in msg
    assert "Finnhub" not in msg


# ---------- field_presence (86bbwachy Phase 4) ----------


def test_field_presence_default_bundle():
    """Default fixture: real news, real short interest, real consensus, but no rating distribution and no
    rating changes."""
    _, presence = build_user_message(_bundle())
    assert presence == {
        "news_block": True,
        "analyst_activity": False,
        "analyst_consensus": True,
        "short_interest": True,
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


def test_field_presence_analyst_activity_true_with_only_rating_changes():
    """A Canadian name with no distribution but real rating changes still has analyst activity."""
    bundle = _bundle(analyst_rating_changes=[{"date": "2026-09-20", "action": "up"}])
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


def test_field_presence_has_no_entry_for_insider_activity():
    """buys_90d/sells_90d are always real counts (0 is a legitimate
    answer, not N/A) -- no meaningful presence/absence to report."""
    _, presence = build_user_message(_bundle())
    assert "insider_activity" not in presence


# ---------- _data_coverage_line (86bbummwp Tier 1a) ----------


def test_data_coverage_line_flags_default_fixtures_real_gaps():
    """Default fixture has no analyst_activity: mentioned, not a bare 'standard.'"""
    _, presence = build_user_message(_bundle())
    line = _data_coverage_line(presence)
    assert "no analyst rating data available" in line
    assert "peer sentiment" not in line
    assert "no news articles available" not in line  # default fixture has real news


def test_data_coverage_line_is_standard_when_everything_is_present():
    """With no permanent gap left (peer sentiment is retired) a fully covered run reads as standard."""
    bundle = _bundle(analyst_recommendation_trends=[
        {"period": "2026-09-01", "strong_buy": 10, "buy": 15, "hold": 5, "sell": 1, "strong_sell": 0},
    ])
    _, presence = build_user_message(bundle)
    line = _data_coverage_line(presence)
    assert "no analyst rating data available" not in line
    assert "standard" in line.lower()


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
            "social_sentiment": "unknown",
            "narrative_momentum": "stable",
            "positioning_assessment": "neutral",
            "analyst_consensus": "buy",
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


# --- a sampled month of news: only the shown headlines are listed, the rest is statistics (BB-023) ---


def _sample(count: int, shown_every: int = 3) -> list[dict]:
    """A scored sample over two ISO weeks (2026-09-07 and 2026-09-14), every third one shown."""
    out = []
    for i in range(count):
        day = 7 + (i % 10)  # Sep 7 to Sep 16
        out.append(
            {
                "id": f"N{i + 1}",
                "headline": f"Headline {i + 1}",
                "source": "Yahoo",
                "quality_tier": "secondary",
                "sentiment": ["positive", "negative", "neutral"][i % 3],
                "date": datetime(2026, 9, day, 12, tzinfo=UTC),
                "shown": i % shown_every == 0,
            }
        )
    return out


def test_only_the_shown_headlines_are_listed():
    bundle = _bundle(news_with_sentiment=_sample(30))

    msg, _ = build_user_message(bundle)

    assert "N1 09-07: Headline 1" in msg and "N4 09-10: Headline 4" in msg  # shown, with their dates
    assert "Headline 2 (" not in msg and "Headline 3 (" not in msg  # scored, not shown


def test_the_coverage_totals_and_weekly_tone_are_reported_when_known():
    bundle = _bundle(
        news_with_sentiment=_sample(30), news_coverage={"window_days": 30, "fetched": 2400}
    )

    msg, _ = build_user_message(bundle)

    assert "2400 articles fetched over the last 30 days; 30 sampled" in msg
    assert "10 listed below" in msg
    assert "Tone of the scored sample: 10 positive, 10 negative, 10 neutral (33% positive, 33% negative, 33% neutral)." in msg
    assert "Week of 2026-09-07:" in msg and "Week of 2026-09-14:" in msg
    assert "sample, so they can show a large shift" in msg


def test_the_tone_shares_are_of_the_whole_sample_not_of_the_non_neutral_ones():
    """KO 2026-10-03: 48 positive, 19 negative, 42 neutral was written up as '72% positive' (48 of 67)."""
    from agents.pass1_sentiment_analyst import _tone_shares

    articles = [{"sentiment": "positive"}] * 48 + [{"sentiment": "negative"}] * 19 + [{"sentiment": "neutral"}] * 42
    assert _tone_shares(articles) == "44% positive, 17% negative, 39% neutral"


def test_weekly_counts_add_up_to_the_sample():
    import re

    bundle = _bundle(
        news_with_sentiment=_sample(30), news_coverage={"window_days": 30, "fetched": 900}
    )

    msg, _ = build_user_message(bundle)

    totals = [int(n) for n in re.findall(r"Week of [\d-]+: .*\(n=(\d+)\)", msg)]
    assert sum(totals) == 30


def test_without_coverage_the_payload_is_unchanged():
    msg, _ = build_user_message(_bundle())

    assert "Coverage:" not in msg and "Week of" not in msg
    assert "N1 09-20: Apple announces new product" in msg


def test_the_anomaly_check_uses_the_whole_scored_sample_not_just_the_shown_ones():
    positive = [
        {**a, "sentiment": "positive", "shown": a["id"] == "N1"} for a in _sample(30)
    ]
    bundle = _bundle(
        news_with_sentiment=positive,
        insider_activity={
            "transactions": [
                {"is_issuer": False, "transaction_type": "sale", "date": "2026-09-10"} for _ in range(3)
            ]
        },
    )

    flags = _anomalies(bundle)

    assert flags and "100% positive" in flags[0]  # 30 of 30, though only 1 headline is shown


def test_insider_line_keeps_only_notable_transactions_for_the_companys_size():
    recent = (datetime.now(UTC).date() - timedelta(days=5)).isoformat()
    sale = {"date": recent, "is_issuer": False, "transaction_type": "sale", "shares": 100, "value": 6_000_000}
    # $28.3B: threshold 0.001% = $283K, so the $6M sale is notable; at $3.0T the threshold is $30M, so it is routine.
    big, small = build_user_message(_bundle(insider_activity={"transactions": [sale]}, price_info={"market_cap": 28.3e9}))[0],         build_user_message(_bundle(insider_activity={"transactions": [sale]}, price_info={"market_cap": 3.0e12}))[0]
    assert "Insider activity (90d): net selling $6.0M from 1 notable of 1 transactions" in big
    assert "none above the notable threshold of $30.0M: routine" in small


# ---------- dominant_themes anchored to real news IDs (2026-10-03) ----------


def _themed(ids):
    return _valid_sentiment_analyst_output(structured_data={
        **_valid_sentiment_analyst_output()["structured_data"],
        "news_sentiment": {"overall": "positive", "sentiment_trend": "stable",
                           "dominant_themes": [{"theme": "t", "sentiment": "positive", "primary_news_id": i} for i in ids]},
    })


def test_a_theme_anchored_to_a_listed_news_id_passes():
    passed, errors = _validate_with_caveats(
        _themed(["N1", "N4"]), canadian_sentiment_inferred=False, material_absent=[], anomalies=[], stale_data=[],
        news_ids={"N1", "N4", "N7"},
    )
    assert passed, errors


def test_a_theme_anchored_to_a_source_label_or_unlisted_id_is_rejected():
    """KO 2026-10-03: a theme's primary_news_id was 'ANALYST'."""
    passed, errors = _validate_with_caveats(
        _themed(["N1", "ANALYST", "N99"]), canadian_sentiment_inferred=False, material_absent=[], anomalies=[], stale_data=[],
        news_ids={"N1", "N4"},
    )
    assert not passed
    assert any("'ANALYST'" in e for e in errors) and any("'N99'" in e for e in errors)


def test_the_average_target_is_shown_to_cents_not_as_yahoo_gave_it():
    """MSFT 2026-10-03: 'Avg target: 578.82245 USD' was copied into evidence as written."""
    msg, _ = build_user_message(_bundle(analyst_consensus={"consensus_rating": "buy", "num_analysts": 3, "target_mean": 578.82245}))
    assert "Avg target: 578.82 USD" in msg and "578.82245" not in msg
    msg, _ = build_user_message(_bundle(analyst_consensus={"consensus_rating": "buy", "num_analysts": 3, "target_mean": None}))
    assert "Avg target: N/A USD" in msg


def test_an_identical_second_monthly_distribution_row_is_not_repeated():
    """Yahoo's 0m and -1m rows are often identical (TD.TO, SHOP.TO, ENB.TO 2026-10-03): the repeat adds tokens, not information."""
    same = {"strong_buy": 5, "buy": 4, "hold": 3, "sell": 0, "strong_sell": 2}
    msg, _ = build_user_message(_bundle(analyst_recommendation_trends=[{"period": "0m", **same}, {"period": "-1m", **same}]))
    assert "0m: strong_buy=5" in msg and "-1m:" not in msg
    changed = {**same, "buy": 6}
    msg, _ = build_user_message(_bundle(analyst_recommendation_trends=[{"period": "0m", **changed}, {"period": "-1m", **same}]))
    assert "0m: strong_buy=5 buy=6" in msg and "-1m: strong_buy=5 buy=4" in msg
