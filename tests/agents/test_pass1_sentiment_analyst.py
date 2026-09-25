"""Tests for agents/pass1_sentiment_analyst.py (86bbuhjup). No harness
equivalent -- see test_pass1_stock_researcher.py's docstring for why."""
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from agents.pass1_sentiment_analyst import build_user_message


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
