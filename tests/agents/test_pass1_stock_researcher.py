"""Tests for agents/pass1_stock_researcher.py (86bbuhjup).

No harness equivalent exists -- simulation/runners/ has no unit tests for
individual runners (they're only exercised indirectly via the sweep), so
these are net-new. Focus is field-completeness: confirming build_user_message
actually renders real DataBundle field values, not just that it doesn't
crash -- the class of bug 86bbt1k1p/86bbt1kct found in the CIO/Shadow payload
builders (fields silently rendering "NOT AVAILABLE"/"N/A" instead of the real
value).
"""
from datetime import UTC, datetime
from types import SimpleNamespace

from agents.pass1_stock_researcher import build_user_message


def _news_item(id_="N1", headline="Company announces new product", tier="primary"):
    return SimpleNamespace(
        id=id_, headline=headline, source="Reuters", quality_tier=tier,
        date=datetime(2026, 9, 20, tzinfo=UTC),
    )


def _filing_digest(section="Business", content="Business digest content."):
    return SimpleNamespace(section=section, content=content, token_count=50)


def _peer_block(peer_id="PEER_1", content="Business: peer summary. Recent news: peer news."):
    return SimpleNamespace(peer_id=peer_id, content=content, token_count=40)


def _management_signals(
    insider="buying", buyback="active", dividend="held", c_suite=None
):
    return SimpleNamespace(
        insider_net_direction_90d=insider,
        buyback_activity=buyback,
        dividend_activity=dividend,
        c_suite_changes_12mo=c_suite,
        changes_detail="",
    )


def _bundle(**overrides) -> SimpleNamespace:
    """SimpleNamespace, not a real DataBundle -- build_user_message only reads
    a handful of attributes off it, and constructing a fully valid DataBundle
    (with its cross-field validators) is unrelated overhead for a rendering
    test. Same convention as test_pass2_view.py's equivalent bundle stand-in."""
    defaults = dict(
        stock=SimpleNamespace(ticker="SHOP.TO", currency="CAD", exchange="TSX"),
        company_info={"name": "Shopify Inc", "sector": "Technology"},
        context=SimpleNamespace(account_type="tfsa", timeline="medium_term"),
        data_vintage=datetime(2026, 9, 23, tzinfo=UTC),
        canadian_data_flags=None,
        price_info={"current_price": 105.0, "market_cap": 1.3e11, "currency": "CAD",
                     "high_52w": 120.0, "low_52w": 60.0},
        risk_metrics={"beta": 1.8},
        dividend_info={"dividend_yield": None, "payout_ratio": None},
        research_sources=SimpleNamespace(
            filing_digests=[_filing_digest()],
            news_items=[_news_item()],
            peer_blocks=[_peer_block()],
            management_signals=_management_signals(),
            missing_sources_list=[],
            has_filing_digest=True,
            transcript_excerpts=[],
        ),
    )
    return SimpleNamespace(**{**defaults, **overrides})


def test_renders_real_header_fields():
    msg = build_user_message(_bundle())
    assert "Technology | TSX | CAD" in msg
    assert "Timeline: medium_term | Account: tfsa" in msg
    assert "2026-09-23" in msg


def test_renders_business_description_from_business_digest():
    msg = build_user_message(_bundle())
    assert "Business digest content." in msg


def test_business_description_honest_when_no_business_digest():
    bundle = _bundle()
    bundle.research_sources.filing_digests = [_filing_digest(section="MDA", content="MDA only.")]
    msg = build_user_message(bundle)
    assert "N/A — no filing digest available" in msg


def test_renders_news_item_with_real_citation_id():
    bundle = _bundle()
    bundle.research_sources.news_items = [_news_item(id_="N7", headline="Real headline")]
    msg = build_user_message(bundle)
    assert "N7: Real headline" in msg


def test_low_quality_tier_news_is_filtered_out():
    bundle = _bundle()
    bundle.research_sources.news_items = [
        _news_item(id_="N1", headline="Good story", tier="primary"),
        _news_item(id_="N2", headline="Low tier story", tier="low"),
    ]
    msg = build_user_message(bundle)
    assert "Good story" in msg
    assert "Low tier story" not in msg


def test_renders_management_signals_from_structured_fields():
    bundle = _bundle()
    bundle.research_sources.management_signals = _management_signals(
        insider="selling", buyback="suspended", dividend="cut", c_suite=2
    )
    msg = build_user_message(bundle)
    assert "Insider activity (90d): selling" in msg
    assert "Buyback activity: suspended" in msg
    assert "Dividend activity: cut" in msg
    assert "C-suite changes (12mo): 2" in msg


def test_renders_beta_from_risk_metrics_not_price_info():
    bundle = _bundle()
    bundle.risk_metrics = {"beta": 1.42}
    msg = build_user_message(bundle)
    assert "Beta: 1.42" in msg


def test_missing_beta_renders_honestly_not_fabricated():
    bundle = _bundle()
    bundle.risk_metrics = {}
    msg = build_user_message(bundle)
    assert "Beta: N/A" in msg


def test_renders_dividend_yield_as_percentage():
    bundle = _bundle()
    bundle.dividend_info = {"dividend_yield": 0.032, "payout_ratio": 0.45}
    msg = build_user_message(bundle)
    assert "Yield: 3.20%" in msg
    assert "Payout ratio: 45.00%" in msg


def test_no_dividend_data_renders_honestly_not_fabricated():
    msg = build_user_message(_bundle())  # defaults: both None
    assert "Yield: N/A" in msg
    assert "Payout ratio: N/A" in msg


def test_no_peers_renders_honestly():
    bundle = _bundle()
    bundle.research_sources.peer_blocks = []
    msg = build_user_message(bundle)
    assert "PEER COMPARABLES: N/A" in msg


def test_no_transcript_excerpts_renders_honestly_not_fabricated():
    """transcript_excerpts is permanently [] today (D3) -- must render as
    honestly unavailable, never a fabricated placeholder."""
    msg = build_user_message(_bundle())
    assert "EARNINGS TRANSCRIPT:" in msg
    assert "not available" in msg


def test_data_coverage_line_reflects_real_missing_sources():
    bundle = _bundle()
    bundle.research_sources.missing_sources_list = ["filing_digests", "peer_blocks"]
    msg = build_user_message(bundle)
    assert "no filing digest available" in msg
    assert "no peer comparables available" in msg


def test_data_coverage_line_standard_when_nothing_missing_except_permanent_gap():
    msg = build_user_message(_bundle())
    # Nothing in missing_sources_list, but the permanent transcript gap always
    # applies -- data_coverage_line is never a bare "standard." today.
    assert "earnings transcript excerpts are not available" in msg


def test_canadian_data_limited_flag_appears_when_sedar_unavailable():
    bundle = _bundle()
    bundle.canadian_data_flags = SimpleNamespace(sedar_filing_available=False)
    msg = build_user_message(bundle)
    assert "CANADIAN DATA LIMITED: true" in msg


def test_canadian_data_limited_flag_absent_for_us_stock():
    """canadian_data_flags is None for US stocks (DataBundle's own
    None-for-US enforcement) -- must not raise or fabricate a flag."""
    msg = build_user_message(_bundle())  # canadian_data_flags=None by default
    assert "CANADIAN DATA LIMITED" not in msg
