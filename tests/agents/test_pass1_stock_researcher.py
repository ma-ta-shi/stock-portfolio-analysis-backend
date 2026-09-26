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

from agents.pass1_stock_researcher import (
    _anomalies,
    _data_coverage,
    _data_coverage_line,
    _stale_data,
    _validate_with_caveats,
    build_user_message,
)


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
            # 86bbummwp Tier 2 -- real fields on ResearchSourcesBundle, well
            # within _stale_data()'s own thresholds by default.
            latest_filing_age_days=30,
            latest_news_age_days=3,
            latest_transcript_age_days=None,  # permanent, D3
        ),
    )
    return SimpleNamespace(**{**defaults, **overrides})


def test_renders_real_header_fields():
    msg, _ = build_user_message(_bundle())
    assert "Technology | TSX | CAD" in msg
    assert "Timeline: medium_term | Account: tfsa" in msg
    assert "2026-09-23" in msg


def test_renders_business_description_from_business_digest():
    msg, _ = build_user_message(_bundle())
    assert "Business digest content." in msg


def test_business_description_honest_when_no_business_digest():
    bundle = _bundle()
    bundle.research_sources.filing_digests = [_filing_digest(section="MDA", content="MDA only.")]
    msg, _ = build_user_message(bundle)
    assert "N/A — no filing digest available" in msg


def test_renders_news_item_with_real_citation_id():
    bundle = _bundle()
    bundle.research_sources.news_items = [_news_item(id_="N7", headline="Real headline")]
    msg, _ = build_user_message(bundle)
    assert "N7: Real headline" in msg


def test_low_quality_tier_news_is_filtered_out():
    bundle = _bundle()
    bundle.research_sources.news_items = [
        _news_item(id_="N1", headline="Good story", tier="primary"),
        _news_item(id_="N2", headline="Low tier story", tier="low"),
    ]
    msg, _ = build_user_message(bundle)
    assert "Good story" in msg
    assert "Low tier story" not in msg


def test_renders_management_signals_from_structured_fields():
    bundle = _bundle()
    bundle.research_sources.management_signals = _management_signals(
        insider="selling", buyback="suspended", dividend="cut", c_suite=2
    )
    msg, _ = build_user_message(bundle)
    assert "Insider activity (90d): selling" in msg
    assert "Buyback activity: suspended" in msg
    assert "Dividend activity: cut" in msg
    assert "C-suite changes (12mo): 2" in msg


def test_renders_beta_from_risk_metrics_not_price_info():
    bundle = _bundle()
    bundle.risk_metrics = {"beta": 1.42}
    msg, _ = build_user_message(bundle)
    assert "Beta: 1.42" in msg


def test_missing_beta_renders_honestly_not_fabricated():
    bundle = _bundle()
    bundle.risk_metrics = {}
    msg, _ = build_user_message(bundle)
    assert "Beta: N/A" in msg


def test_renders_dividend_yield_as_percentage():
    bundle = _bundle()
    bundle.dividend_info = {"dividend_yield": 0.032, "payout_ratio": 0.45}
    msg, _ = build_user_message(bundle)
    assert "Yield: 3.20%" in msg
    assert "Payout ratio: 45.00%" in msg


def test_no_dividend_data_renders_honestly_not_fabricated():
    msg, _ = build_user_message(_bundle())  # defaults: both None
    assert "Yield: N/A" in msg
    assert "Payout ratio: N/A" in msg


def test_no_peers_renders_honestly():
    bundle = _bundle()
    bundle.research_sources.peer_blocks = []
    msg, _ = build_user_message(bundle)
    assert "PEER COMPARABLES: N/A" in msg


def test_no_transcript_excerpts_renders_honestly_not_fabricated():
    """transcript_excerpts is permanently [] today (D3) -- must render as
    honestly unavailable, never a fabricated placeholder."""
    msg, _ = build_user_message(_bundle())
    assert "EARNINGS TRANSCRIPT:" in msg
    assert "not available" in msg


def test_data_coverage_line_reflects_real_missing_sources():
    bundle = _bundle()
    bundle.research_sources.missing_sources_list = ["filing_digests", "peer_blocks"]
    msg, _ = build_user_message(bundle)
    assert "no filing digest available" in msg
    assert "no peer comparables available" in msg


def test_data_coverage_line_standard_when_nothing_missing_except_permanent_gap():
    msg, _ = build_user_message(_bundle())
    # Nothing in missing_sources_list, but the permanent transcript gap always
    # applies -- data_coverage_line is never a bare "standard." today.
    assert "earnings transcript excerpts are not available" in msg


def test_canadian_data_limited_flag_appears_when_sedar_unavailable():
    bundle = _bundle()
    bundle.canadian_data_flags = SimpleNamespace(sedar_filing_available=False)
    msg, _ = build_user_message(bundle)
    assert "CANADIAN DATA LIMITED: true" in msg


def test_canadian_data_limited_flag_absent_for_us_stock():
    """canadian_data_flags is None for US stocks (DataBundle's own
    None-for-US enforcement) -- must not raise or fabricate a flag."""
    msg, _ = build_user_message(_bundle())  # canadian_data_flags=None by default
    assert "CANADIAN DATA LIMITED" not in msg


# ---------- field_presence (86bbwachy Phase 4) ----------


def test_field_presence_all_true_when_default_bundle_is_fully_populated():
    """_bundle()'s own defaults have a real filing digest, news item, and
    peer block -- only dividend data is None by default (see _bundle()'s
    own dividend_info default)."""
    _, presence = build_user_message(_bundle())
    assert presence["business_description"] is True
    assert presence["recent_developments"] is True
    assert presence["filing_highlights"] is True
    assert presence["peers_block"] is True
    assert presence["beta"] is True
    assert presence["dividend_context"] is False  # yield/payout both None by default
    assert presence["earnings_transcript"] is False  # permanent D3 gap, transcript_excerpts=[]


def test_field_presence_business_description_false_without_a_business_section():
    bundle = _bundle()
    bundle.research_sources.filing_digests = [_filing_digest(section="MDA", content="MDA only.")]
    _, presence = build_user_message(bundle)
    assert presence["business_description"] is False
    # filing_highlights renders ALL sections, so a real MDA-only digest
    # still counts as present there even though business_description (the
    # Business-section-specific field) does not -- confirms the two track
    # genuinely different things, not the same signal under two names.
    assert presence["filing_highlights"] is True


def test_field_presence_beta_false_when_missing():
    bundle = _bundle()
    bundle.risk_metrics = {}
    _, presence = build_user_message(bundle)
    assert presence["beta"] is False


def test_field_presence_dividend_context_true_when_only_one_component_present():
    """yield/payout are independent signals -- either one alone is enough
    to count as present, not "both or nothing" (see _dividend_context's
    own comment)."""
    bundle = _bundle()
    bundle.dividend_info = {"dividend_yield": 0.032, "payout_ratio": None}
    _, presence = build_user_message(bundle)
    assert presence["dividend_context"] is True


def test_field_presence_peers_block_false_when_no_peers():
    bundle = _bundle()
    bundle.research_sources.peer_blocks = []
    _, presence = build_user_message(bundle)
    assert presence["peers_block"] is False


def test_field_presence_recent_developments_false_when_all_news_is_low_tier():
    bundle = _bundle()
    bundle.research_sources.news_items = [_news_item(id_="N1", tier="low")]
    _, presence = build_user_message(bundle)
    assert presence["recent_developments"] is False


def test_field_presence_has_no_entry_for_management_signals_or_whole_price_context():
    """Neither block ever renders a genuine "N/A" as a whole (see each
    helper's own code) -- no meaningful present/absent distinction to
    report, so neither key exists in the map at all."""
    _, presence = build_user_message(_bundle())
    assert "management_signals" not in presence
    assert "price_context" not in presence


# ---------- _validate_with_caveats (86bbummwp 1d) ----------


def _valid_stock_researcher_output(**overrides) -> dict:
    base = {
        "assessment_summary": "A solid mid-cap healthcare technology company with recurring revenue.",
        "analysis_confidence": "high",
        "caveats": ["Coverage limited to public filings and news."],
        "key_factors": [
            {"factor": "Recurring revenue", "importance": "high", "sentiment": "positive",
             "evidence": "FILING:MD&A: subscription revenue mix"},
            {"factor": "Roll-up integration risk", "importance": "medium", "sentiment": "negative",
             "evidence": "N1: recent acquisition closed"},
        ],
        "risks": [
            {"risk": "Integration execution risk", "severity": "medium", "evidence": "N1: recent acquisition closed"},
        ],
        "narrative": (
            "COMPANY_X demonstrates a durable recurring-revenue base built on subscription and "
            "platform fees across its clinical network (FILING:MD&A). Recent developments show "
            "continued roll-up acquisition activity, adding scale but introducing integration "
            "execution risk that management will need to manage carefully over the next several "
            "quarters (N1). Competitive position is average relative to national peers, with "
            "revenue roughly comparable to PEER_1 but trailing on operating margin, reflecting the "
            "ongoing cost of integrating acquired clinics onto a single platform. Management is "
            "assessed as competent based on a consistent acquisition and integration track record "
            "to date. Overall the thesis rests on continued execution of the roll-up strategy "
            "translating into durable, growing subscription-like revenue over the medium term "
            "horizon, with integration risk as the primary item to watch going forward."
        ),
        "structured_data": {
            "thesis_archetype": "quality_compounder",
            "competitive_position": "average",
            "moat_assessment": {"overall_moat_durability": "moderate", "moat_trend": "stable", "moats": []},
            "management_assessment": "competent",
            "growth_drivers": ["Roll-up acquisitions (N1)"],
            "competitive_threats": ["Larger national competitors (PEER_1)"],
            "recent_developments": ["Acquisition closed (N1)"],
            "peer_comparison_summary": "Comparable scale to PEER_1 on revenue, trailing on margin.",
        },
    }
    base.update(overrides)
    return base


def _validate(output, has_filing_digest, material_absent=None, anomalies=None, stale_data=None):
    return _validate_with_caveats(
        output,
        has_filing_digest=has_filing_digest,
        material_absent=material_absent or [],
        anomalies=anomalies or [],
        stale_data=stale_data or [],
    )


def test_validate_with_caveats_passes_when_digest_present_and_schema_valid():
    passed, errors = _validate(_valid_stock_researcher_output(), has_filing_digest=True)
    assert passed, errors


def test_validate_with_caveats_fails_on_base_schema_error_regardless_of_digest():
    out = _valid_stock_researcher_output(structured_data={
        **_valid_stock_researcher_output()["structured_data"], "thesis_archetype": "not_a_real_archetype",
    })
    passed, errors = _validate(out, has_filing_digest=True)
    assert not passed
    assert any("thesis_archetype" in e for e in errors)


def test_validate_with_caveats_flags_missing_filing_depth_caveat():
    out = _valid_stock_researcher_output()  # no filing-depth mention
    passed, errors = _validate(out, has_filing_digest=False)
    assert not passed
    assert any("Filing depth limited" in e for e in errors)


def test_validate_with_caveats_passes_with_real_current_phrase_when_digest_absent():
    out = _valid_stock_researcher_output(caveats=[
        "Filing depth limited: no regulatory filing text was available for this company; "
        "analysis relies on the company profile, public news and peer comparison."
    ])
    passed, errors = _validate(out, has_filing_digest=False)
    assert passed, errors


def test_validate_with_caveats_not_checked_when_digest_present():
    out = _valid_stock_researcher_output()  # no filing-depth mention, but digest is present
    passed, errors = _validate(out, has_filing_digest=True)
    assert passed, errors


# ---------- confidence/data-quality coupling rule (86bbummwp follow-on) ----------
# NOTE: validate_stock_researcher's own schema already requires caveats to have
# >=1 item unconditionally (see _validate_with_caveats's own docstring) -- so a
# "confidence high + gap + empty caveats" failure case can't be constructed
# through this composed wrapper without colliding with that unrelated base
# check first. The shared rule's own "empty caveats" branch is covered in
# isolation by test_validators_common.py; this just confirms the composition
# doesn't break a real, valid output when a gap is present.


def test_validate_with_caveats_passes_with_gap_when_real_caveat_already_present():
    out = _valid_stock_researcher_output()  # default fixture already has a real caveat
    passed, errors = _validate(out, has_filing_digest=True, material_absent=["filings"])
    assert passed, errors


# ---------- _data_coverage / _anomalies (86bbummwp Tier 2) ----------


def test_data_coverage_all_present_when_nothing_missing():
    bundle = _bundle()  # default: missing_sources_list=[]
    result = _data_coverage(bundle)
    assert result["present"] == ["filing_digests", "peer_blocks", "news_items"]
    assert result["absent"] == ["transcript_excerpts"]  # permanent, D3


def test_data_coverage_reflects_real_missing_sources():
    bundle = _bundle()
    bundle.research_sources.missing_sources_list = ["filing_digests", "peer_blocks"]
    result = _data_coverage(bundle)
    assert result["present"] == ["news_items"]
    assert set(result["absent"]) == {"filing_digests", "peer_blocks", "transcript_excerpts"}


def test_data_coverage_matches_data_coverage_line_semantics():
    """Both representations must agree on what's missing -- confirmed by
    comparing against _data_coverage_line's own prose for the same bundle."""
    bundle = _bundle()
    bundle.research_sources.missing_sources_list = ["news_items"]
    line = _data_coverage_line(bundle)
    structured = _data_coverage(bundle)
    assert "no recent news available" in line
    assert "news_items" in structured["absent"]
    assert "news_items" not in structured["present"]


def test_anomalies_empty_by_default():
    """Default fixture: buyback=active, insider=buying -- no contradiction."""
    assert _anomalies(_bundle()) == []


def test_anomalies_flags_buyback_active_with_insider_selling():
    bundle = _bundle(research_sources=SimpleNamespace(
        filing_digests=[_filing_digest()], news_items=[_news_item()], peer_blocks=[_peer_block()],
        management_signals=_management_signals(insider="selling", buyback="active"),
        missing_sources_list=[], has_filing_digest=True, transcript_excerpts=[],
    ))
    result = _anomalies(bundle)
    assert len(result) == 1
    assert "buyback" in result[0].lower()
    assert "sellers" in result[0].lower()


def test_anomalies_not_flagged_when_buyback_suspended():
    bundle = _bundle(research_sources=SimpleNamespace(
        filing_digests=[_filing_digest()], news_items=[_news_item()], peer_blocks=[_peer_block()],
        management_signals=_management_signals(insider="selling", buyback="suspended"),
        missing_sources_list=[], has_filing_digest=True, transcript_excerpts=[],
    ))
    assert _anomalies(bundle) == []


def test_anomalies_not_flagged_when_insider_buying():
    bundle = _bundle(research_sources=SimpleNamespace(
        filing_digests=[_filing_digest()], news_items=[_news_item()], peer_blocks=[_peer_block()],
        management_signals=_management_signals(insider="buying", buyback="active"),
        missing_sources_list=[], has_filing_digest=True, transcript_excerpts=[],
    ))
    assert _anomalies(bundle) == []


# ---------- _stale_data (86bbummwp Tier 2) ----------


def test_stale_data_empty_for_default_fixture():
    assert _stale_data(_bundle()) == []


def test_stale_data_flags_stale_filing_past_normal_reporting_cycle():
    bundle = _bundle()
    bundle.research_sources.latest_filing_age_days = 150
    assert "filings" in _stale_data(bundle)


def test_stale_data_filing_within_normal_quarterly_cycle_not_flagged():
    """80 days is a completely normal filing age (companies report roughly
    quarterly) -- must not be flagged."""
    bundle = _bundle()
    bundle.research_sources.latest_filing_age_days = 80
    assert "filings" not in _stale_data(bundle)


def test_stale_data_flags_stale_news():
    bundle = _bundle()
    bundle.research_sources.latest_news_age_days = 45
    assert "news" in _stale_data(bundle)


def test_stale_data_recent_news_not_flagged():
    bundle = _bundle()
    bundle.research_sources.latest_news_age_days = 5
    assert "news" not in _stale_data(bundle)


def test_stale_data_no_filing_at_all_not_flagged_as_stale():
    """age=None means no filing digest exists at all -- a data_coverage gap
    (missing_sources_list), not a staleness signal."""
    bundle = _bundle()
    bundle.research_sources.latest_filing_age_days = None
    assert "filings" not in _stale_data(bundle)


def test_stale_data_transcripts_never_checked():
    """Permanently None today (D3) -- must never appear in stale_data, that's
    data_coverage's job."""
    assert "transcripts" not in _stale_data(_bundle())
