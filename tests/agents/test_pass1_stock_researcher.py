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
    buyback="active", dividend="held", c_suite=None
):
    return SimpleNamespace(
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
        currency_mismatch=None,
        company_info={"name": "Shopify Inc", "sector": "Technology"},
        context=SimpleNamespace(account_type="tfsa", timeline="medium_term"),
        data_vintage=datetime(2026, 9, 23, tzinfo=UTC),
        canadian_data_flags=None,
        price_info={"current_price": 105.0, "market_cap": 1.3e11, "currency": "CAD",
                     "high_52w": 120.0, "low_52w": 60.0},
        risk_metrics={"beta": 1.8},
        insider_activity={"transactions": [], "value_currency": "CAD"},
        dividend_info={"dividend_yield": None, "payout_ratio": None},
        research_sources=SimpleNamespace(
            filing_digests=[_filing_digest(), _filing_digest(section="MDA", content="MDA digest content.")],
            news_items=[_news_item()],
            peer_blocks=[_peer_block()],
            management_signals=_management_signals(),
            missing_sources_list=[],
            has_filing_digest=True,
            transcript_excerpts=[],
            peer_names={"PEER_1": "Peer One Inc."},
            dual_class_flag=False,
            business_profile=None,
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
        buyback="suspended", dividend="cut", c_suite=2
    )
    msg, _ = build_user_message(bundle)
    assert "Insider activity (90d): no qualifying insider purchases or sales" in msg  # the sized summary, not a bare direction
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
    assert "EARNINGS TRANSCRIPT" not in msg  # nothing in it, so the block is left out rather than sent as a dead line


def test_data_coverage_line_reflects_real_missing_sources():
    bundle = _bundle()
    bundle.research_sources.missing_sources_list = ["filing_digests", "peer_blocks"]
    msg, _ = build_user_message(bundle)
    assert "no filing digest available" in msg
    assert "no peer comparables available" in msg


def test_data_coverage_line_standard_when_nothing_is_missing():
    msg, _ = build_user_message(_bundle())
    # The permanent transcript gap is not listed (the same on every run, nothing the model can act on); it stays in
    # the stored data_coverage the confidence rule reads.
    assert "DATA COVERAGE: standard." in msg
    assert "transcript" not in msg.split("BUSINESS DESCRIPTION")[0]


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


def _validate(output, has_filing_digest, material_absent=None, anomalies=None, stale_data=None, valid_news_ids=None):
    return _validate_with_caveats(
        output,
        has_filing_digest=has_filing_digest,
        material_absent=material_absent or [],
        anomalies=anomalies or [],
        stale_data=stale_data or [],
        valid_news_ids=valid_news_ids,
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
    """Default fixture: nothing is flagged."""
    assert _anomalies(_bundle()) == []


def test_a_buyback_with_insider_selling_is_not_a_data_anomaly():
    """It flagged 15 of 59 real runs and forced data quality to "low" ("unreliable" to Pass 2); two true facts are not a
    contradiction in the data, and both reach the model in SIGNALS."""
    bundle = _bundle(research_sources=SimpleNamespace(
        filing_digests=[_filing_digest()], news_items=[_news_item()], peer_blocks=[_peer_block()],
        management_signals=_management_signals(buyback="active"),
        missing_sources_list=[], has_filing_digest=True, transcript_excerpts=[],
    ))
    assert _anomalies(bundle) == []


# ---------- _stale_data (86bbummwp Tier 2) ----------


def test_stale_data_empty_for_default_fixture():
    assert _stale_data(_bundle()) == []


def test_stale_data_flags_stale_filing_past_normal_reporting_cycle():
    bundle = _bundle()
    bundle.research_sources.latest_filing_age_days = 500
    assert "filings" in _stale_data(bundle)


def test_stale_data_an_annual_report_months_old_is_not_flagged():
    """The digests come from the annual report, so 228 days (KO), 235 (ENB.TO) and 218 (BAM.TO) are normal; the old
    120 day limit made all three "low" quality for Pass 2 on every run."""
    bundle = _bundle()
    for age in (80, 228, 365):
        bundle.research_sources.latest_filing_age_days = age
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


def test_the_researcher_payload_states_the_currency_conversion_and_that_filing_amounts_stay_in_the_filing_currency():
    bundle = _bundle(currency_mismatch={"financials_currency": "USD", "quote_currency": "CAD", "converted": True, "usd_cad": 1.4243})
    msg, _ = build_user_message(bundle)
    assert "NOTE: statements are reported in USD and shown here converted to CAD at 1.4243 CAD per USD (Bank of Canada)." in msg
    assert "Amounts quoted inside filing excerpts are in USD." in msg


def test_the_researcher_payload_has_no_currency_note_when_there_is_no_mismatch():
    msg, _ = build_user_message(_bundle())
    assert "NOTE: statements are reported" not in msg


def test_the_business_digest_is_sent_once_and_labelled_for_citation():
    """The Business digest used to be sent twice (BUSINESS DESCRIPTION, then FILING:Business under FILING HIGHLIGHTS)."""
    msg, _ = build_user_message(_bundle())
    assert msg.count("Business digest content.") == 1
    assert "BUSINESS DESCRIPTION (cite as FILING:Business):" in msg
    assert "FILING:MDA: MDA digest content." in msg and "FILING:Business:" not in msg


def test_price_and_market_cap_are_rounded():
    """TD.TO printed 168.5800018310547 CAD and a market cap of 275293962659.2229."""
    bundle = _bundle(price_info={"current_price": 168.5800018310547, "market_cap": 275293962659.2229,
                                  "currency": "CAD", "high_52w": 175.3300018310547, "low_52w": 109.5})
    msg, _ = build_user_message(bundle)
    assert "Current: 168.58 CAD | 52w High: 175.33 | 52w Low: 109.50" in msg
    assert "Market Cap: 275.29B CAD" in msg


def test_insider_activity_is_the_sized_summary_the_sentiment_agent_reads():
    """Every one of 54 real outputs saw a bare "selling"; management was called concerning in 41 of them."""
    bundle = _bundle(price_info={"current_price": 1.0, "market_cap": 1e11, "currency": "CAD", "high_52w": 2.0, "low_52w": 1.0})
    bundle.insider_activity = {"transactions": [
        {"date": datetime.now(UTC).date().isoformat(), "transaction_type": "sale", "value": 1.0e5, "is_issuer": False},
    ], "value_currency": "CAD"}
    msg, _ = build_user_message(bundle)
    assert "none above the notable threshold" in msg and "routine" in msg


def test_the_dividend_record_is_shown_not_only_the_yield():
    """The dividend_compounder archetype is defined by the record, which was in the bundle and never shown."""
    bundle = _bundle(dividend_info={"dividend_yield": 0.0242, "payout_ratio": 0.613, "dividend_growth_5yr": 0.0456,
                                    "consecutive_years_paid": 7})
    msg, _ = build_user_message(bundle)
    assert "Yield: 2.42% | Payout ratio: 61.30% | 5-year dividend growth: 4.6% a year | paid in each of the last 7 years" in msg


# ---------- the merge step between the model's answer and storage ----------


def _answer(**overrides):
    out = {
        "assessment_summary": "COMPANY_X is ahead of PEER_1_COMPANY.",
        "caveats": ["TICKER_X insiders sold shares."],
        "narrative": "COMPANY_X (TICKER_X) leads PEER_1_COMPANY on scale (N1).",
        "structured_data": {"recent_developments": [
            {"news_id": "N1", "significance": "high", "sentiment": "positive"}]},
    }
    out.update(overrides)
    return out


def test_a_bare_peer_token_the_prompt_asks_for_is_replaced_in_the_merged_output():
    """The prompt says to cite `PEER_{num}`; all three newest real runs sent "PEER_1 and PEER_2" on to Bull and Bear."""
    from agents.pass1_stock_researcher import merge_researcher_output

    answer = _answer()
    answer["narrative"] = "Leads PEER_1 on scale (PEER_1)."

    merged = merge_researcher_output(answer, _bundle())

    assert merged["narrative"] == "Leads Peer One Inc. on scale (Peer One Inc.)."


def test_the_real_names_replace_the_anonymization_tokens_everywhere():
    """9 of 61 real outputs carried COMPANY_X or PEER_1_COMPANY into Pass 2: deanonymize_text_fields was never called."""
    from agents.pass1_stock_researcher import merge_researcher_output

    merged = merge_researcher_output(_answer(), _bundle())

    assert merged["narrative"] == "Shopify Inc (SHOP.TO) leads Peer One Inc. on scale (N1)."
    assert merged["assessment_summary"] == "Shopify Inc is ahead of Peer One Inc.."
    assert merged["caveats"] == ["SHOP.TO insiders sold shares."]


def test_a_cited_news_id_is_stored_with_its_headline_date_and_source():
    from agents.pass1_stock_researcher import merge_researcher_output

    dev = merge_researcher_output(_answer(), _bundle())["structured_data"]["recent_developments"]

    assert dev == [{"news_id": "N1", "event": "Company announces new product", "date": "2026-09-20T00:00:00+00:00",
                    "source": "Reuters", "significance": "high", "sentiment": "positive"}]


def test_the_dual_class_caveat_is_added_after_the_names_are_restored():
    from agents.pass1_stock_researcher import merge_researcher_output

    bundle = _bundle()
    bundle.research_sources.dual_class_flag = True
    caveats = merge_researcher_output(_answer(), bundle)["caveats"]

    assert caveats[0] == "SHOP.TO insiders sold shares."
    assert caveats[1].startswith("Dual-class share structure")
    assert len(merge_researcher_output(_answer(), _bundle())["caveats"]) == 1  # no flag, nothing added


def test_a_news_id_the_payload_does_not_list_is_dropped_not_raised_on():
    """Only reachable when the answer failed validation on every attempt."""
    from agents.pass1_stock_researcher import merge_researcher_output

    answer = _answer(structured_data={"recent_developments": [
        {"news_id": "N1", "significance": "high", "sentiment": "positive"},
        {"news_id": "N99", "significance": "low", "sentiment": "neutral"}]})

    dev = merge_researcher_output(answer, _bundle())["structured_data"]["recent_developments"]

    assert [d["news_id"] for d in dev] == ["N1"]


def test_the_model_is_made_to_retry_a_news_id_that_is_not_in_the_payload():
    ok, errors = _validate(_answer(structured_data={"recent_developments": [{"news_id": "N99"}]}), True,
                           valid_news_ids={"N1"})
    assert not ok and any("recent_developments[0].news_id" in e for e in errors)


async def test_the_runner_returns_the_merged_answer(monkeypatch):
    """Through the real run(): what comes back is what gets stored."""
    from agents.pass1_stock_researcher import StockResearcherRunner

    runner = StockResearcherRunner()

    async def fake_call(self, system, user, validator, **kwargs):
        # the validator the runner built must accept the id the payload lists and reject an invented one
        assert all("news_id" not in e for e in validator(_answer())[1])
        assert any("news_id" in e for e in validator(_answer(structured_data={"recent_developments": [{"news_id": "N99"}]}))[1])
        return _answer(), []

    monkeypatch.setattr(StockResearcherRunner, "call_with_validation", fake_call)

    out, errors = await runner.run(_bundle(stock=SimpleNamespace(ticker="SHOP.TO", currency="CAD", exchange="TSX")))

    assert errors == []
    assert "COMPANY_X" not in out["narrative"] and "TICKER_X" not in out["narrative"]
    assert out["structured_data"]["recent_developments"][0]["event"] == "Company announces new product"


def test_with_no_digest_the_business_description_is_the_labelled_company_profile():
    """A name with no digest (pure-Canadian, a fetch failure) had no description of what the company does at all."""
    bundle = _bundle()
    bundle.research_sources.filing_digests = []
    bundle.research_sources.has_filing_digest = False
    bundle.research_sources.missing_sources_list = ["filing_digests"]
    bundle.research_sources.business_profile = "COMPANY_X makes anvils."

    msg, presence = build_user_message(bundle)

    assert "BUSINESS DESCRIPTION (a third-party company profile, not a filing; cite as PROFILE:Business):\nCOMPANY_X makes anvils." in msg
    assert "(the business description is a third-party company profile)" in msg
    assert presence["business_description"] is True and presence["filing_highlights"] is False


def test_with_neither_digest_nor_profile_the_description_says_so():
    bundle = _bundle()
    bundle.research_sources.filing_digests = []
    msg, presence = build_user_message(bundle)
    assert "N/A — no filing digest available for this name." in msg
    assert presence["business_description"] is False


def test_a_missing_company_name_falls_back_to_the_ticker_not_a_gap():
    from agents.pass1_stock_researcher import merge_researcher_output

    bundle = _bundle(company_info={"name": None, "sector": "Technology"})

    assert merge_researcher_output(_answer(), bundle)["narrative"].startswith("SHOP.TO (SHOP.TO) leads")


def test_a_no_moat_answer_passes_the_runners_validator_and_is_stored_with_a_trend_of_none():
    """BAM.TO 2026-10-09: durability 'none' with a trend the enum rejected; through the runner's own validator, and the same dict
    is what the retry loop returns and stores."""
    out = _valid_stock_researcher_output()
    out["structured_data"]["moat_assessment"] = {"overall_moat_durability": "none", "moat_trend": "stable", "moats": []}
    passed, errors = _validate(out, has_filing_digest=True)
    assert passed, errors
    assert out["structured_data"]["moat_assessment"]["moat_trend"] == "none"


def test_a_trend_of_none_beside_a_real_moat_is_rejected_by_the_runners_validator():
    out = _valid_stock_researcher_output()
    out["structured_data"]["moat_assessment"] = {"overall_moat_durability": "moderate", "moat_trend": "none", "moats": []}
    passed, errors = _validate(out, has_filing_digest=True)
    assert not passed and any("moat_trend" in e for e in errors)
