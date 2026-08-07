import json

import pytest
from pydantic import ValidationError

from data.schemas.common import FilingDigest, NewsItem, PeerBlock, TranscriptExcerpt
from data.schemas.research_sources_bundle import ManagementSignals, ResearchSourcesBundle


def _signals(**overrides) -> ManagementSignals:
    defaults = dict(
        c_suite_changes_12mo=False,
        changes_detail="",
        insider_net_direction_90d="neutral",
        buyback_activity="",
        dividend_activity="",
    )
    return ManagementSignals(**{**defaults, **overrides})


def _news_item(item_id: str = "N1") -> NewsItem:
    return NewsItem(
        id=item_id,
        date="2026-08-01T09:00:00",
        headline="Headline",
        source="Reuters",
        quality_tier="primary",
    )


def _transcript() -> TranscriptExcerpt:
    return TranscriptExcerpt(quarter="Q2-2026", type="qa", content="...", token_count=100)


def _bundle(**overrides) -> ResearchSourcesBundle:
    defaults = dict(
        filing_digests=[FilingDigest(section="MDA", content="...", token_count=200)],
        transcript_excerpts=[_transcript()],
        news_items=[_news_item()],
        peer_blocks=[PeerBlock(peer_id="PEER_1", content="...", token_count=100)],
        management_signals=_signals(),
        dual_class_flag=False,
        cik_verified=True,
        sedar_filing_available=False,
        latest_filing_age_days=10,
        latest_transcript_age_days=30,
        latest_news_age_days=1,
        transcript_count=1,
        news_item_count=1,
    )
    return ResearchSourcesBundle(**{**defaults, **overrides})


# ---------- ManagementSignals ----------


def test_management_signals_valid_construction():
    signals = _signals()
    assert signals.insider_net_direction_90d == "neutral"


def test_management_signals_rejects_invalid_insider_direction():
    with pytest.raises(ValidationError):
        _signals(insider_net_direction_90d="hodling")


def test_management_signals_accepts_none_insider_direction():
    """Real gap avoided proactively: None means "couldn't determine,"
    distinct from "neutral" (a genuine, resolved zero-net reading) —
    conflating the two under one enum would lose information a reliability
    scorer needs."""
    signals = _signals(insider_net_direction_90d=None)
    assert signals.insider_net_direction_90d is None


def test_management_signals_is_frozen():
    signals = _signals()
    with pytest.raises(ValidationError):
        signals.c_suite_changes_12mo = True


def test_management_signals_forbids_extra_fields():
    with pytest.raises(ValidationError):
        _signals(unexpected="field")


# ---------- ResearchSourcesBundle: valid construction ----------


def test_valid_construction():
    bundle = _bundle()
    assert bundle.transcript_count == 1
    assert bundle.management_signals.insider_net_direction_90d == "neutral"


# ---------- count/list consistency ----------


def test_transcript_count_must_match_list_length():
    with pytest.raises(ValidationError):
        _bundle(transcript_excerpts=[_transcript()], transcript_count=2)


def test_news_item_count_must_match_list_length():
    with pytest.raises(ValidationError):
        _bundle(news_items=[_news_item()], news_item_count=0)


def test_empty_lists_with_matching_zero_counts_is_valid():
    bundle = _bundle(
        transcript_excerpts=[],
        transcript_count=0,
        news_items=[],
        news_item_count=0,
        latest_transcript_age_days=None,
        latest_news_age_days=None,
    )
    assert bundle.transcript_count == 0


# ---------- graceful degradation: age fields optional when their
# underlying collection is empty ----------


def test_accepts_none_latest_filing_age_when_no_filings():
    bundle = _bundle(filing_digests=[], latest_filing_age_days=None)
    assert bundle.latest_filing_age_days is None


def test_accepts_none_latest_transcript_age_when_no_transcripts():
    bundle = _bundle(transcript_excerpts=[], transcript_count=0, latest_transcript_age_days=None)
    assert bundle.latest_transcript_age_days is None


def test_accepts_none_latest_news_age_when_no_news():
    bundle = _bundle(news_items=[], news_item_count=0, latest_news_age_days=None)
    assert bundle.latest_news_age_days is None


def test_rejects_negative_age():
    with pytest.raises(ValidationError):
        _bundle(latest_filing_age_days=-1)


def test_rejects_negative_count():
    with pytest.raises(ValidationError):
        _bundle(transcript_excerpts=[], transcript_count=-1)


# ---------- age/list-emptiness consistency (real gap caught on review —
# previously unenforced) ----------


def test_rejects_real_age_paired_with_empty_filing_digests():
    with pytest.raises(ValidationError):
        _bundle(filing_digests=[], latest_filing_age_days=5)


def test_rejects_none_age_paired_with_nonempty_filing_digests():
    """Default filing_digests is non-empty — passing a None age for it
    must be rejected, not silently accepted."""
    with pytest.raises(ValidationError):
        _bundle(latest_filing_age_days=None)


def test_rejects_real_age_paired_with_empty_transcripts():
    with pytest.raises(ValidationError):
        _bundle(transcript_excerpts=[], transcript_count=0, latest_transcript_age_days=5)


def test_rejects_none_age_paired_with_nonempty_transcripts():
    with pytest.raises(ValidationError):
        _bundle(latest_transcript_age_days=None)


def test_rejects_real_age_paired_with_empty_news_items():
    with pytest.raises(ValidationError):
        _bundle(news_items=[], news_item_count=0, latest_news_age_days=5)


def test_rejects_none_age_paired_with_nonempty_news_items():
    with pytest.raises(ValidationError):
        _bundle(latest_news_age_days=None)


# ---------- base behavior ----------


def test_is_frozen():
    bundle = _bundle()
    with pytest.raises(ValidationError):
        bundle.dual_class_flag = True


def test_forbids_extra_fields():
    with pytest.raises(ValidationError):
        _bundle(analyst_reports_summary={"legacy": True})


# ---------- model_dump() round-trip (ClickUp 86bawp88h) ----------


def test_management_signals_round_trips_through_model_dump():
    signals = _signals()
    assert ManagementSignals.model_validate(signals.model_dump()) == signals


def test_management_signals_round_trips_with_none_insider_direction():
    signals = _signals(insider_net_direction_90d=None)
    assert ManagementSignals.model_validate(signals.model_dump()) == signals


def test_bundle_round_trips_through_model_dump():
    """The interesting case: confirms all four nested submodel lists
    (filing_digests, transcript_excerpts, news_items, peer_blocks) plus the
    nested ManagementSignals survive dump as real structured data, not
    collapsed into opaque dicts, and reload into an identical bundle."""
    bundle = _bundle()
    dumped = bundle.model_dump()
    assert dumped["filing_digests"] == [{"section": "MDA", "content": "...", "token_count": 200}]
    assert dumped["management_signals"]["insider_net_direction_90d"] == "neutral"
    assert ResearchSourcesBundle.model_validate(dumped) == bundle


def test_bundle_round_trips_with_all_empty_lists():
    bundle = _bundle(
        filing_digests=[],
        transcript_excerpts=[],
        news_items=[],
        peer_blocks=[],
        transcript_count=0,
        news_item_count=0,
        latest_filing_age_days=None,
        latest_transcript_age_days=None,
        latest_news_age_days=None,
    )
    assert ResearchSourcesBundle.model_validate(bundle.model_dump()) == bundle


def test_bundle_json_mode_dump_is_json_serializable_and_round_trips():
    bundle = _bundle()
    dumped = bundle.model_dump(mode="json")
    json.dumps(dumped)  # raises if the nested NewsItem.date isn't JSON-primitive
    assert isinstance(dumped["news_items"][0]["date"], str)
    assert ResearchSourcesBundle.model_validate(dumped) == bundle
