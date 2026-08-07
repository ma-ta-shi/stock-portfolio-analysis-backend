import json
from datetime import date, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from data.schemas.common import (
    CBCommentaryItem,
    FilingDigest,
    NewsItem,
    PeerBlock,
    StockRef,
    TranscriptExcerpt,
)


# ---------- NewsItem ----------


def test_news_item_valid_construction():
    item = NewsItem(
        id="N1",
        date=datetime(2026, 8, 4),
        headline="Bank raises rates",
        source="Reuters",
        quality_tier="primary",
    )
    assert item.id == "N1"


def test_news_item_rejects_invalid_quality_tier():
    with pytest.raises(ValidationError):
        NewsItem(
            id="N1",
            date=datetime(2026, 8, 4),
            headline="Bank raises rates",
            source="Reuters",
            quality_tier="not_a_real_tier",
        )


# ---------- FilingDigest ----------


def test_filing_digest_valid_construction():
    digest = FilingDigest(section="MDA", content="...", token_count=200)
    assert digest.section == "MDA"


def test_filing_digest_rejects_riskfactors_section():
    """RiskFactors was removed in Researcher v1.3 — Business/MDA only."""
    with pytest.raises(ValidationError):
        FilingDigest(section="RiskFactors", content="...", token_count=100)


# ---------- TranscriptExcerpt ----------


def test_transcript_excerpt_valid_construction():
    excerpt = TranscriptExcerpt(quarter="Q2-2026", type="qa", content="...", token_count=150)
    assert excerpt.type == "qa"


def test_transcript_excerpt_rejects_invalid_type():
    with pytest.raises(ValidationError):
        TranscriptExcerpt(quarter="Q2-2026", type="summary", content="...", token_count=150)


# ---------- PeerBlock ----------


def test_peer_block_valid_construction():
    block = PeerBlock(peer_id="PEER_1", content="...", token_count=100)
    assert block.peer_id == "PEER_1"


# ---------- CBCommentaryItem ----------


def test_cb_commentary_item_valid_construction():
    item = CBCommentaryItem(date=date(2026, 8, 4), headline="Fed holds rates steady")
    assert item.headline == "Fed holds rates steady"


# ---------- StockRef ----------


def test_stock_ref_from_stock_maps_fields_correctly():
    fake_stock = SimpleNamespace(
        stock_id=uuid4(),
        canonical_ticker="RY.TO",
        primary_exchange="TSX",
        currency="CAD",
    )
    ref = StockRef.from_stock(fake_stock)
    assert ref.stock_id == fake_stock.stock_id
    assert ref.ticker == "RY.TO"
    assert ref.exchange == "TSX"
    assert ref.currency == "CAD"


def test_stock_ref_does_not_expose_lazy_loaded_relationships():
    """Deliberately excludes aliases/analysis_runs — StockRef must not
    accept them even if a caller tries to pass them through."""
    with pytest.raises(ValidationError):
        StockRef(
            stock_id=uuid4(),
            ticker="RY.TO",
            currency="CAD",
            exchange="TSX",
            aliases=["ROYAL BANK"],
        )


def test_stock_ref_is_frozen():
    ref = StockRef(stock_id=uuid4(), ticker="RY.TO", currency="CAD", exchange="TSX")
    with pytest.raises(ValidationError):
        ref.ticker = "SHOP.TO"


# ---------- model_dump() round-trip (ClickUp 86bawp88h) ----------
# Confirms these submodels — embedded in every other contract in this
# package — serialize to plain dicts/JSON-primitives and rebuild into an
# identical instance, which matters once any of these are cached or logged
# via structlog rather than only ever passed in-process.


def test_news_item_round_trips_through_model_dump():
    item = NewsItem(
        id="N1", date=datetime(2026, 8, 4, 9, 30), headline="Bank raises rates", source="Reuters", quality_tier="primary"
    )
    assert NewsItem.model_validate(item.model_dump()) == item


def test_news_item_json_mode_dump_is_json_serializable_and_round_trips():
    item = NewsItem(
        id="N1", date=datetime(2026, 8, 4, 9, 30), headline="Bank raises rates", source="Reuters", quality_tier="primary"
    )
    dumped = item.model_dump(mode="json")
    json.dumps(dumped)  # raises if anything isn't JSON-primitive (e.g. a raw datetime)
    assert NewsItem.model_validate(dumped) == item


def test_filing_digest_round_trips_through_model_dump():
    digest = FilingDigest(section="MDA", content="...", token_count=200)
    assert FilingDigest.model_validate(digest.model_dump()) == digest


def test_transcript_excerpt_round_trips_through_model_dump():
    excerpt = TranscriptExcerpt(quarter="Q2-2026", type="qa", content="...", token_count=150)
    assert TranscriptExcerpt.model_validate(excerpt.model_dump()) == excerpt


def test_peer_block_round_trips_through_model_dump():
    block = PeerBlock(peer_id="PEER_1", content="...", token_count=100)
    assert PeerBlock.model_validate(block.model_dump()) == block


def test_cb_commentary_item_round_trips_through_model_dump():
    item = CBCommentaryItem(date=date(2026, 8, 4), headline="Fed holds rates steady")
    assert CBCommentaryItem.model_validate(item.model_dump()) == item


def test_cb_commentary_item_json_mode_dump_is_json_serializable():
    item = CBCommentaryItem(date=date(2026, 8, 4), headline="Fed holds rates steady")
    dumped = item.model_dump(mode="json")
    json.dumps(dumped)
    assert dumped["date"] == "2026-08-04"


def test_stock_ref_round_trips_through_model_dump():
    ref = StockRef(stock_id=uuid4(), ticker="RY.TO", currency="CAD", exchange="TSX")
    assert StockRef.model_validate(ref.model_dump()) == ref


def test_stock_ref_json_mode_dump_serializes_uuid_as_str():
    ref = StockRef(stock_id=uuid4(), ticker="RY.TO", currency="CAD", exchange="TSX")
    dumped = ref.model_dump(mode="json")
    json.dumps(dumped)
    assert dumped["stock_id"] == str(ref.stock_id)
    assert StockRef.model_validate(dumped) == ref
