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
