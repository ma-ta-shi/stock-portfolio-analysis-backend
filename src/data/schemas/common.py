"""Shared submodels used across multiple data-pipeline contracts (ClickUp
86bawp7yx). Exact record shapes come from docs/technical/data-pipeline.md §4
— these promote what were plain `dict` fields in that doc's original
@dataclass sketch into real validated models, per this ticket's decision to
build the contracts as Pydantic v2, not dataclasses.
"""

from datetime import date, datetime
from typing import Literal, Protocol
from uuid import UUID

from data.schemas.base import ContractModel


class NewsItem(ContractModel):
    """One news item, with a stable N{num} id assigned globally in
    DataPipeline.prepare() — the LLM references items by this id in its
    own output, and the orchestrator hydrates the full record back in at
    merge time (data-pipeline.md §4, ResearchSourcesBundle).

    `date` is a full datetime, not a date (unlike CBCommentaryItem below)
    — real news articles carry a publish time, not just a publish day
    (confirmed live this session: openbb-tmx's own news results include
    tz-aware timestamps), and recency-within-a-day matters for news
    freshness scoring in a way it doesn't for a central bank statement."""

    id: str
    date: datetime
    headline: str
    source: str
    quality_tier: Literal["primary", "secondary", "low"]


class FilingDigest(ContractModel):
    """One filing-section digest (<=250 tokens). Note: a RiskFactors
    section existed in an earlier Researcher version but was removed in
    v1.3 — Business/MDA are the only two sections produced today."""

    section: Literal["Business", "MDA"]
    content: str
    token_count: int


class TranscriptExcerpt(ContractModel):
    """One earnings-call excerpt. <=400 tokens total per quarter, 2 most
    recent quarters kept.

    `type` includes "mgmt" (added 2026-08-07, contract-vs-consumer audit)
    alongside the original "guidance"/"qa" — the live Stock Researcher
    Agent Prompt.md's payload template has three distinct excerpt slots
    per quarter (`{transcript_mgmt_excerpt}`, `{transcript_guidance_excerpt}`,
    `{transcript_qa_excerpt}`), not two; "mgmt" covers prepared management
    commentary distinct from forward-looking guidance."""

    quarter: str
    type: Literal["mgmt", "guidance", "qa"]
    content: str
    token_count: int


class PeerBlock(ContractModel):
    """One peer-company block (<=200 tokens, 2-3 peers per bundle)."""

    peer_id: str
    content: str
    token_count: int


class CBCommentaryItem(ContractModel):
    """One central-bank commentary headline, timeline-scoped to the
    analysis run's macro lookback window.

    `date` is day-only, not a datetime (unlike NewsItem above) — a rate
    decision or statement is naturally a day-level event, and neither
    FRED's nor BoC Valet's series data carries intraday timestamps."""

    date: date
    headline: str


class StockLike(Protocol):
    """Duck-typed stand-in for api.tables.stock.Stock — avoids importing
    the SQLAlchemy model (and its api.database side effects) into the data
    layer. Same pattern as data.providers.router.StockLike, extended here
    with the extra fields StockRef.from_stock() needs; not reused directly
    from router.py to keep data.schemas independent of data.providers."""

    stock_id: UUID
    canonical_ticker: str
    primary_exchange: str
    currency: str


class StockRef(ContractModel):
    """Lightweight Pydantic snapshot of the fields agents actually need
    from the Stock ORM model (docs/technical/database-design.md), built at
    DataBundle-assembly time rather than embedding the live SQLAlchemy
    object directly.

    Deliberately excludes Stock's lazy-loaded relationships (`aliases`,
    `analysis_runs`) — nothing downstream needs them, and embedding the
    live ORM object via arbitrary_types_allowed=True would both skip
    validation on the field most worth validating and risk triggering
    lazy-loading outside an active DB session (ClickUp 86bawp7yx decision).

    `stock_id` is one field beyond 86bawp7yx's own literal suggestion
    (`ticker`, `currency`, `exchange`) — added so anything downstream that
    needs to correlate a DataBundle back to its canonical DB row (cache
    keys, historical prediction lookups) doesn't have to resolve the
    ticker back to an id itself. Flagging this as a deliberate, disclosed
    addition, not an unstated scope change."""

    stock_id: UUID
    ticker: str
    currency: str
    exchange: str

    @classmethod
    def from_stock(cls, stock: StockLike) -> "StockRef":
        return cls(
            stock_id=stock.stock_id,
            ticker=stock.canonical_ticker,
            currency=stock.currency,
            exchange=stock.primary_exchange,
        )
