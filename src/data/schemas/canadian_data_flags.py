"""CanadianDataFlags contract model (ClickUp 86bawp80f).

Full field spec: docs/technical/data-pipeline.md §4, "CanadianDataFlags".

Migrated 2026-09-13 (ClickUp 86bawptx4, assembly): dropped sentiment_source,
has_transcript, and transcript_source - see the class docstring for why.
"""

from pydantic import NonNegativeInt

from data.schemas.base import ContractModel


class CanadianDataFlags(ContractModel):
    """Per-stock data quality flags for TSX/TSXV stocks.
    Populated by DataPipeline.prepare(); attached to DataBundle.
    None for US stocks - check DataBundle.canadian_data_flags is not None
    before reading. That None-for-US rule is enforced at the DataBundle
    level (ClickUp 86bawp863), not self-enforced by this model.

    Type definition only - these fields feed the mechanical `data_coverage`/
    `stale_data` flags (docs/technical/pass1-confidence-model.md, decision
    D6), not a reliability score. An earlier version of this docstring
    described the exact pre-D6 formula (sentiment_source == "local_llm"
    capping reliability at 70, sedar_filing_available suppressing a
    no-transcript penalty, analyst_count < 3 triggering a deduction) -
    that whole 0-100 reliability_score machinery was removed by D6, along
    with every compute_base_reliability_*() function. The facts these
    fields carry are still real and still consumed, just by the mechanical
    flags now, not the deleted formula.

    sentiment_source deliberately removed, not overlooked (86bawptx4):
    it would always hold the exact same value as DataBundle.sentiment_source
    for a CA stock - one precompute/sentiment.py call scores both markets,
    no CA-specific branch exists. Duplicating a fact across two contracts
    needs a real reason (a genuine second computation path, or a
    performance/access-pattern need) - neither applies here. Read
    DataBundle.sentiment_source instead. If a CA-specific sentiment path
    is ever built, that's the time to reintroduce a dedicated field, not
    before.

    has_transcript/transcript_source also removed (86bawptx4): grepped
    every current agent prompt for both field names and found zero
    references anywhere, including the Stock Researcher prompt an earlier
    version of this ticket assumed would use them. Earnings-call
    transcripts are permanently out of scope on any free tier, either
    market (data-inventory-audit.md, decision D3) - nothing reads these
    today and nothing will under the current design.
    """

    analyst_count: NonNegativeInt
    news_article_count: NonNegativeInt
    news_sources: list[str]  # e.g. ["openbb_tmx", "globenewswire", "newswire_ca"]
    sedar_filing_available: bool
    statcan_available: bool
