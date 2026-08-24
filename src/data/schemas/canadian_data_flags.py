"""CanadianDataFlags contract model (ClickUp 86bawp80f).

Full field spec: docs/technical/data-pipeline.md §4, "CanadianDataFlags".
"""

from typing import Literal

from pydantic import NonNegativeInt, model_validator

from data.schemas.base import ContractModel


class CanadianDataFlags(ContractModel):
    """Per-stock data quality flags for TSX/TSXV stocks.
    Populated by DataPipeline.prepare(); attached to DataBundle.
    None for US stocks — check DataBundle.canadian_data_flags is not None
    before reading. That None-for-US rule is enforced at the DataBundle
    level (ClickUp 86bawp863), not self-enforced by this model.

    Type definition only — the reliability-scoring behavior that consumes
    these fields (sentiment_source == "local_llm" capping reliability at
    70, sedar_filing_available suppressing the no-transcript penalty,
    statcan_available giving a small positive adjustment, analyst_count < 3
    triggering a deduction) lives in the reliability scorer, not here. The
    has_transcript/transcript_source consistency check below is a basic
    data-validity invariant, not scoring logic, so it stays in scope.
    """

    sentiment_source: Literal["finnhub", "local_llm", "keyword"]
    has_transcript: bool
    transcript_source: Literal["finnhub"] | None
    analyst_count: NonNegativeInt
    news_article_count: NonNegativeInt
    news_sources: list[str]  # e.g. ["openbb_tmx", "globenewswire", "newswire_ca"]
    sedar_filing_available: bool
    statcan_available: bool

    @model_validator(mode="after")
    def _check_transcript_consistency(self) -> "CanadianDataFlags":
        if self.has_transcript and self.transcript_source is None:
            raise ValueError("has_transcript=True requires a non-None transcript_source")
        if not self.has_transcript and self.transcript_source is not None:
            raise ValueError("transcript_source must be None when has_transcript=False")
        return self
