"""Data-pipeline contract models (ClickUp 86bawp7yx). See
docs/technical/data-pipeline.md §4 for the full design and
docs/decision-log.md for the module-layout/Pydantic-conventions decision
this package implements.
"""

from data.schemas.base import ContractModel
from data.schemas.canadian_data_flags import CanadianDataFlags
from data.schemas.common import (
    CBCommentaryItem,
    FilingDigest,
    NewsItem,
    PeerBlock,
    StockRef,
    TranscriptExcerpt,
)
from data.schemas.context import AnalysisContext

__all__ = [
    "ContractModel",
    "NewsItem",
    "FilingDigest",
    "TranscriptExcerpt",
    "PeerBlock",
    "CBCommentaryItem",
    "StockRef",
    "AnalysisContext",
    "CanadianDataFlags",
]
