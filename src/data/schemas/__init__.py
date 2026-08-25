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
from data.schemas.data_bundle import DataBundle, get_benchmark
from data.schemas.macro_sources_bundle import MacroSourcesBundle
from data.schemas.research_sources_bundle import ManagementSignals, ResearchSourcesBundle

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
    "MacroSourcesBundle",
    "ManagementSignals",
    "ResearchSourcesBundle",
    "DataBundle",
    "get_benchmark",
]
