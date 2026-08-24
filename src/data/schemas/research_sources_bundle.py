"""ResearchSourcesBundle contract model — NOT YET IMPLEMENTED.

Scaffolded by ClickUp 86bawp7yx (contracts module layout decision) so this
model has a settled location and base class to build against. Actual
implementation is ClickUp 86bawp83r ("Define ResearchSourcesBundle contract
model").

Full field spec: docs/technical/data-pipeline.md §4, "ResearchSourcesBundle".
Inherit from data.schemas.base.ContractModel (frozen=True, extra="forbid").
Reuses NewsItem, FilingDigest, TranscriptExcerpt, PeerBlock from
data.schemas.common. Populated per the precompute/research_sources.py
contract (also not yet built — this ticket's own crosslisting/edgartools
work this session is the raw data source it will read from).
"""
