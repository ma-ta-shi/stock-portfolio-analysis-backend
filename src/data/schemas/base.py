"""Shared Pydantic v2 base for every data-pipeline contract model (ClickUp
86bawp7yx). Backs CanadianDataFlags, MacroSourcesBundle, ResearchSourcesBundle,
DataBundle, and AnalysisContext — see docs/technical/data-pipeline.md §4.
"""

from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    """Base for every data-pipeline contract model.

    frozen=True — these are immutable snapshots handed to agents once per
    analysis run, never mutated mid-run.
    extra="forbid" — catches a precompute module silently adding or
    renaming a field before it reaches an agent prompt, rather than the
    mismatch silently passing through as an ignored extra key.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
