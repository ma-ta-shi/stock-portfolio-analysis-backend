"""DataBundle contract model — NOT YET IMPLEMENTED.

Scaffolded by ClickUp 86bawp7yx (contracts module layout decision) so this
model has a settled location and base class to build against. Actual
implementation is ClickUp 86bawp863 ("Define DataBundle contract model +
get_benchmark()").

Full field spec: docs/technical/data-pipeline.md §4, "DataBundle".
Inherit from data.schemas.base.ContractModel (frozen=True, extra="forbid").
`stock` field should be typed as data.schemas.common.StockRef, not the
Stock ORM model directly. `context` field should be typed as
data.schemas.context.AnalysisContext. `canadian_data_flags`,
`macro_sources`, `research_sources` fields depend on their own
not-yet-built sibling contract modules in this package.
"""
