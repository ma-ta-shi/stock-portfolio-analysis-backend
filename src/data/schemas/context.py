"""AnalysisContext (ClickUp 86bb66wc8) — immutable context threaded through
DataPipeline.prepare() and every agent call for one analysis run.

Placed here, not in a new src/orchestrator/ package: 86bb66wc8 explicitly
defers file placement to this ticket's (86bawp7yx) module-layout decision,
and src/orchestrator/ doesn't exist yet — creating a whole new top-level
package for one 2-field model would be premature.

Field values verified 2026-08-04 against src/api/tables/analysis_runs.py:13-14
(the ORM columns these mirror) rather than trusted from
docs/technical/phase1-archive/orchestration-engine-phase1.md's AnalysisContext
sketch, which uses different, stale values (timeline "1mo"/"3mo"/"12mo" and
account_type "taxable" instead of "trading") — that archived doc predates a
rename and doesn't match CLAUDE.md's own current CLI example
(`--account tfsa --timeline medium_term`) or the live schema.
"""

from typing import Literal

from data.schemas.base import ContractModel


class AnalysisContext(ContractModel):
    account_type: Literal["tfsa", "rrsp", "trading", "general"]
    timeline: Literal["short_term", "medium_term", "long_term"]
