from sqlalchemy import Boolean, String, JSON, Text, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional

class RunQualitySummary(Base):
    """1:1 with `analysis_runs` (86bbwachy Phase 5) -- one row per orchestrator run() invocation,
    written by AnalysisOrchestrator._write_run_quality_summary() in a try/finally wrapper around
    the pipeline's own run() body, so it fires on every one of that method's exit paths (success
    or any of its several failure returns/raises), not just the happy path.

    Deliberately does NOT duplicate exchange/currency/instrument_type/market_cap_bucket/
    disagreement_score/disagreement_class -- all already live directly on `analysis_runs`, which
    this table is 1:1 with; a consumer already has that row for free via run_id. stock_outlook/
    overall_confidence ARE duplicated here from `recommendations` -- a genuine one-join-avoided
    denormalization, since that's a different table.
    """
    __tablename__ = "run_quality_summary"
    summary_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("analysis_runs.run_id"), unique=True)

    wall_clock_ms: Mapped[int]  # datetime.now(UTC) - analysis_runs.triggered_at, captured fresh at
    # write time -- not analysis_runs.completed_at, which is only ever set on the one success path
    total_llm_ms: Mapped[int] = mapped_column(default=0)  # SUM(llm_calls.latency_ms) for this run
    slowest_call_ms: Mapped[Optional[int]]  # MAX(llm_calls.latency_ms) -- None when zero calls ran
    # (undefined for an empty set, unlike the SUM/COUNT fields below which are well-defined as 0)

    total_calls: Mapped[int] = mapped_column(default=0)
    retry_calls: Mapped[int] = mapped_column(default=0)  # llm_calls.attempt > 1 -- no dedicated
    # retry boolean column exists on llm_calls; retries are just later-attempt rows for the same call_site
    total_prompt_tokens: Mapped[int] = mapped_column(default=0)
    total_completion_tokens: Mapped[int] = mapped_column(default=0)
    total_thinking_chars: Mapped[int] = mapped_column(default=0)
    truncated_calls: Mapped[int] = mapped_column(default=0)  # llm_calls.finish_reason == "length"
    empty_content_calls: Mapped[int] = mapped_column(default=0)  # llm_calls.finish_reason == "empty_content"
    validator_failures: Mapped[int] = mapped_column(default=0)  # llm_calls.validator_passed is False

    # Captured from AnalysisOrchestrator's own gate1_check()/gate2_check() calls -- never
    # persisted anywhere before this table. Null when the corresponding gate never ran (e.g. a
    # Gate 1 failure means gate2_* stays null, the run never got that far).
    gate1_passed: Mapped[Optional[bool]] = mapped_column(Boolean)
    gate1_reason: Mapped[Optional[str]] = mapped_column(String(255))
    gate2_passed: Mapped[Optional[bool]] = mapped_column(Boolean)
    gate2_reason: Mapped[Optional[str]] = mapped_column(String(255))

    # Denormalized from `recommendations` (see class docstring) -- null on any run that didn't
    # reach _create_recommendation (most exit paths).
    stock_outlook: Mapped[Optional[str]] = mapped_column(String(20))
    overall_confidence: Mapped[Optional[int]]

    # Entirely new signal (86bbwachy Phase 5) -- no existing code anywhere re-checks a persisted
    # AgentOutput row for empty key_factors/risks/narrative after storage; the real validators only
    # enforce length/count bounds on the LLM's raw output before storage and before retries exhaust.
    # Lists of agent_name, not counts, so a human can see which agent specifically.
    agents_with_empty_key_factors: Mapped[Optional[list]] = mapped_column(JSON)
    agents_with_empty_risks: Mapped[Optional[list]] = mapped_column(JSON)
    agents_with_empty_narrative: Mapped[Optional[list]] = mapped_column(JSON)

    # Nullable, cheap to reserve now, no UI writes these yet -- matches the original spec's own
    # "cheap to add, not required to be filled" framing.
    human_quality_note: Mapped[Optional[str]] = mapped_column(Text)
    human_quality_rating: Mapped[Optional[str]] = mapped_column(String(20))  # good|garbage|suspect

    created_at: Mapped[datetime] = mapped_column(default=func.now())
    # relationships
    run: Mapped["AnalysisRun"] = relationship(back_populates="run_quality_summary")
