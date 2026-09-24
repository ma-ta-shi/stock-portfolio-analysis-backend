from enum import StrEnum
from sqlalchemy import ForeignKey, String, JSON, func, Index, text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional


class RunStatus(StrEnum):
    """The real value set for AnalysisRun.status (86bbuhjup) -- the column
    itself has no enum/CHECK constraint (plain String(30)); this is the
    guard, used by the orchestrator rather than relying on it as free text.

    Values track docs/technical/orchestration-engine.md's "Execution flow"
    section's own status naming (queued -> pass1_running -> pass1_complete ->
    pass2_running -> pass2_complete -> synthesis_running -> completed), not
    that same doc's gate-threshold PROSE alongside it (still describes the
    pre-2026-09-16 "minimum 3/5 reliable" Gate 1 contract, settled otherwise
    -- see docs/decision-log.md) or its RunStatus pseudocode (runs the CIO
    unconditionally and only distinguishes COMPLETE/COMPLETE_WITH_WARNINGS
    after the fact, contradicting the settled Gate 2 rule that a Gate 2
    failure skips the CIO entirely, audit E78). `failed` is this enum's own
    addition, covering both a Gate 1 failure (no usable Pass 1 output at
    all) and a Gate 2 failure (a mandatory advocate missing) as one terminal
    non-error outcome -- there is no separate "completed with warnings"
    state, since Gate 2 failure means the CIO never runs and no
    Recommendation is ever created."""

    QUEUED = "queued"
    PASS1_RUNNING = "pass1_running"
    PASS1_COMPLETE = "pass1_complete"
    PASS2_RUNNING = "pass2_running"
    PASS2_COMPLETE = "pass2_complete"
    SYNTHESIS_RUNNING = "synthesis_running"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"
    run_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("user_profiles.user_id"), index=True)
    stock_id: Mapped[UUID] = mapped_column(ForeignKey("stocks.stock_id"), index=True)
    account_type: Mapped[str] = mapped_column(String(20))  # tfsa|rrsp|trading
    timeline: Mapped[str] = mapped_column(String(20))       # short_term|medium_term|long_term
    triggered_at: Mapped[datetime] = mapped_column(default=func.now())
    triggered_by: Mapped[str] = mapped_column(String(20))    # manual|scheduled|alert
    status: Mapped[str] = mapped_column(String(30), default="queued")
    pass1_completed_at: Mapped[Optional[datetime]]
    pass2_completed_at: Mapped[Optional[datetime]]
    completed_at: Mapped[Optional[datetime]]
    llm_config: Mapped[dict] = mapped_column(JSON)
    disagreement_score: Mapped[Optional[int]]
    disagreement_class: Mapped[Optional[str]] = mapped_column(String(20))
    error_log: Mapped[Optional[list]] = mapped_column(JSON)
    # relationships
    stock: Mapped["Stock"] = relationship(back_populates="analysis_runs")
    agent_outputs: Mapped[list["AgentOutput"]] = relationship(back_populates="run")
    recommendation: Mapped[Optional["Recommendation"]] = relationship(back_populates="run")

    # Real TOCTOU race, confirmed live (2026-09-23): api/routes/analysis.py's
    # create_analysis() checks for an in-progress run, then inserts a new
    # one -- nothing serialized that check-then-write across two concurrent
    # requests, and two genuinely concurrent httpx requests for the same
    # brand-new ticker (tests/api/test_analysis_routes.py) reproduced
    # exactly that: both passed the check, both got 202, two AnalysisRun
    # rows landed for the same stock. A partial unique index (DB-enforced,
    # not just app-level) closes this the same way stocks.canonical_ticker's
    # own UNIQUE constraint already closes the analogous Stock race -- only
    # one NON-terminal run per stock can ever exist at the database level,
    # regardless of request timing. Only excludes completed/failed, matching
    # RunStatus's own docstring: those are the only two terminal states, and
    # a stock legitimately accumulates many of them over time.
    __table_args__ = (
        Index(
            "ix_analysis_runs_one_active_per_stock",
            "stock_id",
            unique=True,
            sqlite_where=text("status NOT IN ('completed', 'failed')"),
            postgresql_where=text("status NOT IN ('completed', 'failed')"),
        ),
    )