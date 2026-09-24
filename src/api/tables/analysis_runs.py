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

    # Run context tags (86bbwachy Phase 1) -- ephemeral, cannot be backfilled
    # onto a historical run once it's past, so they land in this migration
    # rather than being added opportunistically later (run-instrumentation.md
    # §5.6). Populated once, in orchestrator.py's run(), right after
    # DataPipeline.prepare() returns a real bundle -- not at AnalysisRun
    # creation time in api/routes/analysis.py, because instrument_type/
    # market_cap_bucket have no reliable source that early (Stock's own ORM
    # row carries neither asset_type nor market_cap; both only exist on
    # DataBundle.company_info, which doesn't exist until prepare() runs).
    # exchange/currency COULD be set earlier from the Stock row alone, but
    # are set here too so all five context tags come from one place, at one
    # point in the run, rather than a split design.
    exchange: Mapped[Optional[str]] = mapped_column(String(20))  # from bundle.stock.exchange
    currency: Mapped[Optional[str]] = mapped_column(String(3))  # from bundle.stock.currency
    instrument_type: Mapped[Optional[str]] = mapped_column(String(20))  # equity|etf|other, from company_info["asset_type"]
    # Large/mid/small-cap bucketing thresholds are new logic this ticket owns
    # (company_info["market_cap"] is a raw number; nothing else in this
    # codebase buckets it) -- see orchestrator.py's own bucketing helper for
    # the actual thresholds and their reasoning.
    market_cap_bucket: Mapped[Optional[str]] = mapped_column(String(20))  # large|mid|small|micro
    # Stays null until Milestone 5's regime-detection work exists -- nothing
    # in this codebase computes bull/bear/sideways/high_volatility/crisis/
    # transition classifications yet (market_regime_history.py is a coded,
    # currently-unwired table with zero writers). Not a bug that it's always
    # null today.
    #
    # Correction on review: unlike exchange/currency/instrument_type/
    # market_cap_bucket, this one is NOT necessarily unbackfillable later --
    # market_regime_history is keyed by classification_date, not by stock or
    # run, so a future regime-detection system given real historical data
    # could in principle backfill this column for every past run by looking
    # up whichever regime was active on that run's triggered_at date. Added
    # here anyway to match the spec's own decision and because a nullable,
    # currently-unused column costs nothing -- but "added from the first
    # migration because it can't be backfilled" (the reasoning that's
    # actually correct for the other four tags) does not apply to this one
    # specifically, and shouldn't be assumed to if this column is revisited.
    market_regime: Mapped[Optional[str]] = mapped_column(String(20))
    # relationships
    stock: Mapped["Stock"] = relationship(back_populates="analysis_runs")
    agent_outputs: Mapped[list["AgentOutput"]] = relationship(back_populates="run")
    recommendation: Mapped[Optional["Recommendation"]] = relationship(back_populates="run")
    run_quality_summary: Mapped[Optional["RunQualitySummary"]] = relationship(back_populates="run")

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