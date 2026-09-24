from sqlalchemy import ForeignKey, String, JSON, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional

class AgentOutput(Base):
    __tablename__ = "agent_outputs"
    output_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("analysis_runs.run_id"), index=True)
    agent_name: Mapped[str] = mapped_column(String(30), index=True)
    agent_pass: Mapped[str] = mapped_column(String(10))     # pass1|pass2|synthesis
    status: Mapped[str] = mapped_column(String(20))          # completed|failed|timeout|skipped
    recommendation: Mapped[Optional[str]] = mapped_column(String(10))  # bullish|bearish|neutral (null for pass1)
    confidence: Mapped[Optional[int]]                         # 0-100 (null for pass1)
    # reliability_score (0-100) retired 86bbummwp (decision D6, docs/technical/
    # pass1-confidence-model.md): removed, not redesigned -- never LLM-produced again.
    # analysis_confidence (pass1 only) is the real, live replacement signal. Mechanical
    # flags (stale_data/anomalies/data_coverage) are a separate, not-yet-built orchestrator
    # computation -- no speculative column added for them here; see the D6 doc.
    analysis_confidence: Mapped[Optional[str]] = mapped_column(String(20))  # pass1 only
    # {field_name: bool} -- did this agent's own build_user_message() render
    # a real value for this field, or "N/A" (86bbwachy Phase 4). Set only for
    # the 7 call sites that consume raw DataBundle fields directly (Stock
    # Researcher, Fundamental Analyst, Technical Analyst, Sentiment Analyst,
    # Macro Economist, Risk Advisor Stage A, Tax Strategist) -- null for
    # Bull/Bear Advocate, CIO, Shadow CIO, which only ever see compressed
    # Pass 1 output and have no raw-field presence of their own to report.
    # NOT the same thing as D6's own `data_coverage` above (a coarser,
    # data-source-type signal meant for a not-yet-built user-facing badge,
    # deliberately deferred, still unbuilt) -- this is the finer-grained,
    # per-rendered-field instrumentation signal, with an immediate consumer
    # (run_quality_summary, ordinary debugging), not a bet on a future
    # feature's exact shape.
    input_field_coverage: Mapped[Optional[dict]] = mapped_column(JSON)
    key_factors: Mapped[Optional[list]] = mapped_column(JSON)
    risks: Mapped[Optional[list]] = mapped_column(JSON)
    narrative: Mapped[Optional[str]] = mapped_column(Text)
    structured_output: Mapped[Optional[dict]] = mapped_column(JSON)  # agent-specific structured data
    data_sources_used: Mapped[Optional[list]] = mapped_column(JSON)
    input_agent_outputs: Mapped[Optional[list]] = mapped_column(JSON)  # output_ids of inputs (pass2)
    model_used: Mapped[str] = mapped_column(String(50))
    prompt_version: Mapped[str] = mapped_column(String(64))   # content hash
    tokens_used: Mapped[Optional[int]]
    latency_ms: Mapped[Optional[int]]
    error_detail: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    # relationships
    run: Mapped["AnalysisRun"] = relationship(back_populates="agent_outputs")