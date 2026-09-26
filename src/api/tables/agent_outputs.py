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
    # analysis_confidence (pass1 only) is the real, live replacement signal.
    analysis_confidence: Mapped[Optional[str]] = mapped_column(String(20))  # pass1 only
    # D6's three mechanical, orchestrator-computed flags (86bbummwp Tier 2) -- distinct
    # from input_field_coverage below (finer-grained, per-rendered-field). All three are
    # nullable, and null is a real, distinct state from an empty list: null means this
    # agent doesn't implement that check (yet); [] means it does, and found nothing.
    # Pass 1 only, same as analysis_confidence -- Pass 2/CIO/Shadow CIO never set these.
    stale_data: Mapped[Optional[list]] = mapped_column(JSON)          # which series are stale
    anomalies: Mapped[Optional[list]] = mapped_column(JSON)           # pre-flight contradiction findings
    data_coverage: Mapped[Optional[dict]] = mapped_column(JSON)       # {"present": [...], "absent": [...]}
    # D6 section 3's mechanical per-agent rollup (86bbummwp Tier 3) -- high|medium|low,
    # computed purely from this same row's own stale_data/anomalies/data_coverage, distinct
    # from analysis_confidence (the model's own judgment, which can legitimately diverge --
    # an agent can claim high confidence while this is low, exactly the case this flag
    # exists to surface to Pass 2/CIO). Deliberately NOT also rolled up into one combined
    # cross-agent value anywhere -- see docs/technical/pass1-confidence-model.md's own D6
    # section 3 for the spec, and 86bbummwp's Tier 3 plan for why the combined rollup it
    # also names was concluded not to be built at all (misleading for a human viewer just
    # as much as an LLM prompt, not just lacking a consumer).
    data_quality_assessment: Mapped[Optional[str]] = mapped_column(String(10))
    # {field_name: bool} -- did this agent's own build_user_message() render
    # a real value for this field, or "N/A" (86bbwachy Phase 4). Set only for
    # the 7 call sites that consume raw DataBundle fields directly (Stock
    # Researcher, Fundamental Analyst, Technical Analyst, Sentiment Analyst,
    # Macro Economist, Risk Advisor Stage A, Tax Strategist) -- null for
    # Bull/Bear Advocate, CIO, Shadow CIO, which only ever see compressed
    # Pass 1 output and have no raw-field presence of their own to report.
    # NOT the same thing as data_coverage above (a coarser, data-source-type
    # signal, also feeding a not-yet-built user-facing badge) -- this is the
    # finer-grained, per-rendered-field instrumentation signal, with an
    # immediate consumer (run_quality_summary, ordinary debugging).
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