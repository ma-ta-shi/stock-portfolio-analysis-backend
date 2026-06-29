from sqlalchemy import String, JSON, func
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
    reliability_score: Mapped[Optional[int]]                  # 0-100 (pass1 only)
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