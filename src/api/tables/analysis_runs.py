from sqlalchemy import String, JSON, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional

class AnalysisRun(Base):
    __tablename__ = "analysis_runs"
    run_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("user_profiles.user_id"), index=True)
    stock_id: Mapped[UUID] = mapped_column(ForeignKey("stocks.stock_id"), index=True)
    account_type: Mapped[str] = mapped_column(String(20))  # tfsa|rrsp|trading|general
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