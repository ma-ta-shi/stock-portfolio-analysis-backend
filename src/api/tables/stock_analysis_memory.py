from sqlalchemy import String, JSON, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional, Text

class StockAnalysisMemory(Base):
    __tablename__ = "stock_analysis_memory"
    memory_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    stock_id: Mapped[UUID] = mapped_column(ForeignKey("stocks.stock_id"), index=True)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("analysis_runs.run_id"))
    agent_name: Mapped[str] = mapped_column(String(30), index=True)
    analysis_date: Mapped[datetime] = mapped_column(index=True)
    account_type: Mapped[str] = mapped_column(String(20))
    timeline: Mapped[str] = mapped_column(String(20))
    key_insights: Mapped[list] = mapped_column(JSON)  # [{insight, importance, sentiment}]
    recommendation: Mapped[Optional[str]] = mapped_column(String(10))  # For Pass 2/CIO agents
    confidence: Mapped[Optional[int]]
    predicted_return_tier: Mapped[Optional[str]] = mapped_column(String(20))
    outcome_summary: Mapped[Optional[str]] = mapped_column(Text)  # Filled later from feedback
    events_since: Mapped[Optional[list]] = mapped_column(JSON)  # Filled on next analysis: earnings, price changes, news
    created_at: Mapped[datetime] = mapped_column(default=func.now())