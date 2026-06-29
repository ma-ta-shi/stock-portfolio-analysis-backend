from sqlalchemy import String, JSON, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional, Text

class CounterfactualRecord(Base):
    __tablename__ = "counterfactual_records"
    counterfactual_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    recommendation_id: Mapped[UUID] = mapped_column(index=True)  # FK to allocation_recommendations or analysis_runs
    recommendation_source: Mapped[str] = mapped_column(String(30))  # "portfolio_optimizer" | "cio_outlook"
    recommended_ticker: Mapped[str] = mapped_column(String(20))
    recommended_action_type: Mapped[str] = mapped_column(String(20))  # swap | deploy | rebalance | no_action
    counterfactuals: Mapped[list] = mapped_column(JSON)  # [{rank, ticker, action_type, expected_return_at_time, reason_not_chosen}]
    scoring_checkpoint: Mapped[Optional[str]] = mapped_column(String(10))  # 1m | 3m | 6m
    recommended_return_pct: Mapped[Optional[float]]
    counterfactual_returns: Mapped[Optional[list]] = mapped_column(JSON)  # [{rank, return_pct}]
    recommendation_was_best: Mapped[Optional[bool]]
    recommendation_rank_vs_counterfactuals: Mapped[Optional[int]]  # 1 if best, 2 if second, etc.
    created_at: Mapped[datetime] = mapped_column(default=func.now())