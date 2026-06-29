from sqlalchemy import String, JSON, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional

class AgentPortfolioDecision(Base):
    __tablename__ = "agent_portfolio_decisions"
    decision_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("user_profiles.user_id"), index=True)
    cycle_date: Mapped[datetime] = mapped_column(index=True)
    cycle_type: Mapped[str] = mapped_column(String(20))  # weekly_batch|on_demand|earnings_triggered|circuit_breaker
    optimization_input_snapshot: Mapped[dict] = mapped_column(JSON)
    optimization_output: Mapped[dict] = mapped_column(JSON)
    actions_taken: Mapped[list] = mapped_column(JSON)
    actions_skipped: Mapped[Optional[list]] = mapped_column(JSON)
    portfolio_value_before: Mapped[float]
    portfolio_value_after: Mapped[float]
    cash_deployed: Mapped[float] = mapped_column(default=0)
    cash_raised: Mapped[float] = mapped_column(default=0)
    transaction_costs_incurred: Mapped[float] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(default=func.now())