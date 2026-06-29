from sqlalchemy import String, JSON, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional, Text

class AgentPortfolioTransaction(Base):
    __tablename__ = "agent_portfolio_transactions"
    transaction_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("user_profiles.user_id"), index=True)
    stock_id: Mapped[UUID] = mapped_column(ForeignKey("stocks.stock_id"), index=True)
    holding_id: Mapped[Optional[UUID]] = mapped_column(ForeignKey("agent_portfolio_holdings.holding_id"))
    account_type: Mapped[str] = mapped_column(String(20))
    transaction_type: Mapped[str] = mapped_column(String(20))  # buy|sell|dividend|drip|capital_injection
    shares: Mapped[Optional[float]]
    price_per_share: Mapped[float]
    total_amount: Mapped[float]
    fees: Mapped[float] = mapped_column(default=0)
    currency: Mapped[str] = mapped_column(String(3))
    transaction_date: Mapped[datetime]
    cost_basis_at_sell: Mapped[Optional[float]]
    realized_gain_loss: Mapped[Optional[float]]
    realized_gain_loss_pct: Mapped[Optional[float]]
    analysis_run_id: Mapped[Optional[UUID]] = mapped_column(ForeignKey("analysis_runs.run_id"))
    decision_type: Mapped[str] = mapped_column(String(30))  # outlook_driven|swap|rebalance|tax_harvest|capital_deploy|circuit_breaker
    decision_confidence: Mapped[Optional[int]]
    decision_rationale: Mapped[str] = mapped_column(Text)
    cio_outlook_at_decision: Mapped[Optional[str]] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(default=func.now())