from sqlalchemy import String, JSON, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional

class AgentPortfolioHolding(Base):
    __tablename__ = "agent_portfolio_holdings"
    holding_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("user_profiles.user_id"), index=True)
    stock_id: Mapped[UUID] = mapped_column(ForeignKey("stocks.stock_id"), index=True)
    account_type: Mapped[str] = mapped_column(String(20))  # tfsa|rrsp|trading
    shares: Mapped[float]
    average_cost_basis: Mapped[float]
    total_book_cost: Mapped[float]
    is_active: Mapped[bool] = mapped_column(default=True)
    total_realized_gain_loss: Mapped[float] = mapped_column(default=0)
    total_dividends_received: Mapped[float] = mapped_column(default=0)
    total_fees_paid: Mapped[float] = mapped_column(default=0)
    first_buy_date: Mapped[Optional[datetime]]
    last_transaction_date: Mapped[Optional[datetime]]
    added_at: Mapped[datetime] = mapped_column(default=func.now())
    last_updated: Mapped[datetime] = mapped_column(default=func.now())