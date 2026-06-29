from sqlalchemy import String, JSON, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional

class AgentPortfolioAccountState(Base):
    __tablename__ = "agent_portfolio_account_state"
    state_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("user_profiles.user_id"), index=True)
    account_type: Mapped[str] = mapped_column(String(20))  # tfsa|rrsp|trading
    cash_balance: Mapped[float]
    contribution_room_remaining: Mapped[Optional[float]]  # TFSA/RRSP only
    total_contributions: Mapped[float] = mapped_column(default=0)
    total_withdrawals: Mapped[float] = mapped_column(default=0)  # Should be 0 for RRSP
    tfsa_sells_this_quarter: Mapped[Optional[dict]] = mapped_column(JSON)  # {ticker: count}
    realized_capital_gains_ytd: Mapped[float] = mapped_column(default=0)
    realized_capital_losses_ytd: Mapped[float] = mapped_column(default=0)
    superficial_loss_cooldowns: Mapped[Optional[dict]] = mapped_column(JSON)  # {ticker: cooldown_expiry_date}
    last_updated: Mapped[datetime] = mapped_column(default=func.now())