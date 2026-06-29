from sqlalchemy import String, JSON, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional, Text

class PortfolioHolding(Base):
    __tablename__ = "portfolio_holdings"
    holding_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("user_profiles.user_id"), index=True)
    stock_id: Mapped[UUID] = mapped_column(ForeignKey("stocks.stock_id"), index=True)
    account_type: Mapped[str] = mapped_column(String(20))  # tfsa|rrsp|trading
    shares: Mapped[float]  # Current share count (reduced on sells)
    average_cost_basis: Mapped[float]  # Recomputed from buy transactions: total_book_cost / total_shares
    total_book_cost: Mapped[float]  # Sum of (shares * price) across all buy transactions
    is_active: Mapped[bool] = mapped_column(default=True)  # False when all shares sold (closed position)
    # Realized P&L tracking (accumulated from sell transactions)
    total_realized_gain_loss: Mapped[float] = mapped_column(default=0)  # Sum of realized P&L from all sells
    total_dividends_received: Mapped[float] = mapped_column(default=0)  # Sum of net dividends
    total_fees_paid: Mapped[float] = mapped_column(default=0)  # Sum of all transaction fees
    first_buy_date: Mapped[Optional[datetime]]  # Date of first buy transaction
    last_transaction_date: Mapped[Optional[datetime]]  # Date of most recent transaction
    added_at: Mapped[datetime] = mapped_column(default=func.now())
    last_updated: Mapped[datetime] = mapped_column(default=func.now())
    notes: Mapped[Optional[str]] = mapped_column(Text)