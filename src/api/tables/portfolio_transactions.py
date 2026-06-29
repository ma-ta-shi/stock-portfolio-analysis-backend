from sqlalchemy import String, JSON, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional, Text

class PortfolioTransaction(Base):
    __tablename__ = "portfolio_transactions"
    transaction_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("user_profiles.user_id"), index=True)
    stock_id: Mapped[UUID] = mapped_column(ForeignKey("stocks.stock_id"), index=True)
    holding_id: Mapped[Optional[UUID]] = mapped_column(ForeignKey("portfolio_holdings.holding_id"))
    account_type: Mapped[str] = mapped_column(String(20))
    transaction_type: Mapped[str] = mapped_column(String(20))  # buy|sell|dividend|drip|transfer_in|transfer_out
    shares: Mapped[Optional[float]]  # Positive for buy/drip/transfer_in, negative for sell/transfer_out, null for dividends
    price_per_share: Mapped[float]
    total_amount: Mapped[float]  # shares * price, or dividend amount
    fees: Mapped[float] = mapped_column(default=0)
    currency: Mapped[str] = mapped_column(String(3))  # CAD|USD
    transaction_date: Mapped[datetime]
    # Realized P&L (populated for sell transactions only)
    cost_basis_at_sell: Mapped[Optional[float]]  # The average cost basis at time of sell
    realized_gain_loss: Mapped[Optional[float]]  # (sell_price - avg_cost_basis) * shares_sold - fees
    realized_gain_loss_pct: Mapped[Optional[float]]  # Percentage return on the sold portion
    # Feedback integration (populated for sell transactions only)
    active_prediction_id: Mapped[Optional[UUID]]  # If stock had an active prediction at time of sell
    system_outlook_at_sell: Mapped[Optional[str]] = mapped_column(String(20))  # What the system recommended when the user sold
    sell_aligned_with_system: Mapped[Optional[bool]]  # Did the user sell when the system said bearish? Or against a bullish outlook?
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=func.now())