from sqlalchemy import String, JSON, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional, Text

class DividendEvent(Base):
    __tablename__ = "dividend_events"
    dividend_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    transaction_id: Mapped[UUID] = mapped_column(ForeignKey("portfolio_transactions.transaction_id"))
    stock_id: Mapped[UUID] = mapped_column(ForeignKey("stocks.stock_id"), index=True)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("user_profiles.user_id"), index=True)
    account_type: Mapped[str] = mapped_column(String(20))
    ex_date: Mapped[datetime]
    payment_date: Mapped[datetime]
    amount_per_share: Mapped[float]
    total_amount: Mapped[float]
    withholding_tax: Mapped[float] = mapped_column(default=0)  # e.g., 15% US WHT on TFSA
    net_amount: Mapped[float]
    dividend_type: Mapped[str] = mapped_column(String(30))  # eligible_canadian|us_qualified|foreign|return_of_capital
    is_drip: Mapped[bool] = mapped_column(default=False)