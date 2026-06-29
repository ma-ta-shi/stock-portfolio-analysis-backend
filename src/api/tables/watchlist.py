from sqlalchemy import String, JSON, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional, Text

class WatchlistEntry(Base):
    __tablename__ = "watchlist"
    entry_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("user_profiles.user_id"), index=True)
    stock_id: Mapped[UUID] = mapped_column(ForeignKey("stocks.stock_id"), index=True)
    added_at: Mapped[datetime] = mapped_column(default=func.now())
    default_account_type: Mapped[Optional[str]] = mapped_column(String(20))
    default_timeline: Mapped[Optional[str]] = mapped_column(String(20))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(default=True)