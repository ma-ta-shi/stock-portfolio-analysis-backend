from sqlalchemy import String, JSON, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional, Text

class AllocationRecommendation(Base):
    __tablename__ = "allocation_recommendations"
    allocation_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("user_profiles.user_id"), index=True)
    available_capital: Mapped[float]
    account_type: Mapped[str] = mapped_column(String(20))
    ranked_stocks: Mapped[list] = mapped_column(JSON)  # [{stock_id, expected_return, confidence, ...}]
    constraints_applied: Mapped[dict] = mapped_column(JSON)
    market_regime: Mapped[Optional[str]] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(default=func.now())