from sqlalchemy import String, JSON, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional, Text

class MarketRegimeHistory(Base):
    __tablename__ = "market_regime_history"
    regime_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    classification_date: Mapped[datetime] = mapped_column(index=True)
    regime: Mapped[str] = mapped_column(String(20))  # bull|bear|sideways|high_volatility|crisis|transition
    confidence: Mapped[int]  # 0-100
    indicators: Mapped[dict] = mapped_column(JSON)  # {benchmark_trend, vix_level, breadth, ...}
    is_circuit_breaker_active: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(default=func.now())