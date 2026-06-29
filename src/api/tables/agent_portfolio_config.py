from sqlalchemy import String, JSON, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional

class AgentPortfolioConfig(Base):
    __tablename__ = "agent_portfolio_config"
    config_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("user_profiles.user_id"), unique=True)
    is_enabled: Mapped[bool] = mapped_column(default=False)
    initialized_at: Mapped[Optional[datetime]]
    initial_portfolio_snapshot: Mapped[Optional[dict]] = mapped_column(JSON)
    injection_frequency: Mapped[Optional[str]] = mapped_column(String(20))  # biweekly|monthly|none
    injection_amount: Mapped[Optional[float]]
    last_injection_date: Mapped[Optional[datetime]]
    min_buy_confidence: Mapped[int] = mapped_column(default=60)
    min_sell_confidence: Mapped[int] = mapped_column(default=65)
    min_swap_improvement_pct: Mapped[float] = mapped_column(default=3.0)
    max_position_size_pct: Mapped[float] = mapped_column(default=10.0)
    max_sector_concentration_pct: Mapped[float] = mapped_column(default=30.0)
    min_holding_period_days: Mapped[int] = mapped_column(default=0)  # 0 = disabled (cost-based churn control)
    scan_universe: Mapped[str] = mapped_column(String(30), default="watchlist")  # watchlist|tsx60_sp500|custom
    custom_universe_tickers: Mapped[Optional[list]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    last_updated: Mapped[datetime] = mapped_column(default=func.now())