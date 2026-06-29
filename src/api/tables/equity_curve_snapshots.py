from sqlalchemy import String, JSON, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional, Text

class EquityCurveSnapshot(Base):
    __tablename__ = "equity_curve_snapshots"
    snapshot_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("user_profiles.user_id"), index=True)
    snapshot_date: Mapped[datetime] = mapped_column(index=True)
    account_type: Mapped[str] = mapped_column(String(20))  # tfsa|rrsp|trading|combined
    portfolio_value: Mapped[float]
    benchmark_value: Mapped[float]
    cumulative_return_pct: Mapped[float]
    cumulative_alpha_pct: Mapped[float]
    max_drawdown_pct: Mapped[float]
    open_positions: Mapped[int]
    closed_positions_total: Mapped[int]
    win_rate_pct: Mapped[Optional[float]]
    rolling_sharpe_6m: Mapped[Optional[float]]
    signal_alpha_decomposition: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=func.now())