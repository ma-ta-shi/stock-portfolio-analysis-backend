from sqlalchemy import String, JSON, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional, Text

class Prediction(Base):
    __tablename__ = "predictions"
    prediction_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    recommendation_id: Mapped[UUID] = mapped_column(ForeignKey("recommendations.recommendation_id"), unique=True)
    stock_id: Mapped[UUID] = mapped_column(ForeignKey("stocks.stock_id"), index=True)
    price_at_recommendation: Mapped[float]
    benchmark_price_at_recommendation: Mapped[float]
    # Expected return estimation (phased precision — see Feedback Learning doc)
    predicted_return_tier: Mapped[Optional[str]] = mapped_column(String(20))  # high|moderate|low|minimal (Phase 2)
    predicted_return_pct: Mapped[Optional[float]]    # Point estimate (Phase 4, after calibration)
    predicted_return_low: Mapped[Optional[float]]    # 25th percentile (Phase 3)
    predicted_return_high: Mapped[Optional[float]]   # 75th percentile (Phase 3)
    predicted_capital_appreciation_pct: Mapped[Optional[float]]
    predicted_dividend_income_pct: Mapped[Optional[float]]
    predicted_tax_drag_pct: Mapped[Optional[float]]
    return_estimate_confidence: Mapped[Optional[int]]  # 0-100, separate from directional confidence
    estimated_round_trip_cost_pct: Mapped[Optional[float]]  # Pre-computed from position size + liquidity
    # Checkpoints are stored in a separate prediction_checkpoints table (see below)
    # This avoids read-modify-write on a JSON blob and enables SQL queries on individual checkpoints
    direction_scores: Mapped[Optional[dict]] = mapped_column(JSON)
    magnitude_scores: Mapped[Optional[dict]] = mapped_column(JSON)
    relative_score: Mapped[Optional[float]]
    composite_score: Mapped[Optional[float]]
    per_agent_scores: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    last_checked_at: Mapped[Optional[datetime]]
    # relationships
    recommendation: Mapped["Recommendation"] = relationship(back_populates="prediction")
    checkpoints: Mapped[list["PredictionCheckpoint"]] = relationship(back_populates="prediction", order_by="PredictionCheckpoint.due_date")