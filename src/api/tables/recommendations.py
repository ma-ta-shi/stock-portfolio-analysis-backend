from sqlalchemy import String, JSON, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional, Text

class Recommendation(Base):
    __tablename__ = "recommendations"
    recommendation_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("analysis_runs.run_id"), unique=True)
    stock_id: Mapped[UUID] = mapped_column(ForeignKey("stocks.stock_id"), index=True)
    # Two-layer model: Layer 1 = stock outlook (this table), Layer 2 = portfolio optimization (separate system)
    stock_outlook_direction: Mapped[str] = mapped_column(String(20))  # bullish|somewhat_bullish|neutral|somewhat_bearish|bearish
    overall_confidence: Mapped[int]  # 0-100
    account_recommendation: Mapped[dict] = mapped_column(JSON)  # {action, rationale}
    multi_account_recommendations: Mapped[Optional[dict]] = mapped_column(JSON)  # for "general" context
    position_size_suggestion: Mapped[Optional[str]] = mapped_column(String(50))
    key_drivers: Mapped[list] = mapped_column(JSON)  # from stock_outlook.key_drivers
    key_risks: Mapped[list] = mapped_column(JSON)
    synthesis_narrative: Mapped[str] = mapped_column(Text)
    dissenting_views: Mapped[Optional[list]] = mapped_column(JSON)
    bull_case_strength: Mapped[int]
    bear_case_strength: Mapped[int]
    # Expected return estimation (phased precision — see Feedback Learning doc)
    expected_return_tier: Mapped[Optional[str]] = mapped_column(String(20))  # high|moderate|low|minimal (Phase 2)
    expected_return_pct: Mapped[Optional[float]]    # Point estimate (Phase 4)
    expected_return_low: Mapped[Optional[float]]    # 25th percentile (Phase 3)
    expected_return_high: Mapped[Optional[float]]   # 75th percentile (Phase 3)
    expected_return_basis: Mapped[Optional[str]] = mapped_column(Text)  # Plain-English reasoning
    return_estimate_confidence: Mapped[Optional[int]]  # 0-100, separate from directional confidence
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    # relationships
    run: Mapped["AnalysisRun"] = relationship(back_populates="recommendation")
    prediction: Mapped[Optional["Prediction"]] = relationship(back_populates="recommendation")