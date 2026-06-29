from sqlalchemy import String, JSON, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional, Text

class PredictionCheckpoint(Base):
    __tablename__ = "prediction_checkpoints"
    checkpoint_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    prediction_id: Mapped[UUID] = mapped_column(ForeignKey("predictions.prediction_id"), index=True)
    interval: Mapped[str] = mapped_column(String(10))  # 1w|1m|3m|6m|12m
    due_date: Mapped[datetime] = mapped_column(index=True)
    actual_price: Mapped[Optional[float]]
    dividends_per_share: Mapped[Optional[float]]
    benchmark_price: Mapped[Optional[float]]
    benchmark_dividends_per_share: Mapped[Optional[float]]
    actual_total_return_pct: Mapped[Optional[float]]  # (price change + dividends) / rec_price
    actual_return_net_of_costs_pct: Mapped[Optional[float]]
    benchmark_total_return_pct: Mapped[Optional[float]]
    prediction_error_pct: Mapped[Optional[float]]
    return_in_predicted_range: Mapped[Optional[bool]]
    interpolated_expected: Mapped[Optional[float]]
    checked_at: Mapped[Optional[datetime]]
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    # relationships
    prediction: Mapped["Prediction"] = relationship(back_populates="checkpoints")