from sqlalchemy import String, JSON, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional, Text

class ShadowPrediction(Base):
    __tablename__ = "shadow_predictions"
    shadow_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    analysis_run_id: Mapped[UUID] = mapped_column(ForeignKey("analysis_runs.run_id"), index=True)
    primary_outlook_direction: Mapped[str] = mapped_column(String(20))
    primary_confidence: Mapped[int]
    primary_projected_return_tier: Mapped[Optional[str]] = mapped_column(String(20))
    shadow_outlook_direction: Mapped[str] = mapped_column(String(20))
    shadow_confidence: Mapped[int]
    shadow_projected_return_tier: Mapped[Optional[str]] = mapped_column(String(20))
    divergence_magnitude: Mapped[str] = mapped_column(String(10))  # none | minor | moderate | major
    primary_accuracy_at_checkpoint: Mapped[Optional[float]]
    shadow_accuracy_at_checkpoint: Mapped[Optional[float]]
    which_was_closer: Mapped[Optional[str]] = mapped_column(String(10))  # primary | shadow | tie
    created_at: Mapped[datetime] = mapped_column(default=func.now())