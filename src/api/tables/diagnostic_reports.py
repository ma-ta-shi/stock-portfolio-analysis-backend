from sqlalchemy import String, JSON, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional, Text

class DiagnosticReport(Base):
    __tablename__ = "diagnostic_reports"
    diagnostic_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    prediction_id: Mapped[UUID] = mapped_column(ForeignKey("predictions.prediction_id"), index=True)
    checkpoint_interval: Mapped[str] = mapped_column(String(10))
    diagnosis_json: Mapped[dict] = mapped_column(JSON)  # Full root cause analysis
    error_classification: Mapped[str] = mapped_column(String(30))  # direction_wrong|magnitude_over|magnitude_under|timing_error|correct
    primary_error_source: Mapped[Optional[str]] = mapped_column(String(30))  # agent name
    primary_error_type: Mapped[Optional[str]] = mapped_column(String(50))
    diagnosis_quality_score: Mapped[Optional[int]]  # 0-100, from judge step
    judge_feedback: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=func.now())