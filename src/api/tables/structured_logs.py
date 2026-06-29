from sqlalchemy import String, JSON, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional, Text

class StructuredLog(Base):
    __tablename__ = "structured_logs"
    log_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    timestamp: Mapped[datetime] = mapped_column(default=func.now(), index=True)
    run_id: Mapped[Optional[UUID]] = mapped_column(index=True)
    component: Mapped[str] = mapped_column(String(50))
    level: Mapped[str] = mapped_column(String(10))
    message: Mapped[str] = mapped_column(Text)
    context: Mapped[Optional[dict]] = mapped_column(JSON)
    duration_ms: Mapped[Optional[int]]
    error_detail: Mapped[Optional[str]] = mapped_column(Text)