from sqlalchemy import String, JSON, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional, Text

class ErrorRecord(Base):
    __tablename__ = "error_records"
    error_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    timestamp: Mapped[datetime] = mapped_column(default=func.now(), index=True)
    severity: Mapped[str] = mapped_column(String(10), index=True)  # critical|high|medium|low
    component: Mapped[str] = mapped_column(String(50), index=True)  # agent|data_pipeline|llm|api|orchestrator|frontend|feedback|optimizer
    error_type: Mapped[str] = mapped_column(String(50), index=True)
    dedup_subtype: Mapped[Optional[str]] = mapped_column(String(100))  # Structured dedup key
    run_id: Mapped[Optional[UUID]] = mapped_column(index=True)
    agent_name: Mapped[Optional[str]] = mapped_column(String(30))
    stock_ticker: Mapped[Optional[str]] = mapped_column(String(20))
    user_id: Mapped[Optional[UUID]]
    message: Mapped[str] = mapped_column(Text)
    stack_trace: Mapped[Optional[str]] = mapped_column(Text)
    context_json: Mapped[Optional[dict]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20), default="open", index=True)  # open|acknowledged|investigating|resolved|auto_fixed|wont_fix
    resolution_notes: Mapped[Optional[str]] = mapped_column(Text)
    resolved_at: Mapped[Optional[datetime]]
    auto_fix_proposed: Mapped[Optional[str]] = mapped_column(Text)
    auto_fix_status: Mapped[Optional[str]] = mapped_column(String(20))  # proposed|approved|applied|rejected
    occurrence_count: Mapped[int] = mapped_column(default=1)
    first_seen: Mapped[datetime] = mapped_column(default=func.now())
    last_seen: Mapped[datetime] = mapped_column(default=func.now())
    related_error_ids: Mapped[Optional[list]] = mapped_column(JSON)