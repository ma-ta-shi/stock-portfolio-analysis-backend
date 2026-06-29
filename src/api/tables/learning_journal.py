from sqlalchemy import String, JSON, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional, Text

class LearningJournalEntry(Base):
    __tablename__ = "learning_journal"
    entry_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    pattern_type: Mapped[str] = mapped_column(String(40), index=True)
    description: Mapped[str] = mapped_column(Text)
    evidence_json: Mapped[dict] = mapped_column(JSON)
    proposed_action_json: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20), default="proposed")  # proposed|approved|implemented|rejected|superseded
    implemented_at: Mapped[Optional[datetime]]
    prompt_version_before: Mapped[Optional[str]] = mapped_column(String(64))
    prompt_version_after: Mapped[Optional[str]] = mapped_column(String(64))
    post_implementation_accuracy: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=func.now())