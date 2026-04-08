from sqlalchemy import Column, Integer, String, JSON, Boolean, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional

class UserProfile(Base):
    __tablename__ = "user_profiles"
    user_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    display_name: Mapped[str] = mapped_column(String(100))
    financial_profile: Mapped[Optional[dict]] = mapped_column(JSON)  # income, province, risk tolerance
    context_and_goals: Mapped[Optional[dict]] = mapped_column(JSON)  # goals, constraints, preferences
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(default=func.now())