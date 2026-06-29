from sqlalchemy import String, JSON, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional, Text

class ResearchCache(Base):
    __tablename__ = "research_cache"
    cache_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    stock_id: Mapped[UUID] = mapped_column(ForeignKey("stocks.stock_id"), index=True)
    data_type: Mapped[str] = mapped_column(String(30), index=True)
    fetched_at: Mapped[datetime] = mapped_column(default=func.now())
    expires_at: Mapped[datetime]
    data: Mapped[dict] = mapped_column(JSON)
    source: Mapped[str] = mapped_column(String(30))