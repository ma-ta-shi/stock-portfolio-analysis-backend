from sqlalchemy import String, JSON, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional

class Stock(Base):
    __tablename__ = "stocks"
    stock_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    canonical_ticker: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    company_name: Mapped[str] = mapped_column(String(200))
    primary_exchange: Mapped[str] = mapped_column(String(20))
    currency: Mapped[str] = mapped_column(String(3))
    sector: Mapped[Optional[str]] = mapped_column(String(100))
    industry: Mapped[Optional[str]] = mapped_column(String(100))
    isin: Mapped[Optional[str]] = mapped_column(String(12))
    last_updated: Mapped[datetime] = mapped_column(default=func.now())
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON)
    # relationships
    aliases: Mapped[list["StockAlias"]] = relationship(back_populates="stock")
    analysis_runs: Mapped[list["AnalysisRun"]] = relationship(back_populates="stock")