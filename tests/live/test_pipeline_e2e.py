"""DataPipeline.prepare() end-to-end (ClickUp 86bawpty3).

The first point in the DataBundle Assembly epic where a genuinely full,
real pipeline run becomes possible - every provider call and precompute
module wired together for one real US ticker and one real CA ticker, not a
sweep. tests/data/test_pipeline.py already covers the assembly logic
itself (field mapping, is_ca gating, sources_used merge) against fakes;
this file only confirms the real wiring holds together.
"""

from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api.database import Base
from api.tables.agent_outputs import AgentOutput  # noqa: F401 - Stock's mapper needs this reachable
from api.tables.analysis_runs import AnalysisRun  # noqa: F401
from api.tables.prediction_checkpoints import PredictionCheckpoint  # noqa: F401
from api.tables.predictions import Prediction  # noqa: F401
from api.tables.recommendations import Recommendation  # noqa: F401
from api.tables.stock import Stock
from api.tables.user_profile import UserProfile  # noqa: F401
from data.pipeline import DataPipeline, StockNotFoundError
from data.schemas.context import AnalysisContext

pytestmark = pytest.mark.live


async def _make_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(bind=engine, expire_on_commit=False)()


async def _insert_stock(session, **fields):
    stock = Stock(stock_id=uuid4(), **fields)
    session.add(stock)
    await session.commit()
    return stock


async def test_prepare_real_us_stock():
    session = await _make_session()
    stock = await _insert_stock(
        session,
        canonical_ticker="AAPL",
        company_name="Apple Inc.",
        primary_exchange="NASDAQ",
        currency="USD",
        sector="Technology",
        industry="Consumer Electronics",
    )
    context = AnalysisContext(account_type="trading", timeline="medium_term")

    bundle = await DataPipeline().prepare(stock.stock_id, context, session)

    assert bundle.stock.ticker == "AAPL"
    assert bundle.benchmark_ticker == "^GSPC"
    assert bundle.canadian_data_flags is None
    assert bundle.analyst_recommendation_trends is not None
    assert bundle.price_info["current_price"] > 0
    assert bundle.data_freshness["get_price_history"]
    assert bundle.data_freshness["get_price_history:benchmark"]
    assert bundle.data_freshness["get_price_history:sector_etf"]
    assert bundle.tax_metrics
    # Real gap caught on review: raw_price/benchmark_price were originally
    # fetched with only "1y" of history, but risk_metrics.py's own module
    # docstring requires >=3y for max_drawdown_3yr_pct/recovery_3yr_days to
    # resolve at all - a fast unit test with fake compute_all() can't catch
    # this, only a real run against real data can.
    assert bundle.risk_metrics["max_drawdown_3yr_pct"] is not None
    assert bundle.risk_metrics["beta"] is not None


async def test_prepare_real_ca_stock():
    session = await _make_session()
    stock = await _insert_stock(
        session,
        canonical_ticker="RY.TO",
        company_name="Royal Bank of Canada",
        primary_exchange="TSX",
        currency="CAD",
        sector="Finance",
        industry="Banks",
    )
    context = AnalysisContext(account_type="tfsa", timeline="long_term")

    bundle = await DataPipeline().prepare(stock.stock_id, context, session)

    assert bundle.stock.ticker == "RY.TO"
    assert bundle.benchmark_ticker == "^GSPTSE"
    assert bundle.canadian_data_flags is not None
    assert bundle.analyst_recommendation_trends is None
    assert bundle.price_info["current_price"] > 0
    assert bundle.data_freshness["get_price_history"]
    assert bundle.data_freshness["get_price_history:benchmark"]
    assert bundle.data_freshness["get_price_history:sector_etf"]
    assert bundle.tax_metrics
    assert bundle.risk_metrics["max_drawdown_3yr_pct"] is not None
    assert bundle.risk_metrics["beta"] is not None


async def test_prepare_stock_not_found_raises_against_real_db():
    session = await _make_session()
    context = AnalysisContext(account_type="trading", timeline="medium_term")
    with pytest.raises(StockNotFoundError):
        await DataPipeline().prepare(uuid4(), context, session)
