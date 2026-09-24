"""Tests for api/routes/analysis.py (86bbuhjup) -- route-level behavior
(validation, ticker resolution, status mapping, error codes), not the
orchestrator itself (already covered by tests/services/test_orchestrator.py).
_run_analysis_background is patched to a no-op in every test here: it opens
its own AsyncSessionLocal() bound to the real api.database engine (by
design -- see that function's own docstring), which would touch the real
app.db if actually run during a route test.
"""
import asyncio
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from api.database import Base
from api.main import app
from api.routes.analysis import get_async_db
from api.tables.agent_outputs import AgentOutput  # noqa: F401
from api.tables.analysis_runs import AnalysisRun, RunStatus
from api.tables.prediction_checkpoints import PredictionCheckpoint  # noqa: F401
from api.tables.predictions import Prediction  # noqa: F401
from api.tables.recommendations import Recommendation
from api.tables.shadow_predictions import ShadowPrediction  # noqa: F401
from api.tables.stock import Stock
from api.tables.user_profile import UserProfile  # noqa: F401


@pytest.fixture
def test_db():
    """One shared in-memory SQLite DB (StaticPool -- a bare `:memory:` URL
    gets a fresh empty DB per connection, which breaks across the several
    requests one test makes) for the lifetime of a single test."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

    async def override_get_async_db():
        async with session_factory() as db:
            yield db

    app.dependency_overrides[get_async_db] = override_get_async_db

    async def _create():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_create())

    yield session_factory

    app.dependency_overrides.pop(get_async_db, None)


@pytest.fixture
def client(test_db):
    with patch("api.routes.analysis._run_analysis_background", new=AsyncMock()):
        yield TestClient(app)


def _company_info(**overrides):
    defaults = {"name": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics",
                "primary_exchange": "NASDAQ", "currency": "USD", "country": "US", "asset_type": "equity"}
    return {**defaults, **overrides}


@pytest.fixture
def mock_router():
    with patch("api.routes.analysis.Router") as MockRouter:
        instance = AsyncMock()
        instance.get_company_info = AsyncMock(return_value=_company_info())
        MockRouter.return_value.__aenter__ = AsyncMock(return_value=instance)
        MockRouter.return_value.__aexit__ = AsyncMock(return_value=False)
        yield instance


def test_create_analysis_resolves_new_ticker_and_creates_stock(client, mock_router):
    resp = client.post("/api/analysis/", json={
        "ticker": "AAPL", "account_type": "trading", "timeline": "medium_term",
        "user_id": str(uuid4()),
    })
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert "analysis_id" in body


def test_create_analysis_reuses_existing_stock_without_a_router_call(client, test_db):
    async def _seed():
        async with test_db() as db:
            db.add(Stock(stock_id=uuid4(), canonical_ticker="RY.TO", company_name="Royal Bank",
                          primary_exchange="TSX", currency="CAD", sector="Financials"))
            await db.commit()

    asyncio.run(_seed())

    with patch("api.routes.analysis.Router") as MockRouter:
        resp = client.post("/api/analysis/", json={
            "ticker": "RY.TO", "account_type": "tfsa", "timeline": "long_term",
            "user_id": str(uuid4()),
        })
        MockRouter.assert_not_called()
    assert resp.status_code == 202


def test_create_analysis_rejects_etf(client, mock_router):
    mock_router.get_company_info = AsyncMock(return_value=_company_info(asset_type="etf"))
    resp = client.post("/api/analysis/", json={
        "ticker": "SPY", "account_type": "trading", "timeline": "medium_term",
        "user_id": str(uuid4()),
    })
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"]["code"] == "UNSUPPORTED_ASSET_TYPE"


def test_create_analysis_404s_on_unresolvable_ticker(client, mock_router):
    mock_router.get_company_info = AsyncMock(return_value={})
    resp = client.post("/api/analysis/", json={
        "ticker": "BOGUS", "account_type": "trading", "timeline": "medium_term",
        "user_id": str(uuid4()),
    })
    assert resp.status_code == 404
    assert resp.json()["detail"]["error"]["code"] == "STOCK_NOT_FOUND"


def test_create_analysis_409s_when_already_in_progress(client, mock_router):
    first = client.post("/api/analysis/", json={
        "ticker": "AAPL", "account_type": "trading", "timeline": "medium_term",
        "user_id": str(uuid4()),
    })
    assert first.status_code == 202

    second = client.post("/api/analysis/", json={
        "ticker": "AAPL", "account_type": "tfsa", "timeline": "short_term",
        "user_id": str(uuid4()),
    })
    assert second.status_code == 409
    assert second.json()["detail"]["error"]["code"] == "ANALYSIS_IN_PROGRESS"


@pytest.mark.asyncio
async def test_concurrent_create_analysis_for_same_new_ticker_does_not_duplicate():
    """Real race, flagged during a 2026-09-23 review, never previously
    tested: both _resolve_or_create_stock's own SELECT-then-INSERT (for a
    brand-new ticker) and create_analysis's own in-progress-run SELECT-
    then-INSERT are classic TOCTOU races -- nothing serializes the
    check-then-write across two concurrent requests. Two users requesting
    analysis on the same never-before-seen ticker at the same moment is a
    real, plausible scenario, not a contrived one.

    Uses a real on-disk temp SQLite file, NOT the shared `test_db` fixture
    (:memory: + StaticPool) -- StaticPool funnels every session through
    ONE shared underlying connection, which is fine for the sequential
    multi-request tests elsewhere in this file, but two sessions actually
    executing concurrently against that single shared connection collide
    inside SQLAlchemy's own greenlet machinery ("aclose(): asynchronous
    generator is already running"), not just at the SQL level -- an
    artifact of the test double, never something production's real
    ASYNC_DATABASE_URL (a real file, real per-connection pooling, see
    api/database.py) would do. A temp file + the engine's natural pool
    gives each session a genuinely separate connection, matching
    production's actual concurrency shape.

    Router.get_company_info is given a real await point (asyncio.sleep)
    to stand in for genuine network latency -- the exact place the real
    route has its own real await (a live provider call) -- so the race
    window opens deterministically rather than hoping SQLite happens to
    be slow enough on its own.
    """
    import os
    import tempfile

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
        session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

        async def override_get_async_db():
            async with session_factory() as db:
                yield db

        app.dependency_overrides[get_async_db] = override_get_async_db
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        try:
            with patch("api.routes.analysis._run_analysis_background", new=AsyncMock()), \
                 patch("api.routes.analysis.Router") as MockRouter:
                async def _slow_get_company_info(ticker):
                    await asyncio.sleep(0.05)
                    return _company_info(name="Newco", primary_exchange="NASDAQ", currency="USD")

                instance = AsyncMock()
                instance.get_company_info = _slow_get_company_info
                MockRouter.return_value.__aenter__ = AsyncMock(return_value=instance)
                MockRouter.return_value.__aexit__ = AsyncMock(return_value=False)

                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                    body = {"ticker": "NEWCO", "account_type": "trading", "timeline": "medium_term",
                            "user_id": str(uuid4())}
                    resp1, resp2 = await asyncio.gather(
                        ac.post("/api/analysis/", json=body),
                        ac.post("/api/analysis/", json=body),
                    )

            statuses = sorted([resp1.status_code, resp2.status_code])

            async with session_factory() as db:
                stocks = (await db.execute(select(Stock).where(Stock.canonical_ticker == "NEWCO"))).scalars().all()
                runs = (await db.execute(select(AnalysisRun))).scalars().all()
        finally:
            app.dependency_overrides.pop(get_async_db, None)
            await engine.dispose()
    finally:
        os.remove(db_path)

    assert len(stocks) == 1, (
        f"_resolve_or_create_stock's SELECT-then-INSERT let concurrent requests create "
        f"{len(stocks)} Stock rows for the same ticker; statuses were {statuses}"
    )
    assert len(runs) == 1, (
        f"create_analysis's in-progress-run check let concurrent requests create "
        f"{len(runs)} AnalysisRun rows for the same ticker; statuses were {statuses}"
    )
    assert statuses == [202, 409], (
        f"expected exactly one request accepted (202) and one rejected as a duplicate "
        f"(409), got {statuses}"
    )


def test_create_analysis_rejects_invalid_account_type(client, mock_router):
    resp = client.post("/api/analysis/", json={
        "ticker": "AAPL", "account_type": "not_a_real_account", "timeline": "medium_term",
        "user_id": str(uuid4()),
    })
    assert resp.status_code == 422


def test_status_endpoint_maps_internal_status_to_external_vocabulary(client, test_db):
    run_id = uuid4()

    async def _seed():
        async with test_db() as db:
            stock = Stock(stock_id=uuid4(), canonical_ticker="AAPL", company_name="Apple",
                           primary_exchange="NASDAQ", currency="USD")
            db.add(stock)
            await db.commit()
            db.add(AnalysisRun(
                run_id=run_id, user_id=uuid4(), stock_id=stock.stock_id,
                account_type="trading", timeline="medium_term", triggered_by="manual",
                status=RunStatus.PASS1_RUNNING, llm_config={},
            ))
            await db.commit()

    asyncio.run(_seed())

    resp = client.get(f"/api/analysis/{run_id}/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "running"
    assert body["progress"]["pass"] == 1
    assert body["progress"]["agents_complete"] == 0


def test_status_endpoint_404s_on_unknown_id(client):
    resp = client.get(f"/api/analysis/{uuid4()}/status")
    assert resp.status_code == 404
    assert resp.json()["detail"]["error"]["code"] == "ANALYSIS_NOT_FOUND"


def test_result_endpoint_returns_recommendation_fields_when_completed(client, test_db):
    run_id = uuid4()

    async def _seed():
        async with test_db() as db:
            stock = Stock(stock_id=uuid4(), canonical_ticker="AAPL", company_name="Apple",
                           primary_exchange="NASDAQ", currency="USD")
            db.add(stock)
            await db.commit()
            run = AnalysisRun(
                run_id=run_id, user_id=uuid4(), stock_id=stock.stock_id,
                account_type="trading", timeline="medium_term", triggered_by="manual",
                status=RunStatus.COMPLETED, llm_config={}, disagreement_score=20,
                disagreement_class="consensus",
            )
            db.add(run)
            await db.commit()
            db.add(Recommendation(
                run_id=run_id, stock_id=stock.stock_id, stock_outlook_direction="somewhat_bullish",
                overall_confidence=70, account_recommendation={"action": "buy", "rationale": "r"},
                key_drivers=[], key_risks=[], synthesis_narrative="Good buy.",
                bull_case_strength=70, bear_case_strength=30,
            ))
            await db.commit()

    asyncio.run(_seed())

    resp = client.get(f"/api/analysis/{run_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "complete"
    assert body["stock_outlook_direction"] == "somewhat_bullish"
    assert body["synthesis_narrative"] == "Good buy."
    assert body["disagreement_score"] == 20


def test_result_endpoint_404s_on_unknown_id(client):
    resp = client.get(f"/api/analysis/{uuid4()}")
    assert resp.status_code == 404
