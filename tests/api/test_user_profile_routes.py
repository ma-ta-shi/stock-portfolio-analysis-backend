"""Tests for api/routes/user_profile.py (86bc8efe7) -- the modernized async
vertical (schemas/CRUD/routes) plus the new tax-context, room, and
Portfolio-Optimizer-input fields on UserProfile. Real in-memory SQLite via
TestClient, matching tests/api/test_analysis_routes.py's own established
pattern -- not mocked DB calls.
"""
import asyncio

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from api.database import Base
from api.main import app
from api.routes.user_profile import get_async_db
from api.tables.user_profile import UserProfile  # noqa: F401


@pytest.fixture
def test_db():
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
    yield TestClient(app)


def test_create_profile_returns_new_fields_as_none(client):
    resp = client.post("/", json={"display_name": "Test User"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["display_name"] == "Test User"
    assert body["province"] is None
    assert body["tfsa_room_remaining"] is None
    assert body["time_sensitive_cash_needs"] is None


def test_create_accepts_the_full_field_set_in_one_call(client):
    resp = client.post("/", json={
        "display_name": "Test User",
        "province": "ON",
        "income_annual": 95000.0,
        "tfsa_room_remaining": 12000.0,
        "time_sensitive_cash_needs": [
            {"description": "Down payment", "amount": 50000.0, "deadline": "2027-03-01"},
        ],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["province"] == "ON"
    assert body["income_annual"] == 95000.0
    assert body["tfsa_room_remaining"] == 12000.0
    assert body["time_sensitive_cash_needs"][0]["description"] == "Down payment"


def test_get_by_id_returns_the_real_profile(client):
    created = client.post("/", json={"display_name": "Test User"}).json()
    resp = client.get(f"/{created['user_id']}")
    assert resp.status_code == 200
    assert resp.json()["user_id"] == created["user_id"]


def test_get_by_id_404s_for_unknown_user(client):
    resp = client.get("/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


def test_patch_sets_tax_context_and_room(client):
    created = client.post("/", json={"display_name": "Test User"}).json()
    resp = client.patch(
        f"/{created['user_id']}",
        json={
            "province": "ON",
            "income_annual": 95000.0,
            "tfsa_room_remaining": 12000.0,
            "tfsa_room_as_of": "2026-09-28T00:00:00",
            "rrsp_room_remaining": 30000.0,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["province"] == "ON"
    assert body["income_annual"] == 95000.0
    assert body["tfsa_room_remaining"] == 12000.0
    assert body["rrsp_room_remaining"] == 30000.0
    # unset fields on this PATCH are untouched, not nulled
    assert body["display_name"] == "Test User"


def test_patch_rejects_unsupported_province(client):
    created = client.post("/", json={"display_name": "Test User"}).json()
    resp = client.patch(f"/{created['user_id']}", json={"province": "BC"})
    assert resp.status_code == 422


def test_patch_time_sensitive_cash_needs_replaces_whole_list(client):
    created = client.post("/", json={"display_name": "Test User"}).json()
    first = client.patch(
        f"/{created['user_id']}",
        json={"time_sensitive_cash_needs": [
            {"description": "Down payment", "amount": 50000.0, "deadline": "2027-03-01"},
        ]},
    ).json()
    assert len(first["time_sensitive_cash_needs"]) == 1

    second = client.patch(
        f"/{created['user_id']}",
        json={"time_sensitive_cash_needs": [
            {"description": "New need", "amount": 1000.0, "deadline": "2027-06-01"},
        ]},
    ).json()
    # replaced, not appended
    assert len(second["time_sensitive_cash_needs"]) == 1
    assert second["time_sensitive_cash_needs"][0]["description"] == "New need"


def test_sequential_patches_do_not_clobber_each_others_fields(client):
    created = client.post("/", json={"display_name": "Test User"}).json()
    user_id = created["user_id"]

    client.patch(f"/{user_id}", json={"province": "ON"})
    second = client.patch(f"/{user_id}", json={"income_annual": 95000.0}).json()

    # the second PATCH never mentioned province -- exclude_unset must leave it alone
    assert second["province"] == "ON"
    assert second["income_annual"] == 95000.0


def test_patch_404s_for_unknown_user(client):
    resp = client.patch("/00000000-0000-0000-0000-000000000000", json={"income_annual": 1.0})
    assert resp.status_code == 404


def test_delete_removes_the_profile(client):
    created = client.post("/", json={"display_name": "Test Profile"}).json()
    user_id = created["user_id"]

    resp = client.delete(f"/{user_id}")
    assert resp.status_code == 204

    # actually gone, not just reporting success
    assert client.get(f"/{user_id}").status_code == 404


def test_delete_404s_for_unknown_user(client):
    resp = client.delete("/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


def test_get_users_still_lists_everyone(client):
    client.post("/", json={"display_name": "User A"})
    client.post("/", json={"display_name": "User B"})
    resp = client.get("/")
    assert resp.status_code == 200
    assert len(resp.json()) == 2
