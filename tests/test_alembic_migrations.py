"""Confirms Alembic's baseline migration (86bbwachy prerequisite) actually
does what it claims: `alembic upgrade head` on a fresh DB produces a schema
equivalent to `Base.metadata.create_all()`'s own output, and `alembic
downgrade base` reverses cleanly. A permanent regression test, not just the
one-time manual check this migration was authored against -- catches the
exact mistake alembic/env.py's own docstring warns about (a future table
added to main.py's import list but not env.py's, or vice versa) the moment
it happens, not whenever someone next runs autogenerate by hand.

Uses the real `alembic` Python API (not a subprocess shell-out) against a
real temp on-disk SQLite file -- `:memory:` won't work here since Alembic's
own async engine creation opens its own separate connection per call, and an
in-memory SQLite DB is only visible to the connection that created it.
"""
import os
import sqlite3
from pathlib import Path

import pytest
import sqlalchemy
from alembic import command
from alembic.config import Config

from api.database import Base
import api.tables.stock  # noqa: F401
import api.tables.analysis_runs  # noqa: F401
import api.tables.agent_outputs  # noqa: F401
import api.tables.recommendations  # noqa: F401
import api.tables.predictions  # noqa: F401
import api.tables.prediction_checkpoints  # noqa: F401
import api.tables.shadow_predictions  # noqa: F401
import api.tables.user_profile  # noqa: F401

_BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _schema_dump(db_path: Path) -> dict[tuple[str, str], str]:
    """(type, name) -> CREATE statement, for every real table/index --
    excludes sqlite's own internal objects and alembic_version, which only
    exists on the migration-managed side."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE type IN ('table', 'index') AND name NOT LIKE 'sqlite_%' "
            "AND name != 'alembic_version'"
        ).fetchall()
    finally:
        conn.close()
    return {(t, n): sql for t, n, sql in rows}


def _normalize(sql: str | None) -> str | None:
    """Collapse whitespace AND top-level clause ORDER differences between
    create_all()'s and Alembic's DDL emission -- confirmed live (86bbwachy
    prerequisite) that the two are semantically identical but cosmetically
    different in both dimensions: whitespace/line breaks, and the order of
    trailing constraint clauses (FOREIGN KEY / UNIQUE) within a CREATE TABLE
    statement. A plain whitespace-only normalize still flagged those as
    "different" (caught live by this test's own first run) -- this also
    splits the parenthesized column/constraint list on top-level commas and
    sorts it, so `... UNIQUE (x), FOREIGN KEY(y) ...` and
    `... FOREIGN KEY(y), UNIQUE (x) ...` compare equal. CREATE INDEX
    statements have no such clause list and just get whitespace-collapsed.
    """
    if sql is None:
        return None
    collapsed = " ".join(sql.split())
    open_paren = collapsed.find("(")
    if open_paren == -1 or not collapsed.rstrip().endswith(")"):
        return collapsed  # not a "NAME (...)" shape -- nothing to reorder
    preamble = collapsed[:open_paren]
    body = collapsed[open_paren + 1 : collapsed.rfind(")")]

    clauses, depth, start = [], 0, 0
    for i, ch in enumerate(body):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            clauses.append(body[start:i].strip())
            start = i + 1
    clauses.append(body[start:].strip())
    return f"{preamble}({', '.join(sorted(clauses))})"


@pytest.fixture
def alembic_config(tmp_path) -> tuple[Config, Path]:
    db_path = tmp_path / "alembic_test.db"
    cfg = Config(str(_BACKEND_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(_BACKEND_ROOT / "alembic"))
    os.environ["SP_ALEMBIC_DB_URL"] = f"sqlite+aiosqlite:///{db_path}"
    try:
        yield cfg, db_path
    finally:
        os.environ.pop("SP_ALEMBIC_DB_URL", None)


def test_alembic_upgrade_matches_create_all(alembic_config, tmp_path):
    cfg, alembic_db_path = alembic_config
    command.upgrade(cfg, "head")

    create_all_db_path = tmp_path / "create_all_test.db"
    engine = sqlalchemy.create_engine(f"sqlite:///{create_all_db_path}")
    Base.metadata.create_all(bind=engine)
    engine.dispose()

    alembic_schema = _schema_dump(alembic_db_path)
    create_all_schema = _schema_dump(create_all_db_path)

    assert set(alembic_schema) == set(create_all_schema), (
        "alembic upgrade head and Base.metadata.create_all() produced different "
        "table/index sets -- almost always means a table was added to one of "
        "main.py's or alembic/env.py's import lists but not the other."
    )
    mismatches = {
        key: (alembic_schema[key], create_all_schema[key])
        for key in alembic_schema
        if _normalize(alembic_schema[key]) != _normalize(create_all_schema[key])
    }
    assert not mismatches, f"real schema differences (not just formatting): {mismatches}"


def test_alembic_downgrade_reverses_cleanly(alembic_config):
    cfg, db_path = alembic_config
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")

    remaining = _schema_dump(db_path)
    assert remaining == {}, f"downgrade left real tables/indexes behind: {remaining}"
