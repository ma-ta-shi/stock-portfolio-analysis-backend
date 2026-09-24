import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Same src/-on-path mechanism as pyproject.toml's [tool.pytest.ini_options]
# pythonpath = ["src"] -- Alembic has no equivalent auto-config, so this does
# it manually. Must happen before any `api.*`/`data.*` import below.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Real, live-table import list, matching api/main.py's own actual effective
# set (86bbwachy prerequisite) -- deliberately NOT all ~25 coded table files
# under api/tables/. This project's own established convention (confirmed
# twice during 86bbuhjup and again while planning 86bbwachy): a coded table
# isn't a real, live table until something explicitly imports it, since
# Base.metadata.create_all() (and, from here on, Alembic autogenerate) only
# acts on models that have actually been imported into the process. Baselining
# against the aspirational ~25 would silently promote ~17 never-wired tables
# (market_regime_history, equity_curve_snapshots, etc.) to "real" without that
# being a deliberate decision.
#
# 8 tables, not main.py's own explicitly-commented "7" -- found live running
# this exact baseline generation: main.py's own table-import comment block
# only lists 7, but AnalysisRun.user_id has a real FK to user_profiles, which
# main.py registers indirectly (importing the user_profile ROUTE, which
# imports the UserProfile table, not through the explicit table-import
# block). Autogenerate fails immediately with NoReferencedTableError without
# it -- a real gap in main.py's own comment, not just this file's.
#
# When a future ticket wires a new table into main.py (by any import path),
# add the same import here in the same change -- see this file's own git
# history for 86bbwachy's own tables landing this way.
from api.database import ASYNC_DATABASE_URL, Base  # noqa: E402
from api.tables.stock import Stock  # noqa: E402, F401
from api.tables.analysis_runs import AnalysisRun  # noqa: E402, F401
from api.tables.agent_outputs import AgentOutput  # noqa: E402, F401
from api.tables.recommendations import Recommendation  # noqa: E402, F401
from api.tables.predictions import Prediction  # noqa: E402, F401
from api.tables.prediction_checkpoints import PredictionCheckpoint  # noqa: E402, F401
from api.tables.shadow_predictions import ShadowPrediction  # noqa: E402, F401
from api.tables.user_profile import UserProfile  # noqa: E402, F401
from api.tables.llm_calls import LLMCall  # noqa: E402, F401

target_metadata = Base.metadata

# Single source of truth for the DB URL -- read from api/database.py's own
# ASYNC_DATABASE_URL rather than duplicating the connection string in
# alembic.ini, so the two can never silently drift apart. SP_ALEMBIC_DB_URL
# is an escape hatch for generating/testing a migration against a scratch DB
# (e.g. an empty file, to autogenerate a from-scratch CREATE migration)
# without touching the real app.db -- unset in normal use.
config.set_main_option(
    "sqlalchemy.url", os.environ.get("SP_ALEMBIC_DB_URL", ASYNC_DATABASE_URL)
)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """In this scenario we need to create an Engine
    and associate a connection with the context.

    """

    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""

    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
