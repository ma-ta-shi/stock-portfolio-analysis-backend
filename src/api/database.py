from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./app.db"
ASYNC_DATABASE_URL = "sqlite+aiosqlite:///./app.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()

# Async engine/session (86bawpty3) — added alongside the sync ones above rather than replacing
# them: main.py's own Base.metadata.create_all(bind=engine) startup call and user_profile.py's
# existing route both depend on the sync engine/SessionLocal, and this ticket's own scope is
# DataPipeline.prepare()'s real query need, not migrating the rest of the app. CLAUDE.md's own
# rule ("Always use AsyncSession — never sync Session") and its "Database" section (SQLite +
# aiosqlite for dev) already named this as the intended setup; aiosqlite was already installed
# and in requirements.txt, just never wired up. Same physical app.db file either way.
async_engine = create_async_engine(ASYNC_DATABASE_URL)

AsyncSessionLocal = async_sessionmaker(bind=async_engine, autoflush=False, expire_on_commit=False)
