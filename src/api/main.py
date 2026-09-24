import sys

from dotenv import load_dotenv

# Real bug, confirmed live (86bbuhjup, running the full pipeline end-to-end
# for the first time): data/providers/edgartools.py and ca_crosslisting.py
# both read SP_EDGAR_IDENTITY from os.environ at MODULE IMPORT TIME into a
# cached module-level variable. .env only gets loaded into os.environ via
# load_dotenv(), previously called only as a side effect of importing
# data/providers/fmp.py -- so if edgartools.py (or ca_crosslisting.py) is
# imported anywhere in this app's import graph before fmp.py happens to be
# imported, its cached identity is permanently None for that process, even
# though .env genuinely has a real value (confirmed by reproducing this
# import-order race directly). Loading here, unconditionally and first,
# removes the ordering dependency instead of relying on which provider
# module some future import graph happens to touch first.
load_dotenv()

# Fixed 2026-08-04 (86bb7j0kh) — real bug, confirmed live: structlog's
# default logger writes to stdout/stderr using the process's default
# encoding, which on a plain Windows console is cp1252, not UTF-8. Any
# logged exception whose text/traceback contains a non-cp1252 character
# (confirmed live via router.py's own exception-handling logger.warning(
# ..., exc_info=True) call) raises UnicodeEncodeError — crashing the very
# fallback-catching code path that's supposed to degrade gracefully.
# structlog is never explicitly configured anywhere in this codebase, so
# there's no central place to patch its formatter; reconfiguring the
# underlying streams once here fixes it regardless of how/whether
# structlog ever gets a real config. Must run before anything else logs.
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

from fastapi import FastAPI  # noqa: E402 — must follow the encoding fix above
from api.database import Base, engine  # noqa: E402
from api.routes import analysis, user_profile  # noqa: E402

# Base.metadata.create_all() only creates tables for models that have been
# imported somewhere -- SQLAlchemy's declarative registry only knows about a
# model once its module has actually run (86bbuhjup, database-persistence-
# audit.md finding: this file used to import only user_profile, so 24 of 25
# coded ORM models were never live despite being fully coded).
#
# This imports the 7 tables the orchestrator actually writes to -- not the
# full ~25-table audit backlog, which stays deferred per the "get the CIO
# pipeline running first, storage/tracking comes afterwards" steer. The set
# is 7, not the 5 the ticket named, confirmed empirically (not guessed):
# AnalysisRun/AgentOutput/Recommendation/Prediction/ShadowPrediction alone
# raise `InvalidRequestError` from `configure_mappers()` the first time any
# ORM query touches one of these models, because AnalysisRun.stock and
# Prediction.checkpoints are relationship() forward-refs to Stock and
# PredictionCheckpoint -- neither of which create_all() itself needs, but
# both of which SQLAlchemy's mapper configuration does, transitively, the
# moment the app actually uses these models rather than just creating their
# tables. Importing all 7 together is what makes both create_all() AND
# real ORM use succeed, not just the DDL step.
#
# The real live count is 8, not 7 -- found live while generating the
# Alembic baseline migration (86bbwachy prerequisite): AnalysisRun.user_id
# has a real FK to user_profiles, which this file already makes live too,
# just via the `user_profile` route import on the line above rather than
# through this explicit table-import block. alembic/env.py's own model
# imports mirror this file's true effective set (8 tables), not just this
# block's own 7 -- keep both lists in sync when either changes.
from api.tables.stock import Stock  # noqa: E402, F401
from api.tables.analysis_runs import AnalysisRun  # noqa: E402, F401
from api.tables.agent_outputs import AgentOutput  # noqa: E402, F401
from api.tables.recommendations import Recommendation  # noqa: E402, F401
from api.tables.predictions import Prediction  # noqa: E402, F401
from api.tables.prediction_checkpoints import PredictionCheckpoint  # noqa: E402, F401
from api.tables.shadow_predictions import ShadowPrediction  # noqa: E402, F401

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(user_profile.router)
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
