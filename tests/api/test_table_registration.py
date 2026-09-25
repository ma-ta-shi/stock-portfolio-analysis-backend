"""Confirms the DB import-chain fix (86bbuhjup): the real, verified 7-table
dependency closure the orchestrator needs is sufficient for both DDL
creation and ORM mapper configuration -- not the 5 tables the ticket
originally named (confirmed live during this port: importing only
AnalysisRun/AgentOutput/Recommendation/Prediction/ShadowPrediction raises
InvalidRequestError from configure_mappers(), since AnalysisRun.stock and
Prediction.checkpoints are relationship() forward-refs to Stock and
PredictionCheckpoint).

Deliberately imports the 7 table modules directly, the same set api.main
imports, rather than importing api.main itself -- api.main calls
Base.metadata.create_all(bind=engine) unconditionally at IMPORT time
against the real app.db (a pre-existing structural property of that module,
not something this port changed), so importing it here would side-effect
the real database file on every test collection. Uses a throwaway in-memory
engine for the same reason -- Base.metadata is shared/process-global once
these modules are imported, so the isolation has to be on the engine side,
not the metadata side.
"""
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import configure_mappers

from api.database import Base
from api.tables.analysis_runs import AnalysisRun, RunStatus  # noqa: F401
from api.tables.agent_outputs import AgentOutput  # noqa: F401
from api.tables.llm_calls import LLMCall  # noqa: F401
from api.tables.prediction_checkpoints import PredictionCheckpoint  # noqa: F401
from api.tables.predictions import Prediction  # noqa: F401
from api.tables.recommendations import Recommendation  # noqa: F401
from api.tables.run_quality_summary import RunQualitySummary  # noqa: F401
from api.tables.shadow_predictions import ShadowPrediction  # noqa: F401
from api.tables.stock import Stock  # noqa: F401
from api.tables.user_profile import UserProfile  # noqa: F401

# The original 8 from the 86bbuhjup finding this file's own docstring
# describes, plus llm_calls (86bbwachy Phase 2) and run_quality_summary
# (86bbwachy Phase 5) -- both landed after this file was written and neither
# was added here at the time, a real, pre-existing gap this review caught
# and closed rather than left to grow further. The `<=` check below meant
# neither omission ever failed a test -- this set was never a completeness
# guarantee, just a floor -- but a table-registration test whose own set
# quietly stops tracking real tables is worth keeping current regardless.
_EXPECTED_TABLES = {
    "user_profiles",
    "stocks",
    "analysis_runs",
    "agent_outputs",
    "recommendations",
    "predictions",
    "prediction_checkpoints",
    "shadow_predictions",
    "llm_calls",
    "run_quality_summary",
}


def test_all_ten_real_tables_are_registered():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    tables = set(inspect(engine).get_table_names())
    assert _EXPECTED_TABLES <= tables


def test_configure_mappers_succeeds():
    """The real regression this fix targets: AnalysisRun.stock and
    Prediction.checkpoints are relationship() forward-refs to Stock and
    PredictionCheckpoint -- create_all() alone doesn't need either imported,
    but the first real ORM use of any of these models does. Confirmed live
    (86bbuhjup) that importing only the 5 tables the ticket originally named
    raises InvalidRequestError here."""
    configure_mappers()  # raises on failure; no assertion needed


def test_run_status_enum_has_the_documented_value_set():
    assert {s.value for s in RunStatus} == {
        "queued", "pass1_running", "pass1_complete",
        "pass2_running", "pass2_complete", "synthesis_running",
        "completed", "failed",
    }


def test_run_status_values_are_plain_strings_for_db_storage():
    """The column itself stays a plain String(30), not a DB-level enum --
    confirm the enum members serialize as their bare string value, so
    assigning run.status = RunStatus.QUEUED stores "queued", not
    "RunStatus.QUEUED"."""
    assert str(RunStatus.QUEUED) == "queued"
    assert RunStatus.QUEUED == "queued"
