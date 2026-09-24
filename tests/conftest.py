"""Pytest configuration for tests.

Adds src/ to sys.path so that imports like `from data.providers.base import ...`
work correctly in test modules, regardless of where pytest is invoked from.
"""

import sys
from pathlib import Path

# Same fix as src/api/main.py (86bb7j0kh) - a caught, logged exception whose
# text contains a non-cp1252 character crashes structlog's print on a plain
# Windows console instead of logging, which silently turns graceful
# degradation (Router._try_chain's own "log and try next provider") into a
# fatal crash. main.py's fix only runs for the real app; nothing protected
# test runs, where the same masking was confirmed live 86bawptye. Must run
# before anything else in the test session logs.
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Mapper-reachability imports for every currently-live table (mirrors
# api/main.py's own "9, not 7" import list, plus UserProfile -- live there
# only via the user_profile route import, which nothing in the test suite
# pulls in on its own). Constructing a real ORM instance from any of these
# (AnalysisRun, LLMCall, ...) triggers SQLAlchemy's full mapper
# configuration pass, which needs every class its own string-quoted
# relationship() forward refs name (e.g. Mapped[Optional["AnalysisRun"]])
# importable SOMEWHERE in the process by then. Centralized here, in
# conftest.py (auto-loaded for every test session regardless of which file
# runs, alone or as part of the full suite), instead of each test file that
# happens to construct one of these repeating the same noqa import block --
# found duplicated three times over (test_orchestrator.py, test_pipeline.py,
# test_llm_calls.py) before being consolidated here (86bbwachy Phase 3
# review). Deliberately just these 8 (+ UserProfile), not all ~28 coded
# table files in api/tables/ -- importing every scaffold table here would
# quietly make them ALL "live" for the mapper registry, blurring the
# project's own established "a coded table isn't real until it's
# deliberately imported" distinction (see main.py's own comment on this).
from api.tables.agent_outputs import AgentOutput  # noqa: E402, F401
from api.tables.analysis_runs import AnalysisRun  # noqa: E402, F401
from api.tables.llm_calls import LLMCall  # noqa: E402, F401
from api.tables.prediction_checkpoints import PredictionCheckpoint  # noqa: E402, F401
from api.tables.predictions import Prediction  # noqa: E402, F401
from api.tables.recommendations import Recommendation  # noqa: E402, F401
from api.tables.shadow_predictions import ShadowPrediction  # noqa: E402, F401
from api.tables.stock import Stock  # noqa: E402, F401
from api.tables.user_profile import UserProfile  # noqa: E402, F401
