"""Shared start-up for the read-side scripts (diagnose_run, recent_errors, run_batch).

`import _bootstrap` as the FIRST import: it makes stdout/stderr UTF-8, puts
backend/src on the path, and imports every table the report queries touch.

- UTF-8: same fix as api/main.py and analyze.py; a Windows console otherwise raises
  UnicodeEncodeError on non-ASCII text in an error message.
- Table imports: SQLAlchemy resolves relationship() targets by name when a mapper
  is first used, so every class reachable from AnalysisRun must already be
  imported (see run_trace.py). Importing api.database directly, never api.main,
  because api/main.py runs Base.metadata.create_all against the real app.db.
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT / "src"))

from api.tables.agent_outputs import AgentOutput  # noqa: E402, F401
from api.tables.analysis_runs import AnalysisRun  # noqa: E402, F401
from api.tables.error_records import ErrorRecord  # noqa: E402, F401
from api.tables.llm_calls import LLMCall  # noqa: E402, F401
from api.tables.prediction_checkpoints import PredictionCheckpoint  # noqa: E402, F401
from api.tables.predictions import Prediction  # noqa: E402, F401
from api.tables.recommendations import Recommendation  # noqa: E402, F401
from api.tables.run_quality_summary import RunQualitySummary  # noqa: E402, F401
from api.tables.stock import Stock  # noqa: E402, F401
