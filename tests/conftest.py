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
