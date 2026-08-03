"""Pytest configuration for tests.

Adds src/ to sys.path so that imports like `from data.providers.base import ...`
work correctly in test modules, regardless of where pytest is invoked from.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
