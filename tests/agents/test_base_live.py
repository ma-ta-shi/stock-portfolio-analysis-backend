"""Real agent prompt, real fixture, real Ollama call.

Not a synthetic connectivity smoke test: this drives the actual Technical
Analyst prompt from `simulation/runners/pass1_technical_analyst.py` and a
real historical fixture through the newly-ported `agents.base.BaseRunner`,
then checks the result against `simulation/results_gptoss_v1/scenario_02/
pass1_outputs.json` -- a baseline the original (sync, `requests`-based)
harness already captured for this exact prompt/fixture/model combination.
That's the closest available proof that the async port preserved behavior
for the shape of calls it will actually carry, per ClickUp 86bc2d3t9's own
"verify against real, live data" / "test through the actual code path"
requirements.

Lives outside backend/tests/live/ deliberately -- that directory's
conftest.py autouse fixture skips everything unless FMP/Finnhub/EDGAR/FRED
keys are set, which is the wrong gate for a local-Ollama dependency. Still
opt-in via the same `live` marker (pyproject.toml's `addopts = "-m 'not
live'"` excludes it from a default `pytest` run; `pytest -m live` runs it).
"""

import json
import sys
from pathlib import Path

import pytest

from agents.base import MODEL, OLLAMA_HOST, BaseRunner, OllamaUnavailable, _preflight

pytestmark = pytest.mark.live

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Deliberately imported from the old harness, not reimplemented -- this test
# exists to compare the new runner against the prompt/validator the original
# harness actually used, not a reconstruction of it. Prompt content itself is
# 86bbdutn6's concern, not this one.
from simulation.runners.pass1_technical_analyst import (  # noqa: E402
    SYSTEM_PROMPT,
    build_user_message,
)
from simulation.validators.pass1 import validate_technical_analyst  # noqa: E402

_FIXTURE_PATH = _REPO_ROOT / "simulation" / "fixtures" / "scenario_02_canadian_large_cap.json"
_BASELINE_PATH = (
    _REPO_ROOT / "simulation" / "results_gptoss_v1" / "scenario_02" / "pass1_outputs.json"
)


@pytest.fixture(autouse=True)
async def _require_ollama():
    # Reuses the real production _preflight() rather than hand-rolling a
    # second "is Ollama up and is the model pulled" check that could drift
    # out of sync with it -- and as a side effect, this is the only place
    # _preflight() itself gets exercised against a real Ollama instance
    # rather than a fake. Independent of backend/tests/live/conftest.py's
    # autouse fixture (which gates on FMP/Finnhub/EDGAR/FRED keys, not this)
    # -- see module docstring.
    try:
        await _preflight()
    except OllamaUnavailable as exc:
        pytest.skip(f"{exc} (host={OLLAMA_HOST}, model={MODEL!r})")


@pytest.mark.asyncio
async def test_technical_analyst_prompt_runs_end_to_end_against_real_ollama():
    fixture = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    baseline = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))["TECH"]

    async with BaseRunner() as runner:
        result, errors = await runner.call_with_validation(
            SYSTEM_PROMPT,
            build_user_message(fixture),
            validate_technical_analyst,
            max_tokens=3500,
            temperature=0.3,
        )

    # A single failure here can be real model noise (the audit findings docs
    # already document run-to-run variance on this model) -- re-run before
    # treating it as a port defect, same posture the audit itself takes.
    assert errors == [], f"real prompt/fixture failed validation: {errors}"

    # Structural regression check against a known-good prior run of this exact
    # prompt/fixture/model combination -- content is expected to differ
    # (real, not mocked, model output), shape is not.
    assert set(result.keys()) == set(baseline.keys())

    summary = runner.timing_summary()
    assert summary["calls"] >= 1
    assert summary["total_llm_s"] > 0
