"""Real agent prompt, real fixture, real Ollama call.

Not a synthetic connectivity smoke test: this drives the actual, documented
Technical Analyst prompt through the newly-ported `agents.base.BaseRunner`
and checks the result against the real schema's own declared top-level keys.
That's the closest available proof the async port preserved behavior for the
shape of calls it will actually carry, per ClickUp 86bc2d3t9's own "verify
against real, live data" / "test through the actual code path" requirements.

Updated for ClickUp 86bbdutn6: originally imported `SYSTEM_PROMPT` directly
from `simulation/runners/pass1_technical_analyst.py` as its source of "the
real prompt" -- ironic given that constant was exactly the independently-
authored, diverged-from-the-documented-prompt embedded string 86bbdutn6
exists to retire. It's deleted now (correctly), so this test uses
`agents.prompts.load_template()`, the actual mechanism that replaced it, and
fills the same real placeholders `simulation/runners/
pass1_technical_analyst.py`'s own rewired `run()` now fills. Also dropped the
old shape comparison against `simulation/results_gptoss_v1/scenario_02/
pass1_outputs.json`: that baseline was captured running the OLD embedded
prompt, whose schema genuinely differs from the real one (`structured_data` +
`pass2_view` vs. the real prompt's single `interpretive_fields`) -- exactly
the "existing results become non-comparable" cost 86bbdutn6 names explicitly,
not a regression to chase.

Lives outside backend/tests/live/ deliberately -- that directory's
conftest.py autouse fixture skips everything unless FMP/Finnhub/EDGAR/FRED
keys are set, which is the wrong gate for a local-Ollama dependency. Still
opt-in via the same `live` marker (pyproject.toml's `addopts = "-m 'not
live'"` excludes it from a default `pytest` run; `pytest -m live` runs it).
"""

import json
from pathlib import Path

import pytest

from agents.base import MODEL, OLLAMA_HOST, BaseRunner, OllamaUnavailable, _preflight
from agents.prompts import fill, load_template

pytestmark = pytest.mark.live

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Validator still borrowed from the old harness rather than reimplemented --
# that part hasn't changed, only where the prompt text itself comes from.
import sys  # noqa: E402

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from simulation.validators.pass1 import validate_technical_analyst  # noqa: E402

_FIXTURE_PATH = _REPO_ROOT / "simulation" / "fixtures" / "scenario_02_canadian_large_cap.json"


def _build_user_message(fixture: dict) -> str:
    """Small local copy of pass1_technical_analyst.py's own user-message
    builder -- not imported, since this test now deliberately has zero
    dependency on that module (it's testing agents.prompts, not it)."""
    ctx = fixture["context"]
    price = fixture["price_data"]
    tech = fixture["technical_data"]
    fund = fixture["fundamental_data"]
    base_scores = fixture["base_reliability_scores"]
    return (
        f"{ctx['ticker']} ({ctx['company_name']}) | {ctx['sector']} | "
        f"{ctx['primary_exchange']} | {ctx['currency']}\n"
        f"Timeline: {ctx['timeline']} | Account: {ctx['account_type']} | "
        f"As of: {ctx['data_timestamp']}\n\n"
        f"BASE RELIABILITY SCORE: {base_scores['TECH']}/100\n"
        f"EARNINGS PROXIMITY: {fund.get('earnings_proximity_days', 999)} days\n\n"
        f"PRICE DATA:\n  Current: {price['current_price']} {ctx['currency']}\n"
        f"  52w High: {price['price_52w_high']} | 52w Low: {price['price_52w_low']}\n"
        f"TREND: {tech.get('primary_trend', 'N/A')} | "
        f"Support: {tech.get('nearest_support', 'N/A')} | "
        f"Resistance: {tech.get('nearest_resistance', 'N/A')}"
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
    ctx = fixture["context"]
    tech = fixture["technical_data"]
    fund = fixture["fundamental_data"]

    system_prompt = fill(
        load_template("technical_analyst"),
        {
            "canonical_ticker": ctx["ticker"],
            "company_name": ctx["company_name"],
            "sector": ctx["sector"],
            "timeline": ctx["timeline"],
            "timeline_instruction": f"Timeline: {ctx['timeline']}.",
            "data_coverage_line": "Data coverage: standard.",
            "data_warnings": "",
            "memory_brief": "",
            "earnings_proximity_days": str(fund.get("earnings_proximity_days", "N/A")),
            "nearest_support": str(tech.get("nearest_support", "N/A")),
            "nearest_resistance": str(tech.get("nearest_resistance", "N/A")),
            "weekly_trend": str(tech.get("weekly_trend", "N/A")),
            "rs_leadership": "N/A",
            "rsi_zone_adjusted": "N/A",
            "volatility_regime_derived": str(tech.get("volatility_regime", "N/A")),
            "weekly_rsi_zone": "N/A",
        },
    )

    async with BaseRunner() as runner:
        result, errors = await runner.call_with_validation(
            system_prompt,
            _build_user_message(fixture),
            validate_technical_analyst,
            max_tokens=3500,
            temperature=0.3,
        )

    # A single failure here can be real model noise (the audit findings docs
    # already document run-to-run variance on this model) -- re-run before
    # treating it as a port defect, same posture the audit itself takes.
    assert errors == [], f"real prompt/fixture failed validation: {errors}"

    # NOT compared against `baseline`'s top-level keys anymore. `baseline` was
    # captured running the OLD embedded prompt, which used `structured_data` +
    # `pass2_view` as separate top-level keys; the real documented prompt
    # nests the equivalent content under a single `interpretive_fields` key
    # instead -- a real, correct shape difference, not a regression. This is
    # exactly the "existing results become non-comparable" cost ClickUp
    # 86bbdutn6 names explicitly. The real check is against the actual
    # documented schema's own declared keys.
    # Subset, not exact-set-equality. Confirmed via the retry log, not
    # guessed: `validate_technical_analyst` (simulation/validators/pass1.py)
    # still requires `reliability_score`/`data_quality_assessment` as
    # required fields -- it predates the D6 migration (ClickUp 86bbummwp,
    # separate, not-yet-done ticket) that superseded them with
    # `analysis_confidence`. Attempt 1 correctly omitted them per the real
    # prompt; the retry message then told the model they were "missing",
    # and it added them on attempt 2 to satisfy that stale validator. The
    # extra fields are a validator/prompt mismatch this ticket doesn't own
    # fixing, not the extraction/loading mechanism misbehaving.
    real_schema_fields = {
        "assessment_summary",
        "analysis_confidence",
        "caveats",
        "key_factors",
        "risks",
        "narrative",
        "interpretive_fields",
    }
    assert real_schema_fields <= set(result.keys())

    summary = runner.timing_summary()
    assert summary["calls"] >= 1
    assert summary["total_llm_s"] > 0
