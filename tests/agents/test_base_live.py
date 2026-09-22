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
from simulation.runners.compression import (  # noqa: E402
    build_pass2_user_message,
    compress_pass1_outputs,
    extract_confidence_levels,
)
from simulation.runners.pass1_fundamental_analyst import FundamentalAnalystRunner  # noqa: E402
from simulation.runners.pass1_macro_economist import MacroEconomistRunner  # noqa: E402
from simulation.runners.pass1_sentiment_analyst import SentimentAnalystRunner  # noqa: E402
from simulation.runners.pass1_stock_researcher import StockResearcherRunner  # noqa: E402
from simulation.runners.pass1_technical_analyst import TechnicalAnalystRunner  # noqa: E402
from simulation.utils import build_pass1_reliability_warnings, researcher_thesis_archetype  # noqa: E402
from simulation.validators.pass1 import validate_technical_analyst  # noqa: E402
from simulation.validators.pass2 import (  # noqa: E402
    validate_risk_advisor_stage_a,
    validate_risk_advisor_stage_b,
)

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


def _risk_advisor_precomputed_metrics(fixture: dict) -> str:
    """Small local copy of pass2_risk_advisor.py's own
    _precomputed_risk_metrics() -- not imported, for the same reason
    test_technical_analyst_prompt_runs_end_to_end_against_real_ollama's own
    _build_user_message is a local copy: this test has zero dependency on
    that runner module, only on the real prompt templates and the shared
    compression utilities."""
    price = fixture["price_data"]
    oc = fixture.get("orchestrator_precomputed", {})
    return (
        f"BETA: beta = {price.get('beta', 'N/A')} (will be injected into output by orchestrator)\n"
        f"VOL:  52w High={price['price_52w_high']}, 52w Low={price['price_52w_low']}, "
        f"YTD={price['ytd_return_pct']}%\n"
        f"DD:   Max drawdown (1yr) = {oc.get('max_drawdown_1yr_pct', 'N/A')}% "
        f"(will be injected by orchestrator)\n"
        f"LIQ:  Avg dollar volume (20d) = ${price.get('avg_dollar_volume_20', 0):,.0f}\n"
        f"CORR: Portfolio correlation = unknown (not provided for this analysis)\n"
        f"CONC: Standard concentration risk assessment based on position sizing"
    )


@pytest.mark.asyncio
async def test_risk_advisor_stage_a_to_b_continuation_against_real_ollama():
    """86bc2d414: the actual deliverable of this ticket, proven end to end --
    real Risk Advisor Stage A and Stage B prompts, a real fixture, through the
    real `call_with_validation_start`/`call_with_validation_continue` methods
    against real Ollama, not the raw `requests.post()` probe used during that
    ticket's planning (which proved the underlying Ollama API behaves as the
    spec doc claims, but never exercised this specific interface).

    Pass 1 (5 agents) runs through the simulation harness purely to generate
    realistic input, same posture as this file's Technical Analyst test
    reusing simulation validators -- no harness BaseRunner code is under test
    here, only the production `agents.base.BaseRunner`.
    """
    fixture = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    pass1 = {}
    for agent_id, cls in {
        "RSRCH": StockResearcherRunner,
        "FUND": FundamentalAnalystRunner,
        "TECH": TechnicalAnalystRunner,
        "SENT": SentimentAnalystRunner,
        "MACRO": MacroEconomistRunner,
    }.items():
        result, errors = cls().run(fixture)
        pass1[agent_id] = result
    compressed = compress_pass1_outputs(pass1)

    ctx = fixture["context"]
    acct = ctx["account_type"]
    stage_a_system = fill(
        load_template("risk_advisor", stage="a"),
        {
            "ticker": ctx["ticker"],
            "company_name": ctx["company_name"],
            "sector": ctx["sector"],
            "timeline": ctx["timeline"],
            "timeline_instruction": f"Timeline: {ctx['timeline']}.",
            "precomputed_risk_metrics": _risk_advisor_precomputed_metrics(fixture),
            "researcher_thesis_archetype": researcher_thesis_archetype(compressed),
            "pass1_reliability_warnings": build_pass1_reliability_warnings(
                extract_confidence_levels(compressed)
            ) or "(none)",
            "accuracy_brief": "",
            "winning_patterns_brief": "",
            "memory_brief": "",
        },
    )
    stage_a_user = build_pass2_user_message(fixture, compressed, acct)

    async with BaseRunner() as runner:
        stage_a_result, stage_a_errors, context = await runner.call_with_validation_start(
            stage_a_system, stage_a_user, validate_risk_advisor_stage_a,
            max_tokens=3500, temperature=0.3,
        )
        assert stage_a_errors == [], f"Stage A failed validation: {stage_a_errors}"
        assert context is not None, "a successful Stage A call must return a real context"

        stage_b_prompt = fill(
            load_template("risk_advisor", stage="b"),
            {
                "account_type": acct,
                "timeline": ctx["timeline"],
                "account_instruction": f"Account: {acct.upper()}, Timeline: {ctx['timeline']}.",
                "portfolio_context": "(none provided)",
                "precomputed_portfolio_fit_metrics": "(none -- no portfolio_context provided)",
            },
        )
        stage_b_result, stage_b_errors = await runner.call_with_validation_continue(
            stage_b_prompt, context, validate_risk_advisor_stage_b,
            max_tokens=1500, temperature=0.3,
        )

    assert stage_b_errors == [], f"Stage B failed validation: {stage_b_errors}"

    # The mechanical, wording-independent proof that continuation actually
    # reused Stage A's context rather than silently starting fresh -- can't
    # be faked by a plain separate call, unlike free-text content matching,
    # which the model's real non-determinism would make brittle to assert on
    # directly. Confirmed live during planning this reliably lands non-zero
    # on the immediate next call.
    stage_b_log_entry = runner.call_log[-1]
    assert stage_b_log_entry["prompt_eval_cached_count"] > 0, (
        "Stage B's call reused none of Stage A's context -- continuation may "
        "have silently degraded to a fresh call"
    )

    real_stage_b_fields = {
        "correlation_to_existing_portfolio",
        "concentration_risk",
        "liquidity_risk",
        "position_size_recommendation",
        "stop_loss_suggestion",
        "sizing_rationale",
        "caveats",
    }
    assert real_stage_b_fields <= set(stage_b_result.keys())
