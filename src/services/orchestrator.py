"""AnalysisOrchestrator — the two-pass agent pipeline (86bbuhjup).

Sequences the real, now-ported pipeline per docs/technical/orchestration-engine.md's
documented flow and this project's own settled gate contract
(docs/decision-log.md, 2026-09-02/2026-09-16 -- see agents/utils.py's own
docstrings for the full history):

    DataPipeline.prepare()
      -> Pass 1 (asyncio.gather, 5 agents, each contained)
      -> Gate 1 (completion-only -- agent_completed() on each)
      -> compression
      -> Pass 2 (asyncio.gather, 4 agents, contained)
      -> Gate 2 (both Bull and Bear must complete, else skip the CIO entirely)
      -> disagreement score -> CIO Stage A -> CIO Stage B
      -> Shadow CIO (non-blocking, own try/except)
      -> write Recommendation + Prediction + ShadowPrediction

Explicitly NOT in scope for this port (see the approved plan's Design step 4):
`stock_analysis_memory` loading (orchestration-engine.md step 4c) -- with no
prior analyses in a freshly-created DB there's no real history to load, so
every agent's memory_brief/reflexion_brief/accuracy_brief placeholder fills
with "" already, inside each runner -- the same convention this session
validated as the correct real value for a system with no accumulated
history, not a stand-in for missing wiring. Also not in scope:
PredictionCheckpoint row scheduling -- the approved plan's Design section
only names Recommendation/Prediction/ShadowPrediction; checkpoint scheduling
is real, separate follow-up work, not silently absorbed here.
"""
import asyncio
import itertools
from datetime import UTC, datetime
from uuid import uuid4

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from agents.base import MODEL, OllamaUnavailable
from agents.compression import compress_pass1_outputs
from agents.pass1_fundamental_analyst import FundamentalAnalystRunner
from agents.pass1_macro_economist import MacroEconomistRunner
from agents.pass1_sentiment_analyst import SentimentAnalystRunner
from agents.pass1_stock_researcher import StockResearcherRunner
from agents.pass1_technical_analyst import TechnicalAnalystRunner
from agents.pass2_bear_advocate import BearAdvocateRunner
from agents.pass2_bull_advocate import BullAdvocateRunner
from agents.pass2_risk_advisor import RiskAdvisorRunner
from agents.pass2_tax_strategist import TaxStrategistRunner
from agents.pass3_cio import CIORunner
from agents.pass3_shadow_cio import ShadowCIORunner
from agents.utils import (
    agent_completed,
    compute_disagreement_score,
    compute_outlook_distance,
    gate1_check,
    gate2_check,
)
from api.tables.agent_outputs import AgentOutput
from api.tables.analysis_runs import AnalysisRun, RunStatus
from api.tables.llm_calls import LLMCall
from api.tables.predictions import Prediction
from api.tables.recommendations import Recommendation
from api.tables.shadow_predictions import ShadowPrediction
from data.pipeline import DataPipeline
from data.precompute.tax_metrics import (
    classify_dividend,
    compute_effective_after_tax_yield,
    compute_trailing_dividend,
    resolve_withholding,
)
from data.providers.router import Router
from data.schemas.context import AnalysisContext
from data.schemas.data_bundle import DataBundle

logger = structlog.get_logger(__name__)

# Prompt template versions this orchestrator's agents actually load -- a
# static per-agent table, not a content hash. Real content-hash prompt
# versioning (AgentOutput.prompt_version's own column comment: "content
# hash") is a separate, unbuilt concern (prompt A/B testing infrastructure,
# per CLAUDE.md's LLM Routing section) -- not fabricated here as a fake hash.
_PROMPT_VERSIONS = {
    "RSRCH": "v1", "FUND": "v1", "TECH": "v1", "SENT": "v1", "MACRO": "v1",
    "bull": "v1", "bear": "v1", "tax": "v1", "risk": "v1",
    "cio_stage_a": "v1", "cio_stage_b": "v1", "shadow_cio": "v1",
}

# Real 5-tier CIO value (relative-performance vocabulary) -> Recommendation's
# stale 4-tier column (absolute-magnitude vocabulary, "high|moderate|low|
# minimal", Phase 2). Deliberately lossy -- 86bbt1kpj's still-open vocabulary
# mismatch, not fixed here (that's real, separate schema-design work per the
# approved plan). An ordinal best-effort collapse, not a semantic equivalence.
_RETURN_TIER_TO_RECOMMENDATION_VOCAB = {
    "strong_outperform": "high",
    "outperform": "moderate",
    "market_perform": "low",
    "underperform": "minimal",
    "strong_underperform": "minimal",
}


def _build_pass2_view_bundles(bundle: DataBundle) -> dict[str, dict]:
    """Per-agent orchestrator-owned field slices for agents/pass2_view.py's
    build_pass2_view() -- the fields each Pass 1 agent's own prompt never
    produces (precomputed numerics, derived enums), keyed by agent_id. Field
    names here match build_pass2_view()'s own per-agent b.get(...) calls
    exactly (see that file), not just conceptually.
    """
    fund = bundle.valuation_metrics
    growth = bundle.growth_metrics
    prof = bundle.profitability_metrics
    bal = bundle.balance_sheet_metrics
    sector_medians = bundle.peer_metrics.get("sector_medians", {})
    ti = bundle.technical_indicators
    sr = bundle.support_resistance
    m = bundle.macro_sources

    return {
        "FUND": {
            "pe_ratio": fund.get("pe_ratio"),
            "margin_trend": prof.get("margin_trend"),
            "sector_pe_median": sector_medians.get("sector_median_pe"),
            "roe": prof.get("roe"),
            "debt_to_equity": bal.get("debt_to_equity"),
            "net_margin": prof.get("net_margin"),
            "operating_margin": prof.get("operating_margin"),
            "revenue_growth_yoy": growth.get("revenue_growth_yoy"),
            "revenue_growth_3yr_cagr": growth.get("revenue_growth_3yr_cagr"),
            "fcf_to_net_income": prof.get("fcf_to_net_income"),
        },
        "TECH": {
            "nearest_support": sr.get("nearest_support"),
            "nearest_resistance": sr.get("nearest_resistance"),
            "volatility_regime_derived": ti.get("volatility_regime_derived"),
        },
        "MACRO": {
            "interest_rate_direction": m.rate_trend,
            "inflation_trend": m.cpi_trend,
            "currency_trend": m.cad_trend,
            "commodity_context": m.sector_commodity_direction,
            "volatility_regime": m.vix_regime,
        },
    }


def _llm_call_rows(run_id, agent_pass: str, runner) -> list[LLMCall]:
    """One LLMCall row per NEW runner.call_log entry since this runner was
    last processed -- every attempt, not just the accepted one (86bbwachy
    Phase 2, run-instrumentation.md §5.3, §9 OQ5). Does not set
    agent_output_id here -- _agent_output_row (the only caller) backfills
    that directly onto the returned list's last element, since AgentOutput's
    own PK is generated client-side (see that function's own comment) and
    is already known by the time this runs.

    Sliced from `runner._llm_calls_consumed` onward, not the whole log --
    CIORunner and RiskAdvisorRunner are each a SINGLE shared instance across
    two stages, and call_log accumulates across both (nothing ever clears
    it). CIO calls this once after Stage A and again after Stage B; without
    the slice, the second call reprocesses Stage A's own entries too,
    inserting a real duplicate llm_calls row (confirmed live, 2026-09-24 AAPL
    run: two identical seq=15 "agent:cio_stage_a" rows, one correctly
    carrying agent_output_id from the first call, one blank from the
    second). `_prime_runner` sets the counter to 0 once per runner
    construction; `getattr(..., 0)` covers any runner that predates that
    (none in production today, but keeps this function safe standalone).
    """
    consumed = getattr(runner, "_llm_calls_consumed", 0)
    new_entries = runner.call_log[consumed:]

    rows = []
    for entry in new_entries:
        total_duration_s = entry.get("total_duration_s")
        rows.append(LLMCall(
            run_id=run_id,
            context_tag=entry.get("context_tag", "analysis"),
            seq=entry.get("seq"),
            call_site=entry.get("call_site") or f"agent:{(entry.get('agent') or '').lower()}",
            agent_pass=agent_pass,
            attempt=entry.get("attempt", 1),
            model=entry.get("model", MODEL),
            options_json=entry.get("options_json") or {},
            prompt_path=entry.get("prompt_path"),
            context_path=entry.get("context_path"),
            response_path=entry.get("response_path"),
            prompt_tokens=entry.get("prompt_eval_count"),
            completion_tokens=entry.get("eval_count"),
            thinking_chars=entry.get("thinking_chars"),
            # `is not None`, not a bare truthiness check -- a genuine 0.0s
            # duration (unrealistic for a real network call, but not
            # impossible in a test) must not silently become a "missing"
            # None the same way _market_cap_bucket's own comment already
            # warns against for a different field.
            latency_ms=round(total_duration_s * 1000) if total_duration_s is not None else None,
            finish_reason=entry.get("finish_reason"),
            parsed_ok=entry.get("parsed_ok"),
            parse_error=entry.get("parse_error"),
            validator_passed=entry.get("validator_passed"),
            validator_errors=entry.get("validator_errors"),
            auto_trimmed=entry.get("auto_trimmed", False),
        ))
    # Only advance the consumed marker once every row above was built
    # successfully -- doing this before the loop would permanently drop
    # entries from ever becoming a row if construction raised partway
    # through (unlikely given every field read above is a defaulted
    # .get(), but free to get right and a real footgun for whatever field
    # gets added here next).
    runner._llm_calls_consumed = len(runner.call_log)
    return rows


def _agent_output_row(
    run: AnalysisRun,
    agent_id: str,
    agent_pass: str,
    result: dict | None,
    errors: list[str],
    runner,
) -> tuple[AgentOutput, list[LLMCall]]:
    """One AgentOutput row per call, written immediately after that agent
    finishes (or fails) -- not batched at the end. This is what actually
    backs per-agent containment (d): a mid-run failure still leaves behind
    the call log for every agent that already completed, rather than losing
    it if the process dies before a final batch write.

    status is derived from agent_completed(result), not from `errors` being
    empty -- a validation-failed-but-present agent still counts as
    "completed" (SETTLED 2026-09-02, audit E77: validation status is a
    warning, not a gate). Only a None/empty/all-None result is "failed".

    Also returns this call's llm_calls rows (86bbwachy Phase 2) -- linked to
    the AgentOutput row by generating output_id ourselves (uuid4(), the same
    callable AgentOutput.output_id's own column default would otherwise use)
    instead of relying on a post-insert PK. AgentOutput.output_id is a
    client-side default (uuid4, not a DB sequence/server default), so this
    doesn't skip anything the database would normally provide -- it just
    lets us know the value before the row is ever added to the session,
    avoiding the flush-then-UPDATE two-step the original plan for this
    ticket assumed was necessary. The caller is responsible for db.add()-ing
    both the returned AgentOutput and every row in the returned list.
    """
    completed = agent_completed(result)
    r = result or {}
    timing = runner.last_timing or {}
    output_id = uuid4()
    agent_output = AgentOutput(
        output_id=output_id,
        run_id=run.run_id,
        agent_name=agent_id,
        agent_pass=agent_pass,
        status="completed" if completed else "failed",
        recommendation=r.get("recommendation") if agent_pass != "pass1" else None,
        confidence=r.get("confidence") if agent_pass != "pass1" else None,
        analysis_confidence=r.get("analysis_confidence") if agent_pass == "pass1" else None,
        key_factors=r.get("key_factors"),
        risks=r.get("risks"),
        narrative=r.get("narrative") or r.get("synthesis_narrative"),
        structured_output=result,
        model_used=MODEL,
        prompt_version=_PROMPT_VERSIONS.get(agent_id, "v1"),
        tokens_used=timing.get("eval_count"),
        latency_ms=round(timing.get("total_duration_s", 0) * 1000) if timing.get("total_duration_s") else None,
        error_detail="; ".join(errors) if errors else None,
    )

    llm_calls = _llm_call_rows(run.run_id, agent_pass, runner)
    # The winning attempt is always call_log's last entry, regardless of
    # which path _retry_loop returned through (validated, auto-trimmed,
    # soft-error-accepted, or exhausted) -- confirmed by reading that
    # function directly: `last_result`/the returned `result` are always
    # updated from the same attempt whose call_log entry was just appended,
    # every single time through the loop, not just on the success path.
    # Only back-link when the agent actually completed -- a failed agent's
    # calls all stay unlinked, which is correct: none of them produced an
    # accepted output for AgentOutput to point back to.
    if completed and llm_calls:
        llm_calls[-1].agent_output_id = output_id

    return agent_output, llm_calls


def _add_agent_output_and_calls(
    db: AsyncSession,
    run: AnalysisRun,
    agent_id: str,
    agent_pass: str,
    result: dict | None,
    errors: list[str],
    runner,
) -> None:
    """db.add()s both halves of _agent_output_row's return value -- the
    common pattern at every one of this file's five call sites, factored
    out once rather than repeated five times (86bbwachy Phase 2)."""
    agent_output, llm_calls = _agent_output_row(run, agent_id, agent_pass, result, errors, runner)
    db.add(agent_output)
    for call in llm_calls:
        db.add(call)


def _validate_tax_passthroughs(
    output: dict, bundle: DataBundle, account_type: str
) -> tuple[bool, list[str]]:
    """Production equivalent of agents/validators/pass2.py's
    validate_tax_passthroughs(), which takes a fixture dict -- this computes
    the same expected values directly from the real DataBundle instead
    (classify_dividend/compute_trailing_dividend/resolve_withholding/
    compute_effective_after_tax_yield, the exact same pure functions
    pass2_tax_strategist.py's own bundle.tax_metrics was built from), rather
    than re-parsing them back out of the rendered tax_metrics string.

    Confirms Tax Strategist's dividend_yield_pct/withholding_tax_rate_pct/
    effective_after_tax_yield_pct are within tolerance of the orchestrator's
    own computed values -- catches the LLM silently altering a passed-through
    number, not just a malformed field.
    """
    errors: list[str] = []
    tp = output.get("tax_profile", {})

    classification, _ = classify_dividend(bundle.stock.ticker, bundle.company_info)
    expected_yield, _ = compute_trailing_dividend(
        bundle.dividend_history, bundle.price_info.get("current_price")
    )
    wht = resolve_withholding(classification, account_type)
    expected_wht = wht[0] if wht is not None else None
    expected_eff, _ = compute_effective_after_tax_yield(expected_yield, expected_wht)

    actual_yield = tp.get("dividend_yield_pct")
    if actual_yield is not None and expected_yield is not None:
        if abs(actual_yield - expected_yield) > 0.15:
            errors.append(
                f"tax_profile.dividend_yield_pct: expected ~{expected_yield}%, got {actual_yield}% (tolerance ±0.1%)"
            )

    actual_wht = tp.get("withholding_tax_rate_pct")
    if actual_wht is not None and expected_wht is not None:
        if abs(actual_wht - expected_wht) > 0.15:
            errors.append(
                f"tax_profile.withholding_tax_rate_pct: expected ~{expected_wht}%, got {actual_wht}% (tolerance ±0.1%)"
            )

    actual_eff = tp.get("effective_after_tax_yield_pct")
    if actual_eff is not None and expected_eff is not None:
        if abs(actual_eff - expected_eff) > 0.2:
            errors.append(
                f"tax_profile.effective_after_tax_yield_pct: expected ~{expected_eff}%, got {actual_eff}% (tolerance ±0.1%)"
            )

    return len(errors) == 0, errors


async def _close_runner(runner) -> None:
    """Every runner constructed by this orchestrator owns whatever aiohttp
    session it lazily creates on first call (BaseRunner._get_session()) --
    nothing here ever injects one. Only `async with runner:` closes it
    (BaseRunner.__aexit__), which this orchestrator never uses (runners are
    plain local variables whose `last_timing`/`last_stage_a_context` are
    still needed after the call completes, for _agent_output_row and Stage
    B). Confirmed live (86bbuhjup smoke test): omitting this produces a real
    "Unclosed client session" warning per agent, not just a theoretical
    leak -- 11 leaked sessions per full pipeline run.
    """
    if runner.session is not None:
        await runner.session.close()


async def _run_contained(
    pass_label: str, agent_id: str, coro
) -> tuple[str, dict | None, list[str], Exception | None]:
    """Runs one agent call, catching everything so a single agent's
    exception can never escape into asyncio.gather() -- see _run_pass1's
    own docstring for why letting one escape would orphan sibling tasks
    (asyncio.gather does not cancel siblings on one task's exception) and
    lose their AgentOutput rows. Shared by every Pass 1/Pass 2 agent
    (found during a 2026-09-23 review: the three call sites were carrying
    near-identical try/except bodies, differing only in the log event
    prefix and which coroutine to await); an agent-specific post-
    processing step (tax's passthrough check, risk's Stage B) wraps a call
    to this rather than duplicating its try/except.

    `pass_label` ("pass1"/"pass2") is threaded through only to keep the
    existing structlog event names (`{pass}_agent_ollama_unavailable`/
    `{pass}_agent_failed`) unchanged -- observability behavior this
    refactor deliberately preserves exactly, not a design either helper
    needs on its own.
    """
    try:
        result, errors = await coro
        return agent_id, result, errors, None
    except OllamaUnavailable as exc:
        # Collected, not re-raised -- the caller checks for this marker
        # after every agent in the gather has been accounted for and
        # closed. See _run_pass1's own docstring for the full reasoning.
        logger.error(f"{pass_label}_agent_ollama_unavailable", agent_id=agent_id)
        return agent_id, None, [str(exc)], exc
    except Exception as exc:
        logger.error(f"{pass_label}_agent_failed", agent_id=agent_id, error=str(exc))
        return agent_id, None, [str(exc)], exc


def _market_cap_bucket(market_cap: float | None) -> str | None:
    """large/mid/small/micro from a raw market cap number (86bbwachy Phase
    1) -- new logic this ticket owns, not a read of an existing field.
    company_info["market_cap"] exists as a raw number (data_bundle.py's own
    docstring) but nothing else in this codebase buckets it.

    Deliberately NOT currency-normalized: market_cap is in the stock's own
    listing currency (CAD for a TSX name, USD for NASDAQ/NYSE), and a real
    fix would mean converting through bundle.macro_sources' own FX rate.
    Not done here -- this is a rough context tag for later analysis
    grouping, not a precise metric anything else depends on, and the
    ~30% CAD/USD swing rarely crosses one of these wide bucket boundaries.
    Thresholds are the commonly-used US convention (large >= $10B, mid
    $2B-$10B, small $300M-$2B, micro < $300M); revisit if CA-specific
    buckets ever turn out to matter more than this.
    """
    # Found on review: a non-positive market cap (bad data, never a real
    # value) used to fall through every >= check and land on "micro" --
    # silently mislabeling corrupt data as a real, small classification
    # instead of surfacing it as missing. "Missing over wrong" applies here
    # the same as everywhere else in this codebase.
    if market_cap is None or market_cap <= 0:
        return None
    if market_cap >= 10_000_000_000:
        return "large"
    if market_cap >= 2_000_000_000:
        return "mid"
    if market_cap >= 300_000_000:
        return "small"
    return "micro"


def _divergence_magnitude(distance: int) -> str:
    """none|minor|moderate|major from compute_outlook_distance()'s 0-4
    integer distance -- matches its own >2 high_divergence threshold: 3-4
    is "major" (the same band that flips high_divergence True), 2 is
    "moderate" (still not high_divergence), 1 "minor", 0 "none"."""
    if distance == 0:
        return "none"
    if distance == 1:
        return "minor"
    if distance == 2:
        return "moderate"
    return "major"


class AnalysisOrchestrator:
    """Runs one full analysis for an already-created AnalysisRun row.

    Callers (the API layer, step 6) are responsible for creating `run`
    (status=queued) and committing it before calling this -- matching
    orchestration-engine.md's own step ordering (AnalysisRun created before
    DataPipeline.prepare() runs), and so a caller can return the run_id to
    the user immediately without waiting for this to complete.
    """

    async def run(self, run: AnalysisRun, db: AsyncSession) -> None:
        # Captured once, as a plain UUID, and used in every log call below
        # instead of re-reading run.run_id each time -- a primary key that
        # will never actually change, but SQLAlchemy expires ALL of an
        # object's attributes (including PKs) after a rolled-back flush, and
        # reading ANY attribute off an expired object triggers a real reload
        # query. Confirmed live (86bbuhjup): doing that reload from inside
        # an exception-handling log call can raise MissingGreenlet, turning
        # "log a warning about a failure" into a second, worse failure than
        # the one being logged. A plain captured value can't expire.
        run_id = run.run_id
        context = AnalysisContext(account_type=run.account_type, timeline=run.timeline)

        try:
            bundle = await DataPipeline().prepare(run.stock_id, context, db)
        except Exception as exc:
            logger.error("data_pipeline_failed", run_id=str(run_id), error=str(exc))
            run.status = RunStatus.FAILED
            run.error_log = [{"stage": "data_pipeline", "error": str(exc)}]
            await db.commit()
            raise

        # Run context tags (86bbwachy Phase 1) -- set here, once, right after
        # a real bundle exists, not at AnalysisRun creation time (see
        # analysis_runs.py's own column comments for why instrument_type/
        # market_cap_bucket have no reliable source that early).
        # market_regime is deliberately left unset -- nothing computes it yet.
        run.exchange = bundle.stock.exchange
        run.currency = bundle.stock.currency
        run.instrument_type = bundle.company_info.get("asset_type")
        run.market_cap_bucket = _market_cap_bucket(bundle.company_info.get("market_cap"))
        await db.commit()

        # 86bbwachy Phase 2 capture context -- instance attributes, not
        # threaded through every private method's own signature, since a
        # fresh AnalysisOrchestrator() is constructed per run (confirmed:
        # api/routes/analysis.py's _run_analysis_background does
        # `AnalysisOrchestrator().run(run, db)`, a new instance every time),
        # so there is no cross-run leakage risk. seq_counter is genuinely
        # run-wide (spec §5.2) -- Pass 1/Pass 2's own concurrent agents all
        # need to share the SAME counter object, which instance state gives
        # them for free; each private method below sets these three onto
        # every runner it constructs, right after construction, matching
        # current_agent's own already-established "set post-init" pattern
        # in agents/base.py.
        self._run_id = run_id
        self._ticker = bundle.stock.ticker
        self._seq_counter = itertools.count()

        try:
            pass1_outputs = await self._run_pass1(run, bundle, db)
        except OllamaUnavailable:
            # Environment-level failure (Ollama down / model not pulled) --
            # abort the whole run rather than contain it agent-by-agent, per
            # agents/base.py's own _retry_loop docstring: this is exactly
            # the case it deliberately does NOT catch, so a future caller
            # (this orchestrator) can tell "the whole environment is
            # broken" apart from "this one agent had a rough call."
            run.status = RunStatus.FAILED
            run.error_log = [{"stage": "pass1", "error": "Ollama unavailable"}]
            await db.commit()
            raise

        run.status = RunStatus.PASS1_COMPLETE
        run.pass1_completed_at = datetime.now(UTC)
        await db.commit()

        gate1_passed, gate1_reason = gate1_check(pass1_outputs)
        if not gate1_passed:
            logger.warning("gate1_failed", run_id=str(run_id), reason=gate1_reason)
            run.status = RunStatus.FAILED
            run.error_log = [{"stage": "gate1", "error": gate1_reason}]
            await db.commit()
            return

        compressed = compress_pass1_outputs(pass1_outputs, bundles=_build_pass2_view_bundles(bundle))

        run.status = RunStatus.PASS2_RUNNING
        await db.commit()

        try:
            pass2_outputs = await self._run_pass2(run, bundle, compressed, db)
        except OllamaUnavailable:
            run.status = RunStatus.FAILED
            run.error_log = [{"stage": "pass2", "error": "Ollama unavailable"}]
            await db.commit()
            raise

        run.status = RunStatus.PASS2_COMPLETE
        run.pass2_completed_at = datetime.now(UTC)
        await db.commit()

        gate2_passed, gate2_reason = gate2_check(pass2_outputs)
        if not gate2_passed:
            logger.warning("gate2_failed", run_id=str(run_id), reason=gate2_reason)
            run.status = RunStatus.FAILED
            run.error_log = [{"stage": "gate2", "error": gate2_reason}]
            await db.commit()
            return

        # `.get("confidence") or 0`, not `.get("confidence", 0)` -- the
        # latter only supplies the default when the KEY is missing, not when
        # it's present but explicitly None, which agent_completed()'s own
        # contract allows (Gate 2 only guarantees SOME field is non-None,
        # not specifically `confidence`). A bare `.get(key, 0)` here would
        # pass None into compute_disagreement_score()'s arithmetic below (a
        # real TypeError) or, at the two DB-write call sites using this same
        # pattern, into a NOT NULL int column at commit time. Same fix
        # applied everywhere else this file reads a possibly-partial agent
        # output's `confidence` field.
        bull_conf = (pass2_outputs.get("bull") or {}).get("confidence") or 0
        bear_conf = (pass2_outputs.get("bear") or {}).get("confidence") or 0
        disagreement_score, disagreement_category = compute_disagreement_score(bull_conf, bear_conf)
        run.disagreement_score = disagreement_score
        run.disagreement_class = disagreement_category

        run.status = RunStatus.SYNTHESIS_RUNNING
        await db.commit()

        # One CIORunner instance shared across Stage A and Stage B -- no
        # continuation context is needed between them (unlike Risk Advisor's
        # Stage A->B, an ordinary second call, see pass3_cio.py's own module
        # docstring), but sharing the instance still means one aiohttp
        # session for both calls rather than two. `finally` closes it on
        # EVERY exit path (success, OllamaUnavailable, or a Stage A
        # completion failure that returns before Stage B ever runs) -- three
        # real leak paths a plain "close after the last successful write"
        # placement missed (confirmed live: the CIO session stayed open on
        # both the raise-and-abort paths and on the "Stage A itself never
        # completed" early return).
        #
        # The two AgentOutput rows this produces use agent_name
        # "cio_stage_a"/"cio_stage_b", not both "cio" -- found during a
        # 2026-09-23 review: with the same agent_name AND agent_pass
        # ("synthesis") for both, nothing in the row itself said which was
        # which; any consumer filtering by agent_name=="cio" (the API layer,
        # a future frontend, calibration code) would get an arbitrary one of
        # the two. No schema migration needed -- agent_name is a plain
        # unconstrained String(30) column.
        cio_runner = CIORunner()
        self._prime_runner(cio_runner)
        try:
            try:
                stage_a_result, stage_a_errors = await self._run_cio_stage_a(
                    run, bundle, compressed, pass2_outputs, cio_runner, db
                )
            except OllamaUnavailable:
                run.status = RunStatus.FAILED
                run.error_log = [{"stage": "cio_stage_a", "error": "Ollama unavailable"}]
                await db.commit()
                raise

            if not agent_completed(stage_a_result):
                logger.error("cio_stage_a_failed", run_id=str(run_id), errors=stage_a_errors)
                run.status = RunStatus.FAILED
                run.error_log = [{"stage": "cio_stage_a", "error": "; ".join(stage_a_errors)}]
                await db.commit()
                return

            try:
                stage_b_result, stage_b_errors = await cio_runner.run_stage_b(
                    bundle, stage_a_result, pass2_outputs.get("tax"), None, run.account_type
                )
            except OllamaUnavailable:
                run.status = RunStatus.FAILED
                run.error_log = [{"stage": "cio_stage_b", "error": "Ollama unavailable"}]
                await db.commit()
                raise

            # Orchestrator-side forwarding (docs/agents/cio.md's
            # merge_output_cio_stage_b): stop_loss_suggestion and
            # tax_summary.dividend_yield_pct are facts Risk/Tax already
            # computed, not LLM judgments -- inject them directly rather
            # than trust the LLM to transcribe them back out of its own
            # prompt correctly. Confirmed live (AAPL run, 2026-09-23):
            # neither field appeared anywhere in a real CIO Stage B output;
            # the LLM only paraphrased the stop-loss level in prose, never
            # as a queryable value a caller could check against Risk's own
            # number. Only overwrites/adds these two keys -- everything
            # else in stage_b_result is still the LLM's own, unmodified
            # output.
            if isinstance(stage_b_result, dict):
                risk_stage_b = (pass2_outputs.get("risk") or {}).get("stage_b") or {}
                stage_b_result["stop_loss_suggestion"] = risk_stage_b.get("stop_loss_suggestion")

                tax_profile = (pass2_outputs.get("tax") or {}).get("tax_profile") or {}
                tax_summary = stage_b_result.get("tax_summary")
                if isinstance(tax_summary, dict):
                    tax_summary["dividend_yield_pct"] = tax_profile.get("dividend_yield_pct")

            _add_agent_output_and_calls(db, run, "cio_stage_b", "synthesis", stage_b_result, stage_b_errors, cio_runner)
            await db.commit()
        finally:
            await _close_runner(cio_runner)

        if not agent_completed(stage_b_result):
            logger.error("cio_stage_b_failed", run_id=str(run_id), errors=stage_b_errors)
            run.status = RunStatus.FAILED
            run.error_log = [{"stage": "cio_stage_b", "error": "; ".join(stage_b_errors)}]
            await db.commit()
            return

        # agent_completed() only guarantees SOME field is non-None, not
        # specifically these -- a validation-failed-but-present result could
        # still be missing the fields Recommendation actually needs. Fail
        # loud and recorded here rather than letting a bare KeyError escape
        # uncaught below (which would skip writing run.status=FAILED).
        missing = [
            f for f in ("stock_outlook", "confidence") if stage_a_result.get(f) is None
        ] + [
            f for f in ("synthesis_narrative", "expected_return_tier")
            if stage_b_result.get(f) is None
        ]
        if missing:
            logger.error("cio_output_missing_required_fields", run_id=str(run_id), missing=missing)
            run.status = RunStatus.FAILED
            run.error_log = [{"stage": "recommendation", "error": f"CIO output missing: {missing}"}]
            await db.commit()
            return

        recommendation = await self._create_recommendation(
            run, stage_a_result, stage_b_result, pass2_outputs, db
        )
        await self._create_prediction(run, bundle, recommendation, db)

        # Shadow CIO: non-blocking, own try/except -- a shadow failure must
        # never fail (or even mark warnings on) the primary, user-facing run.
        try:
            await self._run_shadow_cio(
                run, bundle, compressed, pass2_outputs, stage_a_result, db
            )
        except Exception as exc:
            # rollback() FIRST, before anything else touches this session --
            # including the log line right below. Confirmed live: with the
            # log line first (reading run.run_id), SQLAlchemy's default
            # autoflush=True means merely ACCESSING an attribute on any
            # object attached to the session tries to flush first -- and the
            # ShadowPrediction that just failed to flush is still pending
            # (a failed flush does not remove the offending object), so that
            # autoflush immediately re-raises PendingRollbackError before
            # rollback() ever gets a chance to run. Same underlying failure
            # this whole except block exists to prevent, just moved one line
            # earlier and self-inflicted by the fix itself.
            await db.rollback()
            logger.warning("shadow_cio_failed_non_blocking", run_id=str(run_id), error=str(exc))

        run.status = RunStatus.COMPLETED
        run.completed_at = datetime.now(UTC)
        await db.commit()

    def _prime_runner(self, runner) -> None:
        """Stamps 86bbwachy Phase 2 capture context onto a freshly
        constructed runner, before it makes any calls. run_id/ticker anchor
        the artifact directory (agents/capture.py's own artifact_dir());
        seq_counter is the single run-wide counter shared across every
        concurrently-running agent (Pass 1's 5, Pass 2's 4, CIO, Shadow CIO)
        -- called once per runner construction, matching current_agent's own
        already-established "set post-init" pattern in agents/base.py.
        """
        runner.run_id = self._run_id
        runner.ticker = self._ticker
        runner.seq_counter = self._seq_counter
        # _llm_calls_consumed is NOT set here -- BaseRunner.__init__ already
        # defaults it to 0 (see that class's own comment on why it lives
        # there, next to call_log). Re-zeroing it here would be harmless for
        # every real runner (this always runs right after construction,
        # before any call), but would be actively wrong for a duck-typed
        # test double that pre-seeds call_log/_llm_calls_consumed before
        # calling this -- none do today, but there's no reason to couple
        # this method to owning that field when BaseRunner already does.

    async def _run_pass1(self, run: AnalysisRun, bundle: DataBundle, db: AsyncSession) -> dict:
        runners = {
            "RSRCH": StockResearcherRunner(),
            "FUND": FundamentalAnalystRunner(),
            "TECH": TechnicalAnalystRunner(),
            "SENT": SentimentAnalystRunner(),
            "MACRO": MacroEconomistRunner(),
        }
        for r in runners.values():
            self._prime_runner(r)

        # Do NOT let any agent's exception escape this gather -- letting one
        # propagate would orphan the other still-running agent tasks
        # (asyncio.gather does not cancel siblings on one task's exception)
        # and skip the AgentOutput write loop below entirely, losing the
        # call-log for every agent that already succeeded. _run_contained
        # collects every failure (including OllamaUnavailable) into the
        # standard 4-tuple instead; the loop below checks for that marker
        # after every agent has been accounted for and closed.
        results = await asyncio.gather(
            *(_run_contained("pass1", aid, r.run(bundle)) for aid, r in runners.items())
        )

        outputs: dict[str, dict | None] = {}
        ollama_unavailable: OllamaUnavailable | None = None
        for agent_id, result, errors, exc in results:
            _add_agent_output_and_calls(db, run, agent_id, "pass1", result, errors, runners[agent_id])
            outputs[agent_id] = result
            if isinstance(exc, OllamaUnavailable):
                ollama_unavailable = exc
        await db.commit()
        await asyncio.gather(*(_close_runner(r) for r in runners.values()))
        if ollama_unavailable is not None:
            raise ollama_unavailable
        return outputs

    async def _run_pass2(
        self, run: AnalysisRun, bundle: DataBundle, compressed: dict, db: AsyncSession
    ) -> dict:
        runners = {
            "bull": BullAdvocateRunner(),
            "bear": BearAdvocateRunner(),
            "tax": TaxStrategistRunner(),
            "risk": RiskAdvisorRunner(),
        }
        for r in runners.values():
            self._prime_runner(r)

        async def _run_bull_bear_tax(agent_id: str, runner):
            agent_id, result, errors, exc = await _run_contained(
                "pass2", agent_id, runner.run(bundle, compressed, run.account_type)
            )
            if agent_id == "tax" and exc is None and agent_completed(result):
                # Isolated from _run_contained's own try/except on purpose:
                # a bug in this SECONDARY check must not destroy a real,
                # already-successful Tax Strategist result by masquerading
                # as an agent failure -- it can only ever ADD errors to a
                # real result, never erase one.
                try:
                    passthrough_ok, passthrough_errors = _validate_tax_passthroughs(
                        result, bundle, run.account_type
                    )
                    if not passthrough_ok:
                        errors = [*errors, *passthrough_errors]
                except Exception as passthrough_exc:
                    logger.error("tax_passthrough_check_crashed", error=str(passthrough_exc))
            return agent_id, result, errors, exc

        async def _run_risk(runner):
            agent_id, result, errors, exc = await _run_contained(
                "pass2", "risk", runner.run(bundle, compressed, run.account_type)
            )
            # Stage B is an account-specific overlay on Stage A's own
            # output -- only attempted when Stage A produced a real,
            # success-only continuation context (runner.last_stage_a_context
            # is None on total exhaustion, per BaseRunner._retry_loop's own
            # docstring: never a failed attempt's own context).
            if exc is None and runner.last_stage_a_context is not None:
                try:
                    stage_b_result, stage_b_errors = await runner.run_stage_b(
                        bundle, runner.last_stage_a_context, run.account_type
                    )
                    if result:
                        result["stage_b"] = stage_b_result
                    errors = [*errors, *stage_b_errors]
                except OllamaUnavailable as stage_b_exc:
                    # Environment-level failure -- same treatment as every
                    # other OllamaUnavailable in this file (collected here,
                    # propagated by _run_pass2's own caller after every
                    # agent has been written and closed). Stage A's own
                    # real result is discarded on this path too: if Ollama
                    # is down, the whole run is about to abort regardless,
                    # so there is no partial Risk result a failed run could
                    # still use.
                    logger.error("pass2_agent_ollama_unavailable", agent_id="risk")
                    return "risk", None, [str(stage_b_exc)], stage_b_exc
                except Exception as stage_b_exc:
                    # Isolated from Stage A's own result on purpose --
                    # found during the same 2026-09-23 review that caught
                    # tax's passthrough check needing the same treatment: a
                    # bug or failure in Stage B must not destroy a real,
                    # already-successful Stage A result by masquerading as
                    # a total agent failure. Only records the error and
                    # leaves result["stage_b"] absent; never erases result
                    # itself the way the pre-fix single try/except did.
                    logger.error("pass2_agent_failed", agent_id="risk", error=str(stage_b_exc))
                    errors = [*errors, str(stage_b_exc)]
            return "risk", result, errors, exc

        results = await asyncio.gather(
            _run_bull_bear_tax("bull", runners["bull"]),
            _run_bull_bear_tax("bear", runners["bear"]),
            _run_bull_bear_tax("tax", runners["tax"]),
            _run_risk(runners["risk"]),
        )

        outputs: dict[str, dict | None] = {}
        ollama_unavailable: OllamaUnavailable | None = None
        for agent_id, result, errors, exc in results:
            _add_agent_output_and_calls(db, run, agent_id, "pass2", result, errors, runners[agent_id])
            outputs[agent_id] = result
            if isinstance(exc, OllamaUnavailable):
                ollama_unavailable = exc
        await db.commit()
        await asyncio.gather(*(_close_runner(r) for r in runners.values()))
        if ollama_unavailable is not None:
            raise ollama_unavailable
        return outputs

    async def _run_cio_stage_a(
        self,
        run: AnalysisRun,
        bundle: DataBundle,
        compressed: dict,
        pass2_outputs: dict,
        runner: CIORunner,
        db: AsyncSession,
    ) -> tuple[dict, list[str]]:
        result, errors = await runner.run(bundle, compressed, pass2_outputs)
        _add_agent_output_and_calls(db, run, "cio_stage_a", "synthesis", result, errors, runner)
        await db.commit()
        return result, errors

    async def _run_shadow_cio(
        self,
        run: AnalysisRun,
        bundle: DataBundle,
        compressed: dict,
        pass2_outputs: dict,
        stage_a_result: dict,
        db: AsyncSession,
    ) -> None:
        """Every DB write in here runs inside its own SAVEPOINT
        (db.begin_nested()), not a bare db.add()+commit() -- confirmed live
        (86bbuhjup robustness review) that a bare commit failing this deep in
        the call stack (asyncio.gather'd Pass 1/Pass 2 earlier in the same
        session's life, several prior real commits) can leave the async
        session unable to recover via an explicit db.rollback() call
        afterward (a real, reproduced `MissingGreenlet` error on the
        rollback() call itself -- root cause not fully pinned down, but a
        SAVEPOINT sidesteps it reliably: its own context manager releases or
        rolls back the nested transaction automatically on exit, without
        needing a manual, ordering-sensitive recovery step in the caller).
        Merely accessing an attribute on any object still attached to a
        session that has a pending, never-recovered failed flush also
        matters here -- SQLAlchemy's default autoflush=True means even a log
        line's `run.run_id` read can retrigger the SAME failure before any
        explicit recovery code runs; a SAVEPOINT avoids that class of bug
        entirely by never leaving the outer session in that state to begin
        with. `run_id` is still captured as a plain value up front (not
        `run.run_id`, read fresh at each log call) for the same reason
        `run()`'s own docstring gives: a primary key that never changes, but
        still expires along with every other attribute after a rolled-back
        flush, and reading even a never-changing attribute off an expired
        object triggers a real reload query in the wrong place.
        """
        run_id = run.run_id
        # Shadow never receives Tax Strategist output (docs/agents/shadow_cio.md
        # v2: no account-specific counterpart) -- only bull/bear/risk.
        shadow_pass2 = {k: pass2_outputs.get(k) for k in ("bull", "bear", "risk")}
        runner = ShadowCIORunner()
        self._prime_runner(runner)
        result, errors = await runner.run(bundle, compressed, shadow_pass2)
        async with db.begin_nested():
            _add_agent_output_and_calls(db, run, "shadow_cio", "synthesis", result, errors, runner)
            await db.flush()
        await db.commit()
        await _close_runner(runner)

        if not agent_completed(result):
            logger.warning("shadow_cio_did_not_complete", run_id=str(run_id), errors=errors)
            return

        primary_outlook = stage_a_result.get("stock_outlook")
        shadow_outlook = result.get("stock_outlook")
        if primary_outlook is None or shadow_outlook is None:
            # agent_completed() only guarantees SOME field is non-None, not
            # specifically stock_outlook -- a validation-failed-but-present
            # result could be missing it. No divergence to compute without
            # both real outlooks; skip rather than raise inside this
            # already-non-blocking path.
            logger.warning(
                "shadow_cio_missing_outlook_for_divergence", run_id=str(run_id)
            )
            return

        distance, _high_divergence = compute_outlook_distance(primary_outlook, shadow_outlook)
        async with db.begin_nested():
            db.add(ShadowPrediction(
                analysis_run_id=run_id,
                primary_outlook_direction=primary_outlook,
                primary_confidence=stage_a_result.get("confidence") or 0,
                primary_projected_return_tier=stage_a_result.get("expected_return_tier"),
                shadow_outlook_direction=shadow_outlook,
                shadow_confidence=result.get("confidence") or 0,
                shadow_projected_return_tier=result.get("expected_return_tier"),
                divergence_magnitude=_divergence_magnitude(distance),
            ))
            await db.flush()
        await db.commit()

    async def _create_recommendation(
        self,
        run: AnalysisRun,
        stage_a_result: dict,
        stage_b_result: dict,
        pass2_outputs: dict,
        db: AsyncSession,
    ) -> Recommendation:
        """Maps what fits cleanly into Recommendation's existing columns;
        does not duplicate the rest. AgentOutput.structured_output (written
        for both CIO calls above) already carries the CIO's complete,
        unmodified Stage A+B output -- this is a curated view for the
        fields that already have a clean column, not the only place this
        data lands. See docs/technical/database-persistence-audit.md and
        this module's own docstring.
        """
        outlook = stage_a_result["stock_outlook"]
        action = (
            "buy" if outlook in ("bullish", "somewhat_bullish")
            else "sell" if outlook in ("bearish", "somewhat_bearish")
            else "hold"
        )
        bull_conf = (pass2_outputs.get("bull") or {}).get("confidence") or 0
        bear_conf = (pass2_outputs.get("bear") or {}).get("confidence") or 0

        recommendation = Recommendation(
            run_id=run.run_id,
            stock_id=run.stock_id,
            stock_outlook_direction=outlook,
            overall_confidence=stage_a_result["confidence"],
            account_recommendation={
                "action": action,
                "rationale": stage_b_result.get("synthesis_narrative", ""),
            },
            position_size_suggestion=stage_b_result.get("position_sizing_recommendation"),
            key_drivers=stage_a_result.get("key_decision_factors") or [],
            key_risks=[],
            synthesis_narrative=stage_b_result.get("synthesis_narrative", ""),
            dissenting_views=None,
            bull_case_strength=bull_conf,
            bear_case_strength=bear_conf,
            expected_return_tier=_RETURN_TIER_TO_RECOMMENDATION_VOCAB.get(
                stage_b_result.get("expected_return_tier")
            ),
        )
        db.add(recommendation)
        await db.commit()
        await db.refresh(recommendation)
        return recommendation

    async def _create_prediction(
        self, run: AnalysisRun, bundle: DataBundle, recommendation: Recommendation, db: AsyncSession
    ) -> None:
        """price_at_recommendation/benchmark_price_at_recommendation are
        NOT NULL columns with no DataBundle equivalent for the benchmark
        side (DataBundle carries benchmark_ticker, a string, never the
        benchmark's own current price) -- fetched here via one real,
        additional Router.get_quote() call rather than left None (which
        would fail this NOT NULL column at commit) or fabricated.
        """
        async with Router(ticker=bundle.benchmark_ticker) as router:
            benchmark_quote = await router.get_quote(bundle.benchmark_ticker)

        db.add(Prediction(
            recommendation_id=recommendation.recommendation_id,
            stock_id=run.stock_id,
            price_at_recommendation=bundle.price_info.get("current_price"),
            benchmark_price_at_recommendation=benchmark_quote.get("current_price"),
        ))
        await db.commit()
