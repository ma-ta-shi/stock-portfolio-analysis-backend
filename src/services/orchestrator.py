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
from sqlalchemy import select
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
from agents.prompts import prompt_fingerprints
from agents.utils import (
    agent_completed,
    compute_disagreement_score,
    compute_outlook_distance,
    gate1_check,
    gate2_check,
)
from api.crud.user_profile import get_user_profile
from api.tables.agent_outputs import AgentOutput
from api.tables.analysis_runs import AnalysisRun, RunStatus
from api.tables.llm_calls import LLMCall
from api.tables.predictions import Prediction
from api.tables.recommendations import Recommendation
from api.tables.run_quality_summary import RunQualitySummary
from api.tables.shadow_predictions import ShadowPrediction
from api.tables.stock import Stock
from api.tables.user_profile import UserProfile
from data.industry_benchmark import pe_vs_industry
from data.precompute.insider import summarize_insider_activity
from data.precompute.sentiment_signals import analyst_changes_for_bundle, summarize_short_interest
from data.degradation import DegradationCollector, reset_collector, set_collector
from data.pipeline import DataPipeline
from data.precompute.tax_metrics import AccountStateInput, UserTaxProfileInput
from data.providers.router import Router
from data.schemas.context import AnalysisContext
from data.schemas.data_bundle import DataBundle
from services.code_version import code_fingerprint
from services.error_recorder import (
    build_error_note,
    describe_exception,
    rule_slug,
    safe_text,
    summarize_call_log,
    write_error_notes,
)
from services.run_liveness import end_run_now

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

# 86bbwachy Phase 5 -- which of the 12 real agent_names are structurally
# excluded from _write_run_quality_summary's degenerate-output detection.
# Confirmed live (real MSFT run, 2026-09-24), not assumed: without these
# exclusions, every one of the excluded agent_names showed up in the
# corresponding list on EVERY real run regardless of actual quality,
# drowning the real signal in structural noise. Verified directly against
# each agent's own validator source, not guessed:
# - key_factors: never referenced anywhere in validators/cio.py or
#   validators/shadow_cio.py -- only the synthesis-pass agents lack it.
# - risks: never referenced anywhere outside validators/pass1.py -- the
#   four Pass 2 advocates and all three synthesis-pass agents structurally
#   never produce it, confirmed by direct grep.
# - narrative: cio_stage_a's own validator (validate_cio_stage_a) never
#   checks synthesis_narrative (only Stage B's does), and shadow_cio's
#   validator never checks narrative/synthesis_narrative at all -- both
#   structural, unlike bull/bear/tax/risk_stage_a and cio_stage_b, which
#   all have real, validated narrative fields.
_NO_KEY_FACTORS_EXPECTED = {"cio_stage_a", "cio_stage_b", "shadow_cio"}
_NO_RISKS_EXPECTED = {"bull", "bear", "tax", "risk", "cio_stage_a", "cio_stage_b", "shadow_cio"}
_NO_NARRATIVE_EXPECTED = {"cio_stage_a", "shadow_cio"}


def _plain(value):
    """A numpy scalar as a plain Python number; anything else unchanged."""
    return value.item() if hasattr(value, "item") and not isinstance(value, (list, dict, str)) else value


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
    industry = bundle.peer_metrics.get("industry_benchmark")
    pe_pct, pe_position = pe_vs_industry(fund.get("pe_ratio"), industry)
    insider = summarize_insider_activity(
        bundle.insider_activity.get("transactions", []), bundle.price_info.get("market_cap"),
        bundle.insider_activity.get("value_currency"), bundle.price_info.get("currency"),
    )
    analyst = analyst_changes_for_bundle(bundle)
    short = summarize_short_interest(bundle.short_interest)
    ti = bundle.technical_indicators
    sr = bundle.support_resistance
    m = bundle.macro_sources
    is_ca_macro = bundle.canadian_data_flags is not None  # the same test the Macro runner uses

    views = {
        "FUND": {
            "pe_ratio": fund.get("pe_ratio"),
            "margin_trend": prof.get("margin_trend"),
            # The industry P/E benchmark (data/industry_benchmark.py) and where this P/E sits in it, computed in code so
            # Pass 2 copies the position instead of judging a noisy median.
            "industry_pe_median": industry["median_pe"] if industry else None,
            "industry_pe_range": f"{industry['p25_pe']:.1f} to {industry['p75_pe']:.1f}" if industry else None,
            "industry_pe_count": industry["companies"] if industry else None,
            "pe_vs_industry_median_pct": pe_pct,
            "pe_vs_industry": pe_position,
            "roe": prof.get("roe"),
            "debt_to_equity": bal.get("debt_to_equity"),
            "net_margin": prof.get("net_margin"),
            "operating_margin": prof.get("operating_margin"),
            "revenue_growth_yoy": growth.get("revenue_growth_yoy"),
            "revenue_growth_yoy_basis": growth.get("revenue_growth_yoy_basis"),
            "revenue_growth_annual": growth.get("revenue_growth_annual"),
            "revenue_growth_3yr_cagr": growth.get("revenue_growth_3yr_cagr"),
            "fcf_to_net_income": prof.get("fcf_to_net_income"),
        },
        "SENT": {
            "insider_activity_90d": insider["text"],
            "insider_materiality": insider["materiality"],
            # Decided in code from the bundle's figures, not by the model (which wrote "normal volatility risk"
            # for a three-value field in 27 of 48 first attempts): {trend, interpretation}.
            "short_interest_interpretation": {"trend": short["trend"], "interpretation": short["interpretation"]},
            "analyst_changes_90d": analyst["text"],
            # The Pass 2 view used to read these from the model's own output, which never contains them
            # (empty in 44 of 46 real runs); they are provider data, so they come from the bundle.
            "consensus_rating": bundle.analyst_consensus.get("consensus_rating"),
            "average_price_target": bundle.analyst_consensus.get("target_mean"),
        },
        "TECH": {
            # numpy scalars from the indicator maths; a real run once crashed on their repr, so plain floats here
            "nearest_support": _plain(sr.get("nearest_support")),
            "nearest_resistance": _plain(sr.get("nearest_resistance")),
            "volatility_regime_derived": ti.get("volatility_regime_derived"),
        },
        "MACRO": {
            # A Canadian stock's view carries the Bank of Canada and Canadian CPI trends (the Macro payload shows
            # both countries for it); it used to carry the Fed's and US CPI's, so Bull, Bear and the CIO read
            # "tightening" for TD.TO while the BoC was pausing. None, not the US figure, when the Canadian one is missing.
            "interest_rate_direction": m.boc_rate_trend if is_ca_macro else m.rate_trend,
            "inflation_trend": m.ca_cpi_trend if is_ca_macro else m.cpi_trend,
            "currency_trend": m.cad_trend,
            "commodity_context": m.sector_commodity_direction,
            "volatility_regime": m.vix_regime,
        },
    }
    # A metric that does not exist for this kind of company (a bank's operating margin, a REIT's P/E) is passed as
    # "not_applicable", not None, so Pass 2 does not read it as missing data.
    na_fields = set((bundle.not_applicable or {}).get("fields", []))
    industry_keys = {"industry_pe_median", "industry_pe_range", "industry_pe_count", "pe_vs_industry_median_pct", "pe_vs_industry"}
    if "pe_ratio" in na_fields:
        # a REIT's earnings multiples mislead: its industry P/E is not applicable either, even though the screener has one
        for key in industry_keys:
            views["FUND"][key] = "not_applicable"
    for key in na_fields:
        if key in views["FUND"] and views["FUND"][key] is None:
            views["FUND"][key] = "not_applicable"
    return views


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
    second). `BaseRunner.__init__` defaults the counter to 0 for every real
    runner; `getattr(..., 0)` only exists for a duck-typed test double that
    isn't a real BaseRunner (e.g. test_orchestrator.py's own _StubRunner).

    The actual entry->LLMCall field mapping lives on LLMCall.from_call_log_entry
    itself (86bbwachy Phase 3 consolidation, api/tables/llm_calls.py) --
    this function's own job is the slicing/consumed-index bookkeeping above,
    which has no precompute equivalent (data/pipeline.py's own
    _precompute_llm_call_rows shares the same classmethod but not this).
    """
    consumed = getattr(runner, "_llm_calls_consumed", 0)
    new_entries = runner.call_log[consumed:]
    rows = [LLMCall.from_call_log_entry(entry, run_id=run_id, agent_pass=agent_pass) for entry in new_entries]
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
        # 86bbwachy Phase 4 -- set by the 7 in-scope runners' own run()
        # right after build_user_message() returns (before the LLM call is
        # even attempted), None for every other agent. Deliberately read
        # here regardless of `completed` -- a failed call still had a real
        # (possibly incomplete) input, and knowing that is exactly the
        # point (see this field's own column comment).
        input_field_coverage=getattr(runner, "last_field_coverage", None),
        # 86bbummwp Tier 2 -- D6's mechanical flags, same getattr/None-default
        # pattern as input_field_coverage above, for the same reason (set
        # before the LLM call, read regardless of completion status). None is
        # a real, distinct state from an empty list here: it means this agent
        # doesn't implement that particular check (yet), not "checked, found
        # nothing" -- see agent_outputs.py's own column comments.
        stale_data=getattr(runner, "last_stale_data", None),
        anomalies=getattr(runner, "last_anomalies", None),
        data_coverage=getattr(runner, "last_data_coverage", None),
        # 86bbummwp Tier 3 -- D6 section 3's per-agent mechanical rollup, same
        # getattr/None-default pattern as the other 3 mechanical flags above.
        data_quality_assessment=getattr(runner, "last_data_quality_assessment", None),
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


async def _close_runner(runner) -> None:
    """Every runner constructed by this orchestrator owns whatever aiohttp
    session it lazily creates on first call (BaseRunner._get_session()) --
    nothing here ever injects one. Only `async with runner:` closes it
    (BaseRunner.__aexit__), which this orchestrator never uses (runners are
    plain local variables whose `last_timing` is still needed after the
    call completes, for _agent_output_row). Confirmed live (86bbuhjup smoke test): omitting this produces a real
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
        return agent_id, None, [safe_text(exc, 4000)], exc
    except Exception as exc:
        logger.error(f"{pass_label}_agent_failed", agent_id=agent_id, error=safe_text(exc, 4000))
        return agent_id, None, [safe_text(exc, 4000)], exc


def _build_tax_inputs(
    user_profile: UserProfile | None,
) -> tuple[AccountStateInput | None, UserTaxProfileInput | None]:
    """(account_state, user_tax_profile) from a real UserProfile row, or
    (None, None) when there is no profile at all -- the same degrade-to-absent
    behavior every other optional-data path in this codebase already uses, not
    a new failure mode (86bc8efvb). Dollar->cents conversion happens here for
    the room fields: UserProfile.tfsa_room_remaining/rrsp_room_remaining are
    dollars, AccountStateInput's own fields are cents (existing convention in
    tax_metrics.py, unchanged here). Builds the full account_state dict
    regardless of account_type -- build_precomputed_tax_metrics()'s own
    per-account_type branching already decides what actually renders, so this
    doesn't need to know or care which account is being analyzed."""
    if user_profile is None:
        return None, None

    account_state: AccountStateInput = {}
    if user_profile.tfsa_room_remaining is not None:
        account_state["tfsa_room_remaining_cents"] = round(user_profile.tfsa_room_remaining * 100)
    if user_profile.tfsa_room_as_of is not None:
        account_state["tfsa_room_as_of"] = user_profile.tfsa_room_as_of
    if user_profile.rrsp_room_remaining is not None:
        account_state["rrsp_room_remaining_cents"] = round(user_profile.rrsp_room_remaining * 100)
    if user_profile.rrsp_room_as_of is not None:
        account_state["rrsp_room_as_of"] = user_profile.rrsp_room_as_of

    user_tax_profile: UserTaxProfileInput = {}
    if user_profile.province is not None:
        user_tax_profile["province"] = user_profile.province
    if user_profile.income_annual is not None:
        user_tax_profile["income_annual"] = user_profile.income_annual
    if user_profile.marginal_tax_rate_override_pct is not None:
        user_tax_profile["marginal_tax_rate_override_pct"] = (
            user_profile.marginal_tax_rate_override_pct
        )

    # Empty dict -> None: a UserProfile row with every relevant field unset
    # should read as genuinely absent, not as "present but empty" -- the
    # downstream precompute functions already treat None as the absence
    # signal, not {}.
    return (account_state or None), (user_tax_profile or None)


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

    # 86bc997wr -- plain values for error notes, set in run() before anything
    # can fail. Class-level defaults (immutable) so _note_error() can never hit
    # an AttributeError, even for a method exercised without going through run().
    _user_id = None
    _bind = None
    _run_context: dict | None = None
    _failure_noted = False
    _cancelled = False
    _degradation: DegradationCollector | None = None

    def _note_error(
        self,
        component: str,
        error_type: str,
        severity: str,
        message: object,
        *,
        agent_name: str | None = None,
        dedup_subtype: str | None = None,
        exc: BaseException | None = None,
        context: dict | None = None,
        terminal: bool = False,
        occurrence_count: int = 1,
    ) -> None:
        """Queue one failure for error_records (86bc997wr). Synchronous, does no
        I/O, touches no ORM object, and never raises -- so it is safe to call
        anywhere in a run: inside an except block, inside an asyncio.gather
        coroutine (an AsyncSession must never be written to from one), and on a
        session poisoned by a failed flush (where even reading run.run_id can
        raise -- see the comments in _run_pipeline). Nothing is written until
        _flush_error_notes() runs once, at the very end of run().

        `terminal=True` marks the branch that ends the run, so run()'s own
        catch-all doesn't record the same failure a second time when the
        exception propagates out.
        """
        try:
            self.__dict__.setdefault("_error_notes", []).append(
                build_error_note(
                    component,
                    error_type,
                    severity,
                    message,
                    run_id=getattr(self, "_run_id", None),
                    user_id=self._user_id,
                    stock_ticker=getattr(self, "_ticker", None),
                    agent_name=agent_name,
                    dedup_subtype=dedup_subtype,
                    exc=exc,
                    context={**(self._run_context or {}), **(context or {})},
                    occurrence_count=occurrence_count,
                )
            )
            if terminal:
                self._failure_noted = True
        except Exception:
            logger.warning("error_note_build_failed")

    def _note_agent_failure(
        self,
        stage: str,
        agent_id: str,
        errors: list[str],
        exc: Exception | None,
        runner=None,
    ) -> None:
        """One agent ended with no usable output. Soft validation failures on an
        agent that still completed are deliberately NOT recorded here -- they
        already live in agent_outputs.error_detail, and recording them too would
        flood this table. OllamaUnavailable is also not recorded per agent: it
        aborts the run and is noted once at the run level instead.

        Self-contained by design: the runner's last few attempts (how each one
        failed, and where its raw prompt and response were saved) go into the
        note, so diagnosing this failure never needs a join. The matching
        agent_outputs row is run_id + agent_name (they are the same string).

        Fingerprint: an exception is fingerprinted by where it was raised
        (build_error_note does that when dedup_subtype is None); a failure with
        no exception (the retries just ran out) by the agent plus the rule that
        kept failing, so two agents' failures never look like one problem.
        """
        first_error = errors[0] if errors else ""
        self._note_error(
            "agent",
            "agent_exception" if exc is not None else "all_retries_exhausted",
            "high",
            "; ".join(errors) or "agent returned no usable output",
            agent_name=agent_id,
            dedup_subtype=(
                None
                if exc is not None
                else f"{agent_id}:{rule_slug(first_error) if first_error else stage}"
            ),
            exc=exc,
            context={
                "stage": stage,
                "errors": [safe_text(e, 300) for e in errors[:10]],
                **summarize_call_log(getattr(runner, "call_log", None)),
            },
        )

    async def _release_session(self, db: AsyncSession) -> None:
        """Roll the run's own session back, quietly. It never raises, so it is safe
        inside an except/finally (a failing rollback must not replace the real
        error), and it is bounded, so a session left broken by a cancelled
        statement cannot hang the cleanup. Frees any write lock the session holds
        so the independent-session writes that follow are not blocked by it."""
        try:
            await asyncio.wait_for(db.rollback(), timeout=5)
        except Exception:
            pass

    async def _record_run_fingerprints(self, run: AnalysisRun, db: AsyncSession) -> None:
        """Store which prompt text and which code this run used (ledger BB-045), merged
        into `llm_config` so the existing `model` entry stays: `prompts_hash` plus the
        per-file `prompt_hashes` (the templates), and `code_hash` (the backend source,
        which writes much of what the model reads). Done before prepare() and committed
        straight away, on a session that is still clean, so it survives a run that
        fails early or is cancelled. Each hash is best effort and independent: a problem
        reading files must never stop a run, nor stop the other hash (a commit that
        fails is not swallowed, because a broken database would fail the run at its next
        commit anyway, and rolling back here would expire the run's attributes).
        Limit: both are taken when the run starts, so a file edited while a run is in
        flight is attributed to the state the run began with."""
        recorded: dict = {}
        try:
            fingerprints = prompt_fingerprints()
            recorded["prompts_hash"] = fingerprints["combined"]
            recorded["prompt_hashes"] = fingerprints["files"]
        except Exception:
            logger.warning("prompt_fingerprints_failed", exc_info=True)
        try:
            recorded["code_hash"] = code_fingerprint()
        except Exception:
            logger.warning("code_fingerprint_failed", exc_info=True)
        if not recorded:
            return
        run.llm_config = {**(run.llm_config or {}), **recorded}
        await db.commit()

    def _drain_degradation(self) -> None:
        """Turn what the data layer collected during this run (a provider that
        raised and was skipped, an LLM sub-call that got no answer, articles left
        unscored) into error notes. Called after prepare() returns OR raises, and
        again at the very end (the benchmark quote and other late fetches report
        too). Draining empties the collector, so nothing is noted twice. These are
        never terminal: the run carried on, on worse data, which is the point."""
        collector = self._degradation
        if collector is None:
            return
        try:
            for event in collector.drain():
                self._note_error(
                    "data_pipeline",
                    event.kind,
                    event.severity,
                    event.message or event.kind,
                    dedup_subtype=event.fingerprint,
                    context={
                        "stage": "data",
                        "provider": event.provider,
                        "op": event.op,
                        "exc_type": event.exc_type,
                        **event.context,
                    },
                    occurrence_count=event.count,
                )
        except Exception:
            logger.warning("degradation_drain_failed")

    def _note_ollama_unavailable(self, stage: str, exc: BaseException) -> None:
        """Environment-level failure that aborts the run. Noted once per run, at
        the handler, not once per agent (up to 9 agents can raise it in one
        pass -- see the OllamaUnavailable skip in _note_agent_failure's callers)."""
        self._note_error(
            "llm",
            "ollama_connection_failed",
            "critical",
            safe_text(exc, 4000) or "Ollama unavailable",
            dedup_subtype=stage,
            exc=exc,
            context={"stage": stage},
            terminal=True,
        )

    def _note_gate_failure(self, gate: str, reason: str | None) -> None:
        self._note_error(
            "orchestrator",
            "gate_failed",
            "high",
            reason or f"{gate} failed",
            dedup_subtype=gate,
            context={"stage": gate},
            terminal=True,
        )

    def _note_cio_incomplete(self, stage: str, errors: list[str], runner=None) -> None:
        first_error = errors[0] if errors else ""
        self._note_error(
            "agent",
            "all_retries_exhausted",
            "high",
            "; ".join(errors) or f"{stage} produced no usable output",
            agent_name=stage,
            dedup_subtype=f"{stage}:{rule_slug(first_error)}" if first_error else stage,
            context={
                "stage": stage,
                "errors": [safe_text(e, 300) for e in errors[:10]],
                **summarize_call_log(getattr(runner, "call_log", None)),
            },
            terminal=True,
        )

    async def _flush_error_notes(self) -> None:
        """Write every queued note in one independent session (never the run's
        own, which may be mid-rollback). Called once, from run()'s outermost
        `finally`, after the run session has committed or rolled back -- which
        is why none of the note sites has to care about session state."""
        try:
            notes = self.__dict__.get("_error_notes")
            if not notes or self._bind is None:
                return
            self._error_notes = []
            await write_error_notes(self._bind, notes)
        except Exception:
            logger.warning("error_notes_flush_failed")

    async def run(self, run: AnalysisRun, db: AsyncSession) -> None:
        """Public entry point -- a thin wrapper around _run_pipeline() so a
        run_quality_summary row (86bbwachy Phase 5) gets written exactly
        once regardless of which of _run_pipeline's own several
        return/raise exits actually fires. `finally` runs on every one of
        them (success, a `return`, or a propagating `raise`) without
        touching _run_pipeline's own internals at all -- this method's own
        signature is unchanged from before this phase, so every existing
        caller (api/routes/analysis.py's _run_analysis_background) needs no
        changes.

        The inner try/except around the summary write is the same
        defensive posture already used below for Shadow CIO: a bug in this
        SECONDARY write must never mask or replace whatever _run_pipeline
        itself raised or returned.
        """
        # Initialized HERE, not inside _run_pipeline -- this method is the
        # one that guarantees _write_run_quality_summary runs, so it must
        # also guarantee these exist regardless of how early _run_pipeline
        # itself fails (even before its own first line, in principle).
        # _write_run_quality_summary reads these unconditionally.
        self._gate1_passed = self._gate1_reason = None
        self._gate2_passed = self._gate2_reason = None
        # 86bc997wr -- plain values for error notes, read here while the run is
        # freshly loaded and the session is known clean, never from a failure
        # handler. Notes are queued through the run and flushed once, below.
        self._error_notes = []
        self._failure_noted = False
        self._degradation = DegradationCollector()
        self._run_id = run.run_id
        self._user_id = run.user_id
        self._bind = db.bind
        self._run_context = {"account_type": run.account_type, "timeline": run.timeline}
        # The ticker for error notes and log lines, resolved up front from the
        # Stock row (one small query on a session that is still clean) rather
        # than from the DataBundle, which doesn't exist yet when the most
        # common failure -- DataPipeline.prepare() -- happens. Overwritten with
        # bundle.stock.ticker once a real bundle exists.
        self._ticker = await self._lookup_ticker(run, db)
        # Every log line emitted while this run executes -- from any layer,
        # including the data providers and Router, with no call-site changes --
        # carries these fields. structlog's default processors already include
        # merge_contextvars, and asyncio tasks spawned in here inherit the
        # context. bound_contextvars unbinds on exit, so nothing leaks into the
        # next run served by the same process.
        log_context = {
            "run_id": str(run.run_id),
            "account_type": run.account_type,
            "timeline": run.timeline,
        }
        if self._ticker:
            log_context["ticker"] = self._ticker
        with structlog.contextvars.bound_contextvars(**log_context):
            await self._run_guarded(run, db)

    async def _lookup_ticker(self, run: AnalysisRun, db: AsyncSession) -> str | None:
        """Best effort: a missing ticker only costs notes and log lines their
        ticker field, never the run."""
        try:
            return (
                await db.execute(
                    select(Stock.canonical_ticker).where(Stock.stock_id == run.stock_id)
                )
            ).scalar_one_or_none()
        except Exception:
            return None

    async def _run_guarded(self, run: AnalysisRun, db: AsyncSession) -> None:
        """run()'s body: the pipeline, then the run_quality_summary write, then
        the single error-notes flush, each guarded so a failure in a later step
        never masks an earlier one."""
        # The data layer reports degradations (a provider that raised and was
        # skipped, an LLM sub-call that got no answer) to whichever collector is
        # current, so nothing below had to change signature. Reset in finally, so
        # it never leaks into the next run served by this process.
        collector_token = set_collector(self._degradation)
        try:
            try:
                await self._run_pipeline(run, db)
            except asyncio.CancelledError:
                # Ctrl+C, a server restart or a shutdown cancels the task
                # mid-run. CancelledError is a BaseException, so the handlers
                # below never see it, and before this the run stayed
                # non-terminal forever (blocking its ticker). Best effort, in a
                # separate session (this one may be mid-operation), and always
                # re-raised: cancellation must still propagate.
                self._cancelled = True
                self._note_error(
                    "orchestrator",
                    "run_cancelled",
                    "medium",
                    "Run cancelled while in progress (the process was stopping or interrupted).",
                    dedup_subtype="cancelled",
                    context={"stage": "cancelled"},
                    terminal=True,
                )
                # Release whatever write lock the run's own session still holds
                # BEFORE writing from a second connection. On an on-disk SQLite
                # file a held write transaction makes the cleanup below wait out
                # the lock timeout and fail, leaving the run stuck (an in-memory
                # test database hides this: it is one shared connection).
                await self._release_session(db)
                await end_run_now(
                    self._bind,
                    self._run_id,
                    stage="cancelled",
                    message="Run cancelled while in progress (the process was stopping or interrupted).",
                )
                raise
            except Exception as exc:
                # Anything that escapes _run_pipeline without one of its own
                # handled branches having recorded it first (a bug in a pass,
                # an unguarded recommendation/prediction write, ...).
                if not self._failure_noted:
                    self._note_error(
                        "orchestrator",
                        "pipeline_exception",
                        "critical",
                        describe_exception(exc),
                        exc=exc,
                        context={"stage": "run"},
                        terminal=True,
                    )
                raise
            finally:
                # A cancelled run's counters are incomplete and its session was
                # just rolled back (every attribute on `run` is now expired, and
                # reading one from async code raises MissingGreenlet), so no
                # summary is written for it: the run is already recorded as
                # cancelled, with a reason.
                if not self._cancelled:
                    try:
                        await self._write_run_quality_summary(run, db)
                    except Exception as exc:
                        # Noted first: the rollback below can itself fail.
                        self._note_error(
                            "orchestrator",
                            "pipeline_exception",
                            "low",
                            f"run_quality_summary write failed: {describe_exception(exc)}",
                            exc=exc,
                            context={"stage": "quality_summary"},
                        )
                        await self._release_session(db)
                        # self._run_id, not run.run_id: after a rollback every
                        # attribute on `run` is expired, and reading one here is
                        # the MissingGreenlet that would replace the real error.
                        logger.warning(
                            "run_quality_summary_write_failed",
                            run_id=str(self._run_id),
                            error=safe_text(exc),
                        )
        finally:
            # Outermost, so it runs even if the summary block's own rollback()
            # raised, and only after the run session has committed or rolled
            # back. A failure here is swallowed inside (see error_recorder).
            self._drain_degradation()  # anything reported after prepare() returned
            reset_collector(collector_token)
            await self._flush_error_notes()

    async def _run_pipeline(self, run: AnalysisRun, db: AsyncSession) -> None:
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

        # 86bbwachy Phase 2/3 capture context -- instance attributes, not
        # threaded through every private method's own signature, since a
        # fresh AnalysisOrchestrator() is constructed per run (confirmed:
        # api/routes/analysis.py's _run_analysis_background does
        # `AnalysisOrchestrator().run(run, db)`, a new instance every time),
        # so there is no cross-run leakage risk. Created BEFORE prepare()
        # runs, not after -- seq_counter is genuinely run-wide (spec §5.2),
        # and Phase 3 wires it into DataPipeline.prepare() too, so
        # precompute's own Ollama calls (sentiment scoring, filing-section
        # summarization -- both of which happen inside prepare(), before
        # Pass 1 ever starts) consume the lowest seq values in the SAME
        # single sequence Pass 1/Pass 2/CIO/Shadow CIO continue afterward,
        # not a separate 0-based numbering of their own. self._ticker is
        # set separately below, once bundle.stock exists -- prepare() needs
        # no ticker from here, it already resolves its own internally as
        # one of its first steps (stock_ref.ticker), well before either of
        # its own precompute LLM calls happen.
        self._run_id = run_id
        self._seq_counter = itertools.count()
        # self._gate1_passed/_gate1_reason/_gate2_passed/_gate2_reason
        # (86bbwachy Phase 5) are initialized in run(), not here -- see that
        # method's own comment on why it owns this instead of _run_pipeline.

        await self._record_run_fingerprints(run, db)

        try:
            bundle = await DataPipeline().prepare(
                run.stock_id, context, db, run_id=run_id, seq_counter=self._seq_counter
            )
        except Exception as exc:
            self._note_error(
                "data_pipeline",
                "prepare_exception",
                "high",
                describe_exception(exc),
                exc=exc,
                context={"stage": "data_pipeline"},
                terminal=True,
            )
            logger.error("data_pipeline_failed", run_id=str(run_id), error=safe_text(exc, 4000))
            run.status = RunStatus.FAILED
            # safe_text: a provider exception can carry a request URL with an
            # API key in it (open ticket 86bbq7dmj), and this is stored.
            run.error_log = [{"stage": "data_pipeline", "error": safe_text(exc)}]
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
        # commits prepare()'s own added-but-not-committed precompute
        # llm_calls rows (Phase 3's DataPipeline.prepare() only db.add()s,
        # matching that function's pre-existing "never commits" posture --
        # see its own docstring) in the same transaction as these tags.
        await db.commit()

        self._ticker = bundle.stock.ticker

        try:
            pass1_outputs, mechanical_quality = await self._run_pass1(run, bundle, db)
        except OllamaUnavailable as exc:
            self._note_ollama_unavailable("pass1", exc)
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
        self._gate1_passed, self._gate1_reason = gate1_passed, gate1_reason
        if not gate1_passed:
            self._note_gate_failure("gate1", gate1_reason)
            logger.warning("gate1_failed", run_id=str(run_id), reason=gate1_reason)
            run.status = RunStatus.FAILED
            run.error_log = [{"stage": "gate1", "error": gate1_reason}]
            await db.commit()
            return

        compressed = compress_pass1_outputs(
            pass1_outputs, bundles=_build_pass2_view_bundles(bundle), mechanical_quality=mechanical_quality
        )

        run.status = RunStatus.PASS2_RUNNING
        await db.commit()

        try:
            pass2_outputs = await self._run_pass2(run, bundle, compressed, db)
        except OllamaUnavailable as exc:
            self._note_ollama_unavailable("pass2", exc)
            run.status = RunStatus.FAILED
            run.error_log = [{"stage": "pass2", "error": "Ollama unavailable"}]
            await db.commit()
            raise

        run.status = RunStatus.PASS2_COMPLETE
        run.pass2_completed_at = datetime.now(UTC)
        await db.commit()

        gate2_passed, gate2_reason = gate2_check(pass2_outputs)
        self._gate2_passed, self._gate2_reason = gate2_passed, gate2_reason
        if not gate2_passed:
            self._note_gate_failure("gate2", gate2_reason)
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
            except OllamaUnavailable as exc:
                self._note_ollama_unavailable("cio_stage_a", exc)
                run.status = RunStatus.FAILED
                run.error_log = [{"stage": "cio_stage_a", "error": "Ollama unavailable"}]
                await db.commit()
                raise

            if not agent_completed(stage_a_result):
                self._note_cio_incomplete("cio_stage_a", stage_a_errors, cio_runner)
                logger.error("cio_stage_a_failed", run_id=str(run_id), errors=stage_a_errors)
                run.status = RunStatus.FAILED
                run.error_log = [
                    {"stage": "cio_stage_a", "error": safe_text("; ".join(stage_a_errors), 1000)}
                ]
                await db.commit()
                return

            try:
                stage_b_result, stage_b_errors = await cio_runner.run_stage_b(
                    bundle, stage_a_result, pass2_outputs.get("tax"), run.account_type
                )
            except OllamaUnavailable as exc:
                self._note_ollama_unavailable("cio_stage_b", exc)
                run.status = RunStatus.FAILED
                run.error_log = [{"stage": "cio_stage_b", "error": "Ollama unavailable"}]
                await db.commit()
                raise

            # Orchestrator-side forwarding (docs/agents/cio.md's
            # merge_output_cio_stage_b): tax_summary.dividend_yield_pct is a
            # fact Tax already computed, not an LLM judgment -- inject it
            # directly rather than trust the LLM to transcribe it back out of
            # its own prompt correctly. Confirmed live (AAPL run, 2026-09-23):
            # it never appeared in a real CIO Stage B output. Only
            # overwrites/adds this one key -- everything else in
            # stage_b_result is still the LLM's own, unmodified output.
            # (The Risk Advisor's stop-loss used to be forwarded here too; it
            # was removed from the pipeline on 2026-10-01: nothing read it.)
            if isinstance(stage_b_result, dict):
                tax_profile = (pass2_outputs.get("tax") or {}).get("tax_profile") or {}
                tax_summary = stage_b_result.get("tax_summary")
                if isinstance(tax_summary, dict):
                    tax_summary["dividend_yield_pct"] = tax_profile.get("dividend_yield_pct")
                    # The comparable cost of holding this stock in this account (annual dividend tax and the
                    # yield after it, same basis in every account), so the Portfolio Optimizer reads one
                    # finalized source when it compares the three accounts' results for a ticker.
                    tax_summary["annual_tax_drag_pct"] = tax_profile.get("annual_tax_drag_pct")
                    tax_summary["effective_after_tax_yield_pct"] = tax_profile.get("effective_after_tax_yield_pct")

            _add_agent_output_and_calls(db, run, "cio_stage_b", "synthesis", stage_b_result, stage_b_errors, cio_runner)
            await db.commit()
        finally:
            await _close_runner(cio_runner)

        if not agent_completed(stage_b_result):
            self._note_cio_incomplete("cio_stage_b", stage_b_errors, cio_runner)
            logger.error("cio_stage_b_failed", run_id=str(run_id), errors=stage_b_errors)
            run.status = RunStatus.FAILED
            run.error_log = [
                {"stage": "cio_stage_b", "error": safe_text("; ".join(stage_b_errors), 1000)}
            ]
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
            self._note_error(
                "agent",
                "output_validation_failed",
                "high",
                f"CIO output missing required fields: {missing}",
                dedup_subtype="missing:" + ",".join(missing),
                context={"stage": "recommendation", "missing": missing},
                terminal=True,
            )
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
            self._note_error(
                "agent",
                "agent_exception",
                "medium",
                describe_exception(exc),
                agent_name="shadow_cio",
                exc=exc,
                context={"stage": "shadow_cio"},
            )
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
            logger.warning(
                "shadow_cio_failed_non_blocking", run_id=str(run_id), error=safe_text(exc, 4000)
            )

        run.status = RunStatus.COMPLETED
        run.completed_at = datetime.now(UTC)
        await db.commit()

    async def _write_run_quality_summary(self, run: AnalysisRun, db: AsyncSession) -> None:
        """86bbwachy Phase 5 -- called from run()'s own finally block, so
        this fires exactly once per real run() invocation regardless of
        which of _run_pipeline's several return/raise exits actually fired.

        Everything here is either a fresh query against already-committed
        rows (every write earlier in _run_pipeline is immediately followed
        by its own db.commit(), so nothing here can be stale) or an
        instance attribute with nowhere else to live (self._gate1_passed
        etc.) -- _run_pipeline's own local scope (pass1_outputs,
        stage_a_result, ...) is long gone by the time this runs. Must
        db.add()+commit() its own row -- nothing upstream does that for it.
        A failure here is caught by run()'s own wrapper, not this method's
        job to guard against.
        """
        llm_calls = (
            await db.execute(select(LLMCall).where(LLMCall.run_id == run.run_id))
        ).scalars().all()
        total_calls = len(llm_calls)
        retry_calls = sum(1 for c in llm_calls if c.attempt > 1)
        total_llm_ms = sum(c.latency_ms or 0 for c in llm_calls)
        call_latencies = [c.latency_ms for c in llm_calls if c.latency_ms is not None]
        slowest_call_ms = max(call_latencies) if call_latencies else None
        total_prompt_tokens = sum(c.prompt_tokens or 0 for c in llm_calls)
        total_completion_tokens = sum(c.completion_tokens or 0 for c in llm_calls)
        total_thinking_chars = sum(c.thinking_chars or 0 for c in llm_calls)
        truncated_calls = sum(1 for c in llm_calls if c.finish_reason == "length")
        empty_content_calls = sum(1 for c in llm_calls if c.finish_reason == "empty_content")
        validator_failures = sum(1 for c in llm_calls if c.validator_passed is False)

        agent_outputs = (
            await db.execute(select(AgentOutput).where(AgentOutput.run_id == run.run_id))
        ).scalars().all()
        # See the module-level _NO_*_EXPECTED sets' own comment for why
        # these exclusions exist and how each was confirmed.
        #
        # An agent that FAILED (status="failed": no usable output at all) is
        # excluded from all three lists (86bc997wr). Its fields are empty
        # because it never produced anything, not because it produced a thin
        # answer, and listing it here made a run that died before reaching
        # these agents read like a data-quality problem (a real failed run
        # showed all five Pass 1 agents "with empty key_factors"). The failure
        # itself is reported as a failure -- run.error_log / error_records.
        produced = [a for a in agent_outputs if a.status != "failed"]
        agents_with_empty_key_factors = [
            a.agent_name for a in produced
            if a.agent_name not in _NO_KEY_FACTORS_EXPECTED and not a.key_factors
        ]
        agents_with_empty_risks = [
            a.agent_name for a in produced
            if a.agent_name not in _NO_RISKS_EXPECTED and not a.risks
        ]
        agents_with_empty_narrative = [
            a.agent_name for a in produced
            if a.agent_name not in _NO_NARRATIVE_EXPECTED and not a.narrative
        ]

        recommendation = (
            await db.execute(select(Recommendation).where(Recommendation.run_id == run.run_id))
        ).scalar_one_or_none()

        # run.triggered_at is a NAIVE datetime -- confirmed live: it's
        # populated via the column's own `default=func.now()` (a DB-side
        # CURRENT_TIMESTAMP under SQLite), which round-trips with no tzinfo
        # attached, unlike completed_at/pass1_completed_at/pass2_completed_at
        # (always set in Python via datetime.now(UTC)). Every timestamp in
        # this app is UTC in practice; attach it explicitly rather than let
        # a naive-minus-aware TypeError crash this method the first time it
        # runs against a real row.
        triggered_at = run.triggered_at
        if triggered_at.tzinfo is None:
            triggered_at = triggered_at.replace(tzinfo=UTC)
        wall_clock_ms = round((datetime.now(UTC) - triggered_at).total_seconds() * 1000)

        summary = RunQualitySummary(
            run_id=run.run_id,
            wall_clock_ms=wall_clock_ms,
            total_llm_ms=total_llm_ms,
            slowest_call_ms=slowest_call_ms,
            total_calls=total_calls,
            retry_calls=retry_calls,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_thinking_chars=total_thinking_chars,
            truncated_calls=truncated_calls,
            empty_content_calls=empty_content_calls,
            validator_failures=validator_failures,
            gate1_passed=self._gate1_passed,
            gate1_reason=self._gate1_reason,
            gate2_passed=self._gate2_passed,
            gate2_reason=self._gate2_reason,
            stock_outlook=recommendation.stock_outlook_direction if recommendation else None,
            overall_confidence=recommendation.overall_confidence if recommendation else None,
            agents_with_empty_key_factors=agents_with_empty_key_factors or None,
            agents_with_empty_risks=agents_with_empty_risks or None,
            agents_with_empty_narrative=agents_with_empty_narrative or None,
        )
        db.add(summary)
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

    async def _run_pass1(
        self, run: AnalysisRun, bundle: DataBundle, db: AsyncSession
    ) -> tuple[dict, dict[str, str]]:
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
        # 86bbummwp Tier 3 -- collected here, while runners are still open, for
        # compress_pass1_outputs() below. Mirrors analysis_confidence's own
        # in-memory-only path: never re-read from the DB within a single run
        # (see agents/compression.py's own docstring on why `bundles` -- and
        # now this -- are threaded in as a param rather than read back from
        # the AgentOutput rows _add_agent_output_and_calls just persisted).
        mechanical_quality: dict[str, str] = {}
        ollama_unavailable: OllamaUnavailable | None = None
        for agent_id, result, errors, exc in results:
            _add_agent_output_and_calls(db, run, agent_id, "pass1", result, errors, runners[agent_id])
            outputs[agent_id] = result
            if not agent_completed(result) and not isinstance(exc, OllamaUnavailable):
                self._note_agent_failure("pass1", agent_id, errors, exc, runners[agent_id])
            quality = getattr(runners[agent_id], "last_data_quality_assessment", None)
            if quality is not None:
                mechanical_quality[agent_id] = quality
            if isinstance(exc, OllamaUnavailable):
                ollama_unavailable = exc
        await db.commit()
        await asyncio.gather(*(_close_runner(r) for r in runners.values()))
        if ollama_unavailable is not None:
            raise ollama_unavailable
        return outputs, mechanical_quality

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

        # 86bc8efvb -- only the tax runner needs this; bull/bear/risk never
        # have and still don't. Fetched here, before the asyncio.gather below,
        # not concurrently with it: this is the ONLY db access anywhere in
        # this method (none of the 4 runners' own .run() calls touch db at
        # all), and AsyncSession isn't safe for concurrent use, so the
        # ordering is load-bearing, not just style. A missing/unset profile
        # degrades to (None, None) the same way every other optional-data
        # path in this codebase already does -- not a new failure mode.
        user_profile = await get_user_profile(db, run.user_id)
        account_state, user_tax_profile = _build_tax_inputs(user_profile)

        async def _run_bull_bear_tax(agent_id: str, runner):
            # Bull/Bear's own runners don't accept account_state/
            # user_tax_profile at all -- this branch is the only place tax
            # data is threaded through, despite bull/bear/tax sharing this one
            # dispatch helper. Confirmed neither pass2_bull_advocate.py nor
            # pass2_bear_advocate.py reference tax_metrics anywhere; this
            # helper's name describes shared execution mechanics (none of the
            # 3 have a Stage A/B split, unlike risk), not a shared data path.
            if agent_id == "tax":
                coro = runner.run(
                    bundle,
                    compressed,
                    run.account_type,
                    account_state=account_state,
                    user_tax_profile=user_tax_profile,
                )
            else:
                # Bull and Bear are account-neutral: one result per ticker and timeline.
                coro = runner.run(bundle, compressed)
            return await _run_contained("pass2", agent_id, coro)

        results = await asyncio.gather(
            _run_bull_bear_tax("bull", runners["bull"]),
            _run_bull_bear_tax("bear", runners["bear"]),
            _run_bull_bear_tax("tax", runners["tax"]),
            # Account-neutral, a single call (the account-specific stage B was removed 2026-10-01):
            # feeds the CIO's stage A and the Shadow CIO only.
            _run_contained("pass2", "risk", runners["risk"].run(bundle, compressed)),
        )

        outputs: dict[str, dict | None] = {}
        ollama_unavailable: OllamaUnavailable | None = None
        for agent_id, result, errors, exc in results:
            _add_agent_output_and_calls(db, run, agent_id, "pass2", result, errors, runners[agent_id])
            outputs[agent_id] = result
            if not agent_completed(result) and not isinstance(exc, OllamaUnavailable):
                self._note_agent_failure("pass2", agent_id, errors, exc, runners[agent_id])
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
            self._note_error(
                "agent",
                "all_retries_exhausted",
                "medium",
                "; ".join(errors) or "Shadow CIO produced no usable output",
                agent_name="shadow_cio",
                dedup_subtype="did_not_complete",
                context={
                    "stage": "shadow_cio",
                    "errors": [safe_text(e, 300) for e in errors[:10]],
                    **summarize_call_log(getattr(runner, "call_log", None)),
                },
            )
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
            self._note_error(
                "agent",
                "output_validation_failed",
                "medium",
                "Shadow CIO or primary CIO outlook missing; no divergence computed",
                agent_name="shadow_cio",
                dedup_subtype="missing_outlook",
                context={"stage": "shadow_cio"},
            )
            logger.warning(
                "shadow_cio_missing_outlook_for_divergence", run_id=str(run_id)
            )
            return

        distance, high_divergence = compute_outlook_distance(primary_outlook, shadow_outlook)
        async with db.begin_nested():
            db.add(ShadowPrediction(
                analysis_run_id=run_id,
                primary_outlook_direction=primary_outlook,
                primary_confidence=stage_a_result.get("confidence") or 0,
                primary_expected_return_tier=stage_a_result.get("expected_return_tier"),
                shadow_outlook_direction=shadow_outlook,
                shadow_confidence=result.get("confidence") or 0,
                shadow_expected_return_tier=result.get("expected_return_tier"),
                divergence_magnitude=_divergence_magnitude(distance),
                primary_cio_outlook_distance=distance,
                high_divergence=high_divergence,
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
            key_drivers=stage_a_result.get("key_decision_factors") or [],
            key_risks=[],
            synthesis_narrative=stage_b_result.get("synthesis_narrative", ""),
            dissenting_views=None,
            bull_case_strength=bull_conf,
            bear_case_strength=bear_conf,
            expected_return_tier=stage_b_result.get("expected_return_tier"),
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
