from sqlalchemy import ForeignKey, String, JSON, Text, Boolean, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional

class LLMCall(Base):
    """One row per LLM HTTP round trip -- every attempt, success or
    failure, not just the accepted one (86bbwachy Phase 2,
    docs/technical/run-instrumentation.md §5.3). Raw prompt/response text
    lives in artifact files on disk (prompt_path/context_path/response_path
    below), not here -- this table is metrics/IDs/params/status only,
    queryable and joinable; the files are the "cat the actual prompt" side.
    """
    __tablename__ = "llm_calls"
    call_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    # Nullable -- standalone precompute (cache warming), audit rigs, and
    # backtest runs have no real AnalysisRun. context_tag says what it was
    # when this is null.
    run_id: Mapped[Optional[UUID]] = mapped_column(ForeignKey("analysis_runs.run_id"), index=True)
    context_tag: Mapped[str] = mapped_column(String(20))  # analysis|cache_warm|audit|backtest
    # Per-run monotonic round-trip counter, assigned by the caller (see
    # orchestrator.py's own seq counter) -- also the artifact filename
    # prefix. Only guaranteed unique per run, not strictly wall-clock
    # ordered under Pass 1/Pass 2's real concurrency (spec §5.2).
    seq: Mapped[Optional[int]]
    # Stable identifier, derived from the runner's own current_agent (see
    # agents/base.py's call_model/_call_generate). Confirmed live, not
    # guessed: agent:fund, agent:tech, agent:rsrch, agent:sent, agent:macro
    # (Pass 1's own short codes), agent:bull, agent:bear, agent:tax,
    # agent:risk_stage_a, agent:risk_stage_b, agent:cio_stage_a,
    # agent:cio_stage_b, agent:shadow_cio; precompute:sentiment/
    # precompute:filing_summarizer from Phase 3. CIO/Risk Advisor get an
    # explicit _stage_a/_stage_b suffix -- the same ambiguity
    # AgentOutput.agent_name already hit once (fixed this session by
    # renaming "cio" to cio_stage_a/cio_stage_b) is not repeated here.
    call_site: Mapped[str] = mapped_column(String(40), index=True)
    agent_pass: Mapped[Optional[str]] = mapped_column(String(10))  # pass1|pass2|synthesis|precompute
    attempt: Mapped[int] = mapped_column(default=1)  # 1-indexed; retries are separate rows
    model: Mapped[str] = mapped_column(String(50))
    # The actual options sent: num_ctx, num_predict, temperature, think,
    # seed if set. Critical -- this is what catches the num_ctx/think
    # silent-regression class root CLAUDE.md's own LLM Routing section
    # warns about (spec §5.3's own words).
    options_json: Mapped[dict] = mapped_column(JSON)
    # Prompt template identity, for grouping calls by template version --
    # distinct from the rendered prompt text in the artifact file. Null for
    # calls with no template (raw precompute prompts have none).
    template_id: Mapped[Optional[str]] = mapped_column(String(60))
    template_version: Mapped[Optional[str]] = mapped_column(String(10))
    # Relative paths into the run artifact directory (runs_dir, see
    # agents/capture.py). context_path is null for calls with no template
    # (raw precompute prompts) -- spec §5.2.
    prompt_path: Mapped[Optional[str]] = mapped_column(String(255))
    context_path: Mapped[Optional[str]] = mapped_column(String(255))
    response_path: Mapped[Optional[str]] = mapped_column(String(255))
    prompt_tokens: Mapped[Optional[int]]  # Ollama's prompt_eval_count
    completion_tokens: Mapped[Optional[int]]  # Ollama's eval_count -- includes reasoning tokens
    # len() of the thinking field. Ollama exposes no separate reasoning-token
    # count; a non-zero/large count here on a think:"low" call is the
    # regression flag (matches BaseRunner's own existing call_log).
    thinking_chars: Mapped[Optional[int]]
    latency_ms: Mapped[Optional[int]]  # Ollama's total_duration
    # Mapped from Ollama's done_reason (stop/length) plus the locally-
    # detected empty_content/http_error/timeout cases agents/base.py
    # already distinguishes via its own exception handling.
    finish_reason: Mapped[Optional[str]] = mapped_column(String(20))
    parsed_ok: Mapped[Optional[bool]] = mapped_column(Boolean)
    parse_error: Mapped[Optional[str]] = mapped_column(Text)
    validator_passed: Mapped[Optional[bool]] = mapped_column(Boolean)  # null for calls with no validator
    validator_errors: Mapped[Optional[list]] = mapped_column(JSON)
    auto_trimmed: Mapped[Optional[bool]] = mapped_column(Boolean)  # did _auto_trim-equivalent salvage fire
    # FK -> agent_outputs.output_id, set only on the attempt that produced
    # the accepted output (null for superseded retries and non-agent calls)
    # -- non-null therefore also marks "this was the accepted attempt".
    # Set directly at construction, not backfilled via a post-insert UPDATE
    # as originally planned: AgentOutput.output_id's own uuid4() default is
    # client-side, so orchestrator.py's _agent_output_row generates that PK
    # itself before either row is ever added to the session, and both rows
    # go in already fully formed. See that function's own docstring.
    agent_output_id: Mapped[Optional[UUID]] = mapped_column(ForeignKey("agent_outputs.output_id"))
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    # relationships
    run: Mapped[Optional["AnalysisRun"]] = relationship()
    agent_output: Mapped[Optional["AgentOutput"]] = relationship()

    @classmethod
    def from_call_log_entry(cls, entry: dict, *, run_id, agent_pass: str) -> "LLMCall":
        """Builds one row from a call_log-shaped dict -- the one place this
        mapping happens, shared by both real producers of that shape:
        agents/base.py's BaseRunner.call_log (services/orchestrator.py's
        own _llm_call_rows) and agents/capture.py's CaptureContext.call_log
        for precompute (data/pipeline.py's own _precompute_llm_call_rows).

        86bbwachy Phase 3, consolidated from what was two near-identical
        ~20-line field mappings in those two files (found on review) --
        call_site/model are hard requirements here (entry["x"], no
        fallback default), not defensive .get(key, default) reads: both
        real producers always set them (agents/base.py's call_model/
        _call_generate, agents/capture.py's record_call), so a fallback
        was only ever masking a test fixture that hadn't bothered to set
        them, not a real production gap.
        """
        total_duration_s = entry.get("total_duration_s")
        return cls(
            run_id=run_id,
            context_tag=entry.get("context_tag", "analysis"),
            seq=entry.get("seq"),
            call_site=entry["call_site"],
            agent_pass=agent_pass,
            attempt=entry.get("attempt", 1),
            model=entry["model"],
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
            # None the same way _market_cap_bucket's own comment (in
            # services/orchestrator.py) already warns against for a
            # different field.
            latency_ms=round(total_duration_s * 1000) if total_duration_s is not None else None,
            finish_reason=entry.get("finish_reason"),
            parsed_ok=entry.get("parsed_ok"),
            parse_error=entry.get("parse_error"),
            validator_passed=entry.get("validator_passed"),
            validator_errors=entry.get("validator_errors"),
            auto_trimmed=entry.get("auto_trimmed", False),
        )
