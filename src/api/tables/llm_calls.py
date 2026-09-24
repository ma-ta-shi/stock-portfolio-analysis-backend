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
    # Stable identifier, e.g. agent:fundamental_analyst, agent:cio_stage_a,
    # agent:cio_stage_b, agent:shadow_cio, agent:risk_stage_b,
    # precompute:sentiment, precompute:filing_summarizer. CIO/Risk Advisor
    # get an explicit _stage_a/_stage_b suffix -- the same ambiguity
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
    # Backfilled via UPDATE after AgentOutput exists, not set at insert
    # time -- AgentOutput's own PK doesn't exist until after the retry loop
    # finishes, which is after this row's own insert.
    agent_output_id: Mapped[Optional[UUID]] = mapped_column(ForeignKey("agent_outputs.output_id"))
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    # relationships
    run: Mapped[Optional["AnalysisRun"]] = relationship()
    agent_output: Mapped[Optional["AgentOutput"]] = relationship()
