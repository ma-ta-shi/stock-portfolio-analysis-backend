"""Shared LLM-call capture primitive (86bbwachy Phase 2,
docs/technical/run-instrumentation.md). The one place every Ollama round
trip's raw prompt/response gets written to disk as artifact files -- used
by agents/base.py's retry-wrapped calls (call_model/_call_generate) and,
from Phase 3, precompute's simpler one-shot calls (sentiment.py,
filing_summarizer.py). Deliberately DB-agnostic: this module never touches
the database. The caller (orchestrator.py for agent calls,
DataPipeline.prepare() for precompute calls -- the two places that already
own a real AsyncSession, see this ticket's own plan for why) inserts the
actual llm_calls row using the paths and metadata this returns.
"""
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

# Matches agents/base.py's own OLLAMA_HOST-style env var convention.
# Default var/runs/ per spec §5.2; must be gitignored (already done, see
# .gitignore's own 2026-09-24 entry -- added proactively, before this file
# existed, specifically so there was never a window where this could slip
# through uncovered).
RUNS_DIR = os.environ.get("SP_RUNS_DIR", "var/runs")


def artifact_dir(run_id: Any, ticker: str, as_of: date | None = None) -> Path:
    """{runs_dir}/{YYYY-MM-DD}/{ticker}_{run_id[:8]}/ -- spec §5.2.
    `run_id` accepts anything str()-able (a UUID object in production, a
    plain string in tests) since this module has no dependency on the real
    ORM/UUID types. A standalone call with no run_id (cache warming, audit
    rigs -- llm_calls.run_id is nullable for exactly this case) uses
    "standalone" in place of the run_id prefix rather than crashing on
    None[:8].
    """
    day = (as_of or date.today()).isoformat()
    run_part = str(run_id)[:8] if run_id else "standalone"
    return Path(RUNS_DIR) / day / f"{ticker}_{run_part}"


def write_call_artifacts(
    *,
    run_id: Any,
    ticker: str,
    seq: int,
    call_site: str,
    attempt: int,
    prompt_text: str,
    context_payload: dict | list | None,
    response_body: dict,
) -> dict[str, str | None]:
    """Writes the three artifact files for one LLM round trip and returns
    their paths, relative to RUNS_DIR (spec §5.2's own prompt_path/
    context_path/response_path columns on llm_calls). context_path is None
    when context_payload is None -- raw precompute prompts (sentiment
    scoring, filing summarization) have no structured template input to
    snapshot, only the plain-string prompt itself; agent calls always pass
    one, since agents/prompts.py's fill() always has a real values dict.

    Real disk I/O, deliberately synchronous: these files are small (a
    rendered prompt, a JSON response body), and forcing microseconds of
    local file writes onto a thread pool would be pure overhead with no
    real benefit, unlike the actual network call this accompanies.
    """
    directory = artifact_dir(run_id, ticker)
    directory.mkdir(parents=True, exist_ok=True)
    # call_site (e.g. "agent:tech") keeps its colon in the llm_calls DB
    # column -- only the filesystem-facing prefix needs it stripped.
    # Confirmed live (2026-09-24, real AAPL run on this Windows dev box):
    # a literal ":" in a Windows/NTFS filename is not a normal character,
    # it starts an Alternate Data Stream. `Path.write_text()` on
    # "0_agent:tech.1.prompt.txt" silently succeeds -- but it creates a
    # 0-byte file literally named "0_agent" plus a *hidden* ADS named
    # "tech.1.prompt.txt" attached to it, invisible to ls/dir/git/backup
    # tools (confirmed via `Get-Item -Stream *`). Pytest's own read-back
    # in the same process didn't catch this because Python's open() on
    # Windows applies the exact same ADS interpretation on both the write
    # and the read, so the round trip "succeeds" even though the artifact
    # is corrupted from every other tool's point of view.
    safe_call_site = call_site.replace(":", "-")
    prefix = f"{seq}_{safe_call_site}.{attempt}"

    prompt_path = directory / f"{prefix}.prompt.txt"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    context_path = None
    if context_payload is not None:
        context_path = directory / f"{prefix}.context.json"
        context_path.write_text(
            json.dumps(context_payload, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )

    response_path = directory / f"{prefix}.response.json"
    response_path.write_text(
        json.dumps(response_body, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    root = Path(RUNS_DIR).resolve()
    return {
        "prompt_path": str(prompt_path.resolve().relative_to(root)),
        "context_path": str(context_path.resolve().relative_to(root)) if context_path else None,
        "response_path": str(response_path.resolve().relative_to(root)),
    }


# Ollama's own done_reason values pass through unchanged; this exists so the
# empty_content/http_error/timeout cases agents/base.py's own exception
# handling already distinguishes (spec §5.3) share one mapping point rather
# than each call site inventing its own string.
def map_finish_reason(
    done_reason: str | None, *, empty_content: bool = False
) -> str | None:
    if empty_content:
        return "empty_content"
    return done_reason


@dataclass
class CaptureContext:
    """Bundles what a precompute LLM call site (sentiment.py's
    _score_article, filing_summarizer.py's summarize_filing_section) needs
    to capture itself the same way agents/base.py's BaseRunner does --
    86bbwachy Phase 3. One object threaded down through the call chain
    (summarize_news, build_research_sources, build_filing_digests all just
    pass this straight through unchanged) instead of three separate
    parameters at every level in between.

    None at every real call site by default -- DataPipeline.prepare() only
    builds one when it received a real seq_counter from the orchestrator
    (see that module's own comments), so every existing precompute test
    that has never heard of capture keeps working completely unchanged: no
    new files written, no new behavior, same as agents/base.py's own
    "gated on ticker being set" precedent.
    """
    run_id: Any
    ticker: str
    seq_counter: Iterator[int]
    call_log: list[dict] = field(default_factory=list)


def record_call(
    capture: CaptureContext,
    *,
    call_site: str,
    model: str,
    options: dict,
    prompt_text: str,
    response_body: dict,
    thinking_chars: int,
    empty_content: bool,
    parsed_ok: bool,
    parse_error: str | None = None,
) -> None:
    """Writes one precompute LLM call's artifact files and appends its
    llm_calls-shaped record to capture.call_log -- the one-shot-call
    counterpart to agents/base.py's own inline capture block inside
    call_model/_call_generate (86bbwachy Phase 3).

    attempt is always 1: neither sentiment.py's _score_article nor
    filing_summarizer.py's summarize_filing_section retries (the
    2026-08-07 no-fallback-chains decision applies here too -- a genuine
    failure is a missing value, not a retried call), so there is never a
    second attempt to number.

    No context_payload -- matches write_call_artifacts' own contract:
    a raw precompute prompt has no structured template input to snapshot,
    only the plain rendered string itself.

    total_duration/prompt_eval_count/eval_count/done_reason are read
    directly off response_body -- confirmed the same top-level shape on
    both Ollama endpoints these two callers use (/api/chat for sentiment,
    /api/generate for filing summarization), matching agents/base.py's own
    call_model/_call_generate, which read the identical fields the same
    way from each endpoint's own body. thinking_chars/empty_content are
    NOT read here, since the two endpoints disagree on where "thinking"
    and the real answer text live (message.thinking/message.content for
    /api/chat, top-level thinking/response for /api/generate) -- the
    caller already knows its own shape and extracts these itself.
    """
    seq = next(capture.seq_counter)
    paths = write_call_artifacts(
        run_id=capture.run_id,
        ticker=capture.ticker,
        seq=seq,
        call_site=call_site,
        attempt=1,
        prompt_text=prompt_text,
        context_payload=None,
        response_body=response_body,
    )
    # response.json() succeeding only guarantees valid JSON, not a dict --
    # a top-level JSON null/string/list/number is just as valid, and a
    # malformed/unexpected Ollama response is exactly the case this
    # function exists to record, not one it can assume away. Guarded here,
    # once, rather than trusting every caller to pre-sanitize response_body
    # before passing it in.
    safe_body = response_body if isinstance(response_body, dict) else {}
    total_duration = safe_body.get("total_duration")
    capture.call_log.append({
        "call_site": call_site,
        "seq": seq,
        "context_tag": "analysis",
        "model": model,
        "options_json": options,
        "attempt": 1,
        "total_duration_s": round(total_duration / 1e9, 2) if isinstance(total_duration, (int, float)) else None,
        "prompt_eval_count": safe_body.get("prompt_eval_count"),
        "eval_count": safe_body.get("eval_count"),
        "thinking_chars": thinking_chars,
        "finish_reason": map_finish_reason(safe_body.get("done_reason"), empty_content=empty_content),
        "parsed_ok": parsed_ok,
        "parse_error": parse_error,
        **paths,
    })
