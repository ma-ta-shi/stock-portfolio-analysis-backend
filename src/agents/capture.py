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
    prefix = f"{seq}_{call_site}.{attempt}"

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
