"""Base runner: Ollama client, retry logic, JSON parsing.

Production port of `simulation/runners/base.py` (ClickUp 86bc2d3t9) — same
retry/validation logic, converted to async + structlog per root CLAUDE.md's
"Architecture Rules" (agents are async, log via structlog, never print()).
`backend/src/multi_agents/` is an unrelated generic tutorial scaffold (wrong
model, sync client, no retry/validation) and is not extended by this module
-- this is an independent replacement for it, not a change to it. Removing
`multi_agents/` itself is out of scope here (no orchestrator references it
yet either way); that's cleanup for whoever wires the real orchestrator in.

Two things that matter and are easy to get wrong (see root CLAUDE.md's LLM
Routing section):

1. `think` MUST be "low", never False. gpt-oss:20b does not treat `think:
   false` as "suppress reasoning" (that flag only works for Qwen-family
   models) — left unset or False, it falls back to full reasoning effort
   (~35-50s/call, 2,700+ hidden thinking tokens) with no error raised.

2. `num_ctx` MUST be set explicitly in `options` on every call. Ollama
   silently truncates input past its own small default context window if
   this is omitted — no error, no log — and gpt-oss:20b then fabricates a
   confident, wrong answer from pretraining instead of failing. See
   docs/technical/ollama-num-ctx-finding.md.

Single-turn calls (`call_model`/`call_with_validation`, used by all ten
existing pipeline agents) go through `/api/chat`, not `/api/generate` —
`/api/generate` with `format: "json"` corrupts gpt-oss:20b output (garbled
text or an empty response despite a real eval_count); `/api/chat` separates
the `thinking` field from `content` before the JSON grammar is applied, so
the two don't collide. See docs/technical/ollama-generate-format-json-finding.md.

`_call_generate`/`call_with_validation_start`/`call_with_validation_continue`
(86bc2d414) are the one exception: Risk Advisor's Stage A->B needs raw
`/api/generate` context-array continuation (Stage B's stop-loss grounding
cites TECH price levels that live only in Stage A's *prompt*, not its JSON
output) -- see docs/technical/two-turn-execution-mechanism.md. Nothing else
in this pipeline uses continuation, including the CIO's own Stage A->B, which
that doc's own "Why CIO doesn't use this mechanism" section explains; do not
route other two-stage agents through this primitive without re-checking that
reasoning first.
"""

import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable

import aiohttp
import structlog

logger = structlog.get_logger(__name__)

# Overridable for local A/B comparison against another Ollama model. Default
# matches CLAUDE.md's current LLM Routing decision (2026-08-26) exactly.
MODEL = os.environ.get("SP_AGENT_MODEL", "gpt-oss:20b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Reasoning effort. Must be a string level, not a bool — see module docstring.
THINK = "low"

# Measured on the RTX 4070 Ti Super (16GB): gpt-oss:20b stays 100% GPU-resident
# at 16K, 32K and 64K num_ctx (no CPU offload). 32K gives the largest current
# payload (CIO, ~12K tokens) comfortable headroom while staying on the flat
# part of the latency-vs-num_ctx curve (docs/technical/ollama-num-ctx-finding.md).
NUM_CTX = 32768

# A CIO-scale call (~10K prompt tokens) measured ~11s; full-length generation
# is longer. Generous ceiling so a hang fails rather than blocking a run.
REQUEST_TIMEOUT_S = 300

MAX_RETRIES = 3


class OllamaUnavailable(RuntimeError):
    """Ollama isn't reachable, or the model isn't pulled."""


# Preflight runs at most once per process, not once per runner instance --
# every one of the ~10 runners constructed for a single analysis used to pay
# its own ~2s GET /api/tags at startup. The lock serializes the race where
# several runners are constructed concurrently (Pass 1/Pass 2 run agents in
# parallel) before the flag is set, so only one of them actually makes the
# request. A failure is deliberately NOT cached -- so the NEXT agent that
# calls in gets its own fresh, uncached attempt (useful if Ollama was still
# starting up). This does NOT mean the agent that just failed gets a second
# try within its own call_with_validation() retry loop -- see the comment
# there for why that's intentional, not a gap.
_preflight_lock = asyncio.Lock()
_preflight_checked = False


async def _preflight(session: aiohttp.ClientSession | None = None) -> None:
    """Fail fast with an actionable message rather than mid-run.

    Accepts an optional injected session so tests can exercise this without
    monkeypatching `aiohttp.ClientSession` itself -- production call sites
    never pass one, since this runs once globally, before any runner-owned
    session exists.
    """
    global _preflight_checked
    if _preflight_checked:
        return

    async with _preflight_lock:
        if _preflight_checked:
            return

        owns_session = session is None
        # Explicit `is None`, not `session or aiohttp.ClientSession()` -- an
        # injected fake that happens to define `__len__`/`__bool__` returning
        # falsy (e.g. a call counter starting at 0) would otherwise be silently
        # discarded in favor of a real session, defeating the injection.
        if session is None:
            session = aiohttp.ClientSession()
        try:
            try:
                async with session.get(
                    f"{OLLAMA_HOST}/api/tags", timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                raise OllamaUnavailable(
                    f"Cannot reach Ollama at {OLLAMA_HOST}. Is it running? ({exc})"
                ) from exc
        finally:
            if owns_session:
                await session.close()

        names = {m.get("name", "") for m in data.get("models", [])}
        if MODEL not in names:
            raise OllamaUnavailable(
                f"Model {MODEL!r} is not pulled. Run: ollama pull {MODEL}\n"
                f"Available: {', '.join(sorted(names)) or '(none)'}"
            )
        _preflight_checked = True


def _parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, stripping markdown fences if present.

    Ported from simulation/utils.py:87 -- the only piece of that module this
    runner needs.

    Uses raw_decode rather than a bare json.loads so trailing non-whitespace
    after the first balanced JSON value is tolerated and discarded, not a
    hard failure -- defense-in-depth (86bc2d414) against the trailing-brace
    failure mode documented in docs/technical/two-turn-execution-mechanism.md
    (a continuation successor turn with a flatter schema than its
    predecessor's has a reproducible tendency to append one stray extra `}`).
    The real fix is the explicit terminal-brace prompt instruction (already
    present in Risk Advisor's real Stage B prompt); this is a backstop for
    future prompt wording that drops it, not a replacement for that fix.
    Shared by both the /api/chat and /api/generate call paths since it's the
    same function -- /api/chat has never actually exhibited this, but there's
    no reason to tolerate it in one path and not the other.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    obj, _ = json.JSONDecoder().raw_decode(text)
    return obj


class BaseRunner:
    """Shared Ollama client + retry/validation loop for every agent runner.

    Owns an aiohttp session across its lifetime (created lazily, or injected
    for tests/composition) rather than opening one per call -- matches the
    session-lifecycle convention already established by
    `data/providers/finnhub.py` and `data/providers/boc.py` (optional
    injected session, `_owns_session` flag, `async with` support), not
    `data/precompute/sentiment.py`'s per-call session (that module is a
    stateless batch function with no object lifecycle; this class is not).
    """

    def __init__(self, session: aiohttp.ClientSession | None = None):
        self.session = session
        self._owns_session = session is None
        self.last_thinking: str | None = None
        self.last_timing: dict | None = None
        # `last_timing` is overwritten on every call, and `call_with_validation`
        # retries call `call_model` repeatedly -- so on its own it reports the
        # cost of the LAST attempt as if it were the cost of the agent.
        # `call_log` accumulates instead, one record per HTTP round trip, so
        # retry cost stays visible.
        self.call_log: list[dict] = []
        self.current_agent: str | None = None
        self.current_attempt: int = 1

    async def __aenter__(self) -> "BaseRunner":
        # Delegates to _get_session() rather than duplicating the
        # ClientSession(...) construction here -- one place decides how a
        # session gets created. Harmless when a session was injected: it just
        # returns it unchanged.
        await self._get_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._owns_session and self.session is not None:
            await self.session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
            )
            self._owns_session = True
        return self.session

    async def call_model(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 2000,
        temperature: float = 0.3,
    ) -> dict:
        await _preflight()

        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "think": THINK,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": NUM_CTX,
            },
        }
        session = await self._get_session()
        async with session.post(f"{OLLAMA_HOST}/api/chat", json=payload) as response:
            response.raise_for_status()
            body = await response.json()

        message = body.get("message", {})
        # Reasoning is kept out of the parsed text on purpose -- see module docstring.
        self.last_thinking = message.get("thinking")
        self.last_timing = {
            "total_duration_s": round(body.get("total_duration", 0) / 1e9, 2),
            "prompt_eval_count": body.get("prompt_eval_count"),
            "eval_count": body.get("eval_count"),
        }

        self.call_log.append(
            {
                "agent": self.current_agent or type(self).__name__,
                "attempt": self.current_attempt,
                **self.last_timing,
                # Reasoning tokens are billed and slow but never appear in the answer;
                # a non-zero count here on a `think: "low"` run is the regression
                # CLAUDE.md warns about.
                "thinking_chars": len(message.get("thinking") or ""),
            }
        )

        text = message.get("content", "")
        if not text.strip():
            # A blank content with a populated thinking field means the model
            # spent its budget reasoning and never emitted an answer -- usually
            # num_predict too low, or `think` set too high.
            raise json.JSONDecodeError("empty content from model", text or "", 0)
        return _parse_json_response(text)

    async def _call_generate(
        self,
        prompt: str,
        system: str | None = None,
        context: list[int] | None = None,
        max_tokens: int = 2000,
        temperature: float = 0.3,
    ) -> tuple[dict, list[int]]:
        """One /api/generate round trip -- the continuation-capable counterpart
        to call_model()'s /api/chat call (86bc2d414). `system` is only
        meaningful when starting a chain (context=None); a continuation call
        passes context and omits system -- confirmed live during 86bc2d414
        planning that Stage A's role definition and JSON output are already
        fully available to the model via context alone, nothing needs
        re-sending. Both `system` and `context` can technically be sent
        together (not asserted against here -- only *omitting* system on a
        continuation call was actually verified, so banning the combination
        would encode an unverified assumption as a hard constraint); today's
        two public callers below just never do it.

        Never sets format:"json" -- see
        docs/technical/ollama-generate-format-json-finding.md: /api/generate
        applies the JSON-grammar constraint to the raw output stream starting
        from token 1, including gpt-oss's own reasoning preamble, and the two
        fight each other, corrupting output. /api/chat avoids this by
        separating `thinking` from `content` first (see call_model above).
        Relies on the prompt's own "respond with ONLY valid JSON" instruction
        instead, which every agent prompt in this project already carries.

        Returns (parsed_result, context) -- `context` is the raw token-ID
        array from THIS call's response, for a caller to save (if starting a
        chain) or discard (if this was itself a continuation successor turn
        nothing continues from).
        """
        await _preflight()

        payload: dict = {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "think": THINK,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": NUM_CTX,
            },
        }
        if system is not None:
            payload["system"] = system
        if context is not None:
            payload["context"] = context

        session = await self._get_session()
        async with session.post(f"{OLLAMA_HOST}/api/generate", json=payload) as response:
            response.raise_for_status()
            body = await response.json()

        # Unlike call_model's message.thinking, /api/generate returns
        # `thinking` at the top level of the response body.
        self.last_thinking = body.get("thinking")
        self.last_timing = {
            "total_duration_s": round(body.get("total_duration", 0) / 1e9, 2),
            "prompt_eval_count": body.get("prompt_eval_count"),
            "eval_count": body.get("eval_count"),
            # Only meaningful for continuation calls (call_model's /api/chat
            # path doesn't capture this, since cache reuse isn't this
            # project's concern there) -- the two-turn-execution-mechanism
            # doc found this isn't worth tracking as a cost metric
            # (eval_duration dominates total latency regardless), but it IS
            # the one deterministic, wording-independent signal that a
            # continuation call actually reused Stage A's context rather than
            # silently starting fresh -- useful for tests and debugging even
            # though it's explicitly not a cost lever.
            "prompt_eval_cached_count": body.get("prompt_eval_cached_count"),
        }
        self.call_log.append(
            {
                "agent": self.current_agent or type(self).__name__,
                "attempt": self.current_attempt,
                **self.last_timing,
                "thinking_chars": len(body.get("thinking") or ""),
            }
        )

        text = body.get("response", "")
        if not text.strip():
            raise json.JSONDecodeError("empty response from model", text or "", 0)
        return _parse_json_response(text), body.get("context", [])

    # Per-agent reminder appended to the retry message. Runners may override with
    # the "Retry Prompt Injection" text from their own prompt spec. Left empty by
    # default so the generic correction instruction below is used alone.
    RETRY_REMINDER: str = ""

    def _build_retry_message(
        self, user_message: str, previous: dict | None, errors: list[str], attempt: int
    ) -> str:
        """Construct the retry turn.

        Shows the model its own previous response and scopes the requested
        edit to just the flagged errors -- sending "fix these errors" without
        showing the model what it previously said produces constraint
        whack-a-mole (fix one field, another regresses) instead of a targeted
        correction.
        """
        parts = [user_message]

        if previous is not None:
            parts.append(
                "YOUR PREVIOUS RESPONSE (attempt "
                f"{attempt}):\n{json.dumps(previous, indent=2, ensure_ascii=False)}"
            )
        else:
            parts.append(f"YOUR PREVIOUS RESPONSE (attempt {attempt}) could not be parsed as JSON.")

        parts.append("VALIDATION ERRORS:\n" + "\n".join(f"- {e}" for e in errors))

        instruction = (
            "Return the SAME analysis with ONLY the errors above corrected. Do not "
            "rewrite, re-derive, or re-order anything that was not flagged -- in "
            "particular, preserve every citation, evidence reference, and field that "
            "was already correct. Changing content that was not flagged is itself an "
            "error."
        )

        # Length violations are the one case where "preserve what was flagged" is
        # self-contradictory: the flagged field is exactly what has to change.
        # Without this carve-out the model obeys the preservation instruction and
        # returns byte-identical output.
        if any("too long" in e or "too short" in e for e in errors):
            # State the exact deficit. _auto_trim can salvage an overshoot but
            # nothing can expand a short field, so a short narrative burns every
            # attempt -- the most frequent terminal failure observed in the
            # harness. Giving a concrete character target is the only mechanical
            # help available.
            for err in errors:
                m = re.match(r"^(\w+): too short \((\d+) chars, min (\d+)", err)
                if m:
                    field, actual, need = m.group(1), int(m.group(2)), int(m.group(3))
                    short_by = need - actual
                    instruction += (
                        f"\n\nSPECIFIC TARGET: `{field}` is {short_by} characters short "
                        f"({actual} of {need} minimum, roughly {max(1, short_by // 6)} more "
                        f"words). Add substantive analysis to reach at least {need} "
                        f"characters -- do not pad with restatement."
                    )
            instruction += (
                "\n\nEXCEPTION -- a length error is flagged above. That field MUST be "
                "rewritten to fall inside its stated range; returning it unchanged is "
                "not a valid correction. If it is too long, condense it while keeping "
                "every citation and substantive point. If it is too short, expand the "
                "existing analysis with more specific detail -- do not pad, and do not "
                "drop citations to make room."
            )

        parts.append(instruction + "\n\nRespond with corrected JSON only.")
        if self.RETRY_REMINDER:
            parts.append(self.RETRY_REMINDER.strip())

        return "\n\n".join(parts)

    # Overshoot up to this fraction over the max is trimmed locally instead of
    # spending a retry. Beyond it, the output is too far off to salvage mechanically.
    AUTO_TRIM_TOLERANCE = 0.20

    @staticmethod
    def _auto_trim(result: dict, errors: list[str]) -> bool:
        """Trim small `too long` overshoots to the last sentence boundary, in place.

        The model can control narrative length roughly but not precisely --
        salvaging a small overshoot locally avoids burning a retry attempt the
        model demonstrably cannot win.

        Returns True if every error was resolved by trimming.
        """
        # Unit-aware: word limits are the Pass 1 norm, not the exception, so a
        # `chars`-only pattern would never match a `max 80 words` error.
        pattern = re.compile(r"^(\w+): too long \((\d+) (chars|words), max (\d+)")
        resolved = 0
        for err in errors:
            m = pattern.match(err)
            if not m:
                continue
            field, actual, unit, limit = (m.group(1), int(m.group(2)), m.group(3), int(m.group(4)))
            text = result.get(field)
            if not isinstance(text, str) or actual > limit * (1 + BaseRunner.AUTO_TRIM_TOLERANCE):
                continue
            # Word limits trim on word boundaries; char limits on characters.
            if unit == "words":
                window = " ".join(text.split()[:limit])
            else:
                window = text[:limit]
            cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
            half = (len(window) if unit == "words" else limit) * 0.5
            trimmed = window[: cut + 1].rstrip() if cut > half else window.rstrip()
            if trimmed:
                result[field] = trimmed
                resolved += 1

        # Over-long ARRAYS are the same class of imprecision as over-long strings.
        # Truncation is by IMPORTANCE, not position: every key_factors/risks entry
        # carries an `importance` of high|medium|low, so dropping the lowest-ranked
        # item is principled where dropping the last one would be arbitrary.
        # Original order is preserved among the survivors.
        #
        # UNDER-counts are deliberately NOT handled: nothing can invent a factor,
        # the same reason a too-short narrative cannot be salvaged.
        rank = {"high": 0, "medium": 1, "low": 2}
        arr_pattern = re.compile(r"^(\w+): need <=(\d+) items?, got (\d+)")
        for err in errors:
            m = arr_pattern.match(err)
            if not m:
                continue
            field, limit, actual = m.group(1), int(m.group(2)), int(m.group(3))
            items = result.get(field)
            if not isinstance(items, list) or len(items) != actual or actual <= limit:
                continue
            if actual > limit * (1 + BaseRunner.AUTO_TRIM_TOLERANCE) + 1:
                continue  # a gross overshoot is a compliance problem
            order = sorted(
                range(actual),
                key=lambda i: (
                    rank.get(
                        (items[i] or {}).get("importance") if isinstance(items[i], dict) else None,
                        1,
                    ),
                    i,
                ),
            )
            keep = sorted(order[:limit])
            result[field] = [items[i] for i in keep]
            resolved += 1

        # Return "I changed something", not "I fixed everything" -- array
        # overshoots arrive bundled with other errors, so the caller re-validates
        # on True and either passes outright or produces a shorter error list for
        # the retry message, which is strictly better than resending the original.
        #
        # Format violations where the VALID value is already present in the
        # string, e.g. `position_sizing_recommendation` coming back as
        # '3-5% (adopted from Risk Advisor)' -- a correct band with the schema's
        # own parenthetical appended. Extracting the band is deterministic;
        # spending an LLM call to re-ask for it is not.
        #
        # Only fires when the extracted value is UNAMBIGUOUS: exactly one
        # band-shaped token in the string. Enum violations are deliberately NOT
        # repaired here -- that's a semantic rule, and silently rewriting it
        # would mask a real defect rather than a format slip.
        band_pattern = re.compile(r"^(\w+): must be a percentage band")
        for err in errors:
            m = band_pattern.match(err)
            if not m:
                continue
            field = m.group(1)
            val = result.get(field)
            if not isinstance(val, str):
                continue
            bands = re.findall(r"\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*%", val)
            if len(bands) == 1:
                result[field] = bands[0].replace(" ", "")
                resolved += 1

        return resolved > 0

    async def _retry_loop(
        self,
        make_call: Callable[[str], Awaitable[tuple[dict, list[int] | None]]],
        user_message: str,
        validator: Callable[[dict], tuple[bool, list[str]]],
    ) -> tuple[dict, list[str], list[int] | None]:
        """Shared retry/validation core (86bc2d414 follow-up) used by
        call_with_validation, call_with_validation_start, and
        call_with_validation_continue -- the three differ only in HOW one
        attempt is made (/api/chat vs /api/generate, with or without an
        incoming continuation context), not in the retry control flow, which
        used to be duplicated three times. This project's own codebase has
        real, documented incidents from exactly this duplication shape
        (researcher_thesis_archetype drifting across 4 copies;
        CONFIDENCE_STATUS_LABELS maintained independently in two files) --
        sharing this loop is fixing the same class of risk before it recurs
        a third time, not speculative cleanup.

        `make_call(current_message)` performs exactly one HTTP round trip and
        returns (parsed_result, context) -- `context` is None from callers
        that have none to offer (call_with_validation's /api/chat path,
        call_with_validation_continue's successor-turn path); only
        call_with_validation_start's caller passes a make_call that returns a
        real one.

        Context is returned ONLY on a successful (validated, or
        auto-trimmed-into-validity) attempt; on exhaustion it's always None,
        deliberately, not the last (invalid) attempt's context -- a caller
        has no legitimate reason to build on output nothing ever confirmed
        was usable (this project's own Gate 1/Gate 2 checks elsewhere
        establish "don't proceed from incomplete upstream output" as the
        norm). Applies uniformly now, not just to the Stage A case that
        originally motivated it -- callers that never pass a real context
        back (call_with_validation, call_with_validation_continue) are
        unaffected either way, since they already discard the third
        return value.

        Deliberately does NOT catch OllamaUnavailable (raised by the
        preflight check inside call_model/_call_generate on the process's
        first call only -- see _preflight_lock above). Both cases it
        represents -- Ollama isn't running, or the model isn't pulled -- are
        non-transient config problems, not network blips: retrying for up to
        MAX_RETRIES backoff cycles cannot fix either one, and doing so would
        flatten OllamaUnavailable's specific, actionable message into a
        generic "Ollama request error" string in `last_errors`,
        indistinguishable from a real timeout. Letting it propagate uncaught
        and structurally distinct is also what will let a future orchestrator
        tell "the whole environment is broken, abort the run" apart from
        "this one agent had a rough call, mark it failed and move on."
        """
        last_result = {}
        last_errors: list[str] = []
        current_message = user_message

        for attempt in range(MAX_RETRIES):
            try:
                self.current_attempt = attempt + 1
                result, context = await make_call(current_message)
                passed, errors = validator(result)
                if self.call_log:
                    self.call_log[-1]["passed"] = passed
                    self.call_log[-1]["errors"] = errors
                if passed:
                    return result, [], context
                # Salvage small length overshoots locally rather than spending a
                # retry the model demonstrably cannot win (see _auto_trim).
                if self._auto_trim(result, errors):
                    passed, errors = validator(result)
                    if passed:
                        logger.info("auto_trimmed_length_bound")
                        return result, [], context
                last_result = result
                last_errors = errors
                current_message = self._build_retry_message(
                    user_message, result, errors, attempt + 1
                )
                logger.warning("validation_attempt_failed", attempt=attempt + 1, errors=errors)
            except json.JSONDecodeError as exc:
                logger.warning("json_parse_error", attempt=attempt + 1, error=str(exc))
                last_errors = [f"JSON parse error: {exc}"]
                current_message = self._build_retry_message(
                    user_message, None, last_errors, attempt + 1
                )
                await asyncio.sleep(1)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.error("ollama_request_error", attempt=attempt + 1, error=str(exc))
                last_errors = [f"Ollama request error: {exc}"]
                await asyncio.sleep(2**attempt)

        logger.error("all_retry_attempts_failed", max_retries=MAX_RETRIES, errors=last_errors)
        return last_result, last_errors, None

    async def call_with_validation(
        self,
        system_prompt: str,
        user_message: str,
        validator: Callable[[dict], tuple[bool, list[str]]],
        max_tokens: int = 2000,
        temperature: float = 0.3,
    ) -> tuple[dict, list[str]]:
        """Call the model with validation and retry on failure. Returns
        (output, errors). Retry/validation control flow lives in
        _retry_loop; see its docstring for what's shared and why."""

        async def _make_call(msg: str) -> tuple[dict, list[int] | None]:
            return await self.call_model(system_prompt, msg, max_tokens, temperature), None

        result, errors, _ = await self._retry_loop(_make_call, user_message, validator)
        return result, errors

    async def call_with_validation_start(
        self,
        system_prompt: str,
        user_message: str,
        validator: Callable[[dict], tuple[bool, list[str]]],
        max_tokens: int = 2000,
        temperature: float = 0.3,
    ) -> tuple[dict, list[str], list[int] | None]:
        """Start of a continuation chain via /api/generate (86bc2d414) -- same
        retry/validation semantics as call_with_validation above (see
        _retry_loop), different endpoint. Each retry attempt is a fresh call
        (no context in), matching how call_with_validation's own retries work
        today (a failed attempt is not itself continued from).

        Context is returned ONLY on success -- see _retry_loop's docstring
        for the full reasoning (a caller has no legitimate reason to continue
        a Stage B call from a Stage A that never validated).
        """

        async def _make_call(msg: str) -> tuple[dict, list[int]]:
            return await self._call_generate(
                msg, system=system_prompt, max_tokens=max_tokens, temperature=temperature
            )

        return await self._retry_loop(_make_call, user_message, validator)

    async def call_with_validation_continue(
        self,
        prompt: str,
        context: list[int],
        validator: Callable[[dict], tuple[bool, list[str]]],
        max_tokens: int = 2000,
        temperature: float = 0.3,
    ) -> tuple[dict, list[str]]:
        """Successor turn of a continuation chain via /api/generate
        (86bc2d414). Every retry attempt re-continues from the SAME `context`
        passed in -- never chains retry N's own new context onto retry N-1's,
        which would silently drift the chain away from Stage A's real state
        with each failed attempt (the closure below captures `context` once,
        from this method's own argument, never from a call's return value).

        num_ctx sizing note: reuses the module's existing NUM_CTX (32768),
        which must cover the CUMULATIVE context (predecessor prompt + response
        + this turn's own incremental prompt), not just this turn's visible
        prompt -- see docs/technical/two-turn-execution-mechanism.md.
        Live-verified during 86bc2d414 planning: a real Risk Advisor Stage A+B
        chain's cumulative prompt_eval_count reached 6,714 tokens for the
        Stage B call, comfortably within 32768.
        """

        async def _make_call(msg: str) -> tuple[dict, list[int] | None]:
            result, _ = await self._call_generate(
                msg, context=context, max_tokens=max_tokens, temperature=temperature
            )
            return result, None

        result, errors, _ = await self._retry_loop(_make_call, prompt, validator)
        return result, errors

    def timing_summary(self) -> dict:
        """Aggregate `call_log` into the numbers the 30-120s pipeline target is judged on.

        `retry_calls` is the figure that matters most: it is invisible in
        `last_timing` by construction, and it is what turns an 11-call analysis
        into a 15-call one.
        """
        log = self.call_log
        if not log:
            return {"calls": 0}
        total = sum(c.get("total_duration_s") or 0 for c in log)
        return {
            "calls": len(log),
            "retry_calls": sum(1 for c in log if c.get("attempt", 1) > 1),
            "total_llm_s": round(total, 1),
            "slowest_call_s": max((c.get("total_duration_s") or 0) for c in log),
            "prompt_tokens": sum(c.get("prompt_eval_count") or 0 for c in log),
            "completion_tokens": sum(c.get("eval_count") or 0 for c in log),
            "thinking_chars": sum(c.get("thinking_chars") or 0 for c in log),
            "per_agent": [
                {
                    k: c.get(k)
                    for k in (
                        "agent",
                        "attempt",
                        "total_duration_s",
                        "eval_count",
                        "thinking_chars",
                    )
                }
                for c in log
            ],
        }
