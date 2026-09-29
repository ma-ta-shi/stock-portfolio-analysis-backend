"""Degradation events: data or model calls that quietly went wrong (86bc997wr).

The data layer is built to keep going. A provider that raises is skipped in
favor of the next one in its chain; an LLM sub-call that times out just leaves
an article unscored or a filing without a digest. That resilience is right, but
it was invisible: the run continued on worse data and the only trace was a
console warning, gone with the scrollback. This module makes those events
recordable without changing any behavior.

How it works, and why it is shaped like this:

- A DegradationCollector is a plain in-memory list of events. The orchestrator
  creates one per run and sets it in a ContextVar; the hook points in the
  providers call `report(...)`, which is a NO-OP when no collector is set. So
  the data layer never imports the services layer, no signature changes, and
  code that runs outside an orchestrated run (tests, scripts) is unaffected.
- A ContextVar rather than a parameter because `Router` is constructed in
  several places the orchestrator cannot reach (inside DataPipeline.prepare(),
  the benchmark quote, the API route). asyncio tasks inherit the value, so the
  gathered provider calls inside prepare() report into the same collector.
- Events are deduplicated per (provider, op, kind) with a count: 250 unscored
  sentiment articles are one event with count 250, not 250 rows.
- Every method here swallows its own failures. Reporting must never change what
  a provider call returns, logs, retries or raises.
- The orchestrator drains the collector into error notes (see
  AnalysisOrchestrator._drain_degradation), which is what makes them queryable
  in error_records.

`classify()` decides an event's severity. It is the single place that encodes
"which failures are expected".
"""

import contextvars
from dataclasses import dataclass, field

_MESSAGE_LIMIT = 500

# FMP endpoints (the path FMPDataProvider._request is called with) whose Router chain
# has another provider after FMP, so an FMP decline does not lose the data. Keep this
# in step with US_CHAINS in providers/router.py; an endpoint missing here is medium.
_FMP_ENDPOINTS_WITH_FALLBACK = frozenset(
    {
        "quote",
        "dividends",
        "historical-price-eod/full",
        "profile",
        "analyst-estimates",
        "earnings",
        "earnings-calendar",
    }
)

# Event kinds.
LINK_FAILED = "link_failed"  # a provider raised; the chain moved to the next link
EMPTY_AFTER_FAILURE = "empty_after_failure"  # no provider produced data and at least one raised
AUTH_FAILED = "auth_failed"  # a 401: the call fails loud, this records that it did
LLM_REQUEST_FAILED = "llm_request_failed"  # a precompute LLM call got no usable response
SENTIMENT_UNSCORED = "sentiment_unscored"  # articles left without a sentiment score
NOT_COVERED = (
    "not_covered"  # the provider answered but will not serve this (402 / 403 / empty profile)
)
FETCH_FAILED = "fetch_failed"  # a call raised or errored and the provider swallowed it
DATA_MISSING = "data_missing"  # the call worked but the expected data is absent


@dataclass
class DegradationEvent:
    provider: str
    op: str
    kind: str
    severity: str
    message: str
    exc_type: str | None = None
    count: int = 1
    context: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.provider, self.op, self.kind)

    @property
    def fingerprint(self) -> str:
        return f"data:{self.provider}:{self.op}:{self.kind}"


def classify(provider: str, op: str, kind: str, context: dict | None = None) -> str:
    """The severity of an event. The single place that encodes how much
    each kind of degradation matters. `low` events are hidden from reports unless
    --all is passed.

    Written from what real runs produced (2026-09-29), not guessed:
    - Healthy runs of a US large-cap (AAPL), a US mid-cap (RDDT, on FMP's free
      tier) and a Canadian ticker (TD.TO) produced ZERO events. The expected
      quirks (FMP 402 on mid-caps, a provider that does not cover a request) do
      not raise, so they never reach a hook: there is nothing to suppress, and
      no "expected" table to invent.
    - A dead Ollama produced 248 failed sentiment calls (one event, counted), the
      248-of-248 unscored articles those left, and 2 failed filing summaries.
    - A rejected FMP key produced an auth_failed event (and the run fails loud).

    Severity therefore follows meaning:
    - auth_failed: high. A bad or expired key needs a human.
    - empty_after_failure: medium. No provider produced data and one raised, so an
      outage may be hiding the data the agents will now lack.
    - link_failed: low. A provider raised, but the chain moved on; if nothing
      answered, empty_after_failure says so.
    - sentiment_unscored: by how many articles were left unscored (>=50% high,
      >=10% medium, else low), since a stray timeout among hundreds of calls is
      normal but a blind Sentiment agent is not.
    - llm_request_failed: the per-call sentiment failures are low (the
      sentiment_unscored event carries their weight); a failed filing summary is
      medium, because each one is a whole document the Stock Researcher lacks.
    - not_covered: low ONLY for an FMP endpoint that has a fallback after it in
      US_CHAINS, medium for everything else. Observed 2026-09-29: FMP answers 402
      for the benchmark and peer quotes on every US run (AAPL x5, KO x3) and for
      every endpoint on a mid-cap (RDDT: 5 endpoints), and each of those has a
      yfinance (or finnhub) link after it, so the data still arrived; medium would
      put 1 to 5 rows of expected noise on every US run. Low stays counted in the
      run summary and shows with --all. Finnhub is the ONLY source for US news,
      peers and recommendation trends, and `ratios-ttm` is FMP-only, so a decline
      there really loses the data: medium, as is any endpoint not listed below.
    - fetch_failed / data_missing (failures below the Router): medium. Each means a
      call errored or came back unusable and the provider swallowed it.
    - Anything not listed defaults to medium: visible until someone decides.
    """
    if kind == AUTH_FAILED:
        return "high"
    if kind == EMPTY_AFTER_FAILURE:
        return "medium"
    if kind == LINK_FAILED:
        return "low"
    if kind == SENTIMENT_UNSCORED:
        try:
            ratio = int(context["unscored"]) / int(context["total"])  # type: ignore[index]
        except (TypeError, KeyError, ValueError, ZeroDivisionError):
            ratio = 1.0
        return "high" if ratio >= 0.5 else "medium" if ratio >= 0.1 else "low"
    if kind == LLM_REQUEST_FAILED:
        return "low" if op == "sentiment_score" else "medium"
    if kind == NOT_COVERED:
        return "low" if provider == "fmp" and op in _FMP_ENDPOINTS_WITH_FALLBACK else "medium"
    return "medium"


class DegradationCollector:
    """An in-memory, deduplicating list of events. Never raises. Unbounded on purpose:
    events are keyed by (provider, op, kind), all fixed strings in the code, so the
    number of distinct keys in one run is small."""

    def __init__(self) -> None:
        self._events: dict[tuple[str, str, str], DegradationEvent] = {}

    def record(
        self,
        provider: str,
        op: str,
        kind: str,
        message: object = "",
        *,
        exc: BaseException | None = None,
        context: dict | None = None,
    ) -> None:
        try:
            severity = classify(provider, op, kind, context)
            key = (provider, op, kind)
            existing = self._events.get(key)
            if existing is not None:
                existing.count += 1
                return
            self._events[key] = DegradationEvent(
                provider=provider,
                op=op,
                kind=kind,
                severity=severity,
                message=str(message)[:_MESSAGE_LIMIT],
                exc_type=type(exc).__name__ if exc is not None else None,
                context=dict(context or {}),
            )
        except Exception:
            pass

    def drain(self) -> list[DegradationEvent]:
        """Return every event recorded so far and start over."""
        try:
            events = list(self._events.values())
            self._events = {}
            return events
        except Exception:
            return []


_current: contextvars.ContextVar[DegradationCollector | None] = contextvars.ContextVar(
    "degradation_collector", default=None
)


def current_collector() -> DegradationCollector | None:
    return _current.get()


def set_collector(collector: DegradationCollector | None) -> contextvars.Token:
    return _current.set(collector)


def reset_collector(token: contextvars.Token) -> None:
    try:
        _current.reset(token)
    except Exception:
        pass


def report(
    provider: str,
    op: str,
    kind: str,
    message: object = "",
    *,
    exc: BaseException | None = None,
    context: dict | None = None,
) -> None:
    """Record an event on the current run's collector, if there is one.

    A no-op outside an orchestrated run. Never raises, and never influences the
    caller: hook points call it AFTER they have already decided what to do.
    """
    try:
        collector = _current.get()
        if collector is not None:
            collector.record(provider, op, kind, message, exc=exc, context=context)
    except Exception:
        pass
