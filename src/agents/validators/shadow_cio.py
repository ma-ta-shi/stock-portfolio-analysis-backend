"""Shadow CIO validator.

Nothing enforced any Shadow CIO rule before 2026-09-01 — there was no runner and no
validator, only a 714-line prompt. Its rules were therefore unenforceable by
construction, which is the §110.6 situation (Tax) repeated: a documented contract with
no mechanism behind it.

SHADOW IS NOT A SMALL CIO. It emits 10 fields to the CIO's 33, and two of its rules are
the OPPOSITE of the primary's:

  * cold-start cap — the CIO is capped at 65 when `cold_start_cap_active`; the shadow is
    explicitly EXEMPT and may exceed it. A validator that copied the CIO's rule would
    reject correct shadow output.
  * data-quality gate — the shadow caps confidence at 45 on low groundedness all round.
    The CIO has no such rule.

Comparative rules are deliberately NOT here. "Shadow lands one notch less bullish than
primary" cannot be evaluated from a shadow output alone; those belong to the paired
comparison, not to single-output validation.
"""
import json
import re

VALID_OUTLOOKS = {"bullish", "somewhat_bullish", "neutral", "somewhat_bearish", "bearish"}
# Same vocabulary as the CIO prompt (both prompts agree; the runner and the old CIO
# validator did not — see prompt-revision-protocol.md §128.3).
VALID_RETURN_TIERS = {"strong_outperform", "outperform", "market_perform",
                      "underperform", "strong_underperform"}
NEUTRAL_RANGE = {"neutral", "somewhat_bullish", "somewhat_bearish"}

REQUIRED = (
    "stock_outlook", "confidence", "expected_return_tier", "thesis_summary",
    "bear_weighting_application_note", "what_would_change_my_mind", "parallel_predictions",
)


def validate_shadow_cio(
    output: dict,
    disagreement_category: str = "consensus",
    low_groundedness: bool = False,
) -> tuple[bool, list[str]]:
    """Validate Shadow CIO output.

    `low_groundedness` is supplied by the caller (it depends on the Pass 2 payload, not
    on the shadow's own output), so the ≤45 gate is only checked when it is known.
    """
    errors: list[str] = []

    for f in REQUIRED:
        if f not in output:
            errors.append(f"Missing required field: {f}")

    # The shadow is a calibration probe, not an advisor. Same prohibition as the CIO.
    for forbidden in ("recommendation", "buy_sell_hold", "action"):
        if forbidden in output:
            errors.append(f"Shadow CIO: must NOT have a '{forbidden}' field")

    outlook = output.get("stock_outlook")
    if outlook is not None and outlook not in VALID_OUTLOOKS:
        errors.append(f"stock_outlook: must be one of {VALID_OUTLOOKS}, got '{outlook}'")

    tier = output.get("expected_return_tier")
    if tier is not None and tier not in VALID_RETURN_TIERS:
        errors.append(f"expected_return_tier: must be one of {VALID_RETURN_TIERS}, got '{tier}'")

    conf = output.get("confidence")
    if conf is not None:
        if not isinstance(conf, int) or not (0 <= conf <= 100):
            errors.append(f"confidence: must be an integer 0-100, got {conf!r}")
        else:
            # Disagreement caps: same thresholds as the primary. NOTE high_conflict is
            # currently unreachable through the real score (§131.1) — it can only be
            # exercised by injecting the category, which is why this takes a parameter.
            if disagreement_category == "split_decision" and conf > 65:
                errors.append(f"confidence: must be <=65 on split_decision, got {conf}")
            if low_groundedness and conf > 45:
                errors.append(
                    f"confidence: must be <=45 when groundedness is low all round, got {conf}"
                )

    if disagreement_category == "high_conflict" and outlook not in NEUTRAL_RANGE:
        errors.append(
            f"stock_outlook: must be in {NEUTRAL_RANGE} on high_conflict, got '{outlook}'"
        )

    ts = str(output.get("thesis_summary") or "")
    if ts and len(ts) < 100:
        errors.append(f"thesis_summary: too short ({len(ts)} chars, min 100)")
    elif ts and len(ts) > 300:
        # Phrased to match base.py's _auto_trim pattern -- "too long (N chars, max M)".
        # It exists precisely because "the model can control narrative length roughly
        # but not precisely", and the shadow overshot at 301/307/349: a 0.3-16% overshoot
        # it burns retries on and cannot reliably win. The previous message
        # ("must be 100-300 chars, got N") did not match the pattern, so the salvage
        # never fired. A prompt edit would not help -- this model cannot count characters.
        errors.append(f"thesis_summary: too long ({len(ts)} chars, max 300)")

    # The field that makes the shadow auditable: it must say whether the bear override
    # actually changed the read, not merely that it was applied.
    note = str(output.get("bear_weighting_application_note") or "").strip()
    if not note:
        errors.append("bear_weighting_application_note: must be non-empty")
    elif len(note) < 20:
        errors.append(
            f"bear_weighting_application_note: too short to state whether the override "
            f"changed the read (got {note!r})"
        )

    if not str(output.get("what_would_change_my_mind") or "").strip():
        errors.append("what_would_change_my_mind: must be non-empty")

    pp = output.get("parallel_predictions")
    if pp is not None:
        if not isinstance(pp, dict):
            errors.append("parallel_predictions: must be an object")
        else:
            for horizon in ("1_month", "3_month"):
                h = pp.get(horizon)
                if not isinstance(h, dict):
                    errors.append(f"parallel_predictions.{horizon}: missing or not an object")
                    continue
                if h.get("direction") not in VALID_OUTLOOKS:
                    errors.append(
                        f"parallel_predictions.{horizon}.direction: must be one of "
                        f"{VALID_OUTLOOKS}, got {h.get('direction')!r}"
                    )
                hc = h.get("confidence")
                if not isinstance(hc, int) or not (0 <= hc <= 100):
                    errors.append(
                        f"parallel_predictions.{horizon}.confidence: must be an integer 0-100"
                    )

    return len(errors) == 0, errors


_NUMBER_RE = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)(?![\w.])")


def _is_interesting(v: float) -> bool:
    """Small integers and plain percentages (confidence scores, common
    percentage figures) are ubiquitous in any output and prove nothing
    about groundedness either way. Ported verbatim from
    backend/_audit/_shadow_grounding.py's own `interesting()` (86bbt1kct)."""
    return abs(v) > 100 or v != int(v)


def _all_numbers(text: str) -> set[float]:
    """Every number in `text`, unfiltered. Ported verbatim from
    backend/_audit/_shadow_grounding.py's own `numbers()`."""
    out = set()
    for m in _NUMBER_RE.finditer(text):
        try:
            out.add(float(m.group(1)))
        except ValueError:
            pass
    return out


def _grounded(v: float, pool: set[float]) -> bool:
    """Present in the payload within rounding tolerance -- the E47 lesson:
    naive string matching scores a correctly-rounded `293.08` as
    ungrounded against a payload's `293.0752`. Ported verbatim from
    backend/_audit/_shadow_grounding.py's own `grounded()`."""
    for p in pool:
        if v == p:
            return True
        if p != 0 and abs(v - p) / abs(p) < 0.01:
            return True
        if abs(v - round(p, 2)) < 1e-9 or abs(v - round(p, 1)) < 1e-9:
            return True
    return False


def check_numeric_groundedness(
    output: dict, compressed_pass1: dict, pass2_outputs: dict
) -> list[str]:
    """Does every non-trivial number in the Shadow's output trace back to
    its real input payload? Informational only -- NEVER fails validation,
    same pattern as check_cold_start_exemption below (86bbt1kct).

    This answers the Shadow's groundedness question WITHOUT requiring
    citations at all (backend/_audit/_shadow_grounding.py's own framing:
    "citations are how E45 tested the CIO's groundedness; they are not
    what groundedness IS"). Built as decision-support for the deferred
    Pass 1 citation-enforcement question -- this project has already paid
    real cost for citation floors added blind (Tax's now-removed
    ≥2-distinct-ID rule, E88/E90; CIO's validator-prompt-mismatched
    `what_would_change_my_mind`, E93) -- run this against real output
    first, decide from the numbers it produces.
    """
    # Pool is EVERY number in the input, unfiltered -- a small int like a
    # confidence score can legitimately ground an output number (e.g. the
    # shadow citing "70" for a reason unrelated to confidence). Only the
    # OUTPUT side is filtered to "interesting" numbers, since that's what
    # we're deciding whether to trust.
    pool = _all_numbers(
        json.dumps({"compressed_pass1": compressed_pass1, "pass2_outputs": pass2_outputs}, default=str)
    )
    output_numbers = {v for v in _all_numbers(json.dumps(output, default=str)) if _is_interesting(v)}
    ungrounded = sorted(v for v in output_numbers if not _grounded(v, pool))
    if not ungrounded:
        return []
    return [
        f"(informational) numeric groundedness: {len(ungrounded)} of {len(output_numbers)} "
        f"non-trivial number(s) in shadow output do not trace to the input payload: {ungrounded}"
    ]


def check_cold_start_exemption(output: dict, cold_start_cap_active: bool) -> list[str]:
    """The shadow is EXEMPT from the cold-start cap — a warning helper, not a failure.

    Kept separate from `validate_shadow_cio` on purpose: exceeding 65 under an active
    cold-start cap is CORRECT shadow behaviour and must never be an error. This exists so
    the exemption can be asserted in tests without implying the opposite rule.
    """
    conf = output.get("confidence")
    if cold_start_cap_active and isinstance(conf, int) and conf > 65:
        return [f"(informational) shadow exercised its cold-start exemption: confidence={conf}"]
    return []
