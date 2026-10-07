"""Deterministic grounding check for Pass 1 text: every figure an agent writes must be in the data it was given.

Until 2026-10-07 nothing enforced this: the Pass 1 validators checked only the shape of the output, and grounding rested on the prompt
asking for cited evidence. Run over 65 real runs (1,113 figures in narratives, summaries and caveats, the system prompt counted as context) it flags 6 (0.5%): one is a
genuinely wrong figure (a Technical narrative wrote "0.07% above support" where the payload said 0.46%) and the rest are percentages the
model derived itself ("a 54% discount"), which the prompts forbid. Corrupting real figures by 35 to 120%, it catches 75 to 85% (the rest
coincide with some payload number: a payload holds hundreds), so it is a backstop, not proof of grounding.
`BaseRunner.GROUND_MODE` decides what a flag does: "once" (Pass 1) fails the first attempt that reaches it, naming the figure, then only
records; "log" (Pass 2, CIO) only records. See docs/technical/agent-grounding-best-practices.md.

A figure is grounded if it appears in the payload as written, or matches a payload number within 0.6% or its own rounding (one unit of
the last digit below the data, half a unit above it: models truncate as well as round) after
the unit conversions models actually make: percent against fraction (24% for 0.2407), thousands, millions and billions ("$9.4bn" for
9,400,000,000; "3.52M" for 3,520,000). Years, counts under 100 and indicator windows ("200-day") are not figures to check.
"""
import re

_SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mm": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9, "t": 1e12, "trillion": 1e12}
_NUM = re.compile(
    r"(?<![\w.])(?P<n>\d[\d,]*(?:\.\d+)?)(?:[\s  ]*(?P<u>%|trillion|billion|million|thousand|bn|mm|[kKmMbBtT])(?![A-Za-z]))?",
    re.I,
)
_RANGE_GAP = re.compile(r"^[\s$]*(?:[-–—]|to)[\s$]*$")  # "1.2-1.5bn": the first end shares the unit of the last
_WINDOW = re.compile(r"^[\s ‑-]*(day|days|week|weeks|month|months|year|years|yr|yrs|session|sessions|d\b|w\b)", re.I)
MAX_NAMED = 3
FLAG_MARK = "is not in the data you were given"  # in every error grounding_errors() writes


def _scaled(match: re.Match) -> tuple[float, str, int] | None:
    """(value in base units, unit, decimals written) for one regex match."""
    raw = match.group("n").replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    unit = (match.group("u") or "").lower()
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    return value * _SCALE.get(unit, 1.0), unit, decimals


def figures(text: str) -> list[tuple[str, float, float]]:
    """(as written, value, rounding allowance) for each figure worth checking in `text`. The allowance is one unit of the last digit
    the writer gave, in the figure's own unit ("24%" is up to 1 point below the data, "3.52M" up to 10,000 below); `_is_grounded` allows
    only half of it above, since models both round and truncate ("22.7" for 22.76, "39%" for 39.56%) but truncation only goes down."""
    found = []
    matches = list(_NUM.finditer(text))
    for i, m in enumerate(matches):
        parsed = _scaled(m)
        if parsed is None:
            continue
        value, unit, decimals = parsed
        nxt = matches[i + 1] if i + 1 < len(matches) else None
        if not unit and nxt and nxt.group("u") and _RANGE_GAP.match(text[m.end(): nxt.start()]):
            unit = nxt.group("u").lower()  # a range written with one unit: "$1.2-1.5bn", "10-15%"
            value *= _SCALE.get(unit, 1.0)
        if not unit and not decimals:
            if 1900 <= value <= 2100:
                continue  # a year
            if value < 100 or _WINDOW.match(text[m.end(): m.end() + 14]):
                continue  # a count, a rank or an indicator window
        rounding = 10 ** (-decimals) * _SCALE.get(unit, 1.0)
        found.append((m.group(0).strip(), value, rounding))
    return found


def _payload_values(payload: str) -> list[float]:
    return [parsed[0] for m in _NUM.finditer(payload) if (parsed := _scaled(m)) is not None]


def _factors(written: str) -> tuple[float, ...]:
    """The conversions a figure may have gone through, by what it is written as. Allowing every conversion for every figure made the
    check lenient: on 1,100 real figures corrupted by 35 to 120%, only 60 to 86% were caught, the rest matched some payload number by
    coincidence (hundreds of numbers times seven factors)."""
    w = written.strip()
    if w.endswith("%"):
        return (1.0, 100.0)  # a percent against a payload percent or fraction (24% for 0.2407)
    if w[-1:].isalpha():
        return (1.0,)  # a scale unit: payload numbers carrying their own unit were already scaled the same way
    return (1.0, 0.01, 100.0)  # a plain number may be a ratio shown as a percent or the reverse


def _is_grounded(written: str, value: float, rounding: float, values: list[float], payload: str) -> bool:
    if written.replace(" ", " ") in payload:
        return True
    # Written below the data by up to one unit of the last digit is a truncation ("22.7" for 22.76); above it, only a rounding up
    # (half a unit). Allowing a unit on both sides raised the false "grounded" rate: more corrupted figures matched some number.
    below, above = max(0.006 * abs(value), rounding), max(0.006 * abs(value), rounding / 2)
    return any(p and -below <= value - p * factor <= above for p in values for factor in _factors(written))


def ungrounded_figures(text: str, payload: str) -> list[str]:
    values = _payload_values(payload)
    return [written for written, value, rounding in figures(text) if not _is_grounded(written, value, rounding, values, payload)]


def grounding_errors(output: dict, payload: str) -> list[str]:
    """Validator errors for the figures in an agent's narrative, summary and caveats that are not in `payload`."""
    if not isinstance(output, dict) or not payload:
        return []
    caveats = output.get("caveats")
    texts = {
        "narrative": output.get("narrative"),
        "assessment_summary": output.get("assessment_summary"),
        "thesis_summary": output.get("thesis_summary"),  # the Pass 2 and CIO equivalent of the summary
        "caveats": " ".join(str(c) for c in caveats) if isinstance(caveats, list) else None,
    }
    errors = []
    for field, text in texts.items():
        if not isinstance(text, str):
            continue
        for written in ungrounded_figures(text, payload)[:MAX_NAMED]:
            errors.append(f"{field}: the figure {written!r} {FLAG_MARK}; state only figures from the data")
    return errors[:MAX_NAMED]
