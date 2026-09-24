"""Shared validator-domain constants and the declared-enum sweep.

Production port of `simulation/validators/common.py` (86bbuhjup) -- pure copy, no
import-path changes needed (no internal simulation.* imports). Consolidated there
from pass1.py/pass2.py, which each carried a byte-identical copy of
`DECLARED_ENUMS`/`_sweep_declared_enums`/`THESIS_ARCHETYPES` -- the exact duplication
shape that already caused a real, silent bug this project (Risk Advisor's
`researcher_thesis_archetype` reading a wrong key, unnoticed until live-tested): a
dict that exists in two places is how one of the two silently falls behind when it's
extended.

Lives in `agents/validators/` rather than `agents/utils.py` -- these are
validator-domain concepts (declared enum vocabularies), not general shared plumbing.
"""

THESIS_ARCHETYPES = {
    "secular_grower",
    "dividend_compounder",
    "cyclical_recovery",
    "quality_compounder",
    "value_trap_candidate",
}

# Enums the PROMPTS declare that are safe to check by leaf key name alone, at any nesting depth
# -- confirmed CONTEXT-INDEPENDENT (same vocabulary everywhere the key appears, across all 13
# real templates) and confirmed LIVE-SAFE (real gpt-oss:20b output matched the declared
# vocabulary across real fixture-driven calls, not just static prompt text -- several
# candidates that looked safe from static extraction alone were found live-noncompliant and are
# deliberately NOT here; see the 86bbuhk82 plan for the full record of what was tried and why
# each entry below passed).
#
# NOT safe to add here: field names that mean different things in different structural
# positions (`sentiment`, `strength`, `direction`, `timing` all vary by agent/context) -- those
# get path-scoped explicit checks instead, following cio.py's own
# `{parent: {field: {values}}}` pattern.
DECLARED_ENUMS = {
    # analysis_confidence is FOUR values, not three -- "insufficient" is declared identically
    # across all five Pass 1 prompts (verified directly against the real templates) but was
    # missing from this set before this consolidation, silently letting a compliant
    # "insufficient" value fail if anything ever checked it via the sweep.
    "analysis_confidence": {"high", "medium", "low", "insufficient"},
    "importance": {"high", "medium", "low"},
    "likelihood": {"high", "medium", "low"},
    "probability": {"high", "medium", "low"},
    "severity": {"high", "medium", "low"},
    # Stock Researcher's recent_developments[].significance -- same 3-value vocabulary,
    # confirmed no conflicting usage elsewhere. Surfaced by the permanent enum-inventory test.
    "significance": {"high", "medium", "low"},
    "tail_risk_level": {"elevated", "moderate", "negligible"},
    # Technical Analyst -- live-verified (2 real samples each, gpt-oss:20b, real fixtures).
    "nearest_level_bias": {"near_support", "near_resistance", "midrange"},
    "pattern_signal": {"bullish", "bearish", "neutral", "none"},
    "momentum_divergence": {"bullish", "bearish", "none"},
    "volume_confirmation": {"confirming", "diverging", "inconclusive"},
    # Macro Economist -- live-verified (impact_on_stock exercised 4x in one real call).
    "outlook": {"accelerating", "stable", "decelerating", "recessionary"},
    "impact_on_stock": {"positive", "neutral", "negative"},
    # Sentiment Analyst -- live-verified.
    "consensus_trend": {"improving", "stable", "deteriorating"},
    "sentiment_trend": {"improving", "stable", "deteriorating"},
    # CIO -- live-verified. Nested identically under bull_case_assessment AND
    # bear_case_assessment with the same vocabulary in both, so not polysemous despite
    # appearing in two structural positions -- exactly the case this sweep exists for.
    "verdict": {"accepted", "partially_accepted", "rejected"},
    "weight_applied": {"high", "medium", "low"},
}

# `momentum_direction` (Technical Analyst) was tried and REJECTED: live-testing found real,
# non-compliant model output ('stabilizing', not in the prompt's declared
# improving|flat|deteriorating), not just a theoretical risk. Deliberately absent from
# DECLARED_ENUMS above pending a prompt-side fix -- see the 86bbuhk82 plan.


def _sweep_declared_enums(obj, errors: list, path: str = "") -> None:
    """Validate every leaf whose key names a declared enum, at any depth."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            here = f"{path}.{k}" if path else k
            allowed = DECLARED_ENUMS.get(k)
            if allowed is not None and isinstance(v, str) and v not in allowed:
                errors.append(
                    f"{here}: must be one of {'|'.join(sorted(allowed))}, got '{v}'"
                )
            _sweep_declared_enums(v, errors, here)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _sweep_declared_enums(v, errors, f"{path}[{i}]")
