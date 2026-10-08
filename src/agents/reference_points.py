"""Numbers the Bull and Bear advocates used to invent, written by code instead.

`asymmetry_assessment` asked the model for "what does the bull case look like if it works, and what is the cost if it does not" and gave it no
numbers. In 8 real runs the answers were boilerplate ("a 5-7% decline" word for word across three tickers, "10-15% upside" across four), a price
target above the analyst target already in the payload, or a ratio ("3:1") with no source. Telling the model to quote only given figures did not
stop it (3 inputs: it still wrote "a 10% pullback", "a 30% gain" against a +26.9% target, "a 5% dividend increase"), and asking for no figures at
all was obeyed in 4 of 8 answers. So the field is now built here from the data (docs/technical/pass2-output-contract.md).

Every figure is quoted from the source that computed it and never recomputed from another source's price: the payload carries a live quote and
a last-bar close that can differ by a few percent, and a support distance recomputed from the quote can land on the wrong side of the price.
"""
from __future__ import annotations

from data.precompute.sentiment_signals import analyst_changes_for_bundle


def _num(value: float, signed: bool = False) -> str:
    """Two decimals at most, trailing zeros dropped (0.46, 4.4, 173.77)."""
    text = f"{value:+.2f}" if signed else f"{value:.2f}"
    return text.rstrip("0").rstrip(".")


def _pct(value: float, signed: bool = False) -> str:
    return _num(value, signed) + "%"


def asymmetry_reference(bundle) -> str:
    """One factual line: the upside and downside reference points the data gives, each naming its source. Missing inputs are left out."""
    upside, downside = [], []

    consensus = getattr(bundle, "analyst_consensus", None) or {}
    target = consensus.get("target_mean")
    target_pct = None
    if target:
        try:
            target_pct = analyst_changes_for_bundle(bundle).get("price_target_vs_current_pct")
        except (AttributeError, TypeError, ValueError):
            pass  # this line is written after the model has answered: a missing input must leave the figure out, never fail the agent
    if target and target_pct is not None:
        upside.append(f"consensus target {_num(target)} ({_pct(target_pct, signed=True)} against the price, SENT)")

    # Technical stores each distance as (price - level) / price, so a resistance above the price is NEGATIVE and a support below it positive;
    # the words above and below carry the direction, so only the size is quoted.
    sr = getattr(bundle, "support_resistance", None) or {}
    if sr.get("nearest_resistance") is not None and sr.get("pct_to_resistance") is not None:
        upside.append(f"nearest resistance {_num(sr['nearest_resistance'])} ({_pct(abs(sr['pct_to_resistance']))} above, TECH, from the last close)")
    if sr.get("nearest_support") is not None and sr.get("pct_to_support") is not None:
        downside.append(f"nearest support {_num(sr['nearest_support'])} ({_pct(abs(sr['pct_to_support']))} below, TECH, from the last close)")

    rm = getattr(bundle, "risk_metrics", None) or {}
    if rm.get("max_drawdown_1yr_pct") is not None:
        downside.append(f"1-year maximum drawdown {_pct(rm['max_drawdown_1yr_pct'])} (DD)")
    if rm.get("annualized_vol_pct") is not None:
        downside.append(f"a typical 12-month move of {_pct(rm['annualized_vol_pct'])} either way (VOL)")

    parts = []
    if upside:
        parts.append("Upside reference: " + "; ".join(upside) + ".")
    if downside:
        parts.append("Downside reference: " + "; ".join(downside) + ".")
    return " ".join(parts) if parts else "Upside and downside reference points are not available in the data."
