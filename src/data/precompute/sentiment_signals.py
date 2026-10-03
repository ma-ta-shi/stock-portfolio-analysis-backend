"""Code-computed Sentiment inputs: analyst rating changes and the short interest read.

The Sentiment Analyst used to be asked for both (an upgrade and downgrade count, a consensus trend, a short interest
trend and a three-value interpretation) with nothing to compute them from, and the model's own answer for the short
interest value failed the validator in 27 of 48 first attempts ("normal volatility risk" instead of "normal"). The
inputs decide both, so they are decided here and the model is only shown the result.

Pure functions, no network: the provider fetches, these summarise. Same pattern as `insider.summarize_insider_activity`.
"""

from datetime import date, datetime, timedelta

ANALYST_WINDOW_DAYS = 90
# Short interest at or above this share of the float is "elevated". One threshold, the one the Sentiment runner's own
# data-warning check already used; days to cover is shown but does not change the read (a retail position has no
# squeeze exposure, see the design doc, decision 13).
ELEVATED_SHORT_INTEREST_PCT = 10.0
# A month-on-month change within this fraction is "stable".
SHORT_INTEREST_STABLE_BAND = 0.05


def _day(value) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _count(n: int, singular: str, plural: str | None = None) -> str:
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def summarize_analyst_changes(
    rows: list[dict] | None,
    *,
    as_of: date,
    target_mean: float | None = None,
    price: float | None = None,
    currency: str | None = None,
) -> dict:
    """Rating changes in the last ANALYST_WINDOW_DAYS and where the average target sits against the price.

    `rows` are the provider's normalised rating changes ({date, firm, action, price_target_action, ...}); `action` is
    Yahoo's "up", "down", "init", "main" (maintained) or "reit". Counts are of events, so one firm that upgrades and
    raises its target is one upgrade and one target raise. A name with no change in the window says so and gives the
    date of the latest one, because for Canadian names that is often months back and silence is itself the answer.
    Always returns a dict with a `text` (never None): "no data" when the provider gave nothing."""
    target_pct = None
    if isinstance(target_mean, (int, float)) and isinstance(price, (int, float)) and target_mean > 0 and price > 0:
        target_pct = round((target_mean / price - 1) * 100, 1)

    dated = [(d, r) for r in rows or [] if (d := _day(r.get("date"))) is not None]
    if not dated:
        text = "no data"
        if target_pct is not None:
            text += f"; average target {target_pct:+.1f}% against the price"
        return {"upgrades": None, "downgrades": None, "initiations": None, "target_raises": None, "target_cuts": None,
                "latest_change": None, "price_target_vs_current_pct": target_pct, "text": text}

    cutoff = as_of - timedelta(days=ANALYST_WINDOW_DAYS)
    window = [r for d, r in dated if cutoff <= d <= as_of]
    count = lambda key, value: sum(1 for r in window if r.get(key) == value)  # noqa: E731
    upgrades, downgrades, initiations = count("action", "up"), count("action", "down"), count("action", "init")
    raises, cuts = count("price_target_action", "Raises"), count("price_target_action", "Lowers")
    latest = max(d for d, _ in dated)

    if window:
        text = (
            f"{_count(upgrades, 'upgrade')}, {_count(downgrades, 'downgrade')}, {_count(initiations, 'new initiation')} "
            f"in the last {ANALYST_WINDOW_DAYS}d; price targets {raises} raised, {cuts} cut ({len(window)} firm updates)"
        )
    else:
        text = f"none in the last {ANALYST_WINDOW_DAYS}d (latest {latest.isoformat()})"
    if target_pct is not None:
        text += f"; average target {target_pct:+.1f}% against the price"
    return {"upgrades": upgrades, "downgrades": downgrades, "initiations": initiations, "target_raises": raises,
            "target_cuts": cuts, "latest_change": latest.isoformat(), "price_target_vs_current_pct": target_pct,
            "text": text}


def analyst_changes_for_bundle(bundle) -> dict:
    """`summarize_analyst_changes` fed from a DataBundle, so the Sentiment payload and the Pass 2 view cannot drift."""
    return summarize_analyst_changes(
        getattr(bundle, "analyst_rating_changes", None),
        as_of=bundle.data_vintage.date(),
        target_mean=bundle.analyst_consensus.get("target_mean"),
        price=(bundle.price_info or {}).get("current_price"),
    )


def summarize_short_interest(short_interest: dict | None) -> dict:
    """Short interest trend and read from the bundle's own figures.

    `trend` is the month-on-month change in shares short ("increasing", "stable", "decreasing" or "unknown");
    `interpretation` is "elevated_volatility_risk", "normal" or "insufficient_data" (no figure at all). `text` is the
    one-line form the agent is shown. Both labels use the same vocabulary the Sentiment output schema used to ask the
    model for."""
    si = short_interest or {}
    pct, dtc = si.get("short_interest_pct"), si.get("days_to_cover")
    shares, prior = si.get("shares_short"), si.get("shares_short_prior_month")

    change_pct = None
    if shares is not None and prior:
        change_pct = round((shares / prior - 1) * 100, 1)
    if change_pct is None:
        trend = "unknown"
    elif change_pct > SHORT_INTEREST_STABLE_BAND * 100:
        trend = "increasing"
    elif change_pct < -SHORT_INTEREST_STABLE_BAND * 100:
        trend = "decreasing"
    else:
        trend = "stable"

    if pct is None:
        interpretation = "insufficient_data"
    elif pct >= ELEVATED_SHORT_INTEREST_PCT:
        interpretation = "elevated_volatility_risk"
    else:
        interpretation = "normal"

    if not si or (pct is None and dtc is None):
        text = "no short interest data"
    else:
        parts = [f"{pct}% of float" if pct is not None else "% of float unknown"]
        if dtc is not None:
            parts.append(f"{dtc} days to cover")
        text = ", ".join(parts)
        if si.get("as_of_date"):
            text += f", as of {si['as_of_date']}"
        if change_pct is not None:
            text += f"; shares short {shares:,} against {prior:,} a month earlier ({change_pct:+.1f}%)"
    return {"trend": trend, "interpretation": interpretation, "change_pct": change_pct, "text": text}
