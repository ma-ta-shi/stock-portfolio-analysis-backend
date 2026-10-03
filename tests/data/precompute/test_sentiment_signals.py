"""Code-computed analyst changes and short interest read (data/precompute/sentiment_signals.py)."""

from datetime import date

from data.precompute.sentiment_signals import summarize_analyst_changes, summarize_short_interest

AS_OF = date(2026, 10, 3)


def _row(day, action="main", pta="Maintains", firm="F"):
    return {"date": day, "firm": firm, "action": action, "price_target_action": pta}


def test_counts_events_in_the_window_and_target_moves():
    rows = [_row("2026-09-30", "up", "Raises"), _row("2026-09-20", "down", "Lowers"), _row("2026-09-10", "init", "Announces"),
            _row("2026-09-01", "main", "Raises"), _row("2026-08-01", "main", "Maintains")]
    out = summarize_analyst_changes(rows, as_of=AS_OF)
    assert (out["upgrades"], out["downgrades"], out["initiations"]) == (1, 1, 1)
    assert (out["target_raises"], out["target_cuts"]) == (2, 1)
    assert "1 upgrade, 1 downgrade, 1 new initiation in the last 90d" in out["text"] and "2 raised, 1 cut" in out["text"]


def test_rows_older_than_90_days_are_not_counted():
    out = summarize_analyst_changes([_row("2026-06-01", "up", "Raises")], as_of=AS_OF)
    assert out["upgrades"] == 0 and out["target_raises"] == 0
    assert out["text"] == "none in the last 90d (latest 2026-06-01)"


def test_a_quiet_canadian_name_says_so_and_gives_the_latest_date():
    """TD.TO 2026-10-03: 48 rows, the newest on 2026-06-01, so nothing in the last 90 days."""
    out = summarize_analyst_changes([_row("2026-06-01", "main", "Raises")], as_of=AS_OF, target_mean=183.4, price=168.07)
    assert "none in the last 90d (latest 2026-06-01)" in out["text"]
    assert out["price_target_vs_current_pct"] == 9.1 and "+9.1% against the price" in out["text"]


def test_no_rows_is_stated_not_hidden():
    out = summarize_analyst_changes([], as_of=AS_OF)
    assert out["text"] == "no data" and out["upgrades"] is None
    assert summarize_analyst_changes(None, as_of=AS_OF, target_mean=110.0, price=100.0)["text"].endswith("+10.0% against the price")


def test_a_bad_row_date_is_skipped_not_fatal():
    out = summarize_analyst_changes([{"date": "garbage", "action": "up"}, _row("2026-09-30", "up")], as_of=AS_OF)
    assert out["upgrades"] == 1


def test_target_against_price_needs_both_positive():
    for target, price in ((None, 100.0), (110.0, None), (0, 100.0), (110.0, 0)):
        assert summarize_analyst_changes([], as_of=AS_OF, target_mean=target, price=price)["price_target_vs_current_pct"] is None


def _si(pct=1.79, dtc=8.79, shares=28_890_182, prior=27_719_808, as_of="2026-09-15"):
    return {"short_interest_pct": pct, "days_to_cover": dtc, "shares_short": shares, "shares_short_prior_month": prior,
            "as_of_date": as_of}


def test_short_interest_td_to_is_normal_and_rising_slightly():
    """TD.TO: 1.79% of float; 28.89M shares against 27.72M a month earlier is +4.2%, inside the stable band."""
    out = summarize_short_interest(_si())
    assert out["interpretation"] == "normal" and out["trend"] == "stable" and out["change_pct"] == 4.2
    assert "1.79% of float, 8.79 days to cover, as of 2026-09-15; shares short 28,890,182 against 27,719,808" in out["text"]


def test_short_interest_trend_labels():
    assert summarize_short_interest(_si(shares=110, prior=100))["trend"] == "increasing"
    assert summarize_short_interest(_si(shares=90, prior=100))["trend"] == "decreasing"
    assert summarize_short_interest(_si(shares=104, prior=100))["trend"] == "stable"
    assert summarize_short_interest(_si(prior=None))["trend"] == "unknown"


def test_short_interest_is_elevated_at_the_threshold():
    assert summarize_short_interest(_si(pct=10.0))["interpretation"] == "elevated_volatility_risk"
    assert summarize_short_interest(_si(pct=9.99))["interpretation"] == "normal"


def test_no_short_interest_is_insufficient_data():
    for si in (None, {}):
        out = summarize_short_interest(si)
        assert out["interpretation"] == "insufficient_data" and out["trend"] == "unknown" and out["text"] == "no short interest data"
    assert summarize_short_interest({"short_interest_pct": None, "days_to_cover": 3.0})["interpretation"] == "insufficient_data"
