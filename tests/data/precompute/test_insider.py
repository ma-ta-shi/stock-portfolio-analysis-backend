from datetime import UTC, datetime, timedelta

from data.precompute.insider import summarize_insider_activity

RECENT = (datetime.now(UTC) - timedelta(days=10)).date().isoformat()
OLD = (datetime.now(UTC) - timedelta(days=200)).date().isoformat()


def _row(kind="sale", value=1_000_000.0, date=RECENT, issuer=False):
    return {"date": date, "is_issuer": issuer, "transaction_type": kind, "shares": 10, "value": value}


def test_the_threshold_is_the_larger_of_a_thousandth_of_a_percent_of_market_cap_and_25k():
    # $28.3B: 0.001% = $283K. $3.8T: $38.2M. A $1B company: 0.001% = $10K, so the $25K floor wins.
    assert summarize_insider_activity([_row(value=1)], 28.3e9)["threshold"] == 283_000
    assert round(summarize_insider_activity([_row(value=1)], 3.82e12)["threshold"]) == 38_200_000
    assert summarize_insider_activity([_row(value=1)], 1e9)["threshold"] == 25_000


def test_real_2026_10_02_activity_lands_where_the_design_intends():
    # MSFT: six sales worth $20.8M in all, the largest $5.2M, against a $38.2M threshold: nothing notable.
    msft = summarize_insider_activity([_row(value=v) for v in (5.18e6, 4.89e6, 4.42e6, 2.0e6, 2.0e6, 2.28e6)], 3.82e12)
    assert msft["materiality"] == "routine" and msft["notable_count"] == 0
    # RDDT: $2-4M sales against a $283K threshold are notable.
    rddt = summarize_insider_activity([_row(value=v) for v in (3.95e6, 2.49e6, 2.25e6, 100_000)], 28.3e9)
    assert rddt["materiality"] == "notable" and rddt["notable_count"] == 3 and rddt["sell_count"] == 4
    assert rddt["direction"] == "selling"


def test_canadian_names_use_the_same_rule_in_their_own_currency():
    # RY.TO (C$385B, threshold C$3.86M): a C$4.86M sale is notable, a C$1.05M one is not.
    out = summarize_insider_activity([_row(value=4_864_067), _row(value=1_047_677)], 3.856e11)
    assert out["materiality"] == "notable" and out["notable_count"] == 1 and out["sell_count"] == 2


def test_the_text_states_what_notable_means_and_the_net_dollars():
    """The old wording ended '(notable means $2.7M or more, 0.001% of market cap)' and a Sentiment run read it as
    'the $71.3M sold is only 0.001% of market cap' (SHOP.TO 2026-10-03): the threshold is no longer written as a ratio."""
    out = summarize_insider_activity([_row(value=2_000_000)] * 3 + [_row(value=1_000)], 28.3e9)
    assert out["text"] == ("net selling $6.0M from 3 notable of 4 transactions "
                           "over 90 days (a transaction of $283K or more counts as notable)")
    assert "market cap" not in out["text"]


def test_notable_buying_nets_against_notable_selling_by_dollars_not_counts():
    rows = [_row("purchase", 400_000), _row("purchase", 400_000), _row("sale", 5_000_000)]
    out = summarize_insider_activity(rows, 1e10)
    assert out["direction"] == "selling" and out["buy_count"] == 2 and out["notable_count"] == 3


def test_issuer_buybacks_stale_and_non_directional_rows_are_excluded():
    rows = [_row(issuer=True), _row(date=OLD), _row("exercise"), _row("gift")]
    out = summarize_insider_activity(rows, 1e10)
    assert out["materiality"] == "none" and "no qualifying insider purchases or sales" in out["text"]


def test_unknown_market_cap_is_unknown_not_a_guess():
    out = summarize_insider_activity([_row(value=74_000_000)], None)
    assert out["materiality"] is None and out["threshold"] is None and "market cap unknown" in out["text"]


def test_an_undated_aggregate_is_not_reported_as_no_activity():
    """ENB.TO 2026-10-02: ten undated quarterly aggregate rows (openbb-tmx fallback). Saying 'none' would be false."""
    agg = {"date": None, "is_issuer": False, "transaction_type": "sale", "shares": 1000, "value": None}
    out = summarize_insider_activity([agg], 1.45e11)
    assert out["materiality"] is None and "undated aggregate" in out["text"]


def test_rows_without_a_value_are_counted_but_never_notable():
    out = summarize_insider_activity([_row(value=None)], 1e10)
    assert out["sell_count"] == 1 and out["materiality"] == "routine"


def test_values_in_a_different_currency_than_the_market_cap_are_not_sized():
    """If the Bank of Canada rate is unavailable the Canadian values stay USD while the cap is CAD: comparing them
    would be off by the exchange rate, so nothing is judged."""
    out = summarize_insider_activity([_row(value=5_000_000)], 2.7e11, currency="USD", market_cap_currency="CAD")
    assert out["materiality"] is None and "values in USD but market cap in CAD" in out["text"]
    ok = summarize_insider_activity([_row(value=5_000_000)], 2.7e11, currency="CAD", market_cap_currency="CAD")
    assert ok["materiality"] in ("routine", "notable") and ok["text"].count("CAD") >= 1


def test_the_currency_code_follows_every_dollar_figure():
    out = summarize_insider_activity([_row(value=2_000_000)] * 3, 28.3e9, currency="CAD", market_cap_currency="CAD")
    assert out["text"].startswith("net selling $6.0M CAD from 3 notable") and "$283K CAD or more" in out["text"]
