"""One presentation currency per ticker (data/precompute/currency.py)."""
import pytest

from data.precompute.currency import (
    PRICE_BASED_MULTIPLES,
    convert_financials,
    convert_insider_values,
    fx_factor,
    to_quote_currency,
)
from data.providers.base import NormalizedFinancials

USD_CAD = 1.4


def _fin(currency="USD") -> NormalizedFinancials:
    quarter = {"period_end": "2026-06-30", "revenue": 1000.0, "net_income": 100.0, "eps": 2.0, "operating_income": 150.0,
               "shares_outstanding": 50.0, "dividends_paid": None}
    return NormalizedFinancials(
        quarters=[dict(quarter)], annual=[dict(quarter)],
        balance_sheet={"total_debt": 400.0, "total_equity": 800.0, "cash_and_equivalents": 50.0},
        currency=currency, ttm={"revenue": 4000.0, "net_income": 400.0},
    )


def test_fx_factor_directions_and_missing_rate():
    assert fx_factor("USD", "USD", None) == 1.0
    assert fx_factor("USD", "CAD", USD_CAD) == USD_CAD
    assert fx_factor("CAD", "USD", USD_CAD) == pytest.approx(1 / USD_CAD)
    assert fx_factor("USD", "CAD", None) is None and fx_factor("USD", "CAD", 0) is None
    assert fx_factor("EUR", "CAD", USD_CAD) is None  # only USD and CAD are converted


def test_conversion_scales_money_and_leaves_counts_and_ratios_alone():
    out = convert_financials(_fin(), USD_CAD, "CAD")
    q = out.quarters[0]
    assert (q["revenue"], q["net_income"], q["eps"]) == pytest.approx((1400.0, 140.0, 2.8))
    assert q["shares_outstanding"] == 50.0 and q["period_end"] == "2026-06-30" and q["dividends_paid"] is None
    assert out.balance_sheet["total_debt"] == pytest.approx(560.0)
    assert out.ttm["revenue"] == pytest.approx(5600.0) and out.currency == "CAD"
    # the input is not modified
    assert _fin().quarters[0]["revenue"] == 1000.0
    # growth and margins (ratios) are the same before and after
    assert q["net_income"] / q["revenue"] == pytest.approx(0.1)


def test_a_matching_currency_is_untouched_and_has_no_mismatch_record():
    fin, info = to_quote_currency(_fin("CAD"), "CAD", USD_CAD)
    assert info is None and fin.quarters[0]["revenue"] == 1000.0


def test_a_usd_reporter_on_a_cad_quote_is_converted_and_recorded():
    fin, info = to_quote_currency(_fin("USD"), "CAD", USD_CAD)
    assert fin.currency == "CAD" and fin.quarters[0]["revenue"] == pytest.approx(1400.0)
    assert info == {"financials_currency": "USD", "quote_currency": "CAD", "converted": True, "usd_cad": USD_CAD}


def test_without_a_rate_or_for_an_unsupported_pair_nothing_is_converted_and_it_says_so():
    for ccy, rate in (("USD", None), ("EUR", USD_CAD)):
        fin, info = to_quote_currency(_fin(ccy), "CAD", rate)
        assert fin.currency == ccy and fin.quarters[0]["revenue"] == 1000.0
        assert info["converted"] is False and info["usd_cad"] is None


def test_insider_values_convert_from_usd_to_the_quote_currency():
    rows = [{"date": "2026-09-01", "value": 1000.0}, {"date": "2026-09-02", "value": None}]
    out, converted = convert_insider_values(rows, "CAD", USD_CAD)
    assert converted and out[0]["value"] == pytest.approx(1400.0) and out[1]["value"] is None
    assert rows[0]["value"] == 1000.0  # input untouched
    same, ok = convert_insider_values(rows, "USD", USD_CAD)
    assert ok and same is rows
    unchanged, ok = convert_insider_values(rows, "CAD", None)
    assert not ok and unchanged is rows


def test_price_based_multiples_list_covers_what_divides_a_price_by_a_statement_amount():
    assert {"pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda", "peg_ratio"} == set(PRICE_BASED_MULTIPLES)


def test_earnings_surprise_amounts_follow_the_statement_conversion_but_the_percent_does_not():
    """SHOP.TO 2026-10-02: eps 0.34 (USD) sat inside a CAD stock."""
    from data.precompute.currency import convert_earnings_surprises

    rows = [{"period_end": "2025-09-30", "eps_actual": 0.34, "eps_estimated": 0.34194, "eps_surprise_pct": -0.57,
             "revenue_actual": None, "revenue_estimated": 2.0e9}]
    mismatch = {"financials_currency": "USD", "quote_currency": "CAD", "converted": True, "usd_cad": USD_CAD}
    out = convert_earnings_surprises(rows, mismatch)
    assert out[0]["eps_actual"] == pytest.approx(0.476) and out[0]["eps_estimated"] == pytest.approx(0.34194 * USD_CAD)
    assert out[0]["revenue_estimated"] == pytest.approx(2.8e9) and out[0]["revenue_actual"] is None
    assert out[0]["eps_surprise_pct"] == -0.57 and rows[0]["eps_actual"] == 0.34  # input untouched
    assert convert_earnings_surprises(rows, None) is rows
    assert convert_earnings_surprises(rows, {**mismatch, "converted": False}) is rows
    assert convert_earnings_surprises(None, mismatch) is None
