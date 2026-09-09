import pytest

from data.precompute.fundamentals import (
    _PEER_METRIC_VALID_RANGE,
    _SECTOR_MEDIAN_KEYS,
    _valid_peer_value,
    compute_all,
    compute_balance_sheet_metrics,
    compute_dividend_info,
    compute_growth_metrics,
    compute_peer_comparison,
    compute_profitability_metrics,
    compute_valuation_metrics,
)
from data.providers.base import NormalizedDividendRecord, NormalizedFinancials, NormalizedQuote


def _quarter(revenue, net_income, eps, operating_income=None, **overrides):
    row = {
        "period_end": "2026-06-30",
        "revenue": revenue,
        "net_income": net_income,
        "eps": eps,
        "operating_income": operating_income,
        "interest_expense": 10.0,
        "tax_expense": 40.0,
        "cost_of_revenue": revenue - 400 if revenue else None,
        "depreciation_amortization": 50.0,
        "dividends_paid": None,
        "shares_outstanding": 100.0,
        "operating_cash_flow": None,
        "capital_expenditures": None,
    }
    row.update(overrides)
    return row


# 5 quarters, newest first — revenue/net_income/eps trending up, enough for
# a 4-quarter TTM sum and a quarters[4] YoY margin_trend comparison.
_QUARTERS = [
    _quarter(1000.0, 150.0, 1.5, 200.0, operating_cash_flow=180.0, capital_expenditures=40.0),
    _quarter(950.0, 140.0, 1.4, 190.0, operating_cash_flow=170.0, capital_expenditures=38.0),
    _quarter(900.0, 130.0, 1.3, 180.0, operating_cash_flow=160.0, capital_expenditures=36.0),
    _quarter(880.0, 120.0, 1.2, 170.0, operating_cash_flow=150.0, capital_expenditures=34.0),
    _quarter(800.0, 100.0, 1.0, 150.0, operating_cash_flow=140.0, capital_expenditures=30.0),
]

# 4 years, newest first — for revenue_growth_yoy (annual[0] vs [1]) and
# revenue_growth_3yr_cagr (annual[0] vs [3]).
_ANNUAL = [
    _quarter(3700.0, 500.0, 5.3, period_end="2025-12-31"),
    _quarter(3200.0, 420.0, 4.5, period_end="2024-12-31"),
    _quarter(2900.0, 380.0, 4.0, period_end="2023-12-31"),
    _quarter(2500.0, 320.0, 3.5, period_end="2022-12-31"),
]

_BALANCE_SHEET = {
    "total_assets": 5000.0,
    "total_liabilities": 3000.0,
    "total_equity": 2000.0,
    "total_debt": 1000.0,
    "cash_and_equivalents": 500.0,
    "current_assets": 1500.0,
    "current_liabilities": 700.0,
}


def _fin(**overrides) -> NormalizedFinancials:
    defaults = dict(quarters=_QUARTERS, annual=_ANNUAL, balance_sheet=_BALANCE_SHEET, currency="USD")
    defaults.update(overrides)
    return NormalizedFinancials(**defaults)


def _price_info(**overrides) -> NormalizedQuote:
    defaults = NormalizedQuote(
        current_price=50.0, market_cap=5000.0, currency="USD", high_52w=60.0, low_52w=40.0
    )
    defaults.update(overrides)
    return defaults


def _dividend_record(ex_date: str, amount: float) -> NormalizedDividendRecord:
    return NormalizedDividendRecord(ex_date=ex_date, payment_date=None, amount_per_share=amount)


def _dividend_history() -> list[NormalizedDividendRecord]:
    # 6 years x 4 quarterly payments, amount increasing slightly each year —
    # enough for dividend_growth_5yr (needs >5 distinct years) and a clean
    # consecutive_years_paid count.
    records = []
    for year, amount in zip(range(2020, 2026), [0.15, 0.16, 0.17, 0.18, 0.19, 0.20]):
        for month in ("02", "05", "08", "11"):
            records.append(_dividend_record(f"{year}-{month}-15", amount))
    return records


# --- compute_growth_metrics ---


def test_compute_growth_metrics_happy_path():
    result = compute_growth_metrics(_fin())

    assert result["revenue_growth_yoy"] == pytest.approx((3700.0 - 3200.0) / 3200.0)
    assert result["revenue_growth_3yr_cagr"] == pytest.approx((3700.0 / 2500.0) ** (1 / 3) - 1)
    assert result["eps_growth_yoy"] == pytest.approx((5.3 - 4.5) / 4.5)
    assert result["earnings_surprises"] is None
    # forward_pe is a VAL-block field (compute_valuation_metrics), not GROWTH —
    # real bug caught on review, was briefly duplicated in both dicts.
    assert "forward_pe" not in result


def test_compute_growth_metrics_insufficient_annual_data_returns_none():
    result = compute_growth_metrics(_fin(annual=[_ANNUAL[0]]))

    assert result["revenue_growth_yoy"] is None
    assert result["revenue_growth_3yr_cagr"] is None


def test_compute_growth_metrics_earnings_surprises_passed_through():
    """86bbdu04a: passed through directly, not collapsed to None when
    empty — None means no data source wired, [] means fetched-but-empty,
    and compute_all()'s missing_fields scan only flags None as missing."""
    surprises = [{"period_end": "2026-07-30", "eps_actual": 2.02, "eps_estimated": 1.89}]

    result = compute_growth_metrics(_fin(), earnings_surprises=surprises)

    assert result["earnings_surprises"] is surprises


def test_compute_growth_metrics_empty_earnings_surprises_stays_empty_list_not_none():
    result = compute_growth_metrics(_fin(), earnings_surprises=[])

    assert result["earnings_surprises"] == []


def test_compute_growth_metrics_cagr_none_when_base_year_non_positive():
    annual = [_ANNUAL[0], _ANNUAL[1], _ANNUAL[2], _quarter(-100.0, 0.0, 0.0, period_end="2022-12-31")]
    result = compute_growth_metrics(_fin(annual=annual))

    assert result["revenue_growth_3yr_cagr"] is None


# --- compute_profitability_metrics ---


def test_compute_profitability_metrics_happy_path():
    result = compute_profitability_metrics(_fin())

    assert result["gross_margin"] == pytest.approx((1000.0 - 600.0) / 1000.0)
    assert result["operating_margin"] == pytest.approx(200.0 / 1000.0)
    assert result["net_margin"] == pytest.approx(150.0 / 1000.0)
    ttm_net_income = 150.0 + 140.0 + 130.0 + 120.0
    assert result["roe"] == pytest.approx(ttm_net_income / 2000.0)
    assert result["fcf_to_net_income"] == pytest.approx((180.0 - 40.0) / 150.0)
    # latest net_margin 0.15 vs quarters[4] net_margin 100/800=0.125 -> +2.5pp -> expanding
    assert result["margin_trend"] == "expanding"


def test_compute_profitability_metrics_roe_none_when_book_equity_negative():
    """Negative book equity makes ROE meaningless — a net loss over negative
    equity reads as a positive ROE (86bbq04wm)."""
    result = compute_profitability_metrics(
        _fin(balance_sheet={**_BALANCE_SHEET, "total_equity": -500.0})
    )

    assert result["roe"] is None


def test_compute_profitability_metrics_empty_quarters_returns_all_none():
    result = compute_profitability_metrics(_fin(quarters=[]))

    assert result == {
        "gross_margin": None, "operating_margin": None, "net_margin": None,
        "roe": None, "margin_trend": None, "fcf_to_net_income": None,
    }


def test_margin_trend_contracting():
    quarters = [
        _quarter(1000.0, 50.0, 0.5),  # net_margin 0.05
        _quarter(1000.0, 50.0, 0.5),
        _quarter(1000.0, 50.0, 0.5),
        _quarter(1000.0, 50.0, 0.5),
        _quarter(1000.0, 150.0, 1.5),  # 4 quarters back: net_margin 0.15
    ]
    result = compute_profitability_metrics(_fin(quarters=quarters))

    assert result["margin_trend"] == "contracting"


def test_margin_trend_stable_within_deadband():
    quarters = [_quarter(1000.0, 150.0, 1.5), _quarter(1000.0, 149.0, 1.49)]
    result = compute_profitability_metrics(_fin(quarters=quarters))

    assert result["margin_trend"] == "stable"


# --- compute_balance_sheet_metrics ---


def test_compute_balance_sheet_metrics_happy_path():
    result = compute_balance_sheet_metrics(_fin())

    assert result == {
        "debt_to_equity": pytest.approx(0.5),
        "current_ratio": pytest.approx(1500.0 / 700.0),
        "interest_coverage": pytest.approx(20.0),
        "free_cash_flow": pytest.approx(140.0),
        "cash_position": 500.0,
    }
    assert "health_rating" not in result


def test_compute_balance_sheet_metrics_missing_interest_expense_omits_only_coverage():
    quarters = [_quarter(1000.0, 150.0, 1.5, 200.0, interest_expense=None)]
    result = compute_balance_sheet_metrics(_fin(quarters=quarters))

    assert result["interest_coverage"] is None
    assert result["debt_to_equity"] == pytest.approx(0.5)  # unaffected


def test_compute_balance_sheet_metrics_missing_total_debt_omits_only_de():
    result = compute_balance_sheet_metrics(_fin(balance_sheet={**_BALANCE_SHEET, "total_debt": None}))

    assert result["debt_to_equity"] is None
    assert result["current_ratio"] is not None  # unaffected


# --- compute_valuation_metrics ---


def test_compute_valuation_metrics_happy_path():
    growth = compute_growth_metrics(_fin())
    result = compute_valuation_metrics(_fin(), _price_info(), growth)

    ttm_eps = 1.5 + 1.4 + 1.3 + 1.2
    assert result["pe_ratio"] == pytest.approx(50.0 / ttm_eps)
    assert result["pb_ratio"] == pytest.approx(50.0 / (2000.0 / 100.0))
    ttm_revenue = 1000.0 + 950.0 + 900.0 + 880.0
    assert result["ps_ratio"] == pytest.approx(5000.0 / ttm_revenue)
    assert result["ev_ebitda"] == pytest.approx((5000.0 + 1000.0 - 500.0) / (200.0 + 50.0))
    assert result["forward_pe"] is None
    assert result["peg_ratio"] == pytest.approx(result["pe_ratio"] / (growth["eps_growth_yoy"] * 100))


def test_compute_valuation_metrics_forward_pe_with_analyst_estimates():
    """86bbdu04a: forward_pe mirrors pe_ratio's own guard style — current
    price divided by an EPS-like denominator, None if either is missing."""
    growth = compute_growth_metrics(_fin())
    analyst_estimates = {"forward_eps": 5.0}

    result = compute_valuation_metrics(_fin(), _price_info(), growth, analyst_estimates)

    assert result["forward_pe"] == pytest.approx(50.0 / 5.0)


def test_compute_valuation_metrics_forward_pe_none_when_forward_eps_missing():
    growth = compute_growth_metrics(_fin())
    analyst_estimates = {"forward_eps": None}

    result = compute_valuation_metrics(_fin(), _price_info(), growth, analyst_estimates)

    assert result["forward_pe"] is None


def test_compute_valuation_metrics_forward_pe_none_with_bare_empty_dict():
    """Real providers now return a bare {} on no data (fmp.py/openbb_tmx.py/
    yfinance.py, 86bbdu04a review) — not {"forward_eps": None} — matching
    get_company_info's/get_quote's own convention. Confirm this actual
    real-world shape degrades gracefully too, not just the {"forward_eps":
    None} shape."""
    growth = compute_growth_metrics(_fin())

    result = compute_valuation_metrics(_fin(), _price_info(), growth, {})

    assert result["forward_pe"] is None


def test_compute_valuation_metrics_thin_quarters_pe_ratio_none_not_approximated():
    """Real design decision: _ttm() returns None below 4 quarters rather
    than approximating from fewer — matches _cagr()'s own precedent."""
    growth = compute_growth_metrics(_fin())
    thin_fin = _fin(quarters=_QUARTERS[:2])

    result = compute_valuation_metrics(thin_fin, _price_info(), growth)

    assert result["pe_ratio"] is None
    assert result["ps_ratio"] is None


def test_ev_ebitda_always_computed_no_sector_conditional():
    """ev_ebitda is computed unconditionally here — sector-based suppression
    is the payload-template's job, not this module's."""
    growth = compute_growth_metrics(_fin())
    result = compute_valuation_metrics(_fin(), _price_info(), growth)

    assert result["ev_ebitda"] is not None


# --- compute_dividend_info ---


def test_compute_dividend_info_happy_path():
    result = compute_dividend_info(_fin(), _price_info(), _dividend_history())

    trailing_annual_dividend = 0.20 * 4
    assert result["dividend_yield"] == pytest.approx(trailing_annual_dividend / 50.0)
    ttm_eps = 1.5 + 1.4 + 1.3 + 1.2
    assert result["payout_ratio"] == pytest.approx(trailing_annual_dividend / ttm_eps)
    assert result["dividend_growth_5yr"] == pytest.approx((0.80 / 0.60) ** (1 / 5) - 1)
    assert result["dividend_type"] == "regular"
    assert result["consecutive_years_paid"] == 6


def test_compute_dividend_info_no_history_returns_none_type():
    result = compute_dividend_info(_fin(), _price_info(), [])

    assert result == {
        "dividend_yield": None,
        "payout_ratio": None,
        "dividend_growth_5yr": None,
        "dividend_type": "none",
        "consecutive_years_paid": 0,
    }


def test_compute_dividend_info_irregular_type_from_special_dividend():
    history = [_dividend_record(f"2025-{m:02d}-15", 0.20) for m in (2, 5, 8, 11)]
    history.append(_dividend_record("2025-12-01", 5.00))  # special dividend spike

    result = compute_dividend_info(_fin(), _price_info(), history)

    assert result["dividend_type"] == "irregular"


def test_compute_dividend_info_consecutive_years_paid_stops_at_gap():
    history = [_dividend_record("2025-06-15", 0.20), _dividend_record("2023-06-15", 0.18)]  # gap: no 2024

    result = compute_dividend_info(_fin(), _price_info(), history)

    assert result["consecutive_years_paid"] == 1


def test_compute_dividend_info_partial_current_year_does_not_distort_growth():
    """Real bug caught testing through the real code path against live
    AAPL data: comparing the most recent CALENDAR year's total (which is
    very likely partial — however many payments have happened so far this
    year) against a complete prior year systematically understates growth,
    even producing a false negative rate. trailing_annual_dividend (a
    genuine trailing-12-month total) must be used as the "now" figure
    instead of by_year[latest_year]."""
    history = [_dividend_record(f"{year}-{m:02d}-15", 0.20) for year in range(2020, 2026) for m in (2, 5, 8, 11)]
    # 2026: only 3 of 4 payments so far (partial year) — would show as a
    # decline if compared directly against a complete 2025.
    history += [_dividend_record(f"2026-{m:02d}-15", 0.20) for m in (2, 5, 8)]

    result = compute_dividend_info(_fin(), _price_info(), history)

    # trailing 4 payments = last 4 chronologically = 2025-11, 2026-02,
    # 2026-05, 2026-08 = 0.80; reference year 2026-5=2021 total = 0.80.
    # Real, correctly-flat growth, not a fabricated decline.
    assert result["dividend_growth_5yr"] == pytest.approx(0.0)


def test_compute_dividend_info_unsorted_input_handled_correctly():
    """dividend_history ordering isn't guaranteed by the caller — this
    must sort internally rather than assume order."""
    shuffled = list(reversed(_dividend_history()))

    result = compute_dividend_info(_fin(), _price_info(), shuffled)

    assert result["consecutive_years_paid"] == 6
    assert result["dividend_type"] == "regular"


# --- compute_peer_comparison ---


def test_compute_peer_comparison_happy_path():
    peer_fin = _fin(quarters=_QUARTERS, annual=_ANNUAL, balance_sheet=_BALANCE_SHEET)
    peer_price = _price_info(current_price=25.0, market_cap=2500.0)
    peer_data = [("PEER1", peer_fin, peer_price), ("PEER2", peer_fin, peer_price)]

    result = compute_peer_comparison(peer_data)

    assert len(result["peer_records"]) == 2
    assert result["peer_records"][0]["ticker"] == "PEER1"
    assert "pe_ratio" in result["peer_records"][0]
    # identical peers -> median equals either peer's own value
    assert result["sector_medians"]["sector_median_pe"] == pytest.approx(
        result["peer_records"][0]["pe_ratio"]
    )
    assert set(result["sector_medians"].keys()) == {
        "sector_median_pe", "sector_median_pb", "sector_median_rev_growth",
        "sector_median_gross_margin", "sector_median_op_margin", "sector_median_roe",
        "sector_median_de", "sector_median_ev_ebitda",
    }


def test_compute_peer_comparison_empty_peers_returns_empty():
    result = compute_peer_comparison([])

    assert result == {"sector_medians": {}, "peer_records": []}


# --- peer-median guard (86bbq04wm) ---


@pytest.mark.parametrize(
    "metric, in_range, out_of_range",
    [
        ("pe_ratio", 20.0, -5.0),
        ("pb_ratio", 3.0, -1.0),
        ("ev_ebitda", 12.0, -8.0),
        ("debt_to_equity", 0.5, -2.0),
        ("revenue_growth_yoy", 0.15, 1.75),  # SNDK spinoff-stub value
        ("gross_margin", 0.4, -0.2),
        ("operating_margin", 0.2, 1.5),
        ("roe", 0.25, 5.0),
    ],
)
def test_valid_peer_value_bounds(metric, in_range, out_of_range):
    assert _valid_peer_value(metric, in_range) == in_range
    assert _valid_peer_value(metric, out_of_range) is None
    assert _valid_peer_value(metric, None) is None


@pytest.mark.parametrize("metric", ["pe_ratio", "pb_ratio", "ev_ebitda", "debt_to_equity"])
def test_valid_peer_value_rejects_negative_and_exact_zero(metric):
    assert _valid_peer_value(metric, -1.0) is None
    assert _valid_peer_value(metric, 0.0) is None


def test_peer_metric_valid_range_covers_all_sector_median_keys():
    assert set(_PEER_METRIC_VALID_RANGE) == set(_SECTOR_MEDIAN_KEYS)


def test_compute_peer_comparison_single_peer_yields_no_medians():
    result = compute_peer_comparison([("SOLO", _fin(), _price_info(current_price=25.0))])

    assert len(result["peer_records"]) == 1
    assert result["peer_records"][0]["pe_ratio"] is not None  # the real value is kept
    assert all(v is None for v in result["sector_medians"].values())


# revenue 10000 vs 3000 -> revenue_growth_yoy 2.33, past the 1.5 upper bound
_EXTREME_GROWTH_ANNUAL = [
    _quarter(10000.0, 500.0, 5.0, period_end="2025-12-31"),
    _quarter(3000.0, 400.0, 4.0, period_end="2024-12-31"),
]


@pytest.mark.parametrize(
    "override, metric, median_key",
    [
        (
            {"balance_sheet": {**_BALANCE_SHEET, "total_equity": -500.0}},
            "pb_ratio",
            "sector_median_pb",
        ),
        (
            {"balance_sheet": {**_BALANCE_SHEET, "total_equity": -500.0}},
            "debt_to_equity",
            "sector_median_de",
        ),
        (
            {"balance_sheet": {**_BALANCE_SHEET, "total_debt": 0.0}},
            "debt_to_equity",
            "sector_median_de",
        ),
        ({"annual": _EXTREME_GROWTH_ANNUAL}, "revenue_growth_yoy", "sector_median_rev_growth"),
    ],
)
def test_compute_peer_comparison_rejects_implausible_values(override, metric, median_key):
    price = _price_info(current_price=25.0)
    peer_data = [("BAD", _fin(**override), price), ("GOOD", _fin(), price)]

    result = compute_peer_comparison(peer_data)

    bad_record = next(r for r in result["peer_records"] if r["ticker"] == "BAD")
    assert bad_record[metric] is None
    # GOOD alone leaves 1 valid contributor < _MIN_PEERS_FOR_SECTOR_MEDIAN
    assert result["sector_medians"][median_key] is None


def test_compute_peer_comparison_three_peers_two_valid_yields_median():
    price = _price_info(current_price=25.0)
    good = _fin()
    bad = _fin(balance_sheet={**_BALANCE_SHEET, "total_equity": -500.0})  # negative pb + de
    peer_data = [("A", good, price), ("B", good, price), ("C", bad, price)]

    result = compute_peer_comparison(peer_data)

    a_pb = result["peer_records"][0]["pb_ratio"]
    assert a_pb is not None
    assert result["peer_records"][2]["pb_ratio"] is None  # C rejected (negative pb)
    assert result["peer_records"][2]["roe"] is None  # C's roe suppressed at source (negative equity)
    assert result["sector_medians"]["sector_median_pb"] == pytest.approx(a_pb)  # median of A, B


# --- compute_all ---


def test_compute_all_happy_path_returns_full_shape():
    result = compute_all(_fin(), _price_info(), _dividend_history(), [])

    assert set(result.keys()) == {
        "valuation_metrics", "growth_metrics", "profitability_metrics",
        "balance_sheet_metrics", "dividend_info", "peer_metrics",
        "quarters_available", "missing_fields",
    }
    assert result["quarters_available"] == 5
    assert result["valuation_metrics"]["pe_ratio"] is not None
    assert result["dividend_info"]["dividend_yield"] is not None


def test_compute_all_currency_mismatch_raises():
    with pytest.raises(ValueError, match="Currency mismatch"):
        compute_all(_fin(currency="CAD"), _price_info(currency="USD"), [], [])


def test_compute_all_missing_fields_lists_none_valued_keys():
    thin_fin = _fin(quarters=_QUARTERS[:1], annual=[])
    result = compute_all(thin_fin, _price_info(), [], [])

    assert "valuation_metrics.pe_ratio" in result["missing_fields"]
    assert "growth_metrics.revenue_growth_yoy" in result["missing_fields"]
    # No analyst_estimates/earnings_surprises passed -> both stay None,
    # correctly flagged as missing (86bbdu04a).
    assert "valuation_metrics.forward_pe" in result["missing_fields"]


def test_compute_all_wires_analyst_estimates_and_earnings_surprises():
    """86bbdu04a: end-to-end through compute_all(), not just the two
    compute_* functions in isolation."""
    analyst_estimates = {"forward_eps": 5.0}
    earnings_surprises = [{"period_end": "2026-07-30", "eps_actual": 2.02, "eps_estimated": 1.89}]

    result = compute_all(
        _fin(), _price_info(), _dividend_history(), [], analyst_estimates, earnings_surprises
    )

    assert result["valuation_metrics"]["forward_pe"] == pytest.approx(50.0 / 5.0)
    assert result["growth_metrics"]["earnings_surprises"] == earnings_surprises
    assert "valuation_metrics.forward_pe" not in result["missing_fields"]
