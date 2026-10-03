"""The industry P/E benchmark (data/industry_benchmark.py). The screener call is faked; the rules are the point."""

from data import industry_benchmark as ib


def _row(symbol, cap, pe):
    return {"symbol": symbol, "marketCap": cap, "trailingPE": pe}


ROWS = [_row("AAA", 100e9, 20.0), _row("BBB", 50e9, 30.0), _row("CCC", 10e9, 10.0), _row("DDD", 5e9, 40.0),
        _row("EEE", 1e9, 25.0), _row("SELF", 60e9, 99.0)]


def test_the_median_is_over_the_positive_pes_of_everyone_but_the_subject():
    out = ib.summarize("SELF", 60e9, ROWS, "Software—Infrastructure", "US")
    assert out.companies == 5 and out.median_pe == 25.0  # 10, 20, 25, 30, 40; SELF's 99 is not in it
    assert out.industry == "Software—Infrastructure" and out.market == "US"


def test_the_subject_is_excluded_case_insensitively():
    assert ib.summarize("self", 60e9, ROWS, "x", "US").companies == 5


def test_fewer_than_five_pe_bearing_companies_gives_no_benchmark():
    four = [_row("A", 1e9, 10.0), _row("B", 1e9, 20.0), _row("C", 1e9, 30.0), _row("D", 1e9, 40.0)]
    assert ib.summarize("X", 1e9, four, "x", "US") is None


def test_losses_missing_pes_and_preferred_lines_do_not_count():
    rows = [_row("A", 1e9, 10.0), _row("B", 1e9, 20.0), _row("C", 1e9, 30.0), _row("D", 1e9, 40.0),
            _row("LOSS", 1e9, -5.0), _row("NOPE", 1e9, None), _row("ENB-PY.TO", 0.0, 12.0)]
    assert ib.summarize("X", 1e9, rows, "x", "US") is None  # only four count
    assert ib.summarize("X", 1e9, rows + [_row("E", 1e9, 50.0)], "x", "US").companies == 5


def test_the_closest_companies_are_the_nearest_in_market_cap_to_the_subject():
    out = ib.summarize("SELF", 60e9, ROWS, "x", "US")
    assert out.closest == ("BBB", "AAA")  # 50B and 100B are nearest to 60B in log terms


def test_without_a_subject_cap_the_closest_are_just_the_first_listed():
    out = ib.summarize("SELF", None, ROWS, "x", "US")
    assert len(out.closest) == 2


def test_the_two_industry_spellings_map_to_the_screeners_name():
    assert ib.screener_industry("Banks - Diversified") == "Banks—Diversified"
    assert ib.screener_industry("Software - Infrastructure") == "Software—Infrastructure"
    assert ib.screener_industry("Oil & Gas Midstream") == "Oil & Gas Midstream"
    assert ib.screener_industry("Not A Real Industry") is None and ib.screener_industry(None) is None


def test_canadian_tickers_are_recognised():
    assert ib.is_canadian("RY.TO") and ib.is_canadian("SGML.V") and not ib.is_canadian("MSFT")


async def test_a_canadian_stock_with_too_few_tsx_companies_uses_the_us_industry(monkeypatch):
    calls = []

    def fake_screen(industry, exchanges):
        calls.append(tuple(exchanges))
        return [_row("T1", 1e9, 10.0)] if exchanges == ib._CA_EXCHANGES else ROWS[:5]

    class _Ticker:
        info = {"industry": "Software - Application", "marketCap": 60e9}

    monkeypatch.setattr(ib, "_screen", fake_screen)
    monkeypatch.setattr("yfinance.Ticker", lambda t: _Ticker())
    out = await ib.industry_benchmark("SHOP.TO")
    assert out.market == "US" and out.companies == 5
    assert calls == [tuple(ib._CA_EXCHANGES), tuple(ib._US_EXCHANGES)]


async def test_a_us_stock_never_looks_at_the_tsx(monkeypatch):
    calls = []
    monkeypatch.setattr(ib, "_screen", lambda i, e: calls.append(tuple(e)) or [_row("T1", 1e9, 10.0)])

    class _Ticker:
        info = {"industry": "Software - Application", "marketCap": 60e9}

    monkeypatch.setattr("yfinance.Ticker", lambda t: _Ticker())
    assert await ib.industry_benchmark("MSFT") is None
    assert calls == [tuple(ib._US_EXCHANGES)]


async def test_an_unmapped_industry_or_a_failure_gives_none_not_a_guess(monkeypatch):
    class _NoIndustry:
        info = {"industry": None, "marketCap": 1e9}

    monkeypatch.setattr("yfinance.Ticker", lambda t: _NoIndustry())
    assert await ib.industry_benchmark("QQQ") is None

    def boom(t):
        raise RuntimeError("screener is down")

    monkeypatch.setattr("yfinance.Ticker", boom)
    assert await ib.industry_benchmark("MSFT") is None


def test_the_middle_half_is_carried_with_the_median():
    out = ib.summarize("SELF", 60e9, ROWS, "x", "US")  # P/Es 10, 20, 25, 30, 40
    assert (out.p25_pe, out.median_pe, out.p75_pe) == (20.0, 25.0, 30.0)
    assert out.as_dict()["p25_pe"] == 20.0 and out.as_dict()["p75_pe"] == 30.0


def test_a_pe_is_positioned_against_the_middle_half_not_just_the_median():
    """MSFT 2026-10-03: +23% against the median but inside the industry's 12.9 to 50.7 middle half: in line."""
    assert ib.pe_position(28.8, 12.9, 50.7) == "within_range"
    assert ib.pe_position(34.3, 6.2, 18.2) == "above_range"
    assert ib.pe_position(16.5, 19.1, 40.0) == "below_range"
    assert ib.pe_position(20.0, 20.0, 30.0) == "within_range"  # the edges are inside
    assert ib.pe_position(None, 1.0, 2.0) is None and ib.pe_position(-5.0, 1.0, 2.0) is None
    assert ib.pe_position(10.0, None, 2.0) is None



def test_a_pe_near_the_median_is_in_line_even_when_the_middle_half_is_narrow():
    """TD.TO 2026-10-03: five Canadian banks give a middle half of 16.9 to 17.55; TD at 17.57 (+3.6% on the median 17.0)
    would be "above the range" on a technicality."""
    assert ib.pe_position(17.57, 16.9, 17.55, 17.0) == "within_range"
    assert ib.pe_position(17.57, 16.9, 17.55) == "above_range"  # without the median rule it is just over
    assert ib.pe_position(20.0, 16.9, 17.55, 17.0) == "above_range"  # +18% is outside both tests
    assert ib.pe_position(15.0, 16.9, 17.55, 17.0) == "below_range"


def test_pe_vs_industry_uses_the_median_rule():
    bench = {"median_pe": 17.0, "p25_pe": 16.9, "p75_pe": 17.55}
    assert ib.pe_vs_industry(17.57, bench) == (3.4, "within_range")


def test_tiny_companies_are_left_out_of_the_universe():
    """BAM.TO 2026-10-03: a TSX asset-management median of 3.3 over 60 companies, many of them tiny."""
    rows = [_row(f"BIG{i}", 5e9, 20.0 + i) for i in range(5)] + [_row(f"TINY{i}", 100e6, 2.0) for i in range(10)]
    out = ib.summarize("SELF", 5e9, rows, "x", "TSX")
    assert out.companies == 5 and out.median_pe == 22.0
    too_few = ib.summarize("SELF", 5e9, rows[:4] + rows[5:], "x", "TSX")
    assert too_few is None  # four large companies are not enough, however many tiny ones there are


def test_a_pe_from_near_zero_earnings_is_not_meaningful():
    bench = {"median_pe": 23.5, "p25_pe": 13.0, "p75_pe": 50.4}
    assert ib.pe_vs_industry(9001.0, bench) == (None, "not_meaningful")
    assert ib.pe_vs_industry(150.0, bench)[1] == "above_range"  # large but still a P/E
