"""The industry P/E benchmark against the real Yahoo screener (data/industry_benchmark.py).

The unit tests fake the network; this is the one place the unofficial screener endpoint and the industry-name mapping
are exercised for real, one US name and one Canadian name. Loose bounds on purpose: P/Es move, the shape must not."""

import pytest

from data.industry_benchmark import industry_benchmark

pytestmark = pytest.mark.live


@pytest.mark.parametrize("ticker,market", [("MSFT", "US"), ("TD.TO", "TSX")])
async def test_a_real_benchmark_has_a_sane_shape(ticker, market):
    bench = await industry_benchmark(ticker)

    assert bench is not None, f"no benchmark for {ticker}: screener down or industry unmapped"
    assert bench.market == market
    assert bench.companies >= 5
    assert 0 < bench.p25_pe <= bench.median_pe <= bench.p75_pe < 500
    assert len(bench.closest) == 2 and ticker not in bench.closest
