"""The industry P/E benchmark, from one Yahoo screener call.

Both advocates build their valuation argument on how a stock's P/E compares with its peers (40 of the 42 peer or sector
citations in 11 recent real runs were about valuation, and they cite the trailing P/E). That comparison used to rest on
a median over the 2 to 5 peers Finnhub listed (MSFT's was three cybersecurity companies, one of them at 61.8, which
produced a "54% discount") or, for Canada, on nine hand-written entries. Here it is the median trailing P/E over every
company in the stock's Yahoo industry in its home market: one vendor call, a median, a minimum count, and no guess when
there is not enough to go on.

Yahoo's trailing P/E is reported earnings, the same definition as the pipeline's own P/E for the subject (they agreed
within about 2% for MSFT, KO, RDDT and SHOP.TO in the 2026-10-02 audit). Finnhub's P/E excludes extraordinary items and
gives industry medians 6 to 11% lower, so it is not mixed in. A Canadian stock whose TSX industry has fewer than five
P/E-bearing companies (Canadian software, for instance) uses the same industry in the US, and says so.

A median alone is not enough: how far an industry's P/Es spread decides whether a premium means anything. Measured
2026-10-03, MSFT's software-infrastructure median is 23.5 over all 82 companies, 34.8 over the largest 30 and 85.3 over
the largest 15, so a "discount" or "premium" to it is noise, while JPM's banks give 15.0 to 15.1 however the universe is
sliced. The benchmark therefore carries the middle half of the industry (25th to 75th percentile) and a position label
computed here: only a stock outside that range is meaningfully cheap or rich against its industry.

The two companies closest in market cap come from the same universe and feed the Researcher's contrast blocks. Their
ranking ignores the currency difference (at most 1.4x); it only orders two contrast companies.
"""
import asyncio
import math
import statistics
from dataclasses import dataclass

import structlog

from data.degradation import FETCH_FAILED
from data.degradation import report as report_degradation

logger = structlog.get_logger(__name__)

MIN_COMPANIES = 5
CLOSEST_COUNT = 2
_US_EXCHANGES = ["NMS", "NYQ", "NGM", "NCM", "ASE"]
_CA_EXCHANGES = ["TOR", "VAN"]
_SCREEN_SIZE = 250  # the largest 250 companies by market cap in the industry
# A P/E within this fraction of the median is in line whatever the spread: with 5 Canadian banks the middle half is
# 16.9 to 17.6, so TD.TO at 17.6 (+3.6%) would otherwise be "above the range".
NEAR_MEDIAN = 0.10
# Companies under this market cap are left out of the universe: on the TSX they are mostly tiny names whose P/Es drag the
# median (BAM.TO's asset-management median was 3.3 over 60 companies). Not currency-converted: the figure only filters.
MIN_MARKET_CAP = 500e6
# Above this a trailing P/E means earnings are near zero (CRWD 9,001), not a valuation: no comparison is made.
MAX_MEANINGFUL_PE = 200


@dataclass(frozen=True)
class IndustryBenchmark:
    industry: str  # Yahoo's screener name, e.g. "Software—Infrastructure"
    market: str  # "US" or "TSX": the universe the median is over
    companies: int  # companies with a positive trailing P/E in the median
    median_pe: float
    p25_pe: float  # the middle half of the industry's P/Es runs from p25 to p75
    p75_pe: float
    closest: tuple[str, ...]  # the companies closest in market cap, for the Researcher

    def as_dict(self) -> dict:
        return {"industry": self.industry, "market": self.market, "companies": self.companies,
                "median_pe": self.median_pe, "p25_pe": self.p25_pe, "p75_pe": self.p75_pe, "closest": list(self.closest)}


def pe_position(pe: float | None, p25: float | None, p75: float | None, median: float | None = None) -> str | None:
    """Where a P/E sits against the industry: "within_range" (inside the middle half, or within NEAR_MEDIAN of the
    median), "below_range" or "above_range"; None when either side is missing. Computed in code so the agents copy it
    instead of judging a noisy median."""
    if not isinstance(pe, (int, float)) or p25 is None or p75 is None or pe <= 0:
        return None
    if p25 <= pe <= p75 or (median and abs(pe / median - 1) <= NEAR_MEDIAN):
        return "within_range"
    return "below_range" if pe < p25 else "above_range"


def _industry_names() -> dict[str, str]:
    """Info-style name ("Banks - Diversified") to the screener's name ("Banks—Diversified"); names without a dash map to themselves."""
    from yfinance.const import EQUITY_SCREENER_EQ_MAP

    names = {}
    for group in EQUITY_SCREENER_EQ_MAP["industry"].values():
        for name in group:
            names[name.replace("—", " - ")] = name
            names[name] = name
    return names


def screener_industry(info_industry: str | None) -> str | None:
    """The screener's spelling of an industry from a Yahoo info name, or None when the screener has no such industry."""
    if not info_industry:
        return None
    return _industry_names().get(info_industry)


def pe_vs_industry(pe: float | None, benchmark: dict | None) -> tuple[float | None, str | None]:
    """(premium or discount to the industry median in percent, position against the middle half) for a P/E and the
    benchmark's dict form (IndustryBenchmark.as_dict). Both None when either side is missing. Used by the Fundamental
    payload and the Pass 2 view so they cannot disagree."""
    if not benchmark or not isinstance(pe, (int, float)) or pe <= 0 or not benchmark.get("median_pe"):
        return None, None
    if pe > MAX_MEANINGFUL_PE:
        return None, "not_meaningful"
    pct = round((pe / benchmark["median_pe"] - 1) * 100, 1)
    return pct, pe_position(pe, benchmark.get("p25_pe"), benchmark.get("p75_pe"), benchmark["median_pe"])


def is_canadian(ticker: str) -> bool:
    return ticker.upper().endswith((".TO", ".V"))


def _screen(industry: str, exchanges: list[str]) -> list[dict]:
    import yfinance as yf
    from yfinance import EquityQuery

    query = EquityQuery("and", [EquityQuery("eq", ["industry", industry]), EquityQuery("is-in", ["exchange", *exchanges])])
    result = yf.screen(query, size=_SCREEN_SIZE, sortField="intradaymarketcap", sortAsc=False)
    return result.get("quotes", [])


def summarize(
    subject: str, subject_cap: float | None, rows: list[dict], industry: str, market: str
) -> IndustryBenchmark | None:
    """The benchmark over `rows`, or None when fewer than MIN_COMPANIES others have a positive P/E.

    Others are the rows other than the subject with a market cap of at least MIN_MARKET_CAP (preferred-share lines carry
    none). Pure,
    so the rules are unit-tested without the network."""
    others = [r for r in rows if r.get("symbol", "").upper() != subject.upper() and (r.get("marketCap") or 0) >= MIN_MARKET_CAP]
    pes = [r["trailingPE"] for r in others if (r.get("trailingPE") or 0) > 0]
    if len(pes) < MIN_COMPANIES:
        return None
    if subject_cap and subject_cap > 0:
        ranked = sorted(others, key=lambda r: abs(math.log(r["marketCap"] / subject_cap)))
    else:
        ranked = others
    p25, median, p75 = statistics.quantiles(pes, n=4, method="inclusive")
    return IndustryBenchmark(industry, market, len(pes), float(median), float(p25), float(p75),
                             tuple(r["symbol"] for r in ranked[:CLOSEST_COUNT]))


def _benchmark_sync(ticker: str) -> IndustryBenchmark | None:
    import yfinance as yf

    info = yf.Ticker(ticker).info
    industry = screener_industry(info.get("industry"))
    if industry is None:
        return None
    cap = info.get("marketCap")
    canadian = is_canadian(ticker)
    home = (_CA_EXCHANGES, "TSX") if canadian else (_US_EXCHANGES, "US")
    result = summarize(ticker, cap, _screen(industry, home[0]), industry, home[1])
    if result is None and canadian:
        result = summarize(ticker, cap, _screen(industry, _US_EXCHANGES), industry, "US")
    return result


async def industry_benchmark(ticker: str) -> IndustryBenchmark | None:
    """The industry P/E benchmark for `ticker`, or None when it cannot be built (an unmapped industry, too few
    companies, or an error: reported, never guessed)."""
    try:
        return await asyncio.to_thread(_benchmark_sync, ticker)
    except Exception as exc:
        logger.warning("industry_benchmark_failed", ticker=ticker, exc_info=True)
        report_degradation("yfinance", "industry_benchmark", FETCH_FAILED, f"{ticker}: {exc}", exc=exc,
                           context={"symbol": ticker})
        return None
