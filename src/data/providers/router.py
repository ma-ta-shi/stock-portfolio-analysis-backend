"""CA/US dispatcher + per-data-type fallback chains (ClickUp 86bb4758g).

Single dispatch point enforcing CLAUDE.md's routing rule, which nothing in
code previously enforced. Resolves the CA-vs-US branch once per stock, then
runs each method's own fallback chain (try the first suitable provider,
treat an empty/missing/not-implemented result as "move to next," stop at
the first non-empty result or exhaustion).

Core principle: provider adapters stay dumb, router owns every completeness
check. Every adapter wraps exactly one external API and returns whatever it
gets, including None/empty — none of them call another provider. edgartools.py's
former internal fallback (financials None -> yfinance) and us_equity.py's
FMP-with-yfinance-fallback logic both moved here; us_equity.py was deleted.

Chains are an explicit static table, not a generic "try every registered
StockDataProvider" loop — get_financials/get_insider_trading must never
fall back to FMP for fundamentals (CLAUDE.md hard rule), which a generic
loop could silently violate.

Out of scope: per-provider ticker normalization (each adapter's own job —
this router passes the canonical ticker through unchanged), MacroDataProvider
routing, building new provider adapters (e.g. Gap 6's openbb-tmx health
check, Gap 1's live peers source, Gap 2/3's openbb-tmx news/insider support
are all still NotImplementedError stubs on their adapters — this router
just skips them the same way it skips any other empty link).
"""

import json
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import structlog

from data.providers.base import NewsProvider, StockDataProvider
from data.providers.edgartools import EdgarToolsDataProvider
from data.providers.finnhub import FinnhubDataProvider
from data.providers.fmp import FMPDataProvider
from data.providers.openbb_tmx import OpenBBTMXProvider
from data.providers.yfinance import YFinanceDataProvider, YFinanceNewsProvider, correct_alignment

logger = structlog.get_logger(__name__)

_DEFAULT_PEERS_PATH = Path(__file__).resolve().parent.parent / "peers.json"

# .TO (TSX) and .V (TSXV) — confirmed live a .TO-only check misses TSXV
# listings like PLAN.V. Fallback only: used when no Stock record exists.
_CA_SUFFIXES = (".TO", ".V")


class StockLike(Protocol):
    """Duck-typed stand-in for api.tables.stock.Stock — avoids importing the
    SQLAlchemy model (and its api.database side effects) into the data layer."""

    primary_exchange: str
    currency: str


def is_canadian_ticker(ticker: str) -> bool:
    """Raw suffix check — last resort when no Stock record exists. Confirmed
    live this alone is unsafe as the primary signal: bare 'T' is AT&T (NYSE,
    USD) while 'T.TO' is TELUS (TSX, CAD) — completely different companies."""
    return ticker.upper().endswith(_CA_SUFFIXES)


def is_canadian(stock: StockLike | None = None, *, ticker: str | None = None) -> bool:
    """Preferred: Stock.primary_exchange/currency — same check as
    get_benchmark(stock) in data-pipeline.md, so both stay in sync. Only
    falls back to ticker-suffix detection when no Stock record exists
    (e.g. a bare string from data/peers.json)."""
    if stock is not None:
        return stock.primary_exchange in ("TSX", "TSXV") or stock.currency == "CAD"
    if ticker is not None:
        return is_canadian_ticker(ticker)
    raise ValueError("is_canadian() needs either a stock or a ticker")


def _is_empty(value: Any) -> bool:
    """Generalized from us_equity.py — the proven "is this missing" check,
    dict/list/DataFrame-aware."""
    if value is None:
        return True
    if isinstance(value, pd.DataFrame):
        return value.empty
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


# Explicit static chains — one list per method, in try-order. Never inferred
# from "which providers happen to implement this method."
US_CHAINS: dict[str, list[str]] = {
    "get_price_history": ["fmp", "yfinance"],
    "get_dividend_history": ["fmp", "yfinance"],
    "get_quote": ["fmp", "yfinance"],
    "get_company_info": ["fmp", "yfinance"],
    "get_financials": ["edgartools", "yfinance"],
    "get_insider_trading": ["edgartools"],  # never FMP — CLAUDE.md hard rule
    "get_analyst_estimates": ["fmp"],
    "get_analyst_ratings": ["fmp"],
    "get_earnings_calendar": ["fmp", "finnhub"],
    "get_peers": ["finnhub"],
    "get_news": ["finnhub"],
    "get_analyst_recommendation_trends": ["finnhub"],
    "get_ratios_ttm": ["fmp"],  # not on the ABC — carried forward from us_equity.py
}

CA_CHAINS: dict[str, list[str]] = {
    "get_price_history": ["openbb_tmx", "yfinance"],
    "get_dividend_history": ["openbb_tmx"],
    # openbb_tmx has no get_quote method at all (confirmed live 2026-08-04,
    # 86bb7j0kh) — yfinance-only, not a real two-link chain. Previously
    # listed openbb_tmx first, which _try_chain's getattr(provider,
    # method_name, None) silently skipped every single call, same as a
    # NotImplementedError would — misleading, not a real fallback.
    "get_quote": ["yfinance"],
    "get_company_info": ["openbb_tmx"],
    "get_financials": ["yfinance"],
    "get_insider_trading": ["openbb_tmx"],
    "get_analyst_estimates": ["openbb_tmx"],
    "get_analyst_ratings": ["yfinance"],
    "get_earnings_calendar": ["openbb_tmx"],
    "get_peers": ["peers_json"],  # static file, not a live provider — Gap 1
    "get_news": ["openbb_tmx"],
    "get_analyst_recommendation_trends": ["yfinance_news"],
    "get_ratios_ttm": [],  # no CA equivalent — empty chain resolves to {} via _try_chain
}


class Router(StockDataProvider, NewsProvider):
    """Single object DataPipeline.prepare() calls uniformly per stock.
    Resolves CA/US once at construction; every method call below runs its
    own fallback chain against that branch's provider table."""

    def __init__(
        self,
        stock: StockLike | None = None,
        ticker: str | None = None,
        *,
        fmp: FMPDataProvider | None = None,
        edgartools: EdgarToolsDataProvider | None = None,
        yfinance: YFinanceDataProvider | None = None,
        yfinance_news: YFinanceNewsProvider | None = None,
        finnhub: FinnhubDataProvider | None = None,
        openbb_tmx: OpenBBTMXProvider | None = None,
        peers_path: Path | str | None = None,
    ) -> None:
        self.is_ca = is_canadian(stock, ticker=ticker)
        self._providers: dict[str, object] = {}
        # yfinance is on both branches' chains; the API-key-gated providers
        # (fmp, finnhub) are only constructed — and only required to have a
        # key set — for the branch that actually uses them.
        self._add_provider("yfinance", yfinance, YFinanceDataProvider, needed=True)
        self._add_provider("openbb_tmx", openbb_tmx, OpenBBTMXProvider, needed=self.is_ca)
        self._add_provider("yfinance_news", yfinance_news, YFinanceNewsProvider, needed=self.is_ca)
        self._add_provider("fmp", fmp, FMPDataProvider, needed=not self.is_ca)
        self._add_provider("edgartools", edgartools, EdgarToolsDataProvider, needed=not self.is_ca)
        self._add_provider("finnhub", finnhub, FinnhubDataProvider, needed=not self.is_ca)
        self._peers_path = Path(peers_path) if peers_path else _DEFAULT_PEERS_PATH

    def _add_provider(self, key: str, override: object | None, ctor: type, *, needed: bool) -> None:
        if override is not None:
            self._providers[key] = override
        elif needed:
            self._providers[key] = ctor()

    async def __aenter__(self) -> "Router":
        for provider in self._providers.values():
            if hasattr(provider, "__aenter__"):
                await provider.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        for provider in self._providers.values():
            if hasattr(provider, "__aexit__"):
                await provider.__aexit__(exc_type, exc_val, exc_tb)

    def _chain(self, method_name: str) -> list[str]:
        return (CA_CHAINS if self.is_ca else US_CHAINS)[method_name]

    async def _try_chain(self, method_name: str, *args, **kwargs) -> tuple[Any, str | None]:
        """Try each provider in method_name's chain in order. NotImplementedError
        and a missing method both mean "this provider doesn't cover this" and
        move to the next link. A RuntimeError shaped like a bad/missing API key
        (mirrors us_equity.py's _try_fmp precedent) fails loud instead of being
        swallowed as a per-call failure. Anything else is logged and treated as
        a fallback trigger. Returns (result, provider_key_that_produced_it)."""
        result: Any = None
        for key in self._chain(method_name):
            provider = self._providers.get(key)
            method = getattr(provider, method_name, None) if provider is not None else None
            if method is None:
                continue
            try:
                result = await method(*args, **kwargs)
            except NotImplementedError:
                result = None
                continue
            except RuntimeError as e:
                if "401" in str(e):
                    raise
                logger.warning(
                    "router_chain_call_failed", method=method_name, provider=key, error=str(e)
                )
                result = None
                continue
            except Exception:
                logger.warning(
                    "router_chain_call_failed", method=method_name, provider=key, exc_info=True
                )
                result = None
                continue
            if not _is_empty(result):
                return result, key
        return result, None

    # ---------- StockDataProvider ----------

    async def get_price_history(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        result, _ = await self._try_chain("get_price_history", ticker, period, interval)
        return result if not _is_empty(result) else pd.DataFrame()

    async def get_financials(
        self, ticker: str, statement: str, period: str
    ) -> tuple[pd.DataFrame, str | None]:
        """Returns (DataFrame, source) — source exposed (86bbb001k) so callers
        can dispatch to the right adapter's normalize_financials() without
        Router itself doing any shape translation. _is_empty()/_try_chain()
        are untouched; this only surfaces a value _try_chain already computed.

        correct_alignment fixes a yfinance-specific column-misalignment bug
        (financial-data-api-research.md §2) — apply it whenever yfinance is
        the source, regardless of whether it's the CA primary or US fallback."""
        result, source = await self._try_chain("get_financials", ticker, statement, period)
        if _is_empty(result):
            return pd.DataFrame(), source
        if source == "yfinance":
            result = correct_alignment(result)
        return result, source

    async def get_company_info(self, ticker: str) -> dict:
        result, _ = await self._try_chain("get_company_info", ticker)
        return result if not _is_empty(result) else {}

    async def get_analyst_estimates(self, ticker: str) -> dict:
        result, _ = await self._try_chain("get_analyst_estimates", ticker)
        return result if not _is_empty(result) else {}

    async def get_analyst_ratings(self, ticker: str) -> dict:
        result, _ = await self._try_chain("get_analyst_ratings", ticker)
        return result if not _is_empty(result) else {}

    async def get_insider_trading(self, ticker: str, days: int = 90) -> list[dict]:
        result, _ = await self._try_chain("get_insider_trading", ticker, days)
        return result if not _is_empty(result) else []

    async def get_peers(self, ticker: str, limit: int = 5) -> list[str]:
        if self.is_ca:
            return self._load_ca_peers(ticker, limit)
        result, _ = await self._try_chain("get_peers", ticker, limit)
        return result if not _is_empty(result) else []

    async def get_earnings_calendar(self, ticker: str) -> list[dict]:
        result, _ = await self._try_chain("get_earnings_calendar", ticker)
        return result if not _is_empty(result) else []

    async def get_dividend_history(self, ticker: str, from_date: str, to_date: str) -> list[dict]:
        result, _ = await self._try_chain("get_dividend_history", ticker, from_date, to_date)
        return result if not _is_empty(result) else []

    async def get_quote(self, ticker: str) -> dict:
        """Not on StockDataProvider — carried forward from us_equity.py."""
        result, _ = await self._try_chain("get_quote", ticker)
        return result if not _is_empty(result) else {}

    async def get_ratios_ttm(self, ticker: str) -> dict:
        """Not on StockDataProvider — FMP-only cross-check, carried forward
        from us_equity.py. No CA equivalent; empty chain on the CA branch."""
        result, _ = await self._try_chain("get_ratios_ttm", ticker)
        return result if not _is_empty(result) else {}

    # ---------- NewsProvider ----------

    async def get_news(self, ticker: str, days: int) -> list[dict]:
        result, _ = await self._try_chain("get_news", ticker, days)
        return result if not _is_empty(result) else []

    async def get_analyst_recommendation_trends(self, ticker: str) -> list[dict]:
        result, _ = await self._try_chain("get_analyst_recommendation_trends", ticker)
        return result if not _is_empty(result) else []

    # ---------- CA get_peers static file (Gap 1 — not a live provider) ----------

    def _load_ca_peers(self, ticker: str, limit: int) -> list[str]:
        try:
            content = self._peers_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            logger.warning("router_peers_json_missing", path=str(self._peers_path))
            return []
        if not content:
            return []
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("router_peers_json_invalid", path=str(self._peers_path))
            return []
        return list(data.get(ticker.upper(), []))[:limit]
