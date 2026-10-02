"""Insider activity filtered to the notable transactions, sized in dollars.

The agents used to see only transaction counts ("Insider sells (90d): 6"), and a Bear advocate read that as
"6 shares of insider selling" and used it as a core argument. The counts hide the real size: on 2026-10-02 the
90-day sales were $20.8M over 6 transactions at MSFT, $49.3M over 80 at RDDT, $87.9M over 12 at KO and $4.3M at
AAPL, which are very different things for companies of very different size.

A transaction is notable when its value is at least max(0.001% of market cap, $25K): the rule the Sentiment
design already specifies (docs/agents/sentiment_analyst.md, decision 7), which was never built. It scales with
company size, so a routine executive sale at a $3.8T company is dropped and the same dollars at a $28B company
are kept. The market cap comes from the quote, which yfinance serves for the US and Canada alike.
"""
from datetime import UTC, datetime, timedelta

WINDOW_DAYS = 90
NOTABLE_PCT_OF_MARKET_CAP = 0.001
NOTABLE_FLOOR = 25_000.0


def _money(value: float) -> str:
    v = abs(value)
    if v >= 1e9:
        return f"${v / 1e9:.1f}B"
    if v >= 1e6:
        return f"${v / 1e6:.1f}M"
    if v >= 1e3:
        return f"${v / 1e3:.0f}K"
    return f"${v:.0f}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def summarize_insider_activity(transactions: list[dict], market_cap: float | None) -> dict:
    """The notable insider purchases and sales of the last 90 days, netted by dollar value.

    Qualifying rows are the ones the Sentiment counts always used: dated within the window, personal (not the
    issuer's own buyback) and purchase or sale only. `materiality` is "none" with no qualifying rows, "routine"
    when there are rows but none is notable, "notable" otherwise, and None when it cannot be judged: the market
    cap is unknown, or the only rows are an undated aggregate (openbb-tmx's quarterly per-owner fallback, e.g.
    ENB.TO). `buy_count` and `sell_count` count every qualifying row, notable or not."""
    cutoff = (datetime.now(UTC) - timedelta(days=WINDOW_DAYS)).date().isoformat()
    rows = [
        t for t in transactions
        if t.get("date") and t["date"] >= cutoff and not t.get("is_issuer")
        and t.get("transaction_type") in ("purchase", "sale")
    ]
    undated = [
        t for t in transactions
        if not t.get("date") and not t.get("is_issuer") and t.get("transaction_type") in ("purchase", "sale")
    ]
    out = {
        "buy_count": sum(1 for t in rows if t["transaction_type"] == "purchase"),
        "sell_count": sum(1 for t in rows if t["transaction_type"] == "sale"),
        "notable_count": 0, "net_value": 0.0, "threshold": None, "direction": None, "materiality": "none", "text": "",
    }
    if not rows:
        if undated:
            # Present but cannot be placed in a window: not the same as none, so do not say none.
            out["materiality"] = None
            out["text"] = "insider data is an undated aggregate, so 90-day activity cannot be sized"
        else:
            out["text"] = "no qualifying insider purchases or sales returned for the last 90 days"
        return out
    if not (isinstance(market_cap, (int, float)) and market_cap > 0):
        out["materiality"] = None
        out["text"] = f"{_plural(len(rows), 'qualifying transaction')} in the last 90 days, market cap unknown so none can be sized"
        return out

    threshold = max(market_cap * NOTABLE_PCT_OF_MARKET_CAP / 100, NOTABLE_FLOOR)
    notable = [t for t in rows if (t.get("value") or 0) >= threshold]
    out["threshold"] = threshold
    out["notable_count"] = len(notable)
    if not notable:
        out["materiality"] = "routine"
        out["text"] = (f"{_plural(len(rows), 'transaction')} in the last 90 days, none above the notable "
                       f"threshold of {_money(threshold)}: routine")
        return out

    buy_value = sum(t["value"] for t in notable if t["transaction_type"] == "purchase")
    sell_value = sum(t["value"] for t in notable if t["transaction_type"] == "sale")
    net = buy_value - sell_value
    out["net_value"] = net
    out["direction"] = "buying" if net > 0 else "selling" if net < 0 else "neutral"
    out["materiality"] = "notable"
    out["text"] = (f"net {out['direction']} {_money(net)} from {len(notable)} notable of {len(rows)} transactions "
                   f"(notable means {_money(threshold)} or more, 0.001% of market cap) over 90 days")
    return out
