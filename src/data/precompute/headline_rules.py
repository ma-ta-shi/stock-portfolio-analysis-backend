"""Sentiment labels decided by pattern, not by the model, for headline types whose label is fixed by their wording.

About 1 headline in 8 is one of these (536 distinct headlines from saved runs, 2026-10-09): the daily "closed at $X, up 1%"
price recaps, "to host a webcast" and transcript items, "X vs Y: which is the better buy" comparisons, and analyst rating or
price-target changes. The model labels the recaps inconsistently (a recap that "falls" came back negative, one that "laps the
market" positive), so a pattern is both more accurate and free. Such items leave the model's batch, so it has less to get wrong
and a shorter prompt.

Pure functions, no network. `rule_label` returns "positive", "negative", "neutral" or None (None: ask the model).

Safeguards: a rule only fires on an item that names the company (a rival's webcast, or another firm's upgrade, is the model's
call, since it may be "unrelated"); a price-target change outranks a reiterated rating in the same headline ("Reiterates Neutral,
Lowers Price Target" is a cut); and the recap pattern needs a market-direction word after "market", so "jumps as market reacts to
the loan" (news) is not a recap.
"""
import html
import re

_MARKET_MOVE = r"(uptick|slip|slips|gains?|declines?|dips?|improves?|rally|rallies|drops?|rises?|falls?|downturn|sell-?off|weakness|strength)"
_RECAP_HEADLINE = re.compile(
    r"\b(than|the|broader|amid|as|while|against)\b.{0,12}\bmarket\b.{0,30}\b(today|here'?s why|key facts|what (you|investors)|"
    r"information for investors|important facts|some facts|facts to note)\b"
    r"|\b(laps|lags|outpaces?|outpaced|trails|outperforms?|underperforms?)\b.{0,20}\b(the )?(stock )?market\b"
    rf"|\b(rises|gains|falls|dips|declines|sinks|slips|climbs|jumps|drops|advances|increases)\b.{{0,40}}\b(as|while|amid|despite)\b.{{0,12}}\bmarket {_MARKET_MOVE}\b",
    re.I,
)
_RECAP_SNIPPET = re.compile(
    r"\b(closed|settled|settling|concluded|reached|stood|closing|ended)\b.{0,40}\$\s?[\d.,]+.{0,80}\b(move|change|shift)\b", re.I)
_SCHEDULED = re.compile(
    r"\bto (host|hold|report|announce|release)\b.{0,50}\b(webcast|conference call|results|earnings|quarter)\b"
    r"|\bearnings (scheduled|call transcript|preview|calendar)\b|\b(analyst|investor) day (transcript|slideshow|presentation)\b"
    r"|\btranscript\b|\bwhat to expect\b.{0,30}\bearnings",
    re.I,
)
_TRENDING = re.compile(r"\btrending stock\b", re.I)
# "X vs Y", "X or Y: which is the better buy": commentary with no new facts about either company.
_COMPARISON = re.compile(
    r"\b(vs\.?|versus)\b.{0,80}\b(better|which|buy|winner|worth|dividend (king|stock)|stock)\b|\bwhich\b.{0,40}\b(better|buy|stock)\b", re.I)
_PT_RAISE = re.compile(r"\b(raises?|boosts?|lifts?|hikes?) (its )?(price target|pt)\b|\bprice target (raised|boosted|lifted)\b", re.I)
_PT_CUT = re.compile(r"\b(lowers?|cuts?|slashes?|trims?|reduces?) (its )?(price target|pt)\b|\bprice target (lowered|cut|slashed|trimmed)\b", re.I)
_UPGRADE = re.compile(r"\bupgraded to (buy|strong buy|outperform|overweight)\b|\bupgraded\b.{0,10}\bbuy\b", re.I)
_DOWNGRADE = re.compile(r"\bdowngraded? to\b|\(rating downgrade\)", re.I)
_REITERATE = re.compile(
    r"\b(reiterates?|maintains?|reinstates?|initiates?)\b.{0,50}\b(neutral|market perform|equal.weight|sector perform|hold rating|a hold)\b"
    r"|\b(reiterates?|reinstates?|initiates?)\s+(a\s+)?hold\b", re.I)
_DIVIDEND_UP = re.compile(r"\b(announces?|declares?|raises?|hikes?|boosts?|lifts?)\b.{0,25}\b(quarterly |annual )?dividend (increase|hike)\b"
                          r"|\b(raises?|hikes?|boosts?|lifts?)\b.{0,15}\b(quarterly |annual )?dividend\b", re.I)

_SUFFIXES = {"inc", "corp", "corporation", "co", "company", "ltd", "limited", "plc", "group", "holdings", "the"}
# A first word like these names no company by itself ("Bank of Montreal" would match any bank headline), so the whole name is used.
_GENERIC_FIRST_WORDS = {"bank", "first", "american", "united", "general", "national", "royal", "canadian", "global", "international",
                        "north", "new", "western", "eastern", "southern", "great", "advanced", "applied", "digital", "energy"}


def _plain(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", html.unescape(text).lower().replace("’", "'"))


def names_company(headline: str, company: str | None, ticker: str | None) -> bool:
    """True when the headline names the company by its first word(s) ('Brookfield', 'Coca-Cola') or its ticker written as a
    ticker: '(ENB)', 'NYSE:ENB', 'TSX:BAM', 'ENB:CA'. A bare lowercase word is never taken for a ticker (ON, PG, V)."""
    plain = _plain(headline)
    if company:
        words = [w for w in _plain(company).split() if w not in _SUFFIXES]
        if words:
            lead = " ".join(words) if words[0] in _GENERIC_FIRST_WORDS and len(words) > 1 else words[0]
            if len(lead) >= 3 and re.search(rf"\b{re.escape(lead)}\b", plain):
                return True
    if ticker:
        base = re.escape(ticker.split(".")[0].upper())
        if re.search(rf"\({base}\)|\({base}:|\b[A-Z]{{2,6}}:{base}\b|\b{base}:[A-Z]{{2}}\b", html.unescape(headline)):
            return True
    return False


def rule_label(headline: str, snippet: str | None, company: str | None, ticker: str | None) -> str | None:
    """"positive" | "negative" | "neutral" when the wording fixes the label and the item names the company, else None."""
    if not names_company(headline, company, ticker):
        return None
    h = html.unescape(headline)
    if _PT_CUT.search(h) or _DOWNGRADE.search(h):
        return "negative"
    if _PT_RAISE.search(h) or _UPGRADE.search(h) or _DIVIDEND_UP.search(h):
        return "positive"
    if _REITERATE.search(h) or _SCHEDULED.search(h) or _TRENDING.search(h) or _RECAP_HEADLINE.search(h) or _COMPARISON.search(h):
        return "neutral"
    if snippet and _RECAP_SNIPPET.search(html.unescape(snippet)):
        return "neutral"
    return None
