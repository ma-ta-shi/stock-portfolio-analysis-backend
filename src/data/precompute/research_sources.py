"""research_sources.py — sourcing and derivation for the Stock Researcher's
ResearchSourcesBundle (ClickUp 86ban0x1u, slices 2a-ii and 2b).

Builds the three sourcing legs of the bundle: filing-section digests
(FILING:Business/FILING:MDA citations), news items (shared N{num} namespace
with precompute/sentiment.py), and peer blocks (PEER_n citations) - plus
the derivation/compliance layer on top: management_signals (the only
management-quality source the live prompt allows), entity anonymization,
dual_class_flag/cik_verified, sedar_filing_available, the mechanical
counters, and build_research_sources(), the top-level function that
assembles all of it into one validator-satisfying ResearchSourcesBundle.
Every citation token the live Stock Researcher Agent Prompt.md can use has
to trace back to something one of these functions actually put in the
payload, or the LLM either invents a plausible-sounding claim from
training data or (correctly, per the prompt's own rules) says
"insufficient_data" for everything - the whole reason this module exists.

News-item sourcing has a load-bearing contract, mirrored from
precompute/sentiment.py::summarize_news() (confirmed by direct read of
that module's own docstring): this module does NOT call
news_id_assignment.assign_news_ids() itself. It takes already-ID-assigned
articles as an input. ID assignment happens once per run, over the widest
window any Pass 1 agent uses (currently this module's own 180-day window),
and the same assigned list is handed to every consumer - both this module
and precompute/sentiment.py. If this module fetched and assigned its own
IDs independently, a legitimately different fetch (timing, provider
retries) could silently produce different N{num} numbering than Sentiment
sees for the same article, breaking the "same article, same N{num}
everywhere" cross-agent citation guarantee news_id_assignment.py exists
to provide. The real "fetch once, assign once, distribute to every
precompute module" orchestration is DataPipeline.prepare()'s job - not yet
built (no data_pipeline.py in this repo as of this slice).
"""

import asyncio
import re
from datetime import datetime, timedelta

import aiohttp
import structlog
from edgar import Company

from data.precompute.filing_summarizer import summarize_filing_section
from data.providers.ca_crosslisting import (
    get_crosslisted_business_overview,
    get_crosslisted_mda,
    get_expected_cik,
    get_us_ticker,
    is_crosslisted,
)
from data.providers.edgartools import EdgarToolsDataProvider
from data.providers.router import Router, StockLike, is_canadian
from data.schemas.common import FilingDigest, NewsItem, PeerBlock
from data.schemas.research_sources_bundle import ManagementSignals, ResearchSourcesBundle

logger = structlog.get_logger(__name__)

_PEER_LIMIT = 2
_PEER_NEWS_DAYS = 30  # deliberately narrower than this module's own 180-day news
# window - peer recent-news is 1-2 flavor headlines embedded in a <=200-token
# block, not a citation-tracked list. A live check during planning found AAPL
# alone returns 245 articles in just a 30-day window; fetching a peer's full
# 180-day window for 2 headlines would be real, unnecessary volume.
_PEER_SUMMARY_TOKEN_BUDGET = 50
_PEER_BLOCK_TOKEN_BUDGET = 200
_NEWS_LABEL = "Recent news: "

_CA_MARKET_SUFFIXES = (".TO", ".V")  # mirrors router.py's own _CA_SUFFIXES (private
# there) - duplicated here rather than imported across a private module boundary,
# same disclosed-duplication posture as _truncate_to_tokens's relationship to
# filing_summarizer.py's _truncate_to_budget above. Keep in sync if router.py's changes.
_DUAL_CLASS_SUFFIXES = ("-A", "-B", ".A", ".B")
_SHARE_CLASS_SUFFIX_RE = re.compile(r",?\s*Class\s+[A-Za-z0-9].*$", re.IGNORECASE)  # used
# by both compute_dual_class_flag (detects a "Class A/B/..." qualifier in the real
# company name - a ticker-suffix-independent dual-class signal) and _derive_short_name
# further down (strips this same qualifier off a name for anonymization matching) -
# one real-world signal, two different uses of it.

_INSIDER_WINDOW_DAYS = 365  # covers both insider_net_direction_90d's own 90-day
# window (filtered client-side from this) and buyback_activity's active/suspended
# split (91-365 day range) - one fetch, two derived views.
_BUYBACK_RECENT_DAYS = 90

_DIVIDEND_LOOKBACK_DAYS = 1825  # ~5 years - wide enough to reliably distinguish
# "never paid, ever" from "paid historically, recently stopped" (suspended)
# without an unbounded fetch. A stock with zero dividends in 5+ years is a
# reasonable "none" for investment-analysis purposes even if it technically paid
# one decades ago. Refined during implementation from the plan's original
# 12-month sketch, which can't distinguish those two cases from a single fetch.
_DIVIDEND_RECENT_DAYS = 365  # the "is this currently active" bar - matches
# c_suite_changes_12mo's own 12-month framing for "current" activity.


async def build_filing_digests(
    ticker: str, stock: StockLike | None = None
) -> tuple[list[FilingDigest], str | None]:
    """Business + MDA digests, plus the freshest filing_date among sections
    that actually produced a digest (not merely among sections retrieved -
    see below). FilingDigest itself carries no date field; Slice 2b's
    latest_filing_age_days comes from the second return value here rather
    than being re-derived, since ResearchSourcesBundle's own validator
    requires that field be None if and only if filing_digests is empty.

    Three paths, decided by is_crosslisted() first, then is_canadian():
      1. Cross-listed CA (in the ~176-ticker map) -> the existing
         ca_crosslisting.py retrieval (40-F/10-K/20-F/6-K, already built).
      2. Not cross-listed, not Canadian -> the general US path
         (EdgarToolsDataProvider.get_filing_section(), 86ban0x1u/2a-i).
      3. Not cross-listed, Canadian (pure CA, unmapped) -> no filing text
         source exists for this population (per the ticket's own explicit
         guidance) - returns ([], None), not an error.

    `stock` is optional and, as of this slice, always None in practice -
    mirrors Router.__init__'s own (also currently always-None-in-practice)
    parameter, so is_canadian() can use the more reliable
    stock.primary_exchange/currency check once a real Stock record is
    threaded through here, without a signature change.

    Both sections are fetched concurrently in paths 1 and 2, and both
    summarize_filing_section() calls share one aiohttp.ClientSession
    (mirrors precompute/sentiment.py's session-sharing pattern). No
    persistent cache - every call re-fetches and re-summarizes; the
    (cik, accession_number, section_id) digest cache is a separate,
    unbuilt ticket (DataBundle Assembly epic)."""
    if is_crosslisted(ticker):
        business_raw, mda_raw = await asyncio.gather(
            get_crosslisted_business_overview(ticker),
            get_crosslisted_mda(ticker),
        )
    elif not is_canadian(stock, ticker=ticker):
        provider = EdgarToolsDataProvider()
        business_raw, mda_raw = await asyncio.gather(
            provider.get_filing_section(ticker, "Business"),
            provider.get_filing_section(ticker, "MDA"),
        )
    else:
        logger.info("no_filing_source_pure_ca", ticker=ticker)
        return [], None

    sections = [
        (section, raw)
        for section, raw in (("Business", business_raw), ("MDA", mda_raw))
        if raw is not None
    ]
    if not sections:
        return [], None

    async with aiohttp.ClientSession() as session:
        digest_results = await asyncio.gather(
            *(summarize_filing_section(session, raw["text"], section) for section, raw in sections)
        )

    # Pair each digest with its own source section's filing_date, then drop
    # the pairs where summarization failed (HTTP error, non-answer, empty
    # digest - summarize_filing_section returns None for all of these). A
    # section that was retrieved but never became a digest must not count
    # toward "latest" either, or this function's own contract above breaks.
    survivors = [
        (digest, raw["filing_date"])
        for digest, (_, raw) in zip(digest_results, sections, strict=True)
        if digest is not None
    ]
    if not survivors:
        return [], None

    digests = [digest for digest, _ in survivors]
    latest_date = max(filing_date for _, filing_date in survivors)
    return digests, latest_date


def build_news_items(id_assigned_articles: list[dict]) -> list[NewsItem]:
    """Pure mapper over news_id_assignment.assign_news_ids()'s output -
    does not fetch news or assign IDs itself (see module docstring). Drops
    the `text` key (an addition for precompute/sentiment.py's own LLM
    scoring call, not part of NewsItem's schema).

    No quality_tier filtering here. The live Stock Researcher prompt's
    payload rendering drops "low" tier headlines before the LLM sees them
    (v1.3) - but precompute/sentiment.py's own prompt still renders every
    tier, including "low" (live-confirmed common, not rare). Filtering
    here would silently break Sentiment's need to see the full set; the
    exact per-prompt filtering decision belongs to whatever still-unbuilt
    code assembles each agent's actual payload from this shared list, not
    to this module (same posture as the vocabulary-mapping deferrals
    elsewhere in this ticket's design)."""
    return [
        NewsItem(
            id=article["id"],
            date=article["date"],
            headline=article["headline"],
            source=article["source"],
            quality_tier=article["quality_tier"],
        )
        for article in id_assigned_articles
    ]


def _truncate_to_tokens(text: str, token_budget: int) -> str:
    """Cut `text` to the last sentence-ending punctuation at or before
    token_budget * 4 characters (~4 chars/token). Same technique as
    filing_summarizer.py's own _truncate_to_budget(), reimplemented here
    rather than shared: that helper is private and hardcoded to its own
    250-token module constant, not parameterized - peer blocks need two
    different, smaller budgets (50 for a business summary, up to 200 for
    a whole block), so sharing it would mean widening a single-call-site
    private helper for a second, unrelated caller. A disclosed, deliberate
    small duplication of a well-understood technique, not a reinvention.

    Real bug caught on real data at the 50-token budget: a genuine
    yfinance business summary opening "Dell Technologies Inc. designs,
    develops..." truncated to just "Dell Technologies Inc." - "Inc. " was
    the only ". " anywhere in the 200-char window, so the naive
    last-". "-in-window rule mistook a corporate-suffix abbreviation for
    the end of a sentence. A sentence-boundary cut that would throw away
    more than half the budget is treated the same as no boundary found at
    all (fall back to a hard character cut) - filing_summarizer.py's own
    250-token budget makes this far less likely to bite in practice (more
    budget means more real sentences in the window), which is why this
    wasn't caught there; the smaller budgets here make it a real, not
    theoretical, failure mode."""
    limit = token_budget * 4
    if len(text) <= limit:
        return text
    window = text[:limit]
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if cut == -1:
        cut = window.rfind(".")
    if cut == -1 or cut < limit // 2:
        return window.rstrip()
    return window[: cut + 1].rstrip()


def _render_peer_content(summary: str | None, news: list[dict]) -> str | None:
    """One human-readable string per peer, business summary + recent news
    pre-joined. PeerBlock's schema (common.py) has only one flat `content`
    field - no room for the two separate placeholders the live prompt
    template actually renders ({peer_N_business_summary}/
    {peer_N_recent_news} as two distinct lines) - so whatever eventually
    renders that template either uses this whole string in one slot or
    parses it back apart; that mapping decision belongs to the still-
    unbuilt payload-rendering layer, not here.

    The business-summary component gets its own fixed 50-token budget;
    news headlines get whatever of the 200-token whole-block budget
    remains, so an unusually long business summary can't silently starve
    the news line rendered in the same block down to nothing (or vice
    versa - the summary's own budget is fixed regardless of headline
    length).

    Returns None (not an empty string) when there is nothing to report at
    all - the caller drops a peer with no content rather than rendering an
    empty PEER_n block, since an empty block isn't citable evidence."""
    business_line = None
    if summary:
        business_line = f"Business: {_truncate_to_tokens(summary, _PEER_SUMMARY_TOKEN_BUDGET)}"

    # Router.get_news()'s raw order is provider-internal, not chronological
    # (live-confirmed: AAPL's first raw article was not its most recent) -
    # sort explicitly before taking the top 2. published_at is a fixed-
    # format "%Y-%m-%d %H:%M:%S" string on both providers, so a plain
    # string sort is correct without parsing.
    #
    # Deduplicate by headline while walking the sorted list, not just take
    # a naive [:2] slice - real bug caught on real data: Router.get_news()
    # merges a crosslisted CA ticker's TMX feed with its matching Finnhub
    # feed and dedupes only on URL, so the same real story reaches both
    # feeds under two different URLs and survives as two entries with an
    # identical headline (confirmed live: TD.TO, RY.TO both had this,
    # ~15-16% of their 180-day news volume). Without this, a peer's "top 2
    # recent stories" could genuinely be the same story twice.
    headlines: list[str] = []
    seen_headlines: set[str] = set()
    for article in sorted(news, key=lambda a: a.get("published_at", ""), reverse=True):
        headline = article.get("headline")
        if not headline or headline in seen_headlines:
            continue
        seen_headlines.add(headline)
        headlines.append(headline)
        if len(headlines) == 2:
            break
    news_line = None
    if headlines:
        # Business line + the "Recent news: " label both come out of the
        # 200-token whole-block budget before truncating headlines to fit.
        # Stays in character units end-to-end rather than converting to a
        # token count twice (once for "used", once for "available") -
        # doing that the first way round left a real, live-caught overflow
        # (a maxed-out summary + long headlines rendered ~204 tokens
        # against the 200 budget) from two floor-divisions compounding.
        used_chars = (len(business_line) if business_line else 0) + len(_NEWS_LABEL)
        remaining_chars = max(0, _PEER_BLOCK_TOKEN_BUDGET * 4 - used_chars)
        truncated_headlines = _truncate_to_tokens("; ".join(headlines), remaining_chars // 4)
        news_line = f"{_NEWS_LABEL}{truncated_headlines}"

    lines = [line for line in (business_line, news_line) if line]
    return "\n".join(lines) if lines else None


async def _fetch_peer_content(peer_ticker: str, news_days: int) -> str | None:
    """One peer's own get_business_summary() and get_news(), fetched
    concurrently, rendered into one content string (see
    _render_peer_content). A fresh Router(ticker=peer_ticker) per peer -
    Router is constructed per-ticker, not parameterizable per-call."""
    async with Router(ticker=peer_ticker) as peer_router:
        summary, news = await asyncio.gather(
            peer_router.get_business_summary(peer_ticker),
            peer_router.get_news(peer_ticker, news_days),
        )
    return _render_peer_content(summary, news)


async def _fetch_company_name(ticker: str) -> str | None:
    """Real company name for anonymization (86ban0x1u/2b) -
    NormalizedCompanyInfo.name via Router.get_company_info(). Shared by
    two callers: build_peer_blocks (only for a peer that already survived
    the content-based drop - fetching a name for a peer that's getting
    dropped anyway would be wasted work) and build_research_sources
    directly (for the main ticker's own name). Returns None (not "") on
    any failure, matching Router.get_company_info()'s own {}
    degrade-on-failure convention; each caller decides what "no name"
    means for its own case (build_peer_blocks drops the peer;
    build_research_sources logs a warning and proceeds with an empty
    string, since the main ticker is the entire subject of the analysis
    and dropping the whole bundle over this would be a much bigger
    behavior change for a near-unreachable edge case).

    build_research_sources's own fetch of this (the main ticker's name,
    not a peer's) now serves two purposes, not one: anonymization
    (unchanged) and compute_dual_class_flag's share-class-qualifier
    signal (finding #47) - a second, real use for the same already-
    fetched value, not a second fetch."""
    async with Router(ticker=ticker) as router:
        info = await router.get_company_info(ticker)
    return info.get("name") or None


async def build_peer_blocks(
    ticker: str, limit: int = _PEER_LIMIT, news_days: int = _PEER_NEWS_DAYS
) -> list[tuple[PeerBlock, str]]:
    """Router.get_peers(ticker, limit) -> every candidate peer's content
    fetched concurrently (asyncio.gather, order-preserving - matches the
    concurrency already used within one peer and within
    build_filing_digests, rather than fetching peers one at a time).

    A peer with nothing citable in either piece is dropped, not rendered
    as an empty block; survivors renumber PEER_1..PEER_n with no gap.
    This is not just tidiness: the live prompt's payload template has no
    peer_id-keyed lookup the way news_items does (confirmed by reading the
    prompt's actual merge-logic code, not just its template) - the only
    Python-side consumption of peer_blocks is a length check
    (`len(sources.peer_blocks)`), which means the {peer_1_...}/
    {peer_2_...} template slots are almost certainly filled positionally.
    A gap or an out-of-order peer_id wouldn't just risk failing citation
    validation, it would put the wrong peer's content in the wrong slot.

    Returns (PeerBlock, real_company_name) pairs, not bare PeerBlocks
    (86ban0x1u/2b, finding #46): PeerBlock's own schema has no room for a
    name field (common.py, extra="forbid"), but Slice 2b's anonymization
    step needs to know each surviving peer's real name to substitute with
    PEER_n_COMPANY - without it, there is no safe way to anonymize that
    peer's content at all. The name is fetched via Router.get_company_info
    only for peers that already survived the content-based drop above -
    fetching a name for a peer that gets dropped anyway for having no
    usable content would be wasted work. A peer whose company_info/name
    lookup itself comes back empty is dropped the same way a
    no-usable-content peer already is, not rendered with a name that
    can't be safely anonymized - so PEER_n numbering is only finalized
    after both filters, never renumbered twice.

    Degrades to [] when get_peers() returns nothing, which it already
    guarantees never raises."""
    async with Router(ticker=ticker) as router:
        peer_tickers = await router.get_peers(ticker, limit=limit)
    if not peer_tickers:
        return []

    contents = await asyncio.gather(
        *(_fetch_peer_content(peer_ticker, news_days) for peer_ticker in peer_tickers)
    )
    survivors: list[tuple[str, str]] = []
    for peer_ticker, content in zip(peer_tickers, contents, strict=True):
        if content is None:
            logger.info("peer_block_dropped_no_data", ticker=ticker, peer_ticker=peer_ticker)
            continue
        survivors.append((peer_ticker, content))
    if not survivors:
        return []

    names = await asyncio.gather(
        *(_fetch_company_name(peer_ticker) for peer_ticker, _ in survivors)
    )

    blocks: list[tuple[PeerBlock, str]] = []
    for (peer_ticker, content), name in zip(survivors, names, strict=True):
        if not name:
            logger.info("peer_block_dropped_no_name", ticker=ticker, peer_ticker=peer_ticker)
            continue
        block = PeerBlock(
            peer_id=f"PEER_{len(blocks) + 1}",
            content=content,
            token_count=max(1, len(content) // 4),
        )
        blocks.append((block, name))
    return blocks


def _cutoff_date(days: int) -> str:
    """ISO date `days` ago - all of this module's date fields (insider
    transaction date, dividend ex_date) are plain "YYYY-MM-DD" strings, so
    a lexicographic string comparison against this is correct without
    parsing, same convention _render_peer_content already uses for
    published_at."""
    return (datetime.now() - timedelta(days=days)).date().isoformat()


def _compute_insider_direction(insider_rows: list[dict]) -> str | None:
    """buying/selling/neutral, or None if no qualifying rows exist to
    judge from (openbb-tmx's aggregate fallback can't support this at all
    - no per-transaction direction beyond one rollup - so this correctly
    lands on None for that path, not a guess). Excludes is_issuer=True
    rows (that's buyback_activity's signal, not personal insider
    direction) and exercise/gift/other transaction types (same exclusion
    intent as the Sentiment prompt's Form-4 M/G exclusion) - only
    purchase/sale rows from real people carry directional conviction.
    Netted by dollar value, not row count: a handful of large sales
    outweighing many small purchases should read as selling, not buying."""
    recent = [r for r in insider_rows if r.get("date") and r["date"] >= _cutoff_date(90)]
    qualifying = [
        r for r in recent if not r["is_issuer"] and r["transaction_type"] in ("purchase", "sale")
    ]
    if not qualifying:
        return None
    purchase_value = sum(r["value"] or 0 for r in qualifying if r["transaction_type"] == "purchase")
    sale_value = sum(r["value"] or 0 for r in qualifying if r["transaction_type"] == "sale")
    if purchase_value > sale_value:
        return "buying"
    if sale_value > purchase_value:
        return "selling"
    return "neutral"


def _compute_buyback_activity(insider_rows: list[dict]) -> str:
    """active/suspended/none, split by recency over the same 365-day
    fetch insider_net_direction_90d uses (finding #38). Disclosed weak in
    two ways, not one: Form 4 doesn't capture large-scale US corporate
    buybacks as insider transactions the way CA sources do (confirmed
    live: AAPL had zero buyback rows across a full 365-day window despite
    a real, large buyback program per its own MD&A digest); and the
    openbb-tmx CA aggregate fallback hardcodes is_issuer=False always, so
    it structurally cannot report a buyback at all when yfinance is empty
    for a ticker (confirmed live: BCE.TO). Both causes land on "none"
    here, the same conservative choice as an ordinary confirmed-zero -
    this function has no way to tell those apart from what Router hands
    it, and "none" never fabricates "active", matching this whole
    module's missing-over-wrong posture."""
    buybacks = [r for r in insider_rows if r["is_issuer"] and r["transaction_type"] == "buyback"]
    if not buybacks:
        return "none"
    recent_cutoff = _cutoff_date(_BUYBACK_RECENT_DAYS)
    has_recent = any(r["date"] and r["date"] >= recent_cutoff for r in buybacks)
    return "active" if has_recent else "suspended"


def _compute_dividend_activity(dividends: list[dict]) -> str:
    """increased/held/cut/suspended/none, from a wide-window fetch
    (_DIVIDEND_LOOKBACK_DAYS). none = zero records anywhere in that
    window (confirmed live: SHOP.TO, a real growth stock that has never
    paid a dividend). suspended = real history exists but none fall
    within the last _DIVIDEND_RECENT_DAYS (a real signal, distinct from
    never having paid one - not the same as "none"). increased/held/cut
    compare the two most recent records overall (not restricted to the
    recent window - a company with one payment 13 months ago and one 4
    months ago still has a real trend to report, confirmed live: RY.TO's
    own 5-payment history shows a clean increasing trend this way)."""
    if not dividends:
        return "none"
    recent_cutoff = _cutoff_date(_DIVIDEND_RECENT_DAYS)
    if not any(d["ex_date"] >= recent_cutoff for d in dividends):
        return "suspended"
    ordered = sorted(dividends, key=lambda d: d["ex_date"])
    if len(ordered) < 2:
        return "held"
    latest, previous = ordered[-1]["amount_per_share"], ordered[-2]["amount_per_share"]
    if latest > previous:
        return "increased"
    if latest < previous:
        return "cut"
    return "held"


async def build_management_signals(ticker: str) -> ManagementSignals:
    """One Router.get_insider_trading(ticker, 365) call feeds both
    insider_net_direction_90d and buyback_activity (finding #38 - one
    provider call, two derived views). dividend_activity from
    Router.get_dividend_history over a separate, wider window
    (_DIVIDEND_LOOKBACK_DAYS, finding #36's none-vs-suspended split needs
    more than 12 months of history to tell apart). c_suite_changes_12mo
    and changes_detail are always None/"" - no provider anywhere in this
    codebase exposes structured executive-change data, and a prior
    version of this exact field once had a roster size mistakenly
    rendered into it, reaching the CIO's evidence base as a fabricated
    instability signal. A plausible-looking guess here is worse than a
    null; do not add one without first grounding it in real examples the
    way the Slice 1 insider-text classifier was."""
    async with Router(ticker=ticker) as router:
        insider_rows, dividends = await asyncio.gather(
            router.get_insider_trading(ticker, _INSIDER_WINDOW_DAYS),
            router.get_dividend_history(
                ticker, _cutoff_date(_DIVIDEND_LOOKBACK_DAYS), _cutoff_date(0)
            ),
        )

    return ManagementSignals(
        c_suite_changes_12mo=None,
        changes_detail="",
        insider_net_direction_90d=_compute_insider_direction(insider_rows),
        buyback_activity=_compute_buyback_activity(insider_rows),
        dividend_activity=_compute_dividend_activity(dividends),
    )


def compute_dual_class_flag(ticker: str, company_name: str) -> bool:
    """Two independent signals, either one sufficient: (1) a ticker-suffix
    check - strips a CA market suffix first, THEN checks -A/-B/.A/.B on
    what remains (checking the raw ticker directly, no strip, is a real
    bug, not a hypothetical one: "RCI-B.TO".endswith("-B") is False, since
    the string actually ends in ".TO" - confirmed by running it before
    this fix existed; confirmed correct on 11 real tickers after the fix:
    RCI-B.TO/CCL-B.TO/BRK-A/BRK-B -> True, RY.TO/AAPL/REI-UN.TO/
    CAR-UN.TO/SHOP.TO -> False, the REIT-unit -UN suffix correctly does
    not false-positive); (2) a share-class qualifier in the real
    company_info name (_SHARE_CLASS_SUFFIX_RE, the same regex
    _derive_short_name already uses to strip this exact qualifier for
    anonymization - reused here to detect its presence instead).

    The second signal is not optional scope-creep - it's a real,
    live-caught miss in the ticker-suffix check alone, confirmed against
    the live Stock Researcher prompt's own named dual-class test fixture:
    "Pass Shopify (dual-class) with dual_class_flag=true." SHOP.TO has no
    distinguishing ticker suffix at all (its Class B is founder-held,
    never publicly traded under any ticker - the same "no suffix exists
    to check" shape as the already-disclosed GOOG/GOOGL gap below), so
    the suffix check alone returns False for the prompt's own primary
    example. Checked live: SHOP.TO's real company_info name is "Shopify
    Inc. Class A Subordinate Voting Shares" - _SHARE_CLASS_SUFFIX_RE
    matches it. Re-checked live that this doesn't introduce a false
    positive for genuinely single-class names: RY.TO ("Royal Bank of
    Canada") and AAPL ("Apple Inc.") both have no "Class N" qualifier and
    correctly don't match. A real company is not named "Class A/B/C"
    unless a second class exists to distinguish it from - a
    naming-convention assumption, not a proven-exhaustive rule, but one
    with no confirmed counterexample in this session's data.

    Confirmed live to still miss GOOG/GOOGL-style dual-class ADRs where
    NEITHER signal fires: no ticker suffix distinguishes them, and no
    yfinance field distinguishes them either (shortName/longName/
    quoteType are byte-identical for both) - disclosed, not silently
    treated as complete coverage. `company_name` is the caller's own
    already-fetched company_info name (build_research_sources reuses
    _fetch_company_name's result for the main ticker, same fetch
    anonymization already needed) - no new I/O added for this check."""
    stripped = ticker.upper()
    for suffix in _CA_MARKET_SUFFIXES:
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)]
            break
    if stripped.endswith(_DUAL_CLASS_SUFFIXES):
        return True
    return bool(_SHARE_CLASS_SUFFIX_RE.search(company_name))


async def compute_cik_verified(
    ticker: str, has_filing_digest: bool, stock: StockLike | None = None
) -> bool:
    """Same is_crosslisted() -> is_canadian(stock, ticker=ticker) path
    selection as build_filing_digests, for the same reason - needs
    `stock` to use the more reliable check when a real Stock record is
    available. CA crosslisted: a live edgar.Company(us_ticker).cik
    resolution cross-checked against ca_us_crosslisting.json's own
    stored cik - a genuine data-integrity check that would catch a stale
    map entry, confirmed live to currently match for RY.TO/SHOP.TO/
    AEM.TO. Any failure resolving the live CIK (a bad/unmapped us_ticker,
    a genuinely invalid company) degrades to False, not an exception -
    matches get_native_filing_section's own established "degrade on any
    failure" convention for this exact library call. Plain US / no
    filing source: returns has_filing_digest directly - the caller
    passes in what build_filing_digests already determined rather than
    this function re-deriving it, which would mean redundantly
    re-fetching and re-summarizing the same filing content a second
    time."""
    if is_crosslisted(ticker):
        us_ticker = get_us_ticker(ticker)
        expected_cik = get_expected_cik(ticker)
        if us_ticker is None or expected_cik is None:
            return False
        try:
            live_cik = await asyncio.to_thread(lambda: Company(us_ticker).cik)
        except Exception:
            logger.info("cik_verification_failed", ticker=ticker, us_ticker=us_ticker)
            return False
        return live_cik == expected_cik
    if not is_canadian(stock, ticker=ticker):
        return has_filing_digest
    return False


_WORD_BOUNDARY_FLAGS = re.IGNORECASE  # case-insensitive: a missed differently-cased
# mention leaking the real name (under-matching) is the failure mode to avoid for a
# safety feature, not the minor cosmetic risk of an over-eager coincidental match.


def _substitute_word_boundary(text: str, target: str, replacement: str) -> str:
    """Word-boundary-anchored, not a bare substring replace - the ticket's
    own E17 finding: `RY` matched inside "advisory" and corrupted a
    business summary. `target` empty/falsy is a no-op, not an error (a
    peer or main ticker whose real name couldn't be resolved must not
    reach this function at all per the caller's own drop-on-failure
    discipline, but guarding here too costs nothing).

    Uses (?<!\\w)...(?!\\w) lookaround, not \\b - a real, live-caught bug:
    plain \\b requires a word/non-word *transition* at that exact
    position, which breaks for the extremely common case of a company
    name ending in punctuation ("Block, Inc.", "Apple Inc."). The
    trailing "." is itself a non-word character, so a trailing \\b right
    after it requires the text that follows to start with a word
    character - which a normal sentence (followed by a space) never
    does, so real company names ending in "Inc."/"Corp."/"Ltd." silently
    failed to anonymize at all. The lookaround form only requires "not a
    word character immediately before/after the match", which correctly
    handles a punctuation-terminated target while still refusing to
    match `RY` inside "advisory" (the character before "ry" there is "o",
    a word character, so (?<!\\w) still correctly blocks it)."""
    if not target:
        return text
    pattern = r"(?<!\w)" + re.escape(target) + r"(?!\w)"
    return re.sub(pattern, replacement, text, flags=_WORD_BOUNDARY_FLAGS)


_CORPORATE_SUFFIX_RE = re.compile(
    r",?\s+(?:Incorporated|Inc|Corporation|Corp|Limited|Ltd|L\.L\.C\.|LLC|PLC|Company|Co)\.?$",
    re.IGNORECASE,
)
_TRAILING_THE_RE = re.compile(r"\s*\(The\)\s*$", re.IGNORECASE)
_LEADING_THE_RE = re.compile(r"^The\s+", re.IGNORECASE)


def _derive_short_name(name: str) -> str | None:
    """A shorter form of a company name, for anonymization to also match
    against - or None if nothing meaningful strips off. Real, live-caught
    need, not a hypothetical one: matching only the exact
    get_company_info() name string left real company names unanonymized
    in the majority of a 3-ticker live check, for three different
    reasons - (1) that field can include noise a filing's own prose never
    repeats, confirmed live: SHOP.TO's name is "Shopify Inc. Class A
    Subordinate Voting Shares", but its business summary just says
    "Shopify Inc., a commerce..."; (2) it can differ in word order from
    how the prose refers to the company, confirmed live: TD.TO's name is
    "Toronto-Dominion Bank (The)", but its business summary says "The
    Toronto-Dominion Bank..."; (3) LLM-summarized MD&A text commonly uses
    a short/possessive reference regardless of the full registered name,
    confirmed live: AAPL's own MD&A digest said "Apple's fiscal 2025..."
    even though get_company_info() correctly returns "Apple Inc." (which
    matched AAPL's Business digest just fine - the gap is specific to
    free-flowing summarized prose, not every occurrence).

    Strips (in order, since a share-class suffix has to come off before a
    trailing corporate suffix becomes visible to strip): a trailing
    "Class A/B/... Shares"-style share-class qualifier; a trailing
    corporate suffix (Inc./Corp./Ltd./LLC/etc., with or without a leading
    comma); a trailing "(The)"; a leading "The ". Confirmed live this
    closes every gap found above - "Apple" (from "Apple Inc.") still
    word-boundary-matches inside "Apple's" (the apostrophe is a
    non-word character, so a boundary exists there); "Toronto-Dominion
    Bank" (from "Toronto-Dominion Bank (The)") matches inside "The
    Toronto-Dominion Bank..." without needing to also generate a
    leading-"The" variant, since the shorter string is a substring either
    way; "Shopify" (from the share-class-qualified name) matches "Shopify
    Inc.,...". Not exhaustive - a company referred to by a completely
    different informal name (a "Old Navy" for "Gap Inc." style case) is a
    real, remaining gap this can't close, disclosed rather than silently
    assumed complete."""
    stripped = _SHARE_CLASS_SUFFIX_RE.sub("", name)
    stripped = _CORPORATE_SUFFIX_RE.sub("", stripped)
    stripped = _TRAILING_THE_RE.sub("", stripped)
    stripped = _LEADING_THE_RE.sub("", stripped)
    stripped = stripped.strip().rstrip(",")
    return stripped if stripped and stripped != name else None


def _name_variants(name: str) -> list[str]:
    """The full name, plus its derived short name if one exists -
    longest first, though substitution order doesn't actually matter for
    correctness here (each pass only replaces literal matches of its own
    target, so a full-name match consumes the text before the short-name
    pass ever sees it, and there's nothing left to double-substitute)."""
    variants = [name]
    short = _derive_short_name(name)
    if short:
        variants.append(short)
    return variants


def anonymize_content(text: str, company_name: str, ticker: str, peer_names: dict[str, str]) -> str:
    """Replaces the main company's name/ticker with COMPANY_X/TICKER_X,
    and each peer's real name (keyed by peer_id, e.g. "PEER_1") with
    f"{peer_id}_COMPANY" - matching the live prompt's own deanonymize
    step (confirmed by direct read of merge_output_researcher, not the
    ticket text). Applied to filing digest content and each peer's own
    PeerBlock.content only - never to news headlines, which were never
    anonymized in the first place (86bawptxr's own explicit instruction).
    A peer's own ticker is not separately templated anywhere in the live
    prompt (only {peer_1_token}, the literal citation string "PEER_1"),
    so no peer-ticker substitution is needed, only peer-name.

    Each name is matched via both its full and short form (_name_variants,
    finding from live 2b testing: the exact company_info name string
    alone left real names unanonymized in most of a 3-ticker check)."""
    result = text
    for variant in _name_variants(company_name):
        result = _substitute_word_boundary(result, variant, "COMPANY_X")
    result = _substitute_word_boundary(result, ticker, "TICKER_X")
    for peer_id, peer_name in peer_names.items():
        for variant in _name_variants(peer_name):
            result = _substitute_word_boundary(result, variant, f"{peer_id}_COMPANY")
    return result


async def build_research_sources(
    ticker: str, id_assigned_articles: list[dict], stock: StockLike | None = None
) -> ResearchSourcesBundle:
    """Top-level assembly - calls every builder above plus the three from
    2a-ii and wires the result into one validator-satisfying
    ResearchSourcesBundle. Not all builders are independent the way
    2a-ii's three were: build_filing_digests must run before
    compute_cik_verified, since the latter's plain-US branch takes
    build_filing_digests's own result as has_filing_digest rather than
    re-deriving it (redundant re-fetch/re-summarization otherwise).
    build_peer_blocks must run before anonymization, since anonymizing
    peer content needs each peer's real name from build_peer_blocks's
    own widened return.

    `id_assigned_articles` is required, not fetched here, for the same
    reason build_news_items() itself never fetches or assigns IDs (see
    the module docstring's news-item-sourcing section): the caller must
    supply the same already-ID-assigned list precompute/sentiment.py
    receives, so both agents cite the same article under the same
    N{num}. This is the one 2a-ii contract that can't be silently
    defaulted to empty here - unlike transcript_excerpts (D3,
    permanently out of scope, always []), a bundle with news_items
    hardcoded empty would misreport real news coverage.

    Fetches the main ticker's own real name via Router.get_company_info()
    for anonymization - a fetch none of 2a-ii's functions needed, since
    company_info is otherwise a top-level DataBundle field (per this
    module's original docstring), not part of ResearchSourcesBundle
    itself. DataPipeline.prepare() doesn't exist yet to share an
    already-fetched copy, so this is a real, disclosed, possibly-
    redundant-later fetch - the same "no cache yet" posture already
    accepted for filing digests. A failure to resolve the main ticker's
    name is logged, not raised: unlike a peer (optional, best-effort),
    the main ticker is the entire subject of the analysis and will have
    a resolvable name for any real, already-vetted ticker - this is a
    near-unreachable edge case in practice, not a peer-style routine one.

    missing_sources_list only ever names filing_digests/peer_blocks/
    news_items - the three list fields that can be meaningfully "empty".
    management_signals is a struct, never comparably empty: c_suite being
    None is a deliberate design choice, not a fetch failure, and
    buyback/dividend/insider-direction landing on none/neutral share the
    same disclosed fetch-failure-vs-confirmed-zero ambiguity Router's own
    _try_chain leaves unresolved for every non-401 failure (tracked
    separately, ClickUp 86bbzgapt) - there is no reliable signal here to
    report as "missing" either.

    Concurrency: compute_cik_verified depends on build_filing_digests's
    result (has_filing_digest), and compute_dual_class_flag depends on
    company_name_task's result (the share-class-qualifier signal). Peer
    blocks and management signals have no dependency on either background
    task, so build_filing_digests (which makes a real local-LLM
    summarization call, the slowest single step in this function) still
    runs as a background task alongside them rather than blocking them -
    confirmed live this saves real wall-clock time, not just a
    theoretical optimization. compute_dual_class_flag itself is a cheap,
    pure sync string/regex check (no I/O of its own), so waiting on
    company_name_task before calling it directly - not via
    asyncio.to_thread, which existed only to fit the old all-independent
    asyncio.gather shape, not because the work needed a thread - costs
    nothing measurable.

    token_count is recomputed for each anonymized FilingDigest/PeerBlock,
    not carried over from the pre-anonymization digest/block - substitution
    isn't length-neutral ("Apple" -> "COMPANY_X" is +4 chars per
    occurrence, "Royal Bank of Canada" -> "COMPANY_X" is -11), so the
    pre-anonymization count would silently drift from what the final
    content actually is. The two fields get different recompute strategies
    because they start from different accuracy baselines:
      - Filing digests already carry a REAL LLM-reported token_count
        (Ollama's own eval_count, via filing_summarizer.py) - checked live
        before picking a strategy here: a flat len//4 approximation
        applied to that same real content overstated RY.TO's real 233-token
        digest as 276 (this specific document tokenizes at ~4.74
        chars/token, not the flat 4 len//4 assumes - an 18% overstatement,
        not a rounding difference). Recomputing from scratch with len//4
        would have thrown away real accuracy the LLM already gave us.
        Instead, scale the real count by how much anonymization changed
        the content's length: new_count = round(old_count *
        len(new_content) / len(old_content)) - anchors to the document's
        own real chars-per-token ratio and only adjusts for what actually
        changed, rather than replacing a real number with a worse guess.
      - Peer blocks never had a real count to begin with (no LLM call
        sources them - finding #28) - build_peer_blocks already computes
        their pre-anonymization token_count via len//4, so recomputing
        post-anonymization the same way is consistent with the number
        that was already there, not a new approximation being introduced."""
    company_name_task = asyncio.create_task(_fetch_company_name(ticker))
    digests_task = asyncio.create_task(build_filing_digests(ticker, stock=stock))

    peer_pairs, management_signals = await asyncio.gather(
        build_peer_blocks(ticker),
        build_management_signals(ticker),
    )

    company_name = await company_name_task or ""
    if not company_name:
        logger.warning("research_sources_no_company_name", ticker=ticker)

    dual_class_flag = compute_dual_class_flag(ticker, company_name)

    digests, latest_filing_date = await digests_task
    has_filing_digest = len(digests) > 0
    cik_verified = await compute_cik_verified(ticker, has_filing_digest, stock=stock)
    news_items = build_news_items(id_assigned_articles)

    peer_names = {block.peer_id: name for block, name in peer_pairs}
    anonymized_digests = []
    for digest in digests:
        content = anonymize_content(digest.content, company_name, ticker, peer_names)
        # len(digest.content) is never 0 here - every digest in `digests` came from
        # summarize_filing_section(), which explicitly returns None (filtered out
        # before this point, never constructs a FilingDigest) for an empty result -
        # confirmed by direct read of that function, not assumed, since this division
        # would otherwise raise ZeroDivisionError.
        scaled_token_count = round(digest.token_count * len(content) / len(digest.content))
        anonymized_digests.append(
            FilingDigest(
                section=digest.section, content=content, token_count=max(1, scaled_token_count)
            )
        )
    anonymized_peer_blocks = []
    for block, _ in peer_pairs:
        content = anonymize_content(block.content, company_name, ticker, peer_names)
        anonymized_peer_blocks.append(
            PeerBlock(peer_id=block.peer_id, content=content, token_count=max(1, len(content) // 4))
        )

    missing_sources_list = []
    if not anonymized_digests:
        missing_sources_list.append("filing_digests")
    if not anonymized_peer_blocks:
        missing_sources_list.append("peer_blocks")
    if not news_items:
        missing_sources_list.append("news_items")

    latest_filing_age_days = (
        (datetime.now().date() - datetime.fromisoformat(latest_filing_date).date()).days
        if latest_filing_date
        else None
    )
    latest_news_age_days = (
        (datetime.now() - max(item.date for item in news_items)).days if news_items else None
    )

    return ResearchSourcesBundle(
        filing_digests=anonymized_digests,
        transcript_excerpts=[],
        news_items=news_items,
        peer_blocks=anonymized_peer_blocks,
        management_signals=management_signals,
        dual_class_flag=dual_class_flag,
        cik_verified=cik_verified,
        sedar_filing_available=has_filing_digest,
        missing_sources_list=missing_sources_list,
        latest_filing_age_days=latest_filing_age_days,
        latest_transcript_age_days=None,
        latest_news_age_days=latest_news_age_days,
        transcript_count=0,
        news_item_count=len(news_items),
    )
