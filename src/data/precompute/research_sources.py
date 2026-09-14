"""research_sources.py — sourcing for the Stock Researcher's ResearchSourcesBundle
(ClickUp 86ban0x1u, this slice tracked as 86ban0x1u/2a-ii).

Builds the three sourcing legs of the bundle: filing-section digests
(FILING:Business/FILING:MDA citations), news items (shared N{num} namespace
with precompute/sentiment.py), and peer blocks (PEER_n citations). Every
citation token the live Stock Researcher Agent Prompt.md can use has to
trace back to something one of these functions actually put in the
payload, or the LLM either invents a plausible-sounding claim from
training data or (correctly, per the prompt's own rules) says
"insufficient_data" for everything - the whole reason this module exists.

Deliberately NOT built here (Slice 2b, a separate PR): management_signals,
entity anonymization, cik_verified/dual_class_flag, sedar_filing_available,
the mechanical counter fields (latest_filing_age_days etc.), and the
top-level function that assembles a full, validator-satisfying
ResearchSourcesBundle. That assembly needs management_signals before a
valid bundle can even be constructed (every field is required), so it
can't live here yet. These three functions are independently testable and
live-verifiable now; Slice 2b wires them together.

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

import aiohttp
import structlog

from data.precompute.filing_summarizer import summarize_filing_section
from data.providers.ca_crosslisting import (
    get_crosslisted_business_overview,
    get_crosslisted_mda,
    is_crosslisted,
)
from data.providers.edgartools import EdgarToolsDataProvider
from data.providers.router import Router, StockLike, is_canadian
from data.schemas.common import FilingDigest, NewsItem, PeerBlock

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


async def build_peer_blocks(
    ticker: str, limit: int = _PEER_LIMIT, news_days: int = _PEER_NEWS_DAYS
) -> list[PeerBlock]:
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

    Degrades to [] when get_peers() returns nothing, which it already
    guarantees never raises."""
    async with Router(ticker=ticker) as router:
        peer_tickers = await router.get_peers(ticker, limit=limit)
    if not peer_tickers:
        return []

    contents = await asyncio.gather(
        *(_fetch_peer_content(peer_ticker, news_days) for peer_ticker in peer_tickers)
    )

    blocks: list[PeerBlock] = []
    for peer_ticker, content in zip(peer_tickers, contents, strict=True):
        if content is None:
            logger.info("peer_block_dropped_no_data", ticker=ticker, peer_ticker=peer_ticker)
            continue
        blocks.append(
            PeerBlock(
                peer_id=f"PEER_{len(blocks) + 1}",
                content=content,
                token_count=max(1, len(content) // 4),
            )
        )
    return blocks
