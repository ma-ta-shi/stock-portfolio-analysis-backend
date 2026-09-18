"""canadian_data_flags.py — assembly for CanadianDataFlags (ClickUp 86bawptx4).

Genuinely new work: CanadianDataFlags is never constructed anywhere else in
this codebase - its type shipped (Data Contracts epic, 86bawp80f) but nothing
builds a real instance. This module is that builder, standalone until
DataPipeline.prepare() exists to call it - same "ships before the
orchestrator" precedent as tax_metrics.py and research_sources.py.

Only ever called for a CA/TSX-listed stock. Deliberately takes no position
on that itself and always returns a fully-populated CanadianDataFlags - the
None-for-US decision belongs to the caller, which already has to know the
stock's market to decide whether to call this at all. Not reusing
data_bundle.py's _is_canadian_stock() here even though it does exactly this
check - it's a private, underscore-prefixed function, and this codebase has
an established convention against importing private symbols across module
boundaries (research_sources.py/tax_metrics.py both duplicate the small
CA-suffix check locally rather than reach into a private import).
"""

from data.schemas.canadian_data_flags import CanadianDataFlags
from data.schemas.macro_sources_bundle import MacroSourcesBundle
from data.schemas.research_sources_bundle import ResearchSourcesBundle


def build_canadian_data_flags(
    research_sources: ResearchSourcesBundle,
    macro_sources: MacroSourcesBundle,
    analyst_consensus: dict,
) -> CanadianDataFlags:
    """Pure function, no I/O - all three inputs are already-fetched/-built
    by the time this is called (ResearchSourcesBundle and MacroSourcesBundle
    are themselves full precompute outputs; analyst_consensus is the same
    raw dict a caller also assigns to DataBundle.analyst_consensus).

    analyst_count: analyst_consensus is Router.get_analyst_ratings()'s real
    shape for a CA ticker (NormalizedAnalystRatings, 86bbpgrxh). CA_CHAINS
    routes this method to openbb_tmx first (the richer source - target plus
    a full rating breakdown in one call), yfinance second as fallback; both
    adapters build the same shape. Real fields are buy_count/hold_count/
    sell_count (int | None each, strong_buy folded into buy_count and
    strong_sell into sell_count by both adapters) - not "rating_breakdown",
    a pre-normalization key that no longer exists anywhere in the codebase
    (real bug found live 86bawptye: this function read that stale key and
    silently returned 0 for every Canadian stock). Sum the three counts,
    treating a missing/None field as 0 - covers both the fully-empty {} case
    (no data at all) and openbb_tmx's own partial case (target/rating
    present but one of the three counts missing), which its own early-return
    guard doesn't catch.

    news_article_count/news_sources: from the already-assembled
    ResearchSourcesBundle, not a separate fetch - the original ticket's
    "openbb-tmx + RSS merge" source is stale (the RSS fallback doesn't
    exist). Confirmed live for RY.TO: 155 items, 7 distinct real sources.

    sedar_filing_available: copied directly from
    ResearchSourcesBundle.sedar_filing_available, not re-derived - both
    fields mean the same real fact (research_sources.py's own
    has_filing_digest), and computing them independently risks the two
    silently disagreeing. Confirmed live True for RY.TO.

    statcan_available: MacroSourcesBundle.statcan_age_days is already None
    exactly when the StatCan fetch failed or the stock isn't Canadian, per
    that schema's own convention. Confirmed live: statcan_age_days=8 for
    RY.TO/SHOP.TO/WELL.TO - the fetch genuinely succeeds today, not a
    field that's permanently None in practice.

    No sentiment_source parameter - deliberately removed from
    CanadianDataFlags entirely (86bawptx4), see that model's own docstring."""
    analyst_count = (
        (analyst_consensus.get("buy_count") or 0)
        + (analyst_consensus.get("hold_count") or 0)
        + (analyst_consensus.get("sell_count") or 0)
    )
    return CanadianDataFlags(
        analyst_count=analyst_count,
        news_article_count=research_sources.news_item_count,
        news_sources=sorted({item.source for item in research_sources.news_items}),
        sedar_filing_available=research_sources.sedar_filing_available,
        statcan_available=macro_sources.statcan_age_days is not None,
    )
