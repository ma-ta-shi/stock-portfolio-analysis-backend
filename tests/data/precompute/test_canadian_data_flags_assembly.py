from data.precompute.canadian_data_flags import build_canadian_data_flags
from data.schemas.common import NewsItem
from data.schemas.macro_sources_bundle import MacroSourcesBundle
from data.schemas.research_sources_bundle import ManagementSignals, ResearchSourcesBundle


def _news_item(item_id: str, source: str) -> NewsItem:
    return NewsItem(
        id=item_id,
        date="2026-08-01T09:00:00",
        headline="Headline",
        source=source,
        quality_tier="primary",
    )


def _research_bundle(
    news_items: list[NewsItem], sedar_filing_available: bool = False
) -> ResearchSourcesBundle:
    """Minimal valid ResearchSourcesBundle - only the fields
    build_canadian_data_flags() actually reads vary by test, everything
    else is a plausible constant matching this schema's own established
    test fixture shape (tests/data/schemas/test_research_sources_bundle.py)."""
    return ResearchSourcesBundle(
        filing_digests=[],
        transcript_excerpts=[],
        news_items=news_items,
        peer_blocks=[],
        peer_names={},
        management_signals=ManagementSignals(
            c_suite_changes_12mo=None,
            changes_detail="",
            insider_net_direction_90d=None,
            buyback_activity="",
            dividend_activity="",
        ),
        dual_class_flag=False,
        cik_verified=False,
        sedar_filing_available=sedar_filing_available,
        missing_sources_list=[],
        latest_filing_age_days=None,
        latest_transcript_age_days=None,
        latest_news_age_days=1 if news_items else None,
        transcript_count=0,
        news_item_count=len(news_items),
    )


def _macro_bundle(statcan_age_days: int | None) -> MacroSourcesBundle:
    """Minimal valid MacroSourcesBundle - only statcan_age_days varies by
    test; every other field is None/a plausible constant satisfying the
    schema's own cross-field validators (mirrors
    tests/data/schemas/test_macro_sources_bundle.py's own fixture)."""
    return MacroSourcesBundle(
        fed_funds_rate=None,
        treasury_2y=None,
        treasury_5y=None,
        treasury_10y=None,
        cpi=None,
        core_cpi=None,
        gdp=None,
        unemployment=None,
        vix=None,
        cad_usd_fred=None,
        wti_crude=None,
        us_cpi_yoy=None,
        us_core_cpi_yoy=None,
        us_gdp_qoq=None,
        us_gdp_4q_trend=None,
        vix_30d_avg=None,
        boc_rate=None,
        cad_usd=None,
        canada_bond_2y=None,
        canada_bond_5y=None,
        canada_bond_10y=None,
        canada_cpi=None,
        policy_rate_90d_delta_bp=None,
        cpi_3m_delta_pp=None,
        cad_usd_90d_change_pct=None,
        commodity_90d_change_pct=None,
        unemployment_6m_delta=None,
        boc_rate_90d_delta_bp=None,
        rate_trend=None,
        boc_rate_trend=None,
        cpi_trend=None,
        cad_trend=None,
        vix_regime=None,
        us_curve_shape=None,
        ca_curve_shape=None,
        sector_commodity_relevant=False,
        sector_commodity_name=None,
        sector_commodity_level=None,
        sector_commodity_direction=None,
        sector_commodity_age_days=None,
        bond_yields_available=False,
        usd_revenue_exposure_pct=None,
        cb_commentary_items=[],
        cb_commentary_count=0,
        cb_stance_note=None,
        policy_rate_age_days=None,
        cpi_age_days=None,
        gdp_age_days=None,
        unemployment_age_days=None,
        cad_usd_age_days=None,
        vix_age_days=None,
        statcan_unemployment_ca=None,
        ca_unemployment_6m_delta=None,
        statcan_housing_starts=None,
        statcan_retail_sales_yoy=None,
        statcan_age_days=statcan_age_days,
        ca_cpi_yoy=None,
        ca_cpi_3m_delta=None,
        ca_cpi_trend=None,
        ca_gdp_qoq=None,
        ca_gdp_4q_trend=None,
    )


# ---------- analyst_count ----------
# Real shape (NormalizedAnalystRatings, 86bbpgrxh): buy_count/hold_count/
# sell_count, built the same way by both CA_CHAINS links (openbb_tmx first,
# yfinance fallback). Real bug found live 86bawptye: this used to read a
# stale "rating_breakdown" key that no longer exists anywhere in the
# codebase, silently returning 0 for every Canadian stock.


def test_analyst_count_sums_buy_hold_sell_counts():
    """Same real RY.TO total (15) as before the fix, expressed in the real
    buy_count/hold_count/sell_count shape - strong_buy(4)+buy(5)=9,
    hold=5, sell(0)+strong_sell(1)=1, matching both adapters' own folding
    of strong_buy/strong_sell into buy_count/sell_count."""
    consensus = {
        "symbol": "RY.TO",
        "current_price": 285.2,
        "consensus_rating": "buy",
        "buy_count": 9,
        "hold_count": 5,
        "sell_count": 1,
    }
    flags = build_canadian_data_flags(_research_bundle([]), _macro_bundle(None), consensus)
    assert flags.analyst_count == 15


def test_analyst_count_sums_available_counts_when_some_are_missing():
    """Real, live-reachable partial case: openbb_tmx.py's get_analyst_ratings()
    early-return guard only checks target_mean/num_analysts/buy_count/rating,
    not hold_count/sell_count - so a real TMX row with buy_ratings set but
    hold_ratings/sell_ratings missing returns buy_count=5, hold_count=None,
    sell_count=None, not {}. Sum what's there, treat missing as 0."""
    consensus = {"buy_count": 5, "hold_count": None, "sell_count": None}
    flags = build_canadian_data_flags(_research_bundle([]), _macro_bundle(None), consensus)
    assert flags.analyst_count == 5


def test_analyst_count_zero_when_consensus_empty():
    flags = build_canadian_data_flags(_research_bundle([]), _macro_bundle(None), {})
    assert flags.analyst_count == 0


# ---------- news_article_count / news_sources ----------


def test_news_article_count_and_sources_from_research_bundle():
    items = [_news_item("N1", "Reuters"), _news_item("N2", "Benzinga"), _news_item("N3", "Reuters")]
    flags = build_canadian_data_flags(_research_bundle(items), _macro_bundle(None), {})
    assert flags.news_article_count == 3
    assert flags.news_sources == ["Benzinga", "Reuters"]  # sorted, deduplicated


def test_news_article_count_zero_with_no_items():
    flags = build_canadian_data_flags(_research_bundle([]), _macro_bundle(None), {})
    assert flags.news_article_count == 0
    assert flags.news_sources == []


# ---------- sedar_filing_available ----------


def test_sedar_filing_available_passes_through_true():
    flags = build_canadian_data_flags(
        _research_bundle([], sedar_filing_available=True), _macro_bundle(None), {}
    )
    assert flags.sedar_filing_available is True


def test_sedar_filing_available_passes_through_false():
    flags = build_canadian_data_flags(
        _research_bundle([], sedar_filing_available=False), _macro_bundle(None), {}
    )
    assert flags.sedar_filing_available is False


# ---------- statcan_available ----------


def test_statcan_available_true_when_age_days_present():
    flags = build_canadian_data_flags(_research_bundle([]), _macro_bundle(8), {})
    assert flags.statcan_available is True


def test_statcan_available_false_when_age_days_none():
    flags = build_canadian_data_flags(_research_bundle([]), _macro_bundle(None), {})
    assert flags.statcan_available is False


# ---------- real end-to-end shape, from the exact live-confirmed RY.TO values ----------


def test_full_build_matches_real_ry_to_live_verification():
    """The exact real values confirmed live this session: Router.get_analyst_ratings("RY.TO")
    -> buy_count/hold_count/sell_count -> analyst_count=15; build_research_sources("RY.TO") ->
    news_article_count=155 with 7 real distinct sources, sedar_filing_available=True;
    compute_macro_sources(is_canadian_stock=True) -> statcan_age_days=8. Reconstructed as a
    fixture here (not a live call, per this test's own non-live scope) to pin the exact
    real-world shape rather than only a synthetic one."""
    consensus = {"buy_count": 9, "hold_count": 5, "sell_count": 1}
    sources = [
        "Benzinga",
        "Canada Newswire via QuoteMedia",
        "ChartMill",
        "Fintel",
        "PR Newswire via QuoteMedia",
        "SeekingAlpha",
        "Yahoo",
    ]
    items = [_news_item(f"N{i}", src) for i, src in enumerate(sources)]
    flags = build_canadian_data_flags(
        _research_bundle(items, sedar_filing_available=True), _macro_bundle(8), consensus
    )
    assert flags.analyst_count == 15
    assert flags.news_article_count == 7
    assert flags.news_sources == sorted(sources)
    assert flags.sedar_filing_available is True
    assert flags.statcan_available is True
