from datetime import datetime, timedelta

import structlog

from data.precompute.research_sources import (
    _compute_buyback_activity,
    _compute_dividend_activity,
    _compute_insider_direction,
    _derive_short_name,
    _render_peer_content,
    _truncate_to_tokens,
    anonymize_content,
    build_filing_digests,
    build_management_signals,
    build_news_items,
    build_peer_blocks,
    build_research_sources,
    compute_cik_verified,
    compute_dual_class_flag,
)
from data.schemas.common import FilingDigest, NewsItem
from data.schemas.research_sources_bundle import ResearchSourcesBundle


def _section(text: str = "raw text", filing_date: str = "2026-01-01", form: str = "10-K") -> dict:
    return {"text": text, "accession_no": "0001-1", "filing_date": filing_date, "source_form": form}


def _digest(section: str, content: str = "a digest") -> FilingDigest:
    return FilingDigest(section=section, content=content, token_count=10)


# --- build_filing_digests: path selection ---


async def test_crosslisted_ticker_uses_ca_crosslisting_path(monkeypatch):
    calls = []
    monkeypatch.setattr("data.precompute.research_sources.is_crosslisted", lambda t: True)

    async def fake_business_overview(ticker):
        calls.append("business")
        return _section("biz text", "2026-03-01")

    async def fake_mda(ticker):
        calls.append("mda")
        return _section("mda text", "2026-02-01")

    monkeypatch.setattr(
        "data.precompute.research_sources.get_crosslisted_business_overview", fake_business_overview
    )
    monkeypatch.setattr("data.precompute.research_sources.get_crosslisted_mda", fake_mda)

    async def fake_summarize(session, text, section):
        return _digest(section, text)

    monkeypatch.setattr("data.precompute.research_sources.summarize_filing_section", fake_summarize)

    digests, latest_date = await build_filing_digests("RY.TO")

    assert set(calls) == {"business", "mda"}
    assert {d.section for d in digests} == {"Business", "MDA"}
    assert latest_date == "2026-03-01"  # the freshest of the two, not just the last one


async def test_non_crosslisted_non_canadian_uses_edgartools_path(monkeypatch):
    monkeypatch.setattr("data.precompute.research_sources.is_crosslisted", lambda t: False)
    monkeypatch.setattr("data.precompute.research_sources.is_canadian", lambda stock, ticker: False)

    class _FakeEdgarProvider:
        async def get_filing_section(self, ticker, section):
            return _section(f"{section} text", "2026-04-01")

    monkeypatch.setattr(
        "data.precompute.research_sources.EdgarToolsDataProvider", _FakeEdgarProvider
    )

    async def fake_summarize(session, text, section):
        return _digest(section, text)

    monkeypatch.setattr("data.precompute.research_sources.summarize_filing_section", fake_summarize)

    digests, latest_date = await build_filing_digests("AAPL")

    assert {d.section for d in digests} == {"Business", "MDA"}
    assert latest_date == "2026-04-01"


async def test_pure_unmapped_ca_ticker_has_no_filing_source(monkeypatch):
    monkeypatch.setattr("data.precompute.research_sources.is_crosslisted", lambda t: False)
    monkeypatch.setattr("data.precompute.research_sources.is_canadian", lambda stock, ticker: True)
    calls = []

    async def fake_summarize(session, text, section):
        calls.append("called")
        return _digest(section, text)

    monkeypatch.setattr("data.precompute.research_sources.summarize_filing_section", fake_summarize)

    digests, latest_date = await build_filing_digests("WELL.TO")

    assert digests == []
    assert latest_date is None
    assert calls == []  # never even tried to summarize - no source exists


async def test_stock_param_forwarded_to_is_canadian(monkeypatch):
    monkeypatch.setattr("data.precompute.research_sources.is_crosslisted", lambda t: False)
    received = {}

    def fake_is_canadian(stock, ticker):
        received["stock"] = stock
        received["ticker"] = ticker
        return True

    monkeypatch.setattr("data.precompute.research_sources.is_canadian", fake_is_canadian)

    sentinel_stock = object()
    await build_filing_digests("WELL.TO", stock=sentinel_stock)

    assert received["stock"] is sentinel_stock
    assert received["ticker"] == "WELL.TO"


# --- build_filing_digests: both sections missing / summarization failures ---


async def test_both_sections_missing_returns_empty_without_summarizing(monkeypatch):
    monkeypatch.setattr("data.precompute.research_sources.is_crosslisted", lambda t: False)
    monkeypatch.setattr("data.precompute.research_sources.is_canadian", lambda stock, ticker: False)

    class _FakeEdgarProvider:
        async def get_filing_section(self, ticker, section):
            return None

    monkeypatch.setattr(
        "data.precompute.research_sources.EdgarToolsDataProvider", _FakeEdgarProvider
    )
    calls = []

    async def fake_summarize(session, text, section):
        calls.append("called")
        return _digest(section, text)

    monkeypatch.setattr("data.precompute.research_sources.summarize_filing_section", fake_summarize)

    digests, latest_date = await build_filing_digests("ZZZZ")

    assert digests == []
    assert latest_date is None
    assert calls == []


async def test_summarization_failure_excluded_from_survivors_and_date(monkeypatch):
    """The load-bearing case for build_filing_digests's date contract:
    MDA's raw section has the newer filing_date, but its summarization
    fails. The surviving Business digest's older date must be what's
    returned - not MDA's, even though MDA's date is fresher - because
    MDA never became a digest."""
    monkeypatch.setattr("data.precompute.research_sources.is_crosslisted", lambda t: False)
    monkeypatch.setattr("data.precompute.research_sources.is_canadian", lambda stock, ticker: False)

    sections = {
        "Business": _section("business text", "2026-01-01"),
        "MDA": _section("mda text", "2026-06-01"),  # newer, but will fail to summarize
    }

    class _FakeEdgarProvider:
        async def get_filing_section(self, ticker, section):
            return sections[section]

    monkeypatch.setattr(
        "data.precompute.research_sources.EdgarToolsDataProvider", _FakeEdgarProvider
    )

    async def fake_summarize(session, text, section):
        return None if section == "MDA" else _digest(section, text)

    monkeypatch.setattr("data.precompute.research_sources.summarize_filing_section", fake_summarize)

    digests, latest_date = await build_filing_digests("AAPL")

    assert len(digests) == 1
    assert digests[0].section == "Business"
    assert latest_date == "2026-01-01"  # Business's date, not MDA's fresher-but-failed one


async def test_all_summarizations_fail_returns_empty(monkeypatch):
    monkeypatch.setattr("data.precompute.research_sources.is_crosslisted", lambda t: False)
    monkeypatch.setattr("data.precompute.research_sources.is_canadian", lambda stock, ticker: False)

    class _FakeEdgarProvider:
        async def get_filing_section(self, ticker, section):
            return _section(f"{section} text")

    monkeypatch.setattr(
        "data.precompute.research_sources.EdgarToolsDataProvider", _FakeEdgarProvider
    )

    async def fake_summarize(session, text, section):
        return None

    monkeypatch.setattr("data.precompute.research_sources.summarize_filing_section", fake_summarize)

    digests, latest_date = await build_filing_digests("AAPL")

    assert digests == []
    assert latest_date is None


async def test_one_raw_section_missing_other_still_summarized(monkeypatch):
    """A real, confirmed-live case (a mapped CA ticker with no MD&A source
    for a given filing), not hypothetical: only one of the two raw
    sections comes back at all - the other must still produce a real
    digest, not cause the whole call to degrade to empty."""
    monkeypatch.setattr("data.precompute.research_sources.is_crosslisted", lambda t: False)
    monkeypatch.setattr("data.precompute.research_sources.is_canadian", lambda stock, ticker: False)

    class _FakeEdgarProvider:
        async def get_filing_section(self, ticker, section):
            return _section("business text", "2026-01-01") if section == "Business" else None

    monkeypatch.setattr(
        "data.precompute.research_sources.EdgarToolsDataProvider", _FakeEdgarProvider
    )
    calls = []

    async def fake_summarize(session, text, section):
        calls.append(section)
        return _digest(section, text)

    monkeypatch.setattr("data.precompute.research_sources.summarize_filing_section", fake_summarize)

    digests, latest_date = await build_filing_digests("SOME.TO")

    assert calls == ["Business"]  # MDA was never retrieved, so never even attempted
    assert len(digests) == 1
    assert digests[0].section == "Business"
    assert latest_date == "2026-01-01"


# --- build_news_items ---


def test_maps_id_assigned_articles_dropping_text_key():
    article = {
        "id": "N1",
        "date": datetime(2026, 9, 1),
        "headline": "Real headline",
        "source": "Reuters",
        "quality_tier": "primary",
        "text": "article body sentiment.py needs, not part of NewsItem",
    }

    result = build_news_items([article])

    assert len(result) == 1
    item = result[0]
    assert isinstance(item, NewsItem)
    assert item.id == "N1"
    assert item.headline == "Real headline"
    assert item.source == "Reuters"
    assert item.quality_tier == "primary"
    assert not hasattr(item, "text")


def test_preserves_all_quality_tiers_including_low():
    articles = [
        {
            "id": f"N{i}",
            "date": datetime(2026, 9, i),
            "headline": "h",
            "source": "s",
            "quality_tier": tier,
        }
        for i, tier in enumerate(("primary", "secondary", "low"), start=1)
    ]

    result = build_news_items(articles)

    assert [item.quality_tier for item in result] == ["primary", "secondary", "low"]


def test_empty_input_returns_empty_list():
    assert build_news_items([]) == []


# --- _truncate_to_tokens ---


def test_truncate_to_tokens_passthrough_when_short():
    text = "A short sentence."
    assert _truncate_to_tokens(text, 50) == text


def test_truncate_to_tokens_cuts_on_sentence_boundary():
    first = "A short leading sentence."
    second = " " + ("More filler content padding this out further. " * 10)
    result = _truncate_to_tokens(first + second, 10)
    assert result.endswith(".")
    assert len(result) <= 40
    assert result.startswith("A short leading sentence.")


def test_truncate_to_tokens_zero_budget_degrades_to_empty():
    assert _truncate_to_tokens("anything", 0) == ""


def test_truncate_to_tokens_ignores_early_abbreviation_period():
    """Real bug caught against real data: a genuine yfinance business
    summary opening "Dell Technologies Inc. designs, develops..."
    truncated to just "Dell Technologies Inc." at a 50-token budget -
    "Inc. " was the only ". " anywhere in the 200-char window, so the
    naive last-sentence-boundary rule mistook the corporate-suffix
    abbreviation for a real sentence end. The result must keep
    substantially more than just the company name."""
    summary = (
        "Dell Technologies Inc. designs, develops, manufactures, markets, "
        "sells, and supports various comprehensive and integrated "
        "solutions, products, and services in the Americas, Europe, the "
        "Middle East, Asia, and internationally. The company operates "
        "through Infrastructure Solutions Group and Client Solutions "
        "Group segments."
    )
    result = _truncate_to_tokens(summary, 50)
    assert result != "Dell Technologies Inc."
    assert len(result) > 100
    assert "designs, develops" in result


# --- _render_peer_content ---


def test_render_peer_content_both_pieces_present():
    content = _render_peer_content(
        "A leading widget maker.",
        [{"headline": "Widgets up", "published_at": "2026-09-01 00:00:00"}],
    )
    assert content is not None
    assert "Business: A leading widget maker." in content
    assert "Recent news: Widgets up" in content


def test_render_peer_content_no_summary_no_news_returns_none():
    assert _render_peer_content(None, []) is None
    assert _render_peer_content("", []) is None


def test_render_peer_content_summary_only():
    content = _render_peer_content("A leading widget maker.", [])
    assert content == "Business: A leading widget maker."


def test_render_peer_content_news_only():
    content = _render_peer_content(
        None, [{"headline": "Widgets up", "published_at": "2026-09-01 00:00:00"}]
    )
    assert content == "Recent news: Widgets up"


def test_render_peer_content_sorts_news_and_takes_top_two():
    news = [
        {"headline": "Oldest", "published_at": "2026-01-01 00:00:00"},
        {"headline": "Newest", "published_at": "2026-09-01 00:00:00"},
        {"headline": "Middle", "published_at": "2026-05-01 00:00:00"},
    ]
    content = _render_peer_content(None, news)
    assert "Newest" in content
    assert "Middle" in content
    assert "Oldest" not in content


def test_render_peer_content_dedupes_repeated_headline():
    """Real bug caught against real data: Router.get_news() merges a
    crosslisted CA ticker's TMX feed with its matching Finnhub feed and
    dedupes only on URL - the same real story reaches both feeds under
    two different URLs and survives as two entries with an identical
    headline (confirmed live on TD.TO and RY.TO, ~15-16% of 180-day news
    volume). The top-2 "recent news" selection must not waste both slots
    on the same story - it should fall through to the next distinct one."""
    news = [
        {"headline": "Same story", "published_at": "2026-09-01 07:00:00"},
        {"headline": "Same story", "published_at": "2026-09-01 07:00:00"},
        {"headline": "Different story", "published_at": "2026-08-01 00:00:00"},
    ]
    content = _render_peer_content(None, news)
    assert content.count("Same story") == 1
    assert "Different story" in content


def test_render_peer_content_stays_within_whole_block_budget():
    """Regression test for a real bug caught during implementation review:
    the "Recent news: " label's own token cost wasn't subtracted before
    truncating headlines to fit, so a worst-case business summary +
    headlines combination rendered ~204 tokens against a 200-token budget.
    A maxed-out 50-token business summary plus two very long headlines
    must still land at or under 200 tokens once rendered."""
    summary = "x" * 200  # ~50 tokens, fills its own fixed budget exactly
    news = [
        {"headline": "y" * 300, "published_at": "2026-09-02 00:00:00"},
        {"headline": "z" * 300, "published_at": "2026-09-01 00:00:00"},
    ]
    content = _render_peer_content(summary, news)
    assert len(content) // 4 <= 200


# --- build_peer_blocks ---


def _fake_router_factory(peers=None, summaries=None, news=None, names=None):
    peers = peers or {}
    summaries = summaries or {}
    news = news or {}
    names = names or {}
    calls = {"get_peers": [], "get_news": [], "get_company_info": []}

    class _FakeRouter:
        def __init__(self, ticker):
            self.ticker = ticker

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get_peers(self, ticker, limit):
            calls["get_peers"].append((ticker, limit))
            return peers.get(ticker, [])[:limit]

        async def get_business_summary(self, ticker):
            return summaries.get(ticker)

        async def get_news(self, ticker, days):
            calls["get_news"].append((ticker, days))
            return news.get(ticker, [])

        async def get_company_info(self, ticker):
            calls["get_company_info"].append(ticker)
            name = names.get(ticker)
            return {"name": name} if name else {}

    return _FakeRouter, calls


async def test_no_peers_returns_empty_list(monkeypatch):
    fake_cls, _ = _fake_router_factory(peers={"AAPL": []})
    monkeypatch.setattr("data.precompute.research_sources.Router", fake_cls)

    assert await build_peer_blocks("AAPL") == []


async def test_renders_and_numbers_peer_blocks(monkeypatch):
    fake_cls, calls = _fake_router_factory(
        peers={"AAPL": ["MSFT", "GOOG"]},
        summaries={"MSFT": "Microsoft makes software.", "GOOG": "Google makes search."},
        news={
            "MSFT": [{"headline": "MSFT news", "published_at": "2026-09-01 00:00:00"}],
            "GOOG": [{"headline": "GOOG news", "published_at": "2026-09-01 00:00:00"}],
        },
        names={"MSFT": "Microsoft Corporation", "GOOG": "Alphabet Inc."},
    )
    monkeypatch.setattr("data.precompute.research_sources.Router", fake_cls)

    result = await build_peer_blocks("AAPL")

    assert [(b.peer_id, name) for b, name in result] == [
        ("PEER_1", "Microsoft Corporation"),
        ("PEER_2", "Alphabet Inc."),
    ]
    assert "Microsoft makes software." in result[0][0].content
    assert "MSFT news" in result[0][0].content
    assert "Google makes search." in result[1][0].content
    assert calls["get_peers"] == [("AAPL", 2)]
    assert ("MSFT", 30) in calls["get_news"]
    assert ("GOOG", 30) in calls["get_news"]
    assert set(calls["get_company_info"]) == {"MSFT", "GOOG"}


async def test_peer_with_no_data_dropped_and_remaining_renumber(monkeypatch):
    fake_cls, _ = _fake_router_factory(
        peers={"AAPL": ["MSFT", "EMPTY", "GOOG"]},
        summaries={"MSFT": "Microsoft makes software.", "GOOG": "Google makes search."},
        news={
            "MSFT": [{"headline": "MSFT news", "published_at": "2026-09-01 00:00:00"}],
            "GOOG": [{"headline": "GOOG news", "published_at": "2026-09-01 00:00:00"}],
        },
        names={"MSFT": "Microsoft Corporation", "GOOG": "Alphabet Inc."},
    )
    monkeypatch.setattr("data.precompute.research_sources.Router", fake_cls)

    result = await build_peer_blocks("AAPL", limit=3)

    # EMPTY has no summary and no news (both default to nothing in the fakes),
    # so it must be dropped - not rendered as a hollow PEER_2 - and GOOG
    # (originally the 3rd candidate) becomes PEER_2, not PEER_3.
    assert [(b.peer_id, name) for b, name in result] == [
        ("PEER_1", "Microsoft Corporation"),
        ("PEER_2", "Alphabet Inc."),
    ]
    assert "Microsoft makes software." in result[0][0].content
    assert "Google makes search." in result[1][0].content


async def test_peer_with_content_but_no_name_dropped_and_remaining_renumber(monkeypatch):
    """finding #46: a peer that survives the content-based drop but whose
    get_company_info() name lookup itself comes back empty must still be
    dropped, not rendered with a name that can't be safely anonymized -
    the same renumbering discipline as the no-content case, but for a
    different reason and via a different code path, so it needs its own
    dedicated test rather than being folded into the one above."""
    fake_cls, _ = _fake_router_factory(
        peers={"AAPL": ["MSFT", "NONAME", "GOOG"]},
        summaries={
            "MSFT": "Microsoft makes software.",
            "NONAME": "Some business summary.",
            "GOOG": "Google makes search.",
        },
        news={
            "MSFT": [{"headline": "MSFT news", "published_at": "2026-09-01 00:00:00"}],
            "GOOG": [{"headline": "GOOG news", "published_at": "2026-09-01 00:00:00"}],
        },
        names={"MSFT": "Microsoft Corporation", "GOOG": "Alphabet Inc."},  # NONAME omitted
    )
    monkeypatch.setattr("data.precompute.research_sources.Router", fake_cls)

    result = await build_peer_blocks("AAPL", limit=3)

    assert [(b.peer_id, name) for b, name in result] == [
        ("PEER_1", "Microsoft Corporation"),
        ("PEER_2", "Alphabet Inc."),
    ]


async def test_limit_and_news_days_forwarded(monkeypatch):
    fake_cls, calls = _fake_router_factory(
        peers={"AAPL": ["MSFT"]}, summaries={"MSFT": "x"}, names={"MSFT": "MSFT Corp"}
    )
    monkeypatch.setattr("data.precompute.research_sources.Router", fake_cls)

    await build_peer_blocks("AAPL", limit=1, news_days=7)

    assert calls["get_peers"] == [("AAPL", 1)]
    assert calls["get_news"] == [("MSFT", 7)]
    assert calls["get_company_info"] == ["MSFT"]


# --- shared helpers for Slice 2b tests ---


def _days_ago(days: int) -> str:
    return (datetime.now() - timedelta(days=days)).date().isoformat()


def _insider_row(
    transaction_type="purchase",
    is_issuer=False,
    date=None,
    value=1000.0,
    shares=100.0,
):
    return {
        "date": date or _days_ago(5),
        "insider_name": "Someone",
        "is_issuer": is_issuer,
        "transaction_type": transaction_type,
        "shares": shares,
        "value": value,
    }


def _dividend_record(ex_date: str, amount: float):
    return {"ex_date": ex_date, "payment_date": None, "amount_per_share": amount}


# --- _compute_insider_direction ---


def test_insider_direction_buying_when_purchases_exceed_sales():
    rows = [
        _insider_row("purchase", value=5000, date=_days_ago(10)),
        _insider_row("sale", value=1000, date=_days_ago(20)),
    ]
    assert _compute_insider_direction(rows) == "buying"


def test_insider_direction_selling_when_sales_exceed_purchases():
    rows = [
        _insider_row("purchase", value=1000, date=_days_ago(10)),
        _insider_row("sale", value=5000, date=_days_ago(20)),
    ]
    assert _compute_insider_direction(rows) == "selling"


def test_insider_direction_neutral_when_values_equal():
    rows = [
        _insider_row("purchase", value=1000, date=_days_ago(10)),
        _insider_row("sale", value=1000, date=_days_ago(20)),
    ]
    assert _compute_insider_direction(rows) == "neutral"


def test_insider_direction_excludes_issuer_rows():
    """is_issuer=True is buyback_activity's signal, not personal insider
    direction - a huge issuer buyback must not count here."""
    rows = [_insider_row("purchase", is_issuer=True, value=100_000, date=_days_ago(5))]
    assert _compute_insider_direction(rows) is None


def test_insider_direction_excludes_non_directional_types():
    rows = [
        _insider_row("exercise", value=1000, date=_days_ago(5)),
        _insider_row("gift", value=1000, date=_days_ago(5)),
        _insider_row("other", value=1000, date=_days_ago(5)),
        _insider_row("buyback", is_issuer=True, value=1000, date=_days_ago(5)),
    ]
    assert _compute_insider_direction(rows) is None


def test_insider_direction_excludes_rows_outside_90_day_window():
    rows = [_insider_row("purchase", value=5000, date=_days_ago(120))]
    assert _compute_insider_direction(rows) is None


def test_insider_direction_none_for_empty_input():
    assert _compute_insider_direction([]) is None


def test_insider_direction_treats_missing_value_as_zero():
    rows = [
        _insider_row("purchase", value=None, date=_days_ago(5)),
        _insider_row("sale", value=100, date=_days_ago(5)),
    ]
    assert _compute_insider_direction(rows) == "selling"


# --- _compute_buyback_activity ---


def test_buyback_none_when_no_buyback_rows():
    rows = [_insider_row("purchase", value=1000, date=_days_ago(5))]
    assert _compute_buyback_activity(rows) == "none"


def test_buyback_active_when_recent_buyback_exists():
    rows = [_insider_row("buyback", is_issuer=True, value=1000, date=_days_ago(30))]
    assert _compute_buyback_activity(rows) == "active"


def test_buyback_suspended_when_only_older_buybacks_exist():
    """A real, meaningful "was doing this, may have paused" signal - not
    the same as never having done it (that's "none")."""
    rows = [_insider_row("buyback", is_issuer=True, value=1000, date=_days_ago(200))]
    assert _compute_buyback_activity(rows) == "suspended"


def test_buyback_active_when_mix_of_recent_and_older():
    rows = [
        _insider_row("buyback", is_issuer=True, value=1000, date=_days_ago(200)),
        _insider_row("buyback", is_issuer=True, value=1000, date=_days_ago(10)),
    ]
    assert _compute_buyback_activity(rows) == "active"


def test_buyback_none_for_empty_input():
    assert _compute_buyback_activity([]) == "none"


# --- _compute_dividend_activity ---


def test_dividend_none_for_empty_input():
    assert _compute_dividend_activity([]) == "none"


def test_dividend_suspended_when_history_exists_but_none_recent():
    records = [_dividend_record(_days_ago(500), 1.0)]
    assert _compute_dividend_activity(records) == "suspended"


def test_dividend_increased_when_latest_exceeds_previous():
    records = [_dividend_record(_days_ago(180), 1.0), _dividend_record(_days_ago(30), 1.2)]
    assert _compute_dividend_activity(records) == "increased"


def test_dividend_cut_when_latest_below_previous():
    records = [_dividend_record(_days_ago(180), 1.2), _dividend_record(_days_ago(30), 1.0)]
    assert _compute_dividend_activity(records) == "cut"


def test_dividend_held_when_latest_equals_previous():
    records = [_dividend_record(_days_ago(180), 1.0), _dividend_record(_days_ago(30), 1.0)]
    assert _compute_dividend_activity(records) == "held"


def test_dividend_held_when_only_one_record_ever():
    records = [_dividend_record(_days_ago(30), 1.0)]
    assert _compute_dividend_activity(records) == "held"


def test_dividend_trend_uses_two_most_recent_regardless_of_window():
    """A payment 13 months ago and one 4 months ago still have a real
    trend to report - the increased/held/cut comparison isn't restricted
    to the recent window, only the active-vs-suspended check is."""
    records = [_dividend_record(_days_ago(400), 1.0), _dividend_record(_days_ago(120), 1.5)]
    assert _compute_dividend_activity(records) == "increased"


# --- build_management_signals ---


def _fake_signals_router_factory(insider_rows=None, dividends=None):
    insider_rows = insider_rows if insider_rows is not None else []
    dividends = dividends if dividends is not None else []
    calls = {"get_insider_trading": [], "get_dividend_history": []}

    class _FakeRouter:
        def __init__(self, ticker):
            self.ticker = ticker

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get_insider_trading(self, ticker, days):
            calls["get_insider_trading"].append((ticker, days))
            return insider_rows

        async def get_dividend_history(self, ticker, from_date, to_date):
            calls["get_dividend_history"].append((ticker, from_date, to_date))
            return dividends

    return _FakeRouter, calls


async def test_build_management_signals_always_none_c_suite(monkeypatch):
    """No provider anywhere exposes structured executive-change data - a
    prior version of this field once had a roster size mistakenly
    rendered into it, reaching the CIO's evidence base as a fabricated
    instability signal. Must always be None/"", never a guess."""
    fake_cls, _ = _fake_signals_router_factory()
    monkeypatch.setattr("data.precompute.research_sources.Router", fake_cls)

    signals = await build_management_signals("AAPL")

    assert signals.c_suite_changes_12mo is None
    assert signals.changes_detail == ""


async def test_build_management_signals_wires_real_derivations(monkeypatch):
    fake_cls, calls = _fake_signals_router_factory(
        insider_rows=[_insider_row("purchase", value=5000, date=_days_ago(5))],
        dividends=[_dividend_record(_days_ago(30), 1.0)],
    )
    monkeypatch.setattr("data.precompute.research_sources.Router", fake_cls)

    signals = await build_management_signals("AAPL")

    assert signals.insider_net_direction_90d == "buying"
    assert signals.dividend_activity == "held"
    assert signals.buyback_activity == "none"
    assert calls["get_insider_trading"] == [("AAPL", 365)]
    assert len(calls["get_dividend_history"]) == 1
    assert calls["get_dividend_history"][0][0] == "AAPL"


# --- compute_dual_class_flag ---


def test_dual_class_flag_true_for_ca_multi_class():
    assert compute_dual_class_flag("RCI-B.TO", "Rogers Communications Inc.") is True
    assert compute_dual_class_flag("CCL-B.TO", "CCL Industries Inc.") is True


def test_dual_class_flag_true_for_us_multi_class():
    assert compute_dual_class_flag("BRK-A", "Berkshire Hathaway Inc.") is True
    assert compute_dual_class_flag("BRK-B", "Berkshire Hathaway Inc.") is True


def test_dual_class_flag_false_for_single_class():
    """Also covers the share-class-qualifier signal (finding #47) not
    false-positiving here: RY.TO/AAPL's real company_info names have no
    "Class N" qualifier, so this holds for both signals at once, not just
    the ticker-suffix one."""
    assert compute_dual_class_flag("RY.TO", "Royal Bank of Canada") is False
    assert compute_dual_class_flag("AAPL", "Apple Inc.") is False


def test_dual_class_flag_false_for_reit_unit_suffix():
    """-UN must not false-positive as dual-class - a real regression
    check: this is exactly the kind of near-miss suffix a naive check
    could get wrong."""
    assert compute_dual_class_flag("REI-UN.TO", "RioCan Real Estate Investment Trust") is False
    assert compute_dual_class_flag("CAR-UN.TO", "Canadian Apartment Properties REIT") is False


def test_dual_class_flag_true_via_share_class_qualifier_in_name():
    """The real, live-caught fix: SHOP.TO has no distinguishing ticker
    suffix at all (its Class B is founder-held, never publicly traded
    under any ticker) - the ticker-suffix signal alone returns False for
    this, missing the live Stock Researcher prompt's own named dual-class
    test fixture ("Pass Shopify (dual-class) with dual_class_flag=true").
    The real company_info name carries the signal the ticker can't:
    confirmed live, SHOP.TO's actual name is "Shopify Inc. Class A
    Subordinate Voting Shares"."""
    assert (
        compute_dual_class_flag("SHOP.TO", "Shopify Inc. Class A Subordinate Voting Shares") is True
    )


def test_dual_class_flag_known_miss_when_neither_signal_fires():
    """Documented gap, not a passing test pretending otherwise: GOOG has
    no distinguishing ticker suffix AND no yfinance field distinguishes
    its company_info name from GOOGL's (shortName/longName/quoteType are
    byte-identical for both, confirmed live) - so neither signal this
    function checks can catch it."""
    assert compute_dual_class_flag("GOOG", "Alphabet Inc.") is False


# --- compute_cik_verified ---


async def test_cik_verified_true_when_live_cik_matches(monkeypatch):
    monkeypatch.setattr("data.precompute.research_sources.is_crosslisted", lambda t: True)
    monkeypatch.setattr("data.precompute.research_sources.get_us_ticker", lambda t: "RY")
    monkeypatch.setattr("data.precompute.research_sources.get_expected_cik", lambda t: 1000275)

    class _FakeCompany:
        def __init__(self, ticker):
            self.cik = 1000275

    monkeypatch.setattr("data.precompute.research_sources.Company", _FakeCompany)

    assert await compute_cik_verified("RY.TO", has_filing_digest=True) is True


async def test_cik_verified_false_when_live_cik_mismatches(monkeypatch):
    monkeypatch.setattr("data.precompute.research_sources.is_crosslisted", lambda t: True)
    monkeypatch.setattr("data.precompute.research_sources.get_us_ticker", lambda t: "RY")
    monkeypatch.setattr("data.precompute.research_sources.get_expected_cik", lambda t: 1000275)

    class _FakeCompany:
        def __init__(self, ticker):
            self.cik = 999999

    monkeypatch.setattr("data.precompute.research_sources.Company", _FakeCompany)

    assert await compute_cik_verified("RY.TO", has_filing_digest=True) is False


async def test_cik_verified_false_when_company_construction_raises(monkeypatch):
    monkeypatch.setattr("data.precompute.research_sources.is_crosslisted", lambda t: True)
    monkeypatch.setattr("data.precompute.research_sources.get_us_ticker", lambda t: "RY")
    monkeypatch.setattr("data.precompute.research_sources.get_expected_cik", lambda t: 1000275)

    def _raise(ticker):
        raise ValueError("not found")

    monkeypatch.setattr("data.precompute.research_sources.Company", _raise)

    assert await compute_cik_verified("RY.TO", has_filing_digest=True) is False


async def test_cik_verified_false_when_not_in_crosslisting_map(monkeypatch):
    monkeypatch.setattr("data.precompute.research_sources.is_crosslisted", lambda t: True)
    monkeypatch.setattr("data.precompute.research_sources.get_us_ticker", lambda t: None)
    monkeypatch.setattr("data.precompute.research_sources.get_expected_cik", lambda t: None)

    assert await compute_cik_verified("RY.TO", has_filing_digest=True) is False


async def test_cik_verified_plain_us_returns_has_filing_digest(monkeypatch):
    """No second source to verify against for a plain US ticker - always
    identical to whatever build_filing_digests already determined."""
    monkeypatch.setattr("data.precompute.research_sources.is_crosslisted", lambda t: False)
    monkeypatch.setattr("data.precompute.research_sources.is_canadian", lambda stock, ticker: False)

    assert await compute_cik_verified("AAPL", has_filing_digest=True) is True
    assert await compute_cik_verified("AAPL", has_filing_digest=False) is False


async def test_cik_verified_false_for_pure_unmapped_ca(monkeypatch):
    monkeypatch.setattr("data.precompute.research_sources.is_crosslisted", lambda t: False)
    monkeypatch.setattr("data.precompute.research_sources.is_canadian", lambda stock, ticker: True)

    assert await compute_cik_verified("WELL.TO", has_filing_digest=False) is False


async def test_cik_verified_forwards_stock_to_is_canadian(monkeypatch):
    monkeypatch.setattr("data.precompute.research_sources.is_crosslisted", lambda t: False)
    received = {}

    def fake_is_canadian(stock, ticker):
        received["stock"] = stock
        return False

    monkeypatch.setattr("data.precompute.research_sources.is_canadian", fake_is_canadian)

    sentinel_stock = object()
    await compute_cik_verified("AAPL", has_filing_digest=True, stock=sentinel_stock)

    assert received["stock"] is sentinel_stock


# --- _derive_short_name ---


def test_derive_short_name_strips_corporate_suffix():
    assert _derive_short_name("Apple Inc.") == "Apple"
    assert _derive_short_name("Microsoft Corporation") == "Microsoft"
    assert _derive_short_name("Block, Inc.") == "Block"


def test_derive_short_name_strips_share_class_qualifier():
    """Real, live-caught case: SHOP.TO's get_company_info() name field is
    "Shopify Inc. Class A Subordinate Voting Shares" - noise no filing
    text ever repeats."""
    assert _derive_short_name("Shopify Inc. Class A Subordinate Voting Shares") == "Shopify"


def test_derive_short_name_normalizes_trailing_the():
    """Real, live-caught case: TD.TO's get_company_info() name field is
    "Toronto-Dominion Bank (The)" - reversed word order from how the
    company's own business summary refers to it ("The Toronto-Dominion
    Bank...")."""
    assert _derive_short_name("Toronto-Dominion Bank (The)") == "Toronto-Dominion Bank"


def test_derive_short_name_normalizes_leading_the():
    assert _derive_short_name("The Bank of Nova Scotia") == "Bank of Nova Scotia"


def test_derive_short_name_none_when_nothing_to_strip():
    """A name with no corporate suffix, share-class qualifier, or "The"
    wrapping has no distinct short form - returning the same string again
    would be a pointless, wasted second substitution pass."""
    assert _derive_short_name("Royal Bank of Canada") is None


# --- anonymize_content ---


def test_anonymize_content_replaces_company_and_ticker():
    text = "Royal Bank of Canada (RY) reported strong earnings."
    result = anonymize_content(text, "Royal Bank of Canada", "RY", {})
    assert "COMPANY_X" in result
    assert "TICKER_X" in result
    assert "Royal Bank of Canada" not in result


def test_anonymize_content_word_boundary_does_not_match_inside_other_words():
    """The ticket's own E17 regression: RY must not match inside
    "advisory" - a plain substring search would wrongly corrupt this."""
    text = "Provided financial advisory services to clients."
    result = anonymize_content(text, "Royal Bank of Canada", "RY", {})
    assert result == text


def test_anonymize_content_replaces_peer_names():
    text = "Block, Inc. competes with Wix.com Ltd. in this market."
    result = anonymize_content(
        text, "Shopify Inc.", "SHOP", {"PEER_1": "Block, Inc.", "PEER_2": "Wix.com Ltd."}
    )
    assert "PEER_1_COMPANY" in result
    assert "PEER_2_COMPANY" in result
    assert "Block, Inc." not in result
    assert "Wix.com Ltd." not in result


def test_anonymize_content_empty_company_name_is_a_no_op_for_that_substitution():
    text = "Some text with no matches."
    result = anonymize_content(text, "", "", {})
    assert result == text


def test_anonymize_content_case_insensitive():
    """A missed differently-cased mention leaking the real name is the
    failure mode to avoid for a safety feature."""
    text = "APPLE INC. reported earnings. apple inc. is a technology company."
    result = anonymize_content(text, "Apple Inc.", "AAPL", {})
    assert "COMPANY_X" in result
    assert "apple" not in result.lower()


# --- anonymize_content: real bugs caught in end-to-end live verification,
# not synthetic cases - each of these failed before _derive_short_name
# existed, confirmed by running build_research_sources against real
# RY.TO/AAPL/SHOP.TO data and inspecting the actual anonymized output. ---


def test_anonymize_content_catches_possessive_short_form():
    """AAPL's own real MD&A digest said "Apple's fiscal 2025 saw..." -
    get_company_info() correctly returns "Apple Inc." (which matches
    AAPL's Business digest just fine), but the exact string "Apple Inc."
    never appears in this possessive MD&A phrasing at all."""
    text = "Apple's fiscal 2025 saw total net sales of $416,161 million."
    result = anonymize_content(text, "Apple Inc.", "AAPL", {})
    assert "apple" not in result.lower()
    assert "COMPANY_X's fiscal 2025" in result


def test_anonymize_content_catches_reversed_the_wrapping():
    """TD.TO's real get_company_info() name is "Toronto-Dominion Bank
    (The)" (reversed word order), but its business summary text says
    "The Toronto-Dominion Bank, together with its subsidiaries..." - the
    exact strings never match without the short-name derivation."""
    text = "The Toronto-Dominion Bank, together with its subsidiaries, provides banking."
    result = anonymize_content(text, "x", "x", {"PEER_1": "Toronto-Dominion Bank (The)"})
    assert "toronto-dominion" not in result.lower()
    assert "PEER_1_COMPANY" in result


def test_anonymize_content_catches_share_class_qualified_name():
    """SHOP.TO's real get_company_info() name is "Shopify Inc. Class A
    Subordinate Voting Shares" - noise its own business summary text
    never repeats ("Shopify Inc., a commerce technology company...")."""
    text = "Shopify Inc., a commerce technology company, provides tools to merchants."
    result = anonymize_content(text, "Shopify Inc. Class A Subordinate Voting Shares", "SHOP", {})
    assert "shopify" not in result.lower()
    assert "COMPANY_X" in result


# --- build_research_sources ---


async def test_build_research_sources_full_assembly(monkeypatch):
    monkeypatch.setattr("data.precompute.research_sources.is_crosslisted", lambda t: False)
    monkeypatch.setattr("data.precompute.research_sources.is_canadian", lambda stock, ticker: False)

    class _FakeEdgarProvider:
        async def get_filing_section(self, ticker, section):
            return _section(f"{section} text about Acme Corporation itself", "2026-01-01")

    monkeypatch.setattr(
        "data.precompute.research_sources.EdgarToolsDataProvider", _FakeEdgarProvider
    )

    async def fake_summarize(session, text, section):
        return _digest(section, text)

    monkeypatch.setattr("data.precompute.research_sources.summarize_filing_section", fake_summarize)

    class _FakeRouter:
        def __init__(self, ticker):
            self.ticker = ticker

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get_company_info(self, ticker):
            if ticker == "AAPL":
                return {"name": "Acme Corporation"}
            if ticker == "PEERCO":
                return {"name": "Peer Company"}
            return {}

        async def get_peers(self, ticker, limit):
            return ["PEERCO"]

        async def get_business_summary(self, ticker):
            return "Peer Company makes widgets."

        async def get_news(self, ticker, days):
            if ticker == "PEERCO":
                return [
                    {
                        "headline": "Peer Company launches new widget",
                        "published_at": "2026-09-01 00:00:00",
                    }
                ]
            return []

        async def get_insider_trading(self, ticker, days):
            return [_insider_row("purchase", value=5000, date=_days_ago(5))]

        async def get_dividend_history(self, ticker, from_date, to_date):
            return []

    monkeypatch.setattr("data.precompute.research_sources.Router", _FakeRouter)

    articles = [
        {
            "id": "N1",
            "date": datetime(2026, 9, 1),
            "headline": "Acme Corp reports earnings",
            "source": "Reuters",
            "quality_tier": "primary",
        }
    ]

    bundle = await build_research_sources("AAPL", articles)

    assert isinstance(bundle, ResearchSourcesBundle)
    assert len(bundle.filing_digests) == 2
    assert all("Acme Corporation" not in d.content for d in bundle.filing_digests)
    assert all("COMPANY_X" in d.content for d in bundle.filing_digests)
    assert len(bundle.peer_blocks) == 1
    assert "Peer Company" not in bundle.peer_blocks[0].content
    assert "PEER_1_COMPANY" in bundle.peer_blocks[0].content
    assert bundle.news_item_count == 1
    assert bundle.news_items[0].id == "N1"
    assert bundle.sedar_filing_available is True
    assert bundle.cik_verified is True  # plain US, has_filing_digest=True
    assert bundle.dual_class_flag is False
    assert bundle.management_signals.insider_net_direction_90d == "buying"
    assert bundle.management_signals.c_suite_changes_12mo is None
    assert bundle.missing_sources_list == []
    # Filing digest token_count must be rescaled from the pre-anonymization
    # digest's own real count (10, from _digest's stub) by how much
    # "Acme Corporation" (16 chars) -> "COMPANY_X" (9 chars) actually
    # shrank the content - not recomputed from scratch via len//4, which
    # would throw away the real LLM-reported count a live check showed
    # overstates true token count by ~18% on real filing text.
    for digest, raw_content in zip(
        bundle.filing_digests,
        ("Business text about Acme Corporation itself", "MDA text about Acme Corporation itself"),
    ):
        expected = max(1, round(10 * len(digest.content) / len(raw_content)))
        assert digest.token_count == expected
        assert digest.token_count < 10  # anonymization shrank this content, count must reflect that
    # Peer blocks never had a real LLM-reported count (build_peer_blocks
    # itself uses len//4, finding #28) - recomputing len//4 post-anonymization
    # stays consistent with that, not a new approximation.
    for block in bundle.peer_blocks:
        assert block.token_count == max(1, len(block.content) // 4)


async def test_build_research_sources_dual_class_flag_wired_from_fetched_company_name(monkeypatch):
    """Integration-level check, not just a compute_dual_class_flag unit
    test: proves build_research_sources actually threads its own fetched
    company_name into compute_dual_class_flag, end to end. The full-
    assembly test above only exercises a company_name with no "Class"
    qualifier, so a False result there would look identical whether the
    wiring were correct or silently broken (e.g. passing "" instead of
    the real fetched name). This ticker/name pair can only produce
    dual_class_flag=True via the company_name-based signal - SHOP has no
    dual-class ticker suffix at all - so it specifically proves the
    fetched name actually reaches the function, not just that the
    function works in isolation."""
    monkeypatch.setattr("data.precompute.research_sources.is_crosslisted", lambda t: False)
    monkeypatch.setattr("data.precompute.research_sources.is_canadian", lambda stock, ticker: True)

    class _FakeRouter:
        def __init__(self, ticker):
            self.ticker = ticker

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get_company_info(self, ticker):
            return {"name": "Shopify Inc. Class A Subordinate Voting Shares"}

        async def get_peers(self, ticker, limit):
            return []

        async def get_insider_trading(self, ticker, days):
            return []

        async def get_dividend_history(self, ticker, from_date, to_date):
            return []

    monkeypatch.setattr("data.precompute.research_sources.Router", _FakeRouter)

    bundle = await build_research_sources("SHOP", [])

    assert bundle.dual_class_flag is True


async def test_build_research_sources_degrades_cleanly_with_nothing_available(monkeypatch):
    """Pure unmapped CA ticker, no peers, no news: every list field empty,
    the bundle must still construct validly (both model_validators
    satisfied) rather than raise."""
    monkeypatch.setattr("data.precompute.research_sources.is_crosslisted", lambda t: False)
    monkeypatch.setattr("data.precompute.research_sources.is_canadian", lambda stock, ticker: True)

    class _FakeRouter:
        def __init__(self, ticker):
            self.ticker = ticker

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get_company_info(self, ticker):
            return {"name": "Some Co"}

        async def get_peers(self, ticker, limit):
            return []

        async def get_insider_trading(self, ticker, days):
            return []

        async def get_dividend_history(self, ticker, from_date, to_date):
            return []

    monkeypatch.setattr("data.precompute.research_sources.Router", _FakeRouter)

    bundle = await build_research_sources("WELL.TO", [])

    assert bundle.filing_digests == []
    assert bundle.peer_blocks == []
    assert bundle.news_items == []
    assert bundle.sedar_filing_available is False
    assert bundle.cik_verified is False
    assert set(bundle.missing_sources_list) == {"filing_digests", "peer_blocks", "news_items"}
    assert bundle.latest_filing_age_days is None
    assert bundle.latest_news_age_days is None


async def test_build_research_sources_logs_when_main_company_name_unresolvable(monkeypatch):
    """A near-unreachable edge case in practice (the main ticker is the
    entire subject of the analysis), but must be logged, not silently
    swallowed, if it ever happens."""
    monkeypatch.setattr("data.precompute.research_sources.is_crosslisted", lambda t: False)
    monkeypatch.setattr("data.precompute.research_sources.is_canadian", lambda stock, ticker: True)

    class _FakeRouter:
        def __init__(self, ticker):
            self.ticker = ticker

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get_company_info(self, ticker):
            return {}

        async def get_peers(self, ticker, limit):
            return []

        async def get_insider_trading(self, ticker, days):
            return []

        async def get_dividend_history(self, ticker, from_date, to_date):
            return []

    monkeypatch.setattr("data.precompute.research_sources.Router", _FakeRouter)

    with structlog.testing.capture_logs() as logs:
        bundle = await build_research_sources("ZZZZ.TO", [])

    assert bundle is not None  # did not raise
    assert any(log["event"] == "research_sources_no_company_name" for log in logs)
