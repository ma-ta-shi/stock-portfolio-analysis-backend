from datetime import datetime

from data.precompute.research_sources import (
    _render_peer_content,
    _truncate_to_tokens,
    build_filing_digests,
    build_news_items,
    build_peer_blocks,
)
from data.schemas.common import FilingDigest, NewsItem


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


def _fake_router_factory(peers=None, summaries=None, news=None):
    peers = peers or {}
    summaries = summaries or {}
    news = news or {}
    calls = {"get_peers": [], "get_news": []}

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
    )
    monkeypatch.setattr("data.precompute.research_sources.Router", fake_cls)

    result = await build_peer_blocks("AAPL")

    assert [b.peer_id for b in result] == ["PEER_1", "PEER_2"]
    assert "Microsoft makes software." in result[0].content
    assert "MSFT news" in result[0].content
    assert "Google makes search." in result[1].content
    assert calls["get_peers"] == [("AAPL", 2)]
    assert ("MSFT", 30) in calls["get_news"]
    assert ("GOOG", 30) in calls["get_news"]


async def test_peer_with_no_data_dropped_and_remaining_renumber(monkeypatch):
    fake_cls, _ = _fake_router_factory(
        peers={"AAPL": ["MSFT", "EMPTY", "GOOG"]},
        summaries={"MSFT": "Microsoft makes software.", "GOOG": "Google makes search."},
        news={
            "MSFT": [{"headline": "MSFT news", "published_at": "2026-09-01 00:00:00"}],
            "GOOG": [{"headline": "GOOG news", "published_at": "2026-09-01 00:00:00"}],
        },
    )
    monkeypatch.setattr("data.precompute.research_sources.Router", fake_cls)

    result = await build_peer_blocks("AAPL", limit=3)

    # EMPTY has no summary and no news (both default to nothing in the fakes),
    # so it must be dropped - not rendered as a hollow PEER_2 - and GOOG
    # (originally the 3rd candidate) becomes PEER_2, not PEER_3.
    assert [b.peer_id for b in result] == ["PEER_1", "PEER_2"]
    assert "Microsoft makes software." in result[0].content
    assert "Google makes search." in result[1].content


async def test_limit_and_news_days_forwarded(monkeypatch):
    fake_cls, calls = _fake_router_factory(peers={"AAPL": ["MSFT"]}, summaries={"MSFT": "x"})
    monkeypatch.setattr("data.precompute.research_sources.Router", fake_cls)

    await build_peer_blocks("AAPL", limit=1, news_days=7)

    assert calls["get_peers"] == [("AAPL", 1)]
    assert calls["get_news"] == [("MSFT", 7)]
