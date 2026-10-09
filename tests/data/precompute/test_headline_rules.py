"""Headlines whose label the wording fixes are labelled by pattern (ledger BB-111)."""
import pytest

from data.precompute.headline_rules import names_company, rule_label
from data.precompute import sentiment as sentiment_module


@pytest.mark.parametrize(
    ("headline", "expected"),
    [
        ("Shopify (SHOP) Laps the Stock Market: Here's Why", "neutral"),
        ("Lucid Group (LCID) Registers a Bigger Fall Than the Market: Important Facts to Note", "neutral"),
        ("Plug Power (PLUG) Stock Sinks As Market Gains: What You Should Know", "neutral"),
        ("Enbridge Inc. to Host Webcast to Discuss 2026 Third Quarter Results on November 6", "neutral"),
        ("Tilray Brands Q1 2027 Earnings Call Transcript", "neutral"),
        ("Procter & Gamble Company (The) (PG) Is a Trending Stock: Facts to Know Before Betting on It", "neutral"),
        ("Brookfield (BAM) Upgraded to Buy: Here's What You Should Know", "positive"),
        ("Piper Sandler Reiterates Neutral on Brookfield Asset Mgmt, Lowers Price Target to $50", "negative"),
        ("Goldman Sachs Reinstates Neutral on Enbridge, Announces $50 Price Target", "neutral"),
        ("Wells Fargo Raises Price Target on Coca-Cola to $80", "positive"),
    ],
)
def test_wording_that_fixes_the_label(headline, expected):
    company = {"Shopify": "Shopify", "Lucid": "Lucid Group", "Plug": "Plug Power", "Enbridge": "Enbridge", "Tilray": "Tilray Brands",
               "Procter": "Procter & Gamble", "Brookfield": "Brookfield Asset Management", "Wells": "Coca-Cola Company"}
    name = next(v for k, v in company.items() if k in headline)
    assert rule_label(headline, None, name, None) == expected


def test_an_item_that_does_not_name_the_company_is_left_to_the_model():
    assert rule_label("Royal Bank of Canada (RY:CA) Presents at Barclays Conference Transcript", None, "Toronto-Dominion Bank", "TD.TO") is None
    assert rule_label("Corteva Soars 11% on JPM Upgrade to Overweight", None, "Corteva", "CTVA") is None  # noun, not a rating verb
    assert rule_label("Shopify Laps the Stock Market", None, "Plug Power", "PLUG") is None


def test_a_price_move_with_real_news_is_not_a_recap():
    assert rule_label("Plug Power Advances 4% on Electrolyzer Supply Agreement", None, "Plug Power", "PLUG") is None
    assert rule_label("Why Plug Power (PLUG) Stock Is Up Today", None, "Plug Power", "PLUG") is None


def test_the_snippet_of_a_daily_recap_is_enough():
    snippet = "In the most recent trading session, Enbridge (ENB) closed at $46.54, indicating a +1.09% shift from the previous trading day."
    assert rule_label("Enbridge stock this week", snippet, "Enbridge", "ENB.TO") == "neutral"


def test_the_company_is_matched_by_first_word_or_a_ticker_written_as_a_ticker():
    assert names_company("Coca-Cola Plans $10B", "The Coca-Cola Company", "KO")
    assert names_company("Why Shopify (SHOP) Laps the Market", None, "SHOP.TO")
    assert names_company("Brookfield Asset Management Ltd. (BAM:CA) Day Transcript", None, "BAM.TO")
    assert names_company("Shares of NYSE:ENB rose", None, "ENB.TO")
    assert not names_company("Pepsi announces cuts", "The Coca-Cola Company", "KO")


def test_a_bare_word_is_never_taken_for_a_ticker_and_generic_first_words_need_a_second_word():
    assert not names_company("Why investors are on edge", None, "ON")  # ON Semiconductor
    assert not names_company("Fed to host webcast on rates", None, "FED")
    assert not names_company("Bank of Canada to host webcast", "Bank of Montreal", "BMO.TO")
    assert names_company("Bank of Montreal to host webcast", "Bank of Montreal", "BMO.TO")


@pytest.mark.parametrize(
    ("headline", "expected"),
    [
        ("Microsoft announces quarterly dividend increase", "positive"),
        ("Microsoft Raises Dividend by 8%", "positive"),
        ("Reddit (RDDT) vs Snap (SNAP): Which is a Better Stock to Buy?", "neutral"),
        ("Shopify vs. Uber Technologies: Which Technology Stock Is a Better Buy in 2026?", "neutral"),
    ],
)
def test_dividend_raises_and_comparisons(headline, expected):
    company = next(c for k, c in {"Microsoft": "Microsoft", "Toronto": "Toronto-Dominion Bank", "Reddit": "Reddit", "Shopify": "Shopify"}.items() if k in headline)
    assert rule_label(headline, None, company, None) == expected


@pytest.mark.parametrize(
    "headline",
    [
        "Wolfspeed Stock Jumps as Market Reacts to Pentagon Loan",  # news, not a recap: no market-direction word after "market"
        "Reddit vs. Anthropic: Court Rules in Reddit's Favor",  # a lawsuit, not a comparison of stocks
        "Microsoft Could Be 18% Overvalued On Its Dividend Increase",  # valuation commentary, not an announcement
        "Plug Power Maintains Its Hold on the Electrolyzer Market",
    ],
)
def test_news_that_looks_like_a_template_is_left_to_the_model(headline):
    company = next(c for k, c in {"Wolfspeed": "Wolfspeed", "Reddit": "Reddit", "Microsoft": "Microsoft", "Plug": "Plug Power"}.items() if k in headline)
    assert rule_label(headline, None, company, None) is None


@pytest.mark.asyncio
async def test_pattern_labelled_items_never_reach_the_model_and_order_is_kept(monkeypatch):
    seen = []

    async def fake_batch(session, items, **kwargs):
        seen.extend(i["headline"] for i in items)
        return ["negative"] * len(items)

    monkeypatch.setattr(sentiment_module, "_score_batch", fake_batch)
    articles = [
        {"id": "N1", "date": "2026-10-01", "headline": "Shopify (SHOP) Laps the Stock Market: Here's Why", "source": "Zacks", "quality_tier": "secondary", "text": ""},
        {"id": "N2", "date": "2026-10-02", "headline": "Shopify cuts 300 jobs", "source": "Reuters", "quality_tier": "primary", "text": ""},
        {"id": "N3", "date": "2026-10-03", "headline": "Shopify to Announce Third-Quarter 2026 Financial Results", "source": "PR", "quality_tier": "primary", "text": ""},
    ]
    result = await sentiment_module.summarize_news(articles, company="Shopify", ticker="SHOP.TO")
    assert seen == ["Shopify cuts 300 jobs"]
    assert [a["sentiment"] for a in result["articles"]] == ["neutral", "negative", "neutral"]
