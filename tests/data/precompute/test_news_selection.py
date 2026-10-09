"""Tests for data/precompute/news_selection.py: a bounded, time-balanced set per agent."""

from datetime import datetime

from data.precompute.news_selection import select_news


def _a(
    day: int, minute: int, headline: str, source: str = "Yahoo", tier: str = "secondary"
) -> dict:
    return {
        "id": f"{day}-{minute}",
        "date": datetime(2026, 9, day, 12, minute),
        "headline": headline,
        "source": source,
        "quality_tier": tier,
    }


def _busy_day(day: int, count: int) -> list[dict]:
    return [_a(day, i % 60, f"Story {day}-{i}") for i in range(count)]


def test_a_busy_day_is_cut_to_the_per_day_sizes():
    result = select_news(_busy_day(1, 100), scored_per_day=6, shown_per_day=2, researcher_per_day=2)

    assert (len(result["scored"]), len(result["shown"]), len(result["researcher"])) == (6, 2, 2)
    assert result["fetched"] == 100


def test_the_selection_does_not_grow_with_ticker_volume():
    """A 100-a-day ticker and a 400-a-day ticker give the same sized selection."""
    small = select_news([a for d in range(1, 31) for a in _busy_day(d, 100)])
    huge = select_news([a for d in range(1, 31) for a in _busy_day(d, 400)])

    assert len(small["scored"]) == len(huge["scored"]) == 30 * 6
    assert len(small["shown"]) == len(huge["shown"]) == 30 * 2


def test_a_quiet_ticker_loses_nothing():
    articles = [_a(d, 0, f"Story {d}") for d in range(1, 11)]  # one a day

    result = select_news(articles)

    assert len(result["scored"]) == len(result["shown"]) == len(result["researcher"]) == 10


def test_every_day_with_articles_is_represented():
    articles = [a for d in (1, 5, 9, 20) for a in _busy_day(d, 50)]

    result = select_news(articles)

    assert {a["date"].day for a in result["shown"]} == {1, 5, 9, 20}


def test_shown_is_a_subset_of_scored_and_both_are_the_same_dicts_oldest_first():
    articles = [a for d in range(1, 6) for a in _busy_day(d, 20)]

    result = select_news(articles)

    scored_ids = {id(a) for a in result["scored"]}
    assert all(id(a) in scored_ids for a in result["shown"])
    assert all(any(a is b for b in articles) for a in result["scored"])  # not copies
    assert [a["date"] for a in result["scored"]] == sorted(a["date"] for a in result["scored"])


def test_one_publisher_does_not_fill_every_slot():
    articles = [_a(1, i, f"Yahoo story {i}", source="Yahoo") for i in range(30)] + [
        _a(1, 40, "Benzinga story", source="Benzinga"),
        _a(1, 41, "Seeking Alpha story", source="SeekingAlpha"),
    ]

    shown = select_news(articles, shown_per_day=3)["shown"]

    assert {a["source"] for a in shown} == {"Yahoo", "Benzinga", "SeekingAlpha"}


def test_a_higher_quality_tier_is_chosen_before_a_lower_one():
    articles = [_a(1, 1, "Low tier newest", tier="low")] + [
        _a(1, 2, "Wire story", source="Reuters", tier="primary")
    ]

    shown = select_news(articles, shown_per_day=1)["shown"]

    assert [a["headline"] for a in shown] == ["Wire story"]


def test_the_researcher_never_gets_low_tier_headlines():
    articles = [_a(1, i, f"Low {i}", tier="low") for i in range(5)] + [
        _a(1, 30, "Fine", tier="secondary")
    ]

    result = select_news(articles)

    assert [a["headline"] for a in result["researcher"]] == ["Fine"]
    assert len(result["scored"]) == 6  # low tier is still scored for the Sentiment sample


def test_near_duplicate_headlines_count_once():
    articles = [
        _a(1, 1, "Apple, Inc. beats!"),
        _a(1, 2, "apple inc beats"),
        _a(1, 3, "Different story"),
    ]

    result = select_news(articles)

    assert result["fetched"] == 2
    assert len(result["scored"]) == 2


def test_the_selection_is_deterministic():
    articles = [a for d in range(1, 8) for a in _busy_day(d, 40)]

    first = select_news(articles)
    second = select_news(list(reversed(articles)))

    assert [a["id"] for a in first["shown"]] == [a["id"] for a in second["shown"]]


def test_no_articles_gives_empty_lists():
    assert select_news([]) == {"scored": [], "shown": [], "scored_days": [], "researcher": [], "fetched": 0}


def test_items_naming_the_company_rank_before_round_ups_that_only_list_the_ticker():
    articles = [
        _a(1, 1, "3 Mega-Cap Stocks with Solid Fundamentals"),
        _a(1, 2, "Stocks to buy now: Nvidia, Tesla"),
        _a(1, 3, "Microsoft unveils new Azure region"),
        _a(1, 4, "MSFT shares rise after earnings", source="Benzinga"),
    ]

    shown = select_news(articles, shown_per_day=2, names=["MSFT", "Microsoft"])["shown"]

    assert {a["headline"] for a in shown} == {
        "Microsoft unveils new Azure region",
        "MSFT shares rise after earnings",
    }


def test_a_name_the_matcher_cannot_find_keeps_the_old_behaviour_so_nothing_is_emptied():
    """Below MIN_NAMED_TO_FILTER named items in the window, the preference is not a filter."""
    articles = [_a(1, 1, "A market round-up"), _a(2, 1, "Microsoft news")]

    shown = select_news(articles, shown_per_day=1, names=["Microsoft"])["shown"]

    assert {a["date"].day for a in shown} == {1, 2}


def _named_window(named_days: int = 6) -> list[dict]:
    """Days 1..named_days each hold one Microsoft item and two items that never name it; day 20 holds only noise."""
    out = []
    for day in range(1, named_days + 1):
        out += [_a(day, 1, f"Microsoft story {day}"), _a(day, 2, f"Round-up {day}"), _a(day, 3, f"Clickbait {day}")]
    out.append(_a(20, 1, "Sawmill expansion announced"))
    return out


def test_with_enough_company_items_the_sentiment_sample_keeps_only_those():
    """RY.TO / ENB.TO / BAM.TO 2026-10-03: 45-62% of the shown headlines never named the company."""
    result = select_news(_named_window(), shown_per_day=2, scored_per_day=3, names=["Microsoft"])

    assert all("Microsoft" in a["headline"] for a in result["shown"] + result["scored"])
    assert {a["date"].day for a in result["shown"]} == {1, 2, 3, 4, 5, 6}  # the noise-only day 20 is not listed


def test_the_researcher_list_keeps_only_company_items_when_the_window_holds_enough():
    """KO 2026-10-06: the list held a crypto exchange valuation, a Monster Beverage stock pick and a Georgia income piece."""
    result = select_news(_named_window(), researcher_per_day=2, names=["Microsoft"])

    assert result["researcher"] and all("Microsoft" in a["headline"] for a in result["researcher"])


def test_the_researcher_list_is_not_emptied_when_the_name_cannot_be_found():
    few = [_a(1, 1, "Microsoft one"), _a(2, 1, "Other news"), _a(3, 1, "More other news")]

    assert len(select_news(few, names=["Microsoft"])["researcher"]) == 3


def test_the_researcher_list_is_capped_but_still_spans_the_window():
    """The model cited a median of 4 of 52 headlines. Keeping only the newest 20 covered 9 to 10 of the 30 days for MSFT, KO
    and SHOP.TO, so the cap is spread evenly through the window instead."""
    articles = [_a(day, 1, f"Microsoft story {day}") for day in range(1, 29)]

    kept = select_news(articles, names=["Microsoft"], researcher_max=5)["researcher"]

    days = [a["date"].day for a in kept]
    assert len(days) == 5 and days[0] == 1 and days[-1] == 28  # first and newest both kept
    assert days == sorted(days) and max(b - a for a, b in zip(days, days[1:])) <= 8  # no big hole in the window


def test_a_list_under_the_cap_is_left_whole_and_a_cap_of_one_keeps_the_newest():
    articles = [_a(day, 1, f"Microsoft story {day}") for day in range(1, 5)]

    assert len(select_news(articles, names=["Microsoft"], researcher_max=10)["researcher"]) == 4
    assert [a["date"].day for a in select_news(articles, names=["Microsoft"], researcher_max=1)["researcher"]] == [4]
    assert select_news(articles, names=["Microsoft"], researcher_max=0)["researcher"] == []


def test_the_filter_needs_the_minimum_number_of_company_items_across_the_window():
    few = [_a(1, 1, "Microsoft one"), _a(2, 1, "Microsoft two"), _a(2, 2, "Other news"), _a(3, 1, "More other news")]

    result = select_news(few, shown_per_day=2, names=["Microsoft"])

    assert any("Microsoft" not in a["headline"] for a in result["shown"])  # 2 named is under the minimum of 5


def test_the_ticker_is_matched_as_a_whole_word_and_without_its_canadian_suffix():
    articles = [_a(1, 1, "SHOP reports"), _a(1, 2, "Workshop updates"), _a(1, 3, "Other")]

    shown = select_news(articles, shown_per_day=1, names=["SHOP.TO"])["shown"]

    assert [a["headline"] for a in shown] == ["SHOP reports"]  # "Workshop" is not SHOP


def test_shown_is_never_larger_than_scored_even_if_the_sizes_are_set_the_wrong_way_round():
    result = select_news(_busy_day(1, 20), scored_per_day=2, shown_per_day=5)

    scored_ids = {id(a) for a in result["scored"]}
    assert len(result["shown"]) == 2 and all(id(a) in scored_ids for a in result["shown"])


# ---------- the listed headlines are chosen after scoring (ledger BB-111) ----------


def test_pick_shown_skips_articles_scored_as_unrelated_and_takes_the_next_best_of_the_day():
    from data.precompute.news_selection import pick_shown

    day = [{"id": f"N{i}", "date": datetime(2026, 9, 10, 12 - i)} for i in range(5)]
    labels = {"N0": "unrelated", "N1": "positive", "N2": "unrelated", "N3": "neutral", "N4": "negative"}
    assert [a["id"] for a in pick_shown([day], labels, shown_per_day=2)] == ["N3", "N1"]  # oldest first


def test_pick_shown_without_labels_is_the_same_as_choosing_before_scoring_and_a_day_of_only_unrelated_shows_nothing():
    from data.precompute.news_selection import pick_shown

    day1 = [{"id": "A", "date": datetime(2026, 9, 1)}, {"id": "B", "date": datetime(2026, 9, 1, 1)}, {"id": "C", "date": datetime(2026, 9, 1, 2)}]
    day2 = [{"id": "D", "date": datetime(2026, 9, 2)}]
    assert [a["id"] for a in pick_shown([day1, day2], {})] == ["A", "B", "D"]
    assert [a["id"] for a in pick_shown([day1, day2], {"D": "unrelated"})] == ["A", "B"]


def test_select_news_returns_the_scored_articles_grouped_by_day_in_rank_order():
    articles = _busy_day(1, 20) + _busy_day(2, 20)
    result = select_news(articles, scored_per_day=6, shown_per_day=2)
    assert [len(d) for d in result["scored_days"]] == [6, 6]
    assert {id(a) for a in result["shown"]} <= {id(a) for d in result["scored_days"] for a in d[:2]}  # shown is each day's first two
    assert {id(a) for d in result["scored_days"] for a in d} == {id(a) for a in result["scored"]}
