"""Score the hand-labelled headlines in tests/data/sentiment_gold.json with the production scorer and report agreement.

Needs Ollama running. Items are scored per company in batches of up to 30 (rules first, the model for the rest), the way the
pipeline does, but each batch holds only labelled items, so the numbers are a regression check, not a re-measurement of the
2026-10-09 figures (those used each saved run's whole batch). Usage, from backend/: python scripts/eval_sentiment_scorer.py
"""
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import aiohttp  # noqa: E402

from data.precompute import sentiment  # noqa: E402
from data.precompute.headline_rules import rule_label  # noqa: E402

GOLD = Path(__file__).resolve().parents[1] / "tests" / "data" / "sentiment_gold.json"


def _three_class(label: str) -> str:
    return "neutral" if label == "unrelated" else label


async def main() -> None:
    items = json.loads(GOLD.read_text(encoding="utf-8"))["items"]
    predicted: dict[str, str] = {}
    by_company = defaultdict(list)
    for item in items:
        label = rule_label(item["headline"], item["text"], item["company"], item["ticker"])
        if label:
            predicted[item["id"]] = label
        else:
            by_company[item["company"]].append(item)
    async with aiohttp.ClientSession() as session:
        for company, group in by_company.items():
            for start in range(0, len(group), sentiment._BATCH_SIZE):
                chunk = group[start : start + sentiment._BATCH_SIZE]
                labels = await sentiment._score_batch(session, chunk, company=company)
                for item, label in zip(chunk, labels, strict=True):
                    predicted[item["id"]] = label or "none"
    for name in ("dev", "fresh", "third", "all"):
        rows = [i for i in items if name == "all" or i["set"] == name]
        agree = sum(_three_class(predicted[i["id"]]) == _three_class(i["label"]) for i in rows)
        strict = sum(predicted[i["id"]] == i["label"] for i in rows)
        polar = sum({_three_class(predicted[i["id"]]), _three_class(i["label"])} == {"positive", "negative"} for i in rows)
        print(f"{name:6} n={len(rows):3}  agree {agree / len(rows):.0%}  strict {strict / len(rows):.0%}  reversed direction {polar}")


if __name__ == "__main__":
    asyncio.run(main())
