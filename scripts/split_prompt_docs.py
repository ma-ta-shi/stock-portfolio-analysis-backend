"""One-time (and again-on-new-version) authoring tool for ClickUp 86bbdutn6.

Splits each real agent prompt design doc (`Agent Prompts/Current Prompts/*.md`,
in the top-level repo) into a loadable runtime template
(`backend/prompts/{slug}/v1.txt`) plus a companion design doc
(`docs/agents/{slug}.md`, top-level repo). The original design doc is left in
place as the editorial source of truth, with one pointer line inserted at the
top.

Not runtime code -- imports the tested extraction/fill functions from
`agents.prompts` rather than reimplementing them, so the highest-risk logic
here is exercised by `backend/tests/agents/test_prompts.py`, not just by
this script's own one-off output.

Cross-repo by necessity: the design docs live in the top-level repo, the
runtime output belongs in `backend/`. Assumes both repos are checked out
together (the actual current layout, and the same assumption
`backend/_audit/`'s own research scripts already make to import
`simulation.*`) -- override with --source-dir if that assumption doesn't
hold for a given checkout.

Usage: python scripts/split_prompt_docs.py [--source-dir PATH] [--dry-run]
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.prompts import (  # noqa: E402
    bake_output_schema,
    extract_block,
    extract_stage_schema,
)

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SOURCE_DIR = _BACKEND_ROOT.parent / "Agent Prompts" / "Current Prompts"
_DEFAULT_DOCS_AGENTS_DIR = _BACKEND_ROOT.parent / "docs" / "agents"
_PROMPTS_OUT_DIR = _BACKEND_ROOT / "prompts"

# (doc filename stem, slug, is_multi_stage)
AGENTS = [
    ("Fundamental Analyst", "fundamental_analyst", False),
    ("Technical Analyst", "technical_analyst", False),
    ("Sentiment Analyst", "sentiment_analyst", False),
    ("Macro Economist", "macro_economist", False),
    ("Stock Researcher", "stock_researcher", False),
    ("Bull Case Advocate", "bull_advocate", False),
    ("Bear Case Advocate", "bear_advocate", False),
    ("Tax Strategist", "tax_strategist", False),
    ("Shadow CIO (calibration agent)", "shadow_cio", False),
    ("Chief Investment Officer (CIO)", "cio", True),
    ("Risk Advisor", "risk_advisor", True),
]

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def _report_unfilled(slug: str, label: str, text: str) -> None:
    placeholders = sorted(set(_PLACEHOLDER.findall(text)))
    print(f"  [{slug}] {label}: {len(text)} chars, placeholders: {placeholders or '(none)'}")


def _write(path: Path, text: str, dry_run: bool) -> None:
    print(f"  -> {path}{' (dry run, not written)' if dry_run else ''}")
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")


def _make_companion_doc(original: str, replacements: dict[str, str], template_path: str) -> str:
    """Original text with each extracted block's raw content swapped for a
    short pointer note -- everything else (changelog, rationale, the
    System-Prompt/User-Message interaction notes) survives untouched."""
    doc = original
    for old, note in replacements.items():
        if old in doc:
            # `old` is a regex-captured group, which includes whatever trailing
            # newline(s) sat before the closing ``` fence. Dropping that
            # whitespace glues the closing fence directly onto `note` with no
            # line break -- preserve it so the fence stays on its own line.
            trailing_ws = re.search(r"\s*$", old).group(0)
            doc = doc.replace(old, note + trailing_ws, 1)
    pointer = (
        f"> Runtime template: {template_path} · This doc: design rationale only, not executed.\n\n"
    )
    return pointer + doc


def _insert_pointer_line(original: str, template_note: str) -> str:
    if original.startswith(">"):
        return original  # already has a pointer line, e.g. re-run after a manual edit
    return f"{template_note}\n\n{original}"


def split_single_stage(doc_stem: str, slug: str, source_dir: Path, dry_run: bool) -> None:
    path = source_dir / f"{doc_stem} Agent Prompt.md"
    original = path.read_text(encoding="utf-8")

    system = extract_block(original, "System Prompt")
    schema = extract_block(original, "Output Schema")
    baked = bake_output_schema(system, schema)
    _report_unfilled(slug, "v1.txt", baked)
    _write(_PROMPTS_OUT_DIR / slug / "v1.txt", baked, dry_run)

    companion = _make_companion_doc(
        original,
        {
            system: "_(system prompt moved to the runtime template -- see the pointer above)_",
            schema: "_(output schema moved to the runtime template -- see the pointer above)_",
        },
        f"backend/prompts/{slug}/v1.txt",
    )
    _write(_DEFAULT_DOCS_AGENTS_DIR / f"{slug}.md", companion, dry_run)

    pointer = (
        f"> Runtime template: backend/prompts/{slug}/v1.txt · Companion doc: docs/agents/{slug}.md"
    )
    _write(path, _insert_pointer_line(original, pointer), dry_run)


def split_multi_stage(doc_stem: str, slug: str, source_dir: Path, dry_run: bool) -> None:
    path = source_dir / f"{doc_stem} Agent Prompt.md"
    original = path.read_text(encoding="utf-8")

    replacements: dict[str, str] = {}
    for stage in ("A", "B"):
        system = extract_block(original, f"System Prompt — Stage {stage}")
        schema = extract_stage_schema(original, "Output Schema", f"Stage {stage}")
        # CIO/Risk Advisor use a stage-qualified placeholder name
        # (`{output_schema_stage_a}` / `_b`), not the bare `{output_schema}`
        # the other 9 agents use -- same pre-bake mechanism, different name.
        baked = bake_output_schema(
            system, schema, placeholder=f"output_schema_stage_{stage.lower()}"
        )
        _report_unfilled(slug, f"v1_stage_{stage.lower()}.txt", baked)
        _write(_PROMPTS_OUT_DIR / slug / f"v1_stage_{stage.lower()}.txt", baked, dry_run)
        replacements[system] = f"_(Stage {stage} system prompt moved to the runtime template)_"
        replacements[schema] = f"_(Stage {stage} output schema moved to the runtime template)_"

    companion = _make_companion_doc(
        original, replacements, f"backend/prompts/{slug}/v1_stage_a.txt + v1_stage_b.txt"
    )
    _write(_DEFAULT_DOCS_AGENTS_DIR / f"{slug}.md", companion, dry_run)

    pointer = (
        f"> Runtime templates: backend/prompts/{slug}/v1_stage_a.txt, v1_stage_b.txt · "
        f"Companion doc: docs/agents/{slug}.md"
    )
    _write(path, _insert_pointer_line(original, pointer), dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=_DEFAULT_SOURCE_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.source_dir.exists():
        parser.error(
            f"{args.source_dir} does not exist -- pass --source-dir if the top-level "
            "repo isn't checked out as a sibling of backend/"
        )

    for doc_stem, slug, is_multi_stage in AGENTS:
        print(f"\n=== {doc_stem} -> {slug} ===")
        (split_multi_stage if is_multi_stage else split_single_stage)(
            doc_stem, slug, args.source_dir, args.dry_run
        )


if __name__ == "__main__":
    main()
