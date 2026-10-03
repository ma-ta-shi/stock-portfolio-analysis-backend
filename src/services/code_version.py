"""Which backend code a run used (ledger BB-045, found while reviewing the prompt hash).

`agents.prompts.prompt_fingerprints()` covers the static template files, but a lot of what the
model reads is written by CODE: the "stale data: ..." and DATA COVERAGE lines
(`agents/utils.py`), the retry message (`agents/base.py`), the compression step, and every
precompute that decides what a field says. Changing a threshold therefore changes what an agent
is told while every prompt file, and so the prompt hash, stays the same. To say whether two runs
are comparable, a run needs to record the code state too.

This hashes every `.py` file AND every `.json` data file under `backend/src` (found in review:
`data/ca_us_crosslisting.json` decides which cross-listings a run
uses, so an edit to them changes behavior with no Python change). It is deliberately not the git
commit: work in progress is usually uncommitted, and a commit id would say "the same code" for two
different working trees. It changes on ANY such edit, related to the model input or not, which errs
on the safe side: two runs with the same code hash and prompt hash ran the same behavior-defining
files. Not covered: the third-party library versions and the model itself (`llm_config.model` is a
name; a re-pulled Ollama model with the same name is not detected).
"""

import hashlib
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1]  # backend/src
_SUFFIXES = (".py", ".json")


def code_fingerprint() -> str:
    """Twelve hex digits over the path and text of every source and data file, in a stable
    order. Text is read with universal newlines so a Windows and a Linux checkout agree."""
    digest = hashlib.sha256()
    for path in sorted(_SRC_DIR.rglob("*")):
        if not path.is_file() or path.suffix not in _SUFFIXES or "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(_SRC_DIR).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_text(encoding="utf-8", errors="replace").encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:12]
