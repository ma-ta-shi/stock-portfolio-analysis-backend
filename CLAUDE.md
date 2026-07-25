# Backend — Stock Picker FastAPI service

Architecture, data provider strategy, ORM models, and agent design: see `src/CLAUDE.md`.
This file covers repo-level conventions and the contribution workflow.

## Contribution Workflow

This repo is **PR-only** — the backend developer reviews and merges everything; Claude never
pushes to `main` or merges its own work. A `PreToolUse` hook (`.claude/hooks/block-protected-branch.cjs`)
structurally blocks commits/pushes/merges against `main` and any force push, so this isn't just a
convention.

1. **Branch**: `git checkout -b feature/<short-desc>` or `fix/<short-desc>` off latest `main` before any change.
   Check with the backend developer which branch is actually canonical to base off — `main`, `feature/agents`,
   and `feature/database` have diverged and don't share a clean linear history yet.
2. **Implement + test together**: write pytest tests alongside the code, not after. Every new route/service
   needs both success and error-path coverage.
3. **Definition of done** before opening a PR:
   - `pytest -xvs --timeout=30` passing
   - `ruff check . && ruff format --check .` clean
   - Layer separation respected (Router → Service → Repository, never skipped)
   - Self-review with `/code-review` on the diff — fix findings before moving on
4. **Commit**: descriptive message, present tense, explains *why* not just *what*.
5. **PR**: `gh pr create` with a Summary + Test plan body. Small, focused diffs — one feature or fix
   per PR, not a batch of unrelated changes.
6. **Never merge**: the PR sits until the backend developer approves and merges it.

## Dev environment

`requirements.txt` only ever had runtime deps (`fastapi`, `pydantic`, `SQLAlchemy`, `starlette`,
`uvicorn`) — no test/lint tooling. Added locally (not yet reviewed by the backend developer):
- `.venv/` — Python 3.12 virtualenv, gitignored
- `requirements-dev.txt` — pulls in `requirements.txt` plus `pytest`, `pytest-asyncio`, `httpx`,
  `ruff`, `alembic`, `aiosqlite`
- `pyproject.toml` — `[tool.pytest.ini_options]` (`asyncio_mode = "auto"`) and `[tool.ruff]`

Run `pip install -r requirements-dev.txt` inside `.venv` before running tests or lint. **This
dependency-management approach (separate requirements-dev.txt vs. pyproject.toml
dependency-groups) hasn't been confirmed with the backend developer yet** — raise it in the first
PR that touches this, since they may already have a preference.

Running `ruff check .` today surfaces ~70 pre-existing lint errors in the backend developer's
existing code (undefined names, unused imports in `src/api/tables/*.py`, `src/multi_agents/*.py`,
`test.py`). Not touched here — not our code to unilaterally fix — but expect CI to fail on `main`
until those are cleaned up or ruff is scoped to new/changed files only.
