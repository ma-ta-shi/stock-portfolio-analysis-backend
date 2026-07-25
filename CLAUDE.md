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

## Setup gap to raise with the backend developer

`requirements.txt` currently has only `fastapi`, `pydantic`, `SQLAlchemy`, `starlette`, `uvicorn`
and their transitive deps — no `pytest`, `pytest-asyncio`, `httpx`, `ruff`, or `alembic`, even
though `src/CLAUDE.md` and the root project docs assume all of them. Confirm the intended dev
dependency setup (e.g. a `requirements-dev.txt` or `pyproject.toml` `[dependency-groups]`) before
adding tests or running lint in a PR — don't silently add a dependency file without agreeing on
the approach first.
