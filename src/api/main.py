import sys

# Fixed 2026-08-04 (86bb7j0kh) — real bug, confirmed live: structlog's
# default logger writes to stdout/stderr using the process's default
# encoding, which on a plain Windows console is cp1252, not UTF-8. Any
# logged exception whose text/traceback contains a non-cp1252 character
# (confirmed live via router.py's own exception-handling logger.warning(
# ..., exc_info=True) call) raises UnicodeEncodeError — crashing the very
# fallback-catching code path that's supposed to degrade gracefully.
# structlog is never explicitly configured anywhere in this codebase, so
# there's no central place to patch its formatter; reconfiguring the
# underlying streams once here fixes it regardless of how/whether
# structlog ever gets a real config. Must run before anything else logs.
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

from fastapi import FastAPI  # noqa: E402 — must follow the encoding fix above
from api.database import Base, engine  # noqa: E402
from api.routes import user_profile  # noqa: E402

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(user_profile.router)
