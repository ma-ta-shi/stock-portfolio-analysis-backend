"""Do the numbers Pass 1 hands to Pass 2 come through right? An independent accuracy audit, US and Canada.

    python scripts/audit_pass1_data.py                       # the default seven tickers
    python scripts/audit_pass1_data.py MSFT SHOP.TO RY.TO     # chosen tickers
    python scripts/audit_pass1_data.py --refresh              # rebuild the cached bundles
    python scripts/audit_pass1_data.py --show-ok              # also list the checks that agree
    python scripts/audit_pass1_data.py --only fundamentals,macro

Builds a real DataBundle per ticker with DataPipeline.prepare() (in an in-memory database, never app.db; about a
minute each because precompute calls the local model; cached under var/audit_bundles for 12 hours) and recomputes the
figures agents rely on from other sources: ratios hand-calculated from raw statements and compared with Yahoo, trailing
revenue and net income against the SEC for cross-listed Canadian names, short interest / analyst / 52-week / volume /
earnings dates against Yahoo, technical indicators from raw price history, macro against FRED, the Bank of Canada and
StatCan, Canadian insider values in CAD, and any Pass 2 field that arrives empty.

Exit code 1 when any unexplained mismatch is found (documented definitional differences are reported as "known").
Needs the network, SP_FRED_API_KEY and SP_EDGAR_IDENTITY from backend/.env, and Ollama running. What it covers and
what it does not: docs/technical/pass1-data-audit-2026-10-02.md.
"""

import argparse
import asyncio
import sys
from pathlib import Path

import logging

import _bootstrap  # noqa: F401  (UTF-8 console, sys.path, table imports; must come first)
import structlog
from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_ROOT / ".env")
# The providers log expected failures with stack traces (a 40-F filer has no XBRL, a StatCan timeout); the report is
# the output here, so only critical log lines show.
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))

from data.pipeline import _fetch_usd_cad  # noqa: E402
from services import data_audit as audit  # noqa: E402

CHECKS = ("fundamentals", "snapshot", "technicals", "macro", "currency", "research", "fundamental_inputs", "gaps")


async def _main(tickers: list[str], refresh: bool, show_ok: bool, only: set[str]) -> int:
    cache_dir = BACKEND_ROOT / "var" / "audit_bundles"
    usd_cad = await _fetch_usd_cad()
    findings: list[audit.Finding] = []
    macro_done = {"us": False}
    for ticker in tickers:
        print(f"building {ticker} ...", flush=True)
        try:
            bundle = await audit.build_bundle(ticker, cache_dir, refresh)
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop the rest
            findings.append(audit.Finding(ticker, "bundle", "build", "skipped", note=f"{type(exc).__name__}: {str(exc)[:80]}"))
            continue
        runners = {
            "fundamentals": lambda b=bundle: audit.check_fundamentals(b, usd_cad),
            "snapshot": lambda b=bundle: audit.check_snapshot(b),
            "technicals": lambda b=bundle: audit.check_technicals(b),
            "currency": lambda b=bundle: audit.check_currency(b),
            "research": lambda b=bundle: audit.check_research(b),
            "fundamental_inputs": lambda b=bundle: audit.check_fundamental_inputs(b),
            "gaps": lambda b=bundle: audit.check_gaps(b),
        }
        for name in CHECKS:
            if name not in only:
                continue
            try:
                if name == "macro":
                    # FRED and the Bank of Canada once for the run; StatCan for every Canadian ticker
                    result = await audit.check_macro(bundle, us=not macro_done["us"])
                    macro_done["us"] = True
                    findings += audit.downgrade_if_stale(result, bundle)
                else:
                    result = runners[name]()
                    # live comparisons drift as the bundle ages; the fundamentals check uses the bundle's own figures
                    findings += audit.downgrade_if_stale(result, bundle) if name in ("snapshot", "technicals", "currency") else result
            except Exception as exc:  # noqa: BLE001
                findings.append(audit.Finding(ticker, name, "check", "skipped", note=f"{type(exc).__name__}: {str(exc)[:80]}"))
        if "fundamentals" in only:
            try:
                findings += await audit.check_sec(bundle, usd_cad)
            except Exception as exc:  # noqa: BLE001
                findings.append(audit.Finding(ticker, "sec", "check", "skipped", note=f"{type(exc).__name__}: {str(exc)[:80]}"))
    print(audit.format_report(findings, show_ok=show_ok))
    return 1 if audit.summarize(findings)["mismatch"] else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tickers", nargs="*", default=audit.DEFAULT_TICKERS)
    parser.add_argument("--refresh", action="store_true", help="rebuild cached bundles")
    parser.add_argument("--show-ok", action="store_true", help="also list checks that agree")
    parser.add_argument("--only", default=",".join(CHECKS), help=f"comma list of: {', '.join(CHECKS)}")
    args = parser.parse_args()
    only = {c.strip() for c in args.only.split(",") if c.strip()}
    unknown = only - set(CHECKS)
    if unknown:
        print(f"Unknown check(s): {', '.join(sorted(unknown))}")
        sys.exit(2)
    sys.exit(asyncio.run(_main(args.tickers, args.refresh, args.show_ok, only)))


if __name__ == "__main__":
    main()
