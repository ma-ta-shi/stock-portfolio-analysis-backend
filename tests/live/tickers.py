"""Shared ticker matrix for the live E2E test suite (ClickUp 86bb7j0kh).

Every category exists to probe a specific, previously-discovered failure
mode — not to be an exhaustive universe. Reused across test_provider_
completeness.py and test_router_chains.py so the matrix is defined once.
See `project_provider_test_tickers` memory / ClickUp 86bb7j0kh for the
original research this codifies.

Three entries below were flagged in that research as "verify live before
locking in" rather than assumed from memory — verified 2026-08-04:
- Delisted ticker: ATVI (Activision Blizzard, acquired/delisted by
  Microsoft) — yfinance confirms `quoteType: NONE`, no name/price.
- Zero-dividend growth stock: AMZN, not GOOG — GOOG and META both now pay
  real dividends (started 2024), confirmed live via `dividendRate`. AMZN's
  is None/None.
- Low-analyst-coverage name: MKO.V (Mako Mining), not TALA.V — TALA.V
  actually has 5 analysts (not low), MKO.V has 1, confirmed live via
  `numberOfAnalystOpinions`. Bonus: MKO.V is already a known-real
  cross-listed TSXV name from the crosslisting work, so it's a real,
  useful ticker rather than a throwaway.
"""

# --- Baseline sweep categories (used for the "does every method work at
# all" pass — deliberately reused rather than an arbitrary anchor pair,
# see ClickUp 86bb7j0kh plan discussion) ---

US_LARGE_CAP = ("AAPL", "MRNA", "FTNT", "ALB")
CA_CROSSLISTED = ("RY.TO", "SHOP.TO", "CVE.TO", "SU.TO")

# --- Edge-case categories, each tested only against the method(s) its
# "why" is actually about ---

# Thin financials / micro-cap — already the sparsest cases in
# test_edgartools.py's market-cap-spread test; reused, not reinvented.
US_MICRO_CAP = ("GRPN", "PLUG")

# FMP paywalls ETFs outright (HTTP 402, confirmed in fmp.py's own
# docstring) — probes get_price_history/get_dividend_history fallback.
US_ETF = ("TAN", "ARKK")

# FMP's dual-class/ADR paywall is per-ticker, not blanket — confirmed live
# 2026-08-04: BABA returns real data on every endpoint tested; GOOG's
# get_company_info now also returns real FMP data (fmp.py's docstring
# claim that dual-class tickers 402 on EVERY endpoint including /profile
# no longer holds for GOOG specifically). Kept for the "doesn't crash"
# pass-1 check regardless of which way the paywall currently falls.
US_DUAL_CLASS_ADR = ("GOOG", "BABA")

# BRK.B: confirmed live 2026-08-04 to still force the get_company_info
# fallback to yfinance (unlike GOOG, above) — the reliable forcing ticker
# for that specific pass-2 chain test.
US_DUAL_CLASS_STILL_PAYWALLED = "BRK.B"

# Never existed — the established "not mapped" placeholder from
# test_router.py, reused rather than inventing a new fake symbol.
INVALID_TICKER = "ZZZZ"

# Existed once, doesn't now — a distinct failure path from ZZZZ's "never
# existed". Verified live 2026-08-04 (see module docstring).
DELISTED_TICKER = "ATVI"

# Hyphenated multi-class share suffix — already broke the crosslisting
# resolver's naive ticker-guess logic; probes whether the *core*
# price/financials/dividend path handles the shape too.
CA_MULTI_CLASS = ("RCI-B.TO", "CCL-B.TO")

# Trust units — real, held securities deliberately excluded from the SEC
# crosslisting work as "not operating companies," but still need core
# price/financials/dividend coverage.
CA_REIT_TRUST_UNIT = ("CAR-UN.TO", "REI-UN.TO")

# TSXV: not uniformly zero-coverage — PLAN.V is a confirmed-uncovered
# negative case, SGML.V a confirmed-covered positive case (both from the
# crosslisting work). Tests the CA price/financials chain, not SEC
# crosslisting (that's a separate ticket's concern).
CA_TSXV_UNCOVERED = "PLAN.V"
CA_TSXV_COVERED = "SGML.V"

# Canadian ETFs/fund products — no operating-company data applies. Paired
# bare/suffixed forms because the router only trusts the suffix or a Stock
# record, never guesses — bare and suffixed forms can resolve completely
# differently (see AMBIGUOUS_COLLISION below).
CA_ETF_SUFFIXED = ("ZQQ.TO", "XGD.TO", "XEI.TO", "VGRO.TO")
CA_ETF_BARE = ("ZQQ", "XGD", "VGRO")

# Ambiguous bare-vs-.TO collisions. T/T.TO: AT&T (NYSE) vs TELUS (TSX),
# confirmed different companies. VGRO bare: confirmed live to silently
# return a *different real security's* data (not a clean empty) — the
# one collision case that needs its own explicit mismatch assertion
# rather than an empty/NotImplementedError expectation.
AMBIGUOUS_US_TICKER = "T"
AMBIGUOUS_CA_TICKER = "T.TO"
VGRO_BARE_COLLISION = "VGRO"

# Confirmed live 2026-08-04: 1 analyst opinion (see module docstring).
LOW_ANALYST_COVERAGE = "MKO.V"

# Dividend edge cases — RY.TO is a long consistent payer; AMZN confirmed
# live 2026-08-04 to have zero dividend (GOOG/META were wrong picks, both
# now pay real dividends as of 2024).
DIVIDEND_PAYER_CA = "RY.TO"
DIVIDEND_NONE_US = "AMZN"
