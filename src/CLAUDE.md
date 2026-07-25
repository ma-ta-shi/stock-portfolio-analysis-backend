# Backend — FastAPI + SQLAlchemy + Multi-Agent System

## Stack Details
- Python 3.12, FastAPI, SQLAlchemy 2.0 async, Alembic, Pydantic v2
- SQLite (dev) / PostgreSQL (prod) via Pydantic Settings
- structlog for structured logging, httpx for async HTTP
- pytest + pytest-asyncio + httpx.AsyncClient for testing
- pandas + pandas-ta for financial data pre-computation

## Directory Layout
- src/api/routes/          — Thin route handlers (delegate to services)
- src/agents/             — AI agent classes (10 agents, BaseAgent subclasses)
- src/data/               — Data provider adapters (openbb-tmx + yfinance for Canadian, edgartools + FMP for US, Finnhub supplementary, FRED + BoC Valet for macro — see Data Provider Strategy below)
- src/llm/                — LLM provider abstraction (Ollama + Claude API with fallback)
- src/models/             — SQLAlchemy ORM models
- src/schemas/            — Pydantic v2 request/response/agent message models
- src/schemas/agents/     — Per-agent input/output Pydantic schemas
- src/services/           — Business logic, agent orchestration
- src/services/orchestrator.py — Central agent runner (Pass 1 → compress → Pass 2 → CIO)
- src/agent_portfolio/    — Agent Run Portfolio: autonomous simulated portfolio (P2/P3)
  - manager.py            — AgentPortfolioManager: main orchestration service
  - constraints.py        — ConstraintEnforcer: validates trades against TFSA/RRSP/Trading rules
  - dividends.py          — Dividend detection, withholding tax, DRIP processing
  - models.py             — ORM models (AgentPortfolioHolding, Transaction, AccountState, Decision, Config)
  - comparison.py         — Performance comparison: agent vs user vs benchmark
- src/core/config.py      — Pydantic Settings with all env vars
- src/core/exceptions.py  — Domain exceptions (converted to HTTP only in routes)
- src/deps.py             — Dependency injection (db session, agents)
- prompts/                — Agent prompt files (prompts/{agent_name}/v{n}.txt)

## Data Provider Strategy

**Updated 2026-06 to match Notion → Financial Data API Research (the authoritative source — last updated May 2026). This table previously described an FMP-centric stack; that description is now stale and has been corrected here.**

Routing rule: detect Canadian stocks by `.TO` suffix. Route all `.TO` symbols through openbb-tmx + yfinance. Route everything else through edgartools (fundamentals/insider) + FMP (prices/news/ratios) + Finnhub (sentiment). Apply this check once in the provider adapter — never scatter it through agent code.

| Provider | Purpose | Limit | Key detail |
|----------|---------|-------|------------|
| **openbb-tmx** | Canadian prices, profiles, dividends, calendar, news (primary for `.TO` stocks) | Free, no key | TMX Money scraper — can break on site changes; startup health check + yfinance price fallback |
| **yfinance** | Canadian fundamentals (statements, ratios, ratings), benchmark history (`^GSPTSE`), bulk quote fallback | Free | Rate-limited (2s/symbol + backoff); apply `correct_alignment()` for the known column-misalignment bug before caching |
| **edgartools** | US fundamentals (income/balance sheet/cash flow) + Form 4 insider transactions, direct from SEC EDGAR | Free (MIT) | Requires `SP_EDGAR_IDENTITY` env var (a name+email string, not an API key — no registration); ~10 req/s SEC fair-use limit, auto-managed |
| **FMP** | US prices, profiles, news, analyst estimates/ratings, dividends, earnings calendar — **NOT fundamentals** | 250 calls/day free | Annual fundamentals only on free tier (quarterly needs Starter+); use edgartools for income statement/balance sheet/cash flow, not FMP |
| **Finnhub** (supplementary) | US news + AI sentiment, US peer identification | 60 calls/min free | US stocks only on free tier |
| **FRED** | US (and Canadian via OECD series) macro data | 120 calls/min free | Free, no daily cap |
| **Bank of Canada Valet API** | Canadian macro data (BoC rate, prime, CAD/USD, GoC yields) | No API key needed | Free, no documented limit |
| **VADER** | Local sentiment fallback for Canadian news (openbb-tmx returns headlines with no sentiment score) | Local — no API calls | pip install; raw text is also passed straight to the Sentiment Analyst LLM regardless |
| **pandas-ta** | Technical indicator computation (RSI, MACD, etc.) | Local — no API calls | Runs on price history already fetched from openbb-tmx/FMP |

**Alpha Vantage is dropped** — 25 calls/day is too restrictive for a 50-stock watchlist.

**Known gaps** (full detail in Notion → Financial Data API Research): Canadian peer identification uses a static `data/peers.json` file, not an API (Gap 1); Canadian insider trading is aggregate-only via openbb-tmx, vs. transactional Form 4 data for US stocks via edgartools (Gap 3); no Canadian earnings call transcripts on any free tier (Gap 4); edgartools falls back to yfinance when XBRL tagging is too sparse for small-cap US filers (Gap 5b).

### Weekly batch schedule (~30 Canadian + ~20 US stocks)

| Day | Content | Provider | ~Calls |
|-----|---------|----------|--------|
| Wednesday | CA fundamentals (30 stocks) | yfinance | ~90 (2s delays) |
| Wednesday | US fundamentals + insider (20 stocks) | edgartools | ~100–120 EDGAR requests |
| Wednesday | US profiles + price history (20 stocks) | FMP | ~40 |
| Thursday | CA profiles/estimates/ratings/dividends/insider (30 stocks) | openbb-tmx | ~150 |
| Thursday | US analyst estimates/ratings/dividends (20 stocks) | FMP | ~60 |
| Friday | CA prices + news (30 stocks) | openbb-tmx | ~60 |
| Friday | US news/bulk quote/earnings calendar/`^GSPC` | FMP | ~60 |
| Friday | TSX benchmark (`^GSPTSE`) | yfinance | 1 |
| Friday | Macro (FRED + BoC, ~10 series) | FRED/BoC | ~18 |
| Friday | US news with sentiment (20 stocks) | Finnhub | 20 |

FMP weekly total: ~160 calls — well within 250/day, ~90-call buffer for ad-hoc analysis. edgartools has no daily cap (~10 req/s SEC limit only). openbb-tmx, yfinance, and Bank of Canada Valet have no documented hard limits.

Total monthly provider cost: **$0** (all free tiers). First paid upgrade trigger unchanged: reassess FMP Starter ($19–22/month) once past 60–70+ stocks or in heavy on-demand-analysis territory.

### Env vars (src/core/config.py)
```
SP_FMP_API_KEY
SP_FINNHUB_API_KEY
SP_FRED_API_KEY              # free at fred.stlouisfed.org
SP_EDGAR_IDENTITY            # e.g. "Dylan Bann dbann913@gmail.com" — not an API key, just a User-Agent string; call set_identity() once on import
# openbb-tmx: no key needed
# Bank of Canada Valet: no key needed
# yfinance: no key needed
```

## SQLAlchemy 2.0 Pattern
```python
# CORRECT — async select() style
async def get_stock(db: AsyncSession, ticker: str) -> Stock | None:
    result = await db.execute(
        select(Stock).where(Stock.ticker == ticker)
        .options(selectinload(Stock.analyses))
    )
    return result.scalar_one_or_none()

# WRONG — never use legacy query() style
db.query(Stock).filter_by(ticker=ticker).first()  # NO
```

## Agent Architecture
- Agents are async classes in src/agents/ inheriting from BaseAgent
- Each agent has Pydantic input/output schemas in src/schemas/agents/
- Stock analysis orchestrator: src/services/orchestrator.py — never bypass it
- Pass 1 (parallel): Fundamental Analyst, Technical Analyst, Sentiment Analyst, Macro Economist, Stock Researcher
- Pass 2 (parallel, after Pass 1 compression): Bull Advocate, Bear Advocate, Risk Advisor, Tax Strategist
- Synthesis: CIO — reconciles Pass 2 perspectives into `stock_outlook` (5-point direction) + executive brief
- Portfolio Optimizer: separate 3-agent chain in src/services/portfolio_optimizer.py — Health Assessor → Opportunity Ranker → Action Synthesizer; runs after stock analyses complete; Action Synthesizer strongly prefers Claude over local Ollama
- Shadow CIO: calibration benchmark agent, runs non-blocking after primary CIO (step 16 in orchestrator); same inputs, bear-biased prompt (bear 2×, bull 0.5×); result stored to shadow_predictions table only — not user-facing; phased to Milestone 2
- Every agent run has a unique trace_id logged via structlog
- LLM routing: Ollama primary, Claude API fallback on timeout/failure

## Key ORM Models (Learning System)
- `counterfactual_records` — stores runner-up alternatives from the Opportunity Ranker alongside the primary recommendation; scored at prediction resolution (`recommended_return`, `counterfactual_returns` JSON array, `was_best` bool, `rank_vs_counterfactuals`)
- `shadow_predictions` — stores Shadow CIO outputs; fields for primary vs shadow outlook (direction, confidence, return tier), `divergence_magnitude`, and scoring fields (`primary_accuracy`, `shadow_accuracy`, `which_was_closer`); indexed on `analysis_run_id` and composite `(which_was_closer, created_at)` for win-rate aggregation

## Key ORM Models (Portfolio)
- `PortfolioHolding` — `shares` = current count (reduced on sells); `average_cost_basis` = recomputed as `total_book_cost / total_shares`; `total_book_cost` = sum of (shares × price) across all buys; `is_active` (False when all shares sold — soft close, never delete); `total_realized_gain_loss` (accumulated from sell transactions); `total_dividends_received`; `total_fees_paid`; `first_buy_date`; `last_transaction_date`. Key indexes: composite `(user_id, stock_id, account_type)` for fast lookups; `(user_id, is_active)` for separating active from closed.
- `PortfolioTransaction` — `transaction_type`: `buy|sell|dividend|drip|transfer_in|transfer_out`; `shares` is Optional (null for cash dividends); sell-only fields: `cost_basis_at_sell` (avg cost basis snapshot at time of sell, for audit), `realized_gain_loss` = `(price_per_share - avg_cost_basis) × abs(shares) - fees`, `realized_gain_loss_pct`; feedback fields: `active_prediction_id`, `system_outlook_at_sell`, `sell_aligned_with_system` (True if system was bearish and user sold; False if system was bullish but user sold anyway — feeds the learning engine). Key indexes: composite `(user_id, stock_id, transaction_date DESC)` for chronological history; `(holding_id, transaction_type)` for cost basis computation.
- `DividendEvent` — linked to a `portfolio_transactions` record (type: `dividend`); fields: `ex_date`, `payment_date`, `amount_per_share`, `total_amount`, `withholding_tax` (e.g. 15% US WHT on TFSA), `net_amount`, `dividend_type` (`eligible_canadian|us_qualified|foreign|return_of_capital`), `is_drip`
- **On sell transaction creation** the service: (1) computes `realized_gain_loss`; (2) snapshots `cost_basis_at_sell`; (3) reduces `holding.shares`; (4) adds P&L to `holding.total_realized_gain_loss`; (5) sets `holding.is_active = False` if shares reach 0; (6) checks active predictions and writes `sell_aligned_with_system`

## Key ORM Models (Agent Portfolio)
All 5 tables live in `src/agent_portfolio/models.py`.

- **`agent_portfolio_holdings`** — mirrors `portfolio_holdings` but autonomous; key fields: `user_id`, `stock_id`, `account_type`, `shares`, `average_cost_basis`, `total_book_cost`, `is_active`, `total_realized_gain_loss`, `total_dividends_received`, `total_fees_paid`, `first_buy_date`, `last_transaction_date`. Key indexes: composite `(user_id, stock_id, account_type)`; `(user_id, is_active)`.
- **`agent_portfolio_transactions`** — all buy/sell/dividend/drip/capital_injection trades the agent makes; sell rows carry `cost_basis_at_sell`, `realized_gain_loss`, `realized_gain_loss_pct`; every row includes decision context: `decision_type` (`outlook_driven|swap|rebalance|tax_harvest|capital_deploy|circuit_breaker`), `decision_confidence`, `decision_rationale`, `cio_outlook_at_decision`, `analysis_run_id`. Key indexes: composite `(user_id, stock_id, transaction_date DESC)`; `(user_id, decision_type)`.
- **`agent_portfolio_account_state`** — one row per user per account; tracks `cash_balance`, `contribution_room_remaining` (TFSA/RRSP), `total_contributions`, `total_withdrawals`, `tfsa_sells_this_quarter` (JSON `{ticker: count}`), `realized_capital_gains_ytd`, `realized_capital_losses_ytd`, `superficial_loss_cooldowns` (JSON `{ticker: cooldown_expiry_date}`). Key index: composite `(user_id, account_type)`.
- **`agent_portfolio_decisions`** — one row per optimization cycle; `cycle_type` (`weekly_batch|on_demand|earnings_triggered|circuit_breaker`), `optimization_input_snapshot` (JSON), `optimization_output` (JSON), `actions_taken` (JSON list), `actions_skipped` (JSON list), `portfolio_value_before/after`, `cash_deployed`, `cash_raised`, `transaction_costs_incurred`. Key index: composite `(user_id, cycle_date DESC)`.
- **`agent_portfolio_config`** — one row per user (unique `user_id`); `is_enabled`, `initialized_at`, `initial_portfolio_snapshot` (JSON), `injection_frequency` (`biweekly|monthly|none`), `injection_amount`, `last_injection_date`, `min_buy_confidence` (default 60), `min_sell_confidence` (default 65), `min_swap_improvement_pct` (default 3.0), `max_position_size_pct` (default 10.0), `max_sector_concentration_pct` (default 30.0), `scan_universe` (`watchlist|tsx60_sp500|custom`), `custom_universe_tickers` (JSON).

## Error Tracking System
- `ErrorRecord` ORM model — tracks every error: `severity` (critical/high/medium/low), `component`, `error_type`, run/agent/stock context, full stack trace, `context` JSON, `resolution_status` (open → acknowledged → resolved/auto_fixed/wont_fix), `auto_fix_proposal` text, `occurrence_count` (deduplication: same error within 24h increments count), `first_seen`/`last_seen`
- `ErrorService` (`src/services/error_service.py`) — every component captures errors here; handles deduplication, root cause linking (e.g. Ollama down → all agent timeouts linked), auto-fix proposal generation, and severity escalation (5+ occurrences → auto-escalate)
- Error type catalog: 20+ types across 7 components (agent, llm, data_pipeline, orchestrator, optimizer, feedback, api, frontend); each has defined severity and auto-fixable flag; key types: `output_validation_failed` (medium, sometimes auto-fixable), `ollama_connection_failed` (critical), `fmp_rate_limited` (medium), `render_error` (medium, auto-fixable)
- Auto-fix workflow: auto-fixable errors get a plain-English proposal → PM reviews on System Health page → approves → copies pre-formatted prompt into Claude Code
- Critical errors trigger a red banner on all pages; low errors are error-dashboard-only; 5+ occurrences auto-escalate severity
- Full specification: Notion Technical Design — Error Tracking and Monitoring System

## Testing
- pytest-asyncio auto mode (configured in pyproject.toml)
- httpx.AsyncClient for API tests, not TestClient
- SQLite in-memory for all database tests
- Mock LLM responses — never make real LLM calls in tests
- Pattern: Arrange → Act → Assert with explicit fixture setup
- All async — no sync test functions for async code
