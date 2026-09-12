# Canadian Tax Rules Reference

Runtime-bundled reference loaded by the Tax Strategist agent (`load_tax_rules_reference()`) and injected into its system prompt as structured context. This is the maintained source of truth at runtime — rule changes are made here and kept in sync with the Notion "Canadian Tax Rules Reference" page and its local mirror at `docs/product/canadian-tax-rules-reference.md`.

**Disclaimer:** For system-prompt injection only. Not tax advice. Rules are summarized for LLM consumption. Always verify against CRA publications for the current tax year.

**Last verified:** 2026-03-01 _(content verified March 2026; the day is nominal — kept as a full date for the parser. The whole reference needs a CRA freshness check before launch: the contribution limits are 2024 figures and the capital-gains inclusion rule above $250k was politically contested.)_

**Coverage:** This reference fully covers **ordinary common and preferred shares of Canadian and US corporations**, in all three account types, plus **Canadian REITs / income trusts in registered accounts** (TFSA/RRSP — 0% withholding, Canadian-source). It does **not** cover: Canadian REIT/trust treatment in a *taxable* account, US REITs, MLPs, non-MLP limited partnerships, ADRs, or non-US foreign withholding rates — the precompute returns those as `classification_confidence: low` / `modelled: false` for the agent to caveat and defer. Expanding this with the deferred `(classification × structure × account)` rows is tracked post-launch (ClickUp 86bbpm06v).

---

# TFSA (Tax-Free Savings Account)

**Core principle:** All investment income earned within a TFSA — capital gains, dividends, interest — is completely tax-free. Withdrawals are tax-free. Contributions are made with after-tax dollars (no deduction).

**Contribution room:**
- Annual limit: $7,000 (2024, 2025). Check CRA for current year.
- Cumulative: Unused room carries forward indefinitely.
- Withdrawals: Amount withdrawn is added back to contribution room on January 1 of the following year.
- Over-contribution penalty: 1% per month on the excess amount.
- Eligibility: Canadian residents aged 18+ with a valid SIN.

**Investment implications:**
- **Best for:** Highest-growth investments, since gains compound tax-free permanently.
- **Capital gains:** Tax-free. A stock that 10x in a TFSA generates zero tax.
- **Canadian dividends:** Tax-free (no dividend tax credit needed since there's no tax).
- **US dividends:** Subject to 15% US withholding tax under the Canada-US tax treaty. The US does not recognize the TFSA as a retirement account, so the treaty exemption does NOT apply. This withholding is NOT recoverable. For a stock yielding 3%, this costs ~0.45% annually.
- **Other foreign dividends:** Subject to withholding tax of the source country. Not recoverable.
- **Losses:** Cannot be used to offset gains elsewhere. A loss in a TFSA is a permanent loss of tax-sheltered room. This makes permanent capital loss risk especially painful in a TFSA.
- **Day trading caution:** CRA may consider frequent trading in a TFSA as carrying on a business, which could make the gains taxable. Keep TFSA investments as genuine investment, not active trading.

---

# RRSP (Registered Retirement Savings Plan)

**Core principle:** Contributions are tax-deductible (reduce taxable income in the year of contribution). All investment income grows tax-deferred. Withdrawals are taxed as ordinary income.

**Contribution room:**
- Annual limit: 18% of previous year's earned income, up to the annual maximum ($31,560 for 2024). Check CRA for current year.
- Cumulative: Unused room carries forward indefinitely.
- Pension adjustment: Employer pension contributions reduce RRSP room.
- Over-contribution: $2,000 lifetime over-contribution buffer. Beyond that, 1% per month penalty.

**Investment implications:**
- **Best for:** Income-generating investments (especially US dividend stocks), assets expected to have high returns, and situations where your tax rate at contribution is higher than your expected tax rate at withdrawal (retirement).
- **Capital gains:** Tax-deferred. Taxed as ordinary income on withdrawal (not as capital gains). This means the 50% inclusion rate advantage of capital gains is lost — all RRSP withdrawals are taxed at your marginal rate.
- **Canadian dividends:** Tax-deferred. Dividend tax credit is NOT available on RRSP withdrawals. The dividend is effectively converted to ordinary income.
- **US dividends:** Withholding-tax-EXEMPT under the Canada-US tax treaty (Article XVIII). The US recognizes the RRSP as a retirement account. This is the key advantage of holding US dividend stocks in an RRSP.
- **Other foreign dividends:** Withholding tax of source country applies. May be recoverable as a foreign tax credit in some cases.
- **Losses:** Same as TFSA — losses inside an RRSP cannot offset gains elsewhere.
- **Withdrawal timing:** Withdrawals in low-income years (early retirement, sabbatical) minimize tax impact.

**RRSP vs TFSA decision framework for the Tax Strategist:**
- High US dividend yield → RRSP (withholding tax exemption)
- High growth / low dividend → TFSA (tax-free capital gains)
- Current marginal rate > expected retirement rate → RRSP (tax deferral advantage)
- Current marginal rate < expected retirement rate → TFSA (no tax on withdrawal)

---

# Taxable (Trading) Account

**Core principle:** All investment income is taxable in the year it's earned. Different types of income have different tax treatments.

**Capital gains:**
- Inclusion rate: 50% (as of current rules — the first $250,000 of net capital gains annually). Gains above $250,000 have a 66.7% inclusion rate.
- Only realized gains are taxable. Unrealized gains (holding positions) are not taxed.
- Holding period: No explicit benefit for long-term vs short-term, but frequent trading may cause CRA to classify gains as business income (100% taxable).
- Capital losses: Can offset capital gains in the current year, or be carried back 3 years or forward indefinitely.

**Eligible Canadian dividends:**
- Receive the enhanced dividend tax credit.
- Gross-up rate: 38% (for eligible dividends).
- Federal dividend tax credit: 15.0198% of the grossed-up amount.
- Net effect: Eligible Canadian dividends are taxed at a lower effective rate than other income, especially for lower income brackets. In some provinces and income levels, the effective rate on eligible dividends can be close to 0%.

**Foreign dividends (including US):**
- Taxed as ordinary income (no dividend tax credit).
- US withholding tax (15%) can be claimed as a foreign tax credit on your Canadian tax return, partially or fully offsetting the Canadian tax.
- Net cost: Usually small or zero for US dividends in a taxable account because the foreign tax credit covers most of the withholding.

**Interest income:**
- Fully taxable as ordinary income. Worst tax treatment of any investment income type.

---

# Superficial Loss Rule

**Rule:** If you sell a security at a loss AND repurchase the same or identical security within 30 calendar days before or after the sale (60-day window total), the capital loss is denied.

**Details:**
- Applies to purchases by you, your spouse, or a corporation controlled by either.
- The denied loss is added to the adjusted cost base (ACB) of the repurchased security.
- "Identical" includes the same stock on different exchanges (e.g., selling RY on NYSE and buying RY.TO on TSX).
- Does NOT apply to transfers between registered accounts (RRSP, TFSA) and taxable accounts — but transferring in-kind to a registered account has its own rules.

**For the Tax Strategist agent:** Flag superficial loss risk whenever suggesting selling a position at a loss in a taxable account, especially if the user might want to rebuy. Suggest waiting 31 days before repurchasing, or buying a similar (but not identical) security as a substitute.

---

# Cross-border considerations (Canada-US tax treaty)

**Withholding tax summary:**

| Account type | US dividend withholding | Recoverable? |
|---|---|---|
| TFSA | 15% | No (treaty doesn't recognize the TFSA) |
| RRSP | 0% | N/A (exempt under treaty Article XVIII) |
| Taxable | 15% | Yes, via foreign tax credit on the Canadian return |

**US estate tax exposure:** Canadian residents holding US-situs assets (US stocks, US real estate) worth over $60,000 USD may be subject to US estate tax on death. The Canada-US tax treaty provides some relief, but this is a consideration for very large US stock holdings.

---

# Withholding tax grid (modelled combinations)

Formal restatement of the withholding treatment for the combinations the precompute models. The WHT resolver's canonical table mirrors this section (test-enforced). Rows marked **not modelled** are deferred (ClickUp 86bbpm06v) — the resolver returns `modelled: false` and the agent caveats.

| `dividend_classification` + `security_structure` | TFSA | RRSP | Taxable |
|---|---|---|---|
| `canadian_eligible` — CA corp, ordinary or preferred | 0% | 0% | 0% withholding; **eligible for the dividend tax credit** (38% gross-up, 15.0198% federal DTC) |
| `trust_distribution` — CA REIT / income trust | **0%** (Canadian-source) | **0%** (Canadian-source) | **not modelled** — no DTC; other-income / capital-gain / return-of-capital mix; effective rate needs the component split |
| `us` — US corp, ordinary or preferred | 15%, non-recoverable | 0% (treaty Art. XVIII) | 15%, recoverable as foreign tax credit |
| `us` + `reit` — US REIT | **not modelled** | **not modelled** | **not modelled** |
| `us` + `mlp` — US MLP | **not modelled** — generally inappropriate for any Canadian account | **not modelled** | **not modelled** |
| `limited_partnership` — non-MLP LP | **not modelled** | **not modelled** | **not modelled** |
| `foreign` — non-US, non-CA | **not modelled** — source-country rate, no treaty relief in a TFSA | **not modelled** | **not modelled** |
| `adr` | **not modelled** — withholding follows the underlying issuer's tax residence | **not modelled** | **not modelled** |
| `none` — non-dividend payer | n/a | n/a | n/a |

For `trust_distribution` in a taxable account: the reference does not yet model the effective rate. State that the distribution carries no dividend tax credit and is taxed less favourably than an eligible dividend, that a portion is typically return of capital (reduces ACB, defers tax) and capital gains (50% inclusion), and that the exact split is only known from the issuer's annual T3 — recommend the investor verify.

---

# Account optimization decision matrix

For the Tax Strategist agent's prompt, this matrix helps determine optimal account placement:

| Stock characteristic | Best account | Rationale |
|---|---|---|
| US stock, high dividend yield (>2%) | RRSP | Withholding tax exemption saves 0.3%+ annually |
| US stock, low/no dividend, high growth | TFSA | Tax-free capital gains on maximum growth |
| Canadian stock, eligible dividend | Taxable | Dividend tax credit makes effective tax rate very low |
| Canadian REIT / income trust | Registered (TFSA or RRSP) | No dividend tax credit is available regardless of account; the trust-distribution disadvantage (other-income treatment, ACB tracking) only bites in a taxable account |
| Canadian stock, high growth | TFSA | Tax-free capital gains |
| Any stock, expected loss / speculative | Taxable | Capital losses can offset gains elsewhere |
| Any stock, income generation for retirement | RRSP | Tax-deferred income, withdraw at lower rate in retirement |
| Short-term trade (weeks) | Taxable | Keep TFSA/RRSP for long-term compounding; frequent TFSA trading risks CRA business income classification |
